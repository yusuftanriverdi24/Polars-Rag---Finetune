"""Build the LanceDB hybrid-search index over the doc chunks.

One LanceDB table holds every chunk with

- a dense vector from ``EMBED_MODEL`` (L2-normalised, float32), searched
  exactly (flat scan; no approximate index, so results are deterministic), and
- a BM25 full-text index (LanceDB native FTS) over ``fts_text`` = title +
  section + API names + chunk text.

The two result lists are fused with reciprocal rank fusion in
``retrieve.py``.

Embedding model: ``ibm-granite/granite-embedding-small-english-r2`` (47M
params, 384-d, 8k context, Apache-2.0) at a pinned revision. It builds the whole
index in about a minute on an L4 and ~2.5 h on a 4-core laptop CPU; queries are
instant on either. ``Qwen/Qwen3-Embedding-0.6B`` scores higher on MTEB but was
projected at ~25 h for a CPU build here, and ``bge-small-en-v1.5`` caps input at
512 tokens and is weaker on code retrieval. See the PLAN.md decisions log.
Queries and documents are embedded as raw text (no instruction prefix).

Arm B2 adds a second embedder, ``Qwen/Qwen3-Embedding-0.6B`` (``--embedder
qwen3``; GPU only in practice), whose queries carry a one-sentence task
instruction in the format its model card recommends ("Instruct: <task>",
newline, "Query:<query>"); documents get none. ``EMBEDDERS`` holds both
configs. The index records which one built it, and ``Retriever`` loads the
same one for queries.

Determinism: chunks come in a fixed order from ``chunks.jsonl`` (hashed), the
model revision is pinned, and the table is rebuilt from scratch every time.
``index_meta.json`` records the chunk-file hash, model/revision, device and a
fingerprint of the vectors (rounded to 1e-4), so two builds can be compared.
Vectors from a GPU build and a CPU build differ in the last bits, and that can
reorder near-ties. That's why every generation records the ids of the chunks it
actually retrieved.

CLI::

    python -m polars_llm.rag.index        # data/rag/chunks.jsonl -> data/rag/index
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

from .chunk import DEFAULT_OUT as DEFAULT_CHUNKS
from .chunk import Chunk, read_chunks

EMBED_MODEL = "ibm-granite/granite-embedding-small-english-r2"
EMBED_REVISION = "2ab6fa8ea2d674564defd37171ae19079b864b33"
EMBED_MAX_SEQ_LENGTH = 2048  # longest chunk is ~1.3k tokens: nothing is truncated
QUERY_INSTRUCTION: str | None = None  # granite-embedding takes raw queries

# One task-specific instruction, written once in the model card's recommended
# format and not tuned. It also covers the draft-code query (instruction + draft).
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a data-processing task, possibly followed by a draft Polars solution, "
    "retrieve the Polars documentation passages that describe the API needed to implement it\nQuery:"
)

EMBEDDERS: dict[str, dict] = {
    "granite": {
        "model": EMBED_MODEL,
        "revision": EMBED_REVISION,
        "max_seq_length": EMBED_MAX_SEQ_LENGTH,
        "query_instruction": QUERY_INSTRUCTION,
        "tokenizer_kwargs": {},
    },
    "qwen3": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "max_seq_length": EMBED_MAX_SEQ_LENGTH,
        "query_instruction": QWEN3_QUERY_INSTRUCTION,
        # Last-token pooling: the model card pads on the left.
        "tokenizer_kwargs": {"padding_side": "left"},
    },
}
DEFAULT_EMBEDDER = "granite"

DEFAULT_INDEX_DIR = Path("data/rag/index")
TABLE_NAME = "chunks"
META_FILE = "index_meta.json"


def embedder_key_for(meta: dict) -> str:
    """Which ``EMBEDDERS`` entry built an index (v1 indexes predate the key)."""

    key = meta.get("embedder")
    if key:
        return key
    for k, cfg in EMBEDDERS.items():
        if cfg["model"] == meta.get("embed_model"):
            return k
    raise ValueError(f"unknown embedder for index meta: {meta.get('embed_model')!r}")


def load_embedder(device: str | None = None, embedder: str = DEFAULT_EMBEDDER):
    from sentence_transformers import SentenceTransformer

    cfg = EMBEDDERS[embedder]
    model = SentenceTransformer(
        cfg["model"], revision=cfg["revision"], device=device, tokenizer_kwargs=cfg["tokenizer_kwargs"] or None
    )
    model.max_seq_length = cfg["max_seq_length"]
    return model


def document_text(c: Chunk) -> str:
    """What gets embedded for a chunk: its title/section header plus the text."""

    head = c.title if not c.section else f"{c.title} | {c.section}"
    return f"{head}\n{c.text}"


def fts_text(c: Chunk) -> str:
    return "\n".join([c.title, c.section, " ".join(c.api_names), c.text])


def embed_documents(model, texts: list[str], batch_size: int = 16):
    import numpy as np

    # Sort by length for efficient batching, then restore the original order.
    order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))
    vecs = model.encode(
        [texts[i] for i in order],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    out = np.empty_like(vecs)
    out[order] = vecs
    return out


def embed_query(model, query: str, instruction: str | None = QUERY_INSTRUCTION):
    kwargs = {"prompt": instruction} if instruction else {}
    return model.encode([query], normalize_embeddings=True, convert_to_numpy=True, **kwargs)[0].astype("float32")


def vectors_fingerprint(vectors) -> str:
    import numpy as np

    return hashlib.sha256(np.round(vectors, 4).astype("float32").tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(
    chunks_path: Path = DEFAULT_CHUNKS,
    index_dir: Path = DEFAULT_INDEX_DIR,
    *,
    device: str | None = None,
    batch_size: int = 16,
    embedder=None,
    embedder_key: str = DEFAULT_EMBEDDER,
) -> dict:
    """(Re)build the index from scratch and return its metadata."""

    import lancedb
    import pyarrow as pa

    chunks = read_chunks(chunks_path)
    cfg = EMBEDDERS[embedder_key]
    model = embedder or load_embedder(device, embedder_key)
    vectors = embed_documents(model, [document_text(c) for c in chunks], batch_size=batch_size)
    dim = int(vectors.shape[1])

    table = pa.table(
        {
            "id": [c.id for c in chunks],
            "kind": [c.kind for c in chunks],
            "source": [c.source for c in chunks],
            "title": [c.title for c in chunks],
            "section": [c.section for c in chunks],
            "api_names": [c.api_names for c in chunks],
            "text": [c.text for c in chunks],
            "n_tokens": pa.array([c.n_tokens for c in chunks], pa.int32()),
            "fts_text": [fts_text(c) for c in chunks],
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1), pa.float32()), dim),
        }
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(index_dir))
    tbl = db.create_table(TABLE_NAME, data=table, mode="overwrite")
    tbl.create_fts_index("fts_text", replace=True, use_tantivy=False)

    meta = {
        "built_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "chunks_file": str(chunks_path),
        "chunks_sha256": _file_sha256(chunks_path),
        "n_chunks": len(chunks),
        "embedder": embedder_key,
        "embed_model": cfg["model"],
        "embed_revision": cfg["revision"],
        "embed_dim": dim,
        "query_instruction": cfg["query_instruction"],
        "device": str(getattr(model, "device", device)),
        "vectors_fingerprint": vectors_fingerprint(vectors),
        "lancedb_version": lancedb.__version__,
        "fts": {"column": "fts_text", "engine": "lancedb-native-bm25"},
        "vector_search": "exact (flat), cosine on normalised vectors",
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def read_index_meta(index_dir: Path = DEFAULT_INDEX_DIR) -> dict:
    return json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the LanceDB hybrid index.")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default=DEFAULT_EMBEDDER)
    args = parser.parse_args(argv)

    meta = build_index(
        Path(args.chunks), Path(args.index_dir), device=args.device, batch_size=args.batch_size,
        embedder_key=args.embedder,
    )
    print(
        f"Indexed {meta['n_chunks']} chunks ({meta['embed_model']}, dim {meta['embed_dim']}, "
        f"device {meta['device']}) -> {args.index_dir}"
    )
    print(f"chunks sha256 {meta['chunks_sha256'][:12]}  vectors fingerprint {meta['vectors_fingerprint'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
