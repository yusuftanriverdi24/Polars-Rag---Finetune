"""Hybrid retrieval: dense + BM25, fused with reciprocal rank fusion.

The query is the task **instruction only** (no setup code / signature), so
retrieval sees exactly what a user would type. Arm B2 can instead use a
*draft-code* query (HyDE-style, ``draft_query``): the instruction followed by
the code of the model's own zero-shot answer, so the query speaks the API's
vocabulary (``unpivot``, ``group_by``) instead of the task's.

1. Dense: exact cosine search over the chunk vectors (top ``N_CANDIDATES``).
2. Sparse: LanceDB native BM25 full-text search over ``fts_text``
   (top ``N_CANDIDATES``).
3. Fusion: ``score(d) = sum_r 1 / (RRF_K + rank_r(d))`` over the two rankings
   (1-based ranks); ties are broken by chunk id, so the order is deterministic.
4. Packing: walk the fused list and take chunks until ``k`` are selected,
   skipping any chunk that would push the *formatted* context past
   ``budget_tokens`` (measured with the generation model's tokenizer).

``format_context`` renders the selected chunks for the prompt's existing
``{context}`` block. Every chunk gets a header with its source title (and
section), so the model can tell API reference from user-guide text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .chunk import TokenCounter
from .index import (
    DEFAULT_INDEX_DIR,
    EMBEDDERS,
    TABLE_NAME,
    embed_query,
    embedder_key_for,
    load_embedder,
    read_index_meta,
)

DEFAULT_K = 5
DEFAULT_BUDGET_TOKENS = 2000
N_CANDIDATES = 50
RRF_K = 60


@dataclass
class Retrieved:
    id: str
    kind: str
    title: str
    section: str
    api_names: list[str]
    text: str
    n_tokens: int
    rrf_score: float = 0.0
    dense_rank: int | None = None
    fts_rank: int | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """What generations.jsonl records per retrieved chunk."""

        return {
            "id": self.id,
            "title": self.title,
            "section": self.section,
            "rrf_score": round(self.rrf_score, 6),
            "dense_rank": self.dense_rank,
            "fts_rank": self.fts_rank,
        }


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of ranked id lists -> [(id, score)], best first.

    Ties are broken by id so the result is fully deterministic.
    """

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def format_chunk(c: Retrieved, index: int) -> str:
    head = c.title if not c.section or c.section == c.title else f"{c.title} ({c.section})"
    return f"[{index}] {head}\n{c.text.strip()}"


def format_context(chunks: list[Retrieved]) -> str:
    """Render retrieved chunks for the prompt's ``{context}`` block."""

    return "\n\n".join(format_chunk(c, i) for i, c in enumerate(chunks, start=1))


def pack_within_budget(
    candidates: list[Retrieved],
    count: TokenCounter,
    *,
    k: int = DEFAULT_K,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> list[Retrieved]:
    """Take candidates in order until ``k`` are selected, skipping any that
    would push the formatted context past ``budget_tokens``."""

    selected: list[Retrieved] = []
    for c in candidates:
        if len(selected) >= k:
            break
        trial = selected + [c]
        if count(format_context(trial)) <= budget_tokens:
            selected = trial
    return selected


QUERY_MODES = ("instruction", "draft")


def draft_query(instruction: str, draft_output: str | None) -> str:
    """Instruction + the code of a draft answer (the model's own zero-shot
    output). Falls back to the instruction alone when the draft has no code."""

    from ..bench.extract import extract_code

    code = extract_code(draft_output).strip()
    return f"{instruction.strip()}\n\n{code}" if code else instruction.strip()


def load_drafts(path: Path) -> dict[str, dict]:
    """task_id -> generation record (``output``, ``latency_s``, ...) from a generations.jsonl."""

    import json

    with Path(path).open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return {r["task_id"]: r for r in records}


def build_query(mode: str, instruction: str, draft_output: str | None = None) -> str:
    if mode == "instruction":
        return instruction
    if mode == "draft":
        return draft_query(instruction, draft_output)
    raise ValueError(f"unknown query mode {mode!r}; expected one of {QUERY_MODES}")


_FTS_UNSAFE = re.compile(r"[^\w\s]")


def fts_query_text(query: str) -> str:
    """Strip punctuation (backticks, quotes, operators) the FTS parser may reject."""

    return " ".join(_FTS_UNSAFE.sub(" ", query).split())


class Retriever:
    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        *,
        count: TokenCounter | None = None,
        embedder=None,
        device: str | None = None,
    ):
        import lancedb

        from .chunk import load_token_counter

        self.index_dir = Path(index_dir)
        self.meta = read_index_meta(self.index_dir)
        self.table = lancedb.connect(str(self.index_dir)).open_table(TABLE_NAME)
        self.embedder_key = embedder_key_for(self.meta)
        self.query_instruction = EMBEDDERS[self.embedder_key]["query_instruction"]
        self.embedder = embedder or load_embedder(device, self.embedder_key)
        self.count = count or load_token_counter()

    def candidates(self, query: str, n: int = N_CANDIDATES) -> list[Retrieved]:
        cols = ["id", "kind", "title", "section", "api_names", "text", "n_tokens"]
        qvec = embed_query(self.embedder, query, self.query_instruction)
        dense = (
            self.table.search(qvec, query_type="vector")
            .distance_type("cosine")
            .select(cols)
            .limit(n)
            .to_list()
        )
        fts_q = fts_query_text(query)
        sparse = (
            self.table.search(fts_q, query_type="fts", fts_columns="fts_text").select(cols).limit(n).to_list()
            if fts_q
            else []
        )
        # Stable ranks: break exact score ties by id.
        dense.sort(key=lambda r: (r["_distance"], r["id"]))
        sparse.sort(key=lambda r: (-r["_score"], r["id"]))

        rows = {r["id"]: r for r in dense + sparse}
        dense_rank = {r["id"]: i for i, r in enumerate(dense, start=1)}
        fts_rank = {r["id"]: i for i, r in enumerate(sparse, start=1)}
        fused = rrf_fuse([[r["id"] for r in dense], [r["id"] for r in sparse]])
        out = []
        for doc_id, score in fused:
            r = rows[doc_id]
            out.append(
                Retrieved(
                    id=doc_id,
                    kind=r["kind"],
                    title=r["title"],
                    section=r["section"],
                    api_names=list(r["api_names"] or []),
                    text=r["text"],
                    n_tokens=int(r["n_tokens"]),
                    rrf_score=score,
                    dense_rank=dense_rank.get(doc_id),
                    fts_rank=fts_rank.get(doc_id),
                )
            )
        return out

    def retrieve(
        self, query: str, *, k: int = DEFAULT_K, budget_tokens: int = DEFAULT_BUDGET_TOKENS
    ) -> list[Retrieved]:
        return pack_within_budget(self.candidates(query), self.count, k=k, budget_tokens=budget_tokens)
