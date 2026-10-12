"""Retrieval: RRF fusion, budget packing, context formatting, and an
end-to-end hybrid search over a tiny LanceDB index with a fake embedder."""

import hashlib
import re

import numpy as np
import pytest

from polars_llm.rag.chunk import Chunk, whitespace_token_counter as count, write_chunks
from polars_llm.rag.retrieve import (
    Retrieved,
    fts_query_text,
    format_context,
    pack_within_budget,
    rrf_fuse,
)


def _r(id_, n_tokens=10, title=None, section="", text=None, api_names=()):
    return Retrieved(
        id=id_,
        kind="api",
        title=title or f"polars.{id_}",
        section=section,
        api_names=list(api_names),
        text=text if text is not None else "tok " * n_tokens,
        n_tokens=n_tokens,
    )


# --- reciprocal rank fusion ---------------------------------------------------


def test_rrf_scores_and_order():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [d for d, _ in fused] == ["a", "c", "b", "d"]


def test_rrf_ties_are_broken_by_id():
    # "x" and "y" each appear once at rank 1 -> identical scores.
    assert [d for d, _ in rrf_fuse([["y"], ["x"]])] == ["x", "y"]


def test_rrf_item_in_both_lists_beats_single_list_items():
    fused = [d for d, _ in rrf_fuse([["a", "both"], ["b", "both"]])]
    assert fused[0] == "both"


# --- formatting + budget ----------------------------------------------------------


def test_format_context_has_numbered_source_titles():
    ctx = format_context(
        [
            _r("Expr.str.to_uppercase", text="Uppercase it."),
            _r("g", title="Strings", section="Modifying strings > Casing", text="Guide text."),
        ]
    )
    assert ctx == (
        "[1] polars.Expr.str.to_uppercase\nUppercase it.\n\n"
        "[2] Strings (Modifying strings > Casing)\nGuide text."
    )
    assert format_context([]) == ""


def test_pack_stops_at_k():
    cands = [_r(f"c{i}") for i in range(10)]
    assert [c.id for c in pack_within_budget(cands, count, k=3, budget_tokens=10_000)] == ["c0", "c1", "c2"]


def test_pack_respects_budget_and_skips_oversized_chunks():
    cands = [_r("small1", 20), _r("huge", 500), _r("small2", 20), _r("small3", 20)]
    selected = pack_within_budget(cands, count, k=3, budget_tokens=80)
    assert [c.id for c in selected] == ["small1", "small2", "small3"]
    assert count(format_context(selected)) <= 80


def test_pack_budget_counts_the_formatted_context():
    # Each formatted chunk = header (2 words) + 30 words -> two fit in 64, not three.
    cands = [_r(f"c{i}", 30) for i in range(3)]
    assert len(pack_within_budget(cands, count, k=5, budget_tokens=64)) == 2


def test_fts_query_text_strips_punctuation():
    assert fts_query_text("Return `name` and the 'total' (sum) of x.") == "Return name and the total sum of x"


# --- end-to-end hybrid search over a real LanceDB index ------------------------


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""

    device = "cpu"
    dim = 64

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z_]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, prompt=None, **kwargs):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def tiny_index(tmp_path):
    from polars_llm.rag.index import build_index

    chunks = [
        Chunk("api:Expr.str.to_uppercase", "api", "s.rst", "polars.Expr.str.to_uppercase", "String",
              ["Expr.str.to_uppercase"], "Convert strings to uppercase letters.", 5),
        Chunk("api:DataFrame.join", "api", "d.rst", "polars.DataFrame.join", "Joining",
              ["DataFrame.join"], "Join two dataframes on a key column, inner left full.", 9),
        Chunk("api:DataFrame.group_by", "api", "d.rst", "polars.DataFrame.group_by", "Aggregation",
              ["DataFrame.group_by"], "Group rows by key columns then aggregate each group.", 9),
        Chunk("guide:x.md#001", "guide", "x.md", "Joins", "Join strategies",
              ["DataFrame.join"], "An inner join keeps rows whose key appears in both dataframes.", 11),
    ]
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    meta = build_index(path, tmp_path / "index", embedder=FakeEmbedder())
    return tmp_path / "index", meta


def test_build_index_meta(tiny_index):
    _, meta = tiny_index
    assert meta["n_chunks"] == 4
    assert meta["embed_dim"] == FakeEmbedder.dim
    assert len(meta["chunks_sha256"]) == 64 and len(meta["vectors_fingerprint"]) == 64


def test_hybrid_retrieval_end_to_end(tiny_index):
    from polars_llm.rag.retrieve import Retriever

    index_dir, _ = tiny_index
    r = Retriever(index_dir, count=count, embedder=FakeEmbedder())
    cands = r.candidates("Inner join the orders dataframe to customers on the key")
    ids = [c.id for c in cands]
    assert set(ids[:2]) == {"api:DataFrame.join", "guide:x.md#001"}
    top = cands[0]
    assert top.dense_rank is not None and top.fts_rank is not None  # found by both
    # Deterministic: the same query gives the same ranking.
    assert [c.id for c in r.candidates("Inner join the orders dataframe to customers on the key")] == ids

    selected = r.retrieve("convert names to uppercase", k=1, budget_tokens=1000)
    assert [c.id for c in selected] == ["api:Expr.str.to_uppercase"]
    assert selected[0].summary()["title"] == "polars.Expr.str.to_uppercase"


# --- Arm B2: embedder registry + draft-code query ---------------------------------


class RecordingEmbedder(FakeEmbedder):
    def __init__(self):
        self.prompts = []

    def encode(self, texts, prompt=None, **kwargs):
        self.prompts.append(prompt)
        return super().encode(texts)


def test_index_records_embedder_and_retriever_uses_its_query_instruction(tmp_path):
    from polars_llm.rag.index import EMBEDDERS, QWEN3_QUERY_INSTRUCTION, build_index
    from polars_llm.rag.retrieve import Retriever

    path = tmp_path / "chunks.jsonl"
    write_chunks([Chunk("api:DataFrame.join", "api", "d.rst", "polars.DataFrame.join", "Joining",
                        ["DataFrame.join"], "Join two dataframes.", 4)], path)
    meta = build_index(path, tmp_path / "idx", embedder=FakeEmbedder(), embedder_key="qwen3")
    assert meta["embedder"] == "qwen3"
    assert meta["embed_model"] == EMBEDDERS["qwen3"]["model"]
    assert meta["query_instruction"] == QWEN3_QUERY_INSTRUCTION
    # Model-card format: "Instruct: <one sentence>\nQuery:" (the query follows directly).
    assert QWEN3_QUERY_INSTRUCTION.startswith("Instruct: ") and QWEN3_QUERY_INSTRUCTION.endswith("\nQuery:")

    emb = RecordingEmbedder()
    r = Retriever(tmp_path / "idx", count=count, embedder=emb)
    r.candidates("join two frames")
    assert emb.prompts == [QWEN3_QUERY_INSTRUCTION]


def test_v1_index_meta_without_embedder_key_maps_to_granite(tiny_index):
    from polars_llm.rag.index import EMBED_MODEL, embedder_key_for
    from polars_llm.rag.retrieve import Retriever

    assert embedder_key_for({"embed_model": EMBED_MODEL}) == "granite"
    emb = RecordingEmbedder()
    Retriever(tiny_index[0], count=count, embedder=emb).candidates("join")
    assert emb.prompts == [None]  # granite: raw query, no instruction


def test_draft_query_appends_the_drafts_code_only():
    from polars_llm.rag.retrieve import draft_query

    draft = "Here you go:\n```python\ndef solve(df):\n    return df.unpivot(index='id')\n```\nHope it helps."
    q = draft_query("Reshape the table from wide to long. ", draft)
    assert q == "Reshape the table from wide to long.\n\ndef solve(df):\n    return df.unpivot(index='id')"


def test_draft_query_falls_back_to_instruction_without_code():
    from polars_llm.rag.retrieve import build_query, draft_query

    assert draft_query("Do X.", None) == "Do X."
    assert draft_query("Do X.", "") == "Do X."
    assert draft_query("Do X.", "```python\n\n```") == "Do X."
    assert build_query("instruction", "Do X.", "```python\nx = 1\n```") == "Do X."
    with pytest.raises(ValueError):
        build_query("hyde", "Do X.")


def test_draft_query_brings_in_the_api_vocabulary(tmp_path):
    from polars_llm.rag.index import build_index
    from polars_llm.rag.retrieve import Retriever, build_query

    path = tmp_path / "chunks.jsonl"
    write_chunks([
        Chunk("api:DataFrame.unpivot", "api", "d.rst", "polars.DataFrame.unpivot", "Reshape",
              ["DataFrame.unpivot"], "Unpivot a DataFrame from wide to long format.\n\n"
              ">>> df.unpivot(index='a', on=['b', 'c'], variable_name='month')", 20),
        Chunk("api:DataFrame.product", "api", "d.rst", "polars.DataFrame.product", "Aggregation",
              ["DataFrame.product"], "Aggregate the columns to their product value per product.", 9),
    ], path)  # fmt: skip
    build_index(path, tmp_path / "idx", embedder=FakeEmbedder())
    r = Retriever(tmp_path / "idx", count=count, embedder=FakeEmbedder())
    instruction = "Turn the per-product columns into one row per product and month."
    draft = "```python\ndef solve(df):\n    return df.unpivot(index='store', variable_name='month')\n```"
    def fts_rank(query):
        return {c.id: c.fts_rank for c in r.candidates(query)}["api:DataFrame.unpivot"]

    # The column name "product" pulls in DataFrame.product; the draft's code
    # moves DataFrame.unpivot to the top of the BM25 ranking.
    assert r.retrieve(build_query("instruction", instruction), k=1)[0].id == "api:DataFrame.product"
    assert fts_rank(build_query("instruction", instruction)) == 2
    assert fts_rank(build_query("draft", instruction, draft)) == 1


def test_load_drafts(tmp_path):
    import json

    from polars_llm.rag.retrieve import load_drafts

    p = tmp_path / "generations.jsonl"
    p.write_text("".join(json.dumps({"task_id": t, "output": o, "latency_s": 1.5}) + "\n"
                         for t, o in [("a_001", "x"), ("a_002", "y")]), encoding="utf-8")
    d = load_drafts(p)
    assert d["a_002"]["output"] == "y" and d["a_001"]["latency_s"] == 1.5
