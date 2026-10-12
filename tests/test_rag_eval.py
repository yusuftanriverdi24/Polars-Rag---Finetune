"""API-recall metric: dynamic API-name extraction and recall aggregation."""

import pytest

from polars_llm.bench.schema import Task
from polars_llm.rag.chunk import Chunk
from polars_llm.rag.eval_retrieval import (
    CORE_NAMES,
    api_recall,
    build_code_map,
    evaluate,
    solution_api_names,
    summarize,
)
from polars_llm.rag.retrieve import Retrieved

NAMES = [
    "col", "lit", "len", "Expr.alias", "Expr.sum", "Expr.over", "Expr.str.to_uppercase",
    "DataFrame.join", "DataFrame.select", "DataFrame.filter", "DataFrame.group_by",
    "DataFrame.with_columns", "GroupBy.agg", "LazyFrame.collect", "LazyFrame.filter",
]  # fmt: skip
MODULES = {"GroupBy.agg": "polars.dataframe.group_by"}


@pytest.fixture(scope="module")
def code_map():
    chunks = [
        Chunk(f"api:{n}", "api", "x.rst", f"polars.{n}", "", [n], "", 1,
              extra={"module": MODULES.get(n, "polars"), "directive": "autosummary"})
        for n in NAMES
    ]  # fmt: skip
    return build_code_map(chunks)


SETUP = 'inputs = {"df": pl.DataFrame({"k": [1, 2, 2], "v": [1, 2, 3]}), "other": pl.DataFrame({"k": [1, 2], "s": ["a", "b"]})}'


def _names(code, code_map, setup=SETUP):
    return solution_api_names(setup, code, code_map)


def test_extracts_methods_namespaces_and_functions(code_map):
    code = (
        "def solve(df, other):\n"
        "    j = df.join(other, on='k', how='inner')\n"
        "    return j.with_columns(pl.col('s').str.to_uppercase().alias('S'))\n"
    )
    assert _names(code, code_map) == [
        "DataFrame.join", "DataFrame.with_columns", "Expr.alias", "Expr.str.to_uppercase", "col",
    ]  # fmt: skip


def test_group_by_agg_and_lazy(code_map):
    code = (
        "def solve(df, other):\n"
        "    return df.lazy().filter(pl.col('v') > 1).collect().group_by('k').agg(pl.col('v').sum())\n"
    )
    names = _names(code, code_map)
    assert {"LazyFrame.filter", "LazyFrame.collect", "DataFrame.group_by", "GroupBy.agg", "Expr.sum", "col"} <= set(names)


def test_internal_calls_and_operators_are_not_counted(code_map):
    # `pl.col('v') > 1` calls Expr.__gt__, which internally calls `lit`; and
    # DataFrame.filter calls LazyFrame.filter/collect internally. None of those
    # internal calls should count, only what the solution itself calls.
    code = "def solve(df, other):\n    return df.filter(pl.col('v') > 1)\n"
    assert _names(code, code_map) == ["DataFrame.filter", "col"]


def test_setup_code_calls_are_not_counted(code_map):
    setup = SETUP + "\n_ = pl.lit(1)"
    assert "lit" not in _names("def solve(df, other):\n    return df\n", code_map, setup=setup)


def test_api_recall_math():
    retrieved = [
        Retrieved("a", "api", "t", "", ["DataFrame.join"], "", 1),
        Retrieved("b", "guide", "t", "", ["Expr.alias", "col"], "", 1),
    ]
    recall, found, missing = api_recall(["DataFrame.join", "Expr.over", "col", "Expr.sum"], retrieved)
    assert recall == 0.5
    assert found == ["DataFrame.join", "col"]
    assert missing == ["Expr.over", "Expr.sum"]
    assert api_recall([], retrieved) == (None, [], [])


def test_summarize_macro_means_per_category_ignoring_none():
    rows = [
        {"category": "join", "recall": 1.0},
        {"category": "join", "recall": 0.5},
        {"category": "string", "recall": 0.0},
        {"category": "string", "recall": None},  # nothing needed -> not scored
    ]
    s = summarize(rows, "recall")
    assert s["by_category"] == {"join": 0.75, "string": 0.0}
    assert s["overall"] == 0.5
    assert s["n_tasks_scored"] == 3


def test_evaluate_core_and_non_core_recall():
    task = Task("join_001", "easy", "join", "Join them.", "inputs = {}", "def solve():", "def solve(): ...")
    needed = {"join_001": ["DataFrame.join", "col", "Expr.over"]}
    cands = {"join_001": [Retrieved("a", "api", "t", "", ["DataFrame.join", "col"], "x", 1)]}
    count = lambda s: len(s.split())  # noqa: E731
    e = evaluate([task], cands, needed, count, k=5, budget=100, details=True)
    row = e["tasks"][0]
    assert row["recall"] == pytest.approx(2 / 3)
    assert "col" in CORE_NAMES
    assert row["recall_non_core"] == 0.5  # DataFrame.join found, Expr.over missing
    assert row["missing"] == ["Expr.over"]
    assert row["retrieved"][0]["id"] == "a"


# --- Arm B2 selection rules ---------------------------------------------------------


def _cfg(variant, nc, all_, budget=2000, k=5, mode="instruction"):
    return {"variant": variant, "query_mode": mode, "k": k, "budget_tokens": budget,
            "recall": all_, "recall_non_core": nc}


def test_choose_best_by_non_core_recall_then_tiebreaks():
    from polars_llm.rag.eval_retrieval_v2 import choose_best

    assert choose_best([_cfg("a", 0.30, 0.5), _cfg("b", 0.35, 0.1)])["variant"] == "b"
    # Equal non-core -> higher all-names recall.
    assert choose_best([_cfg("a", 0.3, 0.2), _cfg("b", 0.3, 0.25)])["variant"] == "b"
    # Then the smaller budget, then the smaller k, then the instruction query.
    assert choose_best([_cfg("a", 0.3, 0.2, budget=3000), _cfg("a", 0.3, 0.2, budget=2000)])["budget_tokens"] == 2000
    assert choose_best([_cfg("a", 0.3, 0.2, k=8), _cfg("a", 0.3, 0.2, k=5)])["k"] == 5
    assert choose_best([_cfg("c", 0.3, 0.2, mode="draft"), _cfg("a", 0.3, 0.2)])["query_mode"] == "instruction"


def test_better_embedder_uses_grid_mean_and_prefers_granite_on_ties():
    from polars_llm.rag.eval_retrieval_v2 import better_embedder

    def grid(*vals):
        return [{"recall_non_core": {"overall": v}} for v in vals]

    assert better_embedder(grid(0.3, 0.3), grid(0.2, 0.5)) == "qwen3"
    assert better_embedder(grid(0.3, 0.3), grid(0.1, 0.5)) == "granite"
    assert better_embedder(grid(0.3), grid(0.3)) == "granite"


def test_required_max_seq_length():
    from polars_llm.rag.eval_retrieval_v2 import required_max_seq_length

    assert required_max_seq_length(2347) == 4096
    assert required_max_seq_length(4400) == 5632  # 4400 + 768 = 5168 -> next multiple of 512
    assert required_max_seq_length(4096 - 768) == 4096
