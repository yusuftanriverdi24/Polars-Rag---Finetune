"""Leakage checks vs the benchmark, the grouped split, downsampling and
categories (training data v1)."""

import numpy as np

from polars_llm.bench.schema import Task
from polars_llm.datagen.build_sft import grouped_split, make_example
from polars_llm.datagen.candidates import assign_category, difficulty, downsample
from polars_llm.datagen.leakage import (
    bench_refs,
    calibrate_instruction_threshold,
    instruction_api_leaks,
    normalize_solution,
    solution_leak,
)

BENCH = Task(
    id="group_by_001", difficulty="easy", category="group_by", instruction="Total sales per region.",
    setup_code="inputs = {}", signature="def solve(df):",
    reference_solution='def solve(df):\n    return df.group_by("region").agg(pl.col("sales").sum())',
)  # fmt: skip
REFS = bench_refs([BENCH], {"group_by_001": ["DataFrame.group_by", "GroupBy.agg", "col", "Expr.sum"]})


def test_normalization_ignores_names_annotations_imports():
    a = 'import polars.selectors as cs\n\ndef solve(orders: pl.DataFrame) -> pl.DataFrame:\n    """doc"""\n    out = orders.head(2)\n    return out'
    b = "def solve(df):\n    x = df.head(2)\n    return x"
    assert normalize_solution(a) == normalize_solution(b)
    assert normalize_solution('def solve(df):\n    return df.head(2)') != normalize_solution('def solve(df):\n    return df.head(3)')
    assert normalize_solution('def solve(df):\n    return df.head(2)', mask_literals=True) == normalize_solution(
        'def solve(df):\n    return df.head(3)', mask_literals=True)


def test_near_identical_solution_is_a_leak():
    sol = 'def solve(t: pl.DataFrame) -> pl.DataFrame:\n    return t.group_by("region").agg(pl.col("sales").sum())'
    names = {"DataFrame.group_by", "GroupBy.agg", "col", "Expr.sum"}
    assert solution_leak(sol, names, REFS) == ("solution", "group_by_001")


def test_same_structure_needs_enough_api_names():
    sol = 'def solve(df):\n    return df.group_by("k").agg(pl.col("v").sum())'
    many = {"DataFrame.group_by", "GroupBy.agg", "Expr.sum"}
    assert solution_leak(sol, many, REFS)[0] in {"solution", "solution_structure"}
    short = 'def solve(df):\n    return df.head(5)'
    refs = bench_refs([Task(id="t", difficulty="easy", category="select_filter", instruction="x",
                            setup_code="inputs = {}", signature="def solve(df):",
                            reference_solution='def solve(df):\n    return df.head(3)')], {"t": ["DataFrame.head"]})
    assert solution_leak(short, {"DataFrame.head"}, refs) is None  # one-call pattern: not leakage


def test_identical_api_set_is_a_leak():
    # Structurally different code (a helper variable, two keys, an alias) but the same API set.
    sol = (
        "def solve(df):\n"
        '    total_expression = pl.col("amount_in_dollars").sum()\n'
        '    return df.group_by("store_identifier", "calendar_day").agg(total_expression)'
    )
    names = {"DataFrame.group_by", "GroupBy.agg", "col", "Expr.sum"}
    assert solution_leak(sol, names, REFS) == ("api_set", "group_by_001")
    assert solution_leak(sol, names - {"Expr.sum"} | {"Expr.mean"}, REFS) is None


def _unit(*rows):
    a = np.array(rows, dtype=float)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_threshold_is_calibrated_on_distinct_benchmark_tasks():
    bench = _unit([1, 0, 0], [0.8, 0.6, 0], [0, 0, 1])  # t0~t1 at 0.8, t2 far from both
    cal = calibrate_instruction_threshold(bench, ["t0", "t1", "t2"])
    assert cal["threshold"] == 0.8 and cal["nn_cosine_distribution"]["max"] == 0.8
    assert cal["nn_cosine_distribution"]["min"] == 0.0
    assert cal["closest_benchmark_pairs"][0] == {"a": "t0", "b": "t1", "cosine": 0.8}  # each pair listed once
    assert len(cal["closest_benchmark_pairs"]) == 2


def test_leak_needs_similar_instruction_and_same_api_set():
    bench = _unit([1, 0], [0, 1])
    bench_names = [["DataFrame.group_by", "GroupBy.agg", "Expr.sum", "col"], ["DataFrame.pivot"]]
    instr = _unit([1, 0.05], [1, 0.05], [0.2, 1], [1, 1])
    names = [
        ["DataFrame.group_by", "GroupBy.agg", "Expr.sum", "DataFrame.select"],  # same non-core set, close -> leak
        ["DataFrame.group_by", "GroupBy.agg", "Expr.mean"],  # close but different APIs -> kept
        ["DataFrame.pivot"],  # same APIs as bench 1, cosine 0.98 -> leak even with a single API
        ["DataFrame.pivot"],  # same APIs, but cosine 0.71 < threshold -> kept
    ]
    out = instruction_api_leaks(instr, bench, names, bench_names, threshold=0.9)
    assert out[0]["leak"][0] == 0 and out[1]["leak"] is None and out[2]["leak"][0] == 1 and out[3]["leak"] is None
    assert out[1]["nearest"] == 0 and out[1]["cosine"] > 0.99  # the closest pair is still reported
    # A match with a less similar task above the threshold counts too, not only the nearest one.
    out = instruction_api_leaks(_unit([1, 0.9]), _unit([1, 0.8], [1, 1]), [["DataFrame.pivot"]],
                                [["DataFrame.unpivot"], ["DataFrame.pivot"]], threshold=0.9)
    assert out[0]["leak"][0] == 1


def _ex(i, page):
    return {"id": f"e{i}", "page": page}


def test_grouped_split_keeps_pages_together_and_is_deterministic():
    examples = [_ex(i, f"p{i % 37}") for i in range(400)]
    train, dev = grouped_split(examples)
    assert not ({e["page"] for e in train} & {e["page"] for e in dev})
    assert 0.07 <= len(dev) / len(examples) <= 0.16
    assert grouped_split(examples) == (train, dev)


def test_make_example_is_a_valid_task():
    rec = {"id": "c1", "difficulty": "easy", "category": "string", "setup_code": "inputs = {}",
           "signature": "def solve() -> pl.DataFrame:", "reference_solution": "def solve():\n    return pl.DataFrame()",
           "check": {"ignore_row_order": False}, "source": "docstring", "page": "Expr.x", "api_names": [],
           "multi_step": False}
    ex = make_example(rec, "concise", "Do it.")
    assert ex["id"] == "c1__concise" and Task.from_dict(ex).instruction == "Do it."


def test_categories_difficulty_and_downsampling():
    assert assign_category(["DataFrame.group_by", "GroupBy.agg"]) == "group_by"
    assert assign_category(["DataFrame.with_columns", "Expr.str.to_uppercase"]) == "string"
    assert assign_category(["DataFrame.join", "Expr.str.strip_chars"]) == "join"
    assert assign_category(["DataFrame.unpivot"]) == "reshape"
    assert assign_category(["Expr.over", "Expr.sum"]) == "window"
    assert assign_category(["DataFrame.select", "Expr.list.len"]) == "list_struct"
    assert assign_category(["DataFrame.select", "col"]) == "select_filter"
    assert (difficulty(1), difficulty(3), difficulty(4)) == ("easy", "medium", "hard")

    recs = [{"id": f"h{i}", "multi_step": False, "non_core_api_names": ["DataFrame.head"], "api_names": []}
            for i in range(10)]
    recs += [{"id": f"m{i}", "multi_step": True, "non_core_api_names": ["a", "b"], "api_names": []} for i in range(5)]
    kept, dropped = downsample(recs)
    assert sum(r["id"].startswith("h") for r in kept) == 2 and sum(r["id"].startswith("m") for r in kept) == 5
    assert dropped == {"trivial": 8}


def test_shared_core_boilerplate_alone_is_not_a_leak():
    ref = Task(id="string_011", difficulty="easy", category="string", instruction="x", setup_code="inputs = {}",
               signature="def solve(df):", reference_solution='def solve(df):\n    return df.with_columns(pl.col("id").str.zfill(4))')
    refs = bench_refs([ref], {"string_011": ["DataFrame.with_columns", "col", "Expr.str.zfill"]})
    sol = 'def solve(df: pl.DataFrame) -> pl.DataFrame:\n    return df.with_columns(all=pl.col("a").list.all())'
    assert solution_leak(sol, {"DataFrame.with_columns", "col", "Expr.list.all"}, refs) is None
