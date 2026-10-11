"""Determinism checks: order-variation classification and the validate_tasks gate.

Nondeterminism is simulated with explicit random shuffles (rows via an unseeded
``sample(shuffle=True)``, columns via ``random.shuffle``) so the tests don't rely
on thread scheduling to show variation.
"""

import json

import polars as pl

from polars_llm.bench import validate_tasks
from polars_llm.bench.determinism import (
    order_variation,
    run_repeated,
    task_determinism_problems,
)
from polars_llm.bench.schema import Task

SETUP = 'inputs = {"df": pl.DataFrame({"g": list(range(30)), "v": list(range(30))})}'
STABLE = "def solve(df):\n    return df.sort('g')"
SHUFFLE_ROWS = "def solve(df):\n    return df.sample(fraction=1.0, shuffle=True)"
SHUFFLE_COLS = (
    "import random\n"
    "def solve(df):\n"
    "    cols = df.columns[:]\n"
    "    random.shuffle(cols)\n"
    "    return df.select(cols)"
)


def _task(reference, **check):
    return Task.from_dict(
        {
            "id": "det_001",
            "difficulty": "easy",
            "category": "group_by",
            "instruction": "x",
            "setup_code": SETUP,
            "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
            "reference_solution": reference,
            "check": check,
        }
    )


# --- order_variation on synthetic frames -------------------------------------


def test_identical_frames_do_not_vary():
    f = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert not order_variation([f, f.clone(), f.clone()]).any


def test_row_order_variation():
    f = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    v = order_variation([f, f.reverse()])
    assert v.row_order and not v.column_order and not v.content


def test_column_order_variation():
    f = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    v = order_variation([f, f.select("b", "a")])
    assert v.column_order and not v.row_order and not v.content
    assert v.distinct_column_orders == [["a", "b"], ["b", "a"]]


def test_content_variation_and_errors():
    f = pl.DataFrame({"a": [1, 2]})
    assert order_variation([f, pl.DataFrame({"a": [1, 3]})]).content
    assert order_variation([f, None]).content  # one run errored
    assert not order_variation([None, None]).any  # consistently failing is stable


# --- repeated runs in the sandbox ---------------------------------------------


def test_run_repeated_stable_code():
    res = run_repeated(SETUP, STABLE, n=5)
    assert len(res.frames) == 5 and all(f is not None for f in res.frames)
    assert not order_variation(res.frames).any


def test_run_repeated_detects_row_and_column_shuffles():
    assert order_variation(run_repeated(SETUP, SHUFFLE_ROWS, n=10).frames).row_order
    assert order_variation(run_repeated(SETUP, SHUFFLE_COLS, n=20).frames).column_order


def test_run_repeated_records_errors():
    res = run_repeated(SETUP, "def solve(df):\n    raise ValueError('boom')", n=3)
    assert res.frames == [None, None, None]
    assert all(e["type"] == "ValueError" for e in res.errors)


# --- task-level problems respect the check flags -------------------------------


def test_task_problems_flag_untolerated_variation_only():
    assert task_determinism_problems(_task(STABLE), n=5) == []

    problems = task_determinism_problems(_task(SHUFFLE_ROWS), n=10)
    assert problems and "row order" in problems[0]
    assert task_determinism_problems(_task(SHUFFLE_ROWS, ignore_row_order=True), n=10) == []

    problems = task_determinism_problems(_task(SHUFFLE_COLS), n=20)
    assert problems and "column order" in problems[0]
    assert task_determinism_problems(_task(SHUFFLE_COLS, ignore_column_order=True), n=20) == []


def test_validate_tasks_fails_on_order_unstable_reference(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(_task(SHUFFLE_ROWS).to_dict()) + "\n", encoding="utf-8")
    assert validate_tasks.main(["--tasks", str(path), "--determinism-runs", "10"]) == 1
    # Tolerated by the check flag -> passes.
    path.write_text(
        json.dumps(_task(SHUFFLE_ROWS, ignore_row_order=True).to_dict()) + "\n", encoding="utf-8"
    )
    assert validate_tasks.main(["--tasks", str(path), "--determinism-runs", "10"]) == 0
    # Disabled -> the unstable reference is not checked.
    path.write_text(json.dumps(_task(SHUFFLE_ROWS).to_dict()) + "\n", encoding="utf-8")
    assert validate_tasks.main(["--tasks", str(path), "--determinism-runs", "0"]) == 0


# --- static pivot column-order guard ------------------------------------------


def test_pivot_task_requires_ignore_column_order():
    from polars_llm.bench.determinism import pivot_column_order_problem

    pivot = "def solve(df):\n    return df.pivot(on='g', index='v', values='v')"
    assert pivot_column_order_problem(_task(pivot)) is not None
    assert pivot_column_order_problem(_task(pivot, ignore_column_order=True)) is None
    # An explicit column order selected after the pivot is fine.
    fixed = "def solve(df):\n    return df.pivot(on='g', index='v', values='v').select('v', '0')"
    assert pivot_column_order_problem(_task(fixed)) is None
    # unpivot is not pivot.
    unpivot = "def solve(df):\n    return df.unpivot(index='g')"
    assert pivot_column_order_problem(_task(unpivot)) is None


def test_every_benchmark_pivot_task_passes_the_guard():
    from polars_llm.bench.determinism import pivot_column_order_problem
    from polars_llm.bench.schema import load_tasks

    assert [t.id for t in load_tasks("benchmark/tasks.jsonl") if pivot_column_order_problem(t)] == []
