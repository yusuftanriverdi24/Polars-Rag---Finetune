"""End-to-end runner tests: extraction -> sandbox -> compare -> classify.

Exercises deliberately correct, wrong, pandas-style and deprecated outputs
through the full scoring pipeline using real subprocesses and Polars.
"""

from polars_llm.bench.classify import (
    DEPRECATED_API,
    PANDAS_LEAK,
    PASS,
    WRONG_RESULT,
)
from polars_llm.bench.runner import score
from polars_llm.bench.schema import Task

TASK = Task.from_dict(
    {
        "id": "t_sel",
        "difficulty": "easy",
        "category": "select_filter",
        "instruction": "Return just column a.",
        "setup_code": 'inputs = {"df": pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})}',
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "reference_solution": "def solve(df):\n    return df.select('a')",
        "check": {},
    }
)


def test_pipeline_classifies_each_output_kind():
    outputs = [
        ("t_sel", "def solve(df):\n    return df.select('a')"),  # correct
        ("t_sel", "```python\ndef solve(df):\n    return df.select('a')\n```"),  # correct, fenced
        ("t_sel", "def solve(df):\n    return df.select('b')"),  # wrong
        ("t_sel", "def solve(df):\n    return df.groupby('a').sum()"),  # pandas leak
        ("t_sel", "def solve(df):\n    return df.with_row_count('i')"),  # deprecated/removed
    ]
    results = score([TASK], outputs, timeout_s=15)
    statuses = [r["status"] for r in results["tasks"]]

    assert statuses == [PASS, PASS, WRONG_RESULT, PANDAS_LEAK, DEPRECATED_API]

    metrics = results["metrics"]
    assert metrics["total"] == 5
    assert metrics["pass"] == 2
    assert metrics["pass_at_1"] == round(2 / 5, 4)
    assert metrics["deprecated_api_count"] == 1
    assert metrics["pass_at_1_by_category"]["select_filter"]["total"] == 5
    assert results["meta"]["polars_version"] == "2.0.0"


def test_unknown_task_id_is_recorded_not_crashed():
    results = score([TASK], [("does_not_exist", "def solve(df): return df")])
    assert results["tasks"][0]["status"] == "runtime_error"
    assert "unknown task" in results["tasks"][0]["reason"]
