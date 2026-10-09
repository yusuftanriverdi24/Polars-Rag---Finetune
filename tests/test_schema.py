import json

import pytest

from polars_llm.bench.schema import CheckFlags, Task, load_tasks

VALID = {
    "id": "t1",
    "difficulty": "easy",
    "category": "select_filter",
    "instruction": "do a thing",
    "setup_code": "inputs = {}",
    "signature": "def solve(df): ...",
    "reference_solution": "def solve(df): return df",
}


def test_from_dict_defaults_check_flags():
    t = Task.from_dict(VALID)
    assert t.check == CheckFlags(False, False)


def test_missing_field_raises():
    bad = {k: v for k, v in VALID.items() if k != "instruction"}
    with pytest.raises(ValueError, match="missing required fields"):
        Task.from_dict(bad)


def test_invalid_difficulty():
    with pytest.raises(ValueError, match="invalid difficulty"):
        Task.from_dict({**VALID, "difficulty": "trivial"})


def test_invalid_category():
    with pytest.raises(ValueError, match="invalid category"):
        Task.from_dict({**VALID, "category": "frobnicate"})


def test_unknown_check_flag():
    with pytest.raises(ValueError, match="unknown check flags"):
        Task.from_dict({**VALID, "check": {"ignore_everything": True}})


def test_load_tasks_roundtrip_and_duplicate_detection(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        json.dumps(VALID) + "\n" + json.dumps({**VALID, "id": "t2"}) + "\n",
        encoding="utf-8",
    )
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["t1", "t2"]

    p.write_text(json.dumps(VALID) + "\n" + json.dumps(VALID) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task id"):
        load_tasks(p)


def test_load_tasks_skips_blank_and_comment_lines(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n# a comment\n" + json.dumps(VALID) + "\n", encoding="utf-8")
    assert len(load_tasks(p)) == 1


def test_benchmark_tasks_file_loads():
    # The committed benchmark set must always load, be unique, and hit the
    # planned 40/50/30 difficulty split across 120 tasks.
    from collections import Counter

    tasks = load_tasks("benchmark/tasks.jsonl")
    assert len(tasks) == 120
    assert len({t.id for t in tasks}) == 120
    by_diff = Counter(t.difficulty for t in tasks)
    assert by_diff == {"easy": 40, "medium": 50, "hard": 30}
    # Every category present.
    assert len({t.category for t in tasks}) == 9
