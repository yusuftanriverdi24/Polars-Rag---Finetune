"""End-to-end generate -> score pipeline test on CPU using the mock model.

Proves the whole flow (prompt build -> generation -> sandbox scoring) works
without a GPU, before any Colab run. The mock answers even-indexed tasks with
their reference solution (-> pass) and odd-indexed tasks with pandas-style
broken code (-> pandas_leak).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate as generate_script  # noqa: E402
import score as score_script  # noqa: E402

from polars_llm.bench.schema import load_tasks  # noqa: E402
from polars_llm.models.mock import make_reference_mock, solve_param_names  # noqa: E402


def test_solve_param_names():
    assert solve_param_names("def solve(df: pl.DataFrame) -> pl.DataFrame:") == ["df"]
    assert solve_param_names(
        "def solve(orders: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:"
    ) == ["orders", "customers"]
    assert solve_param_names("def solve(lf: pl.LazyFrame):") == ["lf"]


def test_generate_then_score_end_to_end(tmp_path):
    tasks = load_tasks("benchmark/tasks.jsonl")[:6]
    model = make_reference_mock(tasks)

    out_dir = tmp_path / "results"
    meta = generate_script.run_generation(tasks, model, "mock", out_dir, seed=123)

    # generations.jsonl exists with one well-formed record per task.
    gen_path = out_dir / "mock" / "generations.jsonl"
    lines = [json.loads(l) for l in gen_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 6
    assert {"task_id", "output", "latency_s", "prompt_tokens", "completion_tokens"} <= set(lines[0])
    assert [l["task_id"] for l in lines] == [t.id for t in tasks]

    # meta captures model + seed.
    assert meta["model"] == "mock"
    assert meta["seed"] == 123
    assert (out_dir / "mock" / "generation_meta.json").exists()

    # Score the generations and check the expected mix of outcomes.
    results = score_script.run_scoring("mock", out_dir, "benchmark/tasks.jsonl", seed=123)
    m = results["metrics"]
    assert m["total"] == 6
    assert m["pass"] == 3  # even-indexed references
    assert m["status_counts"].get("pandas_leak") == 3  # odd-indexed broken code
    assert (out_dir / "mock" / "scores.json").exists()


def test_score_without_generations_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        score_script.run_scoring("nope", tmp_path, "benchmark/tasks.jsonl")
