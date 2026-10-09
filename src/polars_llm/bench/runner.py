"""Benchmark runner: score model outputs and write a results JSON.

Given a list of ``(task_id, model_output)`` pairs it:

1. extracts code from each model output,
2. computes each task's expected output from its reference solution (once,
   cached), under the pinned Polars version,
3. runs the candidate in the sandbox,
4. compares the output and classifies the outcome, and
5. writes a results JSON with per-task records and aggregate metrics.

CLI::

    # score reference solutions as a smoke test (expect ~100% pass)
    python -m polars_llm.bench.runner --tasks benchmark/tasks.jsonl \\
        --use-reference --out results/reference.json

    # score real model outputs (JSONL: {"id": ..., "output": ...} per line)
    python -m polars_llm.bench.runner --tasks benchmark/tasks.jsonl \\
        --outputs outputs.jsonl --out results/arm_a.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import polars as pl

from .classify import classify
from .compare import compare_frames
from .executor import DEFAULT_TIMEOUT_S, ExecutionResult, compute_expected, run_solution
from .extract import extract_code
from .metrics import aggregate
from .schema import Task, load_tasks, tasks_by_id


def _error_summary(result: ExecutionResult) -> dict | None:
    if result.error is None:
        return None
    return {
        "type": result.error.type,
        "message": result.error.message,
        "traceback": result.error.traceback,
    }


def score(
    tasks: list[Task],
    outputs: list[tuple[str, str]],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    seed: int = 0,
    run_meta: dict | None = None,
) -> dict:
    """Score ``outputs`` (task_id, raw_model_output) against ``tasks``."""

    by_id = tasks_by_id(tasks)
    expected_cache: dict[str, ExecutionResult] = {}

    records: list[dict] = []
    for task_id, model_output in outputs:
        task = by_id.get(task_id)
        if task is None:
            records.append(
                {
                    "id": task_id,
                    "difficulty": "unknown",
                    "category": "unknown",
                    "status": "runtime_error",
                    "deprecated_api": False,
                    "pandas_leak_detected": False,
                    "reason": f"unknown task id {task_id!r}",
                    "duration_s": 0.0,
                }
            )
            continue

        # Expected output, computed once per task from its reference solution.
        if task_id not in expected_cache:
            expected_cache[task_id] = compute_expected(
                task.setup_code, task.reference_solution, timeout_s=timeout_s
            )
        expected = expected_cache[task_id]
        if not expected.ok or expected.frame is None:
            records.append(
                {
                    "id": task_id,
                    "difficulty": task.difficulty,
                    "category": task.category,
                    "status": "runtime_error",
                    "deprecated_api": False,
                    "pandas_leak_detected": False,
                    "reason": "reference solution failed to produce expected output "
                    f"(phase={expected.phase})",
                    "reference_error": _error_summary(expected),
                    "duration_s": 0.0,
                }
            )
            continue

        code = extract_code(model_output)
        exec_result = run_solution(task.setup_code, code, timeout_s=timeout_s)

        comparison = None
        if exec_result.ok and exec_result.frame is not None:
            comparison = compare_frames(exec_result.frame, expected.frame, task.check)

        classified = classify(code=code, exec_result=exec_result, comparison=comparison)

        records.append(
            {
                "id": task_id,
                "difficulty": task.difficulty,
                "category": task.category,
                "status": classified.status,
                "deprecated_api": classified.deprecated_api,
                "pandas_leak_detected": classified.pandas_leak_detected,
                "reason": classified.reason,
                "warnings": classified.warnings,
                "error": _error_summary(exec_result),
                "duration_s": round(exec_result.duration_s, 4),
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "extracted_code": code,
            }
        )

    meta = {
        "polars_version": pl.__version__,
        "python_version": sys.version.split()[0],
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seed": seed,
        "timeout_s": timeout_s,
        "n_outputs": len(outputs),
        "n_tasks": len(tasks),
    }
    if run_meta:
        meta.update(run_meta)

    return {"meta": meta, "metrics": aggregate(records), "tasks": records}


def _load_outputs(path: str | Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "output" not in obj:
                raise ValueError(f"{path}:{lineno}: each line needs 'id' and 'output'")
            out.append((str(obj["id"]), str(obj["output"])))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score model outputs against the benchmark.")
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl", help="path to tasks.jsonl")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--outputs", help="JSONL of {'id','output'} model outputs to score")
    src.add_argument(
        "--use-reference",
        action="store_true",
        help="score each task's own reference solution (smoke test; expect ~100%% pass)",
    )
    parser.add_argument("--out", required=True, help="path to write the results JSON")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)

    if args.use_reference:
        outputs = [(t.id, t.reference_solution) for t in tasks]
        run_meta = {"arm": "reference_smoke_test"}
    else:
        outputs = _load_outputs(args.outputs)
        run_meta = {"outputs_file": args.outputs}

    results = score(tasks, outputs, timeout_s=args.timeout, seed=args.seed, run_meta=run_meta)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    m = results["metrics"]
    print(f"Wrote {out_path}")
    print(f"pass@1: {m['pass']}/{m['total']} = {m['pass_at_1']:.1%}")
    print(f"status counts: {m['status_counts']}")
    print(f"deprecated_api flagged: {m['deprecated_api_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
