"""Validate every benchmark task's reference solution.

Runs each task's reference solution in the sandbox under the pinned Polars
version and FAILS (non-zero exit) if any task:

- fails to load/validate against the schema,
- raises an error while producing its expected output,
- raises a deprecation warning, or
- produces output whose row or column order (or content) varies between runs
  in a way the task's ``check`` flags do not tolerate. Polars 2.0's streaming
  engine does not guarantee the order of ``group_by`` / ``join`` / ``unique`` /
  ``pivot`` / ``unpivot`` output, so each reference is run ``--determinism-runs``
  times (default 50; 0 disables) and any untolerated variation is a failure.
  Pivot tasks must also enable ``ignore_column_order`` (or select an explicit
  column order after the pivot), since a correct *candidate* can produce the
  pivoted columns in a different order even when the reference's is stable.

This guards the invariant from PLAN.md Section 4: the expected output is
computed by running clean, up-to-date reference code.

Usage::

    python -m polars_llm.bench.validate_tasks [--tasks benchmark/tasks.jsonl] [--determinism-runs 50]
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from .determinism import DEFAULT_RUNS, pivot_column_order_problem, task_determinism_problems
from .executor import DEFAULT_TIMEOUT_S, compute_expected
from .schema import load_tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--determinism-runs",
        type=int,
        default=DEFAULT_RUNS,
        help="run each reference this many times to check order stability (0 = skip)",
    )
    args = parser.parse_args(argv)

    try:
        tasks = load_tasks(args.tasks)
    except (OSError, ValueError) as exc:
        print(f"FAIL: could not load tasks: {exc}", file=sys.stderr)
        return 1

    print(f"Validating {len(tasks)} tasks against Polars {pl.__version__}\n")

    failures: list[str] = []
    for task in tasks:
        result = compute_expected(task.setup_code, task.reference_solution, timeout_s=args.timeout)
        problems = []
        if not result.ok:
            err = result.error
            detail = f"{err.type}: {err.message}" if err else "unknown error"
            problems.append(f"reference failed (phase={result.phase}): {detail}")
        for w in result.deprecation_warnings:
            problems.append(f"deprecation warning: {w['category']}: {w['message']}")
        pivot_problem = pivot_column_order_problem(task)
        if pivot_problem:
            problems.append(pivot_problem)
        if result.ok and args.determinism_runs > 0:
            problems.extend(task_determinism_problems(task, n=args.determinism_runs))

        if problems:
            mark = "FAIL"
            for p in problems:
                failures.append(f"{task.id}: {p}")
        else:
            mark = "ok  "
        shape = result.frame.shape if result.frame is not None else None
        print(f"  [{mark}] {task.id:<22} {task.difficulty:<6} {task.category:<14} shape={shape}")
        for p in problems:
            print(f"         - {p}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} problem(s) across the task set:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    determinism = (
        f", and give order-stable output over {args.determinism_runs} runs (within check flags)"
        if args.determinism_runs > 0
        else ""
    )
    print(
        f"OK: all {len(tasks)} reference solutions run cleanly with no deprecation "
        f"warnings{determinism}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
