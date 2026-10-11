"""Determinism audit of the benchmark (and of an arm's generations).

1. Runs every task's reference solution ``--runs`` times and flags any task
   whose output varies in row order, column order, or content.
2. Re-runs every generation of an arm whose code touches ``group_by``,
   ``join``, ``unique``, ``pivot`` or ``unpivot`` ``--runs`` times, scoring each
   run against a fresh reference run under the task's *current* ``check``
   flags, and flags any whose pass/fail status changes between runs.

Writes the raw findings to ``--out`` (JSON) and prints a summary. Used for the
benchmark-v1.1 determinism audit recorded in ``benchmark/AUDIT.md``.

Usage::

    python benchmark/determinism_audit.py --runs 50 \\
        --scores results/A/scores.json --out benchmark/determinism_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from polars_llm.bench.compare import compare_frames
from polars_llm.bench.determinism import order_variation, run_repeated
from polars_llm.bench.schema import load_tasks

TOUCHES_UNORDERED_OP = re.compile(r"\bgroup_by\w*\(|\.join\w*\(|\.unique\(|\.pivot\(|\.unpivot\(")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--scores", default="results/A/scores.json")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--out", default="benchmark/determinism_audit.json")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    by_id = {t.id: t for t in tasks}

    # --- 1. reference solutions ---------------------------------------------
    reference_frames = {}
    reference_findings = []
    for t in tasks:
        res = run_repeated(t.setup_code, t.reference_solution, n=args.runs)
        reference_frames[t.id] = res.frames
        v = order_variation(res.frames)
        if v.any:
            reference_findings.append(
                {
                    "id": t.id,
                    "row_order_varies": v.row_order,
                    "column_order_varies": v.column_order,
                    "content_varies": v.content,
                    "column_orders_seen": v.distinct_column_orders,
                    "check": t.check.to_dict(),
                }
            )
        print(f"  ref {t.id:<20} {'VARIES' if v.any else 'stable'}", flush=True)

    # --- 2. arm generations touching unordered ops ----------------------------
    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    generation_findings = []
    audited = []
    for rec in scores["tasks"]:
        code = rec.get("extracted_code") or ""
        task = by_id.get(rec["id"])
        if task is None or not TOUCHES_UNORDERED_OP.search(code):
            continue
        audited.append(rec["id"])
        res = run_repeated(task.setup_code, code, n=args.runs)
        statuses = []
        for got, expected in zip(res.frames, reference_frames[task.id]):
            ok = got is not None and expected is not None and compare_frames(got, expected, task.check).equal
            statuses.append("pass" if ok else "fail")
        n_pass = statuses.count("pass")
        v = order_variation(res.frames)
        flaky = 0 < n_pass < len(statuses)
        print(f"  gen {rec['id']:<20} pass {n_pass}/{len(statuses)}{'  FLAKY' if flaky else ''}", flush=True)
        if flaky or v.any:
            generation_findings.append(
                {
                    "id": rec["id"],
                    "stored_status": rec["status"],
                    "passes": n_pass,
                    "runs": len(statuses),
                    "flaky": flaky,
                    "row_order_varies": v.row_order,
                    "column_order_varies": v.column_order,
                    "content_varies": v.content,
                    "column_orders_seen": v.distinct_column_orders,
                    "check": task.check.to_dict(),
                }
            )

    report = {
        "runs": args.runs,
        "tasks_file": args.tasks,
        "scores_file": args.scores,
        "reference_varies": reference_findings,
        "generations_audited": audited,
        "generations_varying": generation_findings,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nreference solutions with varying output: {len(reference_findings)}/{len(tasks)}")
    for f in reference_findings:
        print(f"  {f['id']}: row={f['row_order_varies']} col={f['column_order_varies']} content={f['content_varies']}")
    flaky = [f for f in generation_findings if f["flaky"]]
    print(f"generations audited: {len(audited)}; with varying output: {len(generation_findings)}; flaky pass/fail: {len(flaky)}")
    for f in generation_findings:
        print(
            f"  {f['id']}: pass {f['passes']}/{f['runs']} (stored {f['stored_status']}) "
            f"row={f['row_order_varies']} col={f['column_order_varies']} content={f['content_varies']}"
        )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
