"""Aggregate metrics over per-task results (PLAN.md Section 4.3).

Computes pass@1 overall and broken down by difficulty and category, a histogram
of statuses, and the ``deprecated_api`` count (tasks that raised a deprecation
warning, recorded even when the result was correct).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .classify import PASS


def _rate(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


def _breakdown(records: list[dict], key: str) -> dict:
    totals: Counter = Counter()
    passes: Counter = Counter()
    for r in records:
        bucket = r.get(key, "unknown")
        totals[bucket] += 1
        if r["status"] == PASS:
            passes[bucket] += 1
    return {
        bucket: {
            "total": totals[bucket],
            "pass": passes[bucket],
            "pass_at_1": _rate(passes[bucket], totals[bucket]),
        }
        for bucket in sorted(totals)
    }


def aggregate(records: list[dict]) -> dict:
    """Build the aggregate metrics block from per-task records."""

    total = len(records)
    passed = sum(1 for r in records if r["status"] == PASS)
    status_counts = Counter(r["status"] for r in records)
    deprecated = sum(1 for r in records if r.get("deprecated_api"))
    pandas_leaks = sum(1 for r in records if r.get("pandas_leak_detected"))

    # Per-category error histograms are handy for the error analysis in Phase 5.
    errors_by_category: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if r["status"] != PASS:
            errors_by_category[r.get("category", "unknown")][r["status"]] += 1

    return {
        "total": total,
        "pass": passed,
        "pass_at_1": _rate(passed, total),
        "pass_at_1_by_difficulty": _breakdown(records, "difficulty"),
        "pass_at_1_by_category": _breakdown(records, "category"),
        "status_counts": dict(sorted(status_counts.items())),
        "deprecated_api_count": deprecated,
        "pandas_leak_count": pandas_leaks,
        "errors_by_category": {k: dict(sorted(v.items())) for k, v in sorted(errors_by_category.items())},
    }
