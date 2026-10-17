"""Post-audit remedy of the round trip: drop the ``api_mismatch`` kept requests.

**Chosen after seeing the audit, not pre-registered** (PLAN.md, "Round-trip
audit and remedy"). The pre-registered audit of the round trip (20 kept, up
to 5 of them ``api_mismatch``) found 4/20 wrong kept requests, all
output-equivalent mismatches (the request describes the output, not the code),
which triggers the ">= 3/20 wrong" rule. The remedy drops every kept request
whose solver code uses none of the reference's non-core API names
(``roundtrip.api_mismatch``), the class all 4 came from or resemble.

``verified.jsonl`` is rebuilt from ``filtered.jsonl`` + the per-request
round-trip verdicts (kept and not ``api_mismatch``), so the step is idempotent
and its report always compares the round-trip kept set with the remedied one.
It also draws the second audit: 20 random requests of the new kept set
(seed 1, not stratified); rule: < 3/20 wrong to proceed to fine-tuning.

CLI::

    python -m polars_llm.datagen.remedy     # then: build_sft --stage split
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from ..io import read_jsonl, write_jsonl
from .roundtrip import DEFAULT_IN, DEFAULT_OUT, DEFAULT_RESULTS

DEFAULT_STATS = Path("results/datagen/remedy_stats.json")
N_AUDIT = 20
AUDIT_SEED = 1
AUDIT_MAX_WRONG = 2  # >= 3 of 20 wrong -> do not proceed to notebook 06


def composition(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "multi_step_share": round(sum(r["multi_step"] for r in rows) / max(len(rows), 1), 3),
        "by_category": dict(sorted(collections.Counter(r["category"] for r in rows).items())),
    }


def remedy(examples: list[dict], results: list[dict]) -> tuple[list[dict], dict]:
    """(remedied kept examples, report). ``examples``: the filtered requests;
    ``results``: one round-trip verdict per request (canaries ignored)."""

    verdict = {r["id"]: r for r in results}
    kept = [e for e in examples if verdict[e["id"]]["kept"]]
    dropped = [e for e in kept if verdict[e["id"]]["api_mismatch"]]
    out = [e for e in kept if not verdict[e["id"]]["api_mismatch"]]

    def show(e):
        v = verdict[e["id"]]
        return {"id": e["id"], "category": e["category"], "instruction": e["instruction"],
                "reference_solution": e["reference_solution"], "solver_code": v["solver_code"],
                "reason": v["reason"], "detail": v["detail"], "api_mismatch": v["api_mismatch"], "audit_label": None}

    report = {
        "rule": "drop every round-trip kept request with api_mismatch = True (chosen after the audit, "
                "not pre-registered)",
        "n_dropped": len(dropped),
        "dropped_by_category": dict(sorted(collections.Counter(e["category"] for e in dropped).items())),
        "before": composition(kept),
        "after": composition(out),
        "audit": {"seed": AUDIT_SEED, "stratified": False,
                  "rule": f"manual: label audit_kept; < {AUDIT_MAX_WRONG + 1}/{N_AUDIT} wrong to proceed to notebook 06"},
        "audit_kept": [show(e) for e in random.Random(AUDIT_SEED).sample(out, min(N_AUDIT, len(out)))],
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drop api_mismatch kept requests; draw the second audit.")
    parser.add_argument("--examples", default=str(DEFAULT_IN), help="filtered requests (round-trip input)")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="round-trip verdicts per request")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    args = parser.parse_args(argv)

    out, report = remedy(read_jsonl(args.examples), read_jsonl(args.results))
    write_jsonl(Path(args.out), out, sort_keys=True)
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"dropped {report['n_dropped']} api_mismatch requests {report['dropped_by_category']}; "
          f"kept {report['before']['n']} -> {report['after']['n']}")
    print(f"Wrote {args.out}, {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
