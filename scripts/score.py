"""Score an arm's generations against the benchmark.

Reads ``results/<arm>/generations.jsonl``, runs the benchmark runner (sandbox
execution + comparison + error classification) on CPU, and writes
``results/<arm>/scores.json``.

Example::

    python scripts/score.py --arm A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polars_llm.bench.runner import score  # noqa: E402
from polars_llm.bench.schema import load_tasks  # noqa: E402


def _load_generations(path: Path) -> list[tuple[str, str]]:
    outputs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "task_id" not in obj or "output" not in obj:
                raise ValueError(f"{path}:{lineno}: each line needs 'task_id' and 'output'")
            outputs.append((str(obj["task_id"]), str(obj["output"])))
    return outputs


def run_scoring(
    arm: str,
    out_dir: Path,
    tasks_path: str,
    *,
    timeout_s: float = 15.0,
    seed: int = 0,
) -> dict:
    """Score ``results/<arm>/generations.jsonl`` and write ``scores.json``."""

    arm_dir = out_dir / arm
    gen_path = arm_dir / "generations.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(f"no generations found at {gen_path}; run generate.py first")

    tasks = load_tasks(tasks_path)
    outputs = _load_generations(gen_path)

    run_meta = {"arm": arm, "generations_file": str(gen_path)}
    meta_path = arm_dir / "generation_meta.json"
    if meta_path.exists():
        run_meta["generation_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))

    results = score(tasks, outputs, timeout_s=timeout_s, seed=seed, run_meta=run_meta)

    out_path = arm_dir / "scores.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    m = results["metrics"]
    print(f"Wrote {out_path}")
    print(f"[{arm}] pass@1: {m['pass']}/{m['total']} = {m['pass_at_1']:.1%}")
    print(f"[{arm}] status counts: {m['status_counts']}")
    print(f"[{arm}] deprecated_api flagged: {m['deprecated_api_count']}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an arm's generations.")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    run_scoring(
        args.arm,
        Path(args.out_dir),
        args.tasks,
        timeout_s=args.timeout,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
