"""Compare the chunks an arm actually retrieved with the retrieval evaluation.

The retrieval evaluation (``results/rag/retrieval_eval.json``) is computed
locally on CPU, while a RAG arm's generation retrieves with an index built on
the Colab GPU. Embeddings differ in the last bits between devices, which can
reorder near-ties, so this reports how often the two agree.

Example::

    python scripts/compare_retrieval.py --arm B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polars_llm.io import read_jsonl  # noqa: E402


def compare(generations: Path, evaluation: Path) -> dict:
    gen = {rec["task_id"]: [r["id"] for r in rec.get("retrieved", [])] for rec in read_jsonl(generations)}
    ev = {t["id"]: [r["id"] for r in t["retrieved"]] for t in json.loads(evaluation.read_text(encoding="utf-8"))["default"]["tasks"]}
    common = sorted(set(gen) & set(ev))
    identical = [t for t in common if gen[t] == ev[t]]
    same_set = [t for t in common if set(gen[t]) == set(ev[t])]
    jaccard = [
        len(set(gen[t]) & set(ev[t])) / len(set(gen[t]) | set(ev[t])) if (gen[t] or ev[t]) else 1.0 for t in common
    ]
    return {
        "n_tasks": len(common),
        "identical_order": len(identical),
        "identical_set": len(same_set),
        "mean_jaccard": round(sum(jaccard) / len(jaccard), 4) if jaccard else None,
        "differing_tasks": [t for t in common if set(gen[t]) != set(ev[t])],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", default="B")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--eval", default="results/rag/retrieval_eval.json")
    args = parser.parse_args(argv)

    r = compare(Path(args.out_dir) / args.arm / "generations.jsonl", Path(args.eval))
    print(
        f"[{args.arm}] retrieved chunks vs local retrieval eval: identical order {r['identical_order']}/{r['n_tasks']}, "
        f"identical set {r['identical_set']}/{r['n_tasks']}, mean Jaccard {r['mean_jaccard']}"
    )
    if r["differing_tasks"]:
        print(f"  differing tasks: {', '.join(r['differing_tasks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
