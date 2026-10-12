"""PART 2 of training data v1: filter instructions, leakage check, split.

Inputs: ``data/sft/candidates.jsonl`` (PART 1) and
``data/sft/instructions_raw.jsonl`` (``instructions.py``). Each surviving
(candidate, style) pair becomes one training example in the benchmark task
format (so ``load_tasks`` and the whole benchmark tooling work on it).

1. ``instructions.instruction_problems`` drops a request that names a method
   the solution uses, contains code, is too short/long or misses an output
   column.
2. Leakage: requests whose embedding (``EMBEDDER``, the RAG model) has cosine
   >= ``leakage.INSTRUCTION_SIM_THRESHOLD`` to any benchmark instruction are
   dropped. (The solution / API-set checks already ran in PART 1.)
3. Split ~90/10 train/dev **grouped by source doc page**: whole pages go to
   one side (pages shuffled with a fixed seed, dev filled up to
   ``DEV_FRACTION`` of the examples), so near-identical examples from one page
   never straddle the split.

CLI::

    python -m polars_llm.datagen.build_sft --device cuda
    python -m polars_llm.datagen.build_sft --embedder fake    # CPU smoke test, no model
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from ..bench.schema import Task, load_tasks
from .instructions import DEFAULT_CANDIDATES, DEFAULT_OUT as DEFAULT_RAW, STYLES, instruction_problems
from .leakage import INSTRUCTION_SIM_THRESHOLD, instruction_leaks

SFT_DIR = Path("data/sft")
DEFAULT_STATS = Path("results/datagen/sft_stats.json")
DEV_FRACTION = 0.10
SEED = 0


def make_example(rec: dict, style: str, instruction: str) -> dict:
    ex = {
        "id": f"{rec['id']}__{style}",
        "difficulty": rec["difficulty"],
        "category": rec["category"],
        "instruction": instruction,
        "setup_code": rec["setup_code"],
        "signature": rec["signature"],
        "reference_solution": rec["reference_solution"],
        "check": rec["check"],
        # provenance (ignored by Task.from_dict)
        "candidate_id": rec["id"],
        "style": style,
        "source": rec["source"],
        "page": rec["page"],
        "api_names": rec["api_names"],
        "multi_step": rec["multi_step"],
    }
    Task.from_dict(ex)  # must be a valid benchmark-format task
    return ex


def grouped_split(examples: list[dict], dev_fraction: float = DEV_FRACTION, seed: int = SEED):
    """(train, dev): whole ``page`` groups on one side; dev ~ ``dev_fraction``."""

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for ex in examples:
        groups[ex["page"]].append(ex)
    pages = sorted(groups)
    random.Random(seed).shuffle(pages)
    target = dev_fraction * len(examples)
    dev_pages: set[str] = set()
    n_dev = 0
    for p in pages:
        if n_dev >= target:
            break
        if n_dev + len(groups[p]) > 1.5 * target and n_dev > 0:
            continue  # don't overshoot with one huge page
        dev_pages.add(p)
        n_dev += len(groups[p])
    train = [ex for ex in examples if ex["page"] not in dev_pages]
    dev = [ex for ex in examples if ex["page"] in dev_pages]
    return train, dev


class FakeEmbedder:
    """Hashing bag-of-words embedder for CPU smoke tests."""

    def encode(self, texts, **_):
        import numpy as np

        out = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def embed(model, texts: list[str]):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8", newline="\n")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "n_candidates": len({r["candidate_id"] for r in rows}),
        "n_pages": len({r["page"] for r in rows}),
        "by_style": dict(collections.Counter(r["style"] for r in rows)),
        "by_source": dict(collections.Counter(r["source"] for r in rows)),
        "by_category": dict(sorted(collections.Counter(r["category"] for r in rows).items())),
        "by_difficulty": dict(sorted(collections.Counter(r["difficulty"] for r in rows).items())),
        "multi_step_share": round(sum(r["multi_step"] for r in rows) / max(len(rows), 1), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter instructions, check leakage, split train/dev.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--instructions", default=str(DEFAULT_RAW))
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--out-dir", default=str(SFT_DIR))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--embedder", choices=["granite", "fake"], default="granite")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    recs = {json.loads(l)["id"]: json.loads(l) for l in Path(args.candidates).read_text(encoding="utf-8").splitlines() if l.strip()}
    raw = [json.loads(l) for l in Path(args.instructions).read_text(encoding="utf-8").splitlines() if l.strip()]

    problems: collections.Counter = collections.Counter()
    problems_by_style: dict[str, collections.Counter] = {s: collections.Counter() for s in STYLES}
    n_unparsed = 0
    examples: list[dict] = []
    for r in raw:
        rec = recs.get(r["id"])
        if rec is None:
            continue
        if not r.get("instructions"):
            n_unparsed += 1
            continue
        for style in STYLES:
            text = r["instructions"][style]
            probs = instruction_problems(text, style, rec)
            for p in probs:
                problems[p] += 1
                problems_by_style[style][p] += 1
            if not probs:
                examples.append(make_example(rec, style, text))
    n_after_filters = len(examples)

    # Instruction-embedding leakage vs the frozen benchmark.
    bench = load_tasks(args.tasks)
    if args.embedder == "fake":
        model = FakeEmbedder()
    else:
        from ..rag.index import load_embedder

        model = load_embedder(args.device, "granite")
    leaks = instruction_leaks(embed(model, [e["instruction"] for e in examples]),
                              embed(model, [t.instruction for t in bench]))
    leak_examples = [
        {"id": e["id"], "benchmark_task": bench[l[0]].id, "cosine": round(l[1], 4)}
        for e, l in zip(examples, leaks) if l is not None
    ]
    examples = [e for e, l in zip(examples, leaks) if l is None]

    train, dev = grouped_split(examples)
    out_dir = Path(args.out_dir)
    train_sha = _write_jsonl(out_dir / "train.jsonl", train)
    dev_sha = _write_jsonl(out_dir / "dev.jsonl", dev)
    assert not ({e["page"] for e in train} & {e["page"] for e in dev})

    report = {
        "n_candidates": len(recs),
        "n_instruction_pairs": len(raw),
        "n_unparsed_writer_outputs": n_unparsed,
        "filter_problems": dict(problems),
        "filter_problems_by_style": {s: dict(c) for s, c in problems_by_style.items()},
        "n_after_filters": n_after_filters,
        "instruction_leakage": {
            "embedder": args.embedder,
            "threshold": INSTRUCTION_SIM_THRESHOLD,
            "dropped": len(leak_examples),
            "examples": leak_examples,
        },
        "all": summarize(examples),
        "train": {**summarize(train), "sha256": train_sha},
        "dev": {**summarize(dev), "sha256": dev_sha},
        "split": {"grouped_by": "page", "dev_fraction_target": DEV_FRACTION, "seed": SEED},
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"filters dropped {dict(problems)} (unparsed {n_unparsed}); leakage dropped {len(leak_examples)}")
    print(f"train {len(train)} / dev {len(dev)} examples ({report['train']['n_pages']} / {report['dev']['n_pages']} pages)")
    print(f"Wrote {out_dir / 'train.jsonl'}, {out_dir / 'dev.jsonl'}, {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
