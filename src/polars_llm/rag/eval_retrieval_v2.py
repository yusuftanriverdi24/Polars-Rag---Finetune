"""Arm B2 retrieval comparison: one round, scored on API recall only.

Variants (all hybrid dense + BM25 with RRF, as in Arm B):

``naive``  corpus v1 + granite, instruction query: the Arm B setup, shown for
           reference and never selected.
``a``      corpus v2 + granite, instruction query.
``b``      corpus v2 + Qwen3-Embedding-0.6B, instruction query.
``c``      corpus v2 + the better of a/b, *draft-code* query: instruction +
           the code of the model's own zero-shot answer (the Arm A generation
           for that task; see ``retrieve.draft_query``).

Each variant is evaluated on the grid k in ``GRID_K`` x context budget in
``GRID_BUDGET``. The needed API names come from the reference solutions,
resolved against the **v1** corpus, so every variant is scored against the
same name sets, even for pages corpus v2 dropped.

Selection rules (fixed before running):

- better of a/b = higher mean non-core recall over the 6 grid points (tie:
  granite, the cheaper model);
- chosen B2 config = highest overall non-core recall among a/b/c x grid; ties
  go to higher all-names recall, then the smaller budget, then the smaller k,
  then the instruction query (cheaper).

pass@1 is never looked at here.

CLI::

    python -m polars_llm.rag.eval_retrieval_v2 --device cuda   # -> results/rag/retrieval_eval_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from ..bench.schema import load_tasks
from .chunk import DEFAULT_OUT as DEFAULT_V1_CHUNKS
from .chunk import read_chunks
from .eval_retrieval import build_code_map, evaluate, longest_prompt_tokens, solution_api_names
from .retrieve import build_query, format_context, load_drafts, pack_within_budget

DEFAULT_OUT = Path("results/rag/retrieval_eval_v2.json")
DEFAULT_DRAFTS = Path("results/A/generations.jsonl")
INDEX_DIRS = {
    "naive": Path("data/rag/index"),
    "granite": Path("data/rag/index_v2_granite"),
    "qwen3": Path("data/rag/index_v2_qwen3"),
}
GRID_K = (5, 8)
GRID_BUDGET = (2000, 3000, 4000)
MAX_NEW_TOKENS = 768
MIN_MAX_SEQ_LENGTH = 4096


def grid_mean_non_core(grid: list[dict]) -> float:
    return round(statistics.fmean(g["recall_non_core"]["overall"] for g in grid), 4)


def better_embedder(grid_granite: list[dict], grid_qwen3: list[dict]) -> str:
    """'granite' or 'qwen3' by mean non-core recall over the grid (tie: granite)."""

    return "qwen3" if grid_mean_non_core(grid_qwen3) > grid_mean_non_core(grid_granite) else "granite"


_MODE_ORDER = {"instruction": 0, "draft": 1}


def choose_best(configs: list[dict]) -> dict:
    """configs: dicts with variant, query_mode, k, budget_tokens, recall, recall_non_core (overall floats)."""

    return min(
        configs,
        key=lambda c: (
            -c["recall_non_core"],
            -c["recall"],
            c["budget_tokens"],
            c["k"],
            _MODE_ORDER[c["query_mode"]],
            c["variant"],
        ),
    )


def required_max_seq_length(max_prompt_tokens: int, max_new_tokens: int = MAX_NEW_TOKENS) -> int:
    """Smallest multiple of 512 (at least 4096) that fits prompt + new tokens."""

    return max(MIN_MAX_SEQ_LENGTH, math.ceil((max_prompt_tokens + max_new_tokens) / 512) * 512)


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_variant(tasks, index_dir: Path, query_mode: str, drafts: dict, needed: dict, *, device=None) -> dict:
    from .retrieve import Retriever

    retriever = Retriever(index_dir, device=device)
    candidates, latencies = {}, []
    for t in tasks:
        query = build_query(query_mode, t.instruction, (drafts.get(t.id) or {}).get("output"))
        t0 = time.perf_counter()
        candidates[t.id] = retriever.candidates(query)
        latencies.append(time.perf_counter() - t0)
    grid = []
    for k in GRID_K:
        for budget in GRID_BUDGET:
            e = evaluate(tasks, candidates, needed, retriever.count, k=k, budget=budget)
            e.pop("tasks")
            grid.append(e)
    out = {
        "index_dir": str(index_dir),
        "index_meta": retriever.meta,
        "query_mode": query_mode,
        "mean_query_latency_s": round(statistics.fmean(latencies), 4),
        "grid_mean_non_core": None,
        "grid": grid,
        "_candidates": candidates,
        "_count": retriever.count,
    }
    out["grid_mean_non_core"] = grid_mean_non_core(grid)
    del retriever
    _free_gpu()
    return out


def _print_table(variants: dict, chosen: dict) -> None:
    print(f"{'variant':<8} {'embedder':<8} {'query':<11} {'k':>2} {'budget':>6}  {'all':>6}  {'non-core':>8}")
    for name, v in variants.items():
        for g in v["grid"]:
            mark = " <- chosen" if name == chosen["variant"] and (g["k"], g["budget_tokens"]) == (
                chosen["k"], chosen["budget_tokens"]) else ""
            print(f"{name:<8} {v['index_meta'].get('embedder', 'granite'):<8} {v['query_mode']:<11} "
                  f"{g['k']:>2} {g['budget_tokens']:>6}  {g['recall']['overall']:>6}  "
                  f"{g['recall_non_core']['overall']:>8}{mark}")
    cats = sorted(next(iter(variants.values()))["grid"][0]["recall_non_core"]["by_category"])
    print("\nnon-core recall per category at each variant's best grid point:")
    print(f"{'category':<14}" + "".join(f"{n:>9}" for n in variants))
    best = {n: max(v["grid"], key=lambda g: g["recall_non_core"]["overall"]) for n, v in variants.items()}
    for cat in cats:
        print(f"{cat:<14}" + "".join(f"{str(best[n]['recall_non_core']['by_category'][cat]):>9}" for n in variants))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Arm B2 retrieval variants by API recall.")
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--chunks", default=str(DEFAULT_V1_CHUNKS), help="v1 chunks (for the needed API names)")
    parser.add_argument("--drafts", default=str(DEFAULT_DRAFTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-naive", action="store_true", help="don't evaluate the v1 (Arm B) index")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    code_map = build_code_map(read_chunks(Path(args.chunks)))
    needed = {t.id: solution_api_names(t.setup_code, t.reference_solution, code_map) for t in tasks}
    drafts = load_drafts(Path(args.drafts))
    missing = [t.id for t in tasks if t.id not in drafts]
    if missing:
        raise ValueError(f"no draft for {len(missing)} tasks, e.g. {missing[:3]}")

    variants: dict[str, dict] = {}
    if not args.skip_naive:
        variants["naive"] = run_variant(tasks, INDEX_DIRS["naive"], "instruction", drafts, needed, device=args.device)
    variants["a"] = run_variant(tasks, INDEX_DIRS["granite"], "instruction", drafts, needed, device=args.device)
    variants["b"] = run_variant(tasks, INDEX_DIRS["qwen3"], "instruction", drafts, needed, device=args.device)
    winner = better_embedder(variants["a"]["grid"], variants["b"]["grid"])
    variants["c"] = run_variant(tasks, INDEX_DIRS[winner], "draft", drafts, needed, device=args.device)

    configs = [
        {
            "variant": name,
            "embedder": v["index_meta"]["embedder"],
            "index_dir": v["index_dir"],
            "query_mode": v["query_mode"],
            "k": g["k"],
            "budget_tokens": g["budget_tokens"],
            "recall": g["recall"]["overall"],
            "recall_non_core": g["recall_non_core"]["overall"],
        }
        for name, v in variants.items()
        if name != "naive"
        for g in v["grid"]
    ]
    chosen = choose_best(configs)
    cv = variants[chosen["variant"]]
    details = evaluate(tasks, cv["_candidates"], needed, cv["_count"], k=chosen["k"],
                       budget=chosen["budget_tokens"], details=True)
    contexts = {
        t.id: format_context(pack_within_budget(cv["_candidates"][t.id], cv["_count"], k=chosen["k"],
                                                budget_tokens=chosen["budget_tokens"]))
        for t in tasks
    }
    prompt_lengths = longest_prompt_tokens(tasks, contexts)
    chosen["max_seq_length"] = required_max_seq_length(prompt_lengths["max_prompt_tokens_with_context"])
    chosen["reason"] = (
        f"highest overall non-core API recall ({chosen['recall_non_core']}) of {len(configs)} configs "
        f"(variants a/b/c x k {list(GRID_K)} x budget {list(GRID_BUDGET)}); ties -> all-names recall, "
        "smaller budget, smaller k, instruction query"
    )

    _print_table(variants, chosen)
    print(f"\nbetter embedder (a vs b, mean non-core over grid): {winner} "
          f"(a {variants['a']['grid_mean_non_core']}, b {variants['b']['grid_mean_non_core']})")
    print(f"chosen: {json.dumps({k: v for k, v in chosen.items() if k != 'reason'})}")
    print(f"prompt lengths: {prompt_lengths}")
    print("mean query latency (s): " + ", ".join(f"{n} {v['mean_query_latency_s']}" for n, v in variants.items()))

    for v in variants.values():
        v.pop("_candidates")
        v.pop("_count")
    report = {
        "n_tasks": len(tasks),
        "needed_names_from": args.chunks,
        "drafts": args.drafts,
        "grid": {"k": list(GRID_K), "budget_tokens": list(GRID_BUDGET)},
        "better_embedder": winner,
        "variants": variants,
        "chosen": chosen,
        "chosen_details": details,
        "prompt_lengths": prompt_lengths,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
