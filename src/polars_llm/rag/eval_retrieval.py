"""Retrieval evaluation without a GPU: API recall@k over the benchmark.

For each task, the Polars API names its ``reference_solution`` uses are
extracted **dynamically**. The reference runs once (in-process; these are our
own, trusted reference solutions) under a profile hook, and every call made
*directly from the solution's code* into a documented API object is recorded
under that object's reference name (``Expr.str.to_uppercase``,
``DataFrame.group_by``, ``GroupBy.agg``, ``col``, ...). Calls Polars makes
internally (e.g. ``DataFrame.with_columns`` -> ``LazyFrame.with_columns``) are
not counted, and decorator wrappers are seen through. Dunder methods
(operators) are ignored.

``API recall@k`` for a task = fraction of those names that appear in the
``api_names`` metadata of the chunks retrieved for its instruction. Reported as
the macro mean over tasks, overall and per category, both over all names and
over the *non-core* names (excluding ubiquitous ones like ``col`` /
``Expr.alias`` / ``DataFrame.select`` that every task uses).

It also sweeps k and the context budget, and reports the longest full prompt
(chat template applied, thinking disabled) so ``max_seq_length`` can be
checked.

CLI::

    python -m polars_llm.rag.eval_retrieval      # -> results/rag/retrieval_eval.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
from pathlib import Path

from ..bench.schema import Task, load_tasks
from .chunk import DEFAULT_OUT as DEFAULT_CHUNKS
from .chunk import Chunk, read_chunks, resolve_object
from .index import DEFAULT_INDEX_DIR
from .retrieve import DEFAULT_BUDGET_TOKENS, DEFAULT_K, Retrieved, format_context, pack_within_budget

DEFAULT_OUT = Path("results/rag/retrieval_eval.json")
SWEEP_K = (1, 3, 5, 8, 10)
SWEEP_BUDGET = (1000, 2000, 3000, 4000)

# Names (almost) every task uses; recall over the rest is the more telling number.
CORE_NAMES = frozenset(
    {
        "col",
        "lit",
        "Expr.alias",
        "DataFrame.select",
        "DataFrame.with_columns",
        "DataFrame.filter",
        "LazyFrame.select",
        "LazyFrame.with_columns",
        "LazyFrame.filter",
        "LazyFrame.collect",
    }
)


# ---------------------------------------------------------------------------
# API names used by a solution
# ---------------------------------------------------------------------------


def _code_targets(raw) -> list:
    if isinstance(raw, property):
        raw = raw.fget
    if inspect.isclass(raw):
        init = raw.__dict__.get("__init__")
        # Unwrap: a shared decorator's wrapper code must never map to one API.
        return [inspect.unwrap(init)] if inspect.isfunction(init) else []
    if inspect.isfunction(raw) or inspect.ismethod(raw):
        return [inspect.unwrap(raw)]
    if callable(raw):  # callable instance, e.g. `pl.col`
        call = getattr(type(raw), "__call__", None)
        return [inspect.unwrap(call)] if inspect.isfunction(call) else []
    return []


def build_code_map(chunks: list[Chunk]) -> dict:
    """code object -> set of documented API names implemented by it."""

    code_map: dict = {}
    seen: set[str] = set()
    for c in chunks:
        if c.kind != "api":
            continue
        name = c.api_names[0]
        if name in seen:
            continue
        seen.add(name)
        module = c.extra.get("module", "polars")
        if c.extra.get("directive") == "automodule":
            import importlib

            mod_name, member = name.split(".", 1)
            raw = getattr(importlib.import_module(f"polars.{mod_name}"), member, None)
        elif module == "polars.functions.col" or name == "col":
            import polars as pl

            raw = pl.col
        else:
            raw = resolve_object(name, module)
        for target in _code_targets(raw):
            code = getattr(target, "__code__", None)
            if code is not None:
                code_map.setdefault(code, set()).add(name)
    return code_map


def solution_api_names(setup_code: str, solution_code: str, code_map: dict) -> list[str]:
    """Documented API names called directly from ``solution_code``'s frames."""

    import polars as pl

    found: set[str] = set()

    def names_for(frame) -> set[str]:
        names = code_map.get(frame.f_code)
        if names:
            return names
        # A decorator wrapper: find the function it wraps in its closure.
        for var in frame.f_code.co_freevars:
            fn = frame.f_locals.get(var)
            if inspect.isfunction(fn):
                code = getattr(inspect.unwrap(fn), "__code__", None)
                if code in code_map:
                    return code_map[code]
        return set()

    def profiler(frame, event, arg):
        # Only frames entered *directly* from the solution's own code count;
        # anything deeper is Polars calling itself.
        if event == "call" and frame.f_back is not None and frame.f_back.f_code.co_filename == "<solution>":
            found.update(names_for(frame))

    ns: dict = {"pl": pl, "__name__": "__retrieval_eval__"}
    exec(compile(setup_code, "<setup>", "exec"), ns)
    exec(compile(solution_code, "<solution>", "exec"), ns)
    prev = sys.getprofile()
    sys.setprofile(profiler)
    try:
        out = ns["solve"](**ns["inputs"])
    finally:
        sys.setprofile(prev)
    if isinstance(out, pl.LazyFrame):
        out.collect()
    return sorted(n for n in found if not n.split(".")[-1].startswith("__"))


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


def api_recall(needed: list[str], retrieved: list[Retrieved]) -> tuple[float | None, list[str], list[str]]:
    """(recall, found, missing); recall is None when nothing is needed."""

    have = {n for c in retrieved for n in c.api_names}
    found = sorted(n for n in needed if n in have)
    missing = sorted(n for n in needed if n not in have)
    recall = len(found) / len(needed) if needed else None
    return recall, found, missing


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def summarize(rows: list[dict], key: str) -> dict:
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r[key])
    return {
        "overall": _mean([r[key] for r in rows]),
        "n_tasks_scored": sum(1 for r in rows if r[key] is not None),
        "by_category": {cat: _mean(v) for cat, v in sorted(by_cat.items())},
    }


def evaluate(
    tasks: list[Task],
    candidates_by_task: dict[str, list[Retrieved]],
    needed_by_task: dict[str, list[str]],
    count,
    *,
    k: int = DEFAULT_K,
    budget: int = DEFAULT_BUDGET_TOKENS,
    details: bool = False,
) -> dict:
    rows = []
    for t in tasks:
        selected = pack_within_budget(candidates_by_task[t.id], count, k=k, budget_tokens=budget)
        needed = needed_by_task[t.id]
        non_core = [n for n in needed if n not in CORE_NAMES]
        r_all, found, missing = api_recall(needed, selected)
        r_nc, _, _ = api_recall(non_core, selected)
        row = {
            "id": t.id,
            "category": t.category,
            "recall": r_all,
            "recall_non_core": r_nc,
            "n_chunks": len(selected),
        }
        if details:
            row.update(
                {
                    "needed": needed,
                    "found": found,
                    "missing": missing,
                    "retrieved": [
                        {"id": c.id, "title": c.title, "section": c.section, "n_tokens": c.n_tokens}
                        for c in selected
                    ],
                    "context_tokens": count(format_context(selected)) if selected else 0,
                }
            )
        rows.append(row)
    return {
        "k": k,
        "budget_tokens": budget,
        "recall": summarize(rows, "recall"),
        "recall_non_core": summarize(rows, "recall_non_core"),
        "mean_chunks": round(statistics.fmean(r["n_chunks"] for r in rows), 2),
        "tasks": rows if details else None,
    }


def longest_prompt_tokens(tasks: list[Task], contexts: dict[str, str]) -> dict:
    """Longest full prompt (chat template, thinking off) with and without context."""

    from transformers import AutoTokenizer

    from ..prompts import build_messages
    from .chunk import TOKENIZER_ID, TOKENIZER_REVISION

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    tok = getattr(tok, "tokenizer", tok)

    def n(task: Task, context: str | None) -> int:
        text = tok.apply_chat_template(
            build_messages(task, context=context),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tok(text, add_special_tokens=False)["input_ids"])

    no_ctx = {t.id: n(t, None) for t in tasks}
    with_ctx = {t.id: n(t, contexts.get(t.id)) for t in tasks}
    worst = max(with_ctx, key=with_ctx.get)
    return {
        "max_prompt_tokens_no_context": max(no_ctx.values()),
        "max_prompt_tokens_with_context": with_ctx[worst],
        "longest_task": worst,
        "mean_prompt_tokens_with_context": round(statistics.fmean(with_ctx.values()), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="API recall@k of the retriever on the benchmark.")
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET_TOKENS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from .retrieve import Retriever

    tasks = load_tasks(args.tasks)
    chunks = read_chunks(Path(args.chunks))
    code_map = build_code_map(chunks)
    needed_by_task = {t.id: solution_api_names(t.setup_code, t.reference_solution, code_map) for t in tasks}

    retriever = Retriever(Path(args.index_dir), device=args.device)
    candidates_by_task = {t.id: retriever.candidates(t.instruction) for t in tasks}

    main_eval = evaluate(
        tasks, candidates_by_task, needed_by_task, retriever.count, k=args.k, budget=args.budget, details=True
    )
    sweep = []
    for k in SWEEP_K:
        for budget in SWEEP_BUDGET:
            e = evaluate(tasks, candidates_by_task, needed_by_task, retriever.count, k=k, budget=budget)
            sweep.append(
                {"k": k, "budget_tokens": budget, "recall": e["recall"]["overall"],
                 "recall_non_core": e["recall_non_core"]["overall"]}
            )

    contexts = {
        r["id"]: format_context(
            pack_within_budget(candidates_by_task[r["id"]], retriever.count, k=args.k, budget_tokens=args.budget)
        )
        for r in main_eval["tasks"]
    }
    prompt_lengths = longest_prompt_tokens(tasks, contexts)

    documented = {n for c in chunks for n in c.api_names if c.kind == "api"}
    all_needed = sorted({n for v in needed_by_task.values() for n in v})
    report = {
        "index_meta": retriever.meta,
        "n_tasks": len(tasks),
        "n_distinct_api_names_needed": len(all_needed),
        "needed_but_undocumented": sorted(set(all_needed) - documented),
        "default": main_eval,
        "sweep": sweep,
        "prompt_lengths": prompt_lengths,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    m = main_eval
    print(f"API recall@{args.k} (budget {args.budget}): all {m['recall']['overall']}  "
          f"non-core {m['recall_non_core']['overall']}  mean chunks {m['mean_chunks']}")
    for cat, v in m["recall"]["by_category"].items():
        print(f"  {cat:<14} all {v}  non-core {m['recall_non_core']['by_category'][cat]}")
    print("sweep (k, budget -> recall all / non-core):")
    for s in sweep:
        print(f"  k={s['k']:<2} budget={s['budget_tokens']:<5} {s['recall']} / {s['recall_non_core']}")
    print(f"prompt lengths: {prompt_lengths}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
