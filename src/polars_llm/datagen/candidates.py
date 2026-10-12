"""PART 1 of training data v1 (local, CPU): task candidates from the docs.

Sources (the same knowledge RAG uses):

- ``docstring``: doctest examples of the installed ``polars==2.0.0`` API, for
  the API pages kept in RAG corpus v2 (``data/rag/chunks_v2.jsonl``);
- ``user_guide``: the user-guide Python snippet files in ``data/raw``.

Pipeline: programs -> ``convert.program_candidates`` -> exact de-duplication
-> ``validate.validate_all`` (sandbox: clean run, no warning, DataFrame result,
same result twice) -> API names (``eval_retrieval.solution_api_names``),
benchmark-style category and difficulty -> leakage vs the frozen benchmark
(``leakage.solution_leak``) -> downsampling of trivial one-liners.

Multi-step = 2+ distinct *non-core* API names (core = ``col``, ``select``,
``filter``, ``with_columns``, ... as in the retrieval eval), since nearly
every example also calls ``select`` + ``col``. Downsampling (fixed seed): a
single-step example whose API is a trivial one (head/tail/slice/describe/...)
is kept at most ``TRIVIAL_CAP`` times per API, any other single-step API at
most ``SINGLE_STEP_CAP`` times; multi-step examples are all kept.

CLI::

    python -m polars_llm.datagen.candidates     # -> data/sft/candidates.jsonl + results/datagen/candidates_stats.json
"""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import random
import statistics
import time
from pathlib import Path

from ..bench.schema import load_tasks
from ..rag.chunk import read_chunks, resolve_object
from ..rag.eval_retrieval import CORE_NAMES, build_code_map, solution_api_names
from .convert import Candidate, docstring_program, program_candidates, user_guide_program
from .leakage import bench_refs, solution_leak
from .validate import validate_all

USER_GUIDE_DIR = Path("data/raw/polars-2.0.0/docs/source/src/python/user-guide")
DEFAULT_V1_CHUNKS = Path("data/rag/chunks.jsonl")
DEFAULT_V2_CHUNKS = Path("data/rag/chunks_v2.jsonl")
DEFAULT_OUT = Path("data/sft/candidates.jsonl")
DEFAULT_STATS = Path("results/datagen/candidates_stats.json")
SEED = 0

TRIVIAL_METHODS = frozenset(
    {
        "head", "tail", "limit", "slice", "clone", "clear", "describe", "glimpse", "rechunk", "shrink_to_fit",
        "lazy", "collect", "to_frame", "reverse", "gather_every", "first", "last", "null_count", "n_unique",
        "estimated_size", "shape", "columns", "schema", "height", "width", "dtypes", "is_empty", "bottom_k",
        "top_k", "max", "min", "sum", "mean", "median", "std", "var", "count", "len", "alias",
    }
)  # fmt: skip
TRIVIAL_CAP = 2
SINGLE_STEP_CAP = 6


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def docstring_programs(chunks_path: Path = DEFAULT_V2_CHUNKS):
    seen: dict[str, str] = {}
    for c in read_chunks(chunks_path):
        if c.kind == "api" and c.api_names and c.api_names[0] not in seen:
            seen[c.api_names[0]] = c.extra.get("module", "polars")
    for name, module in seen.items():
        try:
            obj = resolve_object(name, module)
        except Exception:  # noqa: BLE001 - unresolvable page: no examples
            continue
        prog = docstring_program(name, inspect.getdoc(obj) or "") if obj is not None else None
        if prog is not None:
            yield prog


def user_guide_programs(root: Path = USER_GUIDE_DIR):
    for f in sorted(root.rglob("*.py")):
        prog = user_guide_program("user-guide/" + f.relative_to(root).as_posix(), f.read_text(encoding="utf-8"))
        if prog is not None:
            yield prog


# ---------------------------------------------------------------------------
# tagging
# ---------------------------------------------------------------------------


def namespace(api_name: str) -> str:
    return api_name.rsplit(".", 1)[0] if "." in api_name else "functions"


def assign_category(api_names: list[str]) -> str:
    """Map an example to the benchmark category it most resembles (first match)."""

    names = set(api_names)
    last = {n.rsplit(".", 1)[-1] for n in names}
    ns = {namespace(n) for n in names}
    if last & {"pivot", "unpivot", "transpose", "explode", "unnest", "implode"} and not ns & {"Expr.list", "Expr.struct"}:
        return "reshape"
    if last & {"join", "join_asof", "join_where", "concat", "merge_sorted", "update"}:
        return "join"
    if last & {"over", "rolling", "rolling_by", "group_by_dynamic", "shift", "diff"} or any(
        n.split(".")[-1].startswith(("cum_", "rolling_")) for n in names
    ):
        return "window"
    if "Expr.str" in ns or "Series.str" in ns:
        return "string"
    if ns & {"Expr.dt", "Series.dt"} or last & {"date_range", "datetime_range", "date", "datetime", "duration"}:
        return "datetime"
    if ns & {"Expr.list", "Expr.struct", "Expr.arr", "Series.list", "Series.struct", "Series.arr"}:
        return "list_struct"
    if last & {"group_by", "agg"} or any(namespace(n) in {"GroupBy", "LazyGroupBy"} for n in names):
        return "group_by"
    if "LazyFrame" in ns:
        return "lazy"
    return "select_filter"


def difficulty(n_non_core: int) -> str:
    return "easy" if n_non_core <= 1 else "medium" if n_non_core <= 3 else "hard"


def downsample(records: list[dict], seed: int = SEED) -> tuple[list[dict], dict[str, int]]:
    """Cap single-step examples per API (trivial ones harder); keep all multi-step."""

    rng = random.Random(seed)
    order = sorted(records, key=lambda r: r["id"])
    rng.shuffle(order)
    kept_ids: set[str] = set()
    per_api: dict[str, int] = collections.Counter()
    dropped: dict[str, int] = collections.Counter()
    for r in order:
        if r["multi_step"]:
            kept_ids.add(r["id"])
            continue
        key = (r["non_core_api_names"] or r["api_names"] or ["<none>"])[0]
        cap = TRIVIAL_CAP if key.rsplit(".", 1)[-1] in TRIVIAL_METHODS else SINGLE_STEP_CAP
        if per_api[key] < cap:
            per_api[key] += 1
            kept_ids.add(r["id"])
        else:
            dropped["trivial" if cap == TRIVIAL_CAP else "single_step_cap"] += 1
    return [r for r in records if r["id"] in kept_ids], dict(dropped)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def collect_candidates(programs) -> tuple[list[Candidate], dict]:
    cands: list[Candidate] = []
    rejects: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_programs: collections.Counter = collections.Counter()
    for prog in programs:
        n_programs[prog.source] += 1
        cs, r = program_candidates(prog)
        cands.extend(cs)
        rejects[prog.source].update(r)
    # Exact duplicates (same setup + solution) are kept once.
    unique: dict[tuple[str, str], Candidate] = {}
    dupes = 0
    for c in cands:
        key = (c.setup_code, c.solution)
        if key in unique:
            dupes += 1
        else:
            unique[key] = c
    return list(unique.values()), {
        "programs": dict(n_programs),
        "static_rejects": {k: dict(v) for k, v in rejects.items()},
        "exact_duplicates": dupes,
    }


def build_records(validated, code_map, refs) -> tuple[list[dict], dict]:
    records, leaks, no_api = [], collections.Counter(), 0
    leak_examples: list[dict] = []
    for v in validated:
        c = v.candidate
        try:
            names = solution_api_names(c.setup_code, v.solution, code_map)
        except Exception:  # noqa: BLE001 - validated in the sandbox; be safe
            names = []
        non_core = [n for n in names if n not in CORE_NAMES]
        if not names:  # e.g. `return df`: nothing to learn
            no_api += 1
            continue
        leak = solution_leak(v.solution, set(names), refs)
        if leak:
            leaks[leak[0]] += 1
            leak_examples.append({"id": c.id, "reason": leak[0], "benchmark_task": leak[1]})
            continue
        records.append(
            {
                "id": c.id,
                "source": c.source,
                "page": c.page,
                "label": c.label,
                "category": assign_category(names),
                "difficulty": difficulty(len(non_core)),
                "setup_code": c.setup_code,
                "signature": c.signature,
                "reference_solution": v.solution,
                "check": {"ignore_row_order": v.ignore_row_order, "ignore_column_order": False,
                          "ignore_numeric_width": True},
                "api_names": names,
                "non_core_api_names": non_core,
                "multi_step": len(non_core) >= 2,
                "output_columns": v.output_columns,
                "output_dtypes": v.output_dtypes,
                "n_rows": v.n_rows,
                "preview": v.preview,
            }
        )
    return records, {"dropped": dict(leaks), "examples": leak_examples, "no_api_call_dropped": no_api}


def stats(records: list[dict]) -> dict:
    by_source = collections.Counter(r["source"] for r in records)
    ns = collections.Counter(namespace(n) for r in records for n in set(r["api_names"]))
    return {
        "n": len(records),
        "by_source": dict(by_source),
        "by_category": dict(sorted(collections.Counter(r["category"] for r in records).items())),
        "by_difficulty": dict(sorted(collections.Counter(r["difficulty"] for r in records).items())),
        "multi_step_share": round(sum(r["multi_step"] for r in records) / max(len(records), 1), 3),
        "multi_step_share_by_source": {
            s: round(sum(r["multi_step"] for r in records if r["source"] == s) / n, 3) for s, n in by_source.items()
        },
        "ignore_row_order": sum(r["check"]["ignore_row_order"] for r in records),
        "api_namespaces": dict(ns.most_common()),
        "distinct_api_names": len({n for r in records for n in r["api_names"]}),
        "pages": len({r["page"] for r in records}),
        "mean_non_core_api_names": round(statistics.fmean(len(r["non_core_api_names"]) for r in records), 2)
        if records
        else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training data v1, part 1: task candidates from the docs.")
    parser.add_argument("--v2-chunks", default=str(DEFAULT_V2_CHUNKS), help="API pages (RAG corpus v2)")
    parser.add_argument("--v1-chunks", default=str(DEFAULT_V1_CHUNKS), help="for resolving API names")
    parser.add_argument("--user-guide", default=str(USER_GUIDE_DIR))
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="only the first N candidates (smoke test)")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    programs = list(docstring_programs(Path(args.v2_chunks))) + list(user_guide_programs(Path(args.user_guide)))
    cands, collect_report = collect_candidates(programs)
    if args.limit:
        cands = cands[: args.limit]
    print(f"{len(cands)} unique candidates from {collect_report['programs']}; validating in the sandbox...")

    validated = validate_all(cands, workers=args.workers)
    ok = [v for v in validated if v.ok]
    val_rejects: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for v in validated:
        if not v.ok:
            val_rejects[v.candidate.source][v.reason.split(":")[0] if v.reason.startswith("error") else v.reason] += 1
    print(f"valid: {len(ok)} / {len(validated)}")

    code_map = build_code_map(read_chunks(Path(args.v1_chunks)))
    tasks = load_tasks(args.tasks)
    needed = {t.id: solution_api_names(t.setup_code, t.reference_solution, code_map) for t in tasks}
    records, leak_report = build_records(ok, code_map, bench_refs(tasks, needed))
    before_downsample = stats(records)
    records, down_report = downsample(records)
    final = stats(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "polars_version": __import__("polars").__version__,
        "candidates_by_source": dict(collections.Counter(c.source for c in cands)),
        **collect_report,
        "sandbox_valid_by_source": dict(collections.Counter(v.candidate.source for v in ok)),
        "sandbox_rejects": {k: dict(v) for k, v in val_rejects.items()},
        "leakage": leak_report,
        "before_downsampling": before_downsample,
        "downsampling": {"trivial_cap": TRIVIAL_CAP, "single_step_cap": SINGLE_STEP_CAP, "dropped": down_report},
        "final": final,
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidates_by_source", "sandbox_valid_by_source", "sandbox_rejects")}))
    print(f"leakage dropped: {leak_report['dropped']}; downsampling dropped: {down_report}")
    print(f"final: {json.dumps({k: final[k] for k in ('n', 'by_source', 'multi_step_share', 'by_category')})}")
    print(f"Wrote {out} and {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
