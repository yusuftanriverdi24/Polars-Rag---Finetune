"""Round-trip verification of training requests (data quality, training data v1).

The quality filters only check a request's *form* (method names, code,
length, output columns). A request can pass them and still describe the
code wrongly ("every value" for an ``any_horizontal``) or ambiguously. This
step keeps a request only if a strong solver, given the request, can
reproduce the reference output:

* **Solver:** ``SOLVER_MODEL`` (Qwen3.5-9B, the instruction writer), **not**
  the 4B student: filtering on what the student already solves would keep
  only what it already knows. Limitation: writer and solver are the same
  model, so a misreading they share can pass.
* **Input:** the request + ``setup_code`` + signature, in the same prompt
  template as the RAG arms, with the example's **source doc chunk(s) minus
  the example itself** as context (``oracle.py``: no Examples section, no
  code block the candidate came from; ``answer_in_context`` must be 0). This
  tests whether the request is faithful and unambiguous, not whether the
  solver knows the API, and the solver cannot copy the answer from the doc.
* **Decoding:** greedy, thinking off; adaptive batching (start from free GPU
  memory, halve on CUDA OOM) and checkpoint/resume on Google Drive, exactly
  as for the instruction writer (``instructions.generate_all``).
* **Check:** the solver's code runs in the sandbox; its output must equal the
  reference output under the **candidate's own check flags**, as in the
  benchmark (``ignore_row_order`` only where the reference itself has no
  defined row order; column order is checked; numeric widths ignored).

Pre-registered decision rules (PLAN.md), reported in ``roundtrip_stats.json``:

* ``canaries``: pinned spot-check requests (``data/sft/canaries.json``) with
  their expected verdict: the wrong/ambiguous ones must be rejected, the
  correct ones kept. They are solved alongside, but never enter the data.
* ``audit_kept`` / ``audit_rejected``: 20 random each (seed 0) for a manual
  audit; if >= 3 of the 20 kept requests are wrong, the filter is insufficient.
  The kept sample is stratified: up to 5 of the 20 are ``api_mismatch`` ones.
* ``api_mismatch`` (report only, not a filter): kept requests whose solver
  code uses none of the reference's non-core API names, i.e. possible
  output-equivalent mismatches (the request is satisfied on this input but
  does not describe the code), which an output comparison cannot catch.
* ``decision_rules``: reshape retention < half the overall retention ->
  inspect; fewer than ~1,000 kept with most rejections ``error:*`` -> one retry
  with the error message before moving to data v2.

Outputs: ``data/sft/verified.jsonl`` (kept examples),
``results/datagen/roundtrip_results.jsonl`` (one row per request: verdict,
reason, solver code) and ``results/datagen/roundtrip_stats.json``.

CLI::

    python -m polars_llm.datagen.roundtrip --checkpoint /drive/roundtrip_checkpoint.jsonl
    python -m polars_llm.datagen.roundtrip --model mock     # CPU smoke test
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from ..bench.compare import compare_frames
from ..bench.executor import compute_expected, run_solution
from ..bench.extract import extract_code
from ..bench.schema import Task
from ..prompts.template import build_messages, prompt_fingerprint
from .build_sft import make_example
from .instructions import (
    ALLOC_CONF,
    DEFAULT_CANDIDATES,
    GB_PER_SEQUENCE,
    WRITER_MODEL,
    WRITER_REVISION,
    generate_all,
    gpu_info,
    initial_batch_size,
    load_checkpoint,
)
from .leakage import non_core
from .oracle import answer_in_context

SOLVER_MODEL = WRITER_MODEL
SOLVER_REVISION = WRITER_REVISION
MAX_NEW_TOKENS = 768  # as for the benchmark arms
MAX_SEQ_LENGTH = 8192  # <= 4,000 context tokens + request/setup + 768 new
DEFAULT_IN = Path("data/sft/filtered.jsonl")
DEFAULT_OUT = Path("data/sft/verified.jsonl")
DEFAULT_CANARIES = Path("data/sft/canaries.json")
DEFAULT_CHECKPOINT = Path("data/sft/roundtrip_checkpoint.jsonl")
DEFAULT_RESULTS = Path("results/datagen/roundtrip_results.jsonl")
DEFAULT_STATS = Path("results/datagen/roundtrip_stats.json")
N_AUDIT = 20
N_AUDIT_API_MISMATCH = 5  # of the 20 kept, up to 5 drawn from the api_mismatch ones
AUDIT_MAX_WRONG_KEPT = 2  # >= 3 of 20 wrong kept requests -> the filter is insufficient
RETRY_IF_KEPT_BELOW = 1000
SEED = 0


def solver_messages(example: dict, doc_context: str) -> list[dict]:
    task = Task.from_dict(example)
    return build_messages(task, context=doc_context or None)


def roundtrip_settings(model: str, revision: str | None) -> dict:
    fp = prompt_fingerprint()
    return {"solver_model": model, "solver_revision": revision, "prompt_template_sha256": fp["template_sha256"],
            "max_new_tokens": MAX_NEW_TOKENS}


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def _compare(got, want, check):
    """compare_frames, falling back to row order when ignoring it fails on unsortable dtypes."""

    try:
        return compare_frames(got, want, check)
    except Exception:  # noqa: BLE001 - lists/structs can't be sorted
        return compare_frames(got, want, replace(check, ignore_row_order=False))


def judge(example: dict, solver_output: str, expected, timeout_s: float = 30.0) -> dict:
    """Run the solver's code and compare with the reference output under the
    example's own check flags (as the benchmark does).

    Returns {"kept", "reason", "detail"}; reasons: ``match``, ``no_code``,
    ``error:<Type>``, ``timeout``, ``not_a_dataframe``, ``wrong_columns``
    (different set of names), ``wrong_column_order``, ``wrong_row_count``,
    ``wrong_row_order`` (right rows, order required but different),
    ``wrong_values``."""

    check = Task.from_dict(example).check
    code = extract_code(solver_output)
    if not code.strip() or "def solve" not in code:
        return {"kept": False, "reason": "no_code", "detail": ""}
    res = run_solution(example["setup_code"], code, timeout_s=timeout_s)
    if res.timed_out:
        return {"kept": False, "reason": "timeout", "detail": ""}
    if not res.ok:
        return {"kept": False, "reason": f"error:{res.error.type if res.error else '?'}",
                "detail": (res.error.message if res.error else "")[:300]}
    got, want = res.frame, expected.frame
    if got is None:
        return {"kept": False, "reason": "not_a_dataframe", "detail": ""}
    if set(got.columns) != set(want.columns):
        return {"kept": False, "reason": "wrong_columns", "detail": f"got {got.columns}, expected {want.columns}"[:300]}
    if got.columns != want.columns and not check.ignore_column_order:
        return {"kept": False, "reason": "wrong_column_order",
                "detail": f"got {got.columns}, expected {want.columns}"[:300]}
    if got.height != want.height:
        return {"kept": False, "reason": "wrong_row_count", "detail": f"got {got.height}, expected {want.height}"}
    cmp = _compare(got, want, check)
    if cmp.equal:
        return {"kept": True, "reason": "match", "detail": ""}
    if not check.ignore_row_order and _compare(got, want, replace(check, ignore_row_order=True)).equal:
        return {"kept": False, "reason": "wrong_row_order", "detail": ""}
    return {"kept": False, "reason": "wrong_values", "detail": (cmp.detail or "")[:300]}


def judge_all(examples: list[dict], outputs: dict[str, str], workers: int = 4) -> dict[str, dict]:
    """Verdict per example id; the reference output is computed once per candidate."""

    first = {}
    for e in examples:
        first.setdefault(e["candidate_id"], e)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        refs = list(pool.map(lambda e: compute_expected(e["setup_code"], e["reference_solution"]), first.values()))
    expected = dict(zip(first, refs))

    def one(e):
        ref = expected[e["candidate_id"]]
        if ref is None or not ref.ok or ref.frame is None:
            return e["id"], {"kept": False, "reason": "reference_failed", "detail": ""}
        return e["id"], judge(e, outputs[e["id"]], ref)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, examples))


def api_mismatch(example: dict, solver_code: str) -> bool | None:
    """Report-only: True when the solver's code uses **none** of the reference
    solution's non-core API names (None when the reference has none).

    Static approximation (the solver's code is not executed outside the
    sandbox): a reference name counts as used when its last component
    (``DataFrame.join_asof`` -> ``join_asof``, ``selectors.struct`` ->
    ``struct``) is a called or accessed name in the solver's code. Flags kept
    requests that may be output-equivalent mismatches (right output on this
    input, different operation), the class an output comparison can't catch."""

    names = non_core(example.get("api_names", []))
    if not names:
        return None
    used = _used_names(solver_code)
    return not any(n.rsplit(".", 1)[-1] in used for n in names)


def _used_names(code: str) -> set[str]:
    """Attribute and called names in ``code`` (``len``/``list`` included: Polars API names too)."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            out.add(node.func.id)
    return out


# ---------------------------------------------------------------------------
# canaries
# ---------------------------------------------------------------------------


def load_canaries(path: Path, candidates: dict[str, dict]) -> list[dict]:
    """Pinned spot-check requests as examples (id ``canary:...``) + their expectation."""

    if not path.exists():
        return []
    out = []
    for c in json.loads(path.read_text(encoding="utf-8")):
        ex = make_example(candidates[c["candidate_id"]], c["style"], c["instruction"])
        out.append({**ex, "id": c["id"], "canary": {k: c[k] for k in ("label", "expected", "note")}})
    return out


def canary_report(canaries: list[dict], verdicts: dict[str, dict]) -> dict:
    rows = []
    for c in canaries:
        v = verdicts[c["id"]]
        actual = "keep" if v["kept"] else "reject"
        rows.append({"id": c["id"], "label": c["canary"]["label"], "expected": c["canary"]["expected"],
                     "actual": actual, "reason": v["reason"], "passed": actual == c["canary"]["expected"]})
    return {"n": len(rows), "passed": sum(r["passed"] for r in rows),
            "all_passed": bool(rows) and all(r["passed"] for r in rows), "results": rows}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _rate(rows: list[dict], key) -> dict:
    groups: dict[str, list[bool]] = collections.defaultdict(list)
    for r in rows:
        groups[str(key(r))].append(r["kept"])
    return {k: {"n": len(v), "kept": sum(v), "retention": round(sum(v) / len(v), 3)} for k, v in sorted(groups.items())}


def report(examples: list[dict], verdicts: dict[str, dict], outputs: dict[str, str]) -> dict:
    rows = [{**e, **verdicts[e["id"]]} for e in examples]
    for r in rows:
        r["api_mismatch"] = api_mismatch(r, extract_code(outputs[r["id"]])) if r["kept"] else None
    kept = [r for r in rows if r["kept"]]
    rejected = [r for r in rows if not r["kept"]]
    rng = random.Random(SEED)

    def show(r):
        return {"id": r["id"], "category": r["category"], "instruction": r["instruction"],
                "reference_solution": r["reference_solution"], "solver_code": extract_code(outputs[r["id"]]),
                "reason": r["reason"], "detail": r["detail"], "api_mismatch": r["api_mismatch"],
                "audit_label": None}

    # Audit of kept requests, stratified: up to N_AUDIT_API_MISMATCH api_mismatch ones, the rest from the others.
    mismatched = [r for r in kept if r["api_mismatch"]]
    others = [r for r in kept if not r["api_mismatch"]]
    audit_mm = rng.sample(mismatched, min(N_AUDIT_API_MISMATCH, len(mismatched)))
    audit_kept = audit_mm + rng.sample(others, min(N_AUDIT - len(audit_mm), len(others)))
    applicable = [r for r in kept if r["api_mismatch"] is not None]
    mm_by_cat: dict[str, dict] = {}
    for r in applicable:
        d = mm_by_cat.setdefault(r["category"], {"n_kept": 0, "api_mismatch": 0})
        d["n_kept"] += 1
        d["api_mismatch"] += bool(r["api_mismatch"])

    retention = round(len(kept) / max(len(rows), 1), 3)
    by_category = _rate(rows, lambda r: r["category"])
    errors = sum(r["reason"].startswith("error:") for r in rejected)
    reshape = by_category.get("reshape", {}).get("retention")
    return {
        "n": len(rows),
        "kept": len(kept),
        "retention": retention,
        "reasons": dict(collections.Counter(r["reason"] for r in rows).most_common()),
        "by_source": _rate(rows, lambda r: r["source"]),
        "by_style": _rate(rows, lambda r: r["style"]),
        "by_category": by_category,
        "by_multi_step": _rate(rows, lambda r: r["multi_step"]),
        "by_check_flags": _rate(rows, lambda r: "ignore_row_order" if r["check"].get("ignore_row_order")
                                else "row_order_checked"),
        "both_styles_of_a_candidate": _candidate_agreement(rows),
        "decision_rules": {
            "reshape_retention": reshape,
            "reshape_below_half_overall": reshape is not None and reshape < retention / 2,
            "error_share_of_rejections": round(errors / max(len(rejected), 1), 3),
            "retry_with_error_message": len(kept) < RETRY_IF_KEPT_BELOW and errors > len(rejected) / 2,
            "audit": f"manual: label audit_kept; >= {AUDIT_MAX_WRONG_KEPT + 1}/{N_AUDIT} wrong -> filter insufficient",
        },
        "api_mismatch": {
            "report_only": "kept requests whose solver code uses none of the reference's non-core API names "
                           "(static name match); not a filter",
            "n_kept_with_non_core_names": len(applicable),
            "count": len(mismatched),
            "share": round(len(mismatched) / max(len(applicable), 1), 3),
            "by_category": dict(sorted(mm_by_cat.items())),
        },
        "audit_kept": [show(r) for r in audit_kept],
        "audit_kept_strata": {"api_mismatch": len(audit_mm), "other": len(audit_kept) - len(audit_mm)},
        "audit_rejected": [show(r) for r in rng.sample(rejected, min(N_AUDIT, len(rejected)))],
    }


def _candidate_agreement(rows: list[dict]) -> dict:
    by_cand: dict[str, list[bool]] = collections.defaultdict(list)
    for r in rows:
        by_cand[r["candidate_id"]].append(r["kept"])
    pairs = [v for v in by_cand.values() if len(v) == 2]
    return {"candidates_with_both": len(pairs), "both_kept": sum(all(v) for v in pairs),
            "one_kept": sum(any(v) and not all(v) for v in pairs), "none_kept": sum(not any(v) for v in pairs)}


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------


class MockSolver:
    """CPU smoke test: answers with the reference solution, except for every
    third example where it drops the last output column."""

    name, revision = "mock", None

    def solve_batch(self, items: list[dict]) -> list[tuple[str, float]]:
        out = []
        for it in items:
            code = it["reference_solution"]
            if int(it["id"].encode().hex(), 16) % 3 == 0:
                code = code.replace("def solve", "def _ref", 1) + (
                    "\n\ndef solve(*args, **kwargs):\n    out = _ref(*args, **kwargs)\n    return out.drop(out.columns[-1])"
                )
            out.append((f"```python\n{code}\n```", 0.0))
        return out


class ModelSolver:
    def __init__(self, model_id: str, revision: str | None, seed: int = 0):
        from ..models.hf import HFModel

        self.model = HFModel(model_id, max_seq_length=MAX_SEQ_LENGTH, seed=seed, revision=revision)
        self.name, self.revision = model_id, revision

    def solve_batch(self, items: list[dict]) -> list[tuple[str, float]]:
        gens = self.model.generate([it["messages"] for it in items], max_new_tokens=MAX_NEW_TOKENS,
                                   batch_size=len(items))
        return [(g.text, g.latency_s) for g in gens]

    def free_memory(self) -> None:
        import gc

        gc.collect()
        self.model._torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)
    parser = argparse.ArgumentParser(description="Round-trip verification of training requests.")
    parser.add_argument("--examples", default=str(DEFAULT_IN), help="filtered examples (build_sft --stage filter)")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES), help="for the oracle doc context")
    parser.add_argument("--canaries", default=str(DEFAULT_CANARIES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="append-only JSONL of solver outputs; put it on Google Drive")
    parser.add_argument("--model", default=SOLVER_MODEL, help="HF model id, or 'mock'")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="starting batch size (default: from free memory)")
    parser.add_argument("--gb-per-sequence", type=float, default=2 * GB_PER_SEQUENCE,
                        help="longer prompts than the writer (doc context)")
    parser.add_argument("--workers", type=int, default=4, help="sandbox processes for the check")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    examples = [json.loads(l) for l in Path(args.examples).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]
    candidates = {}
    for line in Path(args.candidates).read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            candidates[c["id"]] = c
    canaries = load_canaries(Path(args.canaries), candidates)

    def context(e):
        return candidates.get(e["candidate_id"], {}).get("doc_context", "")

    # The guarantee the oracle context must give: the answer is not in it.
    leaks = sorted({e["candidate_id"] for e in examples + canaries
                    if answer_in_context(e["reference_solution"], context(e))})
    answer_stats = {"n_candidates": len({e["candidate_id"] for e in examples + canaries}),
                    "with_answer_in_context": len(leaks), "ids": leaks[:50]}
    if leaks:
        raise SystemExit(f"the reference answer is in the oracle context of {len(leaks)} candidates, e.g. {leaks[:3]}; "
                         "run `python -m polars_llm.datagen.candidates --refresh-context`")

    revision = args.revision or (SOLVER_REVISION if args.model == SOLVER_MODEL else None)
    settings = roundtrip_settings(args.model, revision)
    checkpoint = Path(args.checkpoint)
    everything = examples + canaries
    items = [{**e, "messages": solver_messages(e, context(e))} for e in everything]

    def make_row(item, text):
        return {"raw": text}

    done = load_checkpoint(checkpoint, settings)
    run = {"n_resumed": len(done), "n_written": 0, "oom_events": []}
    gpu, source, wall = None, "none (checkpoint complete)", 0.0
    if any(e["id"] not in done for e in everything):
        solver = MockSolver() if args.model == "mock" else ModelSolver(args.model, revision)
        gpu = None if args.model == "mock" else gpu_info()
        if args.batch_size:
            batch_size, source = args.batch_size, "--batch-size"
        elif gpu:
            batch_size = initial_batch_size(gpu["free_gb"], gb_per_sequence=args.gb_per_sequence)
            source = "free GPU memory"
        else:
            batch_size, source = 16, "default (no GPU)"
        print(f"GPU: {gpu['name']}, {gpu['free_gb']} of {gpu['total_gb']} GB free after loading the solver" if gpu
              else "GPU: none")
        print(f"solver {solver.name}; starting batch size {batch_size} ({source})")
        t0 = time.perf_counter()
        done, run = generate_all(items, solver.solve_batch, checkpoint=checkpoint, settings=settings,
                                 batch_size=batch_size, free_memory=getattr(solver, "free_memory", lambda: None),
                                 make_row=make_row)
        wall = time.perf_counter() - t0
    else:
        print(f"checkpoint {checkpoint} already has all {len(everything)} solver outputs: model not loaded")

    outputs = {e["id"]: done[e["id"]]["raw"] for e in everything}
    print(f"checking {len(everything)} solver outputs in the sandbox ({args.workers} workers)...")
    t1 = time.perf_counter()
    verdicts = judge_all(everything, outputs, workers=args.workers)
    check_wall = time.perf_counter() - t1

    kept = [e for e in examples if verdicts[e["id"]]["kept"]]
    for path, rows in ((Path(args.out), kept),
                       (Path(args.results), [{"id": e["id"], "candidate_id": e["candidate_id"], **verdicts[e["id"]],
                                              "solver_code": extract_code(outputs[e["id"]]),
                                              "api_mismatch": api_mismatch(e, extract_code(outputs[e["id"]]))
                                              if verdicts[e["id"]]["kept"] else None} for e in everything])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)

    stats = {
        **settings,
        "decoding": "greedy",
        "context": "oracle: the example's source doc chunk(s) without the example (no Examples section / own code "
                   "block), RAG prompt template",
        "check": "sandbox output == reference output under the candidate's own check flags (as the benchmark)",
        "answer_in_context": answer_stats,
        "n_without_doc_context": sum(not context(e) for e in examples),
        "gpu": gpu,
        "batch_size_source": source,
        "batch_sizes_used": sorted({done[e["id"]].get("batch_size", 0) for e in everything}),
        "this_session": {**run, "generation_wall_time_s": round(wall, 1), "check_wall_time_s": round(check_wall, 1)},
        "generation_time_s_all_sessions": round(sum(done[e["id"]].get("latency_s", 0.0) for e in everything), 1),
        "canaries": canary_report(canaries, verdicts),
        **report(examples, verdicts, outputs),
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"round trip: kept {stats['kept']} / {stats['n']} ({stats['retention']}); reasons {stats['reasons']}")
    print(f"canaries: {stats['canaries']['passed']}/{stats['canaries']['n']} as expected; "
          f"decision rules: {stats['decision_rules']}")
    print(f"Wrote {args.out}, {args.results}, {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
