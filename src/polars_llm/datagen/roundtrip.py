"""Round-trip verification of training requests (data quality, training data v1).

The quality filters only check a request's *form* (method names, code,
length, output columns). A request can pass them and still describe the
code wrongly ("every value" for an ``any_horizontal``) or ambiguously. This
step keeps a request only if a strong solver, given the request, can
reproduce the reference output:

* **Solver:** ``SOLVER_MODEL`` (Qwen3.5-9B, the instruction writer), **not**
  the 4B student: filtering on what the student already solves would keep
  only what it already knows.
* **Input:** the request + ``setup_code`` + signature, in the same prompt
  template as the RAG arms, with the **source doc chunk(s)** of the example
  as context (oracle context, ``candidates.attach_doc_context``). This tests
  whether the request is faithful and unambiguous, not whether the solver
  knows the API.
* **Decoding:** greedy, thinking off; adaptive batching (start from free GPU
  memory, halve on CUDA OOM) and checkpoint/resume on Google Drive, exactly
  as for the instruction writer (``instructions.write_with_checkpoint``).
* **Check:** the solver's code runs in the sandbox; its output must equal the
  reference output with the **row order and column order ignored** (column
  names and values must match; numeric widths ignored as in the benchmark).

Outputs: ``data/sft/verified.jsonl`` (kept examples),
``results/datagen/roundtrip_results.jsonl`` (one row per request: verdict,
reason, solver code) and ``results/datagen/roundtrip_stats.json`` (retention
overall and by source / style / category / multi-step, plus random kept and
rejected examples).

CLI::

    python -m polars_llm.datagen.roundtrip --checkpoint /drive/roundtrip_checkpoint.jsonl
    python -m polars_llm.datagen.roundtrip --model mock     # CPU smoke test
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..bench.compare import compare_frames
from ..bench.executor import compute_expected, run_solution
from ..bench.extract import extract_code
from ..bench.schema import CheckFlags, Task
from ..prompts.template import build_messages, prompt_fingerprint
from .instructions import (
    ALLOC_CONF,
    DEFAULT_CANDIDATES,
    GB_PER_SEQUENCE,
    WRITER_MODEL,
    WRITER_REVISION,
    gpu_info,
    initial_batch_size,
    load_checkpoint,
    write_with_checkpoint,
)

SOLVER_MODEL = WRITER_MODEL
SOLVER_REVISION = WRITER_REVISION
MAX_NEW_TOKENS = 768  # as for the benchmark arms
MAX_SEQ_LENGTH = 8192  # <= 4,000 context tokens + request/setup + 768 new
DEFAULT_IN = Path("data/sft/filtered.jsonl")
DEFAULT_OUT = Path("data/sft/verified.jsonl")
DEFAULT_CHECKPOINT = Path("data/sft/roundtrip_checkpoint.jsonl")
DEFAULT_RESULTS = Path("results/datagen/roundtrip_results.jsonl")
DEFAULT_STATS = Path("results/datagen/roundtrip_stats.json")
CHECK = CheckFlags(ignore_row_order=True, ignore_column_order=True, ignore_numeric_width=True)
N_SHOW = 6
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


def judge(example: dict, solver_output: str, expected, timeout_s: float = 30.0) -> dict:
    """Run the solver's code and compare with the reference output.

    Returns {"kept", "reason", "detail"}; reasons: ``match``, ``no_code``,
    ``error:<Type>``, ``timeout``, ``not_a_dataframe``, ``wrong_columns``,
    ``wrong_row_count``, ``wrong_values``."""

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
        return {"kept": False, "reason": "wrong_columns",
                "detail": f"got {got.columns}, expected {want.columns}"[:300]}
    if got.height != want.height:
        return {"kept": False, "reason": "wrong_row_count", "detail": f"got {got.height}, expected {want.height}"}
    want = want.select(got.columns)
    try:
        cmp = compare_frames(got, want, CHECK)
    except Exception:  # noqa: BLE001 - unsortable dtypes (lists, structs): compare in row order instead
        cmp = compare_frames(got, want, CheckFlags(ignore_column_order=True))
    if not cmp.equal:
        return {"kept": False, "reason": "wrong_values", "detail": (cmp.detail or "")[:300]}
    return {"kept": True, "reason": "match", "detail": ""}


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
    kept = [r for r in rows if r["kept"]]
    rejected = [r for r in rows if not r["kept"]]
    rng = random.Random(SEED)

    def show(r):
        return {"id": r["id"], "instruction": r["instruction"], "reference_solution": r["reference_solution"],
                "solver_code": extract_code(outputs[r["id"]]), "reason": r["reason"], "detail": r["detail"]}

    return {
        "n": len(rows),
        "kept": len(kept),
        "retention": round(len(kept) / max(len(rows), 1), 3),
        "reasons": dict(collections.Counter(r["reason"] for r in rows).most_common()),
        "by_source": _rate(rows, lambda r: r["source"]),
        "by_style": _rate(rows, lambda r: r["style"]),
        "by_category": _rate(rows, lambda r: r["category"]),
        "by_multi_step": _rate(rows, lambda r: r["multi_step"]),
        "both_styles_of_a_candidate": _candidate_agreement(rows),
        "random_kept": [show(r) for r in rng.sample(kept, min(N_SHOW, len(kept)))],
        "random_rejected": [show(r) for r in rng.sample(rejected, min(N_SHOW, len(rejected)))],
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
                    f"\n\ndef solve(*args, **kwargs):\n    out = _ref(*args, **kwargs)\n    return out.drop(out.columns[-1])"
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
    context = {}
    for line in Path(args.candidates).read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            context[c["id"]] = c.get("doc_context", "")
    missing_ctx = sum(not context.get(e["candidate_id"]) for e in examples)

    revision = args.revision or (SOLVER_REVISION if args.model == SOLVER_MODEL else None)
    settings = roundtrip_settings(args.model, revision)
    checkpoint = Path(args.checkpoint)
    items = [{**e, "messages": solver_messages(e, context.get(e["candidate_id"], ""))} for e in examples]

    def make_row(item, text):
        return {"raw": text}

    done = load_checkpoint(checkpoint, settings)
    run = {"n_resumed": len(done), "n_written": 0, "oom_events": []}
    gpu, source, wall = None, "none (checkpoint complete)", 0.0
    if any(e["id"] not in done for e in examples):
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
        run = write_with_checkpoint(items, solver.solve_batch, checkpoint=checkpoint, settings=settings,
                                    batch_size=batch_size, free_memory=getattr(solver, "free_memory", lambda: None),
                                    make_row=make_row)
        wall = time.perf_counter() - t0
        done = load_checkpoint(checkpoint, settings)
    else:
        print(f"checkpoint {checkpoint} already has all {len(examples)} solver outputs: model not loaded")

    outputs = {e["id"]: done[e["id"]]["raw"] for e in examples}
    print(f"checking {len(examples)} solver outputs in the sandbox ({args.workers} workers)...")
    t1 = time.perf_counter()
    verdicts = judge_all(examples, outputs, workers=args.workers)
    check_wall = time.perf_counter() - t1

    kept = [e for e in examples if verdicts[e["id"]]["kept"]]
    for path, rows in ((Path(args.out), kept),
                       (Path(args.results), [{"id": e["id"], "candidate_id": e["candidate_id"], **verdicts[e["id"]],
                                              "solver_code": extract_code(outputs[e["id"]])} for e in examples])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)

    stats = {
        **settings,
        "decoding": "greedy",
        "context": "oracle: the example's source doc chunk(s), RAG prompt template",
        "check": "sandbox output == reference output, row and column order ignored, numeric width ignored",
        "n_without_doc_context": missing_ctx,
        "gpu": gpu,
        "batch_size_source": source,
        "batch_sizes_used": sorted({done[e["id"]].get("batch_size", 0) for e in examples}),
        "this_session": {**run, "generation_wall_time_s": round(wall, 1), "check_wall_time_s": round(check_wall, 1)},
        "generation_time_s_all_sessions": round(sum(done[e["id"]].get("latency_s", 0.0) for e in examples), 1),
        **report(examples, verdicts, outputs),
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"round trip: kept {stats['kept']} / {stats['n']} ({stats['retention']}); reasons {stats['reasons']}")
    print(f"Wrote {args.out}, {args.results}, {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
