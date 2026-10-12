"""PART 2 of training data v1: natural-language requests for each candidate.

An open instruct model (``WRITER_MODEL``, run on the L4; never Claude, so no
style leakage from the Claude-written benchmark) sees a candidate's setup
code, solution and an output preview, and writes **two** requests a user could
have typed: a ``concise`` and a ``descriptive`` one. They must say *what* to
compute, name the output columns (in order), and must not name Polars
methods or contain code.

``instruction_problems`` then filters each request automatically:

``too_short`` / ``too_long``
    word count outside ``LENGTH_LIMITS[style]``.
``code``
    a code fence, ``pl.`` / ``df.``-style attribute access, a call like
    ``name(``, an assignment, ``lambda``/``import``, or backticked code.
``names_method``
    mentions a method/function name the solution calls. Identifiers with an
    underscore or a dot (``group_by``, ``with_columns``, ``str.contains``)
    are matched anywhere as words; plain English words (``sum``, ``sort``,
    ``filter``, ``join``, ``mean``) only in code form (``.sum``, ``sum(``,
    a backticked ``sum``), otherwise every request saying "the sum of" would
    be dropped. Names that are also column names are allowed.
``missing_columns``
    for outputs with at most ``MAX_COLUMNS_TO_CHECK`` columns, an output
    column name that does not appear in the request.

Robust generation (``write_with_checkpoint``):

* **Adaptive batching.** The starting batch size comes from the GPU memory
  that is free after the model is loaded (``initial_batch_size``). On a CUDA
  out-of-memory error the cached memory is released, the batch size is halved
  and the *same* batch is retried, down to 1; the smaller size is kept for the
  rest of the run. ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is set
  before torch is imported, to reduce fragmentation.
* **Checkpoint / resume.** Every finished batch is appended to
  ``--checkpoint`` (on Google Drive in the notebook), flushed and fsynced. On
  restart, candidates already in the checkpoint are skipped; a line cut off by
  a crash is discarded. Rows record the writer model, revision, prompt hash
  and ``max_new_tokens``; a checkpoint written with other settings is refused
  rather than mixed. Once every candidate is done, ``--out`` is written in
  candidate order, independent of how the run was interrupted.

CLI (GPU)::

    python -m polars_llm.datagen.instructions --checkpoint /drive/instructions_checkpoint.jsonl
    python -m polars_llm.datagen.instructions --model mock     # CPU smoke test
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable

WRITER_MODEL = "Qwen/Qwen3.5-9B"
WRITER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STYLES = ("concise", "descriptive")
LENGTH_LIMITS = {"concise": (6, 60), "descriptive": (15, 150)}
MAX_COLUMNS_TO_CHECK = 8
DEFAULT_CANDIDATES = Path("data/sft/candidates.jsonl")
DEFAULT_OUT = Path("data/sft/instructions_raw.jsonl")
DEFAULT_META = Path("results/datagen/instructions_meta.json")
DEFAULT_CHECKPOINT = Path("data/sft/instructions_checkpoint.jsonl")
ALLOC_CONF = "expandable_segments:True"
# Starting batch size from the GPU memory free after loading the writer: keep
# RESERVE_GB free and budget GB_PER_SEQUENCE per prompt (~1-1.5k prompt tokens
# + 400 new; prefill dominates, especially on the torch fallback of the
# linear-attention layers). A power of two, at most MAX_AUTO_BATCH. Only a
# starting point: OOM halving corrects an overestimate.
RESERVE_GB = 2.0
GB_PER_SEQUENCE = 1.0
MAX_AUTO_BATCH = 32

WRITER_SYSTEM = (
    "You write realistic requests that a data analyst would type to ask a colleague for a table transformation. "
    "You describe the desired result in plain English and never write code."
)

WRITER_TEMPLATE = """A function transforms the input table(s) below. Write two requests that a user could have written to get exactly this output from these inputs.

Inputs (variable names: {params}):
```python
{setup_code}
```

Function:
```python
{solution}
```

Output columns, in order: {columns}
Output preview:
{preview}

Write:
1. "concise": one or two short sentences.
2. "descriptive": two to four sentences explaining what to compute and how the result should look.

Rules for both:
- Say WHAT to compute, not how to compute it. Do not mention any function, method, module or library name, and do not use code-like names of operations (for example do not write group_by, with_columns, agg, select).
- Refer to the inputs by their variable names ({params}).
- Name every output column exactly as it is spelled, in backticks, and say in which order the columns should appear{many_columns}.
- If the rows must be in a particular order, say so.
- No code, no code blocks.

Reply with only a JSON object: {{"concise": "...", "descriptive": "..."}}"""


def writer_messages(rec: dict) -> list[dict]:
    cols = rec["output_columns"]
    many = (
        " (with more than 8 output columns you may describe them as a group, e.g. all original columns)"
        if len(cols) > MAX_COLUMNS_TO_CHECK
        else ""
    )
    params = ", ".join(f"`{p}`" for p in _params(rec["signature"]))
    user = WRITER_TEMPLATE.format(
        params=params,
        setup_code=rec["setup_code"],
        solution=rec["reference_solution"],
        columns=", ".join(f"`{c}` ({t})" for c, t in zip(cols, rec["output_dtypes"])),
        preview=rec["preview"],
        many_columns=many,
    )
    return [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": user}]


def writer_prompt_sha256() -> str:
    return hashlib.sha256(f"{WRITER_SYSTEM}\n\x1e\n{WRITER_TEMPLATE}".encode()).hexdigest()


def _params(signature: str) -> list[str]:
    from ..models.mock import solve_param_names

    return solve_param_names(signature)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_writer_output(text: str) -> dict[str, str] | None:
    """{'concise': ..., 'descriptive': ...} from the writer's reply, or None."""

    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out = {s: obj.get(s) for s in STYLES}
    if not all(isinstance(v, str) and v.strip() for v in out.values()):
        return None
    return {s: " ".join(v.split()) for s, v in out.items()}


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def solution_api_identifiers(solution: str) -> set[str]:
    """Method/function names the solution calls or accesses on objects
    (``df.group_by(...).agg`` -> group_by, agg; ``pl.col`` -> col; ``x.str.contains`` -> str, contains)."""

    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)  # keyword args are API vocabulary too (maintain_order, how, ...)
    return {n for n in names if not n.startswith("_")} - {"print", "len", "range", "list", "dict", "str", "int"}


_CODE_PATTERNS = [
    re.compile(r"```"),
    re.compile(r"\b(?:pl|df|lf|cs)\.\w"),
    re.compile(r"\b[A-Za-z_]\w*\([^)]*\)"),  # a call: name(...)
    re.compile(r"\w\s*==?\s*[\w\"'\[]"),  # assignment / comparison in code form
    re.compile(r"\blambda\b|\bimport\b|=>|\bdef\b"),
]
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def has_code(text: str, allowed_tokens: set[str]) -> bool:
    if "```" in text:
        return True
    for tok in _BACKTICK_RE.findall(text):
        if tok not in allowed_tokens and re.search(r"[().=\[\]]", tok):
            return True
    stripped = _BACKTICK_RE.sub(" ", text)  # column names may contain anything
    return any(p.search(stripped) for p in _CODE_PATTERNS)


def mentions_method(text: str, method_names: set[str], allowed_tokens: set[str]) -> str | None:
    lower = text.lower()
    backticked = {t.lower() for t in _BACKTICK_RE.findall(text)}
    allowed = {t.lower() for t in allowed_tokens}
    for name in sorted(method_names):
        n = name.lower()
        if n in allowed:
            continue
        if "_" in n:
            if re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", lower):
                return name
        elif n in backticked or re.search(rf"\.{re.escape(n)}\b|\b{re.escape(n)}\(", lower):
            return name
    return None


def instruction_problems(text: str, style: str, rec: dict) -> list[str]:
    problems = []
    n_words = len(text.split())
    lo, hi = LENGTH_LIMITS[style]
    if n_words < lo:
        problems.append("too_short")
    if n_words > hi:
        problems.append("too_long")
    cols = list(rec["output_columns"])
    allowed = set(cols) | set(_params(rec["signature"]))
    if has_code(text, allowed):
        problems.append("code")
    if mentions_method(text, solution_api_identifiers(rec["reference_solution"]), allowed):
        problems.append("names_method")
    if len(cols) <= MAX_COLUMNS_TO_CHECK:
        lower = text.lower()
        if any(c.lower() not in lower for c in cols):
            problems.append("missing_columns")
    return problems


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------


class MockWriter:
    """Deterministic template writer for CPU smoke tests (no model)."""

    name = "mock"
    revision = None

    def write(self, recs: list[dict], **_) -> list[str]:
        out = []
        for r in recs:
            params = " and ".join(f"`{p}`" for p in _params(r["signature"]))
            cols = ", ".join(f"`{c}`" for c in r["output_columns"])
            out.append(
                json.dumps(
                    {
                        "concise": f"Using {params}, produce a table with the columns {cols} in that order.",
                        "descriptive": (
                            f"Starting from {params}, build the result table described by the task. It should "
                            f"have {r['n_rows']} rows and exactly the columns {cols}, in that order. Keep the "
                            f"values exactly as computed."
                        ),
                    }
                )
            )
        return out


class ModelWriter:
    def __init__(self, model_id: str, revision: str | None, max_seq_length: int = 4096, seed: int = 0):
        from ..models.hf import HFModel

        self.model = HFModel(model_id, max_seq_length=max_seq_length, seed=seed, revision=revision)
        self.name, self.revision = model_id, revision

    def write(self, recs: list[dict], *, batch_size: int = 16, max_new_tokens: int = 400) -> list[str]:
        gens = self.model.generate([writer_messages(r) for r in recs], max_new_tokens=max_new_tokens,
                                   batch_size=batch_size)
        return [g.text for g in gens]

    def write_batch(self, recs: list[dict], max_new_tokens: int) -> list[tuple[str, float]]:
        """Exactly one batch of ``len(recs)``: (text, per-item latency) each."""

        gens = self.model.generate([writer_messages(r) for r in recs], max_new_tokens=max_new_tokens,
                                   batch_size=len(recs))
        return [(g.text, g.latency_s) for g in gens]

    def free_memory(self) -> None:
        gc.collect()
        self.model._torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# adaptive batching + checkpoint/resume
# ---------------------------------------------------------------------------


def gpu_info() -> dict | None:
    """GPU name and free/total memory (GB), or None without CUDA."""

    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {"name": torch.cuda.get_device_name(0), "free_gb": round(free / 2**30, 2),
            "total_gb": round(total / 2**30, 2)}


def initial_batch_size(free_gb: float, *, gb_per_sequence: float = GB_PER_SEQUENCE, reserve_gb: float = RESERVE_GB,
                       max_batch: int = MAX_AUTO_BATCH) -> int:
    """Largest power of two <= ``(free - reserve) / per-sequence``, within [1, max_batch]."""

    fits = int((free_gb - reserve_gb) / gb_per_sequence)
    bs = 1
    while bs * 2 <= min(fits, max_batch):
        bs *= 2
    return bs


def is_oom(exc: BaseException) -> bool:
    """A CUDA out-of-memory error (``torch.OutOfMemoryError``, or its message on older torch)."""

    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def writer_settings(writer, max_new_tokens: int) -> dict:
    """Settings every checkpoint row carries; a resumed run must match them exactly."""

    return {"writer_model": writer.name, "writer_revision": writer.revision,
            "writer_prompt_sha256": writer_prompt_sha256(), "max_new_tokens": max_new_tokens}


def load_checkpoint(path: Path, settings: dict) -> dict[str, dict]:
    """Rows already written, by id. An unreadable line (a crash mid-write) is
    dropped and the file rewritten without it; rows written with other
    settings abort the run."""

    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row_id = row["id"]
        except (json.JSONDecodeError, TypeError, KeyError):
            bad += 1
            continue
        mismatch = {k: (row.get(k), v) for k, v in settings.items() if row.get(k) != v}
        if mismatch:
            raise SystemExit(f"checkpoint {path} was written with other settings {mismatch}; "
                             "use a new --checkpoint path (rows are never mixed)")
        done.setdefault(row_id, row)
    if bad:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values())
        os.replace(tmp, path)
        print(f"checkpoint: dropped {bad} unreadable line(s) from {path}")
    return done


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append and force to disk, so a disconnect can't lose a finished batch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        fh.flush()
        os.fsync(fh.fileno())


def write_with_checkpoint(
    recs: list[dict],
    write_batch: Callable[[list[dict]], list[tuple[str, float]]],
    *,
    checkpoint: Path,
    settings: dict,
    batch_size: int,
    free_memory: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = print,
) -> dict:
    """Write requests for every candidate not yet in ``checkpoint``, halving the
    batch size on CUDA OOM. Returns run stats (resumed/new counts, OOM events)."""

    done = load_checkpoint(checkpoint, settings)
    todo = [r for r in recs if r["id"] not in done]
    log(f"checkpoint {checkpoint}: {len(recs) - len(todo)} of {len(recs)} candidates already written, "
        f"{len(todo)} to go")
    stats = {"n_resumed": len(recs) - len(todo), "n_written": 0, "initial_batch_size": batch_size, "oom_events": []}
    i = 0
    while i < len(todo):
        batch = todo[i : i + batch_size]
        try:
            out = write_batch(batch)
        except Exception as exc:  # noqa: BLE001 - only OOM is handled; anything else re-raises
            if not is_oom(exc):
                raise
            oom = True
        else:
            oom = False
        if oom:
            # Outside the except block, so the traceback (and the tensors it pins) is gone.
            free_memory()
            if batch_size == 1:
                raise RuntimeError(f"CUDA out of memory at batch size 1 (candidate {batch[0]['id']})")
            stats["oom_events"].append({"candidate": batch[0]["id"], "from": batch_size, "to": batch_size // 2})
            batch_size //= 2
            log(f"CUDA OOM: freed memory, retrying the same batch with batch size {batch_size}")
            continue
        rows = [
            {"id": r["id"], "raw": text, "instructions": parse_writer_output(text), "batch_size": len(batch),
             "latency_s": latency, **settings}
            for r, (text, latency) in zip(batch, out)
        ]
        append_rows(checkpoint, rows)
        i += len(batch)
        stats["n_written"] += len(rows)
        log(f"  {stats['n_resumed'] + i}/{len(recs)} written (batch size {batch_size})")
    stats["final_batch_size"] = batch_size
    return stats


def main(argv: list[str] | None = None) -> int:
    # Must be set before torch is imported (the model wrapper imports it lazily).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)
    parser = argparse.ArgumentParser(description="Write two instructions per candidate with an open model.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="append-only JSONL of finished rows; put it on Google Drive to survive disconnects")
    parser.add_argument("--model", default=WRITER_MODEL, help="HF model id, or 'mock'")
    parser.add_argument("--revision", default=None, help=f"default: {WRITER_REVISION} for {WRITER_MODEL}")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="starting batch size (default: from the free GPU memory)")
    parser.add_argument("--gb-per-sequence", type=float, default=GB_PER_SEQUENCE)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    recs = [json.loads(line) for line in Path(args.candidates).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        recs = recs[: args.limit]
    revision = args.revision or (WRITER_REVISION if args.model == WRITER_MODEL else None)
    writer = MockWriter() if args.model == "mock" else ModelWriter(args.model, revision)

    gpu = None if args.model == "mock" else gpu_info()
    if args.batch_size:
        batch_size, source = args.batch_size, "--batch-size"
    elif gpu:
        batch_size = initial_batch_size(gpu["free_gb"], gb_per_sequence=args.gb_per_sequence)
        source = "free GPU memory"
    else:
        batch_size, source = 16, "default (no GPU)"
    if gpu:
        print(f"GPU: {gpu['name']}, {gpu['free_gb']} of {gpu['total_gb']} GB free after loading the writer")
    else:
        print("GPU: none")
    print(f"starting batch size {batch_size} ({source}); "
          f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    if isinstance(writer, MockWriter):
        def write_batch(batch):
            return [(t, 0.0) for t in writer.write(batch)]

        def free_memory():
            return None
    else:
        def write_batch(batch):
            return writer.write_batch(batch, args.max_new_tokens)

        free_memory = writer.free_memory

    settings = writer_settings(writer, args.max_new_tokens)
    checkpoint = Path(args.checkpoint)
    t0 = time.perf_counter()
    run = write_with_checkpoint(recs, write_batch, checkpoint=checkpoint, settings=settings, batch_size=batch_size,
                                free_memory=free_memory)
    wall = time.perf_counter() - t0

    # The final file, in candidate order, whatever the interruption history.
    done = load_checkpoint(checkpoint, settings)
    missing = [r["id"] for r in recs if r["id"] not in done]
    if missing:
        raise SystemExit(f"{len(missing)} candidates missing from the checkpoint, e.g. {missing[:3]}")
    rows = [done[r["id"]] for r in recs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps({k: row[k] for k in ("id", "raw", "instructions")}, ensure_ascii=False) + "\n"
                      for row in rows)
    n_parsed = sum(row["instructions"] is not None for row in rows)
    batch_sizes = sorted({row["batch_size"] for row in rows})
    meta = {
        "writer_model": writer.name,
        "writer_revision": writer.revision,
        "writer_prompt_sha256": writer_prompt_sha256(),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "n_candidates": len(recs),
        "n_parsed": n_parsed,
        "gpu": gpu,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "batch_size_source": source,
        "batch_sizes_used": batch_sizes,
        "this_session": {**run, "wall_time_s": round(wall, 1)},
        "generation_time_s_all_sessions": round(sum(row.get("latency_s", 0.0) for row in rows), 1),
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(recs)} instruction pairs ({n_parsed} parsed; {run['n_resumed']} resumed, {run['n_written']} "
          f"new; batch sizes {batch_sizes}; {len(run['oom_events'])} OOM halvings) with {writer.name} in "
          f"{wall:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
