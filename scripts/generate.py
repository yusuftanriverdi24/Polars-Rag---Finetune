"""Generate model outputs for an experiment arm.

Writes ``results/<arm>/generations.jsonl`` (one record per task: task_id, raw
output, latency, token counts and, for RAG arms, the retrieved chunks) plus
``results/<arm>/generation_meta.json`` (model / version / seed / prompt / RAG
metadata). Scoring is a separate step (``score.py``) so a single expensive GPU
generation pass can be re-scored freely on CPU.

RAG arms (B, D; or ``--rag``) retrieve docs for each task with the task
**instruction** as the query and fill the prompt's ``{context}`` block. Arms
A/C pass no context, so their prompt is unchanged.

Examples::

    # baseline (arm A), real model on Colab GPU
    python scripts/generate.py --arm A --model Qwen/Qwen3.5-4B

    # RAG (arm B): needs the index from `python -m polars_llm.rag.index`
    python scripts/generate.py --arm B --model Qwen/Qwen3.5-4B

    # CPU smoke test of the whole pipeline, no GPU required
    python scripts/generate.py --arm mock --model mock --limit 6
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

# Make `polars_llm` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polars_llm.bench.schema import Task, load_tasks  # noqa: E402
from polars_llm.prompts import build_messages, prompt_fingerprint  # noqa: E402

RAG_ARMS = frozenset({"B", "D"})
DEFAULT_MAX_SEQ_LENGTH = 4096


def _lib_versions() -> dict:
    """Actual installed versions of the key libraries, resolved at runtime."""

    versions = {}
    for lib in (
        "polars", "torch", "transformers", "unsloth", "peft", "trl", "accelerate",
        "sentence_transformers", "lancedb",
    ):  # fmt: skip
        try:
            versions[lib] = __import__(lib).__version__
        except Exception:
            versions[lib] = None
    return versions


def check_prompt_lengths(model, messages_list: list[list[dict]], max_new_tokens: int) -> int | None:
    """Fail loudly if any prompt + max_new_tokens exceeds the model's
    ``max_seq_length`` (the HF wrapper would otherwise truncate silently).
    Returns the longest prompt length, or None if the model can't count tokens."""

    counter = getattr(model, "count_prompt_tokens", None)
    limit = getattr(model, "max_seq_length", None)
    if counter is None or limit is None:
        return None
    longest = max(counter(m) for m in messages_list)
    if longest + max_new_tokens > limit:
        raise ValueError(
            f"longest prompt is {longest} tokens; + max_new_tokens={max_new_tokens} exceeds "
            f"max_seq_length={limit}. Raise --max-seq-length or lower the context budget."
        )
    return longest


def retrieve_contexts(
    tasks: list[Task], index_dir: Path, *, k: int, budget_tokens: int, device: str | None = None
) -> tuple[dict[str, str], dict[str, list[dict]], dict]:
    """Retrieve + format context for every task. Returns (context_map,
    retrieved summaries per task, RAG metadata)."""

    from polars_llm.rag.retrieve import Retriever, format_context

    retriever = Retriever(index_dir, device=device)
    contexts: dict[str, str] = {}
    retrieved: dict[str, list[dict]] = {}
    context_tokens: dict[str, int] = {}
    for t in tasks:
        chunks = retriever.retrieve(t.instruction, k=k, budget_tokens=budget_tokens)
        contexts[t.id] = format_context(chunks)
        retrieved[t.id] = [c.summary() for c in chunks]
        context_tokens[t.id] = retriever.count(contexts[t.id]) if chunks else 0
    meta = {
        "query": "task instruction only",
        "k": k,
        "context_budget_tokens": budget_tokens,
        "index_dir": str(index_dir),
        "index_meta": retriever.meta,
        "max_context_tokens": max(context_tokens.values(), default=0),
        "mean_context_tokens": round(sum(context_tokens.values()) / max(len(context_tokens), 1), 1),
    }
    # Free the embedder (GPU memory) before the generation model is loaded.
    del retriever
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return contexts, retrieved, meta


def run_generation(
    tasks: list[Task],
    model,
    arm: str,
    out_dir: Path,
    *,
    max_new_tokens: int = 768,
    batch_size: int = 8,
    seed: int = 0,
    context_map: dict[str, str] | None = None,
    retrieved_map: dict[str, list[dict]] | None = None,
    extra_meta: dict | None = None,
) -> dict:
    """Run ``model`` over ``tasks`` and write generations + meta. Returns meta."""

    context_map = context_map or {}
    messages_list = [build_messages(t, context=context_map.get(t.id)) for t in tasks]
    longest_prompt = check_prompt_lengths(model, messages_list, max_new_tokens)

    gens = model.generate(messages_list, max_new_tokens=max_new_tokens, batch_size=batch_size)
    if len(gens) != len(tasks):
        raise RuntimeError(f"model returned {len(gens)} generations for {len(tasks)} tasks")

    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    gen_path = arm_dir / "generations.jsonl"
    with gen_path.open("w", encoding="utf-8") as fh:
        for task, gen in zip(tasks, gens):
            record = {
                "task_id": task.id,
                "output": gen.text,
                "latency_s": gen.latency_s,
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
            }
            if retrieved_map is not None:
                record["retrieved"] = retrieved_map.get(task.id, [])
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_latency = round(sum(g.latency_s for g in gens), 3)
    meta = {
        "arm": arm,
        "model": getattr(model, "name", str(model)),
        "adapter": getattr(model, "adapter", None),
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "max_seq_length": getattr(model, "max_seq_length", None),
        "max_prompt_tokens": longest_prompt,
        "enable_thinking": getattr(model, "enable_thinking", None),
        "n_tasks": len(tasks),
        "total_latency_s": total_latency,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lib_versions": _lib_versions(),
        # Ties the run to the exact prompt it was generated with.
        "prompt": prompt_fingerprint(),
    }
    if extra_meta:
        meta.update(extra_meta)
    (arm_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(tasks)} generations to {gen_path}")
    print(f"model={meta['model']} adapter={meta['adapter']} total_latency={total_latency}s")
    return meta


def _build_model(model_arg: str, adapter: str | None, tasks: list[Task], seed: int, max_seq_length: int):
    if model_arg == "mock":
        from polars_llm.models.mock import make_reference_mock

        return make_reference_mock(tasks)
    from polars_llm.models.hf import HFModel

    return HFModel(model_arg, adapter=adapter, seed=seed, max_seq_length=max_seq_length)


def main(argv: list[str] | None = None) -> int:
    from polars_llm.rag.retrieve import DEFAULT_BUDGET_TOKENS, DEFAULT_K

    parser = argparse.ArgumentParser(description="Generate model outputs for an arm.")
    parser.add_argument("--arm", required=True, help="arm name, used as results/<arm>/ dir")
    parser.add_argument("--model", required=True, help="HF model id, or 'mock' for the CPU mock")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter path/id")
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    parser.add_argument(
        "--rag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"retrieve docs into the prompt (default: on for arms {sorted(RAG_ARMS)})",
    )
    parser.add_argument("--rag-index", default="data/rag/index")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--context-budget", type=int, default=DEFAULT_BUDGET_TOKENS)
    parser.add_argument("--rag-device", default=None, help="device for the query embedder")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    use_rag = args.rag if args.rag is not None else args.arm in RAG_ARMS
    context_map = retrieved_map = None
    extra_meta = None
    if use_rag:
        # Retrieve first, so the embedder is freed before the LLM is loaded.
        context_map, retrieved_map, rag_meta = retrieve_contexts(
            tasks, Path(args.rag_index), k=args.k, budget_tokens=args.context_budget, device=args.rag_device
        )
        extra_meta = {"rag": rag_meta}
        print(
            f"retrieved k<={args.k} chunks/task within {args.context_budget} tokens "
            f"(max context {rag_meta['max_context_tokens']} tokens)"
        )

    model = _build_model(args.model, args.adapter, tasks, args.seed, args.max_seq_length)
    run_generation(
        tasks,
        model,
        args.arm,
        Path(args.out_dir),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        context_map=context_map,
        retrieved_map=retrieved_map,
        extra_meta=extra_meta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
