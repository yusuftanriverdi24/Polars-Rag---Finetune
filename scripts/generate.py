"""Generate model outputs for an experiment arm.

Writes ``results/<arm>/generations.jsonl`` (one record per task: task_id, raw
output, latency, token counts and, for RAG arms, the retrieved chunks) plus
``results/<arm>/generation_meta.json`` (model / version / seed / prompt / RAG
metadata). Scoring is a separate step (``score.py``) so a single expensive GPU
generation pass can be re-scored freely on CPU.

RAG arms (B, B2, D; or ``--rag``) retrieve docs for each task and fill the
prompt's ``{context}`` block. Arms A/C pass no context, so their prompt is
unchanged. The query is the task **instruction** (Arm B), or with
``--query-mode draft`` the instruction + the model's own zero-shot draft
(Arm B2: draft -> retrieve -> final answer). The drafts are the Arm A
generations of the same model and prompt (checked against their
``generation_meta.json``); their latency is added to the cost metrics.
``--retrieval-config`` takes the chosen config from
``results/rag/retrieval_eval_v2.json`` (index, query mode, k, budget,
max_seq_length).

Examples::

    # baseline (arm A), real model on Colab GPU
    python scripts/generate.py --arm A --model Qwen/Qwen3.5-4B

    # RAG (arm B): needs the index from `python -m polars_llm.rag.index`
    python scripts/generate.py --arm B --model Qwen/Qwen3.5-4B

    # improved RAG (arm B2): config chosen by eval_retrieval_v2 on API recall
    python scripts/generate.py --arm B2 --model Qwen/Qwen3.5-4B \
        --retrieval-config results/rag/retrieval_eval_v2.json

    # CPU smoke test of the whole pipeline, no GPU required
    python scripts/generate.py --arm mock --model mock --limit 6
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

# Make `polars_llm` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polars_llm.bench.schema import Task, load_tasks  # noqa: E402
from polars_llm.prompts import build_messages, prompt_fingerprint  # noqa: E402

RAG_ARMS = frozenset({"B", "B2", "D"})
DEFAULT_DRAFTS = "results/A/generations.jsonl"
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


def check_draft_provenance(drafts_meta: dict, model: str, adapter: str | None) -> None:
    """Drafts must be the same model's own zero-shot answers (no context, same
    prompt), or the draft-code query would smuggle in outside help."""

    problems = []
    if drafts_meta.get("model") != model:
        problems.append(f"model {drafts_meta.get('model')!r} != {model!r}")
    if drafts_meta.get("adapter") != adapter:
        problems.append(f"adapter {drafts_meta.get('adapter')!r} != {adapter!r}")
    if drafts_meta.get("rag"):
        problems.append("drafts were generated with retrieved context")
    if (drafts_meta.get("prompt") or {}).get("template_sha256") != prompt_fingerprint()["template_sha256"]:
        problems.append("prompt template hash differs")
    if problems:
        raise ValueError("drafts are not this model's zero-shot answers: " + "; ".join(problems))


def retrieve_contexts(
    tasks: list[Task],
    index_dir: Path,
    *,
    k: int,
    budget_tokens: int,
    device: str | None = None,
    query_mode: str = "instruction",
    drafts: dict[str, dict] | None = None,
    retriever=None,
) -> tuple[dict[str, str], dict[str, list[dict]], dict]:
    """Retrieve + format context for every task. Returns (context_map,
    retrieved summaries per task, RAG metadata incl. per-task latencies)."""

    from polars_llm.rag.retrieve import Retriever, build_query, format_context

    if query_mode == "draft":
        missing = [t.id for t in tasks if t.id not in (drafts or {})]
        if missing:
            raise ValueError(f"query_mode='draft' needs a draft for every task; missing {missing[:3]}")
    t_load = time.perf_counter()
    retriever = retriever or Retriever(index_dir, device=device)
    load_s = time.perf_counter() - t_load
    contexts: dict[str, str] = {}
    retrieved: dict[str, list[dict]] = {}
    context_tokens: dict[str, int] = {}
    latency: dict[str, float] = {}
    for t in tasks:
        query = build_query(query_mode, t.instruction, (drafts or {}).get(t.id, {}).get("output"))
        t0 = time.perf_counter()
        chunks = retriever.retrieve(query, k=k, budget_tokens=budget_tokens)
        latency[t.id] = round(time.perf_counter() - t0, 4)
        contexts[t.id] = format_context(chunks)
        retrieved[t.id] = [c.summary() for c in chunks]
        context_tokens[t.id] = retriever.count(contexts[t.id]) if chunks else 0
    meta = {
        "query": "task instruction only" if query_mode == "instruction" else "task instruction + zero-shot draft code",
        "query_mode": query_mode,
        "k": k,
        "context_budget_tokens": budget_tokens,
        "index_dir": str(index_dir),
        "index_meta": retriever.meta,
        "max_context_tokens": max(context_tokens.values(), default=0),
        "mean_context_tokens": round(sum(context_tokens.values()) / max(len(context_tokens), 1), 1),
        "retriever_load_s": round(load_s, 3),
        "retrieval_latency_s": latency,
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
    retrieval_latency: dict[str, float] | None = None,
    draft_latency: dict[str, float] | None = None,
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
            if retrieval_latency is not None:
                record["retrieval_latency_s"] = retrieval_latency.get(task.id)
            if draft_latency is not None:
                record["draft_latency_s"] = draft_latency.get(task.id)
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
    if retrieval_latency is not None or draft_latency is not None:
        # End-to-end cost = draft (if any) + retrieval + final generation.
        r = round(sum((retrieval_latency or {}).values()), 3)
        d = round(sum((draft_latency or {}).values()), 3)
        meta["cost"] = {
            "draft_latency_s": d,
            "retrieval_latency_s": r,
            "generation_latency_s": total_latency,
            "end_to_end_latency_s": round(d + r + total_latency, 3),
            "end_to_end_latency_per_task_s": round((d + r + total_latency) / max(len(tasks), 1), 3),
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
    from polars_llm.rag.retrieve import DEFAULT_BUDGET_TOKENS, DEFAULT_K, QUERY_MODES

    parser = argparse.ArgumentParser(description="Generate model outputs for an arm.")
    parser.add_argument("--arm", required=True, help="arm name, used as results/<arm>/ dir")
    parser.add_argument("--model", required=True, help="HF model id, or 'mock' for the CPU mock")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter path/id")
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--max-seq-length", type=int, default=None,
        help=f"default: the retrieval config's value, else {DEFAULT_MAX_SEQ_LENGTH}",
    )
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
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="instruction")
    parser.add_argument("--drafts", default=DEFAULT_DRAFTS, help="zero-shot generations used by --query-mode draft")
    parser.add_argument(
        "--retrieval-config", default=None,
        help="retrieval_eval_v2.json: use its 'chosen' index / query mode / k / budget / max_seq_length",
    )
    args = parser.parse_args(argv)

    retrieval_config = None
    if args.retrieval_config:
        retrieval_config = json.loads(Path(args.retrieval_config).read_text(encoding="utf-8"))["chosen"]
        args.rag_index = retrieval_config["index_dir"]
        args.query_mode = retrieval_config["query_mode"]
        args.k = retrieval_config["k"]
        args.context_budget = retrieval_config["budget_tokens"]
        if args.max_seq_length is None:
            args.max_seq_length = retrieval_config["max_seq_length"]
    if args.max_seq_length is None:
        args.max_seq_length = DEFAULT_MAX_SEQ_LENGTH

    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    use_rag = args.rag if args.rag is not None else args.arm in RAG_ARMS
    context_map = retrieved_map = retrieval_latency = draft_latency = None
    extra_meta = None
    if use_rag:
        drafts = None
        if args.query_mode == "draft":
            from polars_llm.rag.retrieve import load_drafts

            drafts_path = Path(args.drafts)
            drafts_meta = json.loads((drafts_path.parent / "generation_meta.json").read_text(encoding="utf-8"))
            check_draft_provenance(drafts_meta, args.model, args.adapter)
            drafts = load_drafts(drafts_path)
            draft_latency = {t.id: drafts[t.id]["latency_s"] for t in tasks if t.id in drafts}
        # Retrieve first, so the embedder is freed before the LLM is loaded.
        context_map, retrieved_map, rag_meta = retrieve_contexts(
            tasks, Path(args.rag_index), k=args.k, budget_tokens=args.context_budget, device=args.rag_device,
            query_mode=args.query_mode, drafts=drafts,
        )
        retrieval_latency = rag_meta.pop("retrieval_latency_s")
        if drafts is not None:
            rag_meta["drafts"] = {
                "file": str(drafts_path),
                "model": drafts_meta.get("model"),
                "timestamp_utc": drafts_meta.get("timestamp_utc"),
            }
        if retrieval_config is not None:
            rag_meta["retrieval_config"] = {"file": args.retrieval_config, **retrieval_config}
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
        retrieval_latency=retrieval_latency,
        draft_latency=draft_latency,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
