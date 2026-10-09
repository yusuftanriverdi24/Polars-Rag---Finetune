"""Generate model outputs for an experiment arm.

Writes ``results/<arm>/generations.jsonl`` (one record per task: task_id, raw
output, latency, token counts) plus ``results/<arm>/generation_meta.json``
(model / version / seed metadata). Scoring is a separate step (``score.py``)
so a single expensive GPU generation pass can be re-scored freely on CPU.

Examples::

    # baseline (arm A), real model on Colab GPU
    python scripts/generate.py --arm A --model Qwen/Qwen3.5-4B

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
from polars_llm.prompts import build_messages  # noqa: E402


def _lib_versions() -> dict:
    """Actual installed versions of the key libraries, resolved at runtime."""

    versions = {}
    for lib in ("polars", "torch", "transformers", "unsloth", "peft", "trl", "accelerate"):
        try:
            versions[lib] = __import__(lib).__version__
        except Exception:
            versions[lib] = None
    return versions


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
    extra_meta: dict | None = None,
) -> dict:
    """Run ``model`` over ``tasks`` and write generations + meta. Returns meta."""

    context_map = context_map or {}
    messages_list = [build_messages(t, context=context_map.get(t.id)) for t in tasks]

    gens = model.generate(messages_list, max_new_tokens=max_new_tokens, batch_size=batch_size)
    if len(gens) != len(tasks):
        raise RuntimeError(f"model returned {len(gens)} generations for {len(tasks)} tasks")

    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    gen_path = arm_dir / "generations.jsonl"
    with gen_path.open("w", encoding="utf-8") as fh:
        for task, gen in zip(tasks, gens):
            fh.write(
                json.dumps(
                    {
                        "task_id": task.id,
                        "output": gen.text,
                        "latency_s": gen.latency_s,
                        "prompt_tokens": gen.prompt_tokens,
                        "completion_tokens": gen.completion_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_latency = round(sum(g.latency_s for g in gens), 3)
    meta = {
        "arm": arm,
        "model": getattr(model, "name", str(model)),
        "adapter": getattr(model, "adapter", None),
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "enable_thinking": getattr(model, "enable_thinking", None),
        "n_tasks": len(tasks),
        "total_latency_s": total_latency,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lib_versions": _lib_versions(),
    }
    if extra_meta:
        meta.update(extra_meta)
    (arm_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(tasks)} generations to {gen_path}")
    print(f"model={meta['model']} adapter={meta['adapter']} total_latency={total_latency}s")
    return meta


def _build_model(model_arg: str, adapter: str | None, tasks: list[Task], seed: int):
    if model_arg == "mock":
        from polars_llm.models.mock import make_reference_mock

        return make_reference_mock(tasks)
    from polars_llm.models.hf import HFModel

    return HFModel(model_arg, adapter=adapter, seed=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate model outputs for an arm.")
    parser.add_argument("--arm", required=True, help="arm name, used as results/<arm>/ dir")
    parser.add_argument("--model", required=True, help="HF model id, or 'mock' for the CPU mock")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter path/id")
    parser.add_argument("--tasks", default="benchmark/tasks.jsonl")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    model = _build_model(args.model, args.adapter, tasks, args.seed)
    run_generation(
        tasks,
        model,
        args.arm,
        Path(args.out_dir),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
