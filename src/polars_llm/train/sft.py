"""Arm C fine-tuning: Unsloth bf16 LoRA on Qwen3.5-4B (Colab L4).

Each training example is rendered **exactly** like an Arm A/C inference
prompt: ``build_messages(task, context=None)`` (the shared system prompt +
user template, no ``{context}`` block) through the tokenizer's chat template
with ``add_generation_prompt=True, enable_thinking=False`` (as ``HFModel``
does). The assistant turn is one ```` ```python ```` block with the task's
reference ``solve()``, followed by the template's end-of-turn token. The loss
is on the completion tokens only (prompt labels = -100).

The completion text is taken from the chat template itself: the full
conversation (prompt + assistant message) is rendered and must start with the
generation prompt; whatever follows (the answer + end-of-turn marker) is the
completion. If a template rendered it differently, the example falls back to
``answer + <eos>`` and the count is reported.

Hyperparameters (PLAN.md): LoRA r=16, alpha=16, dropout 0 on all attention
and MLP projections; lr 2e-4, cosine, ~3% warmup, 2 epochs, max_seq_length
2048 (longer examples are dropped and counted), effective batch 16
(4 x 4 accumulation), seed 0, dev loss every epoch, bf16, no quantization.

CLI::

    python -m polars_llm.train.sft --output-dir /content/drive/MyDrive/.../adapter
    python -m polars_llm.train.sft --dry-run        # CPU: tokenize + length stats only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import statistics
import time
from pathlib import Path

from ..bench.schema import Task, load_tasks
from ..prompts import build_messages

BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH = 2048
LORA = {"r": 16, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
TRAIN = {"learning_rate": 2e-4, "lr_scheduler_type": "cosine", "warmup_fraction": 0.03, "num_train_epochs": 2,
         "per_device_train_batch_size": 4, "gradient_accumulation_steps": 4, "weight_decay": 0.0,
         "seed": 0, "optim": "adamw_torch", "max_grad_norm": 1.0}
DEFAULT_TRAIN = Path("data/sft/train.jsonl")
DEFAULT_DEV = Path("data/sft/dev.jsonl")
DEFAULT_META = Path("results/C/train_meta.json")
IGNORE_INDEX = -100


def answer_text(task: Task) -> str:
    """The assistant turn: exactly one python block defining solve()."""

    return f"```python\n{task.reference_solution.strip()}\n```"


def render_prompt(tokenizer, task: Task) -> str:
    return tokenizer.apply_chat_template(
        build_messages(task, context=None), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def completion_text(tokenizer, task: Task, prompt: str) -> tuple[str, bool]:
    """(completion, from_template). The completion includes the end-of-turn marker."""

    answer = answer_text(task)
    full = tokenizer.apply_chat_template(
        build_messages(task, context=None) + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if full.startswith(prompt) and answer in full[len(prompt):]:
        return full[len(prompt):].rstrip("\n"), True
    return answer + (tokenizer.eos_token or ""), False


def tokenize_example(tokenizer, task: Task) -> dict:
    """input_ids / attention_mask / labels with the prompt masked out."""

    prompt = render_prompt(tokenizer, task)
    completion, from_template = completion_text(tokenizer, task, prompt)
    p_ids = tokenizer(text=prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(text=completion, add_special_tokens=False)["input_ids"]
    ids = list(p_ids) + list(c_ids)
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [IGNORE_INDEX] * len(p_ids) + list(c_ids),
        "n_prompt_tokens": len(p_ids),
        "n_completion_tokens": len(c_ids),
        "from_template": from_template,
    }


def tokenize_split(tokenizer, tasks: list[Task], max_seq_length: int = MAX_SEQ_LENGTH) -> tuple[list[dict], dict]:
    rows, too_long, fallback = [], 0, 0
    lengths = []
    for t in tasks:
        ex = tokenize_example(tokenizer, t)
        fallback += not ex["from_template"]
        if len(ex["input_ids"]) > max_seq_length:
            too_long += 1
            continue
        lengths.append(len(ex["input_ids"]))
        rows.append({k: ex[k] for k in ("input_ids", "attention_mask", "labels")})
    stats = {
        "n": len(rows),
        "dropped_too_long": too_long,
        "template_fallbacks": fallback,
        "max_tokens": max(lengths, default=0),
        "mean_tokens": round(statistics.fmean(lengths), 1) if lengths else 0,
        "total_tokens": sum(lengths),
    }
    return rows, stats


def total_steps(n_train: int, cfg: dict = TRAIN) -> int:
    per_step = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    return math.ceil(n_train / per_step) * cfg["num_train_epochs"]


def warmup_steps(n_train: int, cfg: dict = TRAIN) -> int:
    return max(1, math.ceil(cfg["warmup_fraction"] * total_steps(n_train, cfg)))


def _load_text_tokenizer():
    from transformers import AutoTokenizer

    from ..rag.chunk import TOKENIZER_ID, TOKENIZER_REVISION

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    return getattr(tok, "tokenizer", tok)


def train(args) -> dict:  # pragma: no cover - GPU only
    import torch
    from unsloth import FastLanguageModel  # noqa: I001 - unsloth must be imported before transformers
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    torch.manual_seed(TRAIN["seed"])
    model, processor = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length, dtype=torch.bfloat16, load_in_4bit=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model, **LORA, use_gradient_checkpointing="unsloth", random_state=TRAIN["seed"],
    )

    train_tasks, dev_tasks = load_tasks(args.train), load_tasks(args.dev)
    train_rows, train_stats = tokenize_split(tokenizer, train_tasks, args.max_seq_length)
    dev_rows, dev_stats = tokenize_split(tokenizer, dev_tasks, args.max_seq_length)

    targs = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        learning_rate=TRAIN["learning_rate"],
        lr_scheduler_type=TRAIN["lr_scheduler_type"],
        warmup_steps=warmup_steps(len(train_rows)),
        num_train_epochs=TRAIN["num_train_epochs"],
        per_device_train_batch_size=TRAIN["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAIN["gradient_accumulation_steps"],
        weight_decay=TRAIN["weight_decay"],
        max_grad_norm=TRAIN["max_grad_norm"],
        optim=TRAIN["optim"],
        seed=TRAIN["seed"],
        data_seed=TRAIN["seed"],
        bf16=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(dev_rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX),
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train()
    wall = time.perf_counter() - t0

    out = Path(args.output_dir)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    history = trainer.state.log_history
    return {
        "base_model": args.model,
        "adapter_dir": str(out),
        "lora": LORA,
        "train_config": {**TRAIN, "max_seq_length": args.max_seq_length, "warmup_steps": warmup_steps(len(train_rows)),
                         "total_steps": total_steps(len(train_rows)), "effective_batch": 16},
        "data": {"train": str(args.train), "dev": str(args.dev), "train_tokens": train_stats, "dev_tokens": dev_stats},
        "wall_time_s": round(wall, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "train_loss_final": result.training_loss,
        "train_loss_log": [{"step": h["step"], "epoch": h.get("epoch"), "loss": h["loss"]} for h in history if "loss" in h],
        "dev_loss_by_epoch": [{"epoch": h.get("epoch"), "eval_loss": h["eval_loss"]} for h in history if "eval_loss" in h],
        "gpu": torch.cuda.get_device_name(0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unsloth bf16 LoRA fine-tuning (Arm C).")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev", default=str(DEFAULT_DEV))
    parser.add_argument("--output-dir", default="adapters/qwen35-4b-polars-lora-v1")
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--dry-run", action="store_true", help="tokenize + report lengths only (CPU)")
    args = parser.parse_args(argv)

    if args.dry_run:
        tok = _load_text_tokenizer()
        _, tr = tokenize_split(tok, load_tasks(args.train), args.max_seq_length)
        _, dv = tokenize_split(tok, load_tasks(args.dev), args.max_seq_length)
        print(json.dumps({"train": tr, "dev": dv, "total_steps": total_steps(tr["n"]),
                          "warmup_steps": warmup_steps(tr["n"])}, indent=2))
        return 0

    meta = train(args)
    meta["timestamp_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    meta["lib_versions"] = {lib: getattr(__import__(lib), "__version__", None)
                            for lib in ("torch", "transformers", "unsloth", "peft", "trl", "accelerate", "datasets")}
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"trained in {meta['wall_time_s']}s, peak VRAM {meta['peak_vram_gb']} GB; dev loss {meta['dev_loss_by_epoch']}")
    print(f"adapter -> {meta['adapter_dir']}; meta -> {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
