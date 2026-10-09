# Polars Code Generation: RAG vs Fine-Tuning

A controlled comparison of Retrieval-Augmented Generation (RAG) and
parameter-efficient fine-tuning (QLoRA) for teaching a small open-source LLM to
write **correct, up-to-date Polars code** from natural-language instructions.

See [`PLAN.md`](PLAN.md) for the full project plan (it is the single source of
truth).

## Status

- **Phase 0 (setup):** repo skeleton, Polars pinned to `2.0.0`, `uv`-managed env.
- **Phase 1 (benchmark):** sandbox executor, runner, error classifier, and
  120 validated tasks under `benchmark/tasks.jsonl` (see `benchmark/REVIEW.md`).
- **Phase 2 (baseline), in progress:** shared prompt template, HF/Unsloth
  inference wrapper, split `generate` / `score` scripts, and the Arm A Colab
  runner `notebooks/02_baseline.ipynb`.

## Setup

```bash
uv sync --all-extras   # or: uv sync  (benchmark tooling only)
```

## Run the benchmark tooling

Validate that every reference solution runs cleanly under the pinned Polars
version (fails on any error or deprecation warning):

```bash
uv run python -m polars_llm.bench.validate_tasks
```

Score a set of model outputs and write a results JSON:

```bash
uv run python -m polars_llm.bench.runner --help
```

## Benchmark tasks

Tasks are authored per category under `benchmark/tasks_src/<category>.py` (each
exposing a `TASKS` list) and merged into `benchmark/tasks.jsonl`:

```bash
uv run python benchmark/build_tasks.py        # regenerate tasks.jsonl
uv run python -m polars_llm.bench.validate_tasks   # all references must pass
```

## Generation + scoring (arms)

Generation (GPU) and scoring (CPU) are separate steps, so one expensive
generation pass can be re-scored freely:

```bash
# real model (run on a Colab L4/A100 via notebooks/02_baseline.ipynb)
python scripts/generate.py --arm A --model Qwen/Qwen3.5-4B
python scripts/score.py   --arm A        # -> results/A/scores.json

# CPU smoke test of the whole pipeline, no GPU needed
python scripts/generate.py --arm mock --model mock --limit 6
python scripts/score.py   --arm mock
```

The base model (`Qwen/Qwen3.5-4B`) is loaded in bf16 with no 4-bit quantization
and thinking disabled. `transformers` / `torch` / `unsloth` are **Colab-only**
and installed by the notebook; they are not part of the local `uv` environment,
so `generate.py` can only run with `--model mock` locally.

## Tests

```bash
uv run pytest
```
