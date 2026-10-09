# Polars Code Generation: RAG vs Fine-Tuning

A controlled comparison of Retrieval-Augmented Generation (RAG) and
parameter-efficient fine-tuning (QLoRA) for teaching a small open-source LLM to
write **correct, up-to-date Polars code** from natural-language instructions.

See [`PLAN.md`](PLAN.md) for the full project plan (it is the single source of
truth).

## Status

- **Phase 0 (setup):** repo skeleton, Polars pinned to `2.0.0`, `uv`-managed env.
- **Phase 1 (benchmark), in progress:** sandbox executor, runner, error
  classifier, and seed tasks under `benchmark/tasks.jsonl`.

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

## Tests

```bash
uv run pytest
```
