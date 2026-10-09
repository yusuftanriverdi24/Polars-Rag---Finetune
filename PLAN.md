# Polars Code Generation: RAG vs Fine-Tuning

A controlled comparison of Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning (QLoRA) for teaching a small open-source LLM to write **correct, up-to-date Polars code** from natural-language instructions.

> This file is the single source of truth for the project. Claude Code: read it before every task, follow the conventions below, and update the "Decisions log" when a decision changes.

---

## 1. Motivation

The project pins **Polars 2.0.0**, a newly released *major* version that predates the training cutoff of the small open-source models under test — so the model has effectively never seen it. Polars 2.0 is a sharp test case: the long-deprecated 1.x aliases (`groupby`, `apply`, `with_column`, `with_row_count`, `melt`, …) are **removed** in 2.0 and now raise hard errors (`AttributeRemovedError` / `ArgumentRemovedError`) rather than emitting deprecation warnings. A model relying on memorized older Polars — or falling back on pandas habits (`.loc`, `.iloc`, `inplace=`, `groupby`) — therefore fails in concrete, measurable ways. This makes "write correct code against a just-released API the model hasn't memorized" a clean testbed for knowledge-injection methods.

**Framing:** can we *teach a small model a newly released major version (Polars 2.0) that it has never seen* — via retrieval of the 2.0 docs, via fine-tuning on 2.0-grounded examples, or both?

**Research question:** For a small open-source code model, which is more effective for Polars 2.0 code generation — RAG, fine-tuning, or both combined — and what is the cost/performance trade-off?

## 2. Constraints

- **Compute:** Google Colab (free/standard tier, assume a T4 16 GB GPU). Training and open-model inference run on Colab; everything else runs locally on CPU.
- **Timeline:** As fast as possible (~1 week target). Portfolio/CV project → prioritize a clean repo, reproducible results and a strong README over exhaustive experiments.
- **Language:** All tasks, prompts, code and docs in English.

## 3. Experiment arms

| Arm | Model | Context |
|-----|-------|---------|
| A. Baseline | Base model | none |
| B. RAG | Base model | retrieved Polars docs |
| C. Fine-tuned | QLoRA-tuned model | none |
| D. Hybrid | QLoRA-tuned model | retrieved Polars docs |
| E. Reference (optional) | Strong API model (Claude) | none |

All arms use the same prompt template (except the retrieved-context block), greedy decoding (temperature 0), and the same max token limit.

## 4. Evaluation (build this first)

### 4.1 Task format
Each benchmark task is one JSON object in `benchmark/tasks.jsonl`:

```json
{
  "id": "groupby_003",
  "difficulty": "medium",            // easy | medium | hard
  "category": "group_by",            // select_filter, group_by, join, window, lazy, string, datetime, list_struct, reshape
  "instruction": "Natural-language task description.",
  "setup_code": "Python code that builds the input DataFrames (named inputs).",
  "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
  "reference_solution": "Reference implementation of solve().",
  "check": {"ignore_row_order": false, "ignore_column_order": false}
}
```

- Expected output is **computed by running the reference solution** under the pinned Polars version (never stored by hand).
- Target size: **~120 tasks** (≈40 easy / 50 medium / 30 hard), spread across categories.

### 4.2 Execution
- Model output → extract code → run `solve(**inputs)` in a **subprocess sandbox** with a timeout.
- Compare to expected output with `polars.testing.assert_frame_equal` (respecting `check` flags).
- Deprecation warnings are captured and recorded.

### 4.3 Metrics
- **pass@1** (execution accuracy), overall and per difficulty/category
- **Error taxonomy** for failures:
  - `syntax_error`
  - `pandas_leak` (e.g. `pd.`, `.loc`, `.iloc`, `inplace=`, `.groupby(`)
  - `hallucinated_api` (AttributeError / nonexistent function or argument)
  - `deprecated_api` (deprecation warning raised — recorded even when the result is correct)
  - `runtime_error`, `wrong_result`, `timeout`
- **Cost:** latency per task, training time, peak VRAM, data-generation API cost

### 4.4 Leakage control
- Benchmark tasks are created and frozen **before** fine-tuning data is generated.
- Fine-tuning examples are deduplicated against the benchmark (embedding similarity threshold + exact-match checks).

## 5. Data

### 5.1 Documentation corpus (RAG + data generation source)
- Polars User Guide + Python API reference for the **pinned Polars version**, taken from the Polars GitHub repo docs.
- Chunk by section; keep code examples intact with their surrounding explanation.

### 5.2 Fine-tuning data (synthetic)
- Generated with Claude, **grounded in doc chunks**: each generation prompt includes the relevant doc text, so the model writes code against the current API rather than from its own (possibly outdated) memory.
- Every example is **executed under the pinned Polars version**; keep only examples that run, produce the expected result, and raise no deprecation warnings.
- Same format as benchmark tasks (instruction → `solve()` implementation).
- Target: **~2,000–3,000 validated examples**.

## 6. Tech stack

- Environment: Python, `uv`, Polars **pinned to one exact version** (recorded in `pyproject.toml` and in this file)
- Base model: small code model (~1.5B–3B for speed on T4; 7B only if time allows) — chosen in Phase 0
- Fine-tuning: Unsloth (QLoRA), fallback TRL + PEFT
- RAG: sentence-transformers embedding model + LanceDB
- Inference: Hugging Face transformers / Unsloth on Colab; Anthropic API for arm E and data generation
- Tracking: results as JSON in `results/`, plots generated from them

## 7. Repository structure

```
polars-rag-vs-ft/
├── PLAN.md
├── README.md
├── pyproject.toml
├── src/polars_llm/
│   ├── bench/        # task schema, sandbox executor, runner, metrics, error classifier
│   ├── rag/          # doc ingestion, chunking, embedding, retrieval
│   ├── datagen/      # synthetic data generation + validation filter + dedup
│   ├── models/       # inference wrappers (HF/Unsloth, Anthropic)
│   └── prompts/      # prompt templates (shared across arms)
├── benchmark/tasks.jsonl
├── data/             # raw docs (gitignored), sft dataset
├── notebooks/        # thin Colab runners: clone repo, install, call scripts
├── results/          # per-arm JSON results + plots
└── tests/
```

**Workflow:** Code is written locally with Claude Code and pushed to GitHub. Colab notebooks only clone the repo, install dependencies and call scripts (no logic in notebooks). Data generation and benchmark validation run locally on CPU.

## 8. Phases

| # | Phase | Deliverable | Est. |
|---|-------|-------------|------|
| 0 | Setup | Repo skeleton, pinned Polars, base model chosen | 0.5 day |
| 1 | Benchmark | Sandbox executor, runner, error classifier, 120 validated tasks | 1–1.5 days |
| 2 | Baseline | Arm A (+ E) results | 0.5 day |
| 3 | RAG | Doc ingestion, index, retrieval, Arm B results | 1 day |
| 4 | Fine-tuning | Validated SFT dataset, QLoRA adapter, Arm C results | 1.5 days |
| 5 | Hybrid + analysis | Arm D results, comparison tables, plots, error analysis | 1 day |
| 6 | Packaging | README with results, adapter on HF Hub with model card, optional Gradio demo | 0.5–1 day |

## 9. Conventions

- Fixed random seeds everywhere; record seed and model/version info in every results file.
- One exact Polars version for everything (benchmark, data generation, evaluation).
- No hard-coded paths or secrets; API keys from environment variables.
- Every module in `bench/` has unit tests.

## 10. Decisions log

| Decision | Choice |
|----------|--------|
| Compute | Google Colab (T4) |
| Task | Natural language → Polars code |
| Language | English |
| Data generation model | Claude, grounded in pinned-version docs, execution-filtered |
| Polars version | **2.0.0** — pinned exactly in `pyproject.toml`; latest stable on PyPI as of 2026-10-09. NB: APIs deprecated in Polars 1.x (`groupby`, `apply`, `with_row_count`, …) are now *removed* in 2.0 and raise `AttributeRemovedError`/`ArgumentRemovedError` rather than emitting deprecation warnings. The error classifier treats those as `deprecated_api`. |
| Environment manager | `uv` (0.12.x); exact deps locked in `uv.lock` |
| Base model | _TBD in Phase 0_ |

## 11. Polars 2.0 — key breaking changes

Summarized from the official upgrade guide (`docs/source/releases/upgrade/2.md` in the Polars repo at tag `py-2.0.0`, downloaded into `data/raw/` by `src/polars_llm/rag/ingest.py`). These are the changes most likely to trip a model relying on memorized pre-2.0 Polars or on pandas habits — the benchmark deliberately exercises them.

**Removed APIs (now raise `AttributeRemovedError` / `ArgumentRemovedError`, each hinting at the replacement):**
- `melt()` → `unpivot()` (`id_vars`→`index`, `value_vars`→`on`)
- `with_row_count()` → `with_row_index()` — **default column name changed `"row_nr"` → `"index"`**
- `group_by(...).count()` → `group_by(...).len()`
- `DataFrame.pivot(columns=...)` → `on=...`
- `join(join_nulls=...)` → `nulls_equal=...`; `join(how="outer")` → `how="full"` (`outer_coalesce` → `how="full", coalesce=True`)
- `top_k` / `bottom_k(descending=...)` → `reverse=...`
- `str.concat()` → `str.join()` — **default delimiter changed `"-"` → `""`**
- `str.explode()` → `str.split("").explode()` (empty strings now become `null`)
- `replace(default=…/return_dtype=…)` → `replace_strict()`
- `rolling`/`group_by_dynamic(by=...)` → `group_by=...`; `rolling_*(min_periods=...)` → `min_samples=...`
- `dt.datetime()` → `dt.replace_time_zone(None)`; `Series.dt.mean()/median()` → `Series.mean()/median()`
- `approx_n_unique()` method, `LazyFrame.fetch()` → `collect()`+`head()`, `Expr.where()` → `filter()`
- `read_csv(dtypes=...)` → `schema_overrides=...`; `row_count_name/offset` → `row_index_name/offset`
- `assert_frame_equal(check_dtype=...)` → `check_dtypes=...`
- `pl.threadpool_size()` → `pl.thread_pool_size()`; `Array.width` → `size`
- `list.to_struct(n_field_strategy=…, upper_bound=…)` → `fields=…`

**Behavioral changes (silent — no error, but different results):**
- **Lazy API defaults to the streaming engine**, which does **not** guarantee row order for `group_by` / `join` / `unpivot`. Sort explicitly or pass `maintain_order` if you depend on order. (Benchmark: set `ignore_row_order` wherever order is undefined.)
- `pl.concat()` strictness tightened; `explode()` now uses `empty_as_null=False` by default.
- `is_in()` coercion is now strict (no lossy int/float coercion; time-unit conversion; naive-vs-tz datetimes raise).
- Supertype of signed ints and `UInt64` changed `Float64` → `Int128` (affects sum/supertype results).
- `read_csv`/`read_ipc` now dispatch through the lazy `scan_*(...).collect()` path.

**Still-common pandas habits that fail or mislead in Polars:** `df.groupby(...)` / `.apply(...)` / `.loc[...]` / `.iloc[...]` / `inplace=` / `.merge(...)` / `.sort_values(...)` / `.reset_index(...)` have no Polars equivalent (use `group_by`, `map_elements`/expressions, `filter`/`select`, `join`, `sort`); `value_counts()` returns a struct/two-column frame, not a Series.
