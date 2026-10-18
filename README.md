# Polars Code Generation: RAG vs Fine-Tuning

**Question.** Polars 2.0 removed or renamed many APIs that a small LLM learned
from older code (`groupby`, `melt`, `with_row_count`, `pivot(columns=)`,
`how="outer"`, ...). Given only the Polars 2.0 documentation, is it better to
**retrieve** the docs into the prompt (RAG) or to **fine-tune** the model on
examples derived from the same docs?

This README is the project report. [`PLAN.md`](PLAN.md) is the full plan and
decisions log; every number below is recorded there with its provenance.

> **Status: draft.** Arms C and D were run on an A100, while A/B/B2 ran on an L4.
> For hardware parity, C and D will be regenerated on an L4 with the same
> adapter; that run becomes the official result. **The C/D numbers below are the
> A100 run, pending the L4 rerun.**

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3.5-4B` (instruct), bf16, no quantization, thinking off |
| Benchmark | 120 hand-written tasks, git tag `benchmark-v1.1`: 9 categories (select_filter, group_by, join, window, lazy, string, datetime, list_struct, reshape), 40 easy / 50 medium / 30 hard |
| Scoring | the generated `solve()` runs in a sandbox under `polars==2.0.0`; pass = output frame equals the reference's (per-task row/column-order flags, numeric widths ignored). Failures are classified: `wrong_result`, `hallucinated_api`, `deprecated_api`, `runtime_error`, `syntax_error`, ... |
| Decoding | greedy, `max_new_tokens=768`, **one prompt template for all arms**; RAG arms only fill its `{context}` block |
| Metric | pass@1 (one greedy sample per task) |

| Arm | Model | Context |
|---|---|---|
| **A** base | base model | none |
| **B** naive RAG | base model | top-5 doc chunks (2,000 tokens), query = instruction, granite-small-r2 embeddings + BM25 (RRF) |
| **B2** improved RAG | base model | one round of retrieval fixes, chosen on API recall only: filtered corpus, query = instruction + the model's own zero-shot draft, k=8, 4,000 tokens |
| **C** fine-tuned | base + LoRA adapter | none |
| **D** hybrid | base + LoRA adapter | B2's retrieval, with C's output as the draft |

## Results

![pass@1 per arm](results/figures/pass_at_1.png)

| Arm | pass@1 | pass | wrong_result | hallucinated_api | runtime_error | deprecated_api | syntax_error |
|---|---|---|---|---|---|---|---|
| A | **0.408** | 49 | 19 | 30 | 19 | 3 | 0 |
| B | **0.525** | 63 | 23 | 15 | 13 | 5 | 1 |
| B2 | **0.708** | 85 | 17 | 3 | 14 | 1 | 0 |
| C *(A100, pending L4)* | **0.467** | 56 | 23 | 22 | 13 | 3 | 3 |
| D *(A100, pending L4)* | **0.517** | 62 | 24 | 18 | 16 | 0 | 0 |

![pass@1 by category](results/figures/by_category.png)

### Findings

1. **RAG beats fine-tuning on the same docs.** Naive RAG (B, 0.525) already
   beats the fine-tuned model (C, 0.467), and one round of retrieval fixes
   (B2, 0.708) adds another 18 points. Both use only the Polars 2.0 docs.
2. **Fine-tuning fixes API names but weakens multi-step answers.** C vs A:
   27 tasks gained, 20 lost. The gains are mostly corrected names in the
   categories the training data covers well (datetime +6, list_struct +7,
   string +6; 12 of the 27 gains were A `hallucinated_api` failures). The losses
   come from shorter answers that leave out part of the instruction. The median
   solution is 2 lines (A: 3). `.select(` appears in 11 of 120 C answers
   (A: 25) and 11 C wrong results are extra output columns (B2: 3). `.sort(`
   appears 6 times (A: 24). Window tasks use `.over(` in 2/13 answers (A: 8),
   and the model invents names instead (`running_sum`, `mean_over`). Window
   drops to 1/13 and group_by to 7/14. This matches the training data: only 3
   of 1,080 training solutions use `.over(`, and 44% are select/filter
   examples.
3. **The hybrid is worse than RAG alone.** D (0.517) vs B2 (0.708): 7 gained,
   30 lost, with the same failure modes as C (missing projections and sorts,
   invented window APIs). Retrieval still helps the fine-tuned model (C -> D:
   +14 / -8) but much less than the base model (A -> B2: +36 net).
4. **Retrieval quality matters more than retrieval itself.** B's gain over A
   is mostly fewer hallucinated APIs (30 -> 15). B2's changes (dropping SQL
   reference pages and duplicates, the draft-code query, a bigger budget)
   raised non-core API recall from 0.28 to 0.54 and pass@1 by 18 points. The
   gains are largest in window (3 -> 9 of 13) and string (7 -> 11 of 14).
   Reshape stays the hardest category for every arm (at most 4/13).

## Training data pipeline (data v1)

All training data comes from the Polars 2.0 docs, the same source RAG uses.
The instructions were written by an open model, so no Claude-written text
enters training.

| Step | Output |
|---|---|
| Doc examples -> tasks (`datagen/convert.py`): each displayed doctest value / `print` becomes a candidate; its data constructions go to setup, the rest to `solve()` | 1,540 unique candidates |
| Sandbox validation under 2.0.0 (clean, no warnings, DataFrame result, two runs agree) | 1,331 valid |
| Drop solutions that call no documented API (71), leakage control vs the 120 benchmark tasks (6: solution similarity, structure, API set; the instruction-similarity check, calibrated against the benchmark's own nearest-neighbour cosine, dropped 0), downsampling of trivial single-API examples (63) | **1,191 candidates** |
| Requests written by `Qwen/Qwen3.5-9B` (one concise + one descriptive per candidate) + form filters (no method names, no code, length, output columns named) | **2,287 requests** |
| **Round trip**: Qwen3.5-9B solves each request from setup + oracle docs (the example itself removed) and its output must equal the reference | kept **1,428** (0.624) |
| Manual audit of 20 kept: **4/20 wrong** (all "output-equivalent": the request describes the output on this input, not the code) -> remedy: drop every kept request whose solver code uses none of the reference's non-core APIs (`api_mismatch`) | **1,202** (226 dropped) |
| Page-grouped 90/10 split | **train 1,080 / dev 122** |

Training (`train/sft.py`, Unsloth): LoRA r=16, alpha=16 on all attention + MLP
projections, lr 2e-4 cosine, 2 epochs, effective batch 16, loss on the
completion only, same prompt as Arm C. On the A100-80GB it took **438 s**
(136 steps) with **12.45 GB peak VRAM**. Train loss went from 0.375 to 0.061;
**dev loss was 0.189 after epoch 1 and 0.190 after epoch 2** (flat).

## Cost

| Arm | GPU | Draft | Retrieval | Generation | End-to-end (120 tasks) |
|---|---|---|---|---|---|
| B2 | L4 | 175.5 s (Arm A) | 27.5 s | 339.4 s | 542.4 s (4.52 s/task) |
| D *(pending L4)* | A100 | 199.2 s (Arm C) | 31.9 s | 178.9 s | 410.1 s (3.42 s/task) |

The two runs used different GPUs, so these wall times are not comparable.
That is why C/D are being rerun on the L4. In tokens, D's prompts are the same
size as B2's (+0.4%) and its outputs are 19% shorter. D also needs the one-off
training run (438 s on the A100). Arm B's generation took 242.3 s on the L4.

## Limitations

- **AI-labelled audits.** The round-trip audit labels come from an AI
  reviewer (Claude) and were accepted by the user. They are not the judgement
  of a human Polars expert. The second audit (20 requests after the remedy)
  has pre-labels only (19 correct, 1 output-equivalent mismatch); its final
  `audit_label` column is still empty.
- **Conservative round trip.** Of 20 audited rejections, 14 were correct
  requests that the 9B solver could not implement (Wilson 95% CI 0.48-0.85).
  So roughly 600 of the 859 rejected requests were probably usable. The
  filter keeps quality high but removes exactly the hard examples.
- **Under-represented categories.** In train, group_by is 2.2%, reshape 2.6%
  and window 8.0%, against about 11% each in the benchmark. select_filter is
  44% of train (12% of the benchmark), and the multi-step share fell from 0.28
  to 0.18 through the round trip. Arm C's results on these categories are
  limited by the data, not only by the method.
- **Single seed, single sample.** One greedy run per arm. A difference of a
  few tasks (e.g. B 63 vs D 62) is within noise.
- **Flaky tasks.** Polars' streaming engine does not guarantee `group_by`
  order. A determinism audit (each generation re-run 50 times) found 0 flaky
  pass/fail outcomes in A, B, B2 and D, and 2 in C. In both C cases the code
  leaves out a sort that the instruction asks for, so it passes only by chance
  (`lazy_011` passed 27/50 runs, `lazy_012` 21/50). The reported score keeps
  the stored outcome; counting both as failures gives 55/120.
- **Hardware.** A/B/B2 ran on an L4 and the current C/D on an A100. bf16
  greedy outputs can differ between GPUs. The L4 rerun of C/D fixes this; the
  A100 run is kept as a hardware-sensitivity check.
- **One data recipe.** The fine-tuning result is for this data (doc-derived,
  round-trip-filtered). A data mix with balanced categories, more multi-step
  examples and explicit final projections/sorts might do better. It is listed
  as future work in `PLAN.md`.

## Reproduce

```bash
uv sync --all-extras          # local tooling: benchmark, scoring, RAG eval, figures
uv run pytest
uv run python -m polars_llm.bench.validate_tasks   # all 120 references pass under 2.0.0
```

GPU steps run in Colab (the notebooks only clone the repo, install it and call
scripts):

| Notebook | GPU | Produces |
|---|---|---|
| `notebooks/02_baseline.ipynb` | L4 | Arm A: `results/A/` |
| `notebooks/03_rag.ipynb` | L4 | RAG index, retrieval eval, Arm B: `results/B/` |
| `notebooks/04_rag_v2.ipynb` | L4 | corpus v2, retrieval grid, Arm B2: `results/B2/` |
| `notebooks/05_datagen.ipynb` | A100 (L4 fallback) | instructions, filters, leakage, round trip, remedy, split: `data/sft/`, `results/datagen/` |
| `notebooks/06_finetune.ipynb` | L4 | LoRA adapter, Arms C and D, determinism audits: `results/C/`, `results/D/` (`SKIP_TRAINING=True` reuses an existing adapter) |

Generation (GPU) and scoring (CPU) are separate, so any arm can be re-scored
locally:

```bash
python scripts/generate.py --arm A --model Qwen/Qwen3.5-4B      # GPU
uv run python scripts/score.py --arm A                          # -> results/A/scores.json
uv run python benchmark/determinism_audit.py --scores results/A/scores.json --out /tmp/audit.json
uv run python scripts/make_figures.py                           # -> results/figures/*.png
python scripts/generate.py --arm mock --model mock --limit 6    # CPU smoke test, no GPU
```

`transformers` / `torch` / `unsloth` are Colab-only (extra `colab`). Locally,
`generate.py` runs only with `--model mock`.

## Repository

```
benchmark/   tasks (tasks_src/ -> tasks.jsonl), build + determinism audit, AUDIT.md
src/polars_llm/
  bench/     sandbox executor, runner, comparator, error classifier, metrics
  models/    HF/Unsloth wrapper, mock model
  prompts/   the shared prompt template
  rag/       ingest, chunking, corpus v2, index, hybrid retrieval, retrieval eval
  datagen/   doc example conversion, validation, leakage, instructions, round trip, remedy
  train/     SFT (LoRA)
scripts/     generate.py, score.py, make_figures.py, ...
notebooks/   Colab runners 02-06
results/     per-arm generations, scores, metadata and audits; figures/
tests/
```
