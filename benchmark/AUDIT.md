# Benchmark ambiguity audit (pre-freeze, benchmark v1.0)

**Scope:** all 20 `wrong_result` cases in the first real Arm A run
(`Qwen/Qwen3.5-4B`, generic system prompt, `results/A/scores.json` as scored
on 2026-10-11, pass@1 = 47/120 = 0.392).

**Rule:** a task's instruction is changed **only** if it is genuinely ambiguous,
meaning a reader could reasonably produce a different output that is also correct
and the reference would reject it. A task is never changed just because the
model failed it. The reference solution and the expected output were not
changed for any task.

**Verdicts:**
- `model wrong`: the instruction is clear and the model's output doesn't follow it.
- `comparator`: the output is correct and was rejected by the comparator, not
  the instruction (fixed in the comparator).
- `ambiguous`: the instruction did not pin down the expected output (fixed in the
  instruction).

| task | verdict | change made | reason |
|------|---------|-------------|--------|
| `select_filter_012` | model wrong | none | Asks to "return only the rows" and "keep all columns" of the table. The model joined its helper `count` column back on, so the output has a column the table does not have. |
| `group_by_004` | model wrong | none | "Highest `price` within each `category`". The model took the global max with no grouping (5 rows instead of 2). |
| `group_by_012` | model wrong | none | "For each `category`" share of the grand total. The model computed per-row shares without aggregating (4 rows instead of 3). |
| `group_by_013` | comparator | none (comparator) | Values were correct. The model's `when/then(1).sum()` gives `Int32` and the reference's boolean `.sum()` gives `UInt32`. That width/sign difference is incidental, so it is now tolerated by the default-on `ignore_numeric_width` check flag. **Now passes.** |
| `group_by_014` | model wrong | none | The instruction spells out the formula (sum of price*qty / sum of qty). The model computed sum(price)/sum(qty). |
| `join_001` | model wrong | none | Considered as possibly ambiguous about returned columns, but it names them explicitly: "producing one row per order with columns `order_id`, `customer_name`, `amount`". The model returned the raw join, which also has `customer_id` and puts the columns in a different order. |
| `join_003` | model wrong | none | Same check as `join_001`. The instruction says "Return `name` and `dept_name`", which is the same wording that `group_by_004`, `group_by_012` etc. use to define the output columns. The model also returned `dept_id`. |
| `window_002` | model wrong | none | Clear. The model wrote `.over("dept").max()` (the max over the whole column) instead of `.max().over("dept")`. |
| `window_007` | model wrong | none | Clear. The model omitted `.first()`, so it copied `page` unchanged. |
| `lazy_001` | model wrong | none | "Sum of `value` per `grp`". The model summed globally with no grouping. |
| `lazy_003` | model wrong | none | Asks for a new column named `doubled`. The model overwrote `value` and added no column. |
| `lazy_013` | model wrong | none | "Add a column" (keep all rows) holding the per-group sum. The model aggregated with `group_by`, so it dropped rows and the `amount` column. |
| `string_005` | model wrong | none | "Replace **every** forward slash". The model used `str.replace` (first match only). |
| `datetime_003` | model wrong | none | Asks for a column `weekday` with ISO numbers (Mon=1..Sun=7). The model overwrote `d` instead of adding `weekday`, and also added 1 to Polars' already-ISO `dt.weekday()`. |
| `list_struct_001` | model wrong | none | Asks for a column named `tags`. The model left the aggregated column as `tag`. |
| `list_struct_008` | model wrong | none | Asks for a new column `sorted` holding each row's list sorted. The model sorted the rows of `nums` in place (`.sort()` instead of `.list.sort()`) and added no column. |
| `list_struct_013` | model wrong | none | Asks for one row per customer with a list column `top2`. The model returned flat rows with the wrong rank logic and no `top2` column. |
| `reshape_002` | model wrong | none | Clear unpivot spec. The model string-concatenated `q1`/`q2` into both output columns. |
| `reshape_003` | model wrong | none | Clear long-to-wide pivot spec. The model did a `group_by().first()` with no pivot. (The order of the subject columns isn't specified, but the reference's order `math, science` is both first-seen and alphabetical, so this is not a live ambiguity for this input.) |
| `reshape_012` | **ambiguous** | **instruction clarified** | "One column per `product` holding the total `units` sold" doesn't say what a region/segment combination with no sales of a product should hold. The reference (`pivot(..., aggregate_function="sum")`) gives `0`, while a pivot of pre-aggregated data gives `null`, and both are reasonable readings of "total units sold". The instruction now adds "(0 where that combination sold none of the product)". The reference is unchanged. |

## Summary

- 18 × model wrong, no change.
- 1 × comparator (`group_by_013`), fixed by the new default-on `ignore_numeric_width` flag in `bench/compare.py`.
- 1 × ambiguous (`reshape_012`), instruction clarified. The reference and expected output are unchanged.

## Caveat for the Arm A numbers

The existing `results/A/generations.jsonl` was generated **before** the
`reshape_012` clarification, so for that one task the model saw the old
wording. The re-scored Arm A numbers use the new comparator and classifier but
the old generations. For a clean baseline on benchmark v1.0, regenerate Arm A
(about 3 minutes on the L4) before comparing against the RAG / fine-tuned arms.

---

# Determinism audit (benchmark v1.1)

**Why:** in the clean v1.0 Arm A rerun, `reshape_012` scored `pass` on Colab and
`wrong_result` in a local re-score of the *same* generation. Polars 2.0's
default streaming engine does not guarantee the output order of `group_by`,
`join`, `unique`, `pivot` or `unpivot`, so a benchmark score can depend on
chance. This audit looks for every such case.

**Method** (`benchmark/determinism_audit.py`; raw output in
`benchmark/determinism_audit.json`):

1. Run every task's reference solution 50 times in the sandbox and flag any
   task whose output varies in row order, column order or content.
2. Re-run every Arm A generation whose code touches `group_by`, `join`,
   `unique`, `pivot` or `unpivot` 50 times. Score each run against a fresh
   reference run under the task's check flags, and flag any generation whose
   pass/fail status changes between runs.
3. For each flagged task, turn on `ignore_row_order` / `ignore_column_order`
   **only if the instruction does not specify the order**. If the instruction
   specifies the order, leave the task as is. Fixes are made with check flags
   only. No instruction or reference changed, so the Arm A generations stay
   valid and were not regenerated.

## 1. Reference solutions (120 × 50 runs)

- **Content:** stable for all 120 tasks.
- **Column order:** stable for all 120 tasks.
- **Row order:** varies for 22 tasks. Every one already had `ignore_row_order`
  on and says order does not matter, so **no change** was needed:
  - `group_by_001`, `group_by_003` … `group_by_014` (13 tasks)
  - `join_011`
  - `lazy_001`, `lazy_006`, `lazy_007`
  - `string_013`
  - `datetime_011`
  - `list_struct_001`, `list_struct_013`
  - `reshape_011`
- **Tasks that specify an order** (e.g. "Sort the result by…", "keep the
  original order") produced the same order in all 50 runs. That covers every
  task with `ignore_row_order` off.

## 2. Arm A generations touching unordered ops (46 × 50 runs, v1.0 flags)

- **Row order:** varies for 15 generations. All are on tasks where row order is
  ignored or where the generation fails for other reasons, so each one has a
  stable outcome: 50/50 pass, or 0/50 for `group_by_014`, `lazy_013`,
  `list_struct_001` and `reshape_003`.
- **Pass/fail flips between runs:** 1 generation, `reshape_012`, which passed
  39/50. Its values are correct. It pivots after a `group_by`, so its product
  columns come out as `x, y` or `y, x` by chance, and the task checked column
  order.

## Changes (check flags only)

| task | change | reason |
|------|--------|--------|
| `reshape_012` | `ignore_column_order` → on | Flaky in step 2. The instruction asks for "one column per `product`" and does not give their order. |
| `reshape_003` | `ignore_column_order` → on | Same class: a pivot whose instruction ("a column per subject") does not give the order of the pivoted columns. Not flaky for the current Arm A generation, but a correct solution that pivots over unordered data would be. |
| `reshape_006` | `ignore_column_order` → on | Pivot. "One column per `day`" does not give an order. |
| `reshape_008` | `ignore_column_order` → on | Pivot. "One column per `action`" does not give an order. |
| `reshape_010` | `ignore_column_order` → on | Pivot. "Its January and February revenue, plus a column `growth`" lists what to include, not a column order. (Instructions that do fix an order say so explicitly, e.g. "in that order".) |

These five are all of the benchmark's `pivot` tasks.

## After the changes (v1.1 flags)

- The step-2 re-run has **0 of 46** generations flipping between pass and fail.
  `reshape_012` passes 50/50.
- Re-scoring the Arm A v1.0-rerun generations twice locally gives identical
  statuses for all 120 tasks. They also match the statuses scored on Colab.
- **Arm A pass@1 = 49/120 = 0.408**, unchanged. The Colab run had happened to
  get the passing column order for `reshape_012`; that outcome is now
  guaranteed rather than lucky.

## Guarding against regressions (`validate_tasks`)

`python -m polars_llm.bench.validate_tasks` now fails if:

- **Reference order or content varies:** the task's reference output varies
  across `--determinism-runs` runs (default 50) in a way its check flags do not
  tolerate.
- **A pivot task does not tolerate column order:** the reference pivots without
  `ignore_column_order` and without selecting an explicit column order after
  the pivot. The reference run alone can't catch `reshape_012`, because its
  reference was stable; the instability came from a correct *candidate*.

Run against the v1.0 task file, the new validator fails on exactly the five
pivot tasks above.
