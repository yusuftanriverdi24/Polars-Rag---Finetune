"""Instruction writer parsing + automatic instruction filters (training data v1, PART 2)."""

import json

from polars_llm.datagen.instructions import (
    MockWriter,
    instruction_problems,
    mentions_method,
    parse_writer_output,
    solution_api_identifiers,
    writer_messages,
)

REC = {
    "id": "x",
    "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
    "setup_code": 'df = pl.DataFrame({"region": ["a"], "amount": [1]})\ninputs = {"df": df}',
    "reference_solution": (
        "def solve(df: pl.DataFrame) -> pl.DataFrame:\n"
        '    return df.group_by("region", maintain_order=True).agg(pl.col("amount").sum().alias("total"))'
    ),
    "output_columns": ["region", "total"],
    "output_dtypes": ["String", "Int64"],
    "preview": "...",
    "n_rows": 1,
}


def test_parse_writer_output():
    good = 'Sure!\n{"concise": "Total  amount per `region`.", "descriptive": "For each region..."}'
    assert parse_writer_output(good) == {"concise": "Total amount per `region`.", "descriptive": "For each region..."}
    assert parse_writer_output("<think>x</think>" + good)["concise"].startswith("Total")
    assert parse_writer_output("no json here") is None
    assert parse_writer_output('{"concise": "x"}') is None  # a style is missing
    assert parse_writer_output('{"concise": "", "descriptive": "y"}') is None


def test_solution_api_identifiers():
    names = solution_api_identifiers(REC["reference_solution"])
    assert {"group_by", "agg", "col", "sum", "alias", "maintain_order"} <= names


def test_good_instruction_passes():
    text = "From `df`, compute the total `amount` for each `region` as `total`; return `region` then `total`."
    assert instruction_problems(text.replace("`amount`", "amount"), "concise", REC) == []


def test_method_names_are_flagged_but_plain_english_is_not():
    methods = solution_api_identifiers(REC["reference_solution"])
    allowed = {"region", "total", "df"}
    assert mentions_method("Use group_by on region and sum it.", methods, allowed) == "group_by"
    assert mentions_method("Keep the original order (maintain_order).", methods, allowed) == "maintain_order"
    assert mentions_method("Return the sum of amount per region as total.", methods, allowed) is None
    assert mentions_method("Call `.sum` on amount.", methods, allowed) == "sum"
    assert mentions_method("Apply sum() to amount.", methods, allowed) == "sum"
    # A method name that is also a column name is fine.
    assert mentions_method("Return the `sum` column.", methods | {"sum"}, allowed | {"sum"}) is None


def test_code_too_short_too_long_and_missing_columns():
    rec = REC
    assert "code" in instruction_problems("Compute df.group_by('region') with total and region columns.", "concise", rec)
    assert "code" in instruction_problems("```python\nx\n```\nregion total stuff here", "concise", rec)
    assert "too_short" in instruction_problems("Totals by `region`.", "concise", rec)
    assert "too_long" in instruction_problems("word " * 70 + "`region` `total`", "concise", rec)
    assert "missing_columns" in instruction_problems("Sum the amounts for every group in the table please.", "concise", rec)


def test_writer_prompt_and_mock_writer():
    msgs = writer_messages(REC)
    assert msgs[0]["role"] == "system" and "`region` (String), `total` (Int64)" in msgs[1]["content"]
    assert "Do not mention any function, method" in msgs[1]["content"]
    pair = parse_writer_output(MockWriter().write([REC])[0])
    assert instruction_problems(pair["concise"], "concise", REC) == []
    assert instruction_problems(pair["descriptive"], "descriptive", REC) == []
    json.dumps(pair)
