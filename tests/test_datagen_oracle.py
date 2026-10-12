"""Oracle doc context for the round trip: the answer must not be in it."""

import json
from pathlib import Path

from polars_llm.datagen.oracle import (
    answer_expressions,
    answer_in_context,
    strip_candidate_blocks,
    strip_examples,
)

CANDIDATES = Path(__file__).resolve().parents[1] / "data" / "sft" / "candidates.jsonl"

DOC = """polars.DataFrame.unnest(self, columns) -> 'DataFrame'

Decompose struct columns into separate columns.

Parameters
----------
columns
    Name of the struct column(s).

Notes
-----
You can also do this:

>>> df.select(pl.col("t").struct.unnest())
shape: (1, 1)

See Also
--------
DataFrame.unnest

Examples
--------
>>> df = pl.DataFrame({"t": [{"a": 1}]})
>>> df.unnest("t")
shape: (1, 1)
"""


def test_examples_and_stray_doctests_are_dropped():
    out = strip_examples(DOC)
    assert "Decompose struct columns" in out and "Parameters\n----------" in out and "See Also" in out
    assert "Examples" not in out and ">>>" not in out and "unnest(\"t\")" not in out


def test_guide_drops_only_the_candidates_code_block():
    text = "Some prose.\n\n```python\nout = df.pivot(index=\"foo\", on=\"bar\")\n```\n\nMore prose.\n\n```python\nother_code()\n```"
    out = strip_candidate_blocks(text, {'out = df.pivot(index="foo", on="bar")'}, [])
    assert "Some prose." in out and "More prose." in out and "pivot" not in out and "other_code()" in out
    out = strip_candidate_blocks(text, set(), ['df.pivot(index="foo",   on="bar")'])  # whitespace-insensitive
    assert "pivot" not in out


def test_answer_expression_follows_names_and_collect():
    sol = ("def solve(df):\n    q = df.lazy().pivot(index=\"foo\", on=\"bar\", values=\"N\")\n"
           "    out = q.collect()\n    return out")
    assert answer_expressions(sol) == ['df.lazy().pivot(index="foo", on="bar", values="N")']
    assert answer_expressions("def solve(lf):\n    return lf.select(pl.all()).collect()") == [
        "lf.select(pl.all()).collect()", "lf.select(pl.all())"]
    assert answer_expressions("def solve(df):\n    return df") == []  # the input itself
    assert answer_in_context("def solve(df):\n    return df.select(pl.col('a').abs())",
                             ">>> df.select(\n...     pl.col('a').abs()\n... )")


def test_no_candidate_has_its_answer_in_its_own_context():
    recs = [json.loads(l) for l in CANDIDATES.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(recs) > 1000 and all("doc_context" in r for r in recs)
    assert all(answer_expressions(r["reference_solution"]) for r in recs)  # every answer is actually checked
    leaks = [r["id"] for r in recs if answer_in_context(r["reference_solution"], r["doc_context"])]
    assert leaks == []
