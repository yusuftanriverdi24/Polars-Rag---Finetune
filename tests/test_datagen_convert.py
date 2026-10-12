"""Doc example -> task candidate conversion (training data v1, PART 1)."""

import textwrap

from polars_llm.bench.executor import run_solution
from polars_llm.datagen.convert import (
    Program,
    Stmt,
    construction_type,
    docstring_program,
    program_candidates,
    return_statement,
    skip_reason,
    user_guide_program,
)

DOC = textwrap.dedent(
    '''
    Do something.

    Examples
    --------
    >>> from datetime import date
    >>> df = pl.DataFrame({"a": [1, 2, 3], "d": [date(2020, 1, 1)] * 3})
    >>> df.with_columns(pl.col("a").cum_sum().alias("c"))
    shape: (3, 3)
    ...
    >>> lf = pl.LazyFrame({"x": ["a", "b"]})
    >>> lf.select(pl.col("x").str.to_uppercase())  # doctest: +SKIP
    <LazyFrame ...>
    >>> df.write_csv("out.csv")
    '''
)


def test_docstring_example_becomes_a_task_with_inputs_and_signature():
    prog = docstring_program("Expr.cum_sum", DOC)
    cands, rejects = program_candidates(prog)
    assert len(cands) == 1
    c = cands[0]
    assert c.params == ["df"] and c.param_types == ["DataFrame"]
    assert c.signature == "def solve(df: pl.DataFrame) -> pl.DataFrame:"
    # The construction (and the import it needs) goes to the setup, with an inputs dict.
    assert c.setup_code.splitlines() == [
        "from datetime import date",
        'df = pl.DataFrame({"a": [1, 2, 3], "d": [date(2020, 1, 1)] * 3})',
        'inputs = {"df": df}',
    ]
    assert c.solution == (
        "def solve(df: pl.DataFrame) -> pl.DataFrame:\n"
        '    return df.with_columns(pl.col("a").cum_sum().alias("c"))'
    )
    assert rejects == {"doctest_skip": 1}  # write_csv has no displayed output -> no candidate
    res = run_solution(c.setup_code, c.solution)
    assert res.ok and res.frame.columns == ["a", "d", "c"]


def test_multi_step_user_guide_snippet_keeps_steps_in_solve():
    code = textwrap.dedent(
        """
        # --8<-- [start:setup]
        import polars as pl

        df = pl.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})
        threshold = 1
        # --8<-- [end:setup]

        # --8<-- [start:agg]
        big = df.filter(pl.col("v") > threshold)
        out = big.group_by("k", maintain_order=True).agg(pl.col("v").sum())
        print(out)
        # --8<-- [end:agg]
        """
    )
    prog = user_guide_program("user-guide/x.py", code)
    (c,), _ = program_candidates(prog)
    assert c.label == "agg"
    assert "import polars as pl" not in c.solution
    assert c.steps == [
        "threshold = 1",
        'big = df.filter(pl.col("v") > threshold)',
        'out = big.group_by("k", maintain_order=True).agg(pl.col("v").sum())',
    ]
    assert c.solution.endswith("    return out")
    res = run_solution(c.setup_code, c.solution)
    assert res.ok and res.frame.to_dict(as_series=False) == {"k": ["b", "a"], "v": [2, 3]}


def test_value_used_by_setup_and_solution_is_defined_in_both():
    prog = Program("docstring", "p", [
        Stmt("n = 2"),
        Stmt('df = pl.DataFrame({"a": list(range(n))})'),
        Stmt("df.head(n)", displays=True),
    ])  # fmt: skip
    (c,), _ = program_candidates(prog)
    assert "n = 2" in c.setup_code and c.steps == ["n = 2"]
    assert run_solution(c.setup_code, c.solution).ok


def test_unresolved_or_skipped_sources_are_rejected():
    io = Program("user_guide", "p", [Stmt('df = pl.read_csv("x.csv")'), Stmt("print(df.head())")], ["s", "s"])
    assert program_candidates(io)[1] == {"io": 1}
    missing = Program("docstring", "p", [Stmt("other.select(pl.col('a'))", displays=True)])
    assert program_candidates(missing)[1] == {"unresolved_name": 1}
    no_frame = Program("docstring", "p", [Stmt("pl.select(pl.lit(1))", displays=True)])
    assert program_candidates(no_frame)[1] == {"no_input_frame": 1}
    np_dep = Program("docstring", "p", [
        Stmt("import numpy as np"), Stmt('df = pl.DataFrame({"a": [1]})'),
        Stmt("df.select(np.log(pl.col('a')))", displays=True),
    ])  # fmt: skip
    assert program_candidates(np_dep)[1] == {"optional_dependency": 1}
    for code, reason in [("pl.Config.set_tbl_rows(3)", "config"), ("df.plot.line()", "plotting"),
                         ("pl.sql('SELECT 1')", "sql"), ("df.sample(2)", "random")]:
        assert skip_reason(code) == reason
    assert skip_reason("df.set_sorted('a')") is None  # a real API, not config


def test_construction_types_and_lazy_inputs():
    import ast

    def ct(src):
        return construction_type(ast.parse(src).body[0])

    assert ct("df = pl.DataFrame({'a': [1]})") == "DataFrame"
    assert ct("lf = pl.DataFrame({'a': [1]}).lazy()") == "LazyFrame"
    assert ct("lf = pl.LazyFrame({'a': [1]})") == "LazyFrame"
    assert ct("s = pl.Series([1])") == "Series"
    assert ct("df = pl.from_dicts([{'a': 1}])") == "DataFrame"
    assert ct("df2 = df.with_columns(b=1)") is None


def test_return_statement_parenthesises_multiline_and_collects():
    assert return_statement("lf.select('a')", collect=True) == "return lf.select('a').collect()"
    assert return_statement("a + b", collect=True) == "return (a + b).collect()"
    multi = return_statement("lf.filter(x)\n.select('a')", collect=True)
    assert multi == "return (\n    lf.filter(x)\n    .select('a')\n).collect()"


JOIN_ASOF_DOC = textwrap.dedent(
    '''
    Join on nearest key.

    Examples
    --------
    >>> from datetime import date
    >>> gdp = pl.LazyFrame({"date": [date(2016, 1, 1), date(2018, 1, 1)], "gdp": [4164, 4566]})
    >>> population = pl.LazyFrame(
    ...     {"date": [date(2016, 3, 1), date(2018, 8, 1)], "population": [82.19, 82.66]}
    ... ).sort("date")
    >>> population.join_asof(gdp, on="date", strategy="backward").collect()
    shape: (2, 3)
    ...
    '''
)


def test_frame_literal_in_a_step_is_lifted_into_the_inputs():
    (c,), _ = program_candidates(docstring_program("LazyFrame.join_asof", JOIN_ASOF_DOC))
    assert c.params == ["gdp", "population"] and c.param_types == ["LazyFrame", "LazyFrame"]
    assert c.signature == "def solve(gdp: pl.LazyFrame, population: pl.LazyFrame) -> pl.DataFrame:"
    assert 'population = pl.LazyFrame(\n    {"date"' in c.setup_code
    assert c.setup_code.splitlines()[0] == "from datetime import date"
    assert c.setup_code.endswith('inputs = {"gdp": gdp, "population": population}')
    assert 'population = population.sort("date")' in c.solution
    assert "pl.LazyFrame(" not in c.solution and "import" not in c.solution  # date import only needed by setup
    res = run_solution(c.setup_code, c.solution)
    assert res.ok and res.frame.columns == ["date", "population", "gdp"]


def test_inline_frame_literal_and_its_plain_values_are_lifted():
    prog = Program("docstring", "DataFrame.join", [
        Stmt('df = pl.DataFrame({"k": [1, 2], "a": ["x", "y"]})'),
        Stmt("keys = [1, 2]"),
        Stmt('df.join(pl.DataFrame({"k": keys, "b": ["é", "ü"]}), on="k")', displays=True),
    ])  # fmt: skip
    (c,), _ = program_candidates(prog)
    assert c.params == ["df", "other"] and c.param_types == ["DataFrame", "DataFrame"]
    assert c.setup_code.splitlines() == [
        'df = pl.DataFrame({"k": [1, 2], "a": ["x", "y"]})',
        "keys = [1, 2]",
        'other = pl.DataFrame({"k": keys, "b": ["é", "ü"]})',  # non-ASCII: byte offsets handled
        'inputs = {"df": df, "other": other}',
    ]
    assert c.solution.endswith('return df.join(other, on="k")') and "keys" not in c.solution
    assert run_solution(c.setup_code, c.solution).ok

    # A literal built from an input (or inside a lambda) cannot be lifted: the candidate is dropped.
    prog = Program("docstring", "DataFrame.join", [
        Stmt('df = pl.DataFrame({"k": [1, 2]})'),
        Stmt('df.join(pl.DataFrame({"k": df["k"]}), on="k")', displays=True),
    ])  # fmt: skip
    cands, rejects = program_candidates(prog)
    assert not cands and rejects == {"frame_literal_not_liftable": 1}


def test_displayed_literal_chain_gets_an_input():
    prog = Program("docstring", "DataFrame.with_row_index", [
        Stmt('pl.DataFrame({"a": [5, 6]}).with_row_index()', displays=True),
    ])  # fmt: skip
    (c,), _ = program_candidates(prog)
    assert c.params == ["other"] and c.solution.endswith("return other.with_row_index()")
