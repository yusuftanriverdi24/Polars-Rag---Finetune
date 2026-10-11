"""Classifier unit tests over synthetic execution outcomes (no subprocess)."""

from polars_llm.bench.classify import (
    DEPRECATED_API,
    HALLUCINATED_API,
    PANDAS_LEAK,
    PASS,
    RUNTIME_ERROR,
    SYNTAX_ERROR,
    TIMEOUT,
    WRONG_RESULT,
    classify,
    detect_pandas_leak,
)
from polars_llm.bench.compare import Comparison
from polars_llm.bench.executor import ExecError, ExecutionResult


def make_exec(
    *,
    ok=False,
    phase="call",
    error=None,
    warnings=None,
    timed_out=False,
):
    return ExecutionResult(
        ok=ok,
        phase=phase,
        frame=None,
        warnings=warnings or [],
        error=error,
        stdout="",
        stderr="",
        timed_out=timed_out,
        duration_s=0.01,
    )


def err(type_, message, mro=None):
    return ExecError(type=type_, message=message, mro=mro or [type_], module="builtins")


DEP_WARNING = {"category": "DeprecationWarning", "message": "old thing"}


# --- static pandas-leak detection ----------------------------------------

def test_detect_pandas_leak_patterns():
    assert detect_pandas_leak("import pandas as pd")
    assert detect_pandas_leak("df = pd.DataFrame()")
    assert detect_pandas_leak("df.loc[0]")
    assert detect_pandas_leak("df.iloc[0]")
    assert detect_pandas_leak("df.drop('a', inplace=True)")
    assert detect_pandas_leak("df.groupby('a').sum()")
    assert detect_pandas_leak("df.sort_values('a')")
    assert not detect_pandas_leak("df.group_by('a').agg(pl.col('b').sum())")
    assert not detect_pandas_leak("df.filter(pl.col('a') > 1)")


# --- success paths --------------------------------------------------------

def test_pass():
    c = classify(code="def solve(df): return df", exec_result=make_exec(ok=True, phase="ok"),
                 comparison=Comparison(equal=True))
    assert c.status == PASS
    assert not c.deprecated_api


def test_wrong_result():
    c = classify(code="def solve(df): return df", exec_result=make_exec(ok=True, phase="ok"),
                 comparison=Comparison(equal=False, detail="values differ"))
    assert c.status == WRONG_RESULT


def test_pass_but_deprecation_warning_is_flagged():
    # "recorded even when the result is correct" (PLAN.md 4.3)
    c = classify(
        code="def solve(df): return df",
        exec_result=make_exec(ok=True, phase="ok", warnings=[DEP_WARNING]),
        comparison=Comparison(equal=True),
    )
    assert c.status == PASS
    assert c.deprecated_api is True


# --- failure taxonomy -----------------------------------------------------

def test_syntax_error():
    c = classify(code="def solve(df) return df", exec_result=make_exec(
        error=err("SyntaxError", "invalid syntax", ["SyntaxError", "Exception"]), phase="compile"),
        comparison=None)
    assert c.status == SYNTAX_ERROR


def test_pandas_leak_takes_priority_over_attribute_error():
    # .groupby( on a polars frame raises AttributeError in 2.0, but the root
    # cause is the pandas idiom.
    c = classify(
        code="def solve(df): return df.groupby('a').sum()",
        exec_result=make_exec(error=err("AttributeError", "no attribute 'groupby'")),
        comparison=None,
    )
    assert c.status == PANDAS_LEAK


def test_deprecated_removed_api_via_mro():
    c = classify(
        code="def solve(df): return df.with_row_count('i')",
        exec_result=make_exec(error=err(
            "AttributeRemovedError",
            "`with_row_count` was removed in version 2.0; use `with_row_index`",
            ["AttributeRemovedError", "AttributeError", "Exception"],
        )),
        comparison=None,
    )
    assert c.status == DEPRECATED_API
    assert c.deprecated_api is True


def test_deprecated_via_message_even_if_plain_attribute_error():
    c = classify(
        code="def solve(df): return df.foo()",
        exec_result=make_exec(error=err("AttributeError", "foo is deprecated, use bar")),
        comparison=None,
    )
    assert c.status == DEPRECATED_API


def test_hallucinated_attribute_error():
    c = classify(
        code="def solve(df): return df.nonexistent_method()",
        exec_result=make_exec(error=err("AttributeError", "'DataFrame' object has no attribute 'nonexistent_method'")),
        comparison=None,
    )
    assert c.status == HALLUCINATED_API


def test_hallucinated_name_error():
    c = classify(
        code="def solve(df): return undefined_fn(df)",
        exec_result=make_exec(error=err("NameError", "name 'undefined_fn' is not defined")),
        comparison=None,
    )
    assert c.status == HALLUCINATED_API


def test_hallucinated_unexpected_keyword():
    c = classify(
        code="def solve(df): return df.sort('a', bogus=True)",
        exec_result=make_exec(error=err("TypeError", "sort() got an unexpected keyword argument 'bogus'")),
        comparison=None,
    )
    assert c.status == HALLUCINATED_API


def test_runtime_error():
    c = classify(
        code="def solve(df): raise ValueError('boom')",
        exec_result=make_exec(error=err("ValueError", "boom")),
        comparison=None,
    )
    assert c.status == RUNTIME_ERROR


def test_timeout():
    c = classify(
        code="def solve(df):\n    while True: pass",
        exec_result=make_exec(timed_out=True, error=err("Timeout", "exceeded")),
        comparison=None,
    )
    assert c.status == TIMEOUT


# --- error messages that point to a replacement API -> deprecated_api --------


def test_should_be_replaced_message_is_deprecated_api():
    c = classify(
        code="def solve(a, b): return a.join(b, on='k', how='outer')",
        exec_result=make_exec(
            error=err("ValueError", "use of `how='outer'` should be replaced with `how='full'`.")
        ),
        comparison=None,
    )
    assert c.status == DEPRECATED_API
    assert c.deprecated_api


def test_please_use_instead_message_is_deprecated_api():
    msg = (
        "`pl.count()` takes at least one argument. If you want to count the "
        "number of rows, please use `pl.len()` instead."
    )
    c = classify(
        code="def solve(df): return df.group_by('s').agg(pl.count())",
        exec_result=make_exec(error=err("TypeError", msg)),
        comparison=None,
    )
    assert c.status == DEPRECATED_API


def test_unrelated_runtime_errors_stay_runtime_error():
    for type_, msg in [
        ("ColumnNotFoundError", 'unable to find column "id"; valid columns: ["order_id"]'),
        (
            "InvalidOperationError",
            "arithmetic on dtypes f64 and str is not allowed; try an explicit cast first",
        ),
        (
            "DuplicateError",
            "column with name 'count_right' already exists. You may want to try "
            "using the `suffix` parameter to specify a suffix different to the default",
        ),
    ]:
        c = classify(code="def solve(df): ...", exec_result=make_exec(error=err(type_, msg)), comparison=None)
        assert c.status == RUNTIME_ERROR, msg


def test_did_you_mean_hint_is_not_a_replacement_api():
    # A spelling hint for a hallucinated name is not a deprecation.
    c = classify(
        code="def solve(a, b): return a.join(b, on='k', suffixes=('_l', '_r'))",
        exec_result=make_exec(
            error=err(
                "TypeError",
                "DataFrame.join() got an unexpected keyword argument 'suffixes'. Did you mean 'suffix'?",
            )
        ),
        comparison=None,
    )
    assert c.status == HALLUCINATED_API


# --- pandas_style_name secondary tag -----------------------------------------


def _hallucinated(code, type_, msg):
    return classify(code=code, exec_result=make_exec(error=err(type_, msg)), comparison=None)


def test_pandas_style_name_tags_pandas_accessor_methods():
    cases = [
        ("strip", "'ExprStringNameSpace' object has no attribute 'strip'. Did you mean: 'strip_chars'?", "str.strip"),
        ("pad", "'ExprStringNameSpace' object has no attribute 'pad'. Did you mean: 'pad_end'?", "str.pad"),
        ("count", "'ExprStringNameSpace' object has no attribute 'count'. Did you mean: 'contains'?", "str.count"),
        ("len", "'ExprStringNameSpace' object has no attribute 'len'", "str.len"),
        ("day_name", "'ExprDateTimeNameSpace' object has no attribute 'day_name'", "dt.day_name"),
        ("days", "'ExprDateTimeNameSpace' object has no attribute 'days'. Did you mean: 'day'?", "dt.days"),
        ("rename", "'Expr' object has no attribute 'rename'", "rename"),
    ]
    for _, msg, expected in cases:
        c = _hallucinated("def solve(df): ...", "AttributeError", msg)
        assert c.status == HALLUCINATED_API  # primary status unchanged
        assert c.pandas_style_name == expected, msg


def test_pandas_style_name_tags_pandas_keyword():
    c = _hallucinated(
        "def solve(a, b): return a.join(b, on='k', suffixes=('_a', '_b'))",
        "TypeError",
        "DataFrame.join() got an unexpected keyword argument 'suffixes'. Did you mean 'suffix'?",
    )
    assert c.status == HALLUCINATED_API
    assert c.pandas_style_name == "suffixes="


def test_pandas_style_name_tags_property_style_access():
    c = _hallucinated(
        'def solve(df): return df.with_columns(pl.col("d").dt.year.alias("year"))',
        "AttributeError",
        "'function' object has no attribute 'alias'",
    )
    assert c.status == HALLUCINATED_API
    assert c.pandas_style_name == "dt.year"


def test_pandas_style_name_ignores_called_methods_and_non_pandas_names():
    # `.dt.year()` is called correctly; the failure is elsewhere.
    c = _hallucinated(
        'def solve(df): return df.with_columns(pl.list.first("p").alias("x"), pl.col("d").dt.year())',
        "AttributeError",
        "'function' object has no attribute 'first'",
    )
    assert c.pandas_style_name is None
    # Hallucinated, but not a pandas name.
    for msg in [
        "'ExprStringNameSpace' object has no attribute 'to_upper'. Did you mean: 'to_uppercase'?",
        "'ExprDateTimeNameSpace' object has no attribute 'add_days'",
        "'Expr' object has no attribute 'contains'",
    ]:
        assert _hallucinated("def solve(df): ...", "AttributeError", msg).pandas_style_name is None
    c = _hallucinated(
        "def solve(df): ...", "TypeError", "Expr.rank() got an unexpected keyword argument 'dense'"
    )
    assert c.pandas_style_name is None


def test_pandas_style_name_only_on_hallucinated_api():
    # Same pandas-ish name but a different primary status -> no tag.
    c = classify(
        code='def solve(df): return df.filter(df["d"].dt.year == 2023)',
        exec_result=make_exec(
            error=err("TypeError", "'==' not supported between instances of 'method' and 'int'")
        ),
        comparison=None,
    )
    assert c.status == RUNTIME_ERROR
    assert c.pandas_style_name is None


def test_pandas_name_lists_are_absent_from_polars():
    # Guard: a name that exists in Polars 2.0 must not be tagged as pandas-only.
    import polars as pl

    from polars_llm.bench.classify import (
        PANDAS_DT_NAMES,
        PANDAS_OBJECT_NAMES,
        PANDAS_PROPERTY_NAMES,
        PANDAS_STR_NAMES,
    )

    e = pl.col("x")
    assert not [n for n in PANDAS_STR_NAMES if hasattr(e.str, n)]
    assert not [n for n in PANDAS_DT_NAMES if hasattr(e.dt, n)]
    assert not [n for n in PANDAS_OBJECT_NAMES if hasattr(e, n)]
    # ...while the property-style names are real Polars methods.
    assert all(callable(getattr(e.dt, n)) for n in PANDAS_PROPERTY_NAMES)
