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
