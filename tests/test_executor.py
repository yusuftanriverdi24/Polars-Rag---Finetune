"""Executor integration tests: these spawn real subprocesses and use Polars."""

import polars as pl

from polars_llm.bench.executor import compute_expected, run_solution

SETUP = 'inputs = {"df": pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})}'


def test_correct_solution_returns_frame():
    res = run_solution(SETUP, "def solve(df):\n    return df.select('a')")
    assert res.ok
    assert res.phase == "ok"
    assert res.frame is not None
    assert res.frame.equals(pl.DataFrame({"a": [1, 2, 3]}))
    assert res.error is None


def test_lazyframe_result_is_collected():
    res = run_solution(SETUP, "def solve(df):\n    return df.lazy().select('a')")
    assert res.ok
    assert res.frame.equals(pl.DataFrame({"a": [1, 2, 3]}))


def test_syntax_error_reported_at_compile():
    res = run_solution(SETUP, "def solve(df) return df")
    assert not res.ok
    assert res.phase == "compile"
    assert res.error.type == "SyntaxError"


def test_removed_api_raises_attribute_removed_error():
    # with_row_count was removed in Polars 2.0.
    res = run_solution(SETUP, "def solve(df):\n    return df.with_row_count('i')")
    assert not res.ok
    assert res.error.type == "AttributeRemovedError"
    assert "AttributeError" in res.error.mro


def test_deprecation_warning_is_captured_on_success():
    code = (
        "import warnings\n"
        "def solve(df):\n"
        "    warnings.warn('legacy call', DeprecationWarning)\n"
        "    return df\n"
    )
    res = run_solution(SETUP, code)
    assert res.ok
    assert res.deprecation_warnings
    assert res.deprecation_warnings[0]["category"] == "DeprecationWarning"


def test_stdout_is_captured():
    code = "def solve(df):\n    print('hello from sandbox')\n    return df"
    res = run_solution(SETUP, code)
    assert res.ok
    assert "hello from sandbox" in res.stdout


def test_runtime_exception_is_reported():
    res = run_solution(SETUP, "def solve(df):\n    raise ValueError('boom')")
    assert not res.ok
    assert res.error.type == "ValueError"
    assert "boom" in res.error.message


def test_missing_solve_function():
    res = run_solution(SETUP, "x = 1")
    assert not res.ok
    assert res.error.type == "NoSolveFunction"


def test_non_dataframe_return_is_rejected():
    res = run_solution(SETUP, "def solve(df):\n    return 42")
    assert not res.ok
    assert res.error.type == "TypeError"


def test_inputs_must_match_solve_parameters():
    # inputs key is "df" but solve names its parameter "data".
    res = run_solution(SETUP, "def solve(data):\n    return data")
    assert not res.ok
    assert res.error.type == "ValueError"
    assert "do not match inputs keys" in res.error.message


def test_var_keyword_solve_is_accepted():
    res = run_solution(SETUP, "def solve(**kw):\n    return kw['df']")
    assert res.ok
    assert res.frame.equals(pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}))


def test_timeout():
    res = run_solution(SETUP, "def solve(df):\n    while True:\n        pass", timeout_s=3)
    assert res.timed_out
    assert not res.ok
    assert res.error.type == "Timeout"


def test_compute_expected_matches_run_solution():
    res = compute_expected(SETUP, "def solve(df):\n    return df.select('b')")
    assert res.ok
    assert res.frame.equals(pl.DataFrame({"b": [10, 20, 30]}))
