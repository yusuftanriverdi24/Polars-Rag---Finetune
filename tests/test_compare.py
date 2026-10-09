import polars as pl

from polars_llm.bench.compare import compare_frames
from polars_llm.bench.schema import CheckFlags


def test_equal_frames():
    a = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    assert compare_frames(a, a.clone(), CheckFlags()).equal


def test_different_values():
    a = pl.DataFrame({"x": [1, 2]})
    b = pl.DataFrame({"x": [1, 3]})
    assert not compare_frames(a, b, CheckFlags()).equal


def test_column_order_strict_by_default():
    a = pl.DataFrame({"x": [1], "y": [2]})
    b = pl.DataFrame({"y": [2], "x": [1]})
    assert not compare_frames(a, b, CheckFlags()).equal
    assert compare_frames(a, b, CheckFlags(ignore_column_order=True)).equal


def test_row_order_strict_by_default():
    a = pl.DataFrame({"x": [1, 2, 3]})
    b = pl.DataFrame({"x": [3, 1, 2]})
    assert not compare_frames(a, b, CheckFlags()).equal
    assert compare_frames(a, b, CheckFlags(ignore_row_order=True)).equal


def test_dtype_mismatch_fails():
    a = pl.DataFrame({"x": [1, 2]})  # i64
    b = pl.DataFrame({"x": [1.0, 2.0]})  # f64
    result = compare_frames(a, b, CheckFlags())
    assert not result.equal
    assert result.detail  # carries the assert_frame_equal message
