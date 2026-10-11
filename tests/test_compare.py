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


# --- ignore_numeric_width (default on) ---------------------------------------


def _ints(dtype):
    return pl.DataFrame({"x": pl.Series([1, 2], dtype=dtype)})


def test_numeric_width_flag_is_on_by_default():
    assert CheckFlags().ignore_numeric_width is True
    assert CheckFlags.from_dict({}).ignore_numeric_width is True
    assert CheckFlags.from_dict({"ignore_numeric_width": False}).ignore_numeric_width is False


def test_integer_width_and_sign_differences_pass_by_default():
    for a, b in [(pl.Int32, pl.UInt32), (pl.Int32, pl.Int64), (pl.UInt32, pl.Int64), (pl.Int8, pl.UInt64)]:
        assert compare_frames(_ints(a), _ints(b), CheckFlags()).equal, (a, b)


def test_float_width_differences_pass_by_default():
    a = pl.DataFrame({"x": pl.Series([0.1, 2.5], dtype=pl.Float32)})
    b = pl.DataFrame({"x": pl.Series([0.1, 2.5], dtype=pl.Float64)})
    assert compare_frames(a, b, CheckFlags()).equal


def test_numeric_width_differences_fail_when_flag_off():
    strict = CheckFlags(ignore_numeric_width=False)
    assert not compare_frames(_ints(pl.Int32), _ints(pl.UInt32), strict).equal
    a = pl.DataFrame({"x": pl.Series([0.5], dtype=pl.Float32)})
    b = pl.DataFrame({"x": pl.Series([0.5], dtype=pl.Float64)})
    assert not compare_frames(a, b, strict).equal


def test_numeric_width_tolerance_reaches_nested_types():
    a = pl.DataFrame(
        {
            "l": pl.Series([[1, 2]], dtype=pl.List(pl.UInt32)),
            "s": pl.Series([{"a": 1, "b": 0.5}], dtype=pl.Struct({"a": pl.Int32, "b": pl.Float32})),
        }
    )
    b = pl.DataFrame({"l": [[1, 2]], "s": [{"a": 1, "b": 0.5}]})  # i64 / f64
    assert compare_frames(a, b, CheckFlags()).equal


def test_numeric_width_tolerance_still_checks_values():
    a = pl.DataFrame({"x": pl.Series([1, 2], dtype=pl.UInt32)})
    b = pl.DataFrame({"x": [1, 3]})
    assert not compare_frames(a, b, CheckFlags()).equal


def test_large_unsigned_values_survive_normalization():
    big = 2**64 - 1
    a = pl.DataFrame({"x": pl.Series([big], dtype=pl.UInt64)})
    assert compare_frames(a, a.clone(), CheckFlags()).equal
    assert not compare_frames(a, pl.DataFrame({"x": [1]}), CheckFlags()).equal


def test_genuinely_different_types_stay_strict():
    ints = pl.DataFrame({"x": pl.Series([1, 2], dtype=pl.Int32)})
    floats = pl.DataFrame({"x": pl.Series([1.0, 2.0], dtype=pl.Float32)})
    strings = pl.DataFrame({"x": ["1", "2"]})
    assert not compare_frames(ints, floats, CheckFlags()).equal  # int vs float
    assert not compare_frames(ints, strings, CheckFlags()).equal  # int vs string
    assert not compare_frames(
        pl.DataFrame({"l": [[1]]}), pl.DataFrame({"l": [[1.0]]}), CheckFlags()
    ).equal  # list[int] vs list[float]
