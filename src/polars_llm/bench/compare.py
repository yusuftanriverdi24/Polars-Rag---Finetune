"""Output comparison via ``polars.testing.assert_frame_equal``.

Honors the task's ``check`` flags:

- ``ignore_row_order``     -> ``check_row_order=False``
- ``ignore_column_order``  -> ``check_column_order=False``
- ``ignore_numeric_width`` (default on) -> before comparing, both frames are
  cast so every integer type becomes ``Int128`` and every float type becomes
  ``Float64`` (recursively inside ``List`` / ``Array`` / ``Struct``). ``Int128``
  holds every ``Int*``/``UInt*`` value up to 64 bits exactly, and
  ``Float32 -> Float64`` is exact, so the cast never changes a value.

Dtypes are otherwise checked (``check_dtypes=True``, the Polars default). The
numeric-width tolerance exists because the width/signedness of an integer
result is usually an incidental implementation detail (e.g. a boolean ``.sum()``
gives ``UInt32``, a ``when/then(1)`` count gives ``Int32``) rather than a sign
of wrong code. Genuinely different types stay strict: returning an ``i64``
where the reference produced ``f64``, or a string where it produced an
integer, still counts as a mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from polars.testing import assert_frame_equal

from .schema import CheckFlags


@dataclass
class Comparison:
    equal: bool
    detail: str | None = None


def normalize_numeric_dtype(dtype: pl.DataType) -> pl.DataType:
    """Map a dtype to its width-agnostic form (ints -> Int128, floats -> Float64)."""

    if dtype.is_integer():
        return pl.Int128
    if dtype.is_float():
        return pl.Float64
    if isinstance(dtype, pl.List):
        return pl.List(normalize_numeric_dtype(dtype.inner))
    if isinstance(dtype, pl.Array):
        return pl.Array(normalize_numeric_dtype(dtype.inner), dtype.size)
    if isinstance(dtype, pl.Struct):
        return pl.Struct(
            [pl.Field(f.name, normalize_numeric_dtype(f.dtype)) for f in dtype.fields]
        )
    return dtype


def _normalize_numeric_widths(df: pl.DataFrame) -> pl.DataFrame:
    casts = {}
    for name, dtype in df.schema.items():
        target = normalize_numeric_dtype(dtype)
        if target != dtype:
            casts[name] = target
    return df.cast(casts) if casts else df


def compare_frames(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    check: CheckFlags,
) -> Comparison:
    """Return whether ``actual`` equals ``expected`` under the check flags."""

    if check.ignore_numeric_width:
        actual = _normalize_numeric_widths(actual)
        expected = _normalize_numeric_widths(expected)

    try:
        assert_frame_equal(
            actual,
            expected,
            check_row_order=not check.ignore_row_order,
            check_column_order=not check.ignore_column_order,
        )
    except AssertionError as exc:
        return Comparison(equal=False, detail=str(exc))
    return Comparison(equal=True, detail=None)
