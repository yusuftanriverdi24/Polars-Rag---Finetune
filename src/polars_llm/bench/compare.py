"""Output comparison via ``polars.testing.assert_frame_equal``.

Honors the task's ``check`` flags:

- ``ignore_row_order``  -> ``check_row_order=False``
- ``ignore_column_order`` -> ``check_column_order=False``

Dtypes are checked (``check_dtypes=True``, the Polars default): returning an
``i64`` where the reference produced ``f64`` counts as a mismatch, which is the
behavior we want for "correct, up-to-date Polars code".
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


def compare_frames(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    check: CheckFlags,
) -> Comparison:
    """Return whether ``actual`` equals ``expected`` under the check flags."""

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
