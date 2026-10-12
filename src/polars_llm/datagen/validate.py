"""Validate task candidates in the sandbox under the pinned Polars 2.0.0.

Per candidate (all in subprocesses, like benchmark scoring):

1. **Probe** (``run_solution``): the solution runs once with a marker that
   prints the type of the returned value. It must run cleanly and raise **no
   warning at all** (deprecation or otherwise).
2. The final solution is fixed: a LazyFrame result gets ``.collect()``; only
   DataFrame / LazyFrame results are kept (Series, scalars, dicts... are not).
3. **Two runs** (``run_repeated``, n=2, fresh inputs each time) of the final
   solution must give the same DataFrame. When they agree only up to row
   order (e.g. ``group_by`` without ``maintain_order``), the candidate is
   kept with ``check.ignore_row_order = true``.

The first run's result also provides the output preview (columns, dtypes,
first rows) that the instruction writer sees.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import polars as pl
from polars.testing import assert_frame_equal

from ..bench.determinism import run_repeated
from ..bench.executor import run_solution
from .convert import Candidate, build_solution

TYPE_MARKER = "__DATAGEN_RESULT_TYPE__"
PREVIEW_ROWS = 5


@dataclass
class Validated:
    candidate: Candidate
    ok: bool
    reason: str = ""
    solution: str = ""
    ignore_row_order: bool = False
    output_columns: list[str] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)
    n_rows: int = 0
    preview: str = ""


def probe_solution(c: Candidate) -> str:
    """The candidate's solution, with the returned value's type printed."""

    marker_return = (
        f"__r = {c.result_expr}" if "\n" not in c.result_expr else "__r = (\n" + c.result_expr + "\n)"
    )
    steps = c.steps + [marker_return, f'print("{TYPE_MARKER}", type(__r).__name__)']
    return build_solution(c.signature, c.solution_imports, steps, "__r")


def _result_type(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith(TYPE_MARKER):
            return line.split()[-1]
    return None


def preview_text(df: pl.DataFrame, rows: int = PREVIEW_ROWS) -> str:
    with pl.Config(tbl_rows=rows, tbl_cols=12, tbl_width_chars=120, fmt_str_lengths=40):
        return str(df.head(rows))


def frames_agree(a: pl.DataFrame, b: pl.DataFrame) -> str | None:
    """'exact', 'row_order' or None."""

    try:
        assert_frame_equal(a, b)
        return "exact"
    except AssertionError:
        pass
    try:
        assert_frame_equal(a, b, check_row_order=False)
        return "row_order"
    except (AssertionError, Exception):  # unsortable dtypes raise too
        return None


def validate_candidate(c: Candidate, timeout_s: float = 15.0) -> Validated:
    probe = run_solution(c.setup_code, probe_solution(c), timeout_s=timeout_s)
    kind = _result_type(probe.stdout)
    if kind is not None and kind not in {"DataFrame", "LazyFrame"}:
        return Validated(c, False, f"result_type:{kind}")
    if not probe.ok:
        return Validated(c, False, "timeout" if probe.timed_out else f"error:{probe.error.type if probe.error else '?'}")
    if probe.warnings:
        dep = probe.deprecation_warnings
        return Validated(c, False, "deprecation_warning" if dep else "warning")
    if kind not in {"DataFrame", "LazyFrame"}:
        return Validated(c, False, f"result_type:{kind}")

    solution = build_solution(c.signature, c.solution_imports, c.steps, c.result_expr, collect=kind == "LazyFrame")
    rep = run_repeated(c.setup_code, solution, n=2, timeout_s=max(60.0, 4 * timeout_s))
    if rep.timed_out or any(f is None for f in rep.frames):
        return Validated(c, False, "final_run_failed", solution)
    agree = frames_agree(rep.frames[0], rep.frames[1])
    if agree is None:
        return Validated(c, False, "nondeterministic", solution)
    df = rep.frames[0]
    if df.width == 0:
        return Validated(c, False, "empty_result", solution)
    return Validated(
        c,
        True,
        "",
        solution,
        ignore_row_order=agree == "row_order",
        output_columns=df.columns,
        output_dtypes=[str(t) for t in df.dtypes],
        n_rows=df.height,
        preview=preview_text(df),
    )


def validate_all(candidates: list[Candidate], workers: int = 3, timeout_s: float = 15.0) -> list[Validated]:
    """Validate in parallel (each check is its own subprocess); input order kept."""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: validate_candidate(c, timeout_s), candidates))
