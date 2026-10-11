"""Determinism checks: does a solution's output order vary between runs?

Polars 2.0's lazy API defaults to the streaming engine, and ``group_by`` /
``join`` / ``unique`` / ``pivot`` / ``unpivot`` do not guarantee output order,
so the same code can produce the same rows in a different order (or, for a
``pivot`` over unordered data, the same columns in a different order) from one
run to the next. A benchmark task must not pass or fail on that luck.

- :func:`run_repeated` runs a solution ``n`` times in one sandbox subprocess.
- :func:`order_variation` classifies how a list of result frames differ.
- :func:`task_determinism_problems` reports variation in a task's reference
  output that the task's ``check`` flags do not tolerate (used by
  ``validate_tasks``).
- :func:`pivot_column_order_problem` is a static guard for the case a reference
  run cannot reveal: a *correct candidate* that pivots over unordered data
  (e.g. after ``group_by``) gets its pivoted columns in a varying order even
  when the reference's own order is stable (benchmark-v1.1, ``reshape_012``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from .schema import Task

DEFAULT_RUNS = 50
DEFAULT_REPEAT_TIMEOUT_S = 120.0


@dataclass
class RepeatResult:
    frames: list["pl.DataFrame | None"]
    errors: list["dict | None"]
    timed_out: bool = False


@dataclass
class OrderVariation:
    row_order: bool = False  # same rows, different order
    column_order: bool = False  # same columns, different order
    content: bool = False  # different values/shape/dtypes, or a run errored
    distinct_column_orders: list[list[str]] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return self.row_order or self.column_order or self.content


def run_repeated(
    setup_code: str,
    solve_code: str,
    n: int = DEFAULT_RUNS,
    timeout_s: float = DEFAULT_REPEAT_TIMEOUT_S,
) -> RepeatResult:
    """Run ``solve_code`` ``n`` times (fresh inputs each time) in one subprocess."""

    with tempfile.TemporaryDirectory(prefix="polars_repeat_") as tmp:
        tmp_path = Path(tmp)
        payload = tmp_path / "payload.json"
        result = tmp_path / "result.json"
        payload.write_text(
            json.dumps(
                {"setup_code": setup_code, "solve_code": solve_code, "n": n, "out_dir": str(tmp_path)}
            ),
            encoding="utf-8",
        )
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, "-m", "polars_llm.bench._repeat_child", str(payload), str(result)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            return RepeatResult(frames=[None] * n, errors=[{"type": "Timeout"}] * n, timed_out=True)
        if not result.exists():
            err = {"type": "SandboxError", "message": "repeat child produced no result"}
            return RepeatResult(frames=[None] * n, errors=[err] * n)

        runs = json.loads(result.read_text(encoding="utf-8"))["runs"]
        frames = [
            pl.read_ipc(tmp_path / f"{i}.arrow") if r["ok"] else None
            for i, r in enumerate(runs)
        ]
        return RepeatResult(frames=frames, errors=[r["error"] for r in runs])


def _equal(a: pl.DataFrame, b: pl.DataFrame, *, check_row_order: bool) -> bool:
    try:
        assert_frame_equal(a, b, check_row_order=check_row_order)
    except AssertionError:
        return False
    return True


def order_variation(frames: list["pl.DataFrame | None"]) -> OrderVariation:
    """Compare every frame with the first and classify the differences."""

    v = OrderVariation()
    if not frames:
        return v
    if any(f is None for f in frames):
        # Some runs errored: the outcome itself is unstable or broken.
        v.content = not all(f is None for f in frames)
        return v

    first = frames[0]
    orders: list[list[str]] = []
    for f in frames:
        if f.columns not in orders:
            orders.append(f.columns)
    v.distinct_column_orders = orders

    for f in frames[1:]:
        if f.columns != first.columns:
            if sorted(f.columns) != sorted(first.columns):
                v.content = True
                continue
            v.column_order = True
            f = f.select(first.columns)
        if _equal(f, first, check_row_order=True):
            continue
        if _equal(f, first, check_row_order=False):
            v.row_order = True
        else:
            v.content = True
    return v


_PIVOT_RE = re.compile(r"(?<!un)\.pivot\(")


def pivot_column_order_problem(task: Task) -> str | None:
    """A pivot task must tolerate column order unless its reference fixes it.

    The reference "fixes" the order when it ``select``s columns after its last
    ``pivot`` (then the instruction should name that order too).
    """

    ref = task.reference_solution
    pivots = list(_PIVOT_RE.finditer(ref))
    if not pivots or task.check.ignore_column_order:
        return None
    if ".select(" in ref[pivots[-1].end():]:
        return None
    return (
        "reference pivots but ignore_column_order is off: a correct solution's "
        "pivoted column order depends on input/group order, so set "
        "ignore_column_order (or select an explicit column order after the pivot)"
    )


def task_determinism_problems(task: Task, n: int = DEFAULT_RUNS) -> list[str]:
    """Order/content variation in the reference output not covered by ``task.check``."""

    result = run_repeated(task.setup_code, task.reference_solution, n=n)
    if result.timed_out:
        return [f"determinism check timed out over {n} runs"]
    v = order_variation(result.frames)
    problems = []
    if v.content:
        problems.append(f"reference output content varies across {n} runs")
    if v.row_order and not task.check.ignore_row_order:
        problems.append(
            f"reference row order varies across {n} runs but ignore_row_order is off "
            "(make the order deterministic or set ignore_row_order)"
        )
    if v.column_order and not task.check.ignore_column_order:
        problems.append(
            f"reference column order varies across {n} runs ({v.distinct_column_orders}) "
            "but ignore_column_order is off"
        )
    return problems
