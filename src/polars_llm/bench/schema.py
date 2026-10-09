"""Benchmark task schema and JSONL loader (PLAN.md Section 4.1).

A benchmark task is one JSON object per line in ``benchmark/tasks.jsonl``::

    {
      "id": "groupby_003",
      "difficulty": "medium",
      "category": "group_by",
      "instruction": "...",
      "setup_code": "inputs = {\\"df\\": pl.DataFrame({...})}",
      "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
      "reference_solution": "def solve(df): ...",
      "check": {"ignore_row_order": false, "ignore_column_order": false}
    }

Conventions used by the executor:

- ``setup_code`` must define a dict named ``inputs`` mapping the parameter
  names of ``solve`` to their argument values. The sandbox calls
  ``solve(**inputs)``.
- ``reference_solution`` (and model output) must define a callable ``solve``.
- The expected output is always *computed* by running ``reference_solution``
  under the pinned Polars version; it is never stored by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
VALID_CATEGORIES = frozenset(
    {
        "select_filter",
        "group_by",
        "join",
        "window",
        "lazy",
        "string",
        "datetime",
        "list_struct",
        "reshape",
    }
)


@dataclass(frozen=True)
class CheckFlags:
    """Comparison tolerances honored by the output comparator."""

    ignore_row_order: bool = False
    ignore_column_order: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "CheckFlags":
        d = d or {}
        unknown = set(d) - {"ignore_row_order", "ignore_column_order"}
        if unknown:
            raise ValueError(f"unknown check flags: {sorted(unknown)}")
        return cls(
            ignore_row_order=bool(d.get("ignore_row_order", False)),
            ignore_column_order=bool(d.get("ignore_column_order", False)),
        )

    def to_dict(self) -> dict:
        return {
            "ignore_row_order": self.ignore_row_order,
            "ignore_column_order": self.ignore_column_order,
        }


@dataclass(frozen=True)
class Task:
    """A single benchmark task."""

    id: str
    difficulty: str
    category: str
    instruction: str
    setup_code: str
    signature: str
    reference_solution: str
    check: CheckFlags = field(default_factory=CheckFlags)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        required = {
            "id",
            "difficulty",
            "category",
            "instruction",
            "setup_code",
            "signature",
            "reference_solution",
        }
        missing = required - set(d)
        if missing:
            raise ValueError(f"task missing required fields: {sorted(missing)}")
        task = cls(
            id=str(d["id"]),
            difficulty=str(d["difficulty"]),
            category=str(d["category"]),
            instruction=str(d["instruction"]),
            setup_code=str(d["setup_code"]),
            signature=str(d["signature"]),
            reference_solution=str(d["reference_solution"]),
            check=CheckFlags.from_dict(d.get("check")),
        )
        task.validate()
        return task

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "difficulty": self.difficulty,
            "category": self.category,
            "instruction": self.instruction,
            "setup_code": self.setup_code,
            "signature": self.signature,
            "reference_solution": self.reference_solution,
            "check": self.check.to_dict(),
        }

    def validate(self) -> None:
        if not self.id:
            raise ValueError("task id must be non-empty")
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"task {self.id!r}: invalid difficulty {self.difficulty!r} "
                f"(expected one of {sorted(VALID_DIFFICULTIES)})"
            )
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"task {self.id!r}: invalid category {self.category!r} "
                f"(expected one of {sorted(VALID_CATEGORIES)})"
            )
        for field_name in ("instruction", "setup_code", "signature", "reference_solution"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"task {self.id!r}: {field_name} must be non-empty")


def load_tasks(path: str | Path) -> list[Task]:
    """Load and validate all tasks from a JSONL file.

    Blank lines and lines starting with ``#`` are ignored. Raises on malformed
    JSON, schema violations, or duplicate task ids.
    """

    path = Path(path)
    tasks: list[Task] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                task = Task.from_dict(obj)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            if task.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate task id {task.id!r}")
            seen.add(task.id)
            tasks.append(task)
    return tasks


def tasks_by_id(tasks: list[Task]) -> dict[str, Task]:
    return {t.id: t for t in tasks}
