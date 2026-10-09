"""Merge per-category task sources into ``benchmark/tasks.jsonl``.

Each category has a source module ``benchmark/tasks_src/<category>.py`` exposing
a module-level ``TASKS: list[dict]``. Tasks are written with real (triple-
quoted) code strings so they stay readable and editable; this script merges
them, validates every one against the schema, enforces unique ids and that each
task's ``category`` matches its source file, and emits one JSON object per line.

Regenerate with::

    python benchmark/build_tasks.py

All reference solutions must then pass::

    python -m polars_llm.bench.validate_tasks

Source dicts may carry an extra ``touches_2_0_change`` bool (authoring metadata
used by ``benchmark/REVIEW.md``); it is ignored by the task schema and not
written to ``tasks.jsonl``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

# Fixed order so tasks.jsonl is stable across regenerations.
CATEGORY_ORDER = [
    "select_filter",
    "group_by",
    "join",
    "window",
    "lazy",
    "string",
    "datetime",
    "list_struct",
    "reshape",
]

HERE = Path(__file__).parent
SRC_DIR = HERE / "tasks_src"


def _load_module_tasks(path: Path) -> list[dict]:
    spec = importlib.util.spec_from_file_location(f"tasks_src.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    tasks = getattr(module, "TASKS", None)
    if not isinstance(tasks, list):
        raise ValueError(f"{path}: must define a list named TASKS")
    return tasks


def collect_source_tasks() -> list[dict]:
    """Return every source task dict (incl. authoring metadata), in order."""

    from polars_llm.bench.schema import Task  # local import; package must be installed

    out: list[dict] = []
    seen: set[str] = set()
    for category in CATEGORY_ORDER:
        path = SRC_DIR / f"{category}.py"
        if not path.exists():
            continue
        for raw in _load_module_tasks(path):
            # Validate against the schema (raises on malformed tasks).
            task = Task.from_dict(raw)
            if task.category != category:
                raise ValueError(
                    f"{path.name}: task {task.id!r} has category {task.category!r}, "
                    f"expected {category!r}"
                )
            if task.id in seen:
                raise ValueError(f"duplicate task id {task.id!r}")
            seen.add(task.id)
            merged = dict(raw)
            # Normalize code blocks authored as triple-quoted literals.
            for key in ("setup_code", "reference_solution"):
                merged[key] = merged[key].strip("\n")
            merged["touches_2_0_change"] = bool(raw.get("touches_2_0_change", False))
            out.append(merged)
    return out


def build(out_path: Path) -> list[dict]:
    from polars_llm.bench.schema import Task

    source = collect_source_tasks()
    lines = []
    for raw in source:
        # Persist only canonical schema fields (drops touches_2_0_change etc.).
        lines.append(json.dumps(Task.from_dict(raw).to_dict(), ensure_ascii=False))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source


if __name__ == "__main__":
    src = build(HERE / "tasks.jsonl")
    from collections import Counter

    by_cat = Counter(t["category"] for t in src)
    by_diff = Counter(t["difficulty"] for t in src)
    touches = sum(1 for t in src if t["touches_2_0_change"])
    print(f"Wrote {len(src)} tasks to {HERE / 'tasks.jsonl'}")
    print(f"  by difficulty: {dict(by_diff)}")
    print(f"  by category:   {dict(by_cat)}")
    print(f"  touches 2.0 change: {touches} ({touches / max(len(src),1):.0%})")
