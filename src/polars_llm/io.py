"""JSONL reading and writing, shared by every pipeline step.

Rows are written with ``ensure_ascii=False``, which leaves U+0085, U+2028 and
U+2029 unescaped inside strings. ``str.splitlines()`` (and so
``read_text().splitlines()``) splits on those too, cutting such a row into
unparseable pieces. Lines are therefore split on ``"\\n"`` only: the file is
opened with ``newline="\\n"``, and JSON escapes every other control character.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def jsonl_lines(path: str | Path) -> Iterator[tuple[int, str]]:
    """``(line number, text)`` of every non-blank line, split on ``"\\n"`` only."""

    with Path(path).open(encoding="utf-8", newline="\n") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.strip():
                yield lineno, line


def read_jsonl(path: str | Path) -> list[dict]:
    """Every row of a JSONL file, in order."""

    return [json.loads(line) for _, line in jsonl_lines(path)]


def jsonl_line(row, *, sort_keys: bool = False) -> str:
    """One JSONL line, ``ensure_ascii=False`` as every existing file (and its hash)."""

    return json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n"


def write_jsonl(path: str | Path, rows: Iterable, *, sort_keys: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(jsonl_line(r, sort_keys=sort_keys) for r in rows)
