"""Corpus v2: drop chunk families that add noise to retrieval (Arm B2).

Corpus v1 (``chunks.jsonl``, Arm B) is filtered, not re-chunked, so every kept
chunk is byte-identical to its v1 version. A chunk is dropped for the first
matching reason:

``sql_reference``
    Any page under ``reference/sql/`` (SQL clauses, SQL functions, the
    ``SQLContext`` API). They dominated Arm B's retrieval and twice led the
    model to answer with ``pl.sql`` code. The User Guide's SQL chapter is not
    reference material and is kept.
``old_upgrade_guide``
    Upgrade guides for versions before 1.0 (``releases/upgrade/0.*.md``). The
    1.0 guide (which documents e.g. ``melt`` -> ``unpivot``) and the 2.0 guide
    are kept.
``series_duplicate``
    A ``Series.<x>`` API page when an ``Expr.<x>`` page exists (same name,
    including namespaces such as ``str``/``dt``/``list``). Benchmark solutions
    are written with expressions; the Series twin repeats the same docstring
    and crowds the top-k.
``tiny_no_example``
    Fewer than ``TINY_TOKENS`` tokens and no example (no doctest ``>>>`` and
    no fenced code block), e.g. ``QueryResult.n_rows_total``.

The threshold and rules were fixed before any retrieval run, not tuned.

CLI::

    python -m polars_llm.rag.corpus        # data/rag/chunks.jsonl -> data/rag/chunks_v2.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .chunk import DEFAULT_OUT as DEFAULT_V1
from .chunk import Chunk, read_chunks, write_chunks

DEFAULT_V2 = Path("data/rag/chunks_v2.jsonl")
DEFAULT_STATS = Path("results/rag/corpus_v2_stats.json")

TINY_TOKENS = 100
DROP_REASONS = ("sql_reference", "old_upgrade_guide", "series_duplicate", "tiny_no_example")


def has_example(text: str) -> bool:
    return ">>>" in text or "```" in text


def is_sql_reference(c: Chunk) -> bool:
    return "/reference/sql/" in "/" + c.source


def is_old_upgrade_guide(c: Chunk) -> bool:
    name = c.source.replace("\\", "/").rsplit("/", 1)[-1]
    return "/releases/upgrade/" in "/" + c.source and name.startswith("0.")


def _primary_name(c: Chunk) -> str | None:
    return c.api_names[0] if c.kind == "api" and c.api_names else None


def is_series_duplicate(c: Chunk, expr_names: set[str]) -> bool:
    name = _primary_name(c)
    return bool(name and name.startswith("Series.") and "Expr." + name[len("Series."):] in expr_names)


def is_tiny_without_example(c: Chunk) -> bool:
    return c.n_tokens < TINY_TOKENS and not has_example(c.text)


def drop_reason(c: Chunk, expr_names: set[str]) -> str | None:
    if is_sql_reference(c):
        return "sql_reference"
    if is_old_upgrade_guide(c):
        return "old_upgrade_guide"
    if is_series_duplicate(c, expr_names):
        return "series_duplicate"
    if is_tiny_without_example(c):
        return "tiny_no_example"
    return None


def filter_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], dict]:
    """Apply the v2 filters. Returns (kept chunks in v1 order, report)."""

    expr_names = {n for c in chunks if (n := _primary_name(c)) and n.startswith("Expr.")}
    kept: list[Chunk] = []
    dropped: collections.Counter = collections.Counter()
    dropped_by_kind: collections.Counter = collections.Counter()
    for c in chunks:
        reason = drop_reason(c, expr_names)
        if reason is None:
            kept.append(c)
        else:
            dropped[reason] += 1
            dropped_by_kind[(reason, c.kind)] += 1

    def by_kind(cs: list[Chunk]) -> dict:
        return dict(sorted(collections.Counter(c.kind for c in cs).items()))

    names_before = {n for c in chunks for n in c.api_names}
    names_after = {n for c in kept for n in c.api_names}
    report = {
        "tiny_tokens": TINY_TOKENS,
        "before": {"n_chunks": len(chunks), "by_kind": by_kind(chunks)},
        "after": {"n_chunks": len(kept), "by_kind": by_kind(kept)},
        "dropped": {r: dropped.get(r, 0) for r in DROP_REASONS},
        "dropped_by_kind": {f"{r}/{k}": n for (r, k), n in sorted(dropped_by_kind.items())},
        "api_names_before": len(names_before),
        "api_names_after": len(names_after),
    }
    return kept, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter corpus v1 into corpus v2.")
    parser.add_argument("--chunks", default=str(DEFAULT_V1))
    parser.add_argument("--out", default=str(DEFAULT_V2))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    args = parser.parse_args(argv)

    src = Path(args.chunks)
    kept, report = filter_chunks(read_chunks(src))
    import hashlib

    report["v1_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    report["v2_sha256"] = write_chunks(kept, Path(args.out))
    stats = Path(args.stats)
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"corpus v2: {report['before']['n_chunks']} -> {report['after']['n_chunks']} chunks "
          f"(dropped {report['dropped']}) -> {args.out}")
    print(f"by kind: {report['before']['by_kind']} -> {report['after']['by_kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
