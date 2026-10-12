"""Corpus v2 filters (Arm B2): which chunks are dropped and why."""

from polars_llm.rag.chunk import Chunk, read_chunks, write_chunks
from polars_llm.rag.corpus import TINY_TOKENS, drop_reason, filter_chunks, has_example, main

LONG = "word " * 200


def _api(name, text=LONG, n_tokens=200, source=None, part=0):
    return Chunk(f"api:{name}#p{part}", "api", source or "py-polars/docs/source/reference/x.rst",
                 f"polars.{name}", "Section", [name], text, n_tokens)


def _guide(source, text=LONG, n_tokens=300, idx=0):
    return Chunk(f"guide:{source}#{idx:03d}", "guide", source, "Title", "Sec", [], text, n_tokens)


def test_sql_reference_pages_are_dropped_but_user_guide_sql_is_kept():
    clauses = _guide("py-polars/docs/source/reference/sql/clauses.rst")
    sql_api = _api("SQLContext.execute", source="py-polars/docs/source/reference/sql/python_api.rst")
    ug_sql = _guide("docs/source/user-guide/sql/intro.md")
    assert drop_reason(clauses, set()) == "sql_reference"
    assert drop_reason(sql_api, set()) == "sql_reference"
    assert drop_reason(ug_sql, set()) is None


def test_only_pre_1_0_upgrade_guides_are_dropped():
    assert drop_reason(_guide("docs/source/releases/upgrade/0.19.md"), set()) == "old_upgrade_guide"
    assert drop_reason(_guide("docs/source/releases/upgrade/0.20.md"), set()) == "old_upgrade_guide"
    assert drop_reason(_guide("docs/source/releases/upgrade/1.md"), set()) is None  # melt -> unpivot lives here
    assert drop_reason(_guide("docs/source/releases/upgrade/2.md"), set()) is None


def test_series_page_dropped_only_when_expr_twin_exists():
    expr = {"Expr.str.to_uppercase", "Expr.sum"}
    assert drop_reason(_api("Series.str.to_uppercase"), expr) == "series_duplicate"
    assert drop_reason(_api("Series.sum"), expr) == "series_duplicate"
    assert drop_reason(_api("Series.to_frame"), expr) is None  # Series-only API
    assert drop_reason(_api("Expr.sum"), expr) is None
    assert drop_reason(_api("DataFrame.sum"), expr) is None


def test_tiny_chunks_dropped_only_without_examples():
    stub = _api("QueryResult.n_rows_total", text="Total rows that are outputted by the result.", n_tokens=22)
    doctest = _api("Expr.abs", text="Absolute value.\n\n>>> df.select(pl.col('a').abs())", n_tokens=30)
    fenced = _guide("docs/source/user-guide/x.md", text="Short.\n```python\ndf.head()\n```", n_tokens=20)
    at_threshold = _api("Expr.x", text="no example", n_tokens=TINY_TOKENS)
    assert drop_reason(stub, set()) == "tiny_no_example"
    assert drop_reason(doctest, set()) is None
    assert drop_reason(fenced, set()) is None
    assert drop_reason(at_threshold, set()) is None
    assert has_example(">>> 1") and has_example("```\nx\n```") and not has_example("plain")


def test_filter_chunks_keeps_order_and_reports_counts():
    chunks = [
        _api("Expr.sum"),
        _api("Series.sum"),
        _guide("py-polars/docs/source/reference/sql/clauses.rst"),
        _guide("docs/source/releases/upgrade/0.19.md"),
        _api("QueryResult.n_rows_total", text="Total rows.", n_tokens=5),
        _guide("docs/source/user-guide/joins.md"),
    ]
    kept, report = filter_chunks(chunks)
    assert [c.id for c in kept] == [chunks[0].id, chunks[5].id]
    assert report["before"] == {"n_chunks": 6, "by_kind": {"api": 3, "guide": 3}}
    assert report["after"] == {"n_chunks": 2, "by_kind": {"api": 1, "guide": 1}}
    assert report["dropped"] == {
        "sql_reference": 1, "old_upgrade_guide": 1, "series_duplicate": 1, "tiny_no_example": 1,
    }  # fmt: skip


def test_cli_writes_identical_kept_chunks(tmp_path):
    chunks = [_api("Expr.sum"), _api("Series.sum"), _guide("docs/source/user-guide/joins.md")]
    src, out, stats = tmp_path / "v1.jsonl", tmp_path / "v2.jsonl", tmp_path / "stats.json"
    write_chunks(chunks, src)
    assert main(["--chunks", str(src), "--out", str(out), "--stats", str(stats)]) == 0
    v2 = read_chunks(out)
    assert [c.to_dict() for c in v2] == [chunks[0].to_dict(), chunks[2].to_dict()]
    assert stats.exists()
