"""Chunking: API pages, guide sections, code-block integrity, metadata.

Uses a whitespace token counter (no tokenizer download) and tiny synthetic
corpora under tmp_path; API chunks resolve real objects from polars 2.0.0.
"""

from pathlib import Path

import pytest

from polars_llm.rag import chunk as C
from polars_llm.rag.chunk import (
    build_api_chunks,
    build_guide_chunks,
    canonical_api_name,
    markdown_blocks,
    pack_guide_blocks,
    parse_rst_page,
    split_api_doc,
    whitespace_token_counter as count,
)

KNOWN = {
    "Expr.str.to_uppercase",
    "Series.str.to_uppercase",
    "Expr.alias",
    "DataFrame.join",
    "GroupBy.agg",
    "col",
    "selectors.numeric",
}


def test_canonical_api_name():
    assert canonical_api_name("polars.Expr.alias", KNOWN) == "Expr.alias"
    assert canonical_api_name("pl.col", KNOWN) == "col"
    assert canonical_api_name("~polars.DataFrame.join", KNOWN) == "DataFrame.join"
    assert canonical_api_name("polars.dataframe.group_by.GroupBy.agg", KNOWN) == "GroupBy.agg"
    assert canonical_api_name("cs.numeric", KNOWN) == "selectors.numeric"
    # Bare namespace name: Expr is preferred over Series.
    assert canonical_api_name("str.to_uppercase", KNOWN) == "Expr.str.to_uppercase"
    assert canonical_api_name("Expr.alias()", KNOWN) == "Expr.alias"
    assert canonical_api_name("does_not_exist", KNOWN) is None


# --- reference .rst -> API chunks ----------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


RST = """\
======
String
======

The following methods are available under the `expr.str` attribute.

.. currentmodule:: polars
.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

    Expr.str.to_uppercase

Joining
-------

.. autosummary::
   :toctree: api/

    DataFrame.join
"""


def test_parse_rst_page_entries_title_section_module(tmp_path):
    page = parse_rst_page(_write(tmp_path / "string.rst", RST), "ref/string.rst")
    assert page.title == "String"
    names = [(e.name, e.module, e.section) for e in page.entries]
    assert names == [("Expr.str.to_uppercase", "polars", ""), ("DataFrame.join", "polars", "Joining")]
    # Prose survives (role/inline markup simplified); directives don't.
    assert any("expr.str" in text for _, text in page.prose_blocks)
    assert not any("autosummary" in text for _, text in page.prose_blocks)


def test_build_api_chunks_one_chunk_per_page_with_metadata(tmp_path, monkeypatch):
    _write(tmp_path / C.REFERENCE_DIR / "expressions" / "string.rst", RST)
    chunks, _, unresolved = build_api_chunks(tmp_path, count)
    assert unresolved == []
    by_name = {c.api_names[0]: c for c in chunks}
    up = by_name["Expr.str.to_uppercase"]
    assert up.id == "api:Expr.str.to_uppercase"
    assert up.kind == "api" and up.n_parts == 1
    assert up.title == "polars.Expr.str.to_uppercase"
    assert up.section == "String"
    assert up.source.endswith("expressions/string.rst")
    # Signature + description + the docstring's Examples, from polars 2.0.0.
    assert "to_uppercase(" in up.text and "uppercase" in up.text.lower()
    assert "Examples" in up.text and ">>>" in up.text
    assert by_name["DataFrame.join"].section == "String > Joining"


def test_long_api_page_is_split_and_every_part_keeps_the_header():
    header = "polars.DataFrame.thing\nDataFrame.thing(x)"
    params = "Parameters\n----------\n" + "\n".join(f"p{i} : int\n    " + "word " * 20 for i in range(5))
    examples = "Examples\n--------\n" + "\n\n".join(">>> df.thing(%d)\n" % i + "out " * 40 for i in range(40))
    doc = "Do the thing.\n\n" + params + "\n\n" + examples
    parts = split_api_doc(header, doc, count)
    assert len(parts) > 1
    assert all(p.startswith(header) for p in parts)
    assert all(count(p) <= C.API_PART_TOKENS or p.count(">>>") == 1 for p in parts)
    # Every doctest example appears intact in exactly one part.
    for i in range(40):
        assert sum(f">>> df.thing({i})\n" in p for p in parts) == 1


def test_short_api_page_is_not_split():
    assert split_api_doc("polars.f\nf()", "Short doc.", count) == ["polars.f\nf()\n\nShort doc."]


# --- guide markdown -------------------------------------------------------------


def test_markdown_blocks_keep_code_fences_whole_and_track_sections():
    md = (
        "# Title\n\nIntro.\n\n## A\n\nText a.\n\n```python\nx = 1\n\ny = 2\n```\n\n"
        "### A1\n\nDeep.\n\n## B\n\nText b.\n"
    )
    title, blocks = markdown_blocks(md)
    assert title == "Title"
    code = [b for _, b in blocks if b.startswith("```")]
    assert code == ["```python\nx = 1\n\ny = 2\n```"]  # blank line inside kept
    sections = {b: s for s, b in blocks}
    assert sections["Deep."] == "A > A1"
    assert sections["Text b."] == "B"


def test_pack_respects_max_and_never_splits_code():
    big_code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(400)) + "\n```"
    blocks = [("S", "word " * 200), ("S", big_code), ("S", "tail " * 200), ("T", "more " * 200)]
    chunks = pack_guide_blocks(blocks, count)
    assert any(text == big_code for _, text in chunks)  # oversized code block is its own chunk
    for _, text in chunks:
        if text != big_code:
            assert count(text) <= C.GUIDE_MAX_TOKENS
        assert text.count("```") % 2 == 0  # no fence cut in half


def test_pack_splits_oversized_tables_with_header_and_merges_tiny_chunks():
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| r{i} long cell text here | v{i} |" for i in range(200))
    chunks = pack_guide_blocks([("S", table)], count)
    assert len(chunks) > 1
    assert all(t.startswith("| a | b |\n|---|---|") for _, t in chunks)
    # A tiny trailing chunk is merged into its predecessor.
    chunks = pack_guide_blocks([("S", "w " * 450), ("T", "tiny bit")], count)
    assert len(chunks) == 1 and chunks[0][1].endswith("tiny bit")


def test_guide_macro_expansion_and_api_names(tmp_path):
    corpus = tmp_path
    _write(
        corpus / C.SNIPPET_DIR / "user-guide/expressions/strings.py",
        "# --8<-- [start:df]\nresult = df.with_columns(pl.col('a').str.to_uppercase())\n# --8<-- [end:df]\n",
    )
    _write(
        corpus / C.API_LINKS_FILE,
        "python:\n  str.to_uppercase: https://x/api/polars.Expr.str.to_uppercase.html\n",
    )
    _write(
        corpus / "docs/source/user-guide/expressions/strings.md",
        "# Strings\n\n## Casing\n\nUse it like this:\n\n"
        "{{code_block('user-guide/expressions/strings','df',['str.to_uppercase'])}}\n\n"
        '```python exec="on" result="text" session="s"\n--8<-- "python/user-guide/expressions/strings.py:df"\n```\n',
    )
    chunks = build_guide_chunks(corpus, count, KNOWN, rst_pages=[])
    assert len(chunks) == 1
    c = chunks[0]
    assert c.id == "guide:docs/source/user-guide/expressions/strings.md#001"
    assert c.title == "Strings" and c.section == "Casing"
    assert "str.to_uppercase()" in c.text  # snippet inlined
    assert "--8<--" not in c.text and "{{" not in c.text and "<!--api" not in c.text
    assert c.api_names == ["Expr.str.to_uppercase", "col"]


@pytest.mark.parametrize("block", ["```python\n\n```", "<!-- dprint-ignore -->"])
def test_empty_blocks_are_dropped(block):
    _, blocks = markdown_blocks(f"# T\n\n{block}\n\nReal text.\n")
    assert [b for _, b in blocks] == ["Real text."]
