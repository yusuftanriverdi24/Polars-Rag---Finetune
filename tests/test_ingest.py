"""Unit tests for the docs-ingest path selector (no network)."""

from polars_llm.rag.ingest import _keep


def test_keeps_user_guide_and_snippets():
    assert _keep("docs/source/user-guide/expressions/strings.md")
    assert _keep("docs/source/src/python/user-guide/expressions/strings.py")
    assert _keep("docs/source/user-guide/migration/pandas.md")


def test_keeps_upgrade_changelog_and_api_reference():
    assert _keep("docs/source/releases/upgrade/2.md")
    assert _keep("docs/source/releases/changelog.md")
    assert _keep("py-polars/docs/source/reference/dataframe/index.rst")


def test_rejects_unrelated_paths():
    assert not _keep("crates/polars-python/src/c_api/mod.rs")
    assert not _keep("py-polars/polars/dataframe/frame.py")
    assert not _keep("docs/source/src/rust/user-guide/expressions/strings.rs")
    assert not _keep("README.md")
