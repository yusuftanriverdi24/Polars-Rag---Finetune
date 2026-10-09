from polars_llm.bench.extract import extract_code


def test_simple_python_fence():
    out = extract_code("Here you go:\n```python\ndef solve(df):\n    return df\n```\nDone.")
    assert out == "def solve(df):\n    return df"


def test_py_language_tag():
    assert extract_code("```py\nx = 1\n```") == "x = 1"


def test_bare_fence_no_language():
    assert extract_code("```\nx = 1\n```") == "x = 1"


def test_no_fence_returns_stripped_text():
    assert extract_code("  def solve(df):\n    return df  ") == "def solve(df):\n    return df"


def test_prefers_python_tagged_block_over_earlier_text_block():
    text = (
        "Example output:\n```text\nnot code\n```\n"
        "Solution:\n```python\ndef solve(df):\n    return df\n```"
    )
    assert extract_code(text) == "def solve(df):\n    return df"


def test_first_block_when_none_tagged_python():
    text = "```\nfirst = 1\n```\nmid\n```\nsecond = 2\n```"
    assert extract_code(text) == "first = 1"


def test_unterminated_fence_takes_rest():
    # Truncated generation: opening fence but no closing fence.
    assert extract_code("```python\ndef solve(df):\n    return df") == "def solve(df):\n    return df"


def test_empty_and_none():
    assert extract_code("") == ""
    assert extract_code(None) == ""
