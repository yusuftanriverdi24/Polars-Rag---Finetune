from polars_llm.bench.metrics import aggregate


def _rec(id_, status, pandas_style_name=None, **kw):
    return {
        "id": id_,
        "difficulty": "easy",
        "category": "string",
        "status": status,
        "pandas_style_name": pandas_style_name,
        **kw,
    }


def test_pandas_style_name_count_and_breakdown():
    m = aggregate(
        [
            _rec("a", "pass"),
            _rec("b", "hallucinated_api", "str.strip"),
            _rec("c", "hallucinated_api", "suffixes="),
            _rec("d", "hallucinated_api", "suffixes="),
            _rec("e", "hallucinated_api"),  # hallucinated, not pandas-style
            _rec("f", "runtime_error"),
        ]
    )
    assert m["pandas_style_name_count"] == 3
    assert m["pandas_style_names"] == {"str.strip": 1, "suffixes=": 2}
    # The secondary tag never changes the primary status histogram.
    assert m["status_counts"]["hallucinated_api"] == 4
    assert m["pass_at_1"] == round(1 / 6, 4)


def test_records_without_the_tag_count_as_zero():
    m = aggregate([{"id": "a", "status": "pass"}])
    assert m["pandas_style_name_count"] == 0
    assert m["pandas_style_names"] == {}
