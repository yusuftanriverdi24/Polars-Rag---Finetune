"""lazy tasks: LazyFrame query building and `.collect()`."""

TASKS = [
    {
        "id": "lazy_001",
        "difficulty": "medium",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Using the Polars lazy API, filter to rows where `value` is positive, "
            "then compute the sum of `value` per `grp`. Return an eager DataFrame "
            "with columns `grp` and `value`. Row order does not matter."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "grp": ["a", "a", "b", "b", "c"],
            "value": [5, -3, 2, 4, -1],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return (
        lf.filter(pl.col("value") > 0)
        .group_by("grp")
        .agg(pl.col("value").sum())
        .collect()
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
]
