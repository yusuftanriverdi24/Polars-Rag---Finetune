"""group_by tasks: grouping and aggregation."""

TASKS = [
    {
        "id": "group_by_001",
        "difficulty": "easy",
        "category": "group_by",
        "touches_2_0_change": True,  # pandas habit: .groupby(); polars uses .group_by()
        "instruction": (
            "Compute the total `amount` per `category`. Return columns "
            "`category` and `amount`. Row order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "category": ["a", "b", "a", "b", "a"],
            "amount": [10, 5, 7, 3, 1],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("category").agg(pl.col("amount").sum())
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_002",
        "difficulty": "medium",
        "category": "group_by",
        "touches_2_0_change": True,  # pandas habit: groupby; also pl.len() vs count
        "instruction": (
            "For each `category`, compute the number of rows as `n` and the mean "
            "`value` as `avg`. Sort the result by `avg` in descending order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "category": ["x", "y", "x", "y", "z", "x"],
            "value": [1.0, 10.0, 3.0, 6.0, 8.0, 2.0],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return (
        df.group_by("category")
        .agg(
            pl.len().alias("n"),
            pl.col("value").mean().alias("avg"),
        )
        .sort("avg", descending=True)
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
