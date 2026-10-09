"""select_filter tasks: selecting columns and filtering rows."""

TASKS = [
    {
        "id": "select_filter_001",
        "difficulty": "easy",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Return only the rows where `age` is at least 30, keeping just the "
            "`name` and `age` columns (in that order)."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "name": ["Ada", "Bo", "Cy", "Di"],
            "age": [41, 22, 30, 17],
            "city": ["NYC", "LA", "SF", "LA"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("age") >= 30).select("name", "age")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
