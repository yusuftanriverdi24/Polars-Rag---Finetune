"""string tasks: the `.str` namespace."""

TASKS = [
    {
        "id": "string_001",
        "difficulty": "easy",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `upper` containing the uppercase form of `name`, and "
            "keep only rows whose `name` contains the letter 'a' (case-insensitive)."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"name": ["Ada", "Bob", "cora", "Mel"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("name").str.to_uppercase().alias("upper")
    ).filter(pl.col("name").str.contains("(?i)a"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
