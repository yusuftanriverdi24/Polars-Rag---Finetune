"""window tasks: expressions evaluated over groups (`.over(...)`)."""

TASKS = [
    {
        "id": "window_001",
        "difficulty": "hard",
        "category": "window",
        "touches_2_0_change": True,  # window via .over(); pandas uses groupby-transform
        "instruction": (
            "Add a column `rank_in_dept` giving each employee's dense rank by "
            "`salary` (highest salary = rank 1) within their `dept`. Keep all "
            "original columns and preserve the original row order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "dept": ["eng", "eng", "eng", "sales", "sales"],
            "name": ["a", "b", "c", "d", "e"],
            "salary": [100, 120, 120, 80, 90],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("salary")
        .rank(method="dense", descending=True)
        .over("dept")
        .cast(pl.UInt32)
        .alias("rank_in_dept")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
