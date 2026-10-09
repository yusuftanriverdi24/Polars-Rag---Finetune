"""datetime tasks: the `.dt` namespace and temporal parsing."""

TASKS = [
    {
        "id": "datetime_001",
        "difficulty": "medium",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "The `ts` column holds date strings in 'YYYY-MM-DD' format. Parse it "
            "to a Date, then return rows from the year 2023, with an added column "
            "`month` (integer month). Keep columns `ts`, `month` and preserve order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"ts": ["2022-12-31", "2023-01-15", "2023-07-04", "2024-02-01"]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return (
        df.with_columns(pl.col("ts").str.to_date("%Y-%m-%d"))
        .filter(pl.col("ts").dt.year() == 2023)
        .with_columns(pl.col("ts").dt.month().alias("month"))
        .select("ts", "month")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
