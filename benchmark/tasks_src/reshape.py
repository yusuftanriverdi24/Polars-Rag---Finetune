"""reshape tasks: pivot / unpivot / transpose."""

TASKS = [
    {
        "id": "reshape_001",
        "difficulty": "medium",
        "category": "reshape",
        "touches_2_0_change": True,  # pandas habit: melt(); polars 2.0 uses unpivot()
        "instruction": (
            "Reshape from wide to long: given columns `id`, `math`, `science`, "
            "produce columns `id`, `subject`, `score` with one row per "
            "id/subject. Sort by `id` then `subject`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "id": [1, 2],
            "math": [90, 70],
            "science": [85, 95],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.unpivot(
        index="id",
        on=["math", "science"],
        variable_name="subject",
        value_name="score",
    ).sort("id", "subject")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
