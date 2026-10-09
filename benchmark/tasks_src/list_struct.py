"""list_struct tasks: the `.list` / `.struct` namespaces."""

TASKS = [
    {
        "id": "list_struct_001",
        "difficulty": "hard",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each `user` has one row per `tag`. Collapse to one row per `user` "
            "with a column `tags` that is the sorted list of that user's distinct "
            "tags. Row order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "user": ["u1", "u1", "u2", "u2", "u2"],
            "tag": ["z", "a", "m", "b", "b"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("user").agg(
        pl.col("tag").unique().sort().alias("tags")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
]
