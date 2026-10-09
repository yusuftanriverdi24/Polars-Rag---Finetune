"""list_struct tasks: the `.list` / `.struct` namespaces.

Row-wise list/struct operations preserve order. Group-by aggregations that build
lists use `ignore_row_order=True`.
"""

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
    {
        "id": "list_struct_002",
        "difficulty": "easy",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each row's `tags` column holds a list of tags. Add a column `n` with "
            "the number of tags in each list. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"tags": [["a", "b", "c"], ["x"], ["p", "q"]]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("tags").list.len().alias("n"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_003",
        "difficulty": "easy",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each `order` has a list of item names in `items`. Produce one row per "
            "item, keeping the `order` it belongs to alongside the single `items` "
            "value. Preserve the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "order": [1, 2],
            "items": [["pen", "mug"], ["hat"]],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.explode("items")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_004",
        "difficulty": "easy",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each row's `nums` column is a list of integers. Add a column `total` "
            "holding the sum of each list. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"nums": [[1, 2, 3], [10], [4, 6]]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("nums").list.sum().alias("total"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_005",
        "difficulty": "medium",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Keep only the rows whose `tags` list contains the tag \"vip\"."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "user": ["a", "b", "c"],
            "tags": [["vip", "new"], ["new"], ["vip"]],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("tags").list.contains("vip"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_006",
        "difficulty": "medium",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each row's `path` column is a list of steps. Add a column `first_step` "
            "holding the first element of each list. Keep all rows and the "
            "original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"path": [["home", "cart", "pay"], ["home", "faq"]]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("path").list.get(0).alias("first_step"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_007",
        "difficulty": "medium",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "The `point` column is a struct with fields `x` and `y`. Add a plain "
            "column `x` holding that struct's `x` field. Keep all rows and the "
            "original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"point": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("point").struct.field("x").alias("x"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_008",
        "difficulty": "medium",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each row's `nums` column is a list of integers. Add a column `sorted` "
            "that holds the same list sorted in ascending order. Keep all rows and "
            "the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"nums": [[3, 1, 2], [9, 4]]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("nums").list.sort().alias("sorted"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_009",
        "difficulty": "medium",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each row's `scores` column is a list of numbers. Add a column `best` "
            "holding the maximum value in each list. Keep all rows and the "
            "original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"scores": [[10, 40, 20], [5, 5], [99]]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("scores").list.max().alias("best"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_010",
        "difficulty": "hard",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each row's `tags` column is a list that may contain duplicates. Add a "
            "column `n_distinct` with the number of distinct tags in each list. "
            "Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"tags": [["a", "a", "b"], ["x", "y", "z"], ["q", "q"]]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("tags").list.unique().list.len().alias("n_distinct")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_011",
        "difficulty": "hard",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each row's `nums` column is a list of integers. Add a column "
            "`doubled_sum` equal to the sum of each list after doubling every "
            "element. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"nums": [[1, 2, 3], [10, 20]]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("nums").list.eval(pl.element() * 2).list.sum().alias("doubled_sum")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_012",
        "difficulty": "hard",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "The `point` column is a struct with fields `x` and `y`. Expand it so "
            "that `x` and `y` become their own top-level columns (dropping the "
            "struct). Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"point": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.unnest("point")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "list_struct_013",
        "difficulty": "hard",
        "category": "list_struct",
        "touches_2_0_change": False,
        "instruction": (
            "Each `customer` has one row per purchase `amount`. For each customer, "
            "produce their two largest purchase amounts as a list `top2`, ordered "
            "from largest to smallest. Order of customers does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "customer": ["a", "a", "a", "b", "b"],
            "amount": [10, 50, 30, 5, 80],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("customer").agg(
        pl.col("amount").sort(descending=True).head(2).alias("top2")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
]
