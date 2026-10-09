"""window tasks: expressions evaluated over groups with `.over(...)`.

All of these keep every input row and preserve the original row order
(`with_columns`/`filter` are order-preserving), so `ignore_row_order=False`.
"""

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
    {
        "id": "window_002",
        "difficulty": "easy",
        "category": "window",
        "touches_2_0_change": True,  # per-group broadcast via .over(); pandas transform('max')
        "instruction": (
            "Add a column `dept_max` holding, for each row, the maximum `salary` "
            "found in that row's `dept`. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "dept": ["eng", "eng", "sales", "sales"],
            "salary": [100, 140, 80, 90],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("salary").max().over("dept").alias("dept_max")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_003",
        "difficulty": "easy",
        "category": "window",
        "touches_2_0_change": True,  # per-group broadcast via .over()
        "instruction": (
            "Add a column `team_avg` equal to the average `score` of each row's "
            "`team`. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "team": ["a", "a", "b", "b"],
            "score": [2.0, 4.0, 10.0, 20.0],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("score").mean().over("team").alias("team_avg")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_004",
        "difficulty": "easy",
        "category": "window",
        "touches_2_0_change": True,  # count within group via .over()
        "instruction": (
            "Add a column `n_in_category` telling how many rows share each row's "
            "`category`. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"category": ["x", "x", "x", "y", "z", "z"]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.len().over("category").alias("n_in_category")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_005",
        "difficulty": "medium",
        "category": "window",
        "touches_2_0_change": True,  # cum_sum (renamed from cumsum) over a group
        "instruction": (
            "The rows are already in chronological order. Add a column `running` "
            "that holds the running total of `amount` within each `account` "
            "(accumulating top to bottom). Keep the original row order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "account": ["a", "a", "b", "a", "b"],
            "amount": [10, 20, 100, 30, 200],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("amount").cum_sum().over("account").alias("running")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_006",
        "difficulty": "medium",
        "category": "window",
        "touches_2_0_change": True,  # per-group share via .over()
        "instruction": (
            "Add a column `pct_of_group` giving each row's `amount` as a fraction "
            "of the total `amount` in its `group`. Keep all rows and the original "
            "order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "amount": [1, 3, 10, 10],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        (pl.col("amount") / pl.col("amount").sum().over("group")).alias("pct_of_group")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_007",
        "difficulty": "medium",
        "category": "window",
        "touches_2_0_change": True,  # first-of-group broadcast via .over()
        "instruction": (
            "Within each `session` (rows are in order), add a column `first_page` "
            "holding the first `page` value seen in that session. Keep all rows "
            "and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2],
            "page": ["home", "cart", "pay", "home", "faq"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("page").first().over("session").alias("first_page")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_008",
        "difficulty": "medium",
        "category": "window",
        "touches_2_0_change": True,  # deviation from group mean via .over()
        "instruction": (
            "Add a column `diff_from_avg` equal to each row's `value` minus the "
            "average `value` of its `group`. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [10.0, 20.0, 5.0, 15.0],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        (pl.col("value") - pl.col("value").mean().over("group")).alias("diff_from_avg")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_009",
        "difficulty": "medium",
        "category": "window",
        "touches_2_0_change": True,  # lag within group via shift().over()
        "instruction": (
            "The rows are in time order. Add a column `prev_amount` holding the "
            "`amount` from the previous row within the same `account` (missing for "
            "the first row of each account). Keep the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "account": ["a", "a", "a", "b", "b"],
            "amount": [10, 20, 30, 100, 200],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("amount").shift(1).over("account").alias("prev_amount")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_010",
        "difficulty": "medium",
        "category": "window",
        "touches_2_0_change": True,  # ratio to group max via .over()
        "instruction": (
            "Add a column `vs_best` giving each row's `sales` divided by the "
            "highest `sales` within its `region`. Keep all rows and the original "
            "order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "region": ["n", "n", "s", "s"],
            "sales": [50, 100, 20, 40],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        (pl.col("sales") / pl.col("sales").max().over("region")).alias("vs_best")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_011",
        "difficulty": "hard",
        "category": "window",
        "touches_2_0_change": True,  # top-N per group via rank().over() + filter
        "instruction": (
            "For each `category`, keep only the two rows with the highest `value`. "
            "Preserve the original row order and keep all columns."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "category": ["a", "a", "a", "b", "b", "b"],
            "value": [5, 9, 1, 8, 2, 6],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(
        pl.col("value").rank(method="ordinal", descending=True).over("category") <= 2
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_012",
        "difficulty": "hard",
        "category": "window",
        "touches_2_0_change": True,  # cum_max (renamed from cummax) over a group
        "instruction": (
            "The rows are in time order. Add a column `peak` holding the running "
            "maximum of `price` so far within each `stock` (top to bottom). Keep "
            "the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "stock": ["x", "x", "x", "y", "y"],
            "price": [10, 8, 12, 5, 7],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("price").cum_max().over("stock").alias("peak")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "window_013",
        "difficulty": "hard",
        "category": "window",
        "touches_2_0_change": True,  # two window expressions in one pass
        "instruction": (
            "Add two columns: `rnk`, the dense rank of `value` within each `group` "
            "with the largest value ranked 1, and `is_top`, a boolean that is true "
            "only for the row holding the group's maximum `value`. Keep all rows "
            "and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [3, 7, 5, 5],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("value").rank(method="dense", descending=True).over("group").cast(pl.UInt32).alias("rnk"),
        (pl.col("value") == pl.col("value").max().over("group")).alias("is_top"),
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
