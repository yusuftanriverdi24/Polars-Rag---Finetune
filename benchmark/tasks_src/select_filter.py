"""select_filter tasks: selecting columns and filtering rows.

Filter preserves input row order deterministically, so most tasks here keep
`ignore_row_order=False`. Sort-based tasks use unique sort keys to avoid
tie-ordering ambiguity.
"""

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
    {
        "id": "select_filter_002",
        "difficulty": "easy",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Keep only the orders whose `status` is exactly \"shipped\". Return all "
            "columns."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "status": ["shipped", "pending", "shipped", "cancelled", "shipped"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("status") == "shipped")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_003",
        "difficulty": "easy",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "I only care about the product and its price. Return just the `product` "
            "and `price` columns, in that order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "product": ["pen", "mug", "hat"],
            "warehouse": ["A", "B", "A"],
            "price": [2, 8, 15],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.select("product", "price")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_004",
        "difficulty": "easy",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Return the rows whose `score` is between 50 and 80 inclusive."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "player": ["a", "b", "c", "d", "e"],
            "score": [49, 50, 65, 80, 81],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("score").is_between(50, 80))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_005",
        "difficulty": "easy",
        "category": "select_filter",
        "touches_2_0_change": True,  # pandas habit: dropna(); polars uses drop_nulls/is_not_null
        "instruction": (
            "Some rows have a missing `email`. Drop every row where `email` is "
            "missing and return the rest unchanged."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "user": ["a", "b", "c", "d"],
            "email": ["a@x.com", None, "c@x.com", None],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("email").is_not_null())
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_006",
        "difficulty": "easy",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Keep only the rows whose `country` is one of \"US\", \"UK\" or \"DE\"."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "city": ["NYC", "London", "Paris", "Berlin", "Tokyo"],
            "country": ["US", "UK", "FR", "DE", "JP"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("country").is_in(["US", "UK", "DE"]))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_007",
        "difficulty": "medium",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Return the rows that are either in the \"eng\" department OR earn more "
            "than 100. Keep all columns and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "dept": ["eng", "sales", "eng", "sales"],
            "pay": [90, 120, 80, 70],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter((pl.col("dept") == "eng") | (pl.col("pay") > 100))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_008",
        "difficulty": "medium",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Give me the three highest-paid employees, ordered from highest to "
            "lowest salary. Return `name` and `salary`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "name": ["a", "b", "c", "d", "e"],
            "salary": [50, 90, 70, 120, 60],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.sort("salary", descending=True).head(3).select("name", "salary")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_009",
        "difficulty": "medium",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Return only the rows whose `value` is strictly greater than the "
            "average `value` across the whole table."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "item": ["a", "b", "c", "d"],
            "value": [10, 20, 30, 40],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("value") > pl.col("value").mean())
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_010",
        "difficulty": "medium",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Return the rows where the `code` has more than 3 characters."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"code": ["ab", "abcd", "xyz", "hello"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("code").str.len_chars() > 3)
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_011",
        "difficulty": "medium",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "Keep the rows where the actual sales exceeded the target, i.e. where "
            "`sales` is greater than `target`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "region": ["n", "s", "e", "w"],
            "sales": [100, 80, 120, 90],
            "target": [90, 90, 100, 95],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("sales") > pl.col("target"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_012",
        "difficulty": "hard",
        "category": "select_filter",
        "touches_2_0_change": True,  # requires a window count; pandas reaches for groupby/duplicated
        "instruction": (
            "Return only the rows whose `id` value appears more than once anywhere "
            "in the table. Preserve the original row order and keep all columns."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "id": [1, 2, 2, 3, 1, 4],
            "val": ["a", "b", "c", "d", "e", "f"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.len().over("id") > 1)
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_013",
        "difficulty": "hard",
        "category": "select_filter",
        "touches_2_0_change": False,
        "instruction": (
            "The rows are in time order. Return only the rows where `reading` is "
            "strictly higher than the reading in the row immediately before it "
            "(the first row can never qualify)."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "t": [1, 2, 3, 4, 5],
            "reading": [10, 12, 11, 15, 15],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("reading") > pl.col("reading").shift(1))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "select_filter_014",
        "difficulty": "hard",
        "category": "select_filter",
        "touches_2_0_change": True,  # per-group comparison via window; pandas groupby-transform habit
        "instruction": (
            "Within each `category`, keep only the rows whose `amount` is above "
            "that category's own average amount. Preserve the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "category": ["x", "x", "x", "y", "y"],
            "amount": [10, 20, 30, 5, 15],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("amount") > pl.col("amount").mean().over("category"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
