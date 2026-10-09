"""group_by tasks: grouping and aggregation.

group_by does not guarantee output row order (doubly so under the 2.0 streaming
engine), so aggregation results use `ignore_row_order=True` unless the reference
ends with an explicit sort.
"""

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
            "value": [1.0, 10.0, 3.0, 6.0, 9.0, 2.0],
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
    {
        "id": "group_by_003",
        "difficulty": "easy",
        "category": "group_by",
        "touches_2_0_change": True,  # row count per group -> pl.len(); pandas .size()/.count()
        "instruction": (
            "Count how many rows fall under each `status`. Return `status` and the "
            "count as `n`. Row order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"status": ["ok", "fail", "ok", "ok", "fail"]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("status").agg(pl.len().alias("n"))
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_004",
        "difficulty": "easy",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "Find the highest `price` within each `category`. Return `category` "
            "and `price`. Order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "category": ["a", "a", "b", "b", "b"],
            "price": [3, 9, 4, 1, 7],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("category").agg(pl.col("price").max())
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_005",
        "difficulty": "easy",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "For each `product`, compute both the total `qty` and the total "
            "`revenue`. Keep the column names `qty` and `revenue`. Order does not "
            "matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "product": ["pen", "pen", "mug", "mug"],
            "qty": [2, 3, 1, 4],
            "revenue": [4, 6, 8, 32],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("product").agg(
        pl.col("qty").sum(),
        pl.col("revenue").sum(),
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_006",
        "difficulty": "easy",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "Compute the average `rating` per `movie`. Return `movie` and the mean "
            "as `rating`. Order is irrelevant."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "movie": ["m1", "m1", "m2", "m2"],
            "rating": [4.0, 2.0, 5.0, 3.0],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("movie").agg(pl.col("rating").mean())
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_007",
        "difficulty": "medium",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "Compute the total `amount` per `customer`, then keep only the "
            "customers whose total is at least 100. Return `customer` and "
            "`amount`. Order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "customer": ["a", "a", "b", "c", "c"],
            "amount": [60, 50, 30, 80, 40],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return (
        df.group_by("customer")
        .agg(pl.col("amount").sum())
        .filter(pl.col("amount") >= 100)
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_008",
        "difficulty": "medium",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "For each `city`, count how many distinct `customer` values appear. "
            "Return `city` and the distinct count as `customers`. Order does not "
            "matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "city": ["NYC", "NYC", "NYC", "LA", "LA"],
            "customer": ["a", "a", "b", "c", "c"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("city").agg(
        pl.col("customer").n_unique().alias("customers")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_009",
        "difficulty": "medium",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "Compute the total `units` for every combination of `region` and "
            "`product`. Return `region`, `product` and `units`. Order does not "
            "matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "region": ["n", "n", "s", "s", "n"],
            "product": ["x", "y", "x", "x", "x"],
            "units": [1, 2, 3, 4, 5],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("region", "product").agg(pl.col("units").sum())
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_010",
        "difficulty": "medium",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "For each `team`, report the lowest and highest `score` as columns "
            "`lo` and `hi`. Order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "team": ["a", "a", "a", "b", "b"],
            "score": [10, 4, 7, 9, 2],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("team").agg(
        pl.col("score").min().alias("lo"),
        pl.col("score").max().alias("hi"),
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_011",
        "difficulty": "medium",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "For each `account`, sum only the positive `delta` values (ignore "
            "negative ones). Return `account` and the sum as `inflow`. Order does "
            "not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "account": ["a", "a", "a", "b", "b"],
            "delta": [10, -5, 3, -2, 8],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("account").agg(
        pl.col("delta").filter(pl.col("delta") > 0).sum().alias("inflow")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_012",
        "difficulty": "hard",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "For each `category`, compute what share of the grand total `amount` "
            "it accounts for. Return `category` and the share as `share` (a "
            "fraction between 0 and 1). Order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "category": ["a", "a", "b", "c"],
            "amount": [20, 20, 40, 20],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return (
        df.group_by("category")
        .agg(pl.col("amount").sum().alias("amount"))
        .with_columns((pl.col("amount") / pl.col("amount").sum()).alias("share"))
        .select("category", "share")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_013",
        "difficulty": "hard",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "For each `region`, count how many sales were above 100 as `big` and "
            "how many were 100 or below as `small`. Order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "region": ["n", "n", "n", "s", "s"],
            "sale": [150, 50, 200, 90, 300],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("region").agg(
        (pl.col("sale") > 100).sum().alias("big"),
        (pl.col("sale") <= 100).sum().alias("small"),
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "group_by_014",
        "difficulty": "hard",
        "category": "group_by",
        "touches_2_0_change": False,
        "instruction": (
            "Compute a weighted average `price` per `product`, weighting each row "
            "by its `qty` (sum of price*qty divided by sum of qty). Return "
            "`product` and the result as `wavg`. Order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "product": ["x", "x", "y", "y"],
            "price": [10, 20, 30, 50],
            "qty": [1, 3, 2, 2],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("product").agg(
        ((pl.col("price") * pl.col("qty")).sum() / pl.col("qty").sum()).alias("wavg")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
]
