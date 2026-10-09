"""reshape tasks: pivot / unpivot / rename.

`unpivot` and `pivot` outputs are made deterministic with an explicit final
sort (pivot groups do not guarantee row order); group-by reductions over
reshaped data use `ignore_row_order=True`.
"""

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
    {
        "id": "reshape_002",
        "difficulty": "easy",
        "category": "reshape",
        "touches_2_0_change": True,  # melt -> unpivot
        "instruction": (
            "Turn the wide quarterly table into a long one: from `region`, `q1`, "
            "`q2` produce `region`, `quarter`, `value` with one row per "
            "region/quarter. Sort by `region` then `quarter`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "region": ["n", "s"],
            "q1": [10, 30],
            "q2": [20, 40],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.unpivot(
        index="region",
        on=["q1", "q2"],
        variable_name="quarter",
        value_name="value",
    ).sort("region", "quarter")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_003",
        "difficulty": "easy",
        "category": "reshape",
        "touches_2_0_change": True,  # pivot: 2.0 uses on= (was columns=)
        "instruction": (
            "Reshape from long to wide: each row gives a `subject` score for a "
            "student `id`. Produce one row per `id` with a column per subject "
            "holding that student's `score`. Sort the result by `id`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "subject": ["math", "science", "math", "science"],
            "score": [90, 85, 70, 95],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.pivot(on="subject", index="id", values="score").sort("id")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_004",
        "difficulty": "easy",
        "category": "reshape",
        "touches_2_0_change": False,
        "instruction": (
            "Rename the columns: `ds` should become `date` and `y` should become "
            "`value`. Keep the data and order unchanged."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "ds": ["2023-01", "2023-02"],
            "y": [10, 20],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.rename({"ds": "date", "y": "value"})
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_005",
        "difficulty": "easy",
        "category": "reshape",
        "touches_2_0_change": True,  # melt -> unpivot
        "instruction": (
            "Melt the metrics into long form: from `day`, `clicks`, `views`, "
            "`signups` produce `day`, `metric`, `value` with one row per "
            "day/metric. Sort by `day` then `metric`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "day": [1, 2],
            "clicks": [100, 150],
            "views": [1000, 1200],
            "signups": [5, 9],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.unpivot(
        index="day",
        on=["clicks", "views", "signups"],
        variable_name="metric",
        value_name="value",
    ).sort("day", "metric")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_006",
        "difficulty": "medium",
        "category": "reshape",
        "touches_2_0_change": True,  # pivot on= + aggregate_function
        "instruction": (
            "Build a table with one row per `store` and one column per `day`, "
            "where each cell is the total `sales` that store made that day (there "
            "can be several sales per store per day). Sort the result by `store`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "store": ["a", "a", "a", "b", "b"],
            "day": ["mon", "mon", "tue", "mon", "tue"],
            "sales": [10, 5, 7, 3, 4],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.pivot(
        on="day", index="store", values="sales", aggregate_function="sum"
    ).sort("store")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_007",
        "difficulty": "medium",
        "category": "reshape",
        "touches_2_0_change": True,  # melt -> unpivot, then filter
        "instruction": (
            "Reshape `id`, `a`, `b`, `c` into long form (`id`, `name`, `value`), "
            "then keep only the rows whose `value` is greater than 5. Sort by "
            "`id` then `name`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "id": [1, 2],
            "a": [10, 2],
            "b": [3, 8],
            "c": [1, 1],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return (
        df.unpivot(index="id", on=["a", "b", "c"], variable_name="name", value_name="value")
        .filter(pl.col("value") > 5)
        .sort("id", "name")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_008",
        "difficulty": "medium",
        "category": "reshape",
        "touches_2_0_change": True,  # pivot on= ; missing combinations become 0
        "instruction": (
            "Make a table with one row per `user` and one column per `action`, "
            "where each cell counts how many times that user performed that "
            "action. Users who never performed an action should show 0, not a "
            "missing value. Sort the result by `user`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "user": ["a", "a", "b"],
            "action": ["click", "view", "click"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    wide = df.pivot(on="action", index="user", values="action", aggregate_function="len")
    return wide.with_columns(pl.col(pl.UInt32).fill_null(0)).sort("user")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_009",
        "difficulty": "medium",
        "category": "reshape",
        "touches_2_0_change": True,  # melt -> unpivot, keeping one id column
        "instruction": (
            "Keeping `country` as an identifier, turn the `gold`, `silver`, "
            "`bronze` medal columns into long form with columns `country`, "
            "`medal`, `count`. Sort by `country` then `medal`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "country": ["US", "UK"],
            "gold": [10, 4],
            "silver": [8, 6],
            "bronze": [5, 7],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.unpivot(
        index="country",
        on=["gold", "silver", "bronze"],
        variable_name="medal",
        value_name="count",
    ).sort("country", "medal")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_010",
        "difficulty": "hard",
        "category": "reshape",
        "touches_2_0_change": True,  # pivot on=, then a derived column
        "instruction": (
            "Each row gives a product's revenue in a given `month` ('jan' or "
            "'feb'). Produce one row per `product` with its January and February "
            "revenue, plus a column `growth` equal to February minus January. "
            "Sort the result by `product`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "product": ["x", "x", "y", "y"],
            "month": ["jan", "feb", "jan", "feb"],
            "revenue": [100, 130, 50, 40],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    wide = df.pivot(on="month", index="product", values="revenue")
    return wide.with_columns(
        (pl.col("feb") - pl.col("jan")).alias("growth")
    ).sort("product")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_011",
        "difficulty": "hard",
        "category": "reshape",
        "touches_2_0_change": True,  # unpivot then aggregate
        "instruction": (
            "Given a wide table of student scores (`student`, `math`, `science`, "
            "`art`), compute the average score per subject across all students. "
            "Return columns `subject` and `avg`. Order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "student": ["a", "b"],
            "math": [80, 90],
            "science": [70, 50],
            "art": [100, 60],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return (
        df.unpivot(index="student", variable_name="subject", value_name="score")
        .group_by("subject")
        .agg(pl.col("score").mean().alias("avg"))
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "reshape_012",
        "difficulty": "hard",
        "category": "reshape",
        "touches_2_0_change": True,  # pivot on= with a multi-column index
        "instruction": (
            "Build a table keyed by both `region` and `segment` (one row per "
            "combination), with one column per `product` holding the total "
            "`units` sold. Sort the result by `region` then `segment`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "region": ["n", "n", "s", "s"],
            "segment": ["pro", "pro", "pro", "home"],
            "product": ["x", "y", "x", "x"],
            "units": [1, 2, 3, 4],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.pivot(
        on="product", index=["region", "segment"], values="units", aggregate_function="sum"
    ).sort("region", "segment")
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "reshape_013",
        "difficulty": "hard",
        "category": "reshape",
        "touches_2_0_change": True,  # unpivot then per-key reduction
        "instruction": (
            "A wide table holds quarterly revenue per region (`region`, `q1`, "
            "`q2`, `q3`, `q4`). Compute the average revenue per quarter across all "
            "regions. Return `quarter` and `avg`, sorted by `quarter`."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "region": ["n", "s"],
            "q1": [10, 20],
            "q2": [30, 50],
            "q3": [0, 10],
            "q4": [100, 100],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return (
        df.unpivot(index="region", variable_name="quarter", value_name="rev")
        .group_by("quarter")
        .agg(pl.col("rev").mean().alias("avg"))
        .sort("quarter")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
