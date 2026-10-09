"""lazy tasks: LazyFrame query building and `.collect()`.

In Polars 2.0 the lazy API defaults to the streaming engine, which does not
guarantee row order for operations that don't require it. Tasks therefore use
`ignore_row_order=True` unless the reference ends with an explicit `.sort(...)`.
"""

TASKS = [
    {
        "id": "lazy_001",
        "difficulty": "medium",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Using the Polars lazy API, filter to rows where `value` is positive, "
            "then compute the sum of `value` per `grp`. Return an eager DataFrame "
            "with columns `grp` and `value`. Row order does not matter."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "grp": ["a", "a", "b", "b", "c"],
            "value": [5, -3, 2, 4, -1],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return (
        lf.filter(pl.col("value") > 0)
        .group_by("grp")
        .agg(pl.col("value").sum())
        .collect()
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "lazy_002",
        "difficulty": "easy",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily: keep rows where `value` is greater than 0 and return "
            "only the `id` and `value` columns as an eager DataFrame. Order does "
            "not matter."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "id": [1, 2, 3, 4],
            "value": [5, -2, 0, 7],
            "other": ["a", "b", "c", "d"],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return lf.filter(pl.col("value") > 0).select("id", "value").collect()
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "lazy_003",
        "difficulty": "easy",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily: add a column `doubled` equal to twice `value`, and "
            "return the result as an eager DataFrame. Order does not matter."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame({"value": [1, 2, 3]})
}
''',
        "reference_solution": '''
def solve(lf):
    return lf.with_columns((pl.col("value") * 2).alias("doubled")).collect()
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "lazy_004",
        "difficulty": "easy",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily and return the rows sorted by `score` from highest to "
            "lowest, as an eager DataFrame."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "name": ["a", "b", "c"],
            "score": [30, 10, 20],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return lf.sort("score", descending=True).collect()
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "lazy_005",
        "difficulty": "easy",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily: return a single column, the `temp` column renamed to "
            "`temperature`, as an eager DataFrame. Order does not matter."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "temp": [20, 21, 19],
            "city": ["a", "b", "c"],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return lf.select(pl.col("temp").alias("temperature")).collect()
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "lazy_006",
        "difficulty": "medium",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily: among rows where `qty` is at least 2, compute the total "
            "`qty` per `product`. Return `product` and `qty` as an eager "
            "DataFrame. Order does not matter."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "product": ["a", "a", "b", "b", "a"],
            "qty": [1, 3, 5, 2, 4],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return (
        lf.filter(pl.col("qty") >= 2)
        .group_by("product")
        .agg(pl.col("qty").sum())
        .collect()
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "lazy_007",
        "difficulty": "medium",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily with two lazy frames: inner-join `orders` to `products` "
            "on `product` and return `order_id` with its `price`, as an eager "
            "DataFrame. Order does not matter."
        ),
        "signature": "def solve(orders: pl.LazyFrame, products: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "orders": pl.LazyFrame(
        {
            "order_id": [1, 2, 3],
            "product": ["pen", "mug", "pen"],
        }
    ),
    "products": pl.LazyFrame(
        {
            "product": ["pen", "mug"],
            "price": [2, 8],
        }
    ),
}
''',
        "reference_solution": '''
def solve(orders, products):
    return (
        orders.join(products, on="product", how="inner")
        .select("order_id", "price")
        .collect()
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "lazy_008",
        "difficulty": "medium",
        "category": "lazy",
        "touches_2_0_change": True,  # group_by + pl.len(); relies on 2.0 len() aggregate
        "instruction": (
            "Work lazily: for each `city` compute the row count as `n` and the "
            "average `temp` as `avg`, then return the result sorted by `city` "
            "ascending, as an eager DataFrame."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "city": ["b", "a", "b", "a", "a"],
            "temp": [10.0, 20.0, 30.0, 40.0, 60.0],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return (
        lf.group_by("city")
        .agg(pl.len().alias("n"), pl.col("temp").mean().alias("avg"))
        .sort("city")
        .collect()
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "lazy_009",
        "difficulty": "medium",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily: starting from the transactions, keep only debits "
            "(`kind` == \"debit\"), add a column `abs_amount` equal to the "
            "absolute value of `amount`, then keep only rows where `abs_amount` is "
            "above 50. Return the result as an eager DataFrame. Order does not "
            "matter."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "kind": ["debit", "credit", "debit", "debit"],
            "amount": [-100, 200, -30, -80],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return (
        lf.filter(pl.col("kind") == "debit")
        .with_columns(pl.col("amount").abs().alias("abs_amount"))
        .filter(pl.col("abs_amount") > 50)
        .collect()
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "lazy_010",
        "difficulty": "medium",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily: return the two rows with the largest `value`, ordered "
            "from largest to smallest, as an eager DataFrame."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "item": ["a", "b", "c", "d"],
            "value": [7, 2, 9, 4],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return lf.sort("value", descending=True).head(2).collect()
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "lazy_011",
        "difficulty": "hard",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily with two lazy frames: join `sales` to `regions` on "
            "`store`, then compute total `amount` per `region`, and return the "
            "result sorted by `region` ascending, as an eager DataFrame."
        ),
        "signature": "def solve(sales: pl.LazyFrame, regions: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "sales": pl.LazyFrame(
        {
            "store": [1, 2, 1, 3],
            "amount": [10, 20, 30, 40],
        }
    ),
    "regions": pl.LazyFrame(
        {
            "store": [1, 2, 3],
            "region": ["east", "west", "east"],
        }
    ),
}
''',
        "reference_solution": '''
def solve(sales, regions):
    return (
        sales.join(regions, on="store", how="inner")
        .group_by("region")
        .agg(pl.col("amount").sum())
        .sort("region")
        .collect()
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "lazy_012",
        "difficulty": "hard",
        "category": "lazy",
        "touches_2_0_change": False,
        "instruction": (
            "Work lazily: compute the total `amount` per `customer`, keep only "
            "customers whose total is strictly greater than 100, and return the "
            "survivors sorted by total `amount` descending, as an eager "
            "DataFrame with columns `customer` and `amount`."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "customer": ["a", "a", "b", "c", "c"],
            "amount": [60, 80, 50, 70, 90],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return (
        lf.group_by("customer")
        .agg(pl.col("amount").sum())
        .filter(pl.col("amount") > 100)
        .sort("amount", descending=True)
        .collect()
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "lazy_013",
        "difficulty": "hard",
        "category": "lazy",
        "touches_2_0_change": True,  # window expression inside a lazy query
        "instruction": (
            "Work lazily: add a column `group_total` holding the sum of `amount` "
            "within each row's `group`, and return the result as an eager "
            "DataFrame. Order does not matter."
        ),
        "signature": "def solve(lf: pl.LazyFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "lf": pl.LazyFrame(
        {
            "group": ["a", "a", "b", "b"],
            "amount": [1, 2, 10, 20],
        }
    )
}
''',
        "reference_solution": '''
def solve(lf):
    return lf.with_columns(
        pl.col("amount").sum().over("group").alias("group_total")
    ).collect()
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
]
