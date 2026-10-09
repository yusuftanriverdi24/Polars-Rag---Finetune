"""join tasks: combining frames on keys.

Joins do not guarantee output row order (especially under the 2.0 streaming
engine), so join results use `ignore_row_order=True` unless the reference ends
with an explicit sort.
"""

TASKS = [
    {
        "id": "join_001",
        "difficulty": "medium",
        "category": "join",
        "touches_2_0_change": True,  # pandas habit: .merge(); polars uses .join()
        "instruction": (
            "Inner-join `orders` to `customers` on `customer_id`, producing one "
            "row per order with columns `order_id`, `customer_name`, `amount`. "
            "Row order does not matter."
        ),
        "signature": "def solve(orders: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "orders": pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "customer_id": [10, 20, 10, 99],
            "amount": [100, 50, 25, 70],
        }
    ),
    "customers": pl.DataFrame(
        {
            "customer_id": [10, 20, 30],
            "customer_name": ["Ada", "Bo", "Cy"],
        }
    ),
}
''',
        "reference_solution": '''
def solve(orders, customers):
    return (
        orders.join(customers, on="customer_id", how="inner")
        .select("order_id", "customer_name", "amount")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_002",
        "difficulty": "easy",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "Attach each sale's product `price` by matching `product` against the "
            "`products` table. Return `sale_id`, `product`, `price`. Only keep "
            "sales whose product exists in `products`. Order does not matter."
        ),
        "signature": "def solve(sales: pl.DataFrame, products: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "sales": pl.DataFrame(
        {
            "sale_id": [1, 2, 3],
            "product": ["pen", "mug", "pen"],
        }
    ),
    "products": pl.DataFrame(
        {
            "product": ["pen", "mug"],
            "price": [2, 8],
        }
    ),
}
''',
        "reference_solution": '''
def solve(sales, products):
    return sales.join(products, on="product", how="inner").select(
        "sale_id", "product", "price"
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_003",
        "difficulty": "easy",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "For every employee, look up their department name from `depts`. Keep "
            "all employees even if their department is not listed (those get a "
            "missing department name). Return `name` and `dept_name`. Order does "
            "not matter."
        ),
        "signature": "def solve(employees: pl.DataFrame, depts: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "employees": pl.DataFrame(
        {
            "name": ["a", "b", "c"],
            "dept_id": [1, 2, 9],
        }
    ),
    "depts": pl.DataFrame(
        {
            "dept_id": [1, 2],
            "dept_name": ["eng", "sales"],
        }
    ),
}
''',
        "reference_solution": '''
def solve(employees, depts):
    return employees.join(depts, on="dept_id", how="left").select(
        "name", "dept_name"
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_004",
        "difficulty": "easy",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "The orders table refers to a customer by `cust`, while the customers "
            "table calls the same thing `id`. Join them on that relationship and "
            "return `order_id` with the matching `name`. Order does not matter."
        ),
        "signature": "def solve(orders: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "orders": pl.DataFrame(
        {
            "order_id": [1, 2, 3],
            "cust": [10, 20, 10],
        }
    ),
    "customers": pl.DataFrame(
        {
            "id": [10, 20],
            "name": ["Ada", "Bo"],
        }
    ),
}
''',
        "reference_solution": '''
def solve(orders, customers):
    return orders.join(
        customers, left_on="cust", right_on="id", how="inner"
    ).select("order_id", "name")
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_005",
        "difficulty": "easy",
        "category": "join",
        "touches_2_0_change": True,  # anti-join; pandas reaches for merge(indicator=True)
        "instruction": (
            "Return the customers who have never placed an order. Match customers' "
            "`id` against the `customer_id` in `orders`. Return just the customer "
            "`id` and `name`. Order does not matter."
        ),
        "signature": "def solve(customers: pl.DataFrame, orders: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "customers": pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
        }
    ),
    "orders": pl.DataFrame(
        {"customer_id": [1, 1, 3]}
    ),
}
''',
        "reference_solution": '''
def solve(customers, orders):
    return customers.join(
        orders, left_on="id", right_on="customer_id", how="anti"
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_006",
        "difficulty": "medium",
        "category": "join",
        "touches_2_0_change": True,  # full outer join: 2.0 uses how="full" (was "outer")
        "instruction": (
            "Combine the January and February price lists so that every product "
            "appearing in either month is present once, with its `jan` and `feb` "
            "price (missing where a product was absent that month). Return "
            "`product`, `jan`, `feb`. Order does not matter."
        ),
        "signature": "def solve(jan: pl.DataFrame, feb: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "jan": pl.DataFrame(
        {
            "product": ["pen", "mug"],
            "jan": [2, 8],
        }
    ),
    "feb": pl.DataFrame(
        {
            "product": ["mug", "hat"],
            "feb": [9, 15],
        }
    ),
}
''',
        "reference_solution": '''
def solve(jan, feb):
    return jan.join(feb, on="product", how="full", coalesce=True).select(
        "product", "jan", "feb"
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_007",
        "difficulty": "medium",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "Each line item has a `qty`; look up the unit `price` for its "
            "`product` and compute the line `revenue` (qty times price). Return "
            "`product` and `revenue`. Keep only products present in the price "
            "table. Order does not matter."
        ),
        "signature": "def solve(items: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "items": pl.DataFrame(
        {
            "product": ["pen", "mug", "pen"],
            "qty": [3, 2, 1],
        }
    ),
    "prices": pl.DataFrame(
        {
            "product": ["pen", "mug"],
            "price": [2, 8],
        }
    ),
}
''',
        "reference_solution": '''
def solve(items, prices):
    return (
        items.join(prices, on="product", how="inner")
        .with_columns((pl.col("qty") * pl.col("price")).alias("revenue"))
        .select("product", "revenue")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_008",
        "difficulty": "medium",
        "category": "join",
        "touches_2_0_change": True,  # semi-join; not a native pandas operation
        "instruction": (
            "Return only the products that have been ordered at least once (appear "
            "in the `orders` table), keeping the product catalog columns `product` "
            "and `price`. Order does not matter."
        ),
        "signature": "def solve(catalog: pl.DataFrame, orders: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "catalog": pl.DataFrame(
        {
            "product": ["pen", "mug", "hat"],
            "price": [2, 8, 15],
        }
    ),
    "orders": pl.DataFrame(
        {"product": ["pen", "pen", "mug"]}
    ),
}
''',
        "reference_solution": '''
def solve(catalog, orders):
    return catalog.join(orders, on="product", how="semi")
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_009",
        "difficulty": "medium",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "Match actual sales to their targets using both `region` and "
            "`product` together, and return `region`, `product`, `units`, "
            "`target` for the matched rows. Order does not matter."
        ),
        "signature": "def solve(sales: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "sales": pl.DataFrame(
        {
            "region": ["n", "n", "s"],
            "product": ["x", "y", "x"],
            "units": [10, 20, 30],
        }
    ),
    "targets": pl.DataFrame(
        {
            "region": ["n", "s"],
            "product": ["x", "x"],
            "target": [8, 25],
        }
    ),
}
''',
        "reference_solution": '''
def solve(sales, targets):
    return sales.join(targets, on=["region", "product"], how="inner")
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_010",
        "difficulty": "medium",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "Both tables have a column called `value`. Join them on `id` (inner) "
            "and keep both values; name the right table's column `value_right`. "
            "Return `id`, `value`, `value_right`. Order does not matter."
        ),
        "signature": "def solve(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "left": pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    ),
    "right": pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [100, 200, 300],
        }
    ),
}
''',
        "reference_solution": '''
def solve(left, right):
    return left.join(right, on="id", how="inner", suffix="_right")
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_011",
        "difficulty": "hard",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "Each order has an `amount` and belongs to a customer; each customer "
            "lives in a `city`. Report the total `amount` per city. Return `city` "
            "and `amount`. Order does not matter."
        ),
        "signature": "def solve(orders: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "orders": pl.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
            "amount": [10, 20, 40, 5],
        }
    ),
    "customers": pl.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "city": ["NYC", "LA", "NYC"],
        }
    ),
}
''',
        "reference_solution": '''
def solve(orders, customers):
    return (
        orders.join(customers, on="customer_id", how="inner")
        .group_by("city")
        .agg(pl.col("amount").sum())
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_012",
        "difficulty": "hard",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "Each employee row has an `id`, a `name`, and a `manager_id` pointing "
            "at another employee's `id` (the CEO's manager_id is missing). For "
            "each employee return their `name` and their manager's name as "
            "`manager` (missing for the CEO). Order does not matter."
        ),
        "signature": "def solve(emp: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "emp": pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["CEO", "Ann", "Bob"],
            "manager_id": [None, 1, 2],
        }
    )
}
''',
        "reference_solution": '''
def solve(emp):
    managers = emp.select(
        pl.col("id").alias("manager_id"),
        pl.col("name").alias("manager"),
    )
    return emp.join(managers, on="manager_id", how="left").select("name", "manager")
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "join_013",
        "difficulty": "hard",
        "category": "join",
        "touches_2_0_change": False,
        "instruction": (
            "Look up each item's `price` from the price list. Items with no listed "
            "price should be treated as costing 0, not dropped. Return `item` and "
            "`price`. Order does not matter."
        ),
        "signature": "def solve(items: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "items": pl.DataFrame(
        {"item": ["a", "b", "c"]}
    ),
    "prices": pl.DataFrame(
        {
            "item": ["a", "c"],
            "price": [5, 9],
        }
    ),
}
''',
        "reference_solution": '''
def solve(items, prices):
    return items.join(prices, on="item", how="left").with_columns(
        pl.col("price").fill_null(0)
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
]
