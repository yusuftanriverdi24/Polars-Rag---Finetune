"""join tasks: combining frames on keys."""

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
]
