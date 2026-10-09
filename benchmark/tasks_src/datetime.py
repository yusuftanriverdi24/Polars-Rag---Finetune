"""datetime tasks: the `.dt` namespace and temporal parsing.

Date/datetime inputs are built from `datetime` objects so Polars infers Date /
Datetime dtypes. Reference solutions compare against `pl.date(...)` literals to
avoid depending on names imported in setup.
"""

TASKS = [
    {
        "id": "datetime_001",
        "difficulty": "medium",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "The `ts` column holds date strings in 'YYYY-MM-DD' format. Parse it "
            "to a Date, then return rows from the year 2023, with an added column "
            "`month` (integer month). Keep columns `ts`, `month` and preserve order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"ts": ["2022-12-31", "2023-01-15", "2023-07-04", "2024-02-01"]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return (
        df.with_columns(pl.col("ts").str.to_date("%Y-%m-%d"))
        .filter(pl.col("ts").dt.year() == 2023)
        .with_columns(pl.col("ts").dt.month().alias("month"))
        .select("ts", "month")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_002",
        "difficulty": "easy",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `year` holding the calendar year of each date `d`. Keep "
            "all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "d": [
                datetime.date(2023, 1, 15),
                datetime.date(2021, 6, 1),
                datetime.date(2023, 12, 31),
            ]
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("d").dt.year().alias("year"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_003",
        "difficulty": "easy",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `weekday` holding the ISO weekday number of each date "
            "`d` (Monday is 1, Sunday is 7). Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "d": [
                datetime.date(2023, 1, 2),   # Monday
                datetime.date(2023, 1, 7),   # Saturday
                datetime.date(2023, 1, 8),   # Sunday
            ]
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("d").dt.weekday().alias("weekday"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_004",
        "difficulty": "easy",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "Keep only the rows whose date `d` falls in the first half of 2023 "
            "(between 1 January and 30 June 2023, inclusive)."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "d": [
                datetime.date(2022, 12, 1),
                datetime.date(2023, 3, 15),
                datetime.date(2023, 6, 30),
                datetime.date(2023, 7, 1),
            ]
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(
        pl.col("d").is_between(pl.date(2023, 1, 1), pl.date(2023, 6, 30))
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_005",
        "difficulty": "easy",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `s` formatting each date `d` as a string like "
            "'2023/01/15' (year/month/day, zero-padded). Keep all rows and the "
            "original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "d": [
                datetime.date(2023, 1, 15),
                datetime.date(2023, 11, 5),
            ]
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("d").dt.strftime("%Y/%m/%d").alias("s"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_006",
        "difficulty": "medium",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "The `ts` column holds timestamps like '2023-01-01 09:30:00'. Parse "
            "them and add a column `hour` with the hour of day (0-23). Keep all "
            "rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "ts": [
                "2023-01-01 09:30:00",
                "2023-01-01 14:05:00",
                "2023-01-02 23:59:00",
            ]
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("ts").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.hour().alias("hour")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_007",
        "difficulty": "medium",
        "category": "datetime",
        "touches_2_0_change": True,  # duration -> dt.total_days() (dt.days() was removed)
        "instruction": (
            "Each row has a `start` and `end` date. Add a column `days` holding "
            "the number of whole days from `start` to `end`. Keep all rows and "
            "the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "start": [datetime.date(2023, 1, 1), datetime.date(2023, 3, 10)],
            "end": [datetime.date(2023, 1, 8), datetime.date(2023, 3, 12)],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        (pl.col("end") - pl.col("start")).dt.total_days().alias("days")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_008",
        "difficulty": "medium",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `month_start` holding, for each date `d`, the first day "
            "of that date's month. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "d": [
                datetime.date(2023, 1, 15),
                datetime.date(2023, 2, 28),
            ]
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("d").dt.truncate("1mo").alias("month_start")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_009",
        "difficulty": "medium",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `due` that is exactly 7 days after each date `d`. Keep "
            "all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "d": [
                datetime.date(2023, 1, 1),
                datetime.date(2023, 1, 28),
            ]
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("d").dt.offset_by("7d").alias("due"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_010",
        "difficulty": "medium",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "Keep only the rows whose date `d` falls on a weekend (Saturday or "
            "Sunday)."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "d": [
                datetime.date(2023, 1, 6),   # Friday
                datetime.date(2023, 1, 7),   # Saturday
                datetime.date(2023, 1, 8),   # Sunday
                datetime.date(2023, 1, 9),   # Monday
            ]
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("d").dt.weekday() >= 6)
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_011",
        "difficulty": "hard",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "Each row is a purchase with a date `d` and an `amount`. Report the "
            "total `amount` per calendar month, labelling each month by its first "
            "day. Return `month` and `amount`. Order does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "d": [
                datetime.date(2023, 1, 5),
                datetime.date(2023, 1, 20),
                datetime.date(2023, 2, 2),
            ],
            "amount": [10, 15, 7],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by(
        pl.col("d").dt.truncate("1mo").alias("month")
    ).agg(pl.col("amount").sum())
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "datetime_012",
        "difficulty": "hard",
        "category": "datetime",
        "touches_2_0_change": True,  # shift().over() + dt.total_days() (removed dt.days)
        "instruction": (
            "For each `user`, the rows are in visit-date order. Add a column "
            "`gap_days` holding the number of days since that user's previous "
            "visit (missing for each user's first visit). Keep the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
import datetime
inputs = {
    "df": pl.DataFrame(
        {
            "user": ["a", "a", "a", "b", "b"],
            "d": [
                datetime.date(2023, 1, 1),
                datetime.date(2023, 1, 4),
                datetime.date(2023, 1, 10),
                datetime.date(2023, 2, 1),
                datetime.date(2023, 2, 15),
            ],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        (pl.col("d") - pl.col("d").shift(1).over("user"))
        .dt.total_days()
        .alias("gap_days")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "datetime_013",
        "difficulty": "hard",
        "category": "datetime",
        "touches_2_0_change": False,
        "instruction": (
            "The year, month and day are stored in three separate integer "
            "columns. Add a column `weekday` holding the ISO weekday number "
            "(Monday 1 ... Sunday 7) of the date they describe. Keep all rows and "
            "the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "y": [2023, 2023, 2023],
            "m": [1, 1, 1],
            "day": [2, 7, 8],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.date(pl.col("y"), pl.col("m"), pl.col("day")).dt.weekday().alias("weekday")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
