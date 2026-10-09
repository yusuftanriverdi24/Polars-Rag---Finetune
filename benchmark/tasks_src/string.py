"""string tasks: the `.str` namespace."""

TASKS = [
    {
        "id": "string_001",
        "difficulty": "easy",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `upper` containing the uppercase form of `name`, and "
            "keep only rows whose `name` contains the letter 'a' (case-insensitive)."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"name": ["Ada", "Bob", "cora", "Mel"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("name").str.to_uppercase().alias("upper")
    ).filter(pl.col("name").str.contains("(?i)a"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_002",
        "difficulty": "easy",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Replace the `city` column with its lowercase form. Keep all rows and "
            "the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"city": ["NYC", "London", "Paris"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("city").str.to_lowercase())
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_003",
        "difficulty": "easy",
        "category": "string",
        "touches_2_0_change": True,  # str.len_chars (the old str.lengths was removed)
        "instruction": (
            "Add a column `n` giving the number of characters in each `word`. Keep "
            "all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"word": ["hi", "hello", "hey"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("word").str.len_chars().alias("n"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_004",
        "difficulty": "easy",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Keep only the rows whose `sku` begins with the prefix \"AB\"."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"sku": ["AB-1", "CD-2", "AB-3", "EF-4"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.filter(pl.col("sku").str.starts_with("AB"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_005",
        "difficulty": "easy",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "In the `path` column, replace every forward slash '/' with a dash "
            "'-'. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"path": ["a/b/c", "x/y", "z"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("path").str.replace_all("/", "-", literal=True)
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_006",
        "difficulty": "easy",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `full_name` that joins `first` and `last` with a single "
            "space between them. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "first": ["Ada", "Bo"],
            "last": ["Lovelace", "Peep"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.concat_str(["first", "last"], separator=" ").alias("full_name")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_007",
        "difficulty": "easy",
        "category": "string",
        "touches_2_0_change": True,  # str.strip_chars (the old str.strip was renamed)
        "instruction": (
            "The `name` values have stray leading/trailing spaces. Return the "
            "`name` column with that surrounding whitespace removed. Keep all rows "
            "and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"name": ["  Ada ", "Bo  ", " Cy"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("name").str.strip_chars())
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_008",
        "difficulty": "medium",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Each `code` looks like 'ABC-123'. Add a column `digits` containing "
            "just the numeric part as a string (the part after the dash). Keep all "
            "rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"code": ["ABC-123", "XY-7", "ZZZ-900"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("code").str.extract(r"-(\\d+)$", 1).alias("digits")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_009",
        "difficulty": "medium",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Each `ym` value looks like '2023-07'. Add a column `year` holding the "
            "part before the dash (as a string). Keep all rows and the original "
            "order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"ym": ["2023-07", "2021-12", "2020-01"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("ym").str.split("-").list.get(0).alias("year")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_010",
        "difficulty": "medium",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `n_a` counting how many times the lowercase letter 'a' "
            "appears in each `word`. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"word": ["banana", "apple", "kiwi"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("word").str.count_matches("a").alias("n_a")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_011",
        "difficulty": "medium",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Left-pad each `id` string with zeros so it is exactly 4 characters "
            "wide (e.g. '7' becomes '0007'). Return the padded `id`. Keep the "
            "original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"id": ["7", "42", "1234"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("id").str.zfill(4))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_012",
        "difficulty": "medium",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `prefix` containing the first three characters of each "
            "`token`. Keep all rows and the original order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame({"token": ["alpha", "beta", "x"]})
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(pl.col("token").str.slice(0, 3).alias("prefix"))
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
    {
        "id": "string_013",
        "difficulty": "hard",
        "category": "string",
        "touches_2_0_change": True,  # str.join aggregation (was str.concat; default delim changed)
        "instruction": (
            "For each `user`, build a single string listing that user's `tag` "
            "values in alphabetical order, separated by \", \". Return `user` and "
            "the string as `tags`. Order of users does not matter."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {
            "user": ["u1", "u1", "u2", "u2"],
            "tag": ["z", "a", "m", "b"],
        }
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.group_by("user").agg(
        pl.col("tag").sort().str.join(", ").alias("tags")
    )
''',
        "check": {"ignore_row_order": True, "ignore_column_order": False},
    },
    {
        "id": "string_014",
        "difficulty": "hard",
        "category": "string",
        "touches_2_0_change": False,
        "instruction": (
            "Add a column `n_words` counting the words in each `sentence`, where "
            "words are separated by single spaces. Keep all rows and the original "
            "order."
        ),
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "setup_code": '''
inputs = {
    "df": pl.DataFrame(
        {"sentence": ["hello world", "one two three", "solo"]}
    )
}
''',
        "reference_solution": '''
def solve(df):
    return df.with_columns(
        pl.col("sentence").str.split(" ").list.len().alias("n_words")
    )
''',
        "check": {"ignore_row_order": False, "ignore_column_order": False},
    },
]
