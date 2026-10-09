from polars_llm.bench.schema import Task
from polars_llm.prompts import SYSTEM_PROMPT, build_messages, build_user_prompt

TASK = Task.from_dict(
    {
        "id": "t1",
        "difficulty": "easy",
        "category": "group_by",
        "instruction": "Total amount per category.",
        "setup_code": 'inputs = {"df": pl.DataFrame({"category": ["a"], "amount": [1]})}',
        "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
        "reference_solution": "def solve(df): return df",
    }
)


def test_messages_shape_and_roles():
    msgs = build_messages(TASK)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_user_prompt_includes_task_parts():
    user = build_user_prompt(TASK)
    assert TASK.instruction in user
    assert TASK.setup_code in user
    assert TASK.signature in user
    # Asks for a single python block.
    assert "```python" in user


def test_system_prompt_mentions_polars_2_and_forbids_pandas():
    assert "2.0" in SYSTEM_PROMPT
    assert "pandas" in SYSTEM_PROMPT.lower()
    assert "solve" in SYSTEM_PROMPT


def test_context_block_present_only_when_given():
    assert "documentation" not in build_user_prompt(TASK, context=None)
    assert "documentation" not in build_user_prompt(TASK, context="   ")
    withctx = build_user_prompt(TASK, context="pl.DataFrame.group_by(...) aggregates rows.")
    assert "documentation" in withctx
    assert "group_by" in withctx
