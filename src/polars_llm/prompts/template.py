"""The single shared prompt template used by every experiment arm.

The only thing that differs across arms is the optional ``context`` block:

- Arms A (baseline) and C (fine-tuned): ``context=None`` -> no doc block.
- Arms B (RAG) and D (hybrid): ``context`` holds the retrieved Polars 2.0 docs.

``build_messages`` returns a chat ``[system, user]`` message list. The model
wrapper is responsible for rendering it through the tokenizer's chat template
(with thinking disabled for Qwen3.5). The user turn shows the model the task,
the exact `setup_code` that builds the inputs (so it knows the input variable
names, columns and dtypes), and the required signature; it asks for a single
```python block defining ``solve``.
"""

from __future__ import annotations

from ..bench.schema import Task

SYSTEM_PROMPT = (
    "You are an expert Python data engineer who writes correct, idiomatic "
    "Polars code for Polars version 2.0.\n"
    "Rules:\n"
    "- Use only the Polars 2.0 API. `import polars as pl` is already available.\n"
    "- Do not use pandas or any pandas idioms (no `pd.`, `.loc`, `.iloc`, "
    "`inplace=`, `.groupby(`, `.merge(`, `.sort_values(`).\n"
    "- Do not use APIs removed in Polars 2.0 (e.g. `groupby`, `apply`, "
    "`with_row_count`, `melt`, `pivot(columns=...)`); use their current "
    "equivalents (`group_by`, `map_elements`/expressions, `with_row_index`, "
    "`unpivot`, `pivot(on=...)`).\n"
    "- Reply with exactly one ```python code block that defines the `solve` "
    "function with the given signature, and nothing else — no prose, no "
    "explanation, no example usage."
)

_CONTEXT_TEMPLATE = (
    "Here is relevant Polars 2.0 documentation. Use it to write correct, "
    "up-to-date code:\n\n"
    "{context}\n\n"
    "----------------------------------------\n\n"
)

_USER_TEMPLATE = (
    "{context_block}"
    "Write a Polars function for the following task.\n\n"
    "Task:\n{instruction}\n\n"
    "The inputs are created by this setup code (your function receives these "
    "variables as arguments):\n"
    "```python\n{setup_code}\n```\n\n"
    "Complete exactly this function:\n"
    "```python\n{signature}\n    ...\n```\n\n"
    "Return a single ```python block defining `solve` with that exact "
    "signature."
)


def build_user_prompt(task: Task, context: str | None = None) -> str:
    """Render the user turn for a task, optionally with a retrieved-doc block."""

    context_block = ""
    if context and context.strip():
        context_block = _CONTEXT_TEMPLATE.format(context=context.strip())
    return _USER_TEMPLATE.format(
        context_block=context_block,
        instruction=task.instruction.strip(),
        setup_code=task.setup_code.strip(),
        signature=task.signature.strip(),
    )


def build_messages(task: Task, context: str | None = None) -> list[dict]:
    """Build the chat message list (`system` + `user`) for a task."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task, context=context)},
    ]
