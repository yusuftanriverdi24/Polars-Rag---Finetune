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

import hashlib

from ..bench.schema import Task

# Deliberately generic: it names no specific Polars/pandas APIs, because listing
# removed APIs and their 2.0 replacements would leak exactly what the benchmark
# tests and inflate the baseline (see PLAN.md decisions log).
SYSTEM_PROMPT = (
    "You are an expert Python data engineer. Write correct Polars code for "
    "Polars version 2.0. `import polars as pl` is already available.\n"
    "Do not use pandas.\n"
    "Reply with exactly one ```python code block that defines the `solve` "
    "function with the given signature, and nothing else — no prose."
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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_fingerprint() -> dict:
    """Hashes + full text of the prompt template, recorded with every run.

    ``system_prompt_sha256`` identifies the system prompt alone;
    ``template_sha256`` covers the system prompt, user template and context
    template together (any change to what the model is shown changes it).
    """

    template_text = "\n\x1e\n".join([SYSTEM_PROMPT, _USER_TEMPLATE, _CONTEXT_TEMPLATE])
    return {
        "system_prompt_sha256": _sha256(SYSTEM_PROMPT),
        "template_sha256": _sha256(template_text),
        "system_prompt": SYSTEM_PROMPT,
        "user_template": _USER_TEMPLATE,
        "context_template": _CONTEXT_TEMPLATE,
    }


def build_messages(task: Task, context: str | None = None) -> list[dict]:
    """Build the chat message list (`system` + `user`) for a task."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task, context=context)},
    ]
