"""A CPU mock model for proving the generate -> score pipeline without a GPU.

``MockModel`` returns a fixed list of canned completions aligned to the input
order (it ignores the prompt content). ``make_reference_mock`` builds a mock
that answers some tasks with their reference solution (which should score as
``pass``) and others with deliberately broken, pandas-style code (which should
score as ``pandas_leak``), so the full pipeline exercises both outcomes.
"""

from __future__ import annotations

import re

from ..bench.schema import Task
from .base import Generation

_SIG_RE = re.compile(r"def\s+solve\s*\(([^)]*)\)")


def solve_param_names(signature: str) -> list[str]:
    """Extract `solve`'s parameter names from a signature string.

    Strips type annotations and defaults: ``def solve(df: pl.DataFrame) -> ...``
    yields ``["df"]``.
    """

    m = _SIG_RE.search(signature)
    if not m:
        return []
    names = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        name = part.split(":", 1)[0].split("=", 1)[0].strip()
        if name and name != "self":
            names.append(name)
    return names


class MockModel:
    """Returns pre-baked completions in input order; never touches a GPU."""

    name = "mock"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def generate(
        self,
        messages_list: list[list[dict]],
        max_new_tokens: int = 768,
        batch_size: int = 8,
    ) -> list[Generation]:
        out: list[Generation] = []
        for i in range(len(messages_list)):
            text = self.responses[i] if i < len(self.responses) else ""
            out.append(
                Generation(
                    text=text,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_s=0.0,
                )
            )
        return out


def _fenced(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def make_reference_mock(tasks: list[Task]) -> MockModel:
    """Build a mock that alternates correct and broken answers.

    Even-indexed tasks get their reference solution (-> ``pass``); odd-indexed
    tasks get pandas-style broken code using the correct parameter names so it
    passes the signature check but fails at runtime (-> ``pandas_leak``).
    """

    responses: list[str] = []
    for i, task in enumerate(tasks):
        if i % 2 == 0:
            responses.append(_fenced(task.reference_solution))
        else:
            params = solve_param_names(task.signature) or ["df"]
            broken = (
                f"def solve({', '.join(params)}):\n"
                f"    return {params[0]}.groupby('x').sum()"
            )
            responses.append(_fenced(broken))
    return MockModel(responses)
