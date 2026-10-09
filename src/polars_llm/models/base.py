"""Shared types for model wrappers.

Every model wrapper exposes the same ``generate(messages_list, ...)`` method
returning a list of :class:`Generation` aligned to the input order, so the
generation script is identical for the real HF model and the CPU mock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Generation:
    """One model completion plus its cost metadata."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class Model(Protocol):
    """Structural interface implemented by HFModel and MockModel."""

    name: str

    def generate(
        self,
        messages_list: list[list[dict]],
        max_new_tokens: int = 768,
        batch_size: int = 8,
    ) -> list[Generation]: ...
