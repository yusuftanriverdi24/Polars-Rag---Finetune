"""models: inference wrappers (HF/Unsloth, mock) (Phase 2+).

``HFModel`` is intentionally *not* imported here so that importing
``polars_llm.models`` on a CPU-only machine does not pull in ``unsloth`` /
``torch``. Import it explicitly: ``from polars_llm.models.hf import HFModel``.
"""

from .base import Generation, Model
from .mock import MockModel, make_reference_mock, solve_param_names

__all__ = [
    "Generation",
    "Model",
    "MockModel",
    "make_reference_mock",
    "solve_param_names",
]
