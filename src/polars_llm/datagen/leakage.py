"""Leakage control: keep training examples away from the frozen benchmark.

A candidate is dropped when it is *too close* to any benchmark task by one of
(thresholds fixed before looking at the data, PLAN.md):

``solution``
    Normalized solutions (imports and type annotations removed, parameters and
    local variables renamed ``v0, v1, ...``, ``ast.unparse`` formatting) have a
    ``difflib`` similarity of the bodies >= ``SOLUTION_SIM_THRESHOLD``; or their
    *structure* (string/number literals masked as well) is identical.
``api_set``
    The exact same set of non-core Polars API names (as resolved by
    ``eval_retrieval.solution_api_names``).

All three apply only when the candidate uses at least ``MIN_NON_CORE_NAMES``
distinct **non-core** API names (core = ``col``, ``select``, ``with_columns``,
... as in the retrieval eval). A single-API pattern such as
``df.with_columns(pl.col(x).str.to_lowercase())`` is general Polars knowledge,
not a copy of a benchmark task, and short bodies that share only the core
boilerplate (``df.with_columns(pl.col(..)...)``) would otherwise look alike.

``instruction``
    Cosine similarity of instruction embeddings >= ``INSTRUCTION_SIM_THRESHOLD``
    (checked after the instructions are written; see ``build_sft.py``).
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass

from ..rag.eval_retrieval import CORE_NAMES

SOLUTION_SIM_THRESHOLD = 0.85
MIN_NON_CORE_NAMES = 2
INSTRUCTION_SIM_THRESHOLD = 0.85


class _Normalizer(ast.NodeTransformer):
    def __init__(self, mask_literals: bool):
        self.mask = mask_literals
        self.names: dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name not in self.names:
            self.names[name] = f"v{len(self.names)}"
        return self.names[name]

    def visit_FunctionDef(self, node):
        node.returns = None
        node.decorator_list = []
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            a.annotation = None
            a.arg = self._rename(a.arg)
        # Drop a docstring.
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if node.id in self.names or isinstance(node.ctx, ast.Store):
            node.id = self._rename(node.id)
        return node

    def visit_Constant(self, node):
        if self.mask and isinstance(node.value, str):
            return ast.Constant("S")
        if self.mask and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.Constant(0)
        return node


def normalize_solution(code: str, *, mask_literals: bool = False) -> str:
    """Canonical form of a ``solve`` solution for near-duplicate detection."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code.strip()
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    tree = _Normalizer(mask_literals).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _body(normalized: str) -> str:
    """Drop the ``def solve(...):`` header line(s) so boilerplate doesn't inflate similarity."""

    return normalized.split(":\n", 1)[-1] if normalized.startswith("def ") else normalized


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass(frozen=True)
class BenchRef:
    id: str
    normalized: str
    structure: str
    api_names: frozenset[str]


def bench_refs(tasks, needed_by_task: dict[str, list[str]]) -> list[BenchRef]:
    return [
        BenchRef(
            t.id,
            normalize_solution(t.reference_solution),
            normalize_solution(t.reference_solution, mask_literals=True),
            non_core(needed_by_task.get(t.id, [])),
        )
        for t in tasks
    ]


def non_core(api_names) -> frozenset[str]:
    return frozenset(n for n in api_names if n not in CORE_NAMES)


def solution_leak(solution: str, api_names, refs: list[BenchRef]) -> tuple[str, str] | None:
    """(reason, benchmark task id) of the first match, or None."""

    names = non_core(api_names)
    if len(names) < MIN_NON_CORE_NAMES:
        return None
    norm = _body(normalize_solution(solution))
    struct = normalize_solution(solution, mask_literals=True)
    for r in refs:
        ref_norm = _body(r.normalized)
        # Cheap length bound before the quadratic ratio.
        if min(len(norm), len(ref_norm)) / max(len(norm), len(ref_norm), 1) >= SOLUTION_SIM_THRESHOLD:
            if similarity(norm, ref_norm) >= SOLUTION_SIM_THRESHOLD:
                return "solution", r.id
        if struct == r.structure:
            return "solution_structure", r.id
        if names == r.api_names:
            return "api_set", r.id
    return None


def instruction_leaks(instr_vecs, bench_vecs, threshold: float = INSTRUCTION_SIM_THRESHOLD) -> list[tuple[int, float] | None]:
    """For each instruction (rows of L2-normalised ``instr_vecs``), the index and
    cosine of the closest benchmark instruction if it reaches ``threshold``."""

    import numpy as np

    if len(instr_vecs) == 0:
        return []
    sims = np.asarray(instr_vecs) @ np.asarray(bench_vecs).T
    best = sims.argmax(axis=1)
    out: list[tuple[int, float] | None] = []
    for i, j in enumerate(best):
        s = float(sims[i, j])
        out.append((int(j), s) if s >= threshold else None)
    return out
