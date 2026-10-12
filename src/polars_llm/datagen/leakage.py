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

``instruction_and_api_set`` (after the instructions are written; ``build_sft.py``)
    A training request is dropped only if BOTH (a) its embedding cosine to a
    benchmark instruction is >= a **calibrated** threshold, and (b) its
    solution has exactly the same set of non-core API names as that benchmark
    task. The threshold is calibrated on the benchmark itself
    (``calibrate_instruction_threshold``): the nearest-neighbour cosine between
    *distinct* benchmark tasks, which are by construction not leaks of each
    other; the threshold is the maximum of that distribution. A fixed 0.85 was
    meaningless: the embedder puts almost any short Polars request at
    0.85-0.93 to some benchmark task (the first real run dropped 2,036 of 2,137).
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass

from ..rag.eval_retrieval import CORE_NAMES

SOLUTION_SIM_THRESHOLD = 0.85
MIN_NON_CORE_NAMES = 2


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


def nearest_neighbour_cosines(vecs):
    """For each row of L2-normalised ``vecs``: (cosine, index) of its closest *other* row."""

    import numpy as np

    sims = np.asarray(vecs) @ np.asarray(vecs).T
    np.fill_diagonal(sims, -np.inf)
    idx = sims.argmax(axis=1)
    return sims[np.arange(len(sims)), idx], idx


def calibrate_instruction_threshold(bench_vecs, bench_ids: list[str]) -> dict:
    """Threshold = max nearest-neighbour cosine between distinct benchmark tasks,
    plus the distribution it comes from and the closest benchmark pairs."""

    import numpy as np

    nn, idx = nearest_neighbour_cosines(bench_vecs)
    pct = {f"p{q}": round(float(np.percentile(nn, q)), 4) for q in (50, 90, 95, 99)}
    order = np.argsort(-nn)
    pairs, seen = [], set()
    for i in order:
        key = frozenset((bench_ids[i], bench_ids[idx[i]]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"a": bench_ids[i], "b": bench_ids[idx[i]], "cosine": round(float(nn[i]), 4)})
        if len(pairs) == 5:
            break
    return {
        "rule": "max nearest-neighbour cosine between distinct benchmark tasks",
        "threshold": round(float(nn.max()), 4),
        "n_benchmark_tasks": len(bench_ids),
        "nn_cosine_distribution": {"min": round(float(nn.min()), 4), "mean": round(float(nn.mean()), 4), **pct,
                                   "max": round(float(nn.max()), 4)},
        "closest_benchmark_pairs": pairs,
    }


def instruction_api_leaks(instr_vecs, bench_vecs, instr_api_names, bench_api_names, threshold: float) -> list[dict]:
    """Per training request: its nearest benchmark task and cosine, and ``leak``
    = (benchmark index, cosine) of the most similar benchmark task that is
    both >= ``threshold`` AND has the same non-core API set, else None."""

    import numpy as np

    sims = np.asarray(instr_vecs) @ np.asarray(bench_vecs).T if len(instr_vecs) else np.zeros((0, len(bench_vecs)))
    bench_sets = [non_core(n) for n in bench_api_names]
    out = []
    for i, names in enumerate(instr_api_names):
        mine = non_core(names)
        row = sims[i]
        nearest = int(row.argmax())
        leak = None
        for j in np.argsort(-row):
            if row[j] < threshold:
                break
            if bench_sets[j] == mine:
                leak = (int(j), float(row[j]))
                break
        out.append({"nearest": nearest, "cosine": float(row[nearest]), "leak": leak})
    return out
