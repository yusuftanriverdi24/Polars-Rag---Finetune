"""Benchmark tooling: task schema, sandbox executor, comparator, classifier,
metrics and runner (PLAN.md Section 4).
"""

from .classify import (
    DEPRECATED_API,
    HALLUCINATED_API,
    PANDAS_LEAK,
    PASS,
    RUNTIME_ERROR,
    SYNTAX_ERROR,
    TIMEOUT,
    WRONG_RESULT,
    Classification,
    classify,
    detect_pandas_leak,
)
from .compare import Comparison, compare_frames
from .executor import (
    DEFAULT_TIMEOUT_S,
    ExecError,
    ExecutionResult,
    compute_expected,
    run_solution,
)
from .extract import extract_code
from .metrics import aggregate
from .runner import score
from .schema import CheckFlags, Task, load_tasks, tasks_by_id

__all__ = [
    "CheckFlags",
    "Task",
    "load_tasks",
    "tasks_by_id",
    "extract_code",
    "run_solution",
    "compute_expected",
    "ExecutionResult",
    "ExecError",
    "DEFAULT_TIMEOUT_S",
    "compare_frames",
    "Comparison",
    "classify",
    "Classification",
    "detect_pandas_leak",
    "aggregate",
    "score",
    "PASS",
    "SYNTAX_ERROR",
    "PANDAS_LEAK",
    "HALLUCINATED_API",
    "DEPRECATED_API",
    "RUNTIME_ERROR",
    "WRONG_RESULT",
    "TIMEOUT",
]
