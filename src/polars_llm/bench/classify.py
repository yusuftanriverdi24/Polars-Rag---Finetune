"""Error taxonomy classifier (PLAN.md Section 4.3).

Maps an execution outcome (+ the candidate source and the output comparison)
to a single ``status`` plus a ``deprecated_api`` flag that is recorded
independently (a correct result that nevertheless raised a deprecation warning
is still flagged).

Statuses::

    pass             solve ran and the output matched the reference
    syntax_error     the code did not compile
    pandas_leak      the code used pandas idioms (pd., .loc, .iloc, inplace=, .groupby()
    hallucinated_api nonexistent attribute / function / argument (AttributeError,
                     NameError, unexpected-keyword TypeError)
    deprecated_api   used an API removed/deprecated in the pinned version
                     (AttributeRemovedError / ArgumentRemovedError / "was removed"
                     message), or a deprecation warning was raised
    runtime_error    any other exception at run time
    wrong_result     ran cleanly but the output did not match
    timeout          exceeded the sandbox time limit

Priority for a failing run: timeout > syntax_error > pandas_leak >
deprecated_api > hallucinated_api > runtime_error. pandas_leak outranks the
others because in Polars 2.0 pandas idioms surface as generic
``AttributeError``/``TypeError``; the static signal is the more informative
root cause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .compare import Comparison
from .executor import ExecutionResult

# Status constants.
PASS = "pass"
SYNTAX_ERROR = "syntax_error"
PANDAS_LEAK = "pandas_leak"
HALLUCINATED_API = "hallucinated_api"
DEPRECATED_API = "deprecated_api"
RUNTIME_ERROR = "runtime_error"
WRONG_RESULT = "wrong_result"
TIMEOUT = "timeout"

ERROR_STATUSES = frozenset(
    {
        SYNTAX_ERROR,
        PANDAS_LEAK,
        HALLUCINATED_API,
        DEPRECATED_API,
        RUNTIME_ERROR,
        WRONG_RESULT,
        TIMEOUT,
    }
)

# Pandas idioms that should never appear in idiomatic Polars code. The first
# five mirror the examples in PLAN.md Section 4.3; the rest are unambiguous
# pandas-only method names that don't exist on Polars frames.
_PANDAS_PATTERNS = [
    re.compile(r"\bimport\s+pandas\b"),
    re.compile(r"\bpd\."),
    re.compile(r"\.loc\["),
    re.compile(r"\.iloc\["),
    re.compile(r"\binplace\s*="),
    re.compile(r"\.groupby\("),
    re.compile(r"\.sort_values\("),
    re.compile(r"\.reset_index\("),
    re.compile(r"\.merge\("),
]

_SYNTAX_TYPES = {"SyntaxError", "IndentationError", "TabError"}
_REMOVED_TYPES = {"AttributeRemovedError", "ArgumentRemovedError"}
_HALLUCINATION_TYPES = {"AttributeError", "NameError"}
_DEPRECATION_MESSAGE_RE = re.compile(
    r"was removed in version|has been removed|is deprecated|deprecated since|"
    r"has been renamed|renamed to",
    re.IGNORECASE,
)
_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument|got an unexpected")


def detect_pandas_leak(code: str) -> bool:
    """True if the source contains a pandas idiom."""

    return any(pat.search(code) for pat in _PANDAS_PATTERNS)


def _is_removed_or_deprecated_error(error) -> bool:
    names = set(error.mro) | {error.type}
    if names & _REMOVED_TYPES:
        return True
    return bool(_DEPRECATION_MESSAGE_RE.search(error.message))


def _is_hallucinated_api(error) -> bool:
    names = set(error.mro) | {error.type}
    if names & _HALLUCINATION_TYPES:
        return True
    if error.type == "TypeError" and _UNEXPECTED_KWARG_RE.search(error.message):
        return True
    return False


@dataclass
class Classification:
    status: str
    deprecated_api: bool
    pandas_leak_detected: bool
    reason: str
    warnings: list[dict] = field(default_factory=list)


def classify(
    *,
    code: str,
    exec_result: ExecutionResult,
    comparison: "Comparison | None",
) -> Classification:
    """Classify one task attempt into the Section 4.3 taxonomy."""

    pandas_leak_detected = detect_pandas_leak(code)
    deprecation_warnings = exec_result.deprecation_warnings
    deprecated_api = bool(deprecation_warnings)

    def result(status: str, reason: str) -> Classification:
        return Classification(
            status=status,
            deprecated_api=deprecated_api or status == DEPRECATED_API,
            pandas_leak_detected=pandas_leak_detected,
            reason=reason,
            warnings=exec_result.warnings,
        )

    if exec_result.timed_out:
        return result(TIMEOUT, "execution exceeded the sandbox time limit")

    if exec_result.error is not None:
        error = exec_result.error
        names = set(error.mro) | {error.type}

        if names & _SYNTAX_TYPES:
            return result(SYNTAX_ERROR, f"{error.type}: {error.message}")
        if pandas_leak_detected:
            return result(PANDAS_LEAK, f"pandas idiom in code; raised {error.type}")
        if _is_removed_or_deprecated_error(error):
            return result(DEPRECATED_API, f"{error.type}: {error.message}")
        if _is_hallucinated_api(error):
            return result(HALLUCINATED_API, f"{error.type}: {error.message}")
        return result(RUNTIME_ERROR, f"{error.type}: {error.message}")

    # No error: the attempt ran to completion.
    if comparison is None:
        # Should not happen for a successful run, but guard defensively.
        return result(RUNTIME_ERROR, "ran without a comparison to the reference")
    if comparison.equal:
        return result(PASS, "output matched the reference")
    return result(WRONG_RESULT, comparison.detail or "output did not match the reference")
