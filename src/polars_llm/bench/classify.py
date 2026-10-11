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
                     message, or an error message pointing to a replacement API,
                     e.g. "should be replaced with `how='full'`", "please use
                     `pl.len()` instead"), or a deprecation warning was raised
    runtime_error    any other exception at run time
    wrong_result     ran cleanly but the output did not match
    timeout          exceeded the sandbox time limit

Priority for a failing run: timeout > syntax_error > pandas_leak >
deprecated_api > hallucinated_api > runtime_error. pandas_leak outranks the
others because in Polars 2.0 pandas idioms surface as generic
``AttributeError``/``TypeError``; the static signal is the more informative
root cause.

Secondary tag (does not change ``status``)::

    pandas_style_name  a ``hallucinated_api`` failure whose attempted name is a
                       pandas idiom: a pandas accessor method that Polars lacks
                       (``.str.strip``, ``.str.len``, ``.str.pad``,
                       ``.str.count``, ``.dt.day_name``, ``.dt.days``), a pandas
                       keyword (``suffixes=``), a pandas Series/DataFrame
                       method (``.astype``, ``.fillna``, ``.rename`` on an
                       expression), or a pandas *property*-style access of what
                       is a method in Polars (``.dt.year`` without parentheses).
                       Recorded as the ``pandas_style_name`` field (the name, or
                       ``None``).
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
    r"has been renamed|renamed to|"
    # Messages that point the caller to a replacement API, e.g.
    #   "use of `how='outer'` should be replaced with `how='full'`"
    #   "`pl.count()` takes at least one argument. ... please use `pl.len()` instead"
    r"should be replaced (?:with|by)|\buse\b[^\n]{0,60}?\binstead\b",
    re.IGNORECASE,
)
_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument|got an unexpected")

# --- pandas_style_name secondary tag ------------------------------------------

# pandas `.str` accessor methods with no Polars `.str` method of that name.
PANDAS_STR_NAMES = frozenset(
    {
        "len", "strip", "lstrip", "rstrip", "pad", "count", "upper", "lower",
        "title", "capitalize", "cat", "center", "ljust", "rjust", "get",
        "isdigit", "isalpha", "isalnum", "isnumeric", "isspace", "isupper",
        "islower", "startswith", "endswith", "findall", "rsplit", "repeat",
    }
)  # fmt: skip
# pandas `.dt` accessor methods/properties with no Polars `.dt` method of that name.
PANDAS_DT_NAMES = frozenset(
    {
        "day_name", "month_name", "dayofweek", "dayofyear", "weekofyear",
        "days", "seconds", "microseconds", "nanoseconds", "daysinmonth",
        "is_month_end", "is_month_start", "is_year_end", "is_year_start",
        "to_pydatetime", "tz_localize", "tz_convert", "normalize", "floor",
        "ceil",
    }
)  # fmt: skip
# pandas Series/DataFrame methods that don't exist on Polars expressions/frames.
PANDAS_OBJECT_NAMES = frozenset(
    {
        "astype", "isna", "isnull", "notna", "notnull", "fillna", "dropna",
        "nunique", "idxmax", "idxmin", "cumsum", "cumprod", "cummax", "cummin",
        "cumcount", "transform", "rename", "iterrows", "itertuples", "assign",
        "query", "applymap", "tolist", "nlargest", "nsmallest", "set_index",
        "droplevel",
    }
)  # fmt: skip
# pandas keyword arguments (e.g. `merge(suffixes=...)`, `sort_values(ascending=)`).
PANDAS_KWARGS = frozenset(
    {"suffixes", "ascending", "axis", "ignore_index", "left_index", "right_index", "skipna"}
)
# Polars *methods* that pandas exposes as *properties* (`s.dt.year`, no call).
PANDAS_PROPERTY_NAMES = frozenset(
    {
        "year", "month", "day", "hour", "minute", "second", "microsecond",
        "nanosecond", "quarter", "weekday", "week", "date", "time",
    }
)  # fmt: skip

_MISSING_ATTR_RE = re.compile(r"'(\w+)' object has no attribute '(\w+)'")
_UNEXPECTED_KWARG_NAME_RE = re.compile(r"unexpected keyword argument '(\w+)'")
_UNCALLED_ACCESSOR_RE = re.compile(r"\.(dt|str)\.(\w+)\b(?!\s*\()")
_STR_NAMESPACES = {"ExprStringNameSpace", "StringNameSpace"}
_DT_NAMESPACES = {"ExprDateTimeNameSpace", "DateTimeNameSpace"}
_OBJECT_OWNERS = {"Expr", "Series", "DataFrame", "LazyFrame"}


def detect_pandas_style_name(code: str, error) -> str | None:
    """Return the pandas-style name behind a hallucinated-API error, else None.

    ``error`` is the executor's error record (``type`` / ``message``). Returns
    e.g. ``"str.strip"``, ``"dt.day_name"``, ``"suffixes="``, ``"rename"`` or
    ``"dt.year"`` (property-style access of a Polars method).
    """

    if error is None:
        return None
    message = error.message or ""

    m = _UNEXPECTED_KWARG_NAME_RE.search(message)
    if m and m.group(1) in PANDAS_KWARGS:
        return f"{m.group(1)}="

    m = _MISSING_ATTR_RE.search(message)
    if not m:
        return None
    owner, name = m.groups()
    if owner in _STR_NAMESPACES and name in PANDAS_STR_NAMES:
        return f"str.{name}"
    if owner in _DT_NAMESPACES and name in PANDAS_DT_NAMES:
        return f"dt.{name}"
    if owner in _OBJECT_OWNERS and name in PANDAS_OBJECT_NAMES:
        return name
    if owner in {"function", "method"}:
        # e.g. `pl.col("d").dt.year.alias(...)`: the pandas property idiom
        # applied to what is a method in Polars.
        for accessor, attr in _UNCALLED_ACCESSOR_RE.findall(code):
            if attr in PANDAS_PROPERTY_NAMES:
                return f"{accessor}.{attr}"
    return None


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
    pandas_style_name: str | None = None


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

    def result(status: str, reason: str, pandas_style_name: str | None = None) -> Classification:
        return Classification(
            status=status,
            deprecated_api=deprecated_api or status == DEPRECATED_API,
            pandas_leak_detected=pandas_leak_detected,
            reason=reason,
            warnings=exec_result.warnings,
            pandas_style_name=pandas_style_name,
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
            return result(
                HALLUCINATED_API,
                f"{error.type}: {error.message}",
                pandas_style_name=detect_pandas_style_name(code, error),
            )
        return result(RUNTIME_ERROR, f"{error.type}: {error.message}")

    # No error: the attempt ran to completion.
    if comparison is None:
        # Should not happen for a successful run, but guard defensively.
        return result(RUNTIME_ERROR, "ran without a comparison to the reference")
    if comparison.equal:
        return result(PASS, "output matched the reference")
    return result(WRONG_RESULT, comparison.detail or "output did not match the reference")
