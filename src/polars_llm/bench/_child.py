"""Subprocess sandbox entry point (invoked by ``executor.run_solution``).

Usage::

    python -m polars_llm.bench._child <payload.json> <result.json>

The payload JSON provides ``setup_code``, ``solve_code`` and ``result_ipc``.
This process:

1. runs ``setup_code`` to build the ``inputs`` dict,
2. compiles and executes ``solve_code`` to define ``solve``,
3. calls ``solve(**inputs)`` while recording warnings,
4. writes the resulting DataFrame to ``result_ipc`` (Arrow IPC), and
5. writes a structured outcome to ``result.json``.

The parent captures this process's stdout/stderr separately (user ``print``
output lives there); the structured outcome never goes to stdout, so it cannot
be polluted by the executed code.
"""

from __future__ import annotations

import inspect
import json
import sys
import traceback
import warnings


def _error_dict(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "mro": [c.__name__ for c in type(exc).__mro__],
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _warning_dicts(captured) -> list[dict]:
    out = []
    for w in captured:
        out.append(
            {
                "category": getattr(w.category, "__name__", str(w.category)),
                "message": str(w.message),
            }
        )
    return out


def _dump(result: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)


def _check_signature(solve, inputs: dict) -> None:
    """Enforce the solve(**inputs) convention: the keys of ``inputs`` must
    exactly match ``solve``'s parameter names.

    A ``**kwargs`` parameter relaxes the exact-match requirement to "every
    required positional/keyword parameter is supplied". ``*args`` cannot be
    filled from keyword ``inputs`` and is therefore treated as a mismatch.
    """

    params = inspect.signature(solve).parameters
    named = {
        name
        for name, p in params.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)
    }
    has_var_keyword = any(p.kind is p.VAR_KEYWORD for p in params.values())
    keys = set(inputs)

    if has_var_keyword:
        required = {
            name
            for name, p in params.items()
            if p.default is p.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)
        }
        if not required <= keys:
            raise ValueError(
                f"solve() is missing required parameters {sorted(required - keys)} "
                f"for inputs keys {sorted(keys)}"
            )
        return

    if named != keys:
        raise ValueError(
            f"solve() parameters {sorted(named)} do not match inputs keys "
            f"{sorted(keys)} (the setup_code `inputs` dict must name exactly "
            f"solve()'s parameters)"
        )


def main(argv: list[str]) -> int:
    payload_path, result_path = argv[1], argv[2]
    with open(payload_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    setup_code: str = payload["setup_code"]
    solve_code: str = payload["solve_code"]
    result_ipc: str = payload["result_ipc"]
    check_signature: bool = payload.get("check_signature", True)

    result: dict = {
        "ok": False,
        "phase": "setup",
        "error": None,
        "warnings": [],
        "has_result": False,
    }

    import polars as pl

    namespace: dict = {"pl": pl, "__name__": "__sandbox__"}

    # --- setup: build `inputs` ------------------------------------------------
    try:
        exec(compile(setup_code, "<setup>", "exec"), namespace)
        inputs = namespace.get("inputs")
        if not isinstance(inputs, dict):
            raise RuntimeError("setup_code must define a dict named `inputs`")
    except BaseException as exc:  # noqa: BLE001 - report every failure structurally
        result["error"] = _error_dict(exc)
        _dump(result, result_path)
        return 0

    # --- compile the candidate/reference solution -----------------------------
    result["phase"] = "compile"
    try:
        compiled = compile(solve_code, "<solution>", "exec")
    except SyntaxError as exc:
        result["error"] = _error_dict(exc)
        _dump(result, result_path)
        return 0

    # --- define `solve`, call it, normalize the result ------------------------
    # One warnings context spans module-level solution code and the call, so
    # deprecation warnings raised either place are recorded.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        result["phase"] = "exec_defs"
        try:
            exec(compiled, namespace)
        except BaseException as exc:  # noqa: BLE001
            result["warnings"] = _warning_dicts(captured)
            result["error"] = _error_dict(exc)
            _dump(result, result_path)
            return 0

        solve = namespace.get("solve")
        result["phase"] = "call"
        if not callable(solve):
            result["warnings"] = _warning_dicts(captured)
            result["error"] = {
                "type": "NoSolveFunction",
                "module": None,
                "mro": ["NoSolveFunction"],
                "message": "solution did not define a callable `solve`",
                "traceback": "",
            }
            _dump(result, result_path)
            return 0

        try:
            if check_signature:
                _check_signature(solve, inputs)
            out = solve(**inputs)
            if isinstance(out, pl.LazyFrame):
                out = out.collect()
            if isinstance(out, pl.Series):
                out = out.to_frame()
            if not isinstance(out, pl.DataFrame):
                raise TypeError(
                    f"solve() must return a polars DataFrame/LazyFrame/Series, "
                    f"got {type(out).__name__}"
                )
        except BaseException as exc:  # noqa: BLE001
            result["warnings"] = _warning_dicts(captured)
            result["error"] = _error_dict(exc)
            _dump(result, result_path)
            return 0

        result["warnings"] = _warning_dicts(captured)

    # --- serialize the result frame for the parent ----------------------------
    result["phase"] = "serialize"
    try:
        out.write_ipc(result_ipc)
        result["has_result"] = True
    except BaseException as exc:  # noqa: BLE001
        result["error"] = _error_dict(exc)
        _dump(result, result_path)
        return 0

    result["ok"] = True
    result["phase"] = "ok"
    _dump(result, result_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
