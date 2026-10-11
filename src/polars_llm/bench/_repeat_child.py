"""Subprocess entry point for repeated runs (invoked by ``determinism.run_repeated``).

Usage::

    python -m polars_llm.bench._repeat_child <payload.json> <result.json>

The payload provides ``setup_code``, ``solve_code``, ``n`` and ``out_dir``. The
solution is defined once and then called ``n`` times, each time on freshly
built inputs; every result frame is written to ``<out_dir>/<i>.arrow`` and a
per-run outcome list to ``result.json``. Running all repetitions in one
process keeps the audit fast while still exercising Polars' multi-threaded
(and therefore potentially order-unstable) execution on every call.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from ._child import _check_signature


def main(argv: list[str]) -> int:
    payload_path, result_path = argv[1], argv[2]
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    setup_code: str = payload["setup_code"]
    solve_code: str = payload["solve_code"]
    n: int = int(payload["n"])
    out_dir = Path(payload["out_dir"])

    import polars as pl

    runs: list[dict] = []
    namespace: dict = {"pl": pl, "__name__": "__sandbox__"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            exec(compile(solve_code, "<solution>", "exec"), namespace)
            solve = namespace["solve"]
        except BaseException as exc:  # noqa: BLE001
            err = {"type": type(exc).__name__, "message": str(exc)}
            runs = [{"ok": False, "error": err} for _ in range(n)]
            Path(result_path).write_text(json.dumps({"runs": runs}), encoding="utf-8")
            return 0

        for i in range(n):
            try:
                setup_ns: dict = {"pl": pl, "__name__": "__sandbox__"}
                exec(compile(setup_code, "<setup>", "exec"), setup_ns)
                inputs = setup_ns["inputs"]
                _check_signature(solve, inputs)
                out = solve(**inputs)
                if isinstance(out, pl.LazyFrame):
                    out = out.collect()
                if isinstance(out, pl.Series):
                    out = out.to_frame()
                if not isinstance(out, pl.DataFrame):
                    raise TypeError(f"solve() returned {type(out).__name__}")
                out.write_ipc(out_dir / f"{i}.arrow")
                runs.append({"ok": True, "error": None})
            except BaseException as exc:  # noqa: BLE001
                runs.append({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}})

    Path(result_path).write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
