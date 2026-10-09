"""Subprocess sandbox executor (parent side).

Runs a ``solve`` implementation in a fresh Python subprocess (the same
interpreter / pinned Polars as the parent) with a wall-clock timeout, capturing
stdout, stderr, raised warnings, any exception, and the resulting DataFrame.

The result frame is returned via an Arrow IPC file, which preserves dtypes
exactly and avoids the Windows console-encoding pitfalls of printing frames.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

DEFAULT_TIMEOUT_S = 15.0
_MAX_CAPTURE_CHARS = 10_000


@dataclass
class ExecError:
    """A structured description of an exception raised in the sandbox."""

    type: str
    message: str
    mro: list[str] = field(default_factory=list)
    module: str | None = None
    traceback: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ExecError":
        return cls(
            type=d.get("type", "Error"),
            message=d.get("message", ""),
            mro=list(d.get("mro", [])),
            module=d.get("module"),
            traceback=d.get("traceback", ""),
        )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "message": self.message,
            "mro": self.mro,
            "module": self.module,
            "traceback": self.traceback,
        }


@dataclass
class ExecutionResult:
    """Outcome of running one ``solve`` implementation in the sandbox."""

    ok: bool
    phase: str  # setup | compile | exec_defs | call | serialize | ok
    frame: "pl.DataFrame | None"
    warnings: list[dict]
    error: ExecError | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float

    @property
    def deprecation_warnings(self) -> list[dict]:
        return [w for w in self.warnings if _is_deprecation_warning(w)]


def _is_deprecation_warning(w: dict) -> bool:
    name = (w.get("category") or "").lower()
    return (
        name in {"deprecationwarning", "futurewarning", "pendingdeprecationwarning"}
        or "deprecat" in name
        or "removed" in name
    )


def _truncate(s: str) -> str:
    if len(s) > _MAX_CAPTURE_CHARS:
        return s[:_MAX_CAPTURE_CHARS] + f"\n... [truncated {len(s) - _MAX_CAPTURE_CHARS} chars]"
    return s


def run_solution(
    setup_code: str,
    solve_code: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecutionResult:
    """Execute ``solve_code`` against inputs built by ``setup_code`` in a sandbox."""

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="polars_bench_") as tmp:
        tmp_path = Path(tmp)
        payload_path = tmp_path / "payload.json"
        result_path = tmp_path / "result.json"
        result_ipc = tmp_path / "result.arrow"

        payload_path.write_text(
            json.dumps(
                {
                    "setup_code": setup_code,
                    "solve_code": solve_code,
                    "result_ipc": str(result_ipc),
                }
            ),
            encoding="utf-8",
        )

        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [
            sys.executable,
            "-m",
            "polars_llm.bench._child",
            str(payload_path),
            str(result_path),
        ]

        timed_out = False
        stdout = ""
        stderr = ""
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout_s,
                env=env,
            )
            stdout = proc.stdout.decode("utf-8", "replace")
            stderr = proc.stderr.decode("utf-8", "replace")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            if exc.stdout:
                stdout = exc.stdout.decode("utf-8", "replace")
            if exc.stderr:
                stderr = exc.stderr.decode("utf-8", "replace")

        duration = time.perf_counter() - start
        stdout, stderr = _truncate(stdout), _truncate(stderr)

        if timed_out:
            return ExecutionResult(
                ok=False,
                phase="call",
                frame=None,
                warnings=[],
                error=ExecError(
                    type="Timeout",
                    message=f"execution exceeded {timeout_s}s",
                    mro=["Timeout"],
                ),
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                duration_s=duration,
            )

        # The child writes result.json for every non-timeout outcome. If it is
        # missing the child itself crashed (e.g. import failure): surface stderr.
        if not result_path.exists():
            return ExecutionResult(
                ok=False,
                phase="setup",
                frame=None,
                warnings=[],
                error=ExecError(
                    type="SandboxError",
                    message="sandbox did not produce a result; see stderr",
                    mro=["SandboxError"],
                    traceback=stderr,
                ),
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                duration_s=duration,
            )

        outcome = json.loads(result_path.read_text(encoding="utf-8"))

        frame = None
        if outcome.get("has_result"):
            frame = pl.read_ipc(result_ipc)

        error = ExecError.from_dict(outcome["error"]) if outcome.get("error") else None

        return ExecutionResult(
            ok=bool(outcome.get("ok")),
            phase=outcome.get("phase", "unknown"),
            frame=frame,
            warnings=list(outcome.get("warnings", [])),
            error=error,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            duration_s=duration,
        )


def compute_expected(
    setup_code: str,
    reference_solution: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecutionResult:
    """Compute a task's expected output by running its reference solution.

    This is the same sandbox path used for model outputs, so the reference is
    validated under the pinned Polars version exactly as candidates are.
    """

    return run_solution(setup_code, reference_solution, timeout_s=timeout_s)
