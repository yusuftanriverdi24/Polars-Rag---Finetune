"""make_figures: arm resolution, incl. the A100 fallback for C/D (no matplotlib needed)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_figures  # noqa: E402


def _scores(path: Path, p: float) -> None:
    path.mkdir(parents=True)
    (path / "scores.json").write_text(json.dumps({"metrics": {"pass_at_1": p}}), encoding="utf-8")


def test_resolve_arms_prefers_official_then_a100(tmp_path):
    _scores(tmp_path / "A", 0.4)
    _scores(tmp_path / "C", 0.5)
    _scores(tmp_path / "C_a100", 0.9)
    _scores(tmp_path / "D_a100", 0.6)
    entries = make_figures.resolve_arms(tmp_path)
    assert [e["arm"] for e in entries] == ["A", "C", "D"]  # B, B2 missing -> skipped
    by_arm = {e["arm"]: e for e in entries}
    assert by_arm["C"]["metrics"]["pass_at_1"] == 0.5 and not by_arm["C"]["fallback"]
    assert by_arm["D"]["fallback"] and "pending L4" in by_arm["D"]["label"]


def test_resolve_arms_on_repo_results():
    entries = make_figures.resolve_arms(Path("results"))
    assert [e["arm"] for e in entries] == list(make_figures.ARMS)
