"""Draw the README figures from the arms' ``results/<arm>/scores.json``.

Writes ``results/figures/pass_at_1.png`` (pass@1 per arm) and
``results/figures/by_category.png`` (pass@1, arms x categories). Arms C and D
use ``results/C`` / ``results/D`` (the official L4 runs) when they exist, else
the A100 runs ``results/C_a100`` / ``results/D_a100``, labelled as such.

    uv sync --extra figures
    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("A", "B", "B2", "C", "D")
ARM_NAMES = {
    "A": "A  base",
    "B": "B  naive RAG",
    "B2": "B2 improved RAG",
    "C": "C  fine-tuned",
    "D": "D  fine-tuned + RAG",
}
FALLBACK_SUFFIX = "_a100"

# Reference palette (dataviz skill, light mode): ink, surface, blue ramp.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
BLUE_PENDING = "#9ec5f4"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def resolve_arms(results_dir: Path, arms=ARMS) -> list[dict]:
    """One entry per arm: label, scores path, and whether it is the A100 fallback."""

    out = []
    for arm in arms:
        path = results_dir / arm / "scores.json"
        fallback = False
        if not path.exists():
            path = results_dir / f"{arm}{FALLBACK_SUFFIX}" / "scores.json"
            fallback = True
        if not path.exists():
            continue
        label = ARM_NAMES.get(arm, arm) + (" (A100, pending L4)" if fallback else "")
        out.append({"arm": arm, "label": label, "fallback": fallback,
                    "metrics": json.loads(path.read_text(encoding="utf-8"))["metrics"]})
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2, labelsize=9)


def plot_pass_at_1(entries: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(entries) + 1.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    labels = [e["label"] for e in entries][::-1]
    values = [e["metrics"]["pass_at_1"] for e in entries][::-1]
    colors = [BLUE_PENDING if e["fallback"] else BLUE for e in entries][::-1]
    bars = ax.barh(labels, values, height=0.55, color=colors)
    for bar, e in zip(bars, entries[::-1]):
        m = e["metrics"]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{m['pass_at_1']:.3f}  ({m['pass']}/{m['total']})", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("pass@1 (120 tasks, benchmark v1.1, greedy)", color=INK_2, fontsize=9)
    ax.set_title("pass@1 per arm", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_by_category(entries: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cats = sorted(entries[0]["metrics"]["pass_at_1_by_category"])
    grid = [[e["metrics"]["pass_at_1_by_category"][c]["pass_at_1"] for c in cats] for e in entries]
    fig, ax = plt.subplots(figsize=(9.6, 0.55 * len(entries) + 1.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    cmap = ListedColormap(SEQ)
    norm = BoundaryNorm([i / len(SEQ) for i in range(len(SEQ) + 1)], cmap.N)
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    for i, e in enumerate(entries):
        for j, c in enumerate(cats):
            cell = e["metrics"]["pass_at_1_by_category"][c]
            dark = norm(grid[i][j]) >= 3
            ax.text(j, i, f"{cell['pass']}/{cell['total']}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(cats)), [c.replace("_", " ") for c in cats], rotation=0)
    ax.set_yticks(range(len(entries)), [e["label"] for e in entries])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    # Hairline cell borders in the surface color.
    ax.set_xticks([j - 0.5 for j in range(1, len(cats))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(entries))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("pass@1 by category (passed / tasks; darker = higher)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the README figures from results/<arm>/scores.json.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    entries = resolve_arms(Path(args.results_dir))
    if not entries:
        raise SystemExit(f"no scores.json found under {args.results_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pass_at_1(entries, out_dir / "pass_at_1.png")
    plot_by_category(entries, out_dir / "by_category.png")
    for e in entries:
        print(f"{e['label']:<40} {e['metrics']['pass_at_1']:.4f}")
    print(f"Wrote {out_dir / 'pass_at_1.png'} and {out_dir / 'by_category.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
