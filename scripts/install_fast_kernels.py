"""Install the fast linear-attention kernels for Qwen3.5 on Colab, prebuilt only.

Qwen3.5's linear-attention (Gated DeltaNet) layers use ``causal-conv1d`` and
``flash-linear-attention`` (``fla``) when they are importable, and otherwise
fall back to a slow pure-torch reference implementation (same math, much
slower, more memory). Unsloth recommends installing both for Qwen3.5.

This script never compiles anything:

* ``causal-conv1d``: PyPI only has a source distribution (its ``setup.py``
  downloads a wheel *or builds from source*), so we look up the prebuilt
  wheel for this exact CUDA major / torch major.minor / C++11 ABI / Python in
  the recent GitHub releases and ``pip install`` that ``.whl`` URL directly.
  No matching wheel -> skipped.
* ``flash-linear-attention`` + ``fla-core``: pure-Python (Triton) wheels,
  pinned, installed with ``--only-binary=:all: --no-deps`` so pip can neither
  build nor touch the pinned transformers/torch.

Each package is verified by importing it in a fresh subprocess; a package
that fails to import is uninstalled again. Whatever happens the script exits
0: the model still runs on the fallback. The path taken is printed and
written as JSON (``--log``).

    python scripts/install_fast_kernels.py --log results/datagen/fast_kernels.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

RELEASES_API = "https://api.github.com/repos/Dao-AILab/causal-conv1d/releases?per_page=3"
FLA_PINS = ["flash-linear-attention==0.5.2", "fla-core==0.5.2"]  # released 2026-07-27
FLA_CHECK = "from fla.ops.gated_delta_rule import chunk_gated_delta_rule"
CONV_CHECK = "import torch, causal_conv1d; from causal_conv1d import causal_conv1d_fn"


def wheel_suffix(cuda_version: str, torch_version: str, cxx11_abi: bool, python: tuple[int, int]) -> str:
    """The platform part of a causal-conv1d release wheel name, e.g.
    ``+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl``."""

    cuda_major = cuda_version.split(".")[0]
    torch_mm = ".".join(torch_version.split("+")[0].split(".")[:2])
    py = f"cp{python[0]}{python[1]}"
    return f"+cu{cuda_major}torch{torch_mm}cxx11abi{str(cxx11_abi).upper()}-{py}-{py}-linux_x86_64.whl"


def pick_wheel(releases: list[dict], suffix: str) -> tuple[str, str] | None:
    """(release tag, download URL) of the newest release with a wheel ending in ``suffix``."""

    for rel in releases:
        if rel.get("draft") or rel.get("prerelease"):
            continue
        for asset in rel.get("assets", []):
            if re.match(r"causal_conv1d-[\w.]+\+", asset["name"]) and asset["name"].endswith(suffix):
                return rel["tag_name"], asset["browser_download_url"]
    return None


def _pip(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True)


def _imports(code: str) -> tuple[bool, str]:
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return p.returncode == 0, (p.stderr.strip().splitlines() or [""])[-1]


def install_causal_conv1d(env: dict) -> dict:
    ok, _ = _imports(CONV_CHECK)
    if ok:
        return {"path": "already installed"}
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not env.get("cuda"):
        return {"path": "skipped: not linux x86_64 with CUDA"}
    suffix = wheel_suffix(env["cuda"], env["torch"], env["cxx11_abi"], sys.version_info[:2])
    try:
        with urllib.request.urlopen(RELEASES_API, timeout=30) as resp:
            releases = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - offline / rate-limited -> skip, never build
        return {"path": f"skipped: could not list releases ({exc})", "wanted": suffix}
    hit = pick_wheel(releases, suffix)
    if hit is None:
        return {"path": "skipped: no prebuilt wheel for this torch/CUDA/python", "wanted": suffix}
    tag, url = hit
    p = _pip("install", "-q", "--no-deps", url)
    if p.returncode != 0:
        return {"path": "skipped: wheel install failed", "wheel": url, "error": p.stderr.strip()[-300:]}
    ok, err = _imports(CONV_CHECK)
    if not ok:
        _pip("uninstall", "-y", "-q", "causal-conv1d")
        return {"path": "skipped: prebuilt wheel did not import (uninstalled)", "wheel": url, "error": err}
    return {"path": "prebuilt wheel", "release": tag, "wheel": url.rsplit("/", 1)[-1]}


def install_fla(env: dict) -> dict:
    ok, _ = _imports(FLA_CHECK)
    if ok:
        return {"path": "already installed"}
    if platform.system() != "Linux" or not env.get("cuda"):
        return {"path": "skipped: not linux with CUDA"}
    p = _pip("install", "-q", "--only-binary=:all:", "--no-deps", *FLA_PINS)
    if p.returncode != 0:
        return {"path": "skipped: no wheel / install failed", "error": p.stderr.strip()[-300:]}
    ok, err = _imports(FLA_CHECK)
    if not ok:
        _pip("uninstall", "-y", "-q", "flash-linear-attention", "fla-core")
        return {"path": "skipped: installed but did not import (uninstalled)", "error": err}
    return {"path": "prebuilt wheel", "packages": FLA_PINS}


def torch_env() -> dict:
    try:
        import torch
    except ImportError:
        return {}
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "python": platform.python_version()}


def transformers_fast_path() -> dict:
    """What transformers itself will see (checked in a fresh process)."""

    code = (
        "import json; from transformers.utils import import_utils as u; "
        "print(json.dumps({n: bool(getattr(u, n)()) for n in "
        "('is_causal_conv1d_available', 'is_flash_linear_attention_available') if hasattr(u, n)}))"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"error": (p.stderr.strip().splitlines() or ["unknown"])[-1]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log", default=None, help="write the outcome as JSON here")
    args = parser.parse_args(argv)

    env = torch_env()
    report = {"env": env, "causal_conv1d": install_causal_conv1d(env), "flash_linear_attention": install_fla(env)}
    report["transformers_sees"] = transformers_fast_path()
    have = [k for k in ("causal_conv1d", "flash_linear_attention")
            if report[k]["path"] in ("prebuilt wheel", "already installed")]
    report["linear_attention"] = (
        "fast kernels" if len(have) == 2
        else f"partial: only {have[0]} (the rest on the torch fallback)" if have
        else "torch fallback (slow)"
    )
    print(json.dumps(report, indent=2))
    print(f"linear attention: {report['linear_attention']}")
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
