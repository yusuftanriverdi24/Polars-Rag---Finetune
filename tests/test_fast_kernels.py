"""Prebuilt-wheel lookup for the Qwen3.5 linear-attention kernels (never a source build)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from install_fast_kernels import pick_wheel, wheel_suffix  # noqa: E402

BASE = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
NAMES = [
    "causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    "causal_conv1d-1.7.0+cu12torch2.9cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    "causal_conv1d-1.7.0+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    "causal_conv1d-1.7.0+cu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
]
RELEASES = [
    {"tag_name": "v1.8.0rc1", "prerelease": True,
     "assets": [{"name": NAMES[2].replace("1.7.0", "1.8.0rc1"), "browser_download_url": "rc"}]},
    {"tag_name": "v1.7.0", "assets": [{"name": n, "browser_download_url": BASE + n} for n in NAMES]},
]


def test_wheel_suffix_matches_release_naming():
    s = wheel_suffix("12.6", "2.9.0+cu126", True, (3, 12))
    assert s == "+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"


def test_pick_wheel_exact_match_skips_prereleases():
    tag, url = pick_wheel(RELEASES, wheel_suffix("12.6", "2.9.0+cu126", True, (3, 12)))
    assert tag == "v1.7.0" and url.endswith(NAMES[2])
    assert pick_wheel(RELEASES, wheel_suffix("12.8", "2.10.0", True, (3, 12))) is None  # no wheel -> skip
    assert pick_wheel(RELEASES, wheel_suffix("13.0", "2.9.1", True, (3, 12))) is None
