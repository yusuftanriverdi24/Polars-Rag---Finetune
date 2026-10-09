"""Reproducibly download the Polars 2.0.0 documentation corpus.

Downloads a curated subset of the Polars source docs at the exact commit tagged
``py-2.0.0`` into ``data/raw/`` (gitignored). This is the source corpus for both
RAG retrieval and (later) grounded synthetic data generation.

This module only **downloads** — chunking/embedding happen in later steps.

What it fetches (all at the pinned commit, so the corpus is reproducible):

- ``docs/source/user-guide/**``            the User Guide (Markdown)
- ``docs/source/src/python/user-guide/**`` the Python code snippets the guide embeds
- ``docs/source/releases/upgrade/*.md``    version upgrade guides, incl. ``2.md`` (1.x -> 2.0)
- ``docs/source/releases/changelog.md``    the changelog
- ``docs/source/user-guide/migration/*``   pandas / spark migration guides
- ``py-polars/docs/source/reference/**``   the Python API reference (reStructuredText)
- ``docs/source/api/reference.md`` + ``docs/source/_build/API_REFERENCE_LINKS.yml``

Usage::

    python -m polars_llm.rag.ingest                 # download to data/raw/polars-2.0.0
    python -m polars_llm.rag.ingest --dry-run       # list files, download nothing
    python -m polars_llm.rag.ingest --resolve-tag   # re-resolve py-2.0.0 -> commit sha
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "pola-rs/polars"
TAG = "py-2.0.0"
# Commit the py-2.0.0 tag points to. Pinning the sha (not just the tag) makes the
# downloaded corpus byte-for-byte reproducible even if tags are ever moved.
PINNED_COMMIT = "22a147de3d2bb2e44b97338a2510816c7105c9f2"

DEFAULT_DEST = Path("data/raw/polars-2.0.0")
_USER_AGENT = "polars-llm-ingest"


def _keep(path: str) -> bool:
    """Whether a repo path belongs in the documentation corpus."""

    if path.startswith("docs/source/user-guide/"):
        return True
    if path.startswith("docs/source/src/python/user-guide/") and path.endswith(".py"):
        return True
    if path.startswith("docs/source/releases/upgrade/") and path.endswith(".md"):
        return True
    if path == "docs/source/releases/changelog.md":
        return True
    if path.startswith("py-polars/docs/source/reference/") and path.endswith(
        (".rst", ".md")
    ):
        return True
    if path == "docs/source/api/reference.md":
        return True
    if path == "docs/source/_build/API_REFERENCE_LINKS.yml":
        return True
    return False


def _http_get(url: str, *, retries: int = 3, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except Exception as exc:  # noqa: BLE001 - network flakiness: retry
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to GET {url}: {last}")


def resolve_tag_to_commit(tag: str = TAG) -> str:
    """Resolve a git tag to its commit sha via the GitHub API."""

    ref = json.loads(
        _http_get(f"https://api.github.com/repos/{REPO}/git/ref/tags/{tag}")
    )
    obj = ref["object"]
    if obj["type"] == "tag":  # annotated tag -> dereference to the commit
        tag_obj = json.loads(
            _http_get(f"https://api.github.com/repos/{REPO}/git/tags/{obj['sha']}")
        )
        return tag_obj["object"]["sha"]
    return obj["sha"]


def list_corpus_paths(commit: str) -> list[str]:
    """List the doc-corpus file paths present at ``commit``."""

    tree = json.loads(
        _http_get(
            f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1"
        )
    )
    if tree.get("truncated"):
        raise RuntimeError("git tree response was truncated; cannot enumerate reliably")
    return sorted(e["path"] for e in tree["tree"] if e["type"] == "blob" and _keep(e["path"]))


def download(
    dest: Path = DEFAULT_DEST,
    commit: str = PINNED_COMMIT,
    tag: str = TAG,
    *,
    dry_run: bool = False,
) -> dict:
    """Download the corpus to ``dest`` and write a manifest. Returns the manifest."""

    paths = list_corpus_paths(commit)
    print(f"{len(paths)} files match the corpus selection at {REPO}@{commit[:10]}")

    manifest_files: list[dict] = []
    total_bytes = 0
    for i, path in enumerate(paths, start=1):
        if dry_run:
            print(f"  [dry-run] {path}")
            continue
        raw_url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}"
        data = _http_get(raw_url)
        out_path = dest / path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        total_bytes += len(data)
        manifest_files.append({"path": path, "bytes": len(data)})
        if i % 25 == 0 or i == len(paths):
            print(f"  downloaded {i}/{len(paths)} ({total_bytes:,} bytes)")

    manifest = {
        "repo": REPO,
        "tag": tag,
        "commit": commit,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "file_count": len(manifest_files),
        "total_bytes": total_bytes,
        "files": manifest_files,
    }
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {len(manifest_files)} files + manifest.json to {dest}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Polars 2.0.0 docs corpus.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--commit", default=PINNED_COMMIT)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument(
        "--resolve-tag",
        action="store_true",
        help="resolve --tag to a commit sha via the GitHub API instead of the pinned sha",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    commit = resolve_tag_to_commit(args.tag) if args.resolve_tag else args.commit
    download(Path(args.dest), commit=commit, tag=args.tag, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
