"""Oracle doc context for the round-trip check, with the answer removed.

The round-trip solver gets the doc chunk(s) a candidate came from, so it
does not need to *know* the API. But those chunks contain the example itself:
with the code in view, the solver can match the setup to the doc example and
copy the answer, and the check would no longer test the request. So:

* **API pages:** all parts of the page are re-joined (parts 2+ repeat the
  signature line) and the whole ``Examples`` section is dropped; the
  signature, description, Parameters / Returns / See Also / Notes stay.
* **Guide chunks:** the chunk(s) containing the candidate's code, minus every
  code block the candidate came from (a block holding one of its code lines
  or its answer expression); the surrounding prose and other blocks stay.

``answer_in_context`` is the check: the candidate's answer expression (the
reference ``return`` expression; for ``return name``, the value last
assigned to that name; with and without a trailing ``.collect()``), with all
whitespace removed, must not occur in the whitespace-free context.
"""

from __future__ import annotations

import ast
import collections
import re
import statistics
from pathlib import Path

from ..rag.chunk import read_chunks

DEFAULT_V2_CHUNKS = Path("data/rag/chunks_v2.jsonl")
ORACLE_MAX_TOKENS = 4000
ORACLE_MAX_GUIDE_CHUNKS = 2

_HEADING_RE = re.compile(r"^([A-Z][A-Za-z ]+)\n-{3,}[ \t]*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS_RE.sub("", text)


# ---------------------------------------------------------------------------
# the answer
# ---------------------------------------------------------------------------


def answer_expressions(solution: str) -> list[str]:
    """Source of the reference answer: the ``return`` expression of ``solve``
    (or, for ``return name``, the value last assigned to that name), plus the
    same without a trailing ``.collect()`` (added by validation for LazyFrames)."""

    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return []
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"), None)
    if fn is None or not fn.body or not isinstance(fn.body[-1], ast.Return) or fn.body[-1].value is None:
        return []
    value = fn.body[-1].value
    body = fn.body[:-1]
    # `return out` / `return lf.collect()`: follow the names back to the expression that computes them.
    for _ in range(len(body) + 1):
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "collect" \
                and not value.args and isinstance(value.func.value, ast.Name):
            value = value.func.value
        if not isinstance(value, ast.Name):
            break
        k = max((i for i, s in enumerate(body) if isinstance(s, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == value.id for t in s.targets)), default=None)
        if k is None:
            return []  # an input itself: nothing to copy
        value, body = body[k].value, body[:k]
    src = ast.get_source_segment(solution, value) or ast.unparse(value)
    out = [src]
    if src.rstrip().endswith(".collect()"):
        out.append(src.rstrip()[: -len(".collect()")])
    return out


_PROMPT_RE = re.compile(r"^[ \t]*(?:>>>|\.\.\.)[ \t]?", re.MULTILINE)


def answer_in_context(solution: str, context: str) -> bool:
    ctx = _norm(_PROMPT_RE.sub("", context))  # doctest prompts (>>> / ...) split multi-line code
    return any(_norm(a) and _norm(a) in ctx for a in answer_expressions(solution))


# ---------------------------------------------------------------------------
# stripping
# ---------------------------------------------------------------------------


def join_api_parts(parts: list) -> str:
    """One API page from its chunks: part 1, then parts 2+ without the repeated signature line."""

    parts = sorted(parts, key=lambda c: c.part)
    head = parts[0].text.split("\n\n", 1)[0]
    texts = [parts[0].text]
    for c in parts[1:]:
        texts.append(c.text[len(head):].lstrip("\n") if c.text.startswith(head) else c.text)
    return "\n\n".join(t.strip("\n") for t in texts)


def strip_examples(doc: str) -> str:
    """``doc`` without its ``Examples`` section(s) (up to the next numpydoc heading)."""

    heads = list(_HEADING_RE.finditer(doc))
    out, pos = [], 0
    for k, m in enumerate(heads):
        if m.group(1).strip() != "Examples":
            continue
        end = heads[k + 1].start() if k + 1 < len(heads) else len(doc)
        out.append(doc[pos : m.start()])
        pos = end
    out.append(doc[pos:])
    # Doctest snippets outside the Examples heading (e.g. under Notes) are examples too.
    paras = [p for p in re.split(r"\n\s*\n", "".join(out)) if not p.lstrip().startswith(">>>")]
    return "\n\n".join(p.strip("\n") for p in paras if p.strip()).strip()


def strip_candidate_blocks(text: str, code_lines: set[str], answers: list[str]) -> str:
    """``text`` without the code blocks holding the candidate's code or answer."""

    norm_answers = [_norm(a) for a in answers if _norm(a)]

    def keep(m: re.Match) -> str:
        block = m.group(0)
        if any(l in block for l in code_lines) or any(a in _norm(block) for a in norm_answers):
            return ""
        return block

    return re.sub(r"\n{3,}", "\n\n", _FENCE_RE.sub(keep, text)).strip()


def code_lines(rec: dict) -> set[str]:
    """Distinctive source lines of a candidate (setup + solution), for finding its guide chunk."""

    lines = set()
    for code in (rec["setup_code"], rec["reference_solution"]):
        for line in code.splitlines():
            s = line.strip()
            if len(s) >= 15 and not s.startswith(("def solve", "inputs = ", "return ", "import ", "from ")):
                lines.add(s)
    return lines


# ---------------------------------------------------------------------------
# per candidate
# ---------------------------------------------------------------------------


def guide_chunks_for(rec: dict, guide_chunks: list) -> list:
    """The guide chunk(s) containing the most of the candidate's code lines
    (ties: the page named like the snippet file first, then by id)."""

    lines = code_lines(rec)
    scored = [(sum(l in c.text for l in lines), c) for c in guide_chunks]
    best = max((s for s, _ in scored), default=0)
    same_page = rec["page"].rsplit(".", 1)[0] + ".md"  # user-guide/x/y.py -> .../user-guide/x/y.md
    found = sorted((c for s, c in scored if s == best and s > 0),
                   key=lambda c: (not c.source.endswith(same_page), c.id))
    out, total = [], 0
    for c in found[:ORACLE_MAX_GUIDE_CHUNKS]:
        if out and total + c.n_tokens > ORACLE_MAX_TOKENS:
            break
        out.append(c)
        total += c.n_tokens
    return out


def oracle_context(rec: dict, api_chunks: dict[str, list], guide_chunks: list) -> tuple[list[str], str]:
    """(chunk ids, context text) for a candidate, with the answer removed."""

    if rec["source"] == "docstring":
        parts = api_chunks.get(rec["page"], [])
        if not parts:
            return [], ""
        title = parts[0].title
        body = strip_examples(join_api_parts(parts))
        return [c.id for c in sorted(parts, key=lambda c: c.part)], f"[1] {title}\n{body}"
    found = guide_chunks_for(rec, guide_chunks)
    lines, answers = code_lines(rec), answer_expressions(rec["reference_solution"])
    blocks = []
    for k, c in enumerate(found, start=1):
        head = c.title if not c.section or c.section == c.title else f"{c.title} ({c.section})"
        blocks.append(f"[{k}] {head}\n{strip_candidate_blocks(c.text, lines, answers)}")
    return [c.id for c in found], "\n\n".join(blocks)


def attach_doc_context(records: list[dict], chunks_path: Path = DEFAULT_V2_CHUNKS) -> dict:
    """Add ``doc_chunk_ids`` and ``doc_context`` to every record; returns stats
    incl. how many contexts still contain the answer (must be 0)."""

    chunks = read_chunks(chunks_path)
    api_chunks: dict[str, list] = collections.defaultdict(list)
    for c in chunks:
        if c.kind == "api" and c.api_names:
            api_chunks[c.api_names[0]].append(c)
    guide_chunks = [c for c in chunks if c.kind == "guide"]
    missing = collections.Counter()
    for r in records:
        r["doc_chunk_ids"], r["doc_context"] = oracle_context(r, api_chunks, guide_chunks)
        if not r["doc_chunk_ids"]:
            missing[r["source"]] += 1
    return {
        "answer_removed": "API pages: Examples section dropped; guide chunks: the candidate's code block(s) dropped",
        "missing_by_source": dict(missing),
        "mean_chunks": round(statistics.fmean(len(r["doc_chunk_ids"]) for r in records), 2) if records else 0,
        "answer_in_context": answer_leak_stats(records),
    }


def answer_leak_stats(records: list[dict]) -> dict:
    hits = [r["id"] for r in records if answer_in_context(r["reference_solution"], r.get("doc_context", ""))]
    return {"n": len(records), "with_answer_in_context": len(hits), "ids": hits[:50]}
