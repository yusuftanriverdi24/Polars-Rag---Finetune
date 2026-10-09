"""Extract runnable Python code from raw model output.

Models wrap code in Markdown fences and often add prose before/after it. This
module recovers the code with a few robust heuristics:

- Prefer the first ```python / ```py fenced block.
- Otherwise use the first fenced block of any language.
- Handle an unterminated fence (truncated generation) by taking everything
  after the opening fence.
- If there are no fences at all, assume the whole text is code and strip it.
"""

from __future__ import annotations

import re

# Opening fence (``` or ~~~) with an optional language tag, then the body,
# ending either at a closing fence or end-of-string (unterminated block).
_FENCE_RE = re.compile(
    r"(?P<fence>```+|~~~+)[ \t]*(?P<lang>[\w.+-]*)[ \t]*\r?\n"
    r"(?P<body>.*?)"
    r"(?:\r?\n(?P=fence)|\Z)",
    re.DOTALL,
)

_PY_LANGS = {"python", "py", "python3"}


def extract_code(text: str | None) -> str:
    """Return the best-guess Python code contained in ``text``.

    Never raises; returns an empty string for ``None``/empty input.
    """

    if not text:
        return ""

    blocks = [(m.group("lang").lower(), m.group("body")) for m in _FENCE_RE.finditer(text)]
    if blocks:
        for lang, body in blocks:
            if lang in _PY_LANGS:
                return body.strip("\r\n")
        # No explicitly python-tagged block: fall back to the first fenced block.
        return blocks[0][1].strip("\r\n")

    return text.strip()
