"""Helpers for the text that remote VLM APIs return."""

from __future__ import annotations

import re

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence (```json ... ```), leaving the content untouched otherwise."""
    m = _FENCE.match(text)
    return (m.group(1) if m else text).strip()
