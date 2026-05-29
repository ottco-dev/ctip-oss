"""
backend.utils.env_file — Safe .env file read/write utilities.

Used by containers.py (port management) and vlm_providers.py (provider persistence).
Preserves comments, ordering, and all unrelated keys on every write.
"""

from __future__ import annotations

import re
from pathlib import Path

# Canonical .env path — four levels up from this file:
#   backend/utils/env_file.py → backend/utils → backend → (project root)
_DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def get_env_path() -> Path:
    return _DEFAULT_ENV_FILE


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """
    Parse .env into a dict.

    Only non-comment, non-empty KEY=VALUE lines are returned.
    Strips surrounding quotes from values.
    """
    p = path or _DEFAULT_ENV_FILE
    result: dict[str, str] = {}
    if not p.exists():
        return result
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, val = stripped.partition("=")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def write_env_key(key: str, value: str, path: Path | None = None) -> None:
    """
    Update or append KEY="value" in .env, preserving all other content.

    - If the key already exists (with or without quotes), it is replaced in-place.
    - If the key does not exist, it is appended at the end.
    - Surrounding quotes are always written for the new value.
    """
    p = path or _DEFAULT_ENV_FILE
    if not p.exists():
        p.write_text(f'{key}="{value}"\n', encoding="utf-8")
        return

    content = p.read_text(encoding="utf-8")
    pattern = re.compile(rf'^{re.escape(key)}\s*=.*$', re.MULTILINE)
    new_line = f'{key}="{value}"'

    if pattern.search(content):
        content = pattern.sub(new_line, content)
    else:
        content = content.rstrip("\n") + f"\n{new_line}\n"

    p.write_text(content, encoding="utf-8")


def write_env_keys(pairs: dict[str, str], path: Path | None = None) -> None:
    """Write multiple key/value pairs atomically (one read, one write)."""
    p = path or _DEFAULT_ENV_FILE
    if not p.exists():
        lines = "\n".join(f'{k}="{v}"' for k, v in pairs.items()) + "\n"
        p.write_text(lines, encoding="utf-8")
        return

    content = p.read_text(encoding="utf-8")
    for key, value in pairs.items():
        pattern = re.compile(rf'^{re.escape(key)}\s*=.*$', re.MULTILINE)
        new_line = f'{key}="{value}"'
        if pattern.search(content):
            content = pattern.sub(new_line, content)
        else:
            content = content.rstrip("\n") + f"\n{new_line}\n"

    p.write_text(content, encoding="utf-8")
