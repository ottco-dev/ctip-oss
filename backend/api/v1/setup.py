"""
backend.api.v1.setup — First-time configuration API.

Provides endpoints to read, validate, and persist platform configuration
to the .env file without requiring direct filesystem access from the user.

Security notes:
- Sensitive values (passwords, tokens) are redacted in GET responses.
- The .env file is written atomically (temp file → rename).
- Only keys defined in ALLOWED_KEYS can be written (allowlist-only).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/setup", tags=["setup"])

# ── Constants ─────────────────────────────────────────────────────────────────

ENV_FILE = Path(".env")
ENV_EXAMPLE = Path(".env.example")

# Keys that the setup wizard is allowed to read and write.
# All other keys in the .env are left untouched.
ALLOWED_KEYS: set[str] = {
    # Network
    "PUBLIC_DOMAIN",
    "PUBLIC_PORT",
    # Hardware
    "CUDA_DEVICE",
    "CUDA_VISIBLE_DEVICES",
    "VRAM_LIMIT_GB",
    "VRAM_INFERENCE_BUDGET_GB",
    # Storage
    "DATA_ROOT",
    "TRICHOME_ROOT",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "UPLOADS_DIR",
    # Services
    "LABEL_STUDIO_URL",
    "LABEL_STUDIO_API_KEY",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "USE_WANDB",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    # Security
    "SECRET_KEY",
    "API_TOKEN",
    # Annotation
    "CVAT_URL",
    "CVAT_USERNAME",
    # App
    "ENVIRONMENT",
    "LOG_LEVEL",
}

# Keys whose values must be redacted in GET responses
SENSITIVE_KEYS: set[str] = {
    "LABEL_STUDIO_API_KEY",
    "WANDB_API_KEY",
    "SECRET_KEY",
    "API_TOKEN",
    "CVAT_PASSWORD",
}

# ── Schemas ───────────────────────────────────────────────────────────────────


class SetupStatus(BaseModel):
    completed: bool
    """True if the user has run first-time setup at least once."""
    env_exists: bool
    """True if a .env file is present."""
    configured_keys: list[str]
    """Which allowed keys are currently set in .env."""


class ConfigEntry(BaseModel):
    key: str
    value: str
    sensitive: bool = False


class ConfigReadResponse(BaseModel):
    entries: list[ConfigEntry]
    warnings: list[str] = []


class ConfigWriteRequest(BaseModel):
    settings: dict[str, str] = Field(
        ...,
        description="Map of env-var key → value to write to .env.",
    )
    mark_setup_complete: bool = Field(
        default=True,
        description="Append SETUP_COMPLETED=true to .env.",
    )


class ConfigWriteResponse(BaseModel):
    written: list[str]
    skipped: list[str]
    env_path: str


class ValidationResult(BaseModel):
    key: str
    value: str
    valid: bool
    message: str = ""


class ValidateRequest(BaseModel):
    settings: dict[str, str]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, ignoring comments and blank lines."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            result[key] = value
    return result


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    """
    Write updates into the .env file.

    Strategy:
    1. Read all existing lines.
    2. Replace values for keys that already exist.
    3. Append new keys that didn't exist.
    4. Write atomically via temp file.
    """
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f'{key}="{updates[key]}"')
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # Append keys that weren't already present
    remaining = {k: v for k, v in updates.items() if k not in updated_keys}
    if remaining:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")  # blank separator
        new_lines.append("# --- CTIP Setup Wizard ---")
        for key, value in sorted(remaining.items()):
            new_lines.append(f'{key}="{value}"')

    content = "\n".join(new_lines)
    if not content.endswith("\n"):
        content += "\n"

    # Atomic write
    dir_path = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dir_path, delete=False, suffix=".env.tmp"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    tmp_path.replace(path)


def _redact(key: str, value: str) -> str:
    if key in SENSITIVE_KEYS and value:
        return "••••••••"
    return value


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/status", response_model=SetupStatus)
async def get_setup_status() -> SetupStatus:
    """
    Return whether first-time setup has been completed.

    Checks for the presence of SETUP_COMPLETED=true in .env.
    """
    env = _read_env_file(ENV_FILE)
    completed = env.get("SETUP_COMPLETED", "").lower() in ("true", "1", "yes")
    configured = [k for k in ALLOWED_KEYS if k in env and env[k]]
    return SetupStatus(
        completed=completed,
        env_exists=ENV_FILE.exists(),
        configured_keys=sorted(configured),
    )


@router.get("/config", response_model=ConfigReadResponse)
async def get_config() -> ConfigReadResponse:
    """
    Return current platform configuration.

    Sensitive values (API keys, passwords, secret keys) are redacted.
    Reads from .env if present, falls back to .env.example.
    """
    env = _read_env_file(ENV_FILE)
    example = _read_env_file(ENV_EXAMPLE)

    warnings: list[str] = []
    if not ENV_FILE.exists():
        warnings.append(
            ".env file not found — showing defaults from .env.example. "
            "Complete setup to create your .env."
        )

    entries: list[ConfigEntry] = []
    for key in sorted(ALLOWED_KEYS):
        value = env.get(key, example.get(key, ""))
        entries.append(
            ConfigEntry(
                key=key,
                value=_redact(key, value),
                sensitive=key in SENSITIVE_KEYS,
            )
        )

    return ConfigReadResponse(entries=entries, warnings=warnings)


@router.post("/validate", response_model=list[ValidationResult])
async def validate_config(body: ValidateRequest) -> list[ValidationResult]:
    """
    Validate configuration values before writing.

    Returns per-key validation results with human-readable messages.
    """
    results: list[ValidationResult] = []

    for key, value in body.settings.items():
        if key not in ALLOWED_KEYS:
            results.append(
                ValidationResult(
                    key=key,
                    value=value,
                    valid=False,
                    message=f"Key '{key}' is not in the allowed list.",
                )
            )
            continue

        valid = True
        message = "OK"

        if key == "PUBLIC_DOMAIN" and value:
            # Basic domain/IP pattern
            pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-\.]{0,253}[a-zA-Z0-9])?$"
            if not re.match(pattern, value):
                valid = False
                message = "Invalid domain name format."

        elif key == "PUBLIC_PORT":
            try:
                port = int(value)
                if not (1 <= port <= 65535):
                    raise ValueError
            except ValueError:
                valid = False
                message = "Port must be an integer between 1 and 65535."

        elif key == "VRAM_LIMIT_GB":
            try:
                gb = float(value)
                if gb < 1 or gb > 80:
                    raise ValueError
            except ValueError:
                valid = False
                message = "VRAM limit must be a number between 1 and 80 GB."

        elif key in ("LABEL_STUDIO_URL", "MLFLOW_TRACKING_URI", "CVAT_URL"):
            if value and not re.match(r"^https?://", value):
                valid = False
                message = "Must be a valid HTTP/HTTPS URL."

        elif key == "DATA_ROOT":
            p = Path(value).expanduser()
            if not p.is_absolute():
                valid = False
                message = "DATA_ROOT must be an absolute path."

        results.append(
            ValidationResult(key=key, value=value, valid=valid, message=message)
        )

    return results


@router.post("/configure", response_model=ConfigWriteResponse)
async def write_config(body: ConfigWriteRequest) -> ConfigWriteResponse:
    """
    Persist configuration to .env.

    Only keys present in ALLOWED_KEYS are written. Unknown keys are skipped.
    Sensitive values are never echoed back in the response.
    """
    written: list[str] = []
    skipped: list[str] = []
    updates: dict[str, str] = {}

    for key, value in body.settings.items():
        if key not in ALLOWED_KEYS:
            skipped.append(key)
            continue
        updates[key] = value
        written.append(key)

    if body.mark_setup_complete:
        updates["SETUP_COMPLETED"] = "true"

    _write_env_file(ENV_FILE, updates)

    # Invalidate the settings singleton so the next request picks up new values
    try:
        from backend.config import get_settings

        get_settings.cache_clear()
    except Exception:
        pass

    return ConfigWriteResponse(
        written=sorted(written),
        skipped=sorted(skipped),
        env_path=str(ENV_FILE.resolve()),
    )


@router.post("/reset", response_model=dict)
async def reset_setup_status() -> dict[str, str]:
    """
    Clear SETUP_COMPLETED flag so the wizard runs again on next visit.

    Does NOT clear other settings — only resets the completion marker.
    """
    env = _read_env_file(ENV_FILE)
    env.pop("SETUP_COMPLETED", None)
    _write_env_file(ENV_FILE, {"SETUP_COMPLETED": "false"})
    return {"status": "reset", "message": "Setup wizard will appear on next visit."}
