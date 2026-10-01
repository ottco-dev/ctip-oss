"""
Session-grouped train/val/test splits — leakage prevention.

Frames from one microscope session (same plant, slide, magnification, lighting, often near-identical crops) must
never end up on both sides of a split: a model would be rewarded for recognising the session, not the trichomes,
and validation/test metrics would be inflated. All tasks of a session therefore go to exactly one split.

Session key, in order of preference:
  1. ``data["session"]`` / ``data["session_id"]`` / ``meta["session"]`` set on the Label Studio task
  2. the image's parent directory, when it is not a generic upload folder
  3. the file name without its trailing frame / sequence number ("plantA_slide2_0042.jpg" → "plantA_slide2")
  4. the task itself (a session of one) — reported, because then nothing protects against leakage

Splitting is deterministic for a given seed and set of sessions.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse

# folders that say nothing about the imaging session
_GENERIC_DIRS = {"", ".", "upload", "uploads", "images", "image", "data", "media", "files", "local-files", "tasks", "raw"}
# trailing frame / sequence numbers, Label Studio upload hashes ("1a2b3c4d-") and copy markers
_FRAME_SUFFIX = re.compile(r"(?:[_\-. ](?:frame|img|image|shot)?[_\-. ]?\d+|(?:frame|img|image|shot)\d+|\s*\(\d+\))$",
                           re.IGNORECASE)
_LS_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")

SPLITS = ("train", "val", "test")


@dataclass
class SessionSplit:
    splits: dict[str, list[dict]]
    sessions: dict[str, list[str]]                       # split -> session keys
    fallback_sessions: int = 0                           # tasks that formed a session of their own
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "  ".join(f"{s}: {len(self.splits[s])} images / {len(self.sessions[s])} sessions" for s in SPLITS)


def _image_path(url: str) -> PurePosixPath:
    """Path part of a Label Studio image reference (/data/local-files/?d=…, /data/upload/…, s3://…, plain path)."""
    parsed = urlparse(url)
    if parsed.query:
        d = parse_qs(parsed.query).get("d")
        if d:
            return PurePosixPath(unquote(d[0]))
    return PurePosixPath(unquote(parsed.path or url))


def session_key(task: dict) -> tuple[str, bool]:
    """(session key, True if it is only the task itself)."""
    data = task.get("data") or {}
    meta = task.get("meta") or {}
    for value in (data.get("session"), data.get("session_id"), meta.get("session"), meta.get("session_id")):
        if value not in (None, ""):
            return f"session:{value}", False

    path = _image_path(str(data.get("image", "")))
    parent = path.parent.name
    # Label Studio uploads land in /data/upload/<project id>/ - that folder says nothing about the session
    generic = parent.lower() in _GENERIC_DIRS or (parent.isdigit() and path.parent.parent.name.lower() in _GENERIC_DIRS)
    if not generic:
        return f"dir:{path.parent}", False

    stem = _LS_UPLOAD_PREFIX.sub("", path.stem)
    base = stem
    for _ in range(3):                      # "run7_frame0012 (2)" -> "run7"
        base = _FRAME_SUFFIX.sub("", base)
    base = base.strip("_-. ")
    if base and base != stem:
        return f"name:{base.lower()}", False
    return f"task:{task.get('id')}", True


def split_by_session(tasks: list[dict], train_ratio: float, val_ratio: float, seed: int = 42) -> SessionSplit:
    """
    Assign whole sessions to train / val / test so that the image counts approach the ratios.

    Sessions are shuffled with ``seed`` and filled into train, then val, then test. With three or more sessions
    every split gets at least one.
    """
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio > 1:
        raise ValueError(f"invalid split ratios: train={train_ratio}, val={val_ratio}")

    groups: dict[str, list[dict]] = {}
    fallback = 0
    for task in tasks:
        key, alone = session_key(task)
        fallback += alone
        groups.setdefault(key, []).append(task)

    keys = sorted(groups)                          # stable order before the seeded shuffle
    random.Random(seed).shuffle(keys)

    n = len(tasks)
    targets = {"train": n * train_ratio, "val": n * val_ratio}
    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    sessions: dict[str, list[str]] = {s: [] for s in SPLITS}

    def put(split: str, key: str) -> None:
        splits[split].extend(groups[key])
        sessions[split].append(key)

    remaining = list(keys)
    # with three or more sessions, hold back one session each for val and test
    for split, reserve in (("train", 2), ("val", 1)):
        reserve = reserve if len(keys) >= 3 else 0
        while len(remaining) > reserve and len(splits[split]) < targets[split]:
            put(split, remaining.pop(0))
    for key in remaining:
        put("test", key)

    # guarantee non-empty val/test when there are at least three sessions
    if len(keys) >= 3:
        for split in ("val", "test"):
            if not splits[split]:
                donor = max(("train", "val", "test"), key=lambda s: len(sessions[s]))
                key = sessions[donor].pop()
                for t in groups[key]:
                    splits[donor].remove(t)
                put(split, key)

    warnings: list[str] = []
    if len(keys) < 3:
        warnings.append(f"Only {len(keys)} session(s) found — val/test cannot be independent of train. "
                        "Collect images from more sessions before trusting the metrics.")
    if fallback:
        warnings.append(f"{fallback} image(s) have no session information (no session field, folder or file-name "
                        "series) and were treated as their own session — near-duplicates among them can leak. "
                        "Set data.session on the Label Studio tasks.")
    return SessionSplit(splits=splits, sessions=sessions, fallback_sessions=fallback, warnings=warnings)
