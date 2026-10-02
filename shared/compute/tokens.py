"""Bearer tokens for agents: random, shown once, stored only as a SHA-256 hash, compared in constant time."""

from __future__ import annotations

import hashlib
import hmac
import secrets

ENROLL_PREFIX = "ctipe_"
WORKER_PREFIX = "ctipw_"


def new_token(prefix: str) -> str:
    return prefix + secrets.token_urlsafe(32)          # 256 bits


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def matches(token: str, stored_hash: str) -> bool:
    return hmac.compare_digest(token_hash(token), stored_hash)


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
