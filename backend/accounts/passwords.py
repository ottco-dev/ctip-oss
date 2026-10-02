"""Password hashing with scrypt (stdlib): memory-hard, per-user salt, parameters stored with the hash."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

N, R, P = 2**15, 8, 1                  # ~32 MB per hash; a few hundred ms on a VPS core
MIN_LENGTH = 10


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, maxmem=64 * 1024 * 1024, dklen=32)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"scrypt${N}${R}${P}${b64(salt)}${b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, dk = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(dk)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             maxmem=64 * 1024 * 1024, dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, expected)


def password_problem(password: str, username: str = "") -> str | None:
    """Why a new password is not acceptable, or None."""
    if len(password) < MIN_LENGTH:
        return f"at least {MIN_LENGTH} characters"
    if username and username.lower() in password.lower():
        return "must not contain the username"
    if len(set(password)) < 5:
        return "too repetitive"
    return None
