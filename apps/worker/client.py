"""HTTPS client for the coordinator: worker-token auth, TLS required, retries with backoff, SHA-256 on every file."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from shared.compute.schemas import (
    EnrollRequest,
    EnrollResponse,
    Finish,
    Heartbeat,
    HeartbeatResponse,
    Lease,
    LeaseRequest,
)
from shared.compute.tokens import sha256_file

API = "/api/v1/compute/agent"
LOCAL = {"localhost", "127.0.0.1", "::1"}


class CoordinatorError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def check_url(url: str) -> str:
    """Only https - plain http only to this machine (tests, a coordinator on localhost)."""
    u = urlparse(url.rstrip("/"))
    if u.scheme == "https" or (u.scheme == "http" and u.hostname in LOCAL):
        return url.rstrip("/")
    raise ValueError(f"refusing insecure coordinator URL {url!r}: use https://")


class Coordinator:
    def __init__(self, server: str, token: str = "", ca_bundle: str = "", timeout: float = 30.0, retries: int = 5) -> None:
        self.base = check_url(server) + API
        self.s = requests.Session()
        self.s.verify = ca_bundle or True
        self.s.headers["User-Agent"] = "ctip-worker/1"
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"
        self.timeout, self.retries = timeout, retries

    def _call(self, method: str, path: str, retries: int | None = None, **kw) -> requests.Response:
        retries = self.retries if retries is None else retries
        delay = 2.0
        for attempt in range(retries + 1):
            try:
                r = self.s.request(method, self.base + path, timeout=self.timeout, **kw)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == retries:
                    raise CoordinatorError(0, f"coordinator unreachable: {e}") from e
            else:
                if r.status_code < 500:
                    if r.status_code >= 400:
                        try:
                            detail = r.json().get("detail", r.text)
                        except ValueError:
                            detail = r.text
                        raise CoordinatorError(r.status_code, str(detail)[:500])
                    return r
                if attempt == retries:
                    raise CoordinatorError(r.status_code, r.text[:500])
            time.sleep(delay + random.uniform(0, delay / 2))   # jitter: many agents must not retry in lockstep
            delay = min(delay * 2, 60)
        raise CoordinatorError(0, "unreachable")

    def enroll(self, req: EnrollRequest) -> EnrollResponse:
        return EnrollResponse.model_validate(self._call("POST", "/enroll", json=req.model_dump(mode="json")).json())

    def heartbeat(self, hb: Heartbeat) -> HeartbeatResponse:
        return HeartbeatResponse.model_validate(self._call("POST", "/heartbeat", json=hb.model_dump(mode="json")).json())

    def lease(self, req: LeaseRequest) -> Lease | None:
        r = self._call("POST", "/lease", json=req.model_dump(mode="json"))
        return None if r.status_code == 204 or not r.content or r.json() is None else Lease.model_validate(r.json())

    def download(self, artifact_id: str, dest: Path) -> Path:
        """Stream to a temp file, verify the SHA-256 the coordinator announced, then move into place."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self.s.get(self.base + f"/artifacts/{artifact_id}", stream=True, timeout=self.timeout) as r:
            if r.status_code >= 400:
                raise CoordinatorError(r.status_code, r.text[:300])
            expected = r.headers.get("X-Content-SHA256", "")
            with tmp.open("wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)
        got = sha256_file(tmp)
        if not expected or got != expected:
            tmp.unlink(missing_ok=True)
            raise CoordinatorError(0, f"checksum mismatch for artifact {artifact_id}")
        os.replace(tmp, dest)
        return dest

    def upload(self, job_id: str, path: Path, kind: str) -> dict:
        digest = sha256_file(path)
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                with path.open("rb") as fh:                  # reopened per attempt: a retry sends the whole file
                    return self._call("POST", f"/jobs/{job_id}/artifacts", retries=0, files={"file": (path.name, fh)},
                                      data={"kind": kind, "sha256": digest}).json()
            except CoordinatorError as e:
                if (0 < e.status < 500) or attempt == self.retries:
                    raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise CoordinatorError(0, "upload failed")

    def finish(self, job_id: str, f: Finish) -> dict:
        return self._call("POST", f"/jobs/{job_id}/finish", json=f.model_dump(mode="json")).json()

    def cancelled(self, job_id: str) -> None:
        self._call("POST", f"/jobs/{job_id}/cancelled")
