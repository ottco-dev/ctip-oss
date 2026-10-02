"""Database tables of the compute coordinator."""

from __future__ import annotations

import json
import time
import uuid

from sqlmodel import Field, SQLModel


def _id() -> str:
    return uuid.uuid4().hex


class ComputeWorker(SQLModel, table=True):
    __tablename__ = "compute_workers"

    id: str = Field(default_factory=_id, primary_key=True)
    name: str
    token_hash: str = Field(index=True)
    created_at: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    revoked: bool = False
    capabilities_json: str = "{}"
    free_json: str = "{}"

    @property
    def capabilities(self) -> dict:
        return json.loads(self.capabilities_json or "{}")


class ComputeEnrollment(SQLModel, table=True):
    __tablename__ = "compute_enrollments"

    id: str = Field(default_factory=_id, primary_key=True)
    token_hash: str = Field(index=True)
    note: str = ""
    created_at: float = Field(default_factory=time.time)
    expires_at: float
    used_at: float | None = None
    worker_id: str | None = None


class ComputeJob(SQLModel, table=True):
    __tablename__ = "compute_jobs"

    id: str = Field(default_factory=_id, primary_key=True)
    name: str = ""
    kind: str = Field(index=True)
    spec_json: str
    requirements_json: str
    status: str = Field(default="queued", index=True)
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 3
    worker_id: str | None = Field(default=None, index=True)
    lease_expires_at: float | None = None
    cancel_requested: bool = False
    progress: float = 0.0
    epoch: int | None = None
    metrics_json: str = "{}"
    message: str = ""
    checkpoint_artifact: str | None = None
    result_artifacts_json: str = "[]"
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    updated_at: float = Field(default_factory=time.time)


class ComputeArtifact(SQLModel, table=True):
    __tablename__ = "compute_artifacts"

    id: str = Field(default_factory=_id, primary_key=True)
    kind: str                     # dataset | model | checkpoint | result | log
    filename: str
    sha256: str = Field(index=True)
    size: int
    path: str
    job_id: str | None = Field(default=None, index=True)
    created_at: float = Field(default_factory=time.time)
