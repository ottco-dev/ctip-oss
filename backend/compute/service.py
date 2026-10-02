"""Coordinator logic: enrolment, leasing, heartbeats, expiry/resume, completion, artifacts. Framework-free."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from typing import BinaryIO

from sqlmodel import Session, col, select

from backend.compute.models import ComputeArtifact, ComputeEnrollment, ComputeJob, ComputeWorker
from shared.compute import tokens
from shared.compute.matching import fits
from shared.compute.schemas import (
    ACTIVE,
    Capabilities,
    EnrollRequest,
    Finish,
    Free,
    Heartbeat,
    JobStatus,
    Requirements,
)

LEASE_S = 90            # a job is lost when the worker has not reported for this long
HEARTBEAT_S = 20
OFFLINE_S = 120
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
ARTIFACT_KINDS = {"dataset", "model", "checkpoint", "result", "log"}

_lease_lock = threading.Lock()   # one coordinator process; leasing must not hand a job out twice


class ComputeError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


# ── enrolment ────────────────────────────────────────────────────────────────

def create_enrollment(db: Session, note: str = "", ttl_hours: float = 24) -> tuple[str, ComputeEnrollment]:
    token = tokens.new_token(tokens.ENROLL_PREFIX)
    e = ComputeEnrollment(token_hash=tokens.token_hash(token), note=note[:200], expires_at=time.time() + ttl_hours * 3600)
    db.add(e)
    db.commit()
    db.refresh(e)
    return token, e


def enroll(db: Session, req: EnrollRequest) -> tuple[ComputeWorker, str]:
    """Exchange a one-time enrolment token for a worker identity and its own long-lived token."""
    e = db.exec(select(ComputeEnrollment).where(ComputeEnrollment.token_hash == tokens.token_hash(req.enrollment_token))).first()
    if e is None or e.used_at is not None or e.expires_at < time.time():
        raise ComputeError(403, "enrolment token unknown, used or expired")
    worker_token = tokens.new_token(tokens.WORKER_PREFIX)
    w = ComputeWorker(name=req.name, token_hash=tokens.token_hash(worker_token), capabilities_json=req.capabilities.model_dump_json())
    e.used_at, e.worker_id = time.time(), w.id
    db.add(w)
    db.add(e)
    db.commit()
    db.refresh(w)
    return w, worker_token


def authenticate(db: Session, token: str | None) -> ComputeWorker:
    if not token or not token.startswith(tokens.WORKER_PREFIX):
        raise ComputeError(401, "worker token missing")
    w = db.exec(select(ComputeWorker).where(ComputeWorker.token_hash == tokens.token_hash(token))).first()
    if w is None or not tokens.matches(token, w.token_hash):
        raise ComputeError(401, "worker token invalid")
    if w.revoked:
        raise ComputeError(403, "worker revoked")
    return w


# ── jobs ─────────────────────────────────────────────────────────────────────

def submit(db: Session, spec: dict, requirements: Requirements, name: str = "", priority: int = 0, max_attempts: int = 3) -> ComputeJob:
    for key in ("dataset", "model_artifact"):
        if key in spec and db.get(ComputeArtifact, spec[key]) is None:
            raise ComputeError(422, f"{key}: unknown artifact {spec[key]}")
    if spec.get("distributed") and requirements.gpus < 2:
        requirements = requirements.model_copy(update={"gpus": 2})
    job = ComputeJob(name=name[:120], kind=spec["kind"], spec_json=json.dumps(spec), requirements_json=requirements.model_dump_json(),
                     priority=priority, max_attempts=max_attempts)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def lease(db: Session, worker: ComputeWorker, free: Free) -> ComputeJob | None:
    """Hand the best fitting queued job to the worker (priority, then age). One active job per worker."""
    caps = Capabilities.model_validate_json(worker.capabilities_json)
    worker.last_seen, worker.free_json = time.time(), free.model_dump_json()
    db.add(worker)
    with _lease_lock:
        busy = db.exec(select(ComputeJob).where(ComputeJob.worker_id == worker.id, col(ComputeJob.status).in_(list(ACTIVE)))).first()
        if busy is not None:
            db.commit()
            return busy if busy.status == JobStatus.LEASED else None
        queued = db.exec(select(ComputeJob).where(ComputeJob.status == JobStatus.QUEUED)
                         .order_by(col(ComputeJob.priority).desc(), col(ComputeJob.created_at))).all()
        for job in queued:
            if fits(Requirements.model_validate_json(job.requirements_json), job.kind, caps, free):
                job.status, job.worker_id = JobStatus.LEASED, worker.id
                job.attempts += 1
                job.lease_expires_at = time.time() + LEASE_S
                job.updated_at = time.time()
                db.add(job)
                db.commit()
                db.refresh(job)
                return job
        db.commit()
    return None


def _own(db: Session, worker: ComputeWorker, job_id: str) -> ComputeJob:
    job = db.get(ComputeJob, job_id)
    if job is None or job.worker_id != worker.id or job.status not in ACTIVE:
        raise ComputeError(409, f"job {job_id} is not leased to this worker")
    return job


def heartbeat(db: Session, worker: ComputeWorker, hb: Heartbeat) -> tuple[list[str], list[str]]:
    """Record liveness and progress, extend leases. Returns (cancelled by admin, lost lease) job ids."""
    now = time.time()
    worker.last_seen = now
    if hb.capabilities:
        worker.capabilities_json = hb.capabilities.model_dump_json()
    if hb.free:
        worker.free_json = hb.free.model_dump_json()
    db.add(worker)
    cancel: list[str] = []
    lost: list[str] = []
    for p in hb.jobs:
        job = db.get(ComputeJob, p.job_id)
        if job is None or job.worker_id != worker.id or job.status not in ACTIVE:
            lost.append(p.job_id)                         # lease expired meanwhile: someone else may run it
            continue
        if job.cancel_requested:
            cancel.append(job.id)
        if job.status == JobStatus.LEASED:
            job.status, job.started_at = JobStatus.RUNNING, job.started_at or now
        job.progress, job.epoch, job.message = p.progress, p.epoch, p.message
        if p.metrics:
            job.metrics_json = json.dumps(p.metrics)
        job.lease_expires_at, job.updated_at = now + LEASE_S, now
        db.add(job)
    db.commit()
    return cancel, lost


def finish(db: Session, worker: ComputeWorker, job_id: str, f: Finish) -> ComputeJob:
    job = _own(db, worker, job_id)
    now = time.time()
    if f.metrics:
        job.metrics_json = json.dumps(f.metrics)
    job.message, job.updated_at, job.lease_expires_at = f.message[:2000], now, None
    if f.status == "completed":
        job.status, job.progress, job.finished_at = JobStatus.COMPLETED, 1.0, now
    elif f.status == "yielded":                           # worker needs its machine back: resume elsewhere/later
        job.status, job.worker_id = JobStatus.QUEUED, None
        job.attempts = max(0, job.attempts - 1)           # yielding is not a failure
    elif f.retryable and job.attempts < job.max_attempts:
        job.status, job.worker_id = JobStatus.QUEUED, None
    else:
        job.status, job.finished_at = JobStatus.FAILED, now
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def expire(db: Session, now: float | None = None) -> list[str]:
    """Requeue (with checkpoint) or fail jobs whose worker went silent. Returns affected job ids."""
    now = now or time.time()
    affected = []
    for job in db.exec(select(ComputeJob).where(col(ComputeJob.status).in_(list(ACTIVE)), col(ComputeJob.lease_expires_at) < now)).all():
        job.worker_id, job.lease_expires_at, job.updated_at = None, None, now
        if job.attempts >= job.max_attempts:
            job.status, job.finished_at, job.message = JobStatus.FAILED, now, "worker lost too often"
        else:
            job.status, job.message = JobStatus.QUEUED, "worker lost - requeued" + (" (resumes from checkpoint)" if job.checkpoint_artifact else "")
        db.add(job)
        affected.append(job.id)
    db.commit()
    return affected


def cancel(db: Session, job_id: str) -> ComputeJob:
    job = db.get(ComputeJob, job_id)
    if job is None:
        raise ComputeError(404, "no such job")
    if job.status == JobStatus.QUEUED:
        job.status, job.finished_at = JobStatus.CANCELLED, time.time()
    elif job.status in ACTIVE:
        job.cancel_requested = True                       # the worker stops it at its next heartbeat
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def cancelled_by_worker(db: Session, worker: ComputeWorker, job_id: str) -> None:
    job = _own(db, worker, job_id)
    job.status, job.finished_at, job.worker_id, job.message = JobStatus.CANCELLED, time.time(), None, "cancelled"
    db.add(job)
    db.commit()


# ── artifacts ────────────────────────────────────────────────────────────────

def store_artifact(db: Session, root: Path, stream: BinaryIO, filename: str, kind: str, expected_sha256: str | None = None,
                   job_id: str | None = None) -> ComputeArtifact:
    if kind not in ARTIFACT_KINDS:
        raise ComputeError(422, f"unknown artifact kind {kind}")
    root.mkdir(parents=True, exist_ok=True)
    art = ComputeArtifact(kind=kind, filename=Path(filename).name[:200] or "file", sha256="", size=0, path="", job_id=job_id)
    tmp = root / f".{art.id}.part"
    h, size = hashlib.sha256(), 0
    try:
        with tmp.open("wb") as out:
            while block := stream.read(1 << 20):
                size += len(block)
                if size > MAX_ARTIFACT_BYTES:
                    raise ComputeError(413, f"artifact larger than {MAX_ARTIFACT_BYTES // 2**20} MB")
                h.update(block)
                out.write(block)
        digest = h.hexdigest()
        if expected_sha256 and expected_sha256.lower() != digest:
            raise ComputeError(422, "sha256 mismatch - upload corrupted")
        final = root / art.id
        shutil.move(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    art.sha256, art.size, art.path = digest, size, str(final)
    db.add(art)
    db.commit()
    db.refresh(art)
    return art


def worker_may_read(db: Session, worker: ComputeWorker, artifact_id: str) -> ComputeArtifact:
    """A worker may download only the inputs and the checkpoint of a job it currently holds."""
    art = db.get(ComputeArtifact, artifact_id)
    if art is None:
        raise ComputeError(404, "no such artifact")
    for job in db.exec(select(ComputeJob).where(ComputeJob.worker_id == worker.id, col(ComputeJob.status).in_(list(ACTIVE)))).all():
        spec = json.loads(job.spec_json)
        if artifact_id in (spec.get("dataset"), spec.get("model_artifact"), job.checkpoint_artifact):
            return art
    raise ComputeError(403, "artifact not part of a job leased to this worker")


def attach_upload(db: Session, worker: ComputeWorker, job_id: str, art: ComputeArtifact) -> ComputeJob:
    job = _own(db, worker, job_id)
    if art.kind == "checkpoint":
        old = db.get(ComputeArtifact, job.checkpoint_artifact) if job.checkpoint_artifact else None
        job.checkpoint_artifact = art.id
        if old is not None:                               # keep only the newest checkpoint on disk
            Path(old.path).unlink(missing_ok=True)
            db.delete(old)
    else:
        results = json.loads(job.result_artifacts_json)
        results.append(art.id)
        job.result_artifacts_json = json.dumps(results)
    job.updated_at = time.time()
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


# ── views ────────────────────────────────────────────────────────────────────

def worker_view(w: ComputeWorker, now: float | None = None) -> dict:
    now = now or time.time()
    return {"id": w.id, "name": w.name, "online": not w.revoked and now - w.last_seen < OFFLINE_S, "revoked": w.revoked,
            "last_seen": w.last_seen, "created_at": w.created_at, "capabilities": w.capabilities,
            "free": json.loads(w.free_json or "{}")}


def job_view(j: ComputeJob) -> dict:
    return {"id": j.id, "name": j.name, "kind": j.kind, "status": j.status, "priority": j.priority, "attempts": j.attempts,
            "max_attempts": j.max_attempts, "worker_id": j.worker_id, "progress": j.progress, "epoch": j.epoch,
            "metrics": json.loads(j.metrics_json or "{}"), "message": j.message, "spec": json.loads(j.spec_json),
            "requirements": json.loads(j.requirements_json), "checkpoint": j.checkpoint_artifact,
            "results": json.loads(j.result_artifacts_json or "[]"), "cancel_requested": j.cancel_requested,
            "created_at": j.created_at, "started_at": j.started_at, "finished_at": j.finished_at, "updated_at": j.updated_at}
