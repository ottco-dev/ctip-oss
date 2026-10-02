"""
Compute API.

/api/v1/compute/*        admin: enrolment tokens, workers, jobs, artifacts (protected by the CTIP API token)
/api/v1/compute/agent/*  agents: enrol, heartbeat, lease, artifacts, finish (own worker tokens; excluded from the
                         API-token middleware, never accept the admin token)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, TypeAdapter
from sqlmodel import Session, col, select

from backend.compute import service
from backend.compute.models import ComputeArtifact, ComputeJob, ComputeWorker
from backend.config import get_settings
from backend.database import get_session
from shared.compute.schemas import (
    EnrollRequest,
    EnrollResponse,
    Finish,
    Heartbeat,
    HeartbeatResponse,
    JobSpec,
    Lease,
    LeaseRequest,
    Requirements,
)

router = APIRouter(prefix="/compute", tags=["compute"])
agent_router = APIRouter(prefix="/compute/agent", tags=["compute-agent"])
_spec = TypeAdapter(JobSpec)


def artifact_root() -> Path:
    return Path(get_settings().data_root).resolve() / "compute" / "artifacts"


def _raise(e: service.ComputeError):
    raise HTTPException(status_code=e.status, detail=e.detail) from e


# ── admin ────────────────────────────────────────────────────────────────────

class EnrollmentIn(BaseModel):
    note: str = Field("", max_length=200)
    ttl_hours: float = Field(24, gt=0, le=24 * 30)


@router.post("/enrollments")
def create_enrollment(body: EnrollmentIn, db: Session = Depends(get_session)) -> dict:
    """One-time token for a new agent (shown once)."""
    token, e = service.create_enrollment(db, body.note, body.ttl_hours)
    return {"token": token, "expires_at": e.expires_at, "note": e.note}


@router.get("/workers")
def workers(db: Session = Depends(get_session)) -> list[dict]:
    return [service.worker_view(w) for w in db.exec(select(ComputeWorker).order_by(col(ComputeWorker.created_at))).all()]


@router.post("/workers/{worker_id}/revoke")
def revoke(worker_id: str, db: Session = Depends(get_session)) -> dict:
    w = db.get(ComputeWorker, worker_id)
    if w is None:
        raise HTTPException(404, "no such worker")
    w.revoked = True
    db.add(w)
    db.commit()
    for job in db.exec(select(ComputeJob).where(ComputeJob.worker_id == worker_id)).all():
        if job.status in ("leased", "running"):
            job.lease_expires_at = 0                      # requeued by the next maintenance pass
            db.add(job)
    db.commit()
    return service.worker_view(w)


class JobIn(BaseModel):
    spec: dict[str, Any]
    requirements: Requirements = Field(default_factory=Requirements)
    name: str = Field("", max_length=120)
    priority: int = Field(0, ge=-10, le=10)
    max_attempts: int = Field(3, ge=1, le=10)


@router.post("/jobs")
def submit(body: JobIn, db: Session = Depends(get_session)) -> dict:
    try:
        spec = _spec.validate_python(body.spec).model_dump()
    except Exception as e:
        raise HTTPException(422, f"invalid job spec: {e}") from e
    try:
        return service.job_view(service.submit(db, spec, body.requirements, body.name, body.priority, body.max_attempts))
    except service.ComputeError as e:
        _raise(e)


@router.get("/jobs")
def jobs(status: str | None = None, limit: int = 100, db: Session = Depends(get_session)) -> list[dict]:
    q = select(ComputeJob).order_by(col(ComputeJob.created_at).desc()).limit(min(limit, 500))
    if status:
        q = q.where(ComputeJob.status == status)
    return [service.job_view(j) for j in db.exec(q).all()]


@router.get("/jobs/{job_id}")
def job(job_id: str, db: Session = Depends(get_session)) -> dict:
    j = db.get(ComputeJob, job_id)
    if j is None:
        raise HTTPException(404, "no such job")
    return service.job_view(j)


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, db: Session = Depends(get_session)) -> dict:
    try:
        return service.job_view(service.cancel(db, job_id))
    except service.ComputeError as e:
        _raise(e)


@router.post("/artifacts")
def upload_artifact(file: UploadFile = File(...), kind: str = Form("dataset"), sha256: str | None = Form(None),
                    db: Session = Depends(get_session)) -> dict:
    """Upload an input (zipped YOLO dataset with dataset.yaml at its root, or .pt weights)."""
    if kind not in ("dataset", "model"):
        raise HTTPException(422, "admins upload dataset or model artifacts")
    try:
        a = service.store_artifact(db, artifact_root(), file.file, file.filename or kind, kind, sha256)
    except service.ComputeError as e:
        _raise(e)
    return {"id": a.id, "kind": a.kind, "filename": a.filename, "sha256": a.sha256, "size": a.size}


@router.get("/artifacts/{artifact_id}")
def download_artifact(artifact_id: str, db: Session = Depends(get_session)) -> FileResponse:
    a = db.get(ComputeArtifact, artifact_id)
    if a is None:
        raise HTTPException(404, "no such artifact")
    return FileResponse(a.path, filename=a.filename, headers={"X-Content-SHA256": a.sha256})


# ── agents ───────────────────────────────────────────────────────────────────

def _worker(authorization: str | None = Header(None), db: Session = Depends(get_session)) -> ComputeWorker:
    token = authorization[7:].strip() if authorization and authorization.startswith("Bearer ") else None
    try:
        return service.authenticate(db, token)
    except service.ComputeError as e:
        _raise(e)


@agent_router.post("/enroll", response_model=EnrollResponse)
def enroll(body: EnrollRequest, db: Session = Depends(get_session)) -> EnrollResponse:
    try:
        w, token = service.enroll(db, body)
    except service.ComputeError as e:
        _raise(e)
    return EnrollResponse(worker_id=w.id, worker_token=token, heartbeat_s=service.HEARTBEAT_S)


@agent_router.post("/heartbeat", response_model=HeartbeatResponse)
def heartbeat(body: Heartbeat, w: ComputeWorker = Depends(_worker), db: Session = Depends(get_session)) -> HeartbeatResponse:
    cancel, lost = service.heartbeat(db, w, body)
    return HeartbeatResponse(cancel=cancel, lost=lost, lease_s=service.LEASE_S)


@agent_router.post("/lease", response_model=Lease | None)
def lease(body: LeaseRequest, response: Response, w: ComputeWorker = Depends(_worker), db: Session = Depends(get_session)):
    job = service.lease(db, w, body.free)
    if job is None:
        response.status_code = 204
        return None
    return Lease(job_id=job.id, spec=json.loads(job.spec_json), attempt=job.attempts, lease_s=service.LEASE_S,
                 checkpoint=job.checkpoint_artifact)


@agent_router.get("/artifacts/{artifact_id}")
def agent_download(artifact_id: str, w: ComputeWorker = Depends(_worker), db: Session = Depends(get_session)) -> FileResponse:
    try:
        a = service.worker_may_read(db, w, artifact_id)
    except service.ComputeError as e:
        _raise(e)
    return FileResponse(a.path, filename=a.filename, headers={"X-Content-SHA256": a.sha256})


@agent_router.post("/jobs/{job_id}/artifacts")
def agent_upload(job_id: str, request: Request, file: UploadFile = File(...), kind: str = Form(...),
                 sha256: str = Form(...), w: ComputeWorker = Depends(_worker), db: Session = Depends(get_session)) -> dict:
    if kind not in ("checkpoint", "model", "result", "log"):
        raise HTTPException(422, "agents upload checkpoint, model, result or log artifacts")
    try:
        service._own(db, w, job_id)
        a = service.store_artifact(db, artifact_root(), file.file, file.filename or kind, kind, sha256, job_id)
        service.attach_upload(db, w, job_id, a)
    except service.ComputeError as e:
        _raise(e)
    return {"id": a.id, "sha256": a.sha256, "size": a.size}


@agent_router.post("/jobs/{job_id}/finish")
def agent_finish(job_id: str, body: Finish, w: ComputeWorker = Depends(_worker), db: Session = Depends(get_session)) -> dict:
    try:
        return service.job_view(service.finish(db, w, job_id, body))
    except service.ComputeError as e:
        _raise(e)


@agent_router.post("/jobs/{job_id}/cancelled")
def agent_cancelled(job_id: str, w: ComputeWorker = Depends(_worker), db: Session = Depends(get_session)) -> dict:
    try:
        service.cancelled_by_worker(db, w, job_id)
    except service.ComputeError as e:
        _raise(e)
    return {"ok": True}
