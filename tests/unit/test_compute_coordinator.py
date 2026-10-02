"""Compute coordinator: enrolment, auth, leasing by capability, heartbeats, lost workers, resume, yield, artifacts."""

from __future__ import annotations

import io
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from backend.compute import service
from backend.compute.models import ComputeJob
from shared.compute.schemas import (
    Capabilities,
    EnrollRequest,
    Finish,
    Free,
    Heartbeat,
    JobProgress,
    Requirements,
)

GPU = Capabilities(backend="cuda", device="RTX 4060", gpus=1, vram_gb=8, ram_gb=16, cpu_cores=16)
CPU = Capabilities(backend="cpu", device="Ryzen", gpus=0, vram_gb=0, ram_gb=32, cpu_cores=16)
FREE_GPU = Free(vram_gb=6, ram_gb=8)


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from backend.compute import models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def worker(db, caps=GPU, name="pc"):
    token, _ = service.create_enrollment(db, "test")
    return service.enroll(db, EnrollRequest(enrollment_token=token, name=name, capabilities=caps))


def bench(db, req=None, prio=0):
    return service.submit(db, {"kind": "detection_benchmark", "model": "yolo11s", "imgsz": 640, "runs": 5},
                          req or Requirements(), priority=prio)


class TestEnrolment:
    def test_token_once_and_auth(self, db):
        token, _ = service.create_enrollment(db)
        w, wt = service.enroll(db, EnrollRequest(enrollment_token=token, name="a", capabilities=GPU))
        assert wt.startswith("ctipw_") and service.authenticate(db, wt).id == w.id
        with pytest.raises(service.ComputeError, match="used"):
            service.enroll(db, EnrollRequest(enrollment_token=token, name="b", capabilities=GPU))
        with pytest.raises(service.ComputeError):
            service.authenticate(db, "ctipw_wrong")
        w.revoked = True
        db.add(w)
        db.commit()
        with pytest.raises(service.ComputeError, match="revoked"):
            service.authenticate(db, wt)

    def test_expired_token(self, db):
        token, e = service.create_enrollment(db, ttl_hours=1)
        e.expires_at = time.time() - 1
        db.add(e)
        db.commit()
        with pytest.raises(service.ComputeError, match="expired"):
            service.enroll(db, EnrollRequest(enrollment_token=token, name="x", capabilities=GPU))


class TestLeasing:
    def test_requirements_match(self, db):
        gpu_job = bench(db, Requirements(backends=["cuda"], min_vram_gb=4))
        cpu_w, _ = worker(db, CPU, "cpu")
        assert service.lease(db, cpu_w, Free(ram_gb=8)) is None              # wrong backend
        gpu_w, _ = worker(db, GPU, "gpu")
        assert service.lease(db, gpu_w, Free(vram_gb=2, ram_gb=8)) is None   # not enough free VRAM now
        assert service.lease(db, gpu_w, FREE_GPU).id == gpu_job.id

    def test_priority_then_age_and_one_job_per_worker(self, db):
        low, high = bench(db), bench(db, prio=5)
        w, _ = worker(db)
        first = service.lease(db, w, FREE_GPU)
        assert first.id == high.id
        assert service.lease(db, w, FREE_GPU).id == high.id                  # still leased, not started: same job again
        service.heartbeat(db, w, Heartbeat(jobs=[JobProgress(job_id=high.id, progress=0.1)]))
        assert service.lease(db, w, FREE_GPU) is None                        # running: no second job
        assert db.get(ComputeJob, low.id).status == "queued"

    def test_busy_worker_gets_nothing(self, db):
        bench(db)
        w, _ = worker(db)
        assert service.lease(db, w, Free(vram_gb=6, ram_gb=8, busy_by_others=True)) is None

    def test_distributed_needs_multi_gpu(self, db):
        ds = service.store_artifact(db, _root(db), io.BytesIO(b"zip"), "d.zip", "dataset")
        job = service.submit(db, {"kind": "yolo_train", "dataset": ds.id, "distributed": True}, Requirements())
        assert json.loads(job.requirements_json)["gpus"] == 2
        w, _ = worker(db)
        assert service.lease(db, w, FREE_GPU) is None
        multi, _ = worker(db, GPU.model_copy(update={"gpus": 4}), "dgx")
        assert service.lease(db, multi, FREE_GPU).id == job.id


_ROOT: dict = {}


def _root(db):
    import tempfile
    from pathlib import Path

    return _ROOT.setdefault("root", Path(tempfile.mkdtemp()))


class TestLifecycle:
    def test_lost_worker_requeues_with_checkpoint_then_fails(self, db):
        job = bench(db)
        job.max_attempts = 2
        db.add(job)
        db.commit()
        w, _ = worker(db)
        service.lease(db, w, FREE_GPU)
        ck = service.store_artifact(db, _root(db), io.BytesIO(b"weights"), "last.pt", "checkpoint", job_id=job.id)
        service.attach_upload(db, w, job.id, ck)
        assert service.expire(db, now=time.time() + 1000) == [job.id]
        j = db.get(ComputeJob, job.id)
        assert j.status == "queued" and j.checkpoint_artifact == ck.id and "checkpoint" in j.message
        w2, _ = worker(db, name="other")
        lease2 = service.lease(db, w2, FREE_GPU)
        assert lease2.id == job.id and lease2.attempts == 2
        service.expire(db, now=time.time() + 1000)
        assert db.get(ComputeJob, job.id).status == "failed"

    def test_heartbeat_extends_lease_and_reports_cancel(self, db):
        job = bench(db)
        w, _ = worker(db)
        service.lease(db, w, FREE_GPU)
        assert service.heartbeat(db, w, Heartbeat(jobs=[JobProgress(job_id=job.id, progress=0.5, epoch=3, metrics={"map50": 0.2})])) == ([], [])
        j = db.get(ComputeJob, job.id)
        assert j.status == "running" and j.progress == 0.5 and json.loads(j.metrics_json)["map50"] == 0.2
        service.cancel(db, job.id)
        assert service.heartbeat(db, w, Heartbeat(jobs=[JobProgress(job_id=job.id)])) == ([job.id], [])
        service.cancelled_by_worker(db, w, job.id)
        assert db.get(ComputeJob, job.id).status == "cancelled"

    def test_heartbeat_for_foreign_job_is_cancelled(self, db):
        job = bench(db)
        a, _ = worker(db, name="a")
        b, _ = worker(db, name="b")
        service.lease(db, a, FREE_GPU)
        assert service.heartbeat(db, b, Heartbeat(jobs=[JobProgress(job_id=job.id)])) == ([], [job.id])

    def test_yield_requeues_without_counting_an_attempt(self, db):
        job = bench(db)
        w, _ = worker(db)
        service.lease(db, w, FREE_GPU)
        j = service.finish(db, w, job.id, Finish(status="yielded"))
        assert j.status == "queued" and j.attempts == 0 and j.worker_id is None

    def test_retryable_failure_then_final(self, db):
        job = bench(db)
        job.max_attempts = 1
        db.add(job)
        db.commit()
        w, _ = worker(db)
        service.lease(db, w, FREE_GPU)
        assert service.finish(db, w, job.id, Finish(status="failed", retryable=True)).status == "failed"

    def test_queued_cancel_is_immediate(self, db):
        assert service.cancel(db, bench(db).id).status == "cancelled"


class TestArtifacts:
    def test_checksum_and_size(self, db, monkeypatch):
        with pytest.raises(service.ComputeError, match="sha256"):
            service.store_artifact(db, _root(db), io.BytesIO(b"abc"), "x.zip", "dataset", expected_sha256="0" * 64)
        monkeypatch.setattr(service, "MAX_ARTIFACT_BYTES", 10)
        with pytest.raises(service.ComputeError, match="larger"):
            service.store_artifact(db, _root(db), io.BytesIO(b"x" * 20), "x.zip", "dataset")

    def test_worker_reads_only_inputs_of_its_job(self, db):
        ds = service.store_artifact(db, _root(db), io.BytesIO(b"data"), "d.zip", "dataset")
        other = service.store_artifact(db, _root(db), io.BytesIO(b"secret"), "o.zip", "dataset")
        job = service.submit(db, {"kind": "yolo_train", "dataset": ds.id}, Requirements())
        w, _ = worker(db)
        with pytest.raises(service.ComputeError):
            service.worker_may_read(db, w, ds.id)                           # not leased yet
        service.lease(db, w, FREE_GPU)
        assert service.worker_may_read(db, w, ds.id).id == ds.id
        with pytest.raises(service.ComputeError, match="not part"):
            service.worker_may_read(db, w, other.id)
        assert job.id

    def test_new_checkpoint_replaces_old(self, db):
        job = bench(db)
        w, _ = worker(db)
        service.lease(db, w, FREE_GPU)
        a = service.store_artifact(db, _root(db), io.BytesIO(b"1"), "last.pt", "checkpoint", job_id=job.id)
        service.attach_upload(db, w, job.id, a)
        b = service.store_artifact(db, _root(db), io.BytesIO(b"2"), "last.pt", "checkpoint", job_id=job.id)
        service.attach_upload(db, w, job.id, b)
        from pathlib import Path

        assert db.get(ComputeJob, job.id).checkpoint_artifact == b.id and not Path(a.path).exists()


class TestApi:
    """HTTP layer: agent routes take only worker tokens, admin routes stay behind the API token."""

    def test_agent_flow_behind_api_token(self, db, monkeypatch, tmp_path):
        from backend.compute import api
        from backend.database import get_session
        from backend.middleware.auth import APITokenMiddleware

        monkeypatch.setattr(api, "artifact_root", lambda: tmp_path)
        app = FastAPI()
        app.add_middleware(APITokenMiddleware, api_token="admin-secret")
        app.include_router(api.router, prefix="/api/v1")
        app.include_router(api.agent_router, prefix="/api/v1")
        app.dependency_overrides[get_session] = lambda: db
        c = TestClient(app)
        admin = {"Authorization": "Bearer admin-secret"}
        assert c.post("/api/v1/compute/enrollments", json={}).status_code == 401
        token = c.post("/api/v1/compute/enrollments", json={"note": "pc"}, headers=admin).json()["token"]
        r = c.post("/api/v1/compute/agent/enroll", json={"enrollment_token": token, "name": "pc", "capabilities": GPU.model_dump(mode="json")})
        wt = r.json()["worker_token"]
        agent = {"Authorization": f"Bearer {wt}"}
        assert c.post("/api/v1/compute/agent/lease", json={"free": FREE_GPU.model_dump()}, headers=admin).status_code == 401
        assert c.post("/api/v1/compute/agent/lease", json={"free": FREE_GPU.model_dump()}, headers=agent).status_code == 204
        bad = c.post("/api/v1/compute/jobs", json={"spec": {"kind": "detection_benchmark", "runs": 1}}, headers=admin)
        assert bad.status_code == 422                                          # runs >= 3
        evil = c.post("/api/v1/compute/jobs", json={"spec": {"kind": "shell", "cmd": "rm -rf /"}}, headers=admin)
        assert evil.status_code == 422                                         # only known job kinds exist
        jid = c.post("/api/v1/compute/jobs", json={"spec": {"kind": "detection_benchmark"}, "name": "b"}, headers=admin).json()["id"]
        lease = c.post("/api/v1/compute/agent/lease", json={"free": FREE_GPU.model_dump()}, headers=agent).json()
        assert lease["job_id"] == jid and lease["spec"]["runs"] == 20
        hb = c.post("/api/v1/compute/agent/heartbeat", json={"jobs": [{"job_id": jid, "progress": 0.5}]}, headers=agent).json()
        assert hb["cancel"] == [] and hb["lost"] == []
        import hashlib

        data = b"best weights"
        up = c.post(f"/api/v1/compute/agent/jobs/{jid}/artifacts", headers=agent, files={"file": ("best.pt", data)},
                    data={"kind": "model", "sha256": hashlib.sha256(data).hexdigest()})
        assert up.status_code == 200
        done = c.post(f"/api/v1/compute/agent/jobs/{jid}/finish", json={"status": "completed", "metrics": {"fps": 50}}, headers=agent).json()
        assert done["status"] == "completed" and done["results"] == [up.json()["id"]]
        dl = c.get(f"/api/v1/compute/artifacts/{up.json()['id']}", headers=admin)
        assert dl.content == data and dl.headers["X-Content-SHA256"] == hashlib.sha256(data).hexdigest()
        workers = c.get("/api/v1/compute/workers", headers=admin).json()
        assert workers[0]["online"] and workers[0]["capabilities"]["device"] == "RTX 4060"


def test_installers_are_public_and_carry_the_server(monkeypatch):
    from backend.compute import api
    from backend.middleware.auth import APITokenMiddleware

    app = FastAPI()
    app.add_middleware(APITokenMiddleware, api_token="admin-secret")
    app.include_router(api.install_router, prefix="/api/v1")
    c = TestClient(app)
    r = c.get("/api/v1/compute/install/install-worker.sh", headers={"x-forwarded-proto": "https", "x-forwarded-host": "ctip.example.org"})
    assert r.status_code == 200 and 'SERVER="${CTIP_SERVER:-https://ctip.example.org}"' in r.text
    assert "@SERVER@" not in r.text and "ctipe_" in r.text                      # token placeholder only, no secret
    ps = c.get("/api/v1/compute/install/install-worker.ps1", headers={"x-forwarded-proto": "https", "x-forwarded-host": "ctip.example.org"})
    assert ps.status_code == 200 and '"https://ctip.example.org"' in ps.text
    assert c.get("/api/v1/compute/install/../../etc/passwd").status_code in (401, 404)   # never served
    assert c.get("/api/v1/compute/install/other.sh").status_code == 404
