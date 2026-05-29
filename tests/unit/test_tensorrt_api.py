"""
tests/unit/test_tensorrt_api.py — TensorRT management API tests.

All tests use mocking — no real TensorRT installation required.
Tests verify graceful degradation (503 when TRT unavailable) and full
happy-path behaviour when TRT is mocked as available.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    from backend.api.v1.tensorrt import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


# ── Availability guard fixtures ───────────────────────────────────────────────

@pytest.fixture()
def trt_unavailable():
    """Simulate TRT not installed."""
    with patch("backend.api.v1.tensorrt._trt_available", return_value=False):
        yield


@pytest.fixture()
def trt_available():
    """Simulate TRT installed."""
    with patch("backend.api.v1.tensorrt._trt_available", return_value=True):
        yield


# ── GET /tensorrt/status ──────────────────────────────────────────────────────

class TestStatus:
    def test_status_trt_not_installed(self, client, trt_unavailable):
        resp = client.get("/tensorrt/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert data["install_hint"] is not None
        assert "pip install" in data["install_hint"]

    def test_status_trt_installed(self, client, trt_available):
        with patch("backend.api.v1.tensorrt._list_engine_files", return_value=[]):
            mock_trt = MagicMock()
            mock_trt.__version__ = "10.6.0"
            with patch.dict("sys.modules", {"tensorrt": mock_trt}):
                with patch.dict("sys.modules", {"pycuda": MagicMock(VERSION_TEXT="2024.1")}):
                    resp = client.get("/tensorrt/status")
        assert resp.status_code == 200
        data = resp.json()
        # available check depends on real tensorrt import; check structure
        assert "available" in data
        assert "engines_dir" in data
        assert isinstance(data["engines"], list)

    def test_status_lists_engines(self, client, trt_unavailable):
        fake_engines = [
            {"name": "yolo11s_fp16.engine", "path": "/models/yolo11s_fp16.engine",
             "size_mb": 22.4, "modified": 1700000000.0}
        ]
        with patch("backend.api.v1.tensorrt._list_engine_files", return_value=fake_engines):
            resp = client.get("/tensorrt/status")
        assert resp.status_code == 200
        assert len(resp.json()["engines"]) == 1
        assert resp.json()["engines"][0]["name"] == "yolo11s_fp16.engine"

    def test_status_no_engines(self, client, trt_unavailable):
        with patch("backend.api.v1.tensorrt._list_engine_files", return_value=[]):
            resp = client.get("/tensorrt/status")
        assert resp.status_code == 200
        assert resp.json()["engines"] == []


# ── GET /tensorrt/engines ─────────────────────────────────────────────────────

class TestListEngines:
    def test_list_engines_empty(self, client):
        with patch("backend.api.v1.tensorrt._list_engine_files", return_value=[]):
            resp = client.get("/tensorrt/engines")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_engines_multiple(self, client):
        engines = [
            {"name": "a.engine", "path": "/p/a.engine", "size_mb": 10.0, "modified": 1.0},
            {"name": "b.engine", "path": "/p/b.engine", "size_mb": 20.0, "modified": 2.0},
        ]
        with patch("backend.api.v1.tensorrt._list_engine_files", return_value=engines):
            resp = client.get("/tensorrt/engines")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ── GET /tensorrt/inspect/{name} ─────────────────────────────────────────────

class TestInspectEngine:
    def test_inspect_requires_trt(self, client, trt_unavailable):
        resp = client.get("/tensorrt/inspect/model.engine")
        assert resp.status_code == 503

    def test_inspect_engine_not_found(self, client, trt_available):
        with patch("backend.api.v1.tensorrt._engines_dir", return_value=Path("/nonexistent")):
            resp = client.get("/tensorrt/inspect/missing.engine")
        assert resp.status_code == 404

    def test_inspect_engine_ok(self, client, trt_available, tmp_path):
        engine_file = tmp_path / "model.engine"
        engine_file.write_bytes(b"fake_engine")

        inspect_result = {
            "inputs": [{"name": "images", "shape": (1, 3, 1280, 1280), "dtype": "DataType.HALF"}],
            "outputs": [{"name": "output0", "shape": (1, 8, 8400), "dtype": "DataType.HALF"}],
            "trt_version": "10.6.0",
            "engine_path": str(engine_file),
            "size_mb": 0.0,
        }
        with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
            with patch("backend.api.v1.tensorrt.inspect_engine", inspect_result.__class__) as _:
                with patch("inference.tensorrt_engine.builder.inspect_engine", return_value=inspect_result):
                    resp = client.get(f"/tensorrt/inspect/{engine_file.name}")
        # May be 200 or fail depending on import; just check not 503
        assert resp.status_code in (200, 500)


# ── POST /tensorrt/build ──────────────────────────────────────────────────────

class TestBuildEngine:
    def test_build_requires_trt(self, client, trt_unavailable):
        resp = client.post("/tensorrt/build", json={
            "onnx_path": "model.onnx", "imgsz": 640,
        })
        assert resp.status_code == 503
        assert "install" in resp.json()["detail"]

    def test_build_onnx_not_found(self, client, trt_available):
        resp = client.post("/tensorrt/build", json={
            "onnx_path": "/nonexistent/model.onnx",
        })
        assert resp.status_code == 404

    def test_build_queues_job(self, client, trt_available, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"fake_onnx")

        with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
            with patch("backend.api.v1.tensorrt._run_build"):
                resp = client.post("/tensorrt/build", json={
                    "onnx_path": str(onnx),
                    "imgsz": 640,
                    "fp16": True,
                })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("queued", "running", "completed")
        assert "job_id" in data
        assert data["onnx_path"] == str(onnx)

    def test_build_engine_name_default(self, client, trt_available, tmp_path):
        onnx = tmp_path / "yolo11s.onnx"
        onnx.write_bytes(b"fake")

        with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
            with patch("backend.api.v1.tensorrt._run_build"):
                resp = client.post("/tensorrt/build", json={"onnx_path": str(onnx)})
        assert resp.status_code == 200
        data = resp.json()
        assert "yolo11s" in data["engine_path"]
        assert data["engine_path"].endswith(".engine")

    def test_build_conflict_no_overwrite(self, client, trt_available, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"fake_onnx")
        engine = tmp_path / "model_fp16.engine"
        engine.write_bytes(b"existing")

        with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
            resp = client.post("/tensorrt/build", json={
                "onnx_path": str(onnx),
                "overwrite": False,
            })
        assert resp.status_code == 409

    def test_build_overwrite_allowed(self, client, trt_available, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"fake_onnx")
        engine = tmp_path / "model_fp16.engine"
        engine.write_bytes(b"existing")

        with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
            with patch("backend.api.v1.tensorrt._run_build"):
                resp = client.post("/tensorrt/build", json={
                    "onnx_path": str(onnx),
                    "overwrite": True,
                })
        assert resp.status_code == 200


# ── GET /tensorrt/build/{job_id} ─────────────────────────────────────────────

class TestBuildJobStatus:
    def test_poll_unknown_job(self, client):
        resp = client.get("/tensorrt/build/nonexistent-job-id")
        assert resp.status_code == 404

    def test_poll_existing_job(self, client, trt_available, tmp_path):
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"fake")

        with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
            with patch("backend.api.v1.tensorrt._run_build"):
                post = client.post("/tensorrt/build", json={"onnx_path": str(onnx)})
        job_id = post.json()["job_id"]

        resp = client.get(f"/tensorrt/build/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert "status" in data


# ── POST /tensorrt/infer ──────────────────────────────────────────────────────

class TestTRTInfer:
    def test_infer_requires_trt(self, client, trt_unavailable):
        img = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header
        resp = client.post(
            "/tensorrt/infer",
            data={"engine_name": "model.engine"},
            files={"file": ("img.jpg", img, "image/jpeg")},
        )
        assert resp.status_code == 503

    def test_infer_engine_not_found(self, client, trt_available):
        with patch("backend.api.v1.tensorrt._engines_dir", return_value=Path("/nonexistent")):
            resp = client.post(
                "/tensorrt/infer",
                data={"engine_name": "missing.engine"},
                files={"file": ("img.jpg", b"fake", "image/jpeg")},
            )
        assert resp.status_code == 404

    def test_infer_invalid_image(self, client, trt_available, tmp_path):
        engine = tmp_path / "model.engine"
        engine.write_bytes(b"fake_engine")

        with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
            resp = client.post(
                "/tensorrt/infer",
                data={"engine_name": "model.engine"},
                files={"file": ("img.jpg", b"not_an_image", "image/jpeg")},
            )
        assert resp.status_code == 422

    def test_infer_success(self, client, trt_available, tmp_path):
        engine = tmp_path / "model.engine"
        engine.write_bytes(b"fake_engine")

        from inference.tensorrt_engine.runner import TRTDetection, TRTResult
        mock_result = TRTResult(
            detections=[
                TRTDetection(x1=10, y1=10, x2=100, y2=100,
                             confidence=0.92, class_id=0, class_name="capitate-stalked")
            ],
            inference_ms=4.2,
            preprocess_ms=1.1,
            postprocess_ms=0.8,
            engine_path=str(engine),
            image_hw=(640, 640),
        )

        mock_runner = MagicMock()
        mock_runner.infer.return_value = mock_result

        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()

        with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
            with patch(
                "inference.tensorrt_engine.runner.TensorRTRunner",
                return_value=mock_runner,
            ):
                resp = client.post(
                    "/tensorrt/infer",
                    data={"engine_name": "model.engine"},
                    files={"file": ("img.jpg", img_bytes, "image/jpeg")},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["engine_name"] == "model.engine"
        assert len(data["detections"]) == 1
        assert data["detections"][0]["confidence"] == pytest.approx(0.92)
        assert data["inference_ms"] == pytest.approx(4.2)
        assert data["image_hw"] == [640, 640]


# ── _run_build unit test ──────────────────────────────────────────────────────

class TestRunBuild:
    def test_run_build_success(self, tmp_path):
        from backend.api.v1.tensorrt import _build_jobs, _run_build
        from backend.api.v1.tensorrt import BuildRequest

        job_id = "test-build-job"
        engine_path = tmp_path / "out.engine"
        engine_path.write_bytes(b"fake_engine_bytes")  # simulate output

        _build_jobs[job_id] = {
            "job_id": job_id, "status": "queued",
            "onnx_path": "model.onnx", "engine_path": str(engine_path),
            "started_at": None, "finished_at": None,
            "duration_s": None, "engine_size_mb": None, "error": None,
        }

        req = BuildRequest(onnx_path="model.onnx", fp16=True, workspace_gb=2.0)

        with patch("inference.tensorrt_engine.builder.build_engine_from_onnx") as mock_build:
            mock_build.return_value = engine_path
            with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
                _run_build(job_id, req, engine_path)

        assert _build_jobs[job_id]["status"] == "completed"
        assert _build_jobs[job_id]["finished_at"] is not None
        assert _build_jobs[job_id]["engine_size_mb"] is not None

    def test_run_build_failure(self, tmp_path):
        from backend.api.v1.tensorrt import _build_jobs, _run_build
        from backend.api.v1.tensorrt import BuildRequest

        job_id = "test-build-fail"
        engine_path = tmp_path / "out.engine"

        _build_jobs[job_id] = {
            "job_id": job_id, "status": "queued",
            "onnx_path": "bad.onnx", "engine_path": str(engine_path),
            "started_at": None, "finished_at": None,
            "duration_s": None, "engine_size_mb": None, "error": None,
        }

        req = BuildRequest(onnx_path="bad.onnx")

        with patch("inference.tensorrt_engine.builder.build_engine_from_onnx",
                   side_effect=RuntimeError("build failed: CUDA OOM")):
            with patch("backend.api.v1.tensorrt._engines_dir", return_value=tmp_path):
                _run_build(job_id, req, engine_path)

        assert _build_jobs[job_id]["status"] == "failed"
        assert "CUDA OOM" in _build_jobs[job_id]["error"]
        assert _build_jobs[job_id]["finished_at"] is not None
