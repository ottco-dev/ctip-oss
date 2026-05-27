"""
annotation/cvat/client.py — CVAT REST API client wrapper.

Wraps the CVAT REST API for:
  - Project/task management
  - Annotation upload/download (COCO format)
  - Job status polling

CVAT runs at http://localhost:8080 (default docker-compose.annotation.yml)

SDK reference: https://docs.cvat.ai/docs/api_sdk/sdk/
Fallback: direct HTTP via requests when cvat-sdk is not installed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CVATConfig:
    """CVAT connection configuration."""

    host: str = "http://localhost:8080"
    username: str = "admin"
    password: str = "admin"
    timeout_s: float = 30.0
    verify_ssl: bool = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CVATProject:
    id: int
    name: str
    labels: list[dict] = field(default_factory=list)
    task_count: int = 0


@dataclass
class CVATTask:
    id: int
    project_id: int
    name: str
    status: str
    size: int  # number of frames/images
    mode: str = "annotation"  # annotation | interpolation


@dataclass
class CVATJob:
    id: int
    task_id: int
    status: str  # new | in progress | completed | rejected
    assignee: Optional[str] = None
    start_frame: int = 0
    stop_frame: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class CVATClient:
    """
    CVAT REST API client.

    Uses requests for HTTP calls. Automatically handles session auth tokens.

    Usage:
        client = CVATClient(CVATConfig(host="http://localhost:8080"))
        client.connect()
        projects = client.list_projects()
        task_id = client.create_task(project_id=1, name="batch-001", images=[...])
        client.upload_annotations(task_id, annotations)
    """

    def __init__(self, config: CVATConfig | None = None) -> None:
        self.config = config or CVATConfig()
        self._session: Any = None
        self._auth_token: Optional[str] = None

    def connect(self) -> None:
        """Authenticate and create HTTP session."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")

        self._session = requests.Session()
        self._session.verify = self.config.verify_ssl

        resp = self._session.post(
            f"{self.config.host}/api/auth/login",
            json={
                "username": self.config.username,
                "password": self.config.password,
            },
            timeout=self.config.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("key") or data.get("token")
        if token:
            self._auth_token = token
            self._session.headers["Authorization"] = f"Token {token}"

    def disconnect(self) -> None:
        if self._session:
            try:
                self._session.post(f"{self.config.host}/api/auth/logout")
            except Exception:
                pass
            self._session.close()
            self._session = None

    def _get(self, path: str, **kwargs) -> dict:
        resp = self._session.get(
            f"{self.config.host}{path}",
            timeout=self.config.timeout_s,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict | None = None, **kwargs) -> dict:
        resp = self._session.post(
            f"{self.config.host}{path}",
            json=data,
            timeout=self.config.timeout_s,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _patch(self, path: str, data: dict) -> dict:
        resp = self._session.patch(
            f"{self.config.host}{path}",
            json=data,
            timeout=self.config.timeout_s,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def list_projects(self) -> list[CVATProject]:
        """List all CVAT projects."""
        data = self._get("/api/projects")
        return [
            CVATProject(
                id=p["id"],
                name=p["name"],
                labels=p.get("labels", []),
                task_count=p.get("tasks", {}).get("count", 0),
            )
            for p in data.get("results", [])
        ]

    def get_project(self, project_id: int) -> CVATProject:
        data = self._get(f"/api/projects/{project_id}")
        return CVATProject(
            id=data["id"],
            name=data["name"],
            labels=data.get("labels", []),
        )

    def create_project(
        self,
        name: str,
        labels: Optional[list[dict]] = None,
    ) -> CVATProject:
        """
        Create a CVAT project with trichome labels.

        Args:
            name: Project name.
            labels: List of label dicts. Defaults to trichome classes.
        """
        default_labels = [
            {"name": "capitate-stalked", "color": "#60a5fa"},
            {"name": "capitate-sessile", "color": "#a78bfa"},
            {"name": "bulbous", "color": "#34d399"},
            {"name": "non-glandular", "color": "#f87171"},
        ]
        data = self._post(
            "/api/projects",
            data={"name": name, "labels": labels or default_labels},
        )
        return CVATProject(id=data["id"], name=data["name"])

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def list_tasks(self, project_id: Optional[int] = None) -> list[CVATTask]:
        path = f"/api/tasks?project_id={project_id}" if project_id else "/api/tasks"
        data = self._get(path)
        return [
            CVATTask(
                id=t["id"],
                project_id=t.get("project_id", -1),
                name=t["name"],
                status=t.get("status", "unknown"),
                size=t.get("size", 0),
            )
            for t in data.get("results", [])
        ]

    def create_task(
        self,
        project_id: int,
        name: str,
        image_paths: list[str],
    ) -> CVATTask:
        """Create a CVAT task and upload images."""
        task_data = self._post(
            "/api/tasks",
            data={"name": name, "project_id": project_id},
        )
        task_id = task_data["id"]

        # Upload images
        files = [("client_files[" + str(i) + "]", open(p, "rb")) for i, p in enumerate(image_paths)]
        resp = self._session.post(
            f"{self.config.host}/api/tasks/{task_id}/data",
            files=files,
            data={"image_quality": 95},
            timeout=60.0,
        )
        for _, f in files:
            f.close()
        resp.raise_for_status()

        return CVATTask(
            id=task_id,
            project_id=project_id,
            name=name,
            status="annotation",
            size=len(image_paths),
        )

    def get_task_status(self, task_id: int) -> str:
        data = self._get(f"/api/tasks/{task_id}/status")
        return data.get("state", "unknown")

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def upload_annotations(
        self,
        task_id: int,
        annotations: dict,
        format: str = "COCO 1.0",
    ) -> None:
        """Upload annotations in COCO format to a task."""
        import io

        annotation_json = json.dumps(annotations).encode()
        self._session.post(
            f"{self.config.host}/api/tasks/{task_id}/annotations?format={format}",
            files={"annotation_file": ("annotations.json", io.BytesIO(annotation_json), "application/json")},
            timeout=60.0,
        ).raise_for_status()

    def download_annotations(
        self,
        task_id: int,
        format: str = "COCO 1.0",
    ) -> dict:
        """Download annotations from a task in COCO format."""
        # Initiate export
        resp = self._session.get(
            f"{self.config.host}/api/tasks/{task_id}/annotations?format={format}",
            timeout=60.0,
        )
        if resp.status_code == 202:
            # Async export — poll
            for _ in range(30):
                time.sleep(2)
                resp = self._session.get(
                    f"{self.config.host}/api/tasks/{task_id}/annotations?format={format}",
                    timeout=60.0,
                )
                if resp.status_code == 200:
                    break
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
