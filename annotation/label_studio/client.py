"""
annotation/label_studio/client.py — Label Studio REST API client.

Label Studio provides a flexible annotation UI with:
  - Bounding box and polygon annotation
  - Image classification
  - Built-in data quality metrics
  - YOLO/COCO export formats

Default host: http://localhost:8090 (docker-compose.annotation.yml)
"""

from __future__ import annotations

import json
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
class LabelStudioConfig:
    host: str = "http://localhost:8090"
    api_key: str = ""
    timeout_s: float = 30.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LSProject:
    id: int
    title: str
    task_count: int = 0
    annotation_count: int = 0
    label_config: str = ""


@dataclass
class LSTask:
    id: int
    project_id: int
    data: dict = field(default_factory=dict)
    is_labeled: bool = False
    annotation_count: int = 0


@dataclass
class LSAnnotation:
    id: int
    task_id: int
    result: list[dict] = field(default_factory=list)
    score: float = 0.0
    was_cancelled: bool = False
    ground_truth: bool = False


# ---------------------------------------------------------------------------
# Trichome label config (Label Studio XML format)
# ---------------------------------------------------------------------------

TRICHOME_LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="capitate-stalked" background="#60a5fa"/>
    <Label value="capitate-sessile" background="#a78bfa"/>
    <Label value="bulbous" background="#34d399"/>
    <Label value="non-glandular" background="#f87171"/>
  </RectangleLabels>
  <Choices name="maturity" toName="image" choice="single">
    <Choice value="clear"/>
    <Choice value="cloudy"/>
    <Choice value="amber"/>
    <Choice value="mixed"/>
  </Choices>
  <TextArea name="notes" toName="image" placeholder="Optional notes..."
            editable="true" maxSubmissions="1"/>
</View>
"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LabelStudioClient:
    """
    Label Studio REST API client.

    Usage:
        client = LabelStudioClient(LabelStudioConfig(api_key="my-api-key"))
        project_id = client.create_project("Trichome Batch 1")
        client.import_tasks(project_id, image_paths)
        annotations = client.export_annotations(project_id, format="YOLO")
    """

    def __init__(self, config: LabelStudioConfig | None = None) -> None:
        self.config = config or LabelStudioConfig()

    def _headers(self) -> dict:
        return {"Authorization": f"Token {self.config.api_key}", "Content-Type": "application/json"}

    def _get(self, path: str, **kwargs) -> dict:
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed")
        resp = requests.get(
            f"{self.config.host}{path}",
            headers=self._headers(),
            timeout=self.config.timeout_s,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict | list | None = None, **kwargs) -> Any:
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed")
        resp = requests.post(
            f"{self.config.host}{path}",
            headers=self._headers(),
            json=data,
            timeout=self.config.timeout_s,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def list_projects(self) -> list[LSProject]:
        data = self._get("/api/projects")
        return [
            LSProject(
                id=p["id"],
                title=p.get("title", ""),
                task_count=p.get("task_number", 0),
                annotation_count=p.get("num_tasks_with_annotations", 0),
                label_config=p.get("label_config", ""),
            )
            for p in data.get("results", [])
        ]

    def create_project(
        self,
        title: str,
        label_config: Optional[str] = None,
    ) -> int:
        """
        Create a Label Studio project for trichome annotation.

        Returns:
            project_id (int)
        """
        data = self._post(
            "/api/projects",
            data={
                "title": title,
                "label_config": label_config or TRICHOME_LABEL_CONFIG,
                "description": "Cannabis trichome detection and maturity analysis",
            },
        )
        return int(data["id"])

    def get_project(self, project_id: int) -> LSProject:
        data = self._get(f"/api/projects/{project_id}")
        return LSProject(
            id=data["id"],
            title=data.get("title", ""),
            task_count=data.get("task_number", 0),
            annotation_count=data.get("num_tasks_with_annotations", 0),
        )

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def import_tasks(
        self,
        project_id: int,
        image_paths: list[str],
        predictions: Optional[list[dict]] = None,
    ) -> list[int]:
        """
        Import images as tasks into a project.

        Args:
            project_id: Label Studio project ID.
            image_paths: Absolute paths to image files.
            predictions: Optional pre-annotations per image (YOLO/detection format).

        Returns:
            List of created task IDs.
        """
        tasks = []
        for i, path in enumerate(image_paths):
            task: dict = {"data": {"image": path}}
            if predictions and i < len(predictions):
                task["predictions"] = [{"result": predictions[i]}]
            tasks.append(task)

        result = self._post(
            f"/api/projects/{project_id}/import",
            data=tasks,
        )
        return [t["id"] for t in result.get("task_ids", [])]

    def list_tasks(self, project_id: int, page: int = 1, page_size: int = 100) -> list[LSTask]:
        data = self._get(f"/api/tasks?project={project_id}&page={page}&page_size={page_size}")
        return [
            LSTask(
                id=t["id"],
                project_id=project_id,
                data=t.get("data", {}),
                is_labeled=t.get("is_labeled", False),
                annotation_count=t.get("total_annotations", 0),
            )
            for t in data.get("tasks", [])
        ]

    def get_task(self, task_id: int) -> LSTask:
        data = self._get(f"/api/tasks/{task_id}")
        return LSTask(
            id=data["id"],
            project_id=data.get("project", -1),
            data=data.get("data", {}),
            is_labeled=data.get("is_labeled", False),
        )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def get_annotations(self, task_id: int) -> list[LSAnnotation]:
        data = self._get(f"/api/tasks/{task_id}/annotations")
        return [
            LSAnnotation(
                id=a["id"],
                task_id=task_id,
                result=a.get("result", []),
                score=a.get("score", 0.0) or 0.0,
                was_cancelled=a.get("was_cancelled", False),
                ground_truth=a.get("ground_truth", False),
            )
            for a in data
        ]

    def create_annotation(
        self,
        task_id: int,
        result: list[dict],
        ground_truth: bool = False,
    ) -> LSAnnotation:
        """
        Submit an annotation for a task.

        Args:
            task_id: Task ID.
            result: Label Studio result format (list of labeled regions).
            ground_truth: Mark as ground truth.

        Returns:
            Created annotation.
        """
        data = self._post(
            f"/api/tasks/{task_id}/annotations",
            data={"result": result, "ground_truth": ground_truth},
        )
        return LSAnnotation(
            id=data["id"],
            task_id=task_id,
            result=data.get("result", []),
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_annotations(
        self,
        project_id: int,
        export_format: str = "COCO",
    ) -> dict | bytes:
        """
        Export all annotations from a project.

        Args:
            project_id: Project ID.
            export_format: "COCO", "YOLO", "JSON", "CSV"

        Returns:
            Annotation data (dict for JSON/COCO, bytes for others).
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed")

        resp = requests.get(
            f"{self.config.host}/api/projects/{project_id}/export?exportType={export_format}",
            headers=self._headers(),
            timeout=120.0,
        )
        resp.raise_for_status()

        if export_format in ("COCO", "JSON"):
            return resp.json()
        return resp.content

    # ------------------------------------------------------------------
    # Prediction import (pre-annotations from detection model)
    # ------------------------------------------------------------------

    def import_predictions(
        self,
        task_id: int,
        detections: list[dict],
        model_version: str = "yolo11s",
    ) -> None:
        """
        Import model predictions as pre-annotations for a task.

        Args:
            task_id: Label Studio task ID.
            detections: List of detection dicts with x1/y1/x2/y2/confidence/class_name.
            model_version: Model identifier string.
        """
        # Convert detections to Label Studio format
        result = []
        for det in detections:
            # Label Studio uses percentages of image dimensions
            result.append(
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": det.get("x1", 0),
                        "y": det.get("y1", 0),
                        "width": det.get("x2", 0) - det.get("x1", 0),
                        "height": det.get("y2", 0) - det.get("y1", 0),
                        "rectanglelabels": [det.get("class_name", "capitate-stalked")],
                    },
                    "score": det.get("confidence", 0.0),
                }
            )

        self._post(
            f"/api/tasks/{task_id}/annotations",
            data={"result": result, "model_version": model_version},
        )
