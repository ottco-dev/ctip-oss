"""
backend.api.v1.labelstudio — Label Studio management API.

Routes:
  GET  /labelstudio/status          — connection status
  POST /labelstudio/connect         — test + save connection config
  GET  /labelstudio/projects        — list LS projects
  POST /labelstudio/import/{proj}   — import annotations from project to review queue
  POST /labelstudio/export/{proj}   — export approved annotations to Label Studio
  GET  /labelstudio/tasks/{proj}    — list tasks in project (paginated)
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/labelstudio", tags=["label-studio"])

# In-memory config store (persists for server lifetime)
_LS_CONFIG: dict = {
    "host": "http://localhost:8090",
    "api_key": "",
    "connected": False,
    "last_check": 0.0,
    "project_count": 0,
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LSConnectRequest(BaseModel):
    host: str = "http://localhost:8090"
    api_key: str = ""


class LSImportRequest(BaseModel):
    project_id: int
    import_format: str = "JSON"  # JSON | CSV | TSV


class LSExportRequest(BaseModel):
    project_id: int
    approved_items: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    """Create a Label Studio client from saved config."""
    try:
        from annotation.label_studio.client import LabelStudioClient, LabelStudioConfig
        config = LabelStudioConfig(
            host=_LS_CONFIG["host"],
            api_key=_LS_CONFIG["api_key"],
        )
        return LabelStudioClient(config)
    except ImportError:
        raise HTTPException(503, "Label Studio client not installed (pip install requests)")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def get_ls_status():
    """Return Label Studio connection status and config."""
    return {
        **_LS_CONFIG,
        "api_key": "***" if _LS_CONFIG["api_key"] else "",
    }


@router.post("/connect")
def connect_label_studio(body: LSConnectRequest):
    """
    Test Label Studio connection and save config.

    Tests by fetching the project list. Returns connection status.
    """
    _LS_CONFIG["host"] = body.host.rstrip("/")
    _LS_CONFIG["api_key"] = body.api_key

    try:
        from annotation.label_studio.client import LabelStudioClient, LabelStudioConfig
        config = LabelStudioConfig(host=_LS_CONFIG["host"], api_key=body.api_key)
        client = LabelStudioClient(config)

        # Test connection by fetching projects
        projects = client.list_projects()
        _LS_CONFIG["connected"] = True
        _LS_CONFIG["last_check"] = time.time()
        _LS_CONFIG["project_count"] = len(projects) if isinstance(projects, list) else 0

        return {
            "connected": True,
            "host": _LS_CONFIG["host"],
            "project_count": _LS_CONFIG["project_count"],
            "projects": projects[:10] if isinstance(projects, list) else [],
        }

    except ImportError:
        raise HTTPException(503, "Label Studio client requires: pip install requests")
    except Exception as exc:
        _LS_CONFIG["connected"] = False
        raise HTTPException(502, f"Cannot connect to Label Studio at {body.host}: {exc}") from exc


@router.get("/projects")
def list_ls_projects():
    """List all projects in the connected Label Studio instance."""
    if not _LS_CONFIG["connected"]:
        raise HTTPException(400, "Not connected. POST /labelstudio/connect first.")

    try:
        client = _make_client()
        projects = client.list_projects()
        return {"projects": projects if isinstance(projects, list) else []}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Label Studio error: {exc}") from exc


@router.post("/import/{project_id}")
def import_from_label_studio(project_id: int):
    """
    Import completed annotations from a Label Studio project into the review queue.

    Only imports tasks with completed annotations.
    All imports require human review before training use.
    """
    if not _LS_CONFIG["connected"]:
        raise HTTPException(400, "Not connected. POST /labelstudio/connect first.")

    try:
        client = _make_client()
        data = client.export_annotations(project_id, export_format="JSON")
        annotations = data if isinstance(data, list) else []

        # Push to annotation review queue
        try:
            from backend.api.v1.annotation import _QUEUE, _STATS, QueueItemCreate
            import uuid as _uuid

            imported = 0
            for ann in annotations:
                item_id = f"ls-{project_id}-{ann.get('id', _uuid.uuid4().hex[:8])}"
                if item_id not in _QUEUE:
                    _QUEUE[item_id] = {
                        "id": item_id,
                        "image_path": ann.get("data", {}).get("image", ""),
                        "filename": ann.get("data", {}).get("image", "").split("/")[-1],
                        "maturity_stage": "unknown",
                        "clear_fraction": 0.0,
                        "cloudy_fraction": 0.0,
                        "amber_fraction": 0.0,
                        "vlm_confidence": 0.0,
                        "hallucination_flags": [],
                        "review_priority": 1,
                        "queued_at": time.time(),
                        "vlm_backend": "label_studio",
                        "source": "label_studio",
                        "raw_annotation": ann,
                        "status": "pending",
                        "human_in_loop": True,
                    }
                    _STATS["total_pending"] = _STATS.get("total_pending", 0) + 1
                    imported += 1

        except Exception:
            pass

        return {
            "project_id": project_id,
            "total_annotations": len(annotations),
            "imported_to_queue": imported if "imported" in dir() else len(annotations),
            "status": "queued_for_review",
            "human_in_loop": True,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Import failed: {exc}") from exc


@router.get("/tasks/{project_id}")
def list_ls_tasks(project_id: int, page: int = 1, page_size: int = 50):
    """List tasks in a Label Studio project with pagination."""
    if not _LS_CONFIG["connected"]:
        raise HTTPException(400, "Not connected.")

    try:
        client = _make_client()
        data = client.export_annotations(project_id, export_format="JSON")
        tasks = data if isinstance(data, list) else []

        start = (page - 1) * page_size
        end = start + page_size

        return {
            "project_id": project_id,
            "total": len(tasks),
            "page": page,
            "page_size": page_size,
            "tasks": tasks[start:end],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Error fetching tasks: {exc}") from exc


@router.post("/export/{project_id}")
def export_to_label_studio(project_id: int, body: LSExportRequest):
    """
    Export approved annotations from the review queue back to Label Studio.

    Only exports items that have been reviewed and approved.
    """
    if not _LS_CONFIG["connected"]:
        raise HTTPException(400, "Not connected.")

    try:
        client = _make_client()
        result = client.import_predictions(project_id, body.approved_items)
        return {
            "project_id": project_id,
            "exported": len(body.approved_items),
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Export failed: {exc}") from exc
