"""
backend.api.v1.experiments — Experiments CRUD router.

Manages ML experiment records (name, tags, status, run counts, best metrics).
Currently backed by an in-memory store; production deployment should migrate
this to SQLite/PostgreSQL via SQLModel (Alembic migration pending).

Endpoints:
  GET    /experiments              — list all experiments
  POST   /experiments              — create a new experiment
  GET    /experiments/{id}         — get a specific experiment
  PUT    /experiments/{id}         — update an experiment
  DELETE /experiments/{id}         — delete an experiment
  PUT    /experiments/{id}/archive — archive an experiment (soft-delete)
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/experiments", tags=["Experiments"])

# ---------------------------------------------------------------------------
# In-memory store (MVP — replace with SQLModel persistence in Phase 19)
# ---------------------------------------------------------------------------

_EXP_STORE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ExperimentCreate(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = []


class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    status: Optional[str] = None
    is_archived: Optional[bool] = None
    best_map50: Optional[float] = None
    best_run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

@router.get("")
def list_experiments():
    """List all experiments, newest first."""
    return sorted(_EXP_STORE.values(), key=lambda e: e["created_at"], reverse=True)


@router.post("", status_code=201)
def create_experiment(body: ExperimentCreate):
    """Create a new experiment."""
    exp_id = str(uuid.uuid4())[:8]
    now = time.time()
    exp = {
        "id": exp_id,
        "name": body.name,
        "description": body.description,
        "tags": body.tags,
        "status": "active",
        "is_archived": False,
        "created_at": now,
        "updated_at": now,
        "run_count": 0,
        "best_map50": None,
        "best_run_id": None,
    }
    _EXP_STORE[exp_id] = exp
    return exp


@router.get("/{exp_id}")
def get_experiment(exp_id: str):
    """Get a specific experiment by ID."""
    exp = _EXP_STORE.get(exp_id)
    if exp is None:
        raise HTTPException(404, f"Experiment '{exp_id}' not found")
    return exp


@router.put("/{exp_id}")
def update_experiment(exp_id: str, body: ExperimentUpdate):
    """Update experiment fields. Only provided (non-None) fields are changed."""
    exp = _EXP_STORE.get(exp_id)
    if exp is None:
        raise HTTPException(404, f"Experiment '{exp_id}' not found")

    if body.name is not None:
        exp["name"] = body.name
    if body.description is not None:
        exp["description"] = body.description
    if body.tags is not None:
        exp["tags"] = body.tags
    if body.status is not None:
        exp["status"] = body.status
    if body.is_archived is not None:
        exp["is_archived"] = body.is_archived
    if body.best_map50 is not None:
        exp["best_map50"] = body.best_map50
    if body.best_run_id is not None:
        exp["best_run_id"] = body.best_run_id

    exp["updated_at"] = time.time()
    return exp


@router.delete("/{exp_id}", status_code=204)
def delete_experiment(exp_id: str):
    """Permanently delete an experiment."""
    if exp_id not in _EXP_STORE:
        raise HTTPException(404, f"Experiment '{exp_id}' not found")
    del _EXP_STORE[exp_id]


@router.put("/{exp_id}/archive")
def archive_experiment(exp_id: str):
    """Toggle archive status on an experiment."""
    exp = _EXP_STORE.get(exp_id)
    if exp is None:
        raise HTTPException(404, f"Experiment '{exp_id}' not found")
    exp["is_archived"] = not exp.get("is_archived", False)
    exp["updated_at"] = time.time()
    return exp
