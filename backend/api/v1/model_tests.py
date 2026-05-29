"""
backend.api.v1.model_tests — Save/load shareable model test pipeline graphs.

POST /model-tests          — save a graph, returns UUID
GET  /model-tests          — list saved tests
GET  /model-tests/{uuid}   — load a specific test graph
DELETE /model-tests/{uuid} — delete a test
"""

from __future__ import annotations

import json
import time
import uuid as _uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, select

from backend.database import get_session

router = APIRouter(prefix="/model-tests", tags=["model-tests"])


# ---------------------------------------------------------------------------
# DB model
# ---------------------------------------------------------------------------

class ModelTest(SQLModel, table=True):
    __tablename__ = "model_tests"

    id: Optional[int] = Field(default=None, primary_key=True)
    test_uuid: str = Field(index=True, unique=True)
    name: str = Field(default="Untitled test")
    description: str = Field(default="")
    graph_json: str = Field(default="{}")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSaveRequest(BaseModel):
    name: str = "Untitled test"
    description: str = ""
    graph: dict


class TestMeta(BaseModel):
    uuid: str
    name: str
    description: str
    created_at: float
    updated_at: float

    model_config = {"from_attributes": True}


class TestDetail(TestMeta):
    graph: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=TestMeta, status_code=201)
def save_test(
    req: TestSaveRequest,
    session: Session = Depends(get_session),
) -> TestMeta:
    """Save a model test graph. Returns UUID for sharing."""
    test_uuid = str(_uuid.uuid4())
    t = ModelTest(
        test_uuid=test_uuid,
        name=req.name,
        description=req.description,
        graph_json=json.dumps(req.graph),
    )
    session.add(t)
    session.commit()
    session.refresh(t)
    return TestMeta(
        uuid=t.test_uuid,
        name=t.name,
        description=t.description,
        created_at=t.created_at,
        updated_at=t.updated_at,
    )


@router.put("/{test_uuid}", response_model=TestMeta)
def update_test(
    test_uuid: Annotated[str, Path()],
    req: TestSaveRequest,
    session: Session = Depends(get_session),
) -> TestMeta:
    """Overwrite an existing test graph in-place."""
    t = session.exec(select(ModelTest).where(ModelTest.test_uuid == test_uuid)).first()
    if not t:
        raise HTTPException(404, f"Test {test_uuid} not found")
    t.name = req.name
    t.description = req.description
    t.graph_json = json.dumps(req.graph)
    t.updated_at = time.time()
    session.add(t)
    session.commit()
    return TestMeta(
        uuid=t.test_uuid,
        name=t.name,
        description=t.description,
        created_at=t.created_at,
        updated_at=t.updated_at,
    )


@router.get("", response_model=list[TestMeta])
def list_tests(session: Session = Depends(get_session)) -> list[TestMeta]:
    """List all saved tests, newest first."""
    tests = session.exec(select(ModelTest).order_by(ModelTest.created_at.desc())).all()  # type: ignore[arg-type]
    return [
        TestMeta(uuid=t.test_uuid, name=t.name, description=t.description,
                 created_at=t.created_at, updated_at=t.updated_at)
        for t in tests
    ]


@router.get("/{test_uuid}", response_model=TestDetail)
def load_test(
    test_uuid: Annotated[str, Path()],
    session: Session = Depends(get_session),
) -> TestDetail:
    """Load a specific test graph by UUID."""
    t = session.exec(select(ModelTest).where(ModelTest.test_uuid == test_uuid)).first()
    if not t:
        raise HTTPException(404, f"Test {test_uuid} not found")
    return TestDetail(
        uuid=t.test_uuid,
        name=t.name,
        description=t.description,
        graph=json.loads(t.graph_json),
        created_at=t.created_at,
        updated_at=t.updated_at,
    )


@router.delete("/{test_uuid}")
def delete_test(
    test_uuid: Annotated[str, Path()],
    session: Session = Depends(get_session),
) -> dict:
    t = session.exec(select(ModelTest).where(ModelTest.test_uuid == test_uuid)).first()
    if not t:
        raise HTTPException(404, f"Test {test_uuid} not found")
    session.delete(t)
    session.commit()
    return {"deleted": test_uuid}
