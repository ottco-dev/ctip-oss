"""
tests.unit.test_model_tests_api — Unit tests for /model-tests CRUD endpoints.

Coverage:
  POST   /model-tests           — create, UUID generated, defaults applied
  GET    /model-tests           — list (newest first, empty DB case)
  GET    /model-tests/{uuid}    — load detail including graph payload
  PUT    /model-tests/{uuid}    — update name/description/graph in-place
  DELETE /model-tests/{uuid}    — delete, verify gone, 404 on second delete
  Error cases: GET/PUT/DELETE unknown UUID → 404
  Graph payloads: empty dict, nested structure, unicode keys
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from backend.api.v1.model_tests import ModelTest, router
from backend.database import get_session


# ---------------------------------------------------------------------------
# In-memory SQLite test database + app fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    # StaticPool: all operations share ONE connection → in-memory DB persists
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def _override_session():
        with Session(engine) as session:
            yield session

    app = FastAPI()
    app.include_router(router)  # router already carries prefix="/model-tests"
    app.dependency_overrides[get_session] = _override_session

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_GRAPH = {
    "nodes": [
        {"id": "n1", "type": "input", "data": {"label": "Image"}},
        {"id": "n2", "type": "detect", "data": {"model": "yolov8n"}},
    ],
    "edges": [{"source": "n1", "target": "n2"}],
}


def _create(client, name="Test A", description="desc", graph=None):
    if graph is None:
        graph = _SAMPLE_GRAPH
    return client.post("/model-tests", json={"name": name, "description": description, "graph": graph})


# ---------------------------------------------------------------------------
# POST — create
# ---------------------------------------------------------------------------

class TestCreate:
    def test_returns_201(self, client):
        r = _create(client)
        assert r.status_code == 201

    def test_uuid_is_returned(self, client):
        r = _create(client, name="UUID test")
        body = r.json()
        assert "uuid" in body
        assert len(body["uuid"]) == 36  # standard UUID4 format

    def test_name_and_description_round_trip(self, client):
        r = _create(client, name="My graph", description="pipeline v2")
        body = r.json()
        assert body["name"] == "My graph"
        assert body["description"] == "pipeline v2"

    def test_timestamps_are_floats(self, client):
        before = time.time()
        r = _create(client, name="Timestamp test")
        after = time.time()
        body = r.json()
        assert before <= body["created_at"] <= after
        assert before <= body["updated_at"] <= after

    def test_default_name_applied(self, client):
        r = client.post("/model-tests", json={"graph": {}})
        assert r.status_code == 201
        assert r.json()["name"] == "Untitled test"

    def test_empty_graph_accepted(self, client):
        r = client.post("/model-tests", json={"graph": {}})
        assert r.status_code == 201

    def test_nested_graph_accepted(self, client):
        graph = {"a": {"b": {"c": [1, 2, 3]}}, "unicode": "äöü 🔬"}
        r = client.post("/model-tests", json={"name": "nested", "graph": graph})
        assert r.status_code == 201

    def test_missing_graph_returns_422(self, client):
        r = client.post("/model-tests", json={"name": "No graph"})
        assert r.status_code == 422

    def test_duplicate_names_allowed(self, client):
        _create(client, name="Duplicate")
        r2 = _create(client, name="Duplicate")
        assert r2.status_code == 201
        # Each gets a distinct UUID
        r1 = _create(client, name="Duplicate")
        assert r1.json()["uuid"] != r2.json()["uuid"]


# ---------------------------------------------------------------------------
# GET list
# ---------------------------------------------------------------------------

class TestList:
    def test_returns_200_list(self, client):
        r = client.get("/model-tests")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_newest_first(self, client):
        base_name = f"order_{time.time()}"
        ids = []
        for i in range(3):
            r = _create(client, name=f"{base_name}_{i}")
            ids.append(r.json()["uuid"])

        listing = client.get("/model-tests").json()
        listing_ids = [item["uuid"] for item in listing]

        # The last-created item should appear before the first-created
        assert listing_ids.index(ids[2]) < listing_ids.index(ids[0])

    def test_list_items_have_no_graph_field(self, client):
        _create(client, name="list_no_graph")
        listing = client.get("/model-tests").json()
        for item in listing:
            assert "graph" not in item
            assert "uuid" in item
            assert "name" in item
            assert "created_at" in item


# ---------------------------------------------------------------------------
# GET by UUID
# ---------------------------------------------------------------------------

class TestLoadDetail:
    def test_returns_200_with_graph(self, client):
        created = _create(client, name="detail test").json()
        r = client.get(f"/model-tests/{created['uuid']}")
        assert r.status_code == 200
        body = r.json()
        assert "graph" in body
        assert body["graph"] == _SAMPLE_GRAPH

    def test_graph_preserved_exactly(self, client):
        graph = {"nodes": [], "edges": [], "meta": {"version": 3, "tag": "äöü"}}
        created = client.post("/model-tests", json={"graph": graph}).json()
        loaded = client.get(f"/model-tests/{created['uuid']}").json()
        assert loaded["graph"] == graph

    def test_404_for_unknown_uuid(self, client):
        r = client.get("/model-tests/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_uuid_matches_in_detail(self, client):
        created = _create(client).json()
        detail = client.get(f"/model-tests/{created['uuid']}").json()
        assert detail["uuid"] == created["uuid"]


# ---------------------------------------------------------------------------
# PUT — update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_name_and_description(self, client):
        created = _create(client, name="original", description="old").json()
        r = client.put(
            f"/model-tests/{created['uuid']}",
            json={"name": "updated", "description": "new", "graph": _SAMPLE_GRAPH},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "updated"
        assert body["description"] == "new"

    def test_update_graph_reflected_on_load(self, client):
        created = _create(client).json()
        new_graph = {"nodes": [{"id": "x", "type": "output"}], "edges": []}
        client.put(
            f"/model-tests/{created['uuid']}",
            json={"graph": new_graph},
        )
        detail = client.get(f"/model-tests/{created['uuid']}").json()
        assert detail["graph"] == new_graph

    def test_updated_at_increments(self, client):
        created = _create(client).json()
        time.sleep(0.01)
        updated = client.put(
            f"/model-tests/{created['uuid']}",
            json={"graph": {}, "name": "bump"},
        ).json()
        assert updated["updated_at"] >= created["updated_at"]

    def test_created_at_unchanged_after_update(self, client):
        created = _create(client).json()
        updated = client.put(
            f"/model-tests/{created['uuid']}",
            json={"graph": {}, "name": "no-change"},
        ).json()
        assert updated["created_at"] == created["created_at"]

    def test_404_for_unknown_uuid(self, client):
        r = client.put(
            "/model-tests/00000000-0000-0000-0000-000000000000",
            json={"graph": {}, "name": "ghost"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------

class TestDelete:
    def test_returns_deleted_uuid(self, client):
        created = _create(client, name="to delete").json()
        r = client.delete(f"/model-tests/{created['uuid']}")
        assert r.status_code == 200
        assert r.json()["deleted"] == created["uuid"]

    def test_deleted_item_gone_from_list(self, client):
        created = _create(client, name="ephemeral").json()
        client.delete(f"/model-tests/{created['uuid']}")
        listing = [item["uuid"] for item in client.get("/model-tests").json()]
        assert created["uuid"] not in listing

    def test_deleted_item_returns_404_on_load(self, client):
        created = _create(client, name="del-then-get").json()
        client.delete(f"/model-tests/{created['uuid']}")
        r = client.get(f"/model-tests/{created['uuid']}")
        assert r.status_code == 404

    def test_second_delete_returns_404(self, client):
        created = _create(client, name="double-delete").json()
        client.delete(f"/model-tests/{created['uuid']}")
        r = client.delete(f"/model-tests/{created['uuid']}")
        assert r.status_code == 404

    def test_404_for_unknown_uuid(self, client):
        r = client.delete("/model-tests/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404
