"""
Smoke tests for the FastAPI surface.
Disables the Gradio mount so test imports stay fast and side-effect free.
"""
from __future__ import annotations

import os
os.environ.setdefault("ENABLE_WEB_INTERFACE", "false")

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["environment"] == "DataClean-Env"


def test_tasks_listing(client: TestClient):
    r = client.get("/tasks")
    assert r.status_code == 200
    body = r.json()
    assert "tasks" in body and len(body["tasks"]) == 3
    assert "action_schema" in body
    assert "reward_info" in body


def test_reset_then_state_then_step(client: TestClient):
    reset = client.post("/reset", json={"task_id": "task_1", "seed": 42}).json()
    episode_id = reset["episode_id"]
    assert reset["observation"]["task_id"] == "task_1"

    state = client.get(f"/state?episode_id={episode_id}").json()
    assert state["episode_id"] == episode_id
    assert state["step"] == 0
    assert state["done"] is False

    step = client.post("/step", json={
        "episode_id": episode_id,
        "action": {
            "action_type": "remove_duplicates",
            "column": None,
            "params": {},
            "confidence": 0.9,
        },
    }).json()
    assert "reward" in step
    assert "reward_breakdown" in step
    assert step["observation"]["step"] == 1


def test_reset_with_invalid_task(client: TestClient):
    r = client.post("/reset", json={"task_id": "task_999", "seed": 42})
    assert r.status_code == 400


def test_step_unknown_episode(client: TestClient):
    r = client.post("/step", json={
        "episode_id": "00000000-0000-0000-0000-000000000000",
        "action": {"action_type": "done", "confidence": 0.5},
    })
    assert r.status_code == 404


def test_grader_after_reset(client: TestClient):
    reset = client.post("/reset", json={"task_id": "task_1", "seed": 42}).json()
    r = client.post("/grader", json={"episode_id": reset["episode_id"]})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert "quality_scores" in body


def test_baseline_runs(client: TestClient):
    r = client.get("/baseline?seed=42")
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"].keys()) == {"task_1", "task_2", "task_3"}
