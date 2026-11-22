import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(request):
    from triage.config import load_config

    config = load_config()
    if not (config.paths.artifacts_dir / "screener.pt").exists():
        pytest.skip("train screener before running the service tests")
    from triage.serve import app

    with TestClient(app) as running:
        yield running


def test_healthz_reports_the_loaded_model(client):
    payload = client.get("/healthz").json()
    assert payload["status"] == "ok"
    assert payload["labels"] == 78


def test_triage_returns_a_full_decision(client):
    payload = client.post("/triage", json={"ticket_id": "API1", "text": "my new card has still not arrived"}).json()
    assert payload["ticket_id"] == "API1"
    assert payload["status"] in {"auto", "human"}
    assert payload["queue"]
    assert payload["priority"].startswith("P")
    assert payload["reply"]
    assert 0.0 <= payload["confidence"] <= 1.0


def test_batch_endpoint_handles_several_tickets(client):
    body = {"tickets": [{"text": "where is my card"}, {"text": "I lost my card in Paris"}]}
    decisions = client.post("/triage/batch", json=body).json()["decisions"]
    assert len(decisions) == 2
    assert all(decision["executed"] for decision in decisions)


def test_empty_ticket_is_rejected(client):
    assert client.post("/triage", json={"text": ""}).status_code == 422


def test_metrics_count_decisions(client):
    before = client.get("/metrics").json()["requests"]
    client.post("/triage", json={"text": "why was I charged a fee"})
    after = client.get("/metrics").json()
    assert after["requests"] == before + 1
    assert 0.0 <= after["automation_rate"] <= 1.0
