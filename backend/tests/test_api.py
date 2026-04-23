from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.service import get_service


def _sample_query_window_id() -> str:
    service = get_service()
    windows = service.windows(split="test", limit=1000)
    first_usable = next(window for window in windows if window["usable_for_memory"])
    return str(first_usable["window_id"])


def test_health_endpoint_works():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["usable_windows"] > 0
    assert {"hopfield", "siamese_temporal", "som"} <= set(payload["available_models"])


def test_models_endpoint_exposes_three_top_level_modes():
    client = TestClient(app)
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_model"] == "hopfield"
    assert [item["key"] for item in payload["models"]] == ["hopfield", "siamese_temporal", "som"]


def test_dashboard_defaults_to_hopfield():
    client = TestClient(app)
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"]["key"] == "hopfield"
    assert payload["headline_metrics"]["representation_size"] > 0


def test_dashboard_supports_siamese_temporal_and_som():
    client = TestClient(app)

    siamese = client.get("/api/dashboard?model=siamese_temporal")
    som = client.get("/api/dashboard?model=som")

    assert siamese.status_code == 200
    assert som.status_code == 200

    siamese_payload = siamese.json()
    som_payload = som.json()

    assert siamese_payload["selected_model"]["key"] == "siamese_temporal"
    assert som_payload["selected_model"]["key"] == "som"
    assert siamese_payload["headline_metrics"]["representation_size"] > 0
    assert som_payload["headline_metrics"]["representation_size"] > 0


def test_windows_endpoint_accepts_explore_page_limit_request():
    client = TestClient(app)
    response = client.get("/api/windows?limit=1500")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload


def test_retrieve_endpoint_returns_same_core_schema_for_all_models():
    client = TestClient(app)
    window_id = _sample_query_window_id()

    for key in ("hopfield", "siamese_temporal", "som"):
        response = client.post("/api/retrieve", json={"model": key, "window_id": window_id, "k": 5})
        assert response.status_code == 200

        payload = response.json()
        assert payload["model"]["key"] == key
        assert payload["query_window"]["window_id"] == window_id
        assert len(payload["neighbors"]) == 5
        assert "summary_text" in payload
        assert "primary_metrics" in payload
        assert "chart_payload" in payload
        assert "advanced" in payload


def test_legacy_memory_retrieve_alias_preserves_hopfield_access():
    client = TestClient(app)
    window_id = _sample_query_window_id()
    response = client.post("/api/memory/retrieve?model=hopfield", json={"window_id": window_id})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"]["key"] == "hopfield"
    assert payload["query_window"]["window_id"] == window_id


def test_evaluation_endpoint_returns_comparison_rows_for_models_and_baselines():
    client = TestClient(app)
    response = client.get("/api/evaluation?model=som")
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"] == "som"
    row_keys = {row["key"] for row in payload["comparison_rows"]}
    assert {"hopfield", "siamese_temporal", "som"} <= row_keys
    assert payload["comparison_chart"]["kind"] == "primary_metrics"
