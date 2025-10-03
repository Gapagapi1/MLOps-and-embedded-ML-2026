# Prereqs:
#   - docker compose up (mlflow on :5000, model-api on :8000)
#   - MLflow has a registered model "iris-model" with at least version 1.
#   - Version 2 is optional; if missing, the corresponding test will be skipped.

import os
import json
import pytest
import requests



SERVICE_BASE_URL = os.getenv("SERVICE_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def http():
    s = requests.Session()
    # quick health check; if the container isn't up, skip all tests
    try:
        r = s.get(f"{SERVICE_BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"Service not healthy at {SERVICE_BASE_URL} (status {r.status_code})")
    except Exception as e:
        pytest.skip(f"Cannot reach service at {SERVICE_BASE_URL}: {e}")
    return s


def _health(http: requests.Session):
    r = http.get(f"{SERVICE_BASE_URL}/health", timeout=5)
    assert r.status_code == 200, r.text
    return r.json()



def test_predict_success_list_of_lists(http: requests.Session):
    payload = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
        ]
    }
    r = http.post(f"{SERVICE_BASE_URL}/predict", data=json.dumps(payload), timeout=10)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("used") in {"current", "next"}
    assert isinstance(body.get("predictions"), list)
    assert len(body["predictions"]) == 2

    h = _health(http)
    if body["used"] == "current":
        assert body["model_uri"] == h.get("current_model_uri")
    else:
        assert body["model_uri"] == h.get("next_model_uri")


def test_predict_missing_field(http: requests.Session):
    r = http.post(f"{SERVICE_BASE_URL}/predict", data=json.dumps({}), timeout=10)
    assert r.status_code == 400
    assert r.json().get("error") == "missing_field:data"


def test_update_next_and_accept_promote(http: requests.Session):
    # Ensure CURRENT is version 1
    r_set_next_v1 = http.post(
        f"{SERVICE_BASE_URL}/update-model",
        data=json.dumps({"model_uri": "models:/iris-model/1"}),
        timeout=15,
    )
    assert r_set_next_v1.status_code == 200, r_set_next_v1.text
    assert r_set_next_v1.json().get("next_model_uri") == "models:/iris-model/1"

    r_accept_v1 = http.post(f"{SERVICE_BASE_URL}/accept-next-model", timeout=10)
    assert r_accept_v1.status_code == 200, r_accept_v1.text
    state = _health(http)
    assert state.get("current_model_uri") == "models:/iris-model/1"
    assert state.get("next_model_uri") == "models:/iris-model/1"

    # Get a baseline prediction
    payload = {"data": [[5.9, 3.0, 5.1, 1.8]]}
    r_pred_before = http.post(f"{SERVICE_BASE_URL}/predict", data=json.dumps(payload), timeout=15)
    assert r_pred_before.status_code == 200, r_pred_before.text
    pred_before = r_pred_before.json().get("predictions")

    # Try to set NEXT to version 2 (if unavailable, skip)
    r_set_next_v2 = http.post(
        f"{SERVICE_BASE_URL}/update-model",
        data=json.dumps({"model_uri": "models:/iris-model/2"}),
        timeout=15,
    )

    if r_set_next_v2.status_code == 400 and r_set_next_v2.json().get("error") == "load_failed":
        pytest.skip("iris-model version 2 not found")

    assert r_set_next_v2.status_code == 200, r_set_next_v2.text
    assert r_set_next_v2.json().get("next_model_uri") == "models:/iris-model/2"

    # CURRENT should still be v1 until we accept
    state = _health(http)
    assert state.get("current_model_uri") == "models:/iris-model/1"
    assert state.get("next_model_uri") == "models:/iris-model/2"

    # Accept NEXT to CURRENT
    r_accept_v2 = http.post(f"{SERVICE_BASE_URL}/accept-next-model", timeout=10)
    assert r_accept_v2.status_code == 200, r_accept_v2.text

    state = _health(http)
    assert state.get("current_model_uri") == "models:/iris-model/2"
    assert state.get("next_model_uri") == "models:/iris-model/2"

    # Compare predictions (they *may* end up equal depending on the models, but at least we exercise the path)
    r_pred_after = http.post(f"{SERVICE_BASE_URL}/predict", data=json.dumps(payload), timeout=15)
    assert r_pred_after.status_code == 200, r_pred_after.text
    pred_after = r_pred_after.json().get("predictions")

    # If they differ, great; if they don't, still fine.
    assert isinstance(pred_after, list)


def test_update_model_missing_uri(http: requests.Session):
    r = http.post(f"{SERVICE_BASE_URL}/update-model", data=json.dumps({}), timeout=10)
    assert r.status_code == 400
    assert r.json().get("error") == "missing_model_uri"
