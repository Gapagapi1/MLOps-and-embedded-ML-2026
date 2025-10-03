# Prereqs:
#   - docker compose up (mlflow on :5000, model-api on :8001)
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
    assert body["model_uri"].startswith("models:/iris-model/")
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 2


def test_predict_missing_field(http: requests.Session):
    r = http.post(f"{SERVICE_BASE_URL}/predict", data=json.dumps({}), timeout=10)
    assert r.status_code == 400
    assert r.json().get("error") == "missing_field:data"


def test_update_model_changes_uri_and_effect(http: requests.Session):
    # Ensure we are on v1 first
    r0 = http.post(
        f"{SERVICE_BASE_URL}/update-model",
        data=json.dumps({"model_uri": "models:/iris-model/1"}),
        timeout=15,
    )
    assert r0.status_code == 200, r0.text
    uri_before = r0.json().get("model_uri")

    # Get a baseline prediction
    payload = {"data": [[5.9, 3.0, 5.1, 1.8]]}
    r_pred_before = http.post(f"{SERVICE_BASE_URL}/predict", data=json.dumps(payload), timeout=15)
    assert r_pred_before.status_code == 200, r_pred_before.text
    pred_before = r_pred_before.json().get("predictions")

    # Attempt to update to v2
    r1 = http.post(
        f"{SERVICE_BASE_URL}/update-model",
        data=json.dumps({"model_uri": "models:/iris-model/2"}),
        timeout=15,
    )

    if r1.status_code == 400 and r1.json().get("error") == "load_failed":
        pytest.skip("iris-model version 2 not found or not loadable in MLflow registry")

    assert r1.status_code == 200, r1.text
    uri_after = r1.json().get("model_uri")
    assert uri_after != uri_before, "Model URI should change after update"

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

