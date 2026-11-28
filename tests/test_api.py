import pytest
from fastapi.testclient import TestClient

from api import main

TEST_API_KEY = "test-agroyield-api-key"

VALID_PAYLOAD = {
    "start_year": 2020,
    "area": 21400,
    "state_code": "28",
    "district_code": "502",
    "crop_code": "202.0",
    "crop_name": "Arhar/Tur",
    "crop_type": "Pulses",
    "season": "Kharif",
}


class FakePredictor:
    """Mock predictor that avoids loading the real CatBoost model."""

    def __init__(self, prediction=2.5):
        self.prediction = prediction

    def predict(self, payload):
        return self.prediction

    @staticmethod
    def validate_prediction(prediction):
        import math

        return math.isfinite(prediction) and prediction >= 0


@pytest.fixture
def client_factory(monkeypatch):
    """Create a test client with a mocked predictor."""

    def create_client(prediction=2.5):
        monkeypatch.setenv("AGROYIELD_API_KEY", TEST_API_KEY)

        monkeypatch.setattr(
            main,
            "AgroYieldPredictor",
            lambda: FakePredictor(prediction),
        )

        return TestClient(main.app)

    return create_client


def test_health_check(client_factory):
    with client_factory() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True
    assert response.json()["model_name"] == "catboost_v1"


def test_prediction_rejects_missing_api_key(client_factory):
    with client_factory() as client:
        response = client.post(
            "/api/v1/predict",
            json=VALID_PAYLOAD,
        )

    assert response.status_code == 401


def test_prediction_rejects_invalid_api_key(client_factory):
    with client_factory() as client:
        response = client.post(
            "/api/v1/predict",
            json=VALID_PAYLOAD,
            headers={"X-API-Key": "incorrect-key"},
        )

    assert response.status_code == 401


def test_prediction_succeeds_with_valid_api_key(client_factory):
    with client_factory() as client:
        response = client.post(
            "/api/v1/predict",
            json=VALID_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "success"
    assert result["predicted_yield"] == 2.5
    assert result["unit"] == "tonnes/hectare"
    assert result["model_name"] == "catboost_v1"
    assert result["warnings"] == []


def test_prediction_rejects_invalid_input(client_factory):
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["area"] = 0

    with client_factory() as client:
        response = client.post(
            "/api/v1/predict",
            json=invalid_payload,
            headers={"X-API-Key": TEST_API_KEY},
        )

    assert response.status_code == 422


def test_prediction_warns_for_unevaluated_year(client_factory):
    payload = VALID_PAYLOAD.copy()
    payload["start_year"] = 2023

    with client_factory() as client:
        response = client.post(
            "/api/v1/predict",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "success"
    assert result["warnings"]
    assert "outside the model's evaluated" in result["warnings"][0]


def test_negative_prediction_is_not_returned_as_valid(client_factory):
    with client_factory(prediction=-1.5) as client:
        response = client.post(
            "/api/v1/predict",
            json=VALID_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "invalid_prediction"
    assert result["predicted_yield"] is None
    assert any("negative or non-finite" in warning
               for warning in result["warnings"])
