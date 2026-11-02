from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_fare_prediction():
    response = client.post(
        "/predict",
        json={
            "trip_distance": 5.0,
            "trip_duration": 15.0,
            "pickup_hour": 14,
            "passenger_count": 2
        }
    )

    assert response.status_code == 200
    assert "predicted_fare" in response.json()
    assert response.json()["currency"] == "USD"


def test_invalid_trip_distance():
    response = client.post(
        "/predict",
        json={
            "trip_distance": -5,
            "trip_duration": 15,
            "pickup_hour": 14,
            "passenger_count": 2
        }
    )

    assert response.status_code == 422
