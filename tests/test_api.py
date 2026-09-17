from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "API is runningo"

def test_predict():
    response = client.post(
        "/predict",
        json={
            "age": 30,
            "tenure_months": 12,
            "monthly_charges": 70.0,
            "support_calls": 3,
            "contract_type": "Monthly",
            "usage_hours": 100.0
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["churn_prediction"] in [0, 1]
    assert 0 <= data["churn_probability"] <= 1