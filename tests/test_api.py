import random
from http.client import responses

from fastapi.testclient import TestClient
from main import app
from src.llm.guardrails import GUARDRAILS_PHRASES


def test_api_works():
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200


def test_diagnose_endpoint():
    with TestClient(app) as client:
        payload = {
            "age": 20,
            "gender": "female",
            "symptoms": ["fever", "cough"]
        }
        response = client.post("/diagnose", json=payload)

        # Response status
        assert response.status_code == 200
        data = response.json()

        # Response structure
        assert "possible_diseases" in data
        assert isinstance(data["possible_diseases"], list)
        assert len(data["possible_diseases"]) > 0

        # Disease details structure
        disease = data["possible_diseases"][0]
        assert "name" in disease
        assert "icd_code" in disease
        assert "reasoning" in disease


def test_diagnose_validation():
    with TestClient(app) as client:
        # Invalid age
        payload = {
            "age": -5,
            "gender": "male",
            "symptoms": ["headache", "temperature"]
        }
        response = client.post("/diagnose", json=payload)
        assert response.status_code == 422

        # Invalid gender
        payload = {
            "age": 30,
            "gender": "woman",
            "symptoms": ["headache", "temperature"]
        }
        response = client.post("/diagnose", json=payload)
        assert response.status_code == 422

        # No symptoms
        payload = {
            "age": 20,
            "gender": "female",
            "symptoms": []
        }
        response = client.post("/diagnose", json=payload)
        assert response.status_code == 422


def test_diagnose_prompt_injection():
    with TestClient(app) as client:
        payload = {
            "age": 25,
            "gender": "male",
            "symptoms": [
                "fever",
                "cough",
                f"lorem ipsum {random.choice(GUARDRAILS_PHRASES)} dolor sit amet consectetur",
                "headache"
            ]
        }

        response = client.post("/diagnose", json=payload)
        assert response.status_code == 403
