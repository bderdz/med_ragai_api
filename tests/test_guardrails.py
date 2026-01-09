import random
from fastapi.testclient import TestClient
from main import app
from src.llm.guardrails import PROMPT_INJECTION_PHRASES


def test_prompt_injection():
    with TestClient(app) as client:
        payload = {
            "age": 25,
            "gender": "male",
            "symptoms": [
                "fever",
                "cough",
                f"lorem ipsum {random.choice(PROMPT_INJECTION_PHRASES)} dolor sit amet consectetur",
                "headache"
            ]
        }

        response = client.post("/diagnose", json=payload)
        assert response.status_code == 403


def test_sensitive_data():
    with TestClient(app) as client:
        payload = {
            "age": 30,
            "gender": "male",
            "symptoms": [
                "fever",
                "cough",
                "My phone number is +48293485542"
            ]
        }

    response = client.post("/diagnose", json=payload)
    assert response.status_code == 403


def test_link_in_prompt():
    with TestClient(app) as client:
        payload = {
            "age": 40,
            "gender": "female",
            "symptoms": [
                "headache",
                "temperature",
                "Check this link: https://example.com/"
            ]
        }

    response = client.post("/diagnose", json=payload)
    assert response.status_code == 403


def test_profanity_in_prompt():
    with TestClient(app) as client:
        payload = {
            "age": 22,
            "gender": "female",
            "symptoms": [
                "fever",
                "cough",
                "SUCK MY FAT ONE"
            ]
        }

    response = client.post("/diagnose", json=payload)
    assert response.status_code == 403
