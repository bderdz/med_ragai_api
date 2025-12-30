import httpx
from typing import Any

from src.schemas import SymptomsInput


async def get_diagnosis_tool(gender: str, age: int, symptoms: list[str]) -> dict[str, Any]:
    """
    Calls the /diagnose API endpoint with the provided patient information.
    Returns the diagnosis result or an error message.
    Format:
    {
        "possible_diseases": [
            {
                "name": "Disease Name",
                "icd_code": "ICD Code",
                "reasoning": "Explanation"
            },
            ...
        ]
    }
    or in case of error:
    {
        "error": "Error message"
    }
    """
    # Input validation
    print("TOOL INPUT", gender, age, symptoms)
    if age <= 0:
        return {"error": "Age must be a positive integer"}
    if not gender:
        return {"error": "Please select a gender"}
    if not symptoms:
        return {"error": "Please enter at least one symptom"}

    body = SymptomsInput(age=age, gender=gender, symptoms=symptoms)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post("http://localhost:8000/diagnose", json=body.model_dump())
            response.raise_for_status()
            print(response)
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"API ERROR {e.response.status_code}: {e.response.text}"}
        except httpx.RequestError as e:
            return {"error": f"REQUEST ERROR: {e}"}
