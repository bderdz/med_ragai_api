import httpx, logging
from typing import Any
from src.llm.dispatcher import ToolValidationError, ToolError
from src.schemas import SymptomsInput

logger = logging.getLogger(__name__)


async def get_diagnosis_tool(gender: str, age: int, symptoms: list[str]) -> dict[str, Any]:
    """
    Calls the /diagnose API endpoint with the provided patient information.
    Returns the diagnosis result as a JSON dictionary with next schemas: DiagnoseResponse(list[DiseaseDetails])
    """
    # Input validation
    logger.debug(f"EXECUTING TOOL 'get_diagnosis_tool' args: gender={gender}, age={age}, symptoms={symptoms}")
    if not age or age <= 0:
        raise ToolValidationError(f"Age must be a positive integer. Received age: {age}")
    if not gender or gender not in ["male", "female"]:
        raise ToolValidationError("Gender is required and must be 'male' or 'female'")
    if not symptoms:
        raise ToolValidationError("At least one symptom must be provided")

    body = SymptomsInput(age=age, gender=gender, symptoms=symptoms)
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post("http://localhost:8000/diagnose", json=body.model_dump())
            response.raise_for_status()

            logger.debug(f"TOOL 'get_diagnosis_tool' OUTPUT: {response}")
            return response.json()
        except httpx.HTTPStatusError as e:
            error_msg = f"API ERROR {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            raise ToolError(error_msg)
        except httpx.RequestError as e:
            logger.error(f"API CONNECTION ERROR: {e}")
            raise ToolError(f"Failed to connect to the diagnosis API: {e}")
