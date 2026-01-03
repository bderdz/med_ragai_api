from http.client import HTTPException

from fastapi import FastAPI, Depends, APIRouter, HTTPException
from src.schemas import SymptomsInput, DiagnoseResponse
from src.dependencies import get_rag_assistant
from src.llm.guardrails import SecurityError

router = APIRouter()


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(symptoms: SymptomsInput, rag_assistant=Depends(get_rag_assistant)) -> DiagnoseResponse:
    try:
        response: DiagnoseResponse = rag_assistant.diagnose(symptoms)
        return response
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
