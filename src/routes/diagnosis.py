from fastapi import FastAPI, Depends, APIRouter
from src.schemas import SymptomsInput, DiagnoseResponse
from langchain_core.messages import HumanMessage
from src.dependencies import get_rag_assistant

router = APIRouter()


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(symptoms: SymptomsInput, rag_assistant=Depends(get_rag_assistant)) -> DiagnoseResponse:
    response: DiagnoseResponse = rag_assistant.diagnose(symptoms)
    return response
