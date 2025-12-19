from fastapi import FastAPI, Depends, APIRouter
from src.schemas import SymptomsInput, DiagnoseResponse
from langchain_core.messages import HumanMessage
from src.dependencies import get_agent_state

router = APIRouter()


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(symptoms: SymptomsInput, agent=Depends(get_agent_state)) -> DiagnoseResponse:
    prompt = f"Gender: {symptoms.gender}, age: {symptoms.age}, symptoms: {', '.join(symptoms.symptoms)}."
    response = agent.invoke({"messages": [HumanMessage(content=prompt)]})["messages"][-1].content
    print(response)
    return DiagnoseResponse(possible_diseases=[])
