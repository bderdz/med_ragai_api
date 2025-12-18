import uvicorn
from typing import Literal
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.vectors_store import get_vectors_store
from src.rag_agent import get_agent


class SymptomsInput(BaseModel):
    age: int
    gender: Literal["male", "female"]
    symptoms: list[str]


class DiagnoseResponse(BaseModel):
    possible_diseases: list[str]


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectors_store = get_vectors_store()
    app.state.agent = get_agent(vectors_store)

    yield


app = FastAPI(lifespan=lifespan)


def get_agent_state(request: Request):
    if not hasattr(request.app.state, "agent"):
        raise RuntimeError("ERROR: Agent not initialized")
    return request.app.state.agent


@app.get("/")
async def root():
    return {"message": "Welcome to the Health Diagnosis RAG AI API"}


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(symptoms: SymptomsInput, agent=Depends(get_agent_state)):
    prompt = f"Gender: {symptoms.gender}, age: {symptoms.age}, symptoms: {', '.join(symptoms.symptoms)}."
    response = agent.invoke({"messages": [HumanMessage(content=prompt)]})["messages"][-1].content
    print(response)
    return DiagnoseResponse(possible_diseases=[])


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
