from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from rag_agent import agent
from langchain_core.messages import HumanMessage

app = FastAPI()


class SymptomsInput(BaseModel):
    age: int
    gender: Literal["male", "female"]
    symptoms: list[str]


class DiagnoseResponse(BaseModel):
    possible_diseases: list[str]


@app.get("/")
async def root():
    return {"message": "Welcome to the Health Diagnosis RAG AI API"}


@app.get("/diagnose", response_model=DiagnoseResponse)
async def diagnose(symptoms: SymptomsInput):
    prompt = f"Gender: {symptoms.gender}, age: {symptoms.age}, symptoms: {', '.join(symptoms.symptoms)}."
    response = agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })["messages"][-1].content

    possible_diseases = response.split(", ")
    return DiagnoseResponse(possible_diseases)
