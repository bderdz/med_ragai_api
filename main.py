import json
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Symptoms(BaseModel):
    age: int
    gender: Literal["male", "female"]
    symptoms: list[str]


@app.get("/")
async def root():
    return {"message": "Welcome to the Health Diagnosis RAG AI API"}


@app.get("/diagnose")
async def diagnose(symptoms: Symptoms):
    return symptoms
