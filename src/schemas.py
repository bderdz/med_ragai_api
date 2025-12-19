from typing import Literal

from pydantic import BaseModel


class SymptomsInput(BaseModel):
    """Input schema for diagnose endpoint"""
    age: int
    gender: Literal["male", "female"]
    symptoms: list[str]


class DiagnoseResponse(BaseModel):
    """Response schema for diagnose endpoint"""
    possible_diseases: list[str]
