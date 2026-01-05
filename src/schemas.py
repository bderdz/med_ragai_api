from typing import Literal
from pydantic import BaseModel, Field


class SymptomsInput(BaseModel):
    """Input schema for diagnose endpoint"""
    age: int = Field(
        ...,
        gt=0,
        lt=101,
        description="Patient's age (must be positive integer)")
    gender: Literal["male", "female"]
    symptoms: list[str] = Field(
        ...,
        min_length=1,
        description="List of patient's symptoms (at least one symptom required)")


class DiseaseDetails(BaseModel):
    """Schema for a single disease suggested by the model"""
    name: str = Field(..., description="Name of the disease from the context")
    icd_code: str = Field(..., description="ICD code of the disease")
    reasoning: str = Field(
        ...,
        description="Short explanation why this disease fits the patient's symptoms, gender and age.")


class DiagnoseResponse(BaseModel):
    """Response schema for diagnose endpoint"""
    possible_diseases: list[DiseaseDetails]
