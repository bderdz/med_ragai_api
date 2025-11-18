from typing import TypedDict
import pandas as pd
from pydantic import BaseModel

DATASET_FILENAME = "DiseaseAndSymptoms.csv"

df = pd.read_csv(DATASET_FILENAME)

# List of symptom columns
symptoms_cols = [col for col in df.columns if col.startswith("Symptom_")]
# Clean and format symptom columns
for col in symptoms_cols:
    df[col] = df[col].str.lower().str.replace("_", " ").str.strip()

df["Symptoms"] = df[symptoms_cols].values.tolist()
df["Symptoms"] = df["Symptoms"].apply(lambda x: [symptom for symptom in x if pd.notna(symptom)])
df = df.groupby("Disease")["Symptoms"].apply(lambda x: list(set(s for sublist in x for s in sublist))).reset_index()


# Prepare documents for RAG
class MetaData(TypedDict):
    disease: str
    source: str


def create_content(row: pd.Series) -> str:
    if not row["Symptoms"]:
        return f"disease: {row['Disease']}\n symptoms: not available"

    return f"disease: {row['Disease']}\n symptoms: {', '.join(row['Symptoms'])}"


def create_metadata(row: pd.Series) -> MetaData:
    return MetaData(disease=row["Disease"], source="DiseaseAndSymptoms.csv")


df["content"] = df.apply(create_content, axis=1)
df["metadata"] = df.apply(create_metadata, axis=1)

df_rag = df[["content", "metadata"]]
df_rag.to_json("disease_documents.jsonl", orient="records", lines=True, force_ascii=False)
