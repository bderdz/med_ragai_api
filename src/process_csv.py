from typing import TypedDict
import pandas as pd

# To print full dataframe while debugging
pd.set_option('display.max_colwidth', None)

DATASET_FILENAME = "../dataset/processed/disease_symptoms.csv"

df = pd.read_csv(DATASET_FILENAME)

# List of symptom columns
symptoms_cols = df.columns.drop(['prognosis', 'icd_code'])


# Prepare documents for RAG
class MetaData(TypedDict):
    disease: str
    source: str


def create_content(row: pd.Series) -> str:
    symptoms: list[str] = []

    for symptom, prob in row.items():
        if symptom not in symptoms_cols:
            continue
        if prob > 0.0:
            symptom = symptom.replace("_", " ")
            symptoms.append(f"{symptom} {prob}%")
    return f"DISEASE: {row['prognosis']}\nICD: {row['icd_code']}\nSYMPTOMS: {', '.join(symptoms) if symptoms else 'Not specified'}"


def create_metadata(row: pd.Series) -> MetaData:
    return MetaData(disease=row["prognosis"], source=DATASET_FILENAME)


df_docs = pd.DataFrame()
df_docs['content'] = df.apply(create_content, axis=1)
df_docs["metadata"] = df.apply(create_metadata, axis=1)

df_docs.to_json("docs/disease_documents.jsonl", orient="records", lines=True, force_ascii=False)
