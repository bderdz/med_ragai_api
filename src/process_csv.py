import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
import pandas as pd


class MetaData(TypedDict):
    """
     Metadata template for documents
    """
    disease: str
    icd_code: str
    source: str


def prepare_docs(df: pd.DataFrame, source: str) -> list[Document]:
    docs = []
    # List of symptom columns
    symptom_cols = df.columns.drop(['prognosis', 'icd_code'])

    for _, row in df.iterrows():
        disease = row['prognosis']
        icd_code = row['icd_code']
        content = [f'Disease: {disease} ICD CODE: {icd_code}',
                   'Symptoms and probabilities of appearance:']

        symptoms = []
        for symptom in symptom_cols:
            if row[symptom] > 0.0:
                symptoms.append(
                    f"- {symptom.replace('_', ' ')} {row[symptom]}%")

        doc_content = "\n".join(content + symptoms)
        doc_metadata = MetaData(
            disease=disease, icd_code=icd_code, source=source)
        doc = Document(page_content=doc_content, metadata=doc_metadata)
        docs.append(doc)

    print(f"INFO: Loaded {len(docs)} documents from {source}.")
    return docs
