import logging, os, pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.rag.vectors_store import get_vectors_store
from logs import init_logging

load_dotenv()
DATASET_FILENAME = os.getenv("DATASET_FILENAME")

# Set up logging
metrics_logger = logging.getLogger("metrics")

# Initialize CrossEncoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank_docs(query: str, docs: list[Document], top_k: int) -> list[Document]:
    # Reranking
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    # Choose only Top 6 docs
    top_k_docs = [doc for _, doc in scored_docs[:top_k]]

    return top_k_docs


def recall_evaluation(sample_size: int = 30, k=6, rerank: bool = False) -> None:
    df = pd.read_csv(DATASET_FILENAME)
    symptom_cols = df.columns.drop(['prognosis', 'icd_code'])
    vector_store = get_vectors_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k if not rerank else k * 2})

    test_set = df.sample(n=sample_size)
    hits = 0

    for _, row in test_set.iterrows():
        disease = row['prognosis'].lower()

        symptoms = []
        for symptom in symptom_cols:
            if row[symptom] > 0.0:
                symptoms.append(symptom.replace('_', ' '))

        query = ", ".join(symptoms)
        docs = retriever.invoke(query)
        if rerank:
            docs = rerank_docs(query, docs, top_k=k)

        hit = False
        for doc in docs:
            if disease in doc.page_content.lower():
                hit = True
                break

        if hit:
            hits += 1
        else:
            logging.info(f"RECALL MISS: rerank={rerank} disease={disease} symptoms={query} retrieved={[doc.metadata for doc in docs]}")

    recall = hits / sample_size
    metrics_logger.info(f"RECALL EVALUATION: rerank={rerank} sample_size={sample_size} k={k} hits={hits} recall={recall:.2%}")


if __name__ == '__main__':
    init_logging()
    recall_evaluation(sample_size=80, k=6)
    recall_evaluation(sample_size=80, k=6, rerank=True)
