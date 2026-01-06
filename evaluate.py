import logging, os, pandas as pd
from dotenv import load_dotenv
from src.rag.vectors_store import get_vectors_store
from logs import init_logging

load_dotenv()
DATASET_FILENAME = os.getenv("DATASET_FILENAME")

# Set up logging
metrics_logger = logging.getLogger("metrics")


def recall_evaluation(sample_size: int = 30, k=6):
    df = pd.read_csv(DATASET_FILENAME)
    symptom_cols = df.columns.drop(['prognosis', 'icd_code'])
    vector_store = get_vectors_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

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

        hit = False
        for doc in docs:
            if disease in doc.page_content.lower():
                hit = True
                break

        if hit:
            hits += 1
        else:
            logging.info(f"RECALL MISS: Disease='{disease}' Symptoms='{query}' Retrieved='{[doc.page_content for doc in docs]}'")

    recall = hits / sample_size
    metrics_logger.info(f"RECALL EVALUATION: sample_size={sample_size} k={k} hits={hits} recall={recall:.2%}")


if __name__ == '__main__':
    init_logging()
    recall_evaluation(sample_size=50, k=6)
