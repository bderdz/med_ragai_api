import os
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.process_csv import prepare_docs

load_dotenv()

DATASET_FILENAME = os.getenv("DATASET_FILENAME")
DB_PATH = os.getenv("DB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def get_vectors_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

    if os.path.exists(DB_PATH):
        print("INFO: Loading existing vector store from:", DB_PATH)
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        print("INFO: Vector store not found. Creating new vector store in:", DB_PATH)
        df = pd.read_csv(DATASET_FILENAME)
        docs = prepare_docs(df, DATASET_FILENAME)

        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_PATH)
