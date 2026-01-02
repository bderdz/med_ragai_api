import logging, os, pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.rag.process_csv import prepare_docs

logger = logging.getLogger(__name__)

load_dotenv()

DATASET_FILENAME = os.getenv("DATASET_FILENAME")
DB_PATH = os.getenv("DB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def get_vectors_store() -> Chroma:
    """
    Load or create a Chroma vector store from the CSV dataset by using prepare_docs.
    Uses HuggingFaceEmbeddings local embeddings model specified in EMBEDDING_MODEL env. variable.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

    if os.path.exists(DB_PATH):
        logger.info(f"Loading existing vector store from: {DB_PATH}")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        logger.info(f"Vector store not found. Creating new vector store in: f{DB_PATH}")
        df = pd.read_csv(DATASET_FILENAME)
        docs = prepare_docs(df, DATASET_FILENAME)

        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_PATH)
