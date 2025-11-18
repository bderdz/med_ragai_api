import json
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_classic.agents import Agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.documents import Document

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.1
)

# Load documents
documents: list[Document] = []

with open("disease_docs/disease_documents.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        doc = Document(page_content=data["content"], metadata=data["metadata"])
        documents.append(doc)

    print("INFO: Loaded", len(documents), "documents.")

# RAG vector store setup
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="disease_retriever",
    description="Contains information about diseases and their symptoms. Use this tool to look up diseases based on symptoms provided by the patient."
)

# Agent setup
system = """
You are highly capable medical assistant. Your task is to analyze patient symptoms, age and gender and suggest list of possible diseases they might have.
You must only provide disease suggestions based on the information you retrieve using the 'disease_retriever' tool. DON'T use any other knowledge you might have.
Based on the retrieved medical data, you must provide a prioritized list of the 3 most probable differential diagnoses that match the patient's symptoms.
OUTPUT FORMAT:
disease 1, disease 2, disease 3
If you are unable to find any relevant information, respond with "No relevant data found to suggest possible diseases."
"""

agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt=system
)

print("INFO: RAG Agent is loaded and ready.")
