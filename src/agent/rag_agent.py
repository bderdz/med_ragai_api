import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
SYSTEM = """
You are highly capable medical assistant. Your task is to analyze patient symptoms, age and gender and suggest list of possible diseases they might have.
You must only provide disease suggestions based on the information you retrieve using the 'disease_retriever' tool. DON'T use any other knowledge you might have.
Based on the retrieved top-k probable disease with their symptoms and probabilities of appearance, you must select a list of the 3 most probable diagnoses that match the patient's symptoms.
OUTPUT FORMAT:
disease 1, disease 2, disease 3
If you are unable to find any relevant information, respond with "No relevant data found to suggest possible diseases."
"""


def get_agent(vectors_store: Chroma):
    """
    Creates a RAG agent using Gemini model with a retriever tool for searching diseases based on symptoms.
    Needs a Chroma vector store as input.
    """
    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2
    )
    retriever = vectors_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="disease_retriever",
        description="Contains information about diseases and their symptoms. Use this tool to look up diseases based on symptoms provided by the patient."
    )

    agent = create_agent(
        model=model,
        tools=[retriever_tool],
        system_prompt=SYSTEM
    )

    print("INFO: RAG Agent created successfully.")
    return agent
