import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.4
)

# RAG vector store setup
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = Chroma.from_texts(texts=[], embedding_function=embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="...",
    description="..."
)

# Agent setup
agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt="..."
)
