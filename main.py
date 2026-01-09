import os, uvicorn, gradio as gr
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from src.rag.vectors_store import get_vectors_store
from src.llm import DiagnosisAssistant
from src.routes import diagnosis
from src.ui import ChatAgentUI
from logs import init_logging

load_dotenv()
init_logging()

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


# API initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    vectors_store = get_vectors_store()
    rag_assistant = DiagnosisAssistant(vectors_store=vectors_store)
    app.state.rag_assistant = rag_assistant
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(diagnosis.router)

# UI layer above API with chat local agent
chat_ui = ChatAgentUI(local_model=LOCAL_MODEL)
app = gr.mount_gradio_app(app, chat_ui.ui, path="/")
