import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from src.rag.vectors_store import get_vectors_store
from src.rag import DiagnosisAssistant
from src.routes import diagnosis
from src.ui import load_ui
import gradio as gr

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectors_store = get_vectors_store()
    rag_assistant = DiagnosisAssistant(vectors_store=vectors_store)
    app.state.rag_assistant = rag_assistant
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(diagnosis.router)

# @app.get("/")
# async def root() -> dict[str, str]:
#     return {"message": "Welcome to the Health Diagnosis RAG AI API"}


api_ui = load_ui()
app = gr.mount_gradio_app(app, api_ui, path="/")

if __name__ == '__main__':
    # Develop only
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
