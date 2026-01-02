import os, logging, uvicorn, gradio as gr
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from src.rag.vectors_store import get_vectors_store
from src.llm import DiagnosisAssistant
from src.routes import diagnosis
from src.ui import ChatAgentUI

load_dotenv()

LOCAL_MODEL = os.getenv("LOCAL_MODEL")

# Logging
LOG_FILE = "logs/app.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logging.getLogger("src").setLevel(logging.DEBUG)


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

# Develop only
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
