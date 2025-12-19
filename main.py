import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from src.agent import get_vectors_store, get_agent
from src.routes import diagnosis

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectors_store = get_vectors_store()
    app.state.agent = get_agent(vectors_store)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(diagnosis.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Health Diagnosis RAG AI API"}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
