from fastapi import Request

from src.llm import DiagnosisAssistant


def get_rag_assistant(request: Request) -> DiagnosisAssistant:
    if not hasattr(request.app.state, "rag_assistant"):
        raise RuntimeError("ERROR: RAG Diagnosis Assistant not initialized")
    return request.app.state.rag_assistant
