import logging

from fastapi import Request

from src.llm import DiagnosisAssistant

logger = logging.getLogger(__name__)


def get_rag_assistant(request: Request) -> DiagnosisAssistant:
    if not hasattr(request.app.state, "rag_assistant"):
        logger.error("RAG Diagnosis Assistant not initialized")
        raise RuntimeError("RAG Diagnosis Assistant not initialized")
    return request.app.state.rag_assistant
