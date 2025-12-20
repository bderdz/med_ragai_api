from fastapi import Request


def get_rag_assistant(request: Request):
    if not hasattr(request.app.state, "rag_assistant"):
        raise RuntimeError("ERROR: RAG Diagnosis Assistant not initialized")
    return request.app.state.rag_assistant
