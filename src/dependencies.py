from fastapi import Request


def get_agent_state(request: Request):
    if not hasattr(request.app.state, "agent"):
        raise RuntimeError("ERROR: Agent not initialized")
    return request.app.state.agent
