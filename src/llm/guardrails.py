import logging


class SecurityError(Exception):
    """Guardrails related security error (prompt injection etc.)"""
    pass


GUARDRAILS_PHRASES = [
    "ignore previous instructions",
    "ignore your instructions",
    "you are now a",
    "forget everything above",
    "developer mode",
    "override safety",
    "disregard guidelines",
    "system prompt",
    "jailbreak",
    "act as if",
    "pretend you are",
    "roleplay as",
    "simulate being",
    "bypass restrictions",
    "ignore safeguards",
    "admin override",
    "root access",
]


def detect_prompt_injection(prompt: str) -> None:
    lowered_prompt = prompt.lower()
    for phrase in GUARDRAILS_PHRASES:
        if phrase in lowered_prompt:
            logging.warning(f"SECURITY: Prompt injection detected with phrase: '{phrase}'")
            raise SecurityError(f"Request rejected due to prompt injection attempt. Phrase: '{phrase}'")
