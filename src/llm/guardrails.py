import logging
import re


class SecurityError(Exception):
    """Guardrails related security error (prompt injection etc.)"""
    pass


PROMPT_INJECTION_PHRASES = [
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

RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"(?:\+?48)?\s?(?:\d[ -]?){9,}")
RE_PESEL = re.compile(r"\b\d{11}\b")
RE_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
RE_IBAN = re.compile(r"\bPL\d{26}\b", re.IGNORECASE)
RE_LINK = re.compile(r"https?://\S+")
PROFANITY = {"fuck", "bitch", "shit", "asshole", "suck my fat one", "damn", "idiot", "stupid", "dumb"}


def detect_prompt_injection(prompt: str) -> None:
    lowered_prompt = prompt.lower()
    for phrase in PROMPT_INJECTION_PHRASES:
        if phrase in lowered_prompt:
            logging.warning(f"SECURITY: Prompt injection detected with phrase: '{phrase}'")
            raise SecurityError(f"Request rejected due to prompt injection attempt. Phrase: '{phrase}'")


def detect_profanity(prompt: str) -> None:
    lowered_prompt = prompt.lower()
    for bad_word in PROFANITY:
        if bad_word in lowered_prompt:
            logging.warning(f"SECURITY: Profanity detected with word: '{bad_word}'")
            raise SecurityError(f"Request rejected due to profanity attempt. Word: '{bad_word}'")


def detect_links(prompt: str) -> None:
    if RE_LINK.search(prompt):
        logging.warning("SECURITY: Sensitive data detected - link/URL")
        raise SecurityError("Request rejected due to sensitive data (link/URL) detected.")


def detect_sensitive_data(prompt: str) -> None:
    if RE_EMAIL.search(prompt):
        logging.warning("SECURITY: Sensitive data detected - email address")
        raise SecurityError("Request rejected due to sensitive data (email address) detected.")
    if RE_PHONE.search(prompt):
        logging.warning("SECURITY: Sensitive data detected - phone number")
        raise SecurityError("Request rejected due to sensitive data (phone number) detected.")
    if RE_PESEL.search(prompt):
        logging.warning("SECURITY: Sensitive data detected - PESEL number")
        raise SecurityError("Request rejected due to sensitive data (PESEL number) detected.")
    if RE_CARD.search(prompt):
        logging.warning("SECURITY: Sensitive data detected - credit card number")
        raise SecurityError("Request rejected due to sensitive data (credit card number) detected.")
    if RE_IBAN.search(prompt):
        logging.warning("SECURITY: Sensitive data detected - IBAN number")
        raise SecurityError("Request rejected due to sensitive data (IBAN number) detected.")


def run_guardrails(prompt: str) -> None:
    detect_prompt_injection(prompt)
    detect_sensitive_data(prompt)
    detect_links(prompt)
    detect_profanity(prompt)
