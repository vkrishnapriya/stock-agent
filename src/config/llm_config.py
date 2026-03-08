"""
src/config/llm_config.py
LLM provider routing. Always call get_llm() — never instantiate providers directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_GEMINI_MODEL = "gemini-2.0-flash-exp"
_OLLAMA_DEFAULT_MODEL = "llama3.1:8b"


@lru_cache(maxsize=1)
def get_llm(provider: str | None = None) -> Any:
    """Return the configured LLM instance.

    Args:
        provider: Override the LLM_PROVIDER env var. Values: "gemini" | "ollama".

    Returns:
        A LangChain chat model compatible with CrewAI agents.
    """
    resolved = provider or os.environ.get("LLM_PROVIDER", "gemini").lower()

    if resolved == "gemini":
        return _build_gemini()
    if resolved == "ollama":
        return _build_ollama()

    raise ValueError(
        f"Unknown LLM_PROVIDER '{resolved}'. Valid options: gemini, ollama"
    )


def _build_gemini() -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-untyped]

    api_key = os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to .env.local or your environment."
        )

    log.info("llm.configured", provider="gemini", model=_GEMINI_MODEL)
    return ChatGoogleGenerativeAI(
        model=_GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True,
    )


def _build_ollama() -> Any:
    from langchain_ollama import ChatOllama  # type: ignore[import-untyped]

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", _OLLAMA_DEFAULT_MODEL)

    log.info("llm.configured", provider="ollama", model=model, base_url=base_url)
    return ChatOllama(model=model, base_url=base_url, temperature=0.1)
