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

_CLAUDE_DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
_GEMINI_MODEL = "gemini/gemini-2.5-flash-lite"
_GROQ_DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"
_OLLAMA_DEFAULT_MODEL = "llama3.1:8b"


@lru_cache(maxsize=1)
def get_llm(provider: str | None = None) -> Any:
    """Return the configured LLM instance.

    Args:
        provider: Override the LLM_PROVIDER env var. Values: "claude" | "gemini" | "groq" | "ollama".

    Returns:
        A crewai.LLM instance compatible with CrewAI agents.
    """
    from src.config.settings import get_settings
    resolved = (provider or get_settings().llm_provider).lower()

    if resolved == "claude":
        return _build_claude()
    if resolved == "gemini":
        return _build_gemini()
    if resolved == "groq":
        return _build_groq()
    if resolved == "ollama":
        return _build_ollama()

    raise ValueError(
        f"Unknown LLM_PROVIDER '{resolved}'. Valid options: claude, gemini, groq, ollama"
    )


def _build_claude() -> Any:
    from crewai import LLM

    from src.config.settings import get_settings

    api_key = get_settings().anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to .env.local or your environment."
        )

    # pydantic_settings reads .env.local into Settings fields but does NOT write to
    # os.environ. CrewAI's internal pydantic-conversion retry calls LiteLLM directly
    # and checks os.environ for the key — ensure it's always present there.
    os.environ["ANTHROPIC_API_KEY"] = api_key

    model = os.environ.get("CLAUDE_MODEL", _CLAUDE_DEFAULT_MODEL)
    log.info("llm.configured", provider="claude", model=model)
    return LLM(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_retries=5,
        timeout=120,
    )


def _build_gemini() -> Any:
    from crewai import LLM

    from src.config.settings import get_settings

    api_key = get_settings().gemini_api_key or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to .env.local or your environment."
        )

    os.environ["GEMINI_API_KEY"] = api_key
    log.info("llm.configured", provider="gemini", model=_GEMINI_MODEL)
    return LLM(
        model=_GEMINI_MODEL,
        api_key=api_key,
        temperature=0.1,
        max_retries=5,
        timeout=120,
    )


def _build_groq() -> Any:
    from crewai import LLM

    from src.config.settings import get_settings

    api_key = get_settings().groq_api_key or os.environ.get("GROQ_API_KEY") or ""
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to .env.local or your environment."
        )

    os.environ["GROQ_API_KEY"] = api_key
    model = os.environ.get("GROQ_MODEL", _GROQ_DEFAULT_MODEL)
    log.info("llm.configured", provider="groq", model=model)
    return LLM(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_retries=5,
        timeout=120,
    )


def _build_ollama() -> Any:
    from crewai import LLM

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", _OLLAMA_DEFAULT_MODEL)

    log.info("llm.configured", provider="ollama", model=model, base_url=base_url)
    return LLM(
        model=f"ollama/{model}",
        base_url=base_url,
        temperature=0.1,
    )
