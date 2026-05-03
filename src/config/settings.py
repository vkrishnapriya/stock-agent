"""
src/config/settings.py
Application settings loaded from environment / .env.local via Pydantic BaseSettings.
Always import the singleton via: from src.config.settings import get_settings
"""

from __future__ import annotations

from datetime import time
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    llm_provider: Literal["claude", "gemini", "groq", "ollama"] = "claude"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # ── Broker ────────────────────────────────────────────────────────────────
    broker_mode: Literal["paper", "live"] = "paper"
    kite_api_key: str = ""
    kite_secret: str = ""
    kite_access_token: str = ""
    upstox_api_key: str = ""
    upstox_secret: str = ""

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "groq/llama-3.3-70b-versatile"

    # ── News / Data APIs ──────────────────────────────────────────────────────
    tavily_api_key: str = ""
    gnews_api_key: str = ""

    # ── Infrastructure ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/stockagent"
    )

    # ── App ───────────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Risk management ───────────────────────────────────────────────────────
    max_position_size_pct: float = 20.0  # max single position as % of portfolio

    # ── Market hours (IST) — override only for testing ────────────────────────
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30

    # ── Derived (populated by model_validator) ────────────────────────────────
    market_open_time: time = time(9, 15)
    market_close_time: time = time(15, 30)

    @field_validator("market_open_hour")
    @classmethod
    def _validate_open_hour(cls, v: int) -> int:
        if v not in (9,):
            raise ValueError("market_open_hour must be 9 for IST NSE session")
        return v

    @field_validator("market_open_minute")
    @classmethod
    def _validate_open_minute(cls, v: int) -> int:
        if not (0 <= v <= 30):
            raise ValueError(
                "market_open_minute must be 0–30 (NSE opens between 09:00–09:30 IST)"
            )
        return v

    @field_validator("market_close_hour")
    @classmethod
    def _validate_close_hour(cls, v: int) -> int:
        if not (15 <= v <= 16):
            raise ValueError(
                "market_close_hour must be 15 or 16 (NSE closes between 15:00–16:00 IST)"
            )
        return v

    @field_validator("market_close_minute")
    @classmethod
    def _validate_close_minute(cls, v: int) -> int:
        if not (0 <= v <= 59):
            raise ValueError("market_close_minute must be 0–59")
        return v

    @model_validator(mode="after")
    def _build_times(self) -> "Settings":
        self.market_open_time = time(self.market_open_hour, self.market_open_minute)
        self.market_close_time = time(self.market_close_hour, self.market_close_minute)
        if self.market_open_time >= self.market_close_time:
            raise ValueError(
                f"market_open_time ({self.market_open_time}) must be "
                f"before market_close_time ({self.market_close_time})"
            )
        return self

    @property
    def is_live_trading(self) -> bool:
        return self.broker_mode == "live"

    @property
    def is_gemini(self) -> bool:
        return self.llm_provider == "gemini"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
