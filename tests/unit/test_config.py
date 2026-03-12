"""
tests/unit/test_config.py
Unit tests for src/config/settings.py — Settings validation and derived fields.
"""

from __future__ import annotations

from datetime import time

import pytest
from pydantic import ValidationError

from src.config.settings import Settings, get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    """Construct a Settings instance with env-file loading disabled."""
    defaults = dict(
        gemini_api_key="test-key",
        _env_file=None,  # skip .env.local on disk
    )
    defaults.update(overrides)
    return Settings.model_validate(defaults)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_llm_provider_default(self):
        s = make_settings()
        assert s.llm_provider == "gemini"

    def test_broker_mode_default(self):
        s = make_settings()
        assert s.broker_mode == "paper"

    def test_log_level_default(self):
        s = make_settings()
        assert s.log_level == "INFO"

    def test_redis_url_default(self):
        s = make_settings()
        assert s.redis_url == "redis://localhost:6379/0"

    def test_database_url_default(self):
        s = make_settings()
        assert "stockagent" in s.database_url

    def test_market_open_time_default(self):
        s = make_settings()
        assert s.market_open_time == time(9, 15)

    def test_market_close_time_default(self):
        s = make_settings()
        assert s.market_close_time == time(15, 30)


# ---------------------------------------------------------------------------
# Derived time fields (_build_times model_validator)
# ---------------------------------------------------------------------------

class TestDerivedTimes:
    def test_open_time_built_from_components(self):
        s = make_settings(market_open_hour=9, market_open_minute=0)
        assert s.market_open_time == time(9, 0)

    def test_close_time_built_from_components(self):
        s = make_settings(market_close_hour=15, market_close_minute=45)
        assert s.market_close_time == time(15, 45)

    def test_open_and_close_times_are_always_ordered(self):
        # open_hour is locked to 9 and close_hour to 15-16 by field validators,
        # so open < close is guaranteed; verify the extreme boundary still passes.
        s = make_settings(
            market_open_hour=9,
            market_open_minute=30,
            market_close_hour=15,
            market_close_minute=0,
        )
        assert s.market_open_time < s.market_close_time


# ---------------------------------------------------------------------------
# Field validators — market_open_hour
# ---------------------------------------------------------------------------

class TestMarketOpenHour:
    def test_valid_open_hour(self):
        s = make_settings(market_open_hour=9)
        assert s.market_open_hour == 9

    def test_invalid_open_hour_raises(self):
        with pytest.raises(ValidationError, match="market_open_hour must be 9"):
            make_settings(market_open_hour=10)


# ---------------------------------------------------------------------------
# Field validators — market_open_minute
# ---------------------------------------------------------------------------

class TestMarketOpenMinute:
    @pytest.mark.parametrize("minute", [0, 15, 30])
    def test_valid_open_minutes(self, minute: int):
        s = make_settings(market_open_minute=minute)
        assert s.market_open_minute == minute

    @pytest.mark.parametrize("minute", [-1, 31, 60])
    def test_invalid_open_minute_raises(self, minute: int):
        with pytest.raises(ValidationError, match="market_open_minute"):
            make_settings(market_open_minute=minute)


# ---------------------------------------------------------------------------
# Field validators — market_close_hour
# ---------------------------------------------------------------------------

class TestMarketCloseHour:
    @pytest.mark.parametrize("hour", [15, 16])
    def test_valid_close_hours(self, hour: int):
        s = make_settings(market_close_hour=hour)
        assert s.market_close_hour == hour

    @pytest.mark.parametrize("hour", [14, 17])
    def test_invalid_close_hour_raises(self, hour: int):
        with pytest.raises(ValidationError, match="market_close_hour"):
            make_settings(market_close_hour=hour)


# ---------------------------------------------------------------------------
# Field validators — market_close_minute
# ---------------------------------------------------------------------------

class TestMarketCloseMinute:
    @pytest.mark.parametrize("minute", [0, 30, 59])
    def test_valid_close_minutes(self, minute: int):
        s = make_settings(market_close_minute=minute)
        assert s.market_close_minute == minute

    @pytest.mark.parametrize("minute", [-1, 60])
    def test_invalid_close_minute_raises(self, minute: int):
        with pytest.raises(ValidationError, match="market_close_minute"):
            make_settings(market_close_minute=minute)


# ---------------------------------------------------------------------------
# Literal field validation
# ---------------------------------------------------------------------------

class TestLiteralFields:
    @pytest.mark.parametrize("provider", ["gemini", "ollama"])
    def test_valid_llm_providers(self, provider: str):
        s = make_settings(llm_provider=provider)
        assert s.llm_provider == provider

    def test_invalid_llm_provider_raises(self):
        with pytest.raises(ValidationError):
            make_settings(llm_provider="openai")

    @pytest.mark.parametrize("mode", ["paper", "live"])
    def test_valid_broker_modes(self, mode: str):
        s = make_settings(broker_mode=mode)
        assert s.broker_mode == mode

    def test_invalid_broker_mode_raises(self):
        with pytest.raises(ValidationError):
            make_settings(broker_mode="simulation")

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels(self, level: str):
        s = make_settings(log_level=level)
        assert s.log_level == level

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValidationError):
            make_settings(log_level="VERBOSE")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_is_live_trading_false_for_paper(self):
        s = make_settings(broker_mode="paper")
        assert s.is_live_trading is False

    def test_is_live_trading_true_for_live(self):
        s = make_settings(broker_mode="live")
        assert s.is_live_trading is True

    def test_is_gemini_true(self):
        s = make_settings(llm_provider="gemini")
        assert s.is_gemini is True

    def test_is_gemini_false_for_ollama(self):
        s = make_settings(llm_provider="ollama")
        assert s.is_gemini is False


# ---------------------------------------------------------------------------
# get_settings singleton
# ---------------------------------------------------------------------------

class TestGetSettings:
    def test_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_singleton_returns_same_object(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
