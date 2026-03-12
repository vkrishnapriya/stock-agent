"""
tests/unit/test_technical_analyst.py
Unit tests for technical analysis tools and agent.

No live network calls are made — yfinance.Ticker is mocked throughout.
Rate limiting and Redis cache calls are patched to no-ops.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.models.signals import TechnicalSignal


# ---------------------------------------------------------------------------
# OHLCV fixture helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(
    rows: int = 252,
    start_price: float = 1800.0,
    trend: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic daily OHLCV DataFrame.

    Args:
        rows:        Number of bars (252 ≈ one trading year).
        start_price: Starting close price in INR.
        trend:       Daily drift percentage (0.3 = mild uptrend).
        seed:        Random seed for reproducibility.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    closes = [start_price]
    for _ in range(rows - 1):
        pct = trend / 100 + rng.normal(0, 1.0) / 100
        closes.append(closes[-1] * (1 + pct))

    closes = pd.array(closes, dtype=float)
    highs  = closes * (1 + rng.uniform(0.001, 0.015, rows))
    lows   = closes * (1 - rng.uniform(0.001, 0.015, rows))
    opens  = closes * (1 + rng.normal(0, 0.005, rows))
    vols   = rng.integers(500_000, 5_000_000, rows).astype(float)

    index = pd.bdate_range("2025-01-01", periods=rows)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=index,
    )


def _make_ohlcv_json(rows: int = 252, **kwargs) -> str:
    """Return the OHLCV DataFrame as a JSON records string (daily format)."""
    df = _make_ohlcv_df(rows, **kwargs)
    df.index.name = "Date"
    return df.reset_index().assign(Date=lambda d: d["Date"].astype(str)).to_json(orient="records")


def _make_intraday_json(rows: int = 80) -> str:
    """Return an intraday OHLCV JSON string (5-minute bars)."""
    df = _make_ohlcv_df(rows, start_price=2000.0, seed=99)
    df.index = pd.date_range("2025-06-01 09:15", periods=rows, freq="5min")
    df.index.name = "Datetime"
    return df.reset_index().assign(Datetime=lambda d: d["Datetime"].astype(str)).to_json(orient="records")


def _mock_ticker(df: pd.DataFrame) -> MagicMock:
    """Return a MagicMock yfinance.Ticker whose history() returns *df*."""
    ticker = MagicMock()
    ticker.history.return_value = df
    return ticker


# ---------------------------------------------------------------------------
# Shared patches
# ---------------------------------------------------------------------------

def _patch_rate_limit():
    """Patch BaseTool._check_rate_limit to a no-op coroutine."""
    return patch(
        "src.tools.base_tool.BaseTool._check_rate_limit",
        new_callable=AsyncMock,
    )


@contextmanager
def _patch_cache():
    """Context manager that patches BaseTool cache helpers to no-ops."""
    with patch(
        "src.tools.base_tool.BaseTool._get_cached",
        new_callable=AsyncMock,
        return_value=None,
    ), patch(
        "src.tools.base_tool.BaseTool._set_cached",
        new_callable=AsyncMock,
    ):
        yield


# ===========================================================================
# OHLCVFetchTool
# ===========================================================================


class TestOHLCVFetchTool:
    def setup_method(self):
        from src.tools.market.yfinance_tools import OHLCVFetchTool
        self.tool = OHLCVFetchTool()

    def _call_run(self, symbol="INFY.NS", period="1y", interval="1d"):
        df = _make_ohlcv_df()
        with patch("yfinance.Ticker", return_value=_mock_ticker(df)), \
             _patch_rate_limit(), \
             _patch_cache():
            return self.tool._run(symbol=symbol, period=period, interval=interval)

    def test_returns_json_string(self):
        result = self._call_run()
        assert isinstance(result, str)
        records = json.loads(result)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_json_has_required_columns(self):
        result = self._call_run()
        record = json.loads(result)[0]
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in record, f"Missing column: {col}"

    def test_daily_interval_uses_date_key(self):
        result = self._call_run(interval="1d")
        record = json.loads(result)[0]
        assert "Date" in record

    def test_intraday_interval_uses_datetime_key(self):
        df = _make_ohlcv_df()
        # Simulate 5-minute bars by giving DataFrame a datetime index
        df.index = pd.date_range("2025-06-01 09:15", periods=len(df), freq="5min")
        with patch("yfinance.Ticker", return_value=_mock_ticker(df)), \
             _patch_rate_limit(), \
             _patch_cache():
            result = self.tool._run(symbol="INFY.NS", period="1d", interval="5m")
        record = json.loads(result)[0]
        assert "Datetime" in record

    def test_empty_dataframe_raises_value_error(self):
        empty_ticker = MagicMock()
        empty_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=empty_ticker), \
             _patch_rate_limit(), \
             _patch_cache():
            with pytest.raises(ValueError, match="No OHLCV data"):
                self.tool._run(symbol="INVALID.NS")

    def test_cache_hit_skips_yfinance(self):
        cached_json = _make_ohlcv_json(rows=10)
        get_patch = patch(
            "src.tools.base_tool.BaseTool._get_cached",
            new_callable=AsyncMock,
            return_value=cached_json,
        )
        with patch("yfinance.Ticker") as mock_yf, \
             _patch_rate_limit(), \
             get_patch:
            result = self.tool._run(symbol="INFY.NS")
        mock_yf.assert_not_called()
        assert json.loads(result)  # valid JSON

    def test_symbol_uppercased(self):
        df = _make_ohlcv_df()
        with patch("yfinance.Ticker") as mock_yf, \
             _patch_rate_limit(), \
             _patch_cache():
            mock_yf.return_value = _mock_ticker(df)
            self.tool._run(symbol="infy.ns")
        call_args = mock_yf.call_args[0][0]
        assert call_args == "INFY.NS"


# ===========================================================================
# IndicatorEngineTool
# ===========================================================================


class TestIndicatorEngineTool:
    def setup_method(self):
        from src.tools.technical.indicator_engine import IndicatorEngineTool
        self.tool = IndicatorEngineTool()

    def _call_run(self, ohlcv_json: str | None = None, symbol: str = "INFY") -> str:
        ohlcv_json = ohlcv_json or _make_ohlcv_json(rows=252)
        with _patch_rate_limit(), _patch_cache():
            return self.tool._run(ohlcv_json=ohlcv_json, symbol=symbol)

    def test_returns_valid_technical_signal(self):
        result = self._call_run()
        signal = TechnicalSignal.model_validate_json(result)
        assert signal.symbol == "INFY"

    def test_rsi_within_bounds(self):
        result = self._call_run()
        signal = TechnicalSignal.model_validate_json(result)
        assert 0.0 <= signal.rsi <= 100.0

    def test_score_within_bounds(self):
        result = self._call_run()
        signal = TechnicalSignal.model_validate_json(result)
        assert -100.0 <= signal.score <= 100.0

    def test_trend_consistent_with_score(self):
        result = self._call_run()
        signal = TechnicalSignal.model_validate_json(result)
        if signal.score > 20:
            assert signal.trend == "BULLISH"
        elif signal.score < -20:
            assert signal.trend == "BEARISH"
        else:
            assert signal.trend == "NEUTRAL"

    def test_support_below_resistance(self):
        result = self._call_run()
        signal = TechnicalSignal.model_validate_json(result)
        assert signal.support_level_inr < signal.resistance_level_inr

    def test_support_and_resistance_positive(self):
        result = self._call_run()
        signal = TechnicalSignal.model_validate_json(result)
        assert signal.support_level_inr > 0
        assert signal.resistance_level_inr > 0

    def test_volume_signal_valid(self):
        result = self._call_run()
        signal = TechnicalSignal.model_validate_json(result)
        assert signal.volume_signal in ("HIGH", "LOW", "NORMAL")

    def test_intraday_ohlcv_produces_signal(self):
        intraday_json = _make_intraday_json(rows=80)
        result = self._call_run(ohlcv_json=intraday_json, symbol="RELIANCE")
        signal = TechnicalSignal.model_validate_json(result)
        assert signal.symbol == "RELIANCE"

    def test_short_series_still_produces_signal(self):
        """SMA200 will be NaN for <200 bars — tool should handle gracefully."""
        result = self._call_run(ohlcv_json=_make_ohlcv_json(rows=60), symbol="TCS")
        signal = TechnicalSignal.model_validate_json(result)
        assert -100 <= signal.score <= 100

    def test_symbol_uppercased_in_output(self):
        result = self._call_run(symbol="wipro")
        signal = TechnicalSignal.model_validate_json(result)
        assert signal.symbol == "WIPRO"

    def test_missing_ohlcv_column_raises(self):
        bad_df = _make_ohlcv_df(rows=60)
        bad_df = bad_df.drop(columns=["Volume"])
        bad_df.index.name = "Date"
        bad_json = bad_df.reset_index().assign(Date=lambda d: d["Date"].astype(str)).to_json(orient="records")
        with _patch_rate_limit(), _patch_cache():
            with pytest.raises(ValueError, match="missing columns"):
                self.tool._run(ohlcv_json=bad_json, symbol="INFY")

    @pytest.mark.parametrize("rows", [15, 30, 60, 252])
    def test_various_series_lengths(self, rows: int):
        result = self._call_run(ohlcv_json=_make_ohlcv_json(rows=rows))
        signal = TechnicalSignal.model_validate_json(result)
        assert -100 <= signal.score <= 100


# ===========================================================================
# Composite score helper (unit-test the math)
# ===========================================================================


class TestCompositeScore:
    def test_all_bullish_inputs_give_positive_score(self):
        from src.tools.technical.indicator_engine import _compute_composite_score

        score = _compute_composite_score(
            rsi=75.0,
            macd_hist=5.0,
            close=100.0,
            sma50=90.0,
            sma200=80.0,
            bb_upper=110.0,
            bb_lower=90.0,
        )
        assert score > 0

    def test_all_bearish_inputs_give_negative_score(self):
        from src.tools.technical.indicator_engine import _compute_composite_score

        score = _compute_composite_score(
            rsi=25.0,
            macd_hist=-5.0,
            close=100.0,
            sma50=110.0,
            sma200=120.0,
            bb_upper=115.0,
            bb_lower=95.0,
        )
        assert score < 0

    def test_price_at_sma_scores_bearish(self):
        from src.tools.technical.indicator_engine import _compute_composite_score

        # Price exactly at SMA50 and SMA200 is NOT strictly above → bearish trend component.
        # RSI=50 → 0, MACD=0 → 0, mid-BB → 0, trend contribution = -25 (both MAs bearish).
        score = _compute_composite_score(
            rsi=50.0,
            macd_hist=0.0,
            close=100.0,
            sma50=100.0,
            sma200=100.0,
            bb_upper=105.0,
            bb_lower=95.0,
        )
        assert score == pytest.approx(-25.0, abs=1.0)

    def test_score_clamped_to_100(self):
        from src.tools.technical.indicator_engine import _compute_composite_score

        score = _compute_composite_score(
            rsi=100.0, macd_hist=1000.0, close=100.0,
            sma50=1.0, sma200=1.0, bb_upper=101.0, bb_lower=99.0,
        )
        assert score <= 100.0

    def test_score_clamped_to_minus_100(self):
        from src.tools.technical.indicator_engine import _compute_composite_score

        score = _compute_composite_score(
            rsi=0.0, macd_hist=-1000.0, close=100.0,
            sma50=999.0, sma200=999.0, bb_upper=101.0, bb_lower=99.0,
        )
        assert score >= -100.0


# ===========================================================================
# Pivot point helpers
# ===========================================================================


class TestPivotCalculations:
    def test_classic_pivot_formula(self):
        from src.tools.technical.support_resistance import _classic_pivots

        pivots = _classic_pivots(prev_high=2100.0, prev_low=1900.0, prev_close=2000.0)
        expected_P = (2100 + 1900 + 2000) / 3
        assert pivots["P"] == pytest.approx(expected_P, rel=1e-4)
        assert pivots["R1"] == pytest.approx(2 * expected_P - 1900, rel=1e-4)
        assert pivots["S1"] == pytest.approx(2 * expected_P - 2100, rel=1e-4)

    def test_camarilla_pivot_formula(self):
        from src.tools.technical.support_resistance import _camarilla_pivots

        pivots = _camarilla_pivots(prev_high=2100.0, prev_low=1900.0, prev_close=2000.0)
        rng = 2100 - 1900
        assert pivots["R3"] == pytest.approx(2000 + rng * 1.1 / 4, rel=1e-4)
        assert pivots["S3"] == pytest.approx(2000 - rng * 1.1 / 4, rel=1e-4)
        assert pivots["R4"] > pivots["R3"]
        assert pivots["S4"] < pivots["S3"]

    def test_nearest_levels_returns_closest(self):
        from src.tools.technical.support_resistance import _nearest_levels

        support, resistance = _nearest_levels(
            close=100.0,
            candidates=[80.0, 90.0, 95.0, 105.0, 110.0, 120.0],
        )
        assert support == 95.0
        assert resistance == 105.0

    def test_nearest_levels_handles_all_above(self):
        from src.tools.technical.support_resistance import _nearest_levels

        support, resistance = _nearest_levels(
            close=50.0,
            candidates=[80.0, 90.0, 100.0],
        )
        assert support < resistance  # safety guard applied


# ===========================================================================
# SupportResistanceTool
# ===========================================================================


class TestSupportResistanceTool:
    def setup_method(self):
        from src.tools.technical.support_resistance import SupportResistanceTool
        self.tool = SupportResistanceTool()

    def _call_run(self, symbol: str = "INFY.NS") -> dict:
        df = _make_ohlcv_df(rows=252)
        with patch("yfinance.Ticker", return_value=_mock_ticker(df)), \
             _patch_rate_limit(), \
             _patch_cache():
            raw = self.tool._run(symbol=symbol)
        return json.loads(raw)

    def test_returns_json_with_required_keys(self):
        result = self._call_run()
        for key in ("symbol", "classic_pivot", "camarilla_pivot",
                    "week52_high", "week52_low", "nearest_support", "nearest_resistance"):
            assert key in result, f"Missing key: {key}"

    def test_classic_pivot_has_seven_levels(self):
        result = self._call_run()
        cp = result["classic_pivot"]
        for k in ("P", "R1", "R2", "R3", "S1", "S2", "S3"):
            assert k in cp

    def test_camarilla_pivot_has_four_levels(self):
        result = self._call_run()
        cam = result["camarilla_pivot"]
        for k in ("R3", "R4", "S3", "S4"):
            assert k in cam

    def test_52w_high_above_low(self):
        result = self._call_run()
        assert result["week52_high"] > result["week52_low"]

    def test_nearest_support_below_resistance(self):
        result = self._call_run()
        assert result["nearest_support"] < result["nearest_resistance"]

    def test_symbol_uppercased(self):
        result = self._call_run(symbol="infy.ns")
        assert result["symbol"] == "INFY.NS"

    def test_empty_dataframe_raises(self):
        empty_ticker = MagicMock()
        empty_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=empty_ticker), \
             _patch_rate_limit(), \
             _patch_cache():
            with pytest.raises(ValueError, match="No price data"):
                self.tool._run(symbol="FAKE.NS")

    def test_cache_hit_skips_yfinance(self):
        cached = json.dumps({
            "symbol": "INFY.NS",
            "classic_pivot": {}, "camarilla_pivot": {},
            "week52_high": 2100.0, "week52_low": 1500.0,
            "sma_50": 1850.0, "sma_200": 1750.0,
            "nearest_support": 1800.0, "nearest_resistance": 1900.0,
            "last_close": 1850.0,
        })
        get_patch = patch(
            "src.tools.base_tool.BaseTool._get_cached",
            new_callable=AsyncMock,
            return_value=cached,
        )
        with patch("yfinance.Ticker") as mock_yf, _patch_rate_limit(), get_patch:
            result = json.loads(self.tool._run(symbol="INFY.NS"))
        mock_yf.assert_not_called()
        assert result["week52_high"] == 2100.0


# ===========================================================================
# TechnicalAnalysisAgent
# ===========================================================================


class TestTechnicalAnalysisAgent:
    def test_build_returns_crewai_agent(self):
        from crewai import Agent
        from src.agents.technical_analyst import TechnicalAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent_obj = TechnicalAnalysisAgent()
            agent = agent_obj.build()
        assert isinstance(agent, Agent)

    def test_agent_has_three_tools(self):
        from src.agents.technical_analyst import TechnicalAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent_obj = TechnicalAnalysisAgent()
        assert len(agent_obj.tools) == 3

    def test_tool_types(self):
        from src.agents.technical_analyst import TechnicalAnalysisAgent
        from src.tools.market.yfinance_tools import OHLCVFetchTool
        from src.tools.technical.indicator_engine import IndicatorEngineTool
        from src.tools.technical.support_resistance import SupportResistanceTool

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent_obj = TechnicalAnalysisAgent()
        tool_types = {type(t) for t in agent_obj.tools}
        assert OHLCVFetchTool in tool_types
        assert IndicatorEngineTool in tool_types
        assert SupportResistanceTool in tool_types

    def test_agent_role_and_goal(self):
        from src.agents.technical_analyst import TechnicalAnalysisAgent, _GOAL, _ROLE

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent_obj = TechnicalAnalysisAgent()
            agent = agent_obj.build()
        assert agent.role == _ROLE
        assert agent.goal == _GOAL

    def test_agent_defaults_applied(self):
        from src.agents.technical_analyst import TechnicalAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent_obj = TechnicalAnalysisAgent()
            agent = agent_obj.build()
        assert agent.verbose is True
        assert agent.max_iter == 5

    def test_build_task_output_pydantic(self):
        from src.agents.technical_analyst import TechnicalAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent_obj = TechnicalAnalysisAgent()
            task = agent_obj.build_task(symbol="INFY")
        assert task.output_pydantic is TechnicalSignal

    def test_build_task_description_contains_symbol(self):
        from src.agents.technical_analyst import TechnicalAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent_obj = TechnicalAnalysisAgent()
            task = agent_obj.build_task(symbol="WIPRO", period="6mo")
        assert "WIPRO" in task.description
        assert "6mo" in task.description

    def test_output_model_class_attribute(self):
        from src.agents.technical_analyst import TechnicalAnalysisAgent

        assert TechnicalAnalysisAgent.output_model is TechnicalSignal

    def test_extra_tools_appended(self):
        from src.agents.technical_analyst import TechnicalAnalysisAgent

        extra = MagicMock()
        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent_obj = TechnicalAnalysisAgent(extra_tools=[extra])
        assert len(agent_obj.tools) == 4
        assert agent_obj.tools[-1] is extra
