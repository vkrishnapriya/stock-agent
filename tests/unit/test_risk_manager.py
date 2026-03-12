"""
tests/unit/test_risk_manager.py
Unit tests for Risk Management Agent and its three tools.

Uses pytest-mock to isolate yfinance and NSESession calls.
Redis is bypassed via patched _get_cached / _set_cached helpers.
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(
    rows: int = 252,
    start: float = 1500.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame with a realistic random walk."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, rows)
    closes = start * np.cumprod(1 + returns)
    volumes = rng.integers(500_000, 5_000_000, rows).astype(float)
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    df = pd.DataFrame(
        {
            "Open": closes * (1 - rng.uniform(0, 0.005, rows)),
            "High": closes * (1 + rng.uniform(0, 0.01, rows)),
            "Low": closes * (1 - rng.uniform(0, 0.01, rows)),
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


def _mock_ticker(df: pd.DataFrame) -> MagicMock:
    """Return a MagicMock yfinance.Ticker whose .history() returns *df*."""
    ticker = MagicMock()
    ticker.history.return_value = df
    return ticker


def _patch_rate_limit():
    return patch(
        "src.tools.base_tool.BaseTool._check_rate_limit",
        new_callable=AsyncMock,
    )


@contextmanager
def _patch_cache():
    with (
        patch(
            "src.tools.base_tool.BaseTool._get_cached",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "src.tools.base_tool.BaseTool._set_cached",
            new_callable=AsyncMock,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# PositionSizerTool unit tests
# ---------------------------------------------------------------------------


class TestKellyFraction:
    def test_positive_edge(self):
        from src.tools.risk.position_sizer import _kelly_fraction

        # f* = 0.55 - 0.45/2 = 0.55 - 0.225 = 0.325
        assert _kelly_fraction(0.55, 2.0) == pytest.approx(0.325, abs=1e-6)

    def test_negative_edge_returns_zero(self):
        from src.tools.risk.position_sizer import _kelly_fraction

        # Win rate too low for given R:R → negative Kelly → clamped to 0
        assert _kelly_fraction(0.20, 1.0) == 0.0

    def test_fifty_fifty_equal_rr(self):
        from src.tools.risk.position_sizer import _kelly_fraction

        # f* = 0.5 - 0.5/1 = 0.0 (break-even system)
        assert _kelly_fraction(0.50, 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_high_win_rate(self):
        from src.tools.risk.position_sizer import _kelly_fraction

        fraction = _kelly_fraction(0.70, 3.0)
        assert 0.0 < fraction < 1.0


class TestPositionSizerTool:
    def test_basic_output_keys(self):
        from src.tools.risk.position_sizer import PositionSizerTool

        df = _make_price_df()
        nifty_df = _make_price_df(start=22_000.0, seed=99)

        def _ticker(sym):
            return _mock_ticker(nifty_df if sym == "^NSEI" else df)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.position_sizer.yf.Ticker", side_effect=_ticker):
                tool = PositionSizerTool()
                result = json.loads(
                    tool._run(
                        symbol="INFY",
                        entry_price=1500.0,
                        stop_loss=1430.0,
                        portfolio_value=1_000_000.0,
                    )
                )

        expected_keys = {
            "symbol", "entry_price", "stop_loss", "risk_per_share",
            "kelly_fraction", "half_kelly_value_inr", "fixed_frac_value_inr",
            "position_size_inr", "position_size_pct", "quantity",
            "lot_size", "volatility_pct", "beta", "max_drawdown_pct",
        }
        assert expected_keys.issubset(result.keys())

    def test_stop_loss_above_entry_raises(self):
        from src.tools.risk.position_sizer import PositionSizerTool

        with _patch_rate_limit(), _patch_cache():
            tool = PositionSizerTool()
            with pytest.raises(ValueError, match="stop_loss"):
                tool._run(
                    symbol="INFY",
                    entry_price=1500.0,
                    stop_loss=1600.0,   # above entry!
                    portfolio_value=1_000_000.0,
                )

    def test_position_respects_max_cap(self):
        from src.tools.risk.position_sizer import PositionSizerTool

        df = _make_price_df()
        nifty_df = _make_price_df(start=22_000.0, seed=99)

        def _ticker(sym):
            return _mock_ticker(nifty_df if sym == "^NSEI" else df)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.position_sizer.yf.Ticker", side_effect=_ticker):
                tool = PositionSizerTool()
                result = json.loads(
                    tool._run(
                        symbol="ZOMATO",    # not F&O → no lot rounding that inflates size
                        entry_price=200.0,
                        stop_loss=199.0,    # tiny stop → huge fixed-frac → must cap
                        portfolio_value=1_000_000.0,
                    )
                )

        # Position must never exceed 20% of portfolio (default max)
        assert result["position_size_pct"] <= 20.0

    def test_fno_symbol_rounds_to_lot(self):
        from src.tools.risk.position_sizer import PositionSizerTool

        df = _make_price_df()
        nifty_df = _make_price_df(start=22_000.0, seed=99)

        def _ticker(sym):
            return _mock_ticker(nifty_df if sym == "^NSEI" else df)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.position_sizer.yf.Ticker", side_effect=_ticker):
                tool = PositionSizerTool()
                result = json.loads(
                    tool._run(
                        symbol="INFY",  # lot size = 300
                        entry_price=1500.0,
                        stop_loss=1430.0,
                        portfolio_value=1_000_000.0,
                    )
                )

        # INFY lot size is 300; quantity must be a multiple
        assert result["lot_size"] == 300
        assert result["quantity"] % 300 == 0

    def test_non_fno_symbol_quantity_is_one_or_more(self):
        from src.tools.risk.position_sizer import PositionSizerTool

        df = _make_price_df()
        nifty_df = _make_price_df(start=22_000.0, seed=99)

        def _ticker(sym):
            return _mock_ticker(nifty_df if sym == "^NSEI" else df)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.position_sizer.yf.Ticker", side_effect=_ticker):
                tool = PositionSizerTool()
                result = json.loads(
                    tool._run(
                        symbol="ZOMATO",  # not in NIFTY_LOTS
                        entry_price=200.0,
                        stop_loss=185.0,
                        portfolio_value=500_000.0,
                    )
                )

        assert result["lot_size"] is None
        assert result["quantity"] >= 1

    def test_symbol_uppercased(self):
        from src.tools.risk.position_sizer import PositionSizerTool

        df = _make_price_df()
        nifty_df = _make_price_df(start=22_000.0, seed=99)

        def _ticker(sym):
            return _mock_ticker(nifty_df if sym == "^NSEI" else df)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.position_sizer.yf.Ticker", side_effect=_ticker):
                tool = PositionSizerTool()
                result = json.loads(
                    tool._run(
                        symbol="infy",
                        entry_price=1500.0,
                        stop_loss=1430.0,
                        portfolio_value=1_000_000.0,
                    )
                )

        assert result["symbol"] == "INFY"

    def test_fixed_frac_beats_kelly_when_kelly_is_large(self):
        from src.tools.risk.position_sizer import PositionSizerTool, _kelly_fraction

        df = _make_price_df()
        nifty_df = _make_price_df(start=22_000.0, seed=99)

        # Very high win-rate gives large Kelly
        win_rate = 0.9
        rr = 5.0
        kelly = _kelly_fraction(win_rate, rr)
        assert kelly > 0.5  # confirm Kelly is large

        def _ticker(sym):
            return _mock_ticker(nifty_df if sym == "^NSEI" else df)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.position_sizer.yf.Ticker", side_effect=_ticker):
                tool = PositionSizerTool()
                result = json.loads(
                    tool._run(
                        symbol="ZOMATO",
                        entry_price=200.0,
                        stop_loss=180.0,
                        portfolio_value=1_000_000.0,
                        win_rate=win_rate,
                        reward_risk_ratio=rr,
                    )
                )

        # Fixed-frac: 2% of 1M / 20 * 200 = 20,000 INR
        fixed_frac = (0.02 * 1_000_000 / 20.0) * 200.0
        assert result["position_size_inr"] <= fixed_frac + result["quantity"] * 200.0

    def test_insufficient_data_uses_defaults(self):
        from src.tools.risk.position_sizer import PositionSizerTool

        short_df = _make_price_df(rows=5)  # too few rows

        def _ticker(sym):
            return _mock_ticker(short_df)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.position_sizer.yf.Ticker", side_effect=_ticker):
                tool = PositionSizerTool()
                result = json.loads(
                    tool._run(
                        symbol="ZOMATO",
                        entry_price=200.0,
                        stop_loss=180.0,
                        portfolio_value=500_000.0,
                    )
                )

        # Falls back to defaults: vol=30, beta=1, dd=15
        assert result["volatility_pct"] == 30.0
        assert result["beta"] == 1.0
        assert result["max_drawdown_pct"] == 15.0


# ---------------------------------------------------------------------------
# CircuitBreakerTool unit tests
# ---------------------------------------------------------------------------


class TestDetectBand:
    def test_20pct_band(self):
        from src.tools.risk.circuit_checker import _detect_band

        # 20% above last price
        assert _detect_band(1200.0, 1000.0) == "20%"

    def test_10pct_band(self):
        from src.tools.risk.circuit_checker import _detect_band

        assert _detect_band(1100.0, 1000.0) == "10%"

    def test_5pct_band(self):
        from src.tools.risk.circuit_checker import _detect_band

        assert _detect_band(1050.0, 1000.0) == "5%"

    def test_2pct_band(self):
        from src.tools.risk.circuit_checker import _detect_band

        assert _detect_band(1020.0, 1000.0) == "2%"

    def test_zero_last_price_returns_20pct(self):
        from src.tools.risk.circuit_checker import _detect_band

        assert _detect_band(1200.0, 0.0) == "20%"


class TestCircuitBreakerTool:
    def _nse_response(
        self,
        last_price: float = 1500.0,
        upper_cp: float = 1800.0,
        lower_cp: float = 1200.0,
    ) -> dict:
        return {
            "priceInfo": {
                "lastPrice": last_price,
                "upperCP": upper_cp,
                "lowerCP": lower_cp,
            }
        }

    def test_returns_required_keys(self):
        from src.tools.risk.circuit_checker import CircuitBreakerTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.risk.circuit_checker.NSESession.get",
                return_value=self._nse_response(),
            ):
                tool = CircuitBreakerTool()
                result = json.loads(tool._run(symbol="RELIANCE"))

        assert {"symbol", "band", "upper_limit", "lower_limit", "last_price", "near_circuit"} == set(
            result.keys()
        )

    def test_band_detected_correctly(self):
        from src.tools.risk.circuit_checker import CircuitBreakerTool

        # upper=1800, last=1500 → (1800-1500)/1500 = 20%
        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.risk.circuit_checker.NSESession.get",
                return_value=self._nse_response(1500.0, 1800.0, 1200.0),
            ):
                tool = CircuitBreakerTool()
                result = json.loads(tool._run(symbol="RELIANCE"))

        assert result["band"] == "20%"
        assert result["near_circuit"] is False

    def test_near_upper_circuit_emits_warning(self):
        from src.tools.risk.circuit_checker import CircuitBreakerTool, CircuitBreakerWarning

        # Price within 0.5% of upper circuit
        last = 1791.0
        upper = 1800.0
        lower = 1200.0

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.risk.circuit_checker.NSESession.get",
                return_value=self._nse_response(last, upper, lower),
            ):
                tool = CircuitBreakerTool()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = json.loads(tool._run(symbol="RELIANCE"))

        assert result["near_circuit"] is True
        assert any(issubclass(x.category, CircuitBreakerWarning) for x in w)

    def test_near_lower_circuit_emits_warning(self):
        from src.tools.risk.circuit_checker import CircuitBreakerTool, CircuitBreakerWarning

        last = 1209.0
        upper = 1800.0
        lower = 1200.0  # last is 0.75% above lower → within 1%

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.risk.circuit_checker.NSESession.get",
                return_value=self._nse_response(last, upper, lower),
            ):
                tool = CircuitBreakerTool()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = json.loads(tool._run(symbol="RELIANCE"))

        assert result["near_circuit"] is True
        assert any(issubclass(x.category, CircuitBreakerWarning) for x in w)

    def test_missing_limits_raises(self):
        from src.tools.risk.circuit_checker import CircuitBreakerTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.risk.circuit_checker.NSESession.get",
                return_value={"priceInfo": {"lastPrice": 1500.0}},
            ):
                tool = CircuitBreakerTool()
                with pytest.raises(ValueError, match="circuit limits"):
                    tool._run(symbol="RELIANCE")

    def test_symbol_uppercased(self):
        from src.tools.risk.circuit_checker import CircuitBreakerTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.risk.circuit_checker.NSESession.get",
                return_value=self._nse_response(),
            ):
                tool = CircuitBreakerTool()
                result = json.loads(tool._run(symbol="reliance"))

        assert result["symbol"] == "RELIANCE"

    def test_cache_hit_skips_nse(self):
        from src.tools.risk.circuit_checker import CircuitBreakerTool

        cached_json = json.dumps(
            {
                "symbol": "RELIANCE",
                "band": "20%",
                "upper_limit": 1800.0,
                "lower_limit": 1200.0,
                "last_price": 1500.0,
                "near_circuit": False,
            }
        )

        with _patch_rate_limit():
            with (
                patch(
                    "src.tools.base_tool.BaseTool._get_cached",
                    new_callable=AsyncMock,
                    return_value=cached_json,
                ),
                patch(
                    "src.tools.base_tool.BaseTool._set_cached",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.tools.risk.circuit_checker.NSESession.get"
                ) as mock_nse,
            ):
                tool = CircuitBreakerTool()
                result = json.loads(tool._run(symbol="RELIANCE"))

        mock_nse.assert_not_called()
        assert result["band"] == "20%"


# ---------------------------------------------------------------------------
# LiquidityCheckerTool unit tests
# ---------------------------------------------------------------------------


class TestLiquidityCheckerTool:
    def test_ok_flag_for_small_trade(self):
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool

        df = _make_price_df(rows=60)  # 3 months of data

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.liquidity_checker.yf.Ticker", return_value=_mock_ticker(df)):
                tool = LiquidityCheckerTool()
                # Close is ~1500, volume ~2.75M avg → ADV ~4.1B INR
                # Trade of 100k should be well under 10%
                result = json.loads(
                    tool._run(symbol="INFY.NS", intended_trade_value_inr=100_000.0)
                )

        assert result["liquidity_flag"] == "OK"
        assert result["position_pct_of_adv"] < 10.0

    def test_warn_flag_for_medium_trade(self):
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool

        # Make a thin-volume stock: 10k shares/day at ₹100 → ADV = ₹1M
        rng = np.random.default_rng(0)
        closes = np.full(60, 100.0)
        volumes = np.full(60, 10_000.0)
        df = pd.DataFrame(
            {
                "Open": closes,
                "High": closes * 1.005,
                "Low": closes * 0.995,
                "Close": closes,
                "Volume": volumes,
            },
            index=pd.date_range("2025-01-01", periods=60, freq="B"),
        )

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.liquidity_checker.yf.Ticker", return_value=_mock_ticker(df)):
                tool = LiquidityCheckerTool()
                # ADV = 10k * 100 = 1M INR; 150k = 15% of ADV → WARN
                result = json.loads(
                    tool._run(symbol="SMALLCAP.NS", intended_trade_value_inr=150_000.0)
                )

        assert result["liquidity_flag"] == "WARN"

    def test_avoid_flag_for_large_trade(self):
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool

        closes = np.full(60, 100.0)
        volumes = np.full(60, 10_000.0)
        df = pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": volumes},
            index=pd.date_range("2025-01-01", periods=60, freq="B"),
        )

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.liquidity_checker.yf.Ticker", return_value=_mock_ticker(df)):
                tool = LiquidityCheckerTool()
                # ADV = 1M INR; 400k = 40% → AVOID
                result = json.loads(
                    tool._run(symbol="SMALLCAP.NS", intended_trade_value_inr=400_000.0)
                )

        assert result["liquidity_flag"] == "AVOID"

    def test_returns_required_keys(self):
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool

        df = _make_price_df(rows=60)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.liquidity_checker.yf.Ticker", return_value=_mock_ticker(df)):
                tool = LiquidityCheckerTool()
                result = json.loads(
                    tool._run(symbol="INFY.NS", intended_trade_value_inr=50_000.0)
                )

        assert {
            "symbol", "adv_shares", "adv_inr", "avg_close",
            "intended_trade_value_inr", "position_pct_of_adv", "liquidity_flag",
        } == set(result.keys())

    def test_insufficient_data_raises(self):
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool

        df = _make_price_df(rows=5)  # fewer than 20 bars

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.liquidity_checker.yf.Ticker", return_value=_mock_ticker(df)):
                tool = LiquidityCheckerTool()
                with pytest.raises(ValueError, match="Insufficient data"):
                    tool._run(symbol="INFY.NS", intended_trade_value_inr=100_000.0)

    def test_symbol_uppercased(self):
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool

        df = _make_price_df(rows=60)

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.liquidity_checker.yf.Ticker", return_value=_mock_ticker(df)):
                tool = LiquidityCheckerTool()
                result = json.loads(
                    tool._run(symbol="infy.ns", intended_trade_value_inr=100_000.0)
                )

        assert result["symbol"] == "INFY.NS"

    def test_adv_uses_20_day_window(self):
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool

        # First 40 bars: volume 1M; last 20: volume 2M → ADV should use only last 20
        volumes = np.concatenate([np.full(40, 1_000_000.0), np.full(20, 2_000_000.0)])
        closes = np.full(60, 100.0)
        df = pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": volumes},
            index=pd.date_range("2025-01-01", periods=60, freq="B"),
        )

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.risk.liquidity_checker.yf.Ticker", return_value=_mock_ticker(df)):
                tool = LiquidityCheckerTool()
                result = json.loads(
                    tool._run(symbol="TEST.NS", intended_trade_value_inr=50_000.0)
                )

        # ADV should be ~2M shares * 100 = 200M INR
        assert result["adv_shares"] == pytest.approx(2_000_000.0, rel=0.01)


# ---------------------------------------------------------------------------
# RiskManagementAgent unit tests
# ---------------------------------------------------------------------------


class TestRiskManagementAgent:
    def test_build_returns_crewai_agent(self):
        from src.agents.risk_manager import RiskManagementAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = RiskManagementAgent()
            crewai_agent = agent.build()

        from crewai import Agent
        assert isinstance(crewai_agent, Agent)

    def test_agent_has_three_tools(self):
        from src.agents.risk_manager import RiskManagementAgent

        agent = RiskManagementAgent()
        assert len(agent.tools) == 3

    def test_tool_types(self):
        from src.agents.risk_manager import RiskManagementAgent
        from src.tools.risk.circuit_checker import CircuitBreakerTool
        from src.tools.risk.liquidity_checker import LiquidityCheckerTool
        from src.tools.risk.position_sizer import PositionSizerTool

        agent = RiskManagementAgent()
        tool_types = {type(t) for t in agent.tools}
        assert tool_types == {PositionSizerTool, CircuitBreakerTool, LiquidityCheckerTool}

    def test_agent_role_contains_risk(self):
        from src.agents.risk_manager import RiskManagementAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = RiskManagementAgent()
            crewai_agent = agent.build()

        assert "Risk" in crewai_agent.role

    def test_agent_defaults_applied(self):
        from src.agents.risk_manager import RiskManagementAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = RiskManagementAgent()
            crewai_agent = agent.build()

        assert crewai_agent.verbose is True
        assert crewai_agent.max_iter == 5

    def test_build_task_output_pydantic(self):
        from src.agents.risk_manager import RiskManagementAgent
        from src.models.signals import RiskAssessment

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            with patch("src.agents.risk_manager.get_portfolio_state", return_value=None):
                agent = RiskManagementAgent()
                task = agent.build_task(
                    symbol="INFY",
                    entry_price=1500.0,
                    stop_loss=1430.0,
                    portfolio_value=1_000_000.0,
                )

        assert task.output_pydantic is RiskAssessment

    def test_build_task_description_contains_symbol(self):
        from src.agents.risk_manager import RiskManagementAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            with patch("src.agents.risk_manager.get_portfolio_state", return_value=None):
                agent = RiskManagementAgent()
                task = agent.build_task(
                    symbol="RELIANCE",
                    entry_price=2800.0,
                    stop_loss=2700.0,
                    portfolio_value=2_000_000.0,
                )

        assert "RELIANCE" in task.description
        # description formats with commas: "2,800.00" or raw "2800.0" in step params
        assert "2800" in task.description
        assert "2700" in task.description

    def test_build_task_injects_portfolio_context(self):
        from src.agents.risk_manager import RiskManagementAgent

        portfolio_state = {
            "holdings": [{"symbol": "TCS", "quantity": 10}],
            "available_cash_inr": 500_000.0,
        }

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            with patch(
                "src.agents.risk_manager.get_portfolio_state",
                return_value=portfolio_state,
            ):
                agent = RiskManagementAgent()
                task = agent.build_task(
                    symbol="INFY",
                    entry_price=1500.0,
                    stop_loss=1430.0,
                    portfolio_value=1_000_000.0,
                )

        assert "500,000" in task.description or "500000" in task.description
        assert "1" in task.description  # 1 holding

    def test_output_model_class_attribute(self):
        from src.agents.risk_manager import RiskManagementAgent
        from src.models.signals import RiskAssessment

        agent = RiskManagementAgent()
        assert agent.output_model is RiskAssessment

    def test_extra_tools_appended(self):
        from src.agents.risk_manager import RiskManagementAgent

        extra = MagicMock()
        agent = RiskManagementAgent(extra_tools=[extra])
        assert len(agent.tools) == 4
        assert agent.tools[-1] is extra


# ---------------------------------------------------------------------------
# PortfolioState Redis helpers
# ---------------------------------------------------------------------------


class TestPortfolioStateHelpers:
    def test_set_and_get_round_trip(self):
        from src.agents.risk_manager import get_portfolio_state, set_portfolio_state

        state = {"holdings": [], "available_cash_inr": 100_000.0}
        stored: list = []

        with patch("src.agents.risk_manager.redis.from_url") as mock_redis_factory:
            mock_client = MagicMock()
            mock_redis_factory.return_value = mock_client

            # set
            mock_client.set.side_effect = lambda key, val: stored.append((key, val))
            set_portfolio_state(state)

        assert stored  # set was called

    def test_get_returns_none_on_redis_error(self):
        from src.agents.risk_manager import get_portfolio_state

        with patch("src.agents.risk_manager.redis.from_url", side_effect=Exception("conn refused")):
            result = get_portfolio_state()

        assert result is None

    def test_set_swallows_redis_error(self):
        from src.agents.risk_manager import set_portfolio_state

        with patch("src.agents.risk_manager.redis.from_url", side_effect=Exception("conn refused")):
            # Should not raise
            set_portfolio_state({"holdings": []})
