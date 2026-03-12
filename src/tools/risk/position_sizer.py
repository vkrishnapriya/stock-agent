"""
src/tools/risk/position_sizer.py
PositionSizerTool — Kelly criterion + fixed fractional position sizing.

Computes the optimal position size for a proposed trade given:
  * Entry and stop-loss prices  → per-share risk
  * Portfolio value             → absolute risk budget
  * Win rate + reward:risk      → Kelly fraction

Sizing logic
------------
1. Half-Kelly: f* × 0.5 × portfolio_value
2. Fixed fractional: (2% of portfolio_value / risk_per_share) × entry_price
3. Final size = min(half_kelly, fixed_frac), capped at max_position_size_pct.

Also fetches 1-year daily OHLCV from yfinance to compute:
  * Annualised volatility (%)
  * Beta vs ^NSEI (Nifty 50)
  * Maximum historical drawdown (%)
"""

from __future__ import annotations

import json
import math
from typing import Any, ClassVar

import pandas as pd
import structlog
import yfinance as yf
from pydantic import BaseModel, Field

from src.config.market_hours import NIFTY_LOTS
from src.config.settings import get_settings
from src.tools.base_tool import BaseTool

log = structlog.get_logger(__name__)

_NIFTY_SYMBOL = "^NSEI"
_FIXED_RISK_PCT = 0.02   # 2 % of portfolio risked per trade
_HALF_KELLY = 0.5        # use half-Kelly to reduce ruin risk


class PositionSizerInput(BaseModel):
    """Input schema for :class:`PositionSizerTool`."""

    symbol: str = Field(description="NSE symbol without exchange suffix (e.g. 'INFY').")
    entry_price: float = Field(gt=0.0, description="Intended entry price in INR.")
    stop_loss: float = Field(gt=0.0, description="Stop-loss price in INR.")
    portfolio_value: float = Field(gt=0.0, description="Total portfolio value in INR.")
    win_rate: float = Field(
        default=0.55, ge=0.0, le=1.0, description="Historical win rate (0–1)."
    )
    reward_risk_ratio: float = Field(
        default=2.0, gt=0.0, description="Expected reward:risk ratio."
    )


def _kelly_fraction(win_rate: float, reward_risk_ratio: float) -> float:
    """Compute full Kelly fraction; returns 0.0 if edge is negative."""
    # f* = W − (1 − W) / R
    return max(0.0, win_rate - (1.0 - win_rate) / reward_risk_ratio)


def _compute_volatility_beta_drawdown(
    symbol_ns: str,
) -> tuple[float, float, float]:
    """Fetch 1-year daily OHLCV and return (annualised_vol_pct, beta, max_drawdown_pct).

    Falls back to conservative defaults if yfinance returns insufficient data.
    """
    df = yf.Ticker(symbol_ns).history(period="1y", interval="1d", auto_adjust=True)

    if df.empty or len(df) < 20:
        log.warning("position_sizer.insufficient_data", symbol=symbol_ns)
        return 30.0, 1.0, 15.0

    stock_ret = df["Close"].pct_change().dropna()

    # ── Beta vs Nifty ───────────────────────────────────────────────────────
    beta = 1.0
    try:
        nifty_df = yf.Ticker(_NIFTY_SYMBOL).history(
            period="1y", interval="1d", auto_adjust=True
        )
        nifty_ret = nifty_df["Close"].pct_change().dropna()
        aligned = pd.concat([stock_ret, nifty_ret], axis=1, join="inner")
        aligned.columns = ["stock", "nifty"]
        s_stock: pd.Series = aligned["stock"]  # type: ignore[assignment]
        s_nifty: pd.Series = aligned["nifty"]  # type: ignore[assignment]
        var_nifty = float(s_nifty.var())
        if var_nifty > 0:
            beta = float(s_stock.cov(s_nifty) / var_nifty)
    except Exception:
        pass

    # ── Annualised volatility ────────────────────────────────────────────────
    vol_pct = float(stock_ret.std() * math.sqrt(252) * 100)

    # ── Maximum drawdown ────────────────────────────────────────────────────
    close = df["Close"]
    roll_max = close.cummax()
    max_dd_pct = float(abs(((close - roll_max) / roll_max * 100).min()))

    return round(vol_pct, 2), round(beta, 4), round(max_dd_pct, 2)


class PositionSizerTool(BaseTool):
    """Compute position size using Kelly criterion with fixed-fractional fallback.

    Example::

        tool = PositionSizerTool()
        result = json.loads(tool._run(
            symbol="INFY", entry_price=1500.0, stop_loss=1430.0,
            portfolio_value=1_000_000.0,
        ))
        print(result["position_size_inr"], result["quantity"])
    """

    name: str = "position_sizer"
    description: str = (
        "Computes optimal position size using Kelly criterion and fixed-fractional "
        "risk (2% per trade). Also returns volatility_pct, beta vs Nifty, and "
        "max_drawdown_pct. F&O symbols are rounded to the nearest lot size."
    )
    args_schema: type[BaseModel] = PositionSizerInput

    rate_limit_key: ClassVar[str] = "position_sizer"
    rate_limit_per_minute: ClassVar[int] = 30

    # ------------------------------------------------------------------

    def _run(  # type: ignore[override]
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
        win_rate: float = 0.55,
        reward_risk_ratio: float = 2.0,
    ) -> str:
        symbol = symbol.strip().upper()
        log.info(
            "position_sizer.start",
            symbol=symbol,
            entry=entry_price,
            stop=stop_loss,
        )

        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            raise ValueError(
                f"stop_loss ({stop_loss}) must be strictly below "
                f"entry_price ({entry_price})"
            )

        max_pos_pct = getattr(get_settings(), "max_position_size_pct", 20.0) / 100.0

        # ── Kelly sizing ────────────────────────────────────────────────────
        kelly = _kelly_fraction(win_rate, reward_risk_ratio)
        half_kelly_value = kelly * _HALF_KELLY * portfolio_value

        # ── Fixed-fractional sizing ─────────────────────────────────────────
        max_risk_inr = _FIXED_RISK_PCT * portfolio_value
        fixed_frac_value = (max_risk_inr / risk_per_share) * entry_price

        # Conservative: take the smaller of the two
        position_size_inr = min(half_kelly_value, fixed_frac_value)

        # Cap at max_position_size_pct
        position_size_inr = min(max(position_size_inr, 0.0), max_pos_pct * portfolio_value)

        # ── Quantity: round to lot size for F&O symbols ─────────────────────
        raw_qty = int(position_size_inr / entry_price)
        lot_size = NIFTY_LOTS.get(symbol)
        if lot_size and lot_size > 1:
            quantity = max(lot_size, (raw_qty // lot_size) * lot_size)
        else:
            quantity = max(1, raw_qty)

        actual_position_inr = round(quantity * entry_price, 2)

        # ── Risk metrics from yfinance (cached 1 h) ─────────────────────────
        symbol_ns = f"{symbol}.NS"
        cached = self._run_sync_cache_get(f"vbd:{symbol_ns}")
        if cached:
            vbd = cached if isinstance(cached, dict) else json.loads(cached)
            vol_pct = vbd["vol"]
            beta = vbd["beta"]
            max_dd_pct = vbd["dd"]
        else:
            vol_pct, beta, max_dd_pct = _compute_volatility_beta_drawdown(symbol_ns)
            self._run_sync_cache_set(
                f"vbd:{symbol_ns}",
                {"vol": vol_pct, "beta": beta, "dd": max_dd_pct},
                3_600,
            )

        result = {
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_per_share": round(risk_per_share, 2),
            "kelly_fraction": round(kelly, 4),
            "half_kelly_value_inr": round(half_kelly_value, 2),
            "fixed_frac_value_inr": round(fixed_frac_value, 2),
            "position_size_inr": actual_position_inr,
            "position_size_pct": round(actual_position_inr / portfolio_value * 100, 2),
            "quantity": quantity,
            "lot_size": lot_size,
            "volatility_pct": vol_pct,
            "beta": beta,
            "max_drawdown_pct": max_dd_pct,
        }

        log.info(
            "position_sizer.done",
            symbol=symbol,
            position_inr=actual_position_inr,
            quantity=quantity,
        )
        return json.dumps(result)

    # ------------------------------------------------------------------
    # Sync cache bridge (same pattern as OHLCVFetchTool)
    # ------------------------------------------------------------------

    def _run_sync_cache_get(self, key: str) -> Any | None:
        import asyncio
        import concurrent.futures

        async def _get() -> Any:
            return await self._get_cached(key)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _get()).result()
        except RuntimeError:
            return asyncio.run(_get())

    def _run_sync_cache_set(self, key: str, value: Any, ttl: int) -> None:
        import asyncio
        import concurrent.futures

        async def _set() -> None:
            await self._set_cached(key, value, ttl)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, _set()).result()
        except RuntimeError:
            asyncio.run(_set())
