"""
src/tools/technical/indicator_engine.py
IndicatorEngineTool — computes technical indicators from OHLCV data.

Accepts the JSON output of OHLCVFetchTool, computes the following indicators
using the `ta` library, and returns a :class:`~src.models.signals.TechnicalSignal`
serialised as JSON.

Indicators computed
-------------------
* RSI(14)
* MACD(12, 26, 9) — histogram value reported as ``macd_signal``
* Bollinger Bands(20, 2) — upper / lower bands
* VWAP — only for intraday data (interval carries time-of-day info)
* OBV — used to determine volume signal
* SMA(50), SMA(200)

Composite score  (-100 → +100)
-------------------------------
+-------+---------+---------------------------------------------+
| RSI   |   35%   | (rsi − 50) × 2 → -100..+100                |
| MACD  |   30%   | histogram normalised by 1% of close price  |
| Trend |   25%   | price vs SMA50 + SMA200, averaged           |
| BB    |   10%   | price position within Bollinger Band        |
+-------+---------+---------------------------------------------+
"""

from __future__ import annotations

import json
from typing import ClassVar

import pandas as pd
import structlog
import ta
from pydantic import BaseModel, Field

from src.models.signals import TechnicalSignal
from src.tools.base_tool import BaseTool

log = structlog.get_logger(__name__)

_INTRADAY_KEYS = {"Datetime", "datetime"}


class IndicatorInput(BaseModel):
    """Input schema for :class:`IndicatorEngineTool`."""

    ohlcv_json: str = Field(
        description="JSON array of OHLCV records as returned by OHLCVFetchTool."
    )
    symbol: str = Field(
        description="Clean NSE symbol without exchange suffix (e.g. 'INFY', 'RELIANCE')."
    )


def _safe_float(series: pd.Series, idx: int = -1) -> float | None:
    """Return a float from the series at *idx*, or None if NaN / empty."""
    try:
        val = series.iloc[idx]
        return None if pd.isna(val) else float(val)
    except (IndexError, TypeError):
        return None


def _compute_composite_score(
    rsi: float,
    macd_hist: float,
    close: float,
    sma50: float | None,
    sma200: float | None,
    bb_upper: float,
    bb_lower: float,
) -> float:
    """Weighted composite technical score in [-100, +100]."""
    # 1. RSI component (35 %)
    rsi_comp = (rsi - 50.0) * 2.0  # -100..+100

    # 2. MACD histogram (30 %): normalise by 1 % of price
    price_unit = close * 0.01
    macd_comp = max(-100.0, min(100.0, (macd_hist / price_unit) * 10.0)) if price_unit else 0.0

    # 3. Trend vs SMAs (25 %): average contribution from each available MA
    trend_parts: list[float] = []
    if sma50 is not None:
        trend_parts.append(100.0 if close > sma50 else -100.0)
    if sma200 is not None:
        trend_parts.append(100.0 if close > sma200 else -100.0)
    trend_comp = sum(trend_parts) / len(trend_parts) if trend_parts else 0.0

    # 4. Bollinger Band position (10 %): centre = 0, top = +100, bottom = -100
    bb_range = bb_upper - bb_lower
    bb_comp = ((close - bb_lower) / bb_range - 0.5) * 200.0 if bb_range > 0 else 0.0

    raw = rsi_comp * 0.35 + macd_comp * 0.30 + trend_comp * 0.25 + bb_comp * 0.10
    return max(-100.0, min(100.0, round(raw, 2)))


def _support_resistance(
    close: float,
    sma50: float | None,
    sma200: float | None,
    bb_upper: float,
    bb_lower: float,
) -> tuple[float, float]:
    """Derive nearest support and resistance from indicator levels."""
    below = [v for v in [sma50, sma200, bb_lower] if v is not None and v < close]
    above = [v for v in [sma50, sma200, bb_upper] if v is not None and v > close]

    support = max(below) if below else bb_lower
    resistance = min(above) if above else bb_upper

    # Ensure support < resistance (safety guard)
    if support >= resistance:
        support = bb_lower
        resistance = bb_upper if bb_upper > support else support * 1.05

    return round(support, 2), round(resistance, 2)


class IndicatorEngineTool(BaseTool):
    """Compute technical indicators from OHLCV JSON and return a TechnicalSignal.

    Example::

        tool = IndicatorEngineTool()
        signal_json = tool._run(ohlcv_json=ohlcv_str, symbol="INFY")
        signal = TechnicalSignal.model_validate_json(signal_json)
    """

    name: str = "indicator_engine"
    description: str = (
        "Computes RSI, MACD, Bollinger Bands, VWAP (intraday), OBV, SMA(50/200) "
        "from raw OHLCV JSON produced by ohlcv_fetch, then returns a composite "
        "TechnicalSignal (score -100..+100) as JSON."
    )
    args_schema: type[BaseModel] = IndicatorInput

    rate_limit_key: ClassVar[str] = "indicator_engine"
    rate_limit_per_minute: ClassVar[int] = 120  # CPU-bound; generous limit

    # ------------------------------------------------------------------

    def _run(self, ohlcv_json: str, symbol: str) -> str:  # type: ignore[override]
        symbol = symbol.strip().upper()
        log.info("indicator_engine.start", symbol=symbol)

        # ── Parse OHLCV ────────────────────────────────────────────────
        records = json.loads(ohlcv_json)
        df = pd.DataFrame(records)

        is_intraday = any(k in df.columns for k in _INTRADAY_KEYS)
        dt_col = "Datetime" if "Datetime" in df.columns else "Date"
        if dt_col in df.columns:
            df.index = pd.to_datetime(df[dt_col])
            df = df.drop(columns=[dt_col])

        # Normalise column names
        df.columns = [c.title() for c in df.columns]
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV data missing columns: {missing}")

        df = df.sort_index()
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # ── Indicators ─────────────────────────────────────────────────
        rsi_series  = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd_ind    = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        bb_ind      = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        obv_series  = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        sma50_series  = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        sma200_series = ta.trend.SMAIndicator(close, window=200).sma_indicator()

        # Latest values
        rsi_val    = _safe_float(rsi_series) or 50.0
        macd_hist  = _safe_float(macd_ind.macd_diff()) or 0.0
        bb_upper   = _safe_float(bb_ind.bollinger_hband()) or float(close.iloc[-1]) * 1.02
        bb_lower   = _safe_float(bb_ind.bollinger_lband()) or float(close.iloc[-1]) * 0.98
        obv_now    = _safe_float(obv_series) or 0.0
        obv_prev   = _safe_float(obv_series, -6) or obv_now
        sma50_val  = _safe_float(sma50_series)
        sma200_val = _safe_float(sma200_series)
        close_val  = float(close.iloc[-1])

        # VWAP (intraday only)
        vwap_val: float | None = None
        if is_intraday and len(df) >= 14:
            try:
                vwap_series = ta.volume.VolumeWeightedAveragePrice(
                    high, low, close, vol
                ).volume_weighted_average_price()
                vwap_val = _safe_float(vwap_series)
            except Exception:
                pass  # VWAP needs at least 14 bars; silently skip

        # ── Derived signals ────────────────────────────────────────────
        score = _compute_composite_score(
            rsi=rsi_val,
            macd_hist=macd_hist,
            close=close_val,
            sma50=sma50_val,
            sma200=sma200_val,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
        )

        trend: str
        if score > 20:
            trend = "BULLISH"
        elif score < -20:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # Volume signal: OBV 5-bar momentum
        volume_signal: str
        if obv_prev and obv_prev != 0:
            obv_change_pct = (obv_now - obv_prev) / abs(obv_prev) * 100
            if obv_change_pct > 3:
                volume_signal = "HIGH"
            elif obv_change_pct < -3:
                volume_signal = "LOW"
            else:
                volume_signal = "NORMAL"
        else:
            volume_signal = "NORMAL"

        support, resistance = _support_resistance(
            close=close_val,
            sma50=sma50_val,
            sma200=sma200_val,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
        )

        signal = TechnicalSignal(
            symbol=symbol,
            score=score,
            trend=trend,  # type: ignore[arg-type]
            rsi=round(rsi_val, 2),
            macd_signal=round(macd_hist, 4),
            support_level_inr=support,
            resistance_level_inr=resistance,
            volume_signal=volume_signal,  # type: ignore[arg-type]
        )

        log.info(
            "indicator_engine.done",
            symbol=symbol,
            score=score,
            trend=trend,
            rsi=rsi_val,
        )
        return signal.model_dump_json()
