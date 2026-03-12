"""
src/tools/technical/support_resistance.py
SupportResistanceTool — pivot points, 52-week range, and key MA levels.

Calculates:
  * Classic pivot points  (P, R1–R3, S1–S3)
  * Camarilla pivot points (R3, R4, S3, S4)
  * 52-week high / low via yfinance
  * SMA(50) and SMA(200) from 1-year daily OHLCV
  * Nearest support and resistance from all computed levels

Returns a JSON object with all levels.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pandas as pd
import structlog
import ta
import yfinance as yf
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool

log = structlog.get_logger(__name__)

_CACHE_TTL = 3_600  # 1 hour — pivot points change once per day


class SRInput(BaseModel):
    """Input schema for :class:`SupportResistanceTool`."""

    symbol: str = Field(
        description=(
            "Ticker with exchange suffix: 'INFY.NS' (NSE) or 'INFY.BO' (BSE). "
            "Pivot points use the previous session's OHLC from Yahoo Finance."
        )
    )


def _classic_pivots(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    """Calculate Classic pivot points for the current session."""
    P  = (prev_high + prev_low + prev_close) / 3
    R1 = 2 * P - prev_low
    S1 = 2 * P - prev_high
    R2 = P + (prev_high - prev_low)
    S2 = P - (prev_high - prev_low)
    R3 = prev_high + 2 * (P - prev_low)
    S3 = prev_low  - 2 * (prev_high - P)
    return {
        "P":  round(P,  2),
        "R1": round(R1, 2), "R2": round(R2, 2), "R3": round(R3, 2),
        "S1": round(S1, 2), "S2": round(S2, 2), "S3": round(S3, 2),
    }


def _camarilla_pivots(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    """Calculate Camarilla pivot points (R3/R4, S3/S4 most-watched)."""
    rng = prev_high - prev_low
    R3 = prev_close + rng * 1.1 / 4
    R4 = prev_close + rng * 1.1 / 2
    S3 = prev_close - rng * 1.1 / 4
    S4 = prev_close - rng * 1.1 / 2
    return {
        "R3": round(R3, 2), "R4": round(R4, 2),
        "S3": round(S3, 2), "S4": round(S4, 2),
    }


def _nearest_levels(
    close: float, candidates: list[float]
) -> tuple[float, float]:
    """Return (nearest_support, nearest_resistance) from *candidates* vs *close*."""
    below = [v for v in candidates if v < close]
    above = [v for v in candidates if v > close]
    support    = max(below) if below else min(candidates)
    resistance = min(above) if above else max(candidates)
    if support >= resistance:
        resistance = support * 1.01  # safety margin
    return round(support, 2), round(resistance, 2)


class SupportResistanceTool(BaseTool):
    """Derive support and resistance levels from pivot math and moving averages.

    Example::

        tool = SupportResistanceTool()
        result = json.loads(tool._run(symbol="INFY.NS"))
        print(result["nearest_support"], result["nearest_resistance"])
    """

    name: str = "support_resistance"
    description: str = (
        "Calculates Classic and Camarilla pivot points using the previous "
        "session's OHLC, the 52-week high/low, and SMA(50)/SMA(200) levels "
        "for a given NSE/BSE stock. Returns JSON with all key price levels."
    )
    args_schema: type[BaseModel] = SRInput

    rate_limit_key: ClassVar[str] = "yfinance"
    rate_limit_per_minute: ClassVar[int] = 60

    # ------------------------------------------------------------------

    def _run(self, symbol: str) -> str:  # type: ignore[override]
        symbol = symbol.strip().upper()

        cached = self._run_sync_cache_get(f"sr:{symbol}")
        if cached:
            log.debug("support_resistance.cache_hit", symbol=symbol)
            return cached if isinstance(cached, str) else json.dumps(cached)

        log.info("support_resistance.fetching", symbol=symbol)
        ticker = yf.Ticker(symbol)
        df: pd.DataFrame = ticker.history(period="1y", interval="1d", auto_adjust=True)

        if df.empty:
            raise ValueError(
                f"No price data returned for '{symbol}'. "
                "Ensure symbol has '.NS' or '.BO' suffix."
            )

        df.columns = [c.title() for c in df.columns]
        df = df.sort_index()

        # Previous session OHLC for pivot calculations
        prev_high  = float(df["High"].iloc[-2])
        prev_low   = float(df["Low"].iloc[-2])
        prev_close = float(df["Close"].iloc[-2])
        last_close = float(df["Close"].iloc[-1])

        # 52-week high / low
        w52_high = float(df["High"].max())
        w52_low  = float(df["Low"].min())

        # Moving averages
        close_series = df["Close"]
        sma50_series  = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()
        sma200_series = ta.trend.SMAIndicator(close_series, window=200).sma_indicator()

        sma50  = round(float(sma50_series.iloc[-1]),  2) if not pd.isna(sma50_series.iloc[-1])  else None
        sma200 = round(float(sma200_series.iloc[-1]), 2) if not pd.isna(sma200_series.iloc[-1]) else None

        # Pivot points
        classic    = _classic_pivots(prev_high, prev_low, prev_close)
        camarilla  = _camarilla_pivots(prev_high, prev_low, prev_close)

        # All candidate levels for nearest support/resistance
        all_levels: list[float] = [
            classic["S1"], classic["S2"], classic["R1"], classic["R2"],
            camarilla["S3"], camarilla["S4"], camarilla["R3"], camarilla["R4"],
        ]
        if sma50  is not None: all_levels.append(sma50)
        if sma200 is not None: all_levels.append(sma200)
        all_levels += [w52_high, w52_low]

        nearest_support, nearest_resistance = _nearest_levels(last_close, all_levels)

        result = {
            "symbol":             symbol,
            "last_close":         round(last_close, 2),
            "classic_pivot":      classic,
            "camarilla_pivot":    camarilla,
            "week52_high":        round(w52_high, 2),
            "week52_low":         round(w52_low, 2),
            "sma_50":             sma50,
            "sma_200":            sma200,
            "nearest_support":    nearest_support,
            "nearest_resistance": nearest_resistance,
        }

        result_json = json.dumps(result)
        self._run_sync_cache_set(f"sr:{symbol}", result_json, _CACHE_TTL)
        log.info(
            "support_resistance.done",
            symbol=symbol,
            support=nearest_support,
            resistance=nearest_resistance,
        )
        return result_json

    # ------------------------------------------------------------------
    # Sync cache bridge (same pattern as OHLCVFetchTool)
    # ------------------------------------------------------------------

    def _run_sync_cache_get(self, key: str):
        import asyncio, concurrent.futures

        async def _get():
            return await self._get_cached(key)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _get()).result()
        except RuntimeError:
            return asyncio.run(_get())

    def _run_sync_cache_set(self, key: str, value, ttl: int) -> None:
        import asyncio, concurrent.futures

        async def _set():
            await self._set_cached(key, value, ttl)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, _set()).result()
        except RuntimeError:
            asyncio.run(_set())
