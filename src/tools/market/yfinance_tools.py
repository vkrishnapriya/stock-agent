"""
src/tools/market/yfinance_tools.py
OHLCVFetchTool — fetches OHLCV price/volume data from Yahoo Finance.

Returns a JSON-serialised list of OHLCV records suitable for downstream
indicator computation.  Results are cached in Redis:
  * Intraday intervals (≤1h): 60 seconds
  * Daily / weekly / monthly:  3600 seconds
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal

import pandas as pd
import structlog
import yfinance as yf
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool

log = structlog.get_logger(__name__)

# Intervals that carry intraday (time-of-day) information
_INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

_LIVE_TTL  = 60       # seconds — intraday cache
_DAILY_TTL = 3_600    # seconds — end-of-day cache


class OHLCVInput(BaseModel):
    """Input schema for :class:`OHLCVFetchTool`."""

    symbol: str = Field(
        description=(
            "Ticker symbol with exchange suffix. "
            "NSE stocks: append '.NS'  (e.g. 'INFY.NS', 'RELIANCE.NS'). "
            "BSE stocks: append '.BO' (e.g. 'INFY.BO')."
        )
    )
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"] = Field(
        default="1y",
        description="Historical look-back period.",
    )
    interval: Literal[
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk", "1mo"
    ] = Field(
        default="1d",
        description="Bar interval.  Use '1d' for daily, '5m'/'15m' for intraday.",
    )


class OHLCVFetchTool(BaseTool):
    """Fetch OHLCV data for a single symbol from Yahoo Finance.

    Returns a JSON string representing a list of records with keys:
    ``Date`` (or ``Datetime`` for intraday), ``Open``, ``High``, ``Low``,
    ``Close``, ``Volume``.

    Example::

        tool = OHLCVFetchTool()
        json_str = tool._run(symbol="INFY.NS", period="6mo", interval="1d")
    """

    name: str = "ohlcv_fetch"
    description: str = (
        "Fetches historical OHLCV (Open/High/Low/Close/Volume) data for a "
        "given NSE/BSE stock symbol using Yahoo Finance.  Returns a JSON "
        "array of price bars. Use symbol format 'TICKER.NS' for NSE stocks."
    )
    args_schema: type[BaseModel] = OHLCVInput

    rate_limit_key: ClassVar[str] = "yfinance"
    rate_limit_per_minute: ClassVar[int] = 60

    # ------------------------------------------------------------------

    def _run(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> str:
        symbol = symbol.strip().upper()
        is_intraday = interval in _INTRADAY_INTERVALS
        ttl = _LIVE_TTL if is_intraday else _DAILY_TTL

        cache_key = f"{symbol}:{period}:{interval}"
        cached = self._run_sync_cache_get(cache_key)
        if cached:
            log.debug("ohlcv_fetch.cache_hit", symbol=symbol, interval=interval)
            return cached if isinstance(cached, str) else json.dumps(cached)

        log.info("ohlcv_fetch.fetching", symbol=symbol, period=period, interval=interval)
        ticker = yf.Ticker(symbol)
        df: pd.DataFrame = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            raise ValueError(
                f"No OHLCV data returned for '{symbol}' "
                f"(period={period}, interval={interval}). "
                "Check symbol format — NSE stocks need '.NS' suffix."
            )

        # Normalise column names (yfinance may capitalise differently)
        df.columns = [c.title() for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Datetime" if is_intraday else "Date"
        df = df.reset_index()
        df["Datetime" if is_intraday else "Date"] = (
            df["Datetime" if is_intraday else "Date"].astype(str)
        )

        # Limit to last 60 bars to keep context size small
        df = df.tail(60)
        result = df.to_json(orient="records")
        self._run_sync_cache_set(cache_key, result, ttl)
        log.info("ohlcv_fetch.done", symbol=symbol, rows=len(df))
        return result

    # ------------------------------------------------------------------
    # Sync cache helpers (wraps async cache in a straightforward way)
    # ------------------------------------------------------------------

    def _run_sync_cache_get(self, key: str) -> Any | None:
        import asyncio
        import concurrent.futures

        async def _get():
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

        async def _set():
            await self._set_cached(key, value, ttl)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, _set()).result()
        except RuntimeError:
            asyncio.run(_set())
