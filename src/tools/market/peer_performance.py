"""
src/tools/market/peer_performance.py
PeerPricePerformanceTool — compare 1M/3M price returns for a batch of NSE stocks.

Fetches 3 months of daily adjusted-close prices via yfinance for each symbol
in the input list, then computes:
  * 1-month return  (~21 trading days back from today)
  * 3-month return  (full window start to today)

Used by CompetitorAnalysisAgent to rank the target stock against its peers.

Usage::

    tool = PeerPricePerformanceTool()
    data = json.loads(tool._run(symbols_csv="INFY,TCS,WIPRO"))
    for symbol, perf in data.items():
        print(symbol, perf["return_1m_pct"], perf["return_3m_pct"])
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import structlog
import yfinance as yf
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool

log = structlog.get_logger(__name__)

_CACHE_TTL = 3_600        # 1 hour
_TRADING_DAYS_1M = 21     # approximate trading days in one calendar month


class PeerPerformanceInput(BaseModel):
    """Input schema for :class:`PeerPricePerformanceTool`."""

    symbols_csv: str = Field(
        description=(
            "Comma-separated NSE symbols without exchange suffix, "
            "e.g. 'INFY,TCS,WIPRO'. The tool appends '.NS' internally."
        )
    )


class PeerPricePerformanceTool(BaseTool):
    """Compute 1-month and 3-month price returns for a batch of NSE stocks.

    Downloads 3 months of daily data in a single yfinance batch call, then
    computes returns for each symbol.  Falls back to the full available range
    for the 1M return when fewer than 21 rows of data exist.

    Example::

        tool = PeerPricePerformanceTool()
        data = json.loads(tool._run(symbols_csv="INFY,TCS,WIPRO"))
        for sym, perf in data.items():
            print(sym, perf["return_1m_pct"], perf["return_3m_pct"])
    """

    name: str = "peer_price_performance"
    description: str = (
        "Fetches 3 months of daily price data for a comma-separated list of NSE "
        "symbols and computes their 1-month (~21 trading days) and 3-month price "
        "returns. Returns {symbol: {return_1m_pct, return_3m_pct, current_price}}."
    )
    args_schema: type[BaseModel] = PeerPerformanceInput

    rate_limit_key: ClassVar[str] = "yfinance"
    rate_limit_per_minute: ClassVar[int] = 60

    # ------------------------------------------------------------------

    def _run(self, symbols_csv: str) -> str:  # type: ignore[override]
        symbols = [s.strip().upper() for s in symbols_csv.split(",") if s.strip()]
        cache_key = f"pperf:{','.join(sorted(symbols))}"

        cached = self._run_sync_cache_get(cache_key)
        if cached:
            log.debug("peer_price_performance.cache_hit", symbols=symbols)
            return cached if isinstance(cached, str) else json.dumps(cached)

        log.info("peer_price_performance.fetching", symbols=symbols)
        ns_symbols = [f"{s}.NS" for s in symbols]

        # Single batch download — much faster than per-ticker calls
        df = yf.download(
            ns_symbols,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        result: dict[str, Any] = {}
        for sym, ns_sym in zip(symbols, ns_symbols):
            try:
                # yfinance returns a flat DataFrame for a single ticker,
                # a MultiIndex DataFrame for multiple tickers.
                if len(ns_symbols) == 1:
                    close = df["Close"]
                else:
                    close = df["Close"][ns_sym]

                close = close.dropna()
                if close.empty:
                    result[sym] = {
                        "return_1m_pct": None,
                        "return_3m_pct": None,
                        "current_price": None,
                    }
                    continue

                current_price = round(float(close.iat[-1]), 2)

                # 3M return: oldest available price to today
                price_3m_ago = float(close.iat[0])
                return_3m = round((current_price / price_3m_ago - 1) * 100, 2)

                # 1M return: ~21 trading days back; fall back to 3M if insufficient
                if len(close) >= _TRADING_DAYS_1M:
                    price_1m_ago = float(close.iat[-_TRADING_DAYS_1M])
                    return_1m = round((current_price / price_1m_ago - 1) * 100, 2)
                else:
                    return_1m = return_3m

                result[sym] = {
                    "return_1m_pct": return_1m,
                    "return_3m_pct": return_3m,
                    "current_price": current_price,
                }
            except Exception as exc:
                log.warning(
                    "peer_price_performance.symbol_failed",
                    symbol=sym,
                    error=str(exc),
                )
                result[sym] = {
                    "return_1m_pct": None,
                    "return_3m_pct": None,
                    "current_price": None,
                }

        result_json = json.dumps(result)
        self._run_sync_cache_set(cache_key, result_json, _CACHE_TTL)
        log.info("peer_price_performance.done", count=len(result))
        return result_json

    # ------------------------------------------------------------------
    # Sync cache bridge
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
