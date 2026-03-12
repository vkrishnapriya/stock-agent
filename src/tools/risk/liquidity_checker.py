"""
src/tools/risk/liquidity_checker.py
LiquidityCheckerTool — 20-day average daily volume (ADV) analysis.

Computes the 20-day ADV in share and INR terms from yfinance daily data
and determines whether an intended trade would represent a material portion
of normal daily turnover:

  OK    — trade < 10 % of ADV  (low market impact)
  WARN  — trade 10–25 % of ADV (moderate impact; consider splitting)
  AVOID — trade > 25 % of ADV  (high market impact risk)

ADV data is cached for 1 hour.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pandas as pd
import structlog
import yfinance as yf
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool

log = structlog.get_logger(__name__)

_ADV_WINDOW = 20
_WARN_THRESHOLD = 0.10    # 10 % of ADV
_AVOID_THRESHOLD = 0.25   # 25 % of ADV
_CACHE_TTL = 3_600


class LiquidityInput(BaseModel):
    """Input schema for :class:`LiquidityCheckerTool`."""

    symbol: str = Field(
        description=(
            "Ticker with exchange suffix: 'INFY.NS' (NSE) or 'INFY.BO' (BSE)."
        )
    )
    intended_trade_value_inr: float = Field(
        gt=0.0,
        description="Intended trade size in INR (quantity × entry_price).",
    )


class LiquidityCheckerTool(BaseTool):
    """Assess market-impact risk by comparing trade size to the 20-day ADV.

    Example::

        tool = LiquidityCheckerTool()
        result = json.loads(tool._run(
            symbol="INFY.NS",
            intended_trade_value_inr=500_000.0,
        ))
        print(result["liquidity_flag"], result["position_pct_of_adv"])
    """

    name: str = "liquidity_checker"
    description: str = (
        "Computes the 20-day average daily volume (ADV) in INR for an NSE/BSE stock "
        "and checks whether an intended trade exceeds 10% of ADV. "
        "Returns adv_shares, adv_inr, position_pct_of_adv, and "
        "liquidity_flag (OK / WARN / AVOID)."
    )
    args_schema: type[BaseModel] = LiquidityInput

    rate_limit_key: ClassVar[str] = "yfinance"
    rate_limit_per_minute: ClassVar[int] = 60

    # ------------------------------------------------------------------

    def _run(  # type: ignore[override]
        self,
        symbol: str,
        intended_trade_value_inr: float,
    ) -> str:
        symbol = symbol.strip().upper()

        # Try cache for ADV data only (trade value changes per call)
        cached = self._run_sync_cache_get(f"liq:{symbol}")
        if cached:
            adv_data = cached if isinstance(cached, dict) else json.loads(cached)
        else:
            log.info("liquidity_checker.fetching", symbol=symbol)
            ticker = yf.Ticker(symbol)
            df: pd.DataFrame = ticker.history(period="3mo", interval="1d", auto_adjust=True)

            if df.empty or len(df) < _ADV_WINDOW:
                raise ValueError(
                    f"Insufficient data for '{symbol}'. "
                    f"Need at least {_ADV_WINDOW} trading days of history."
                )

            df.columns = [c.title() for c in df.columns]
            df = df.sort_index()
            recent = df.tail(_ADV_WINDOW)

            avg_volume = float(recent["Volume"].mean())
            avg_close = float(recent["Close"].mean())
            adv_inr = round(avg_volume * avg_close, 2)

            adv_data = {
                "adv_shares": round(avg_volume, 0),
                "adv_inr": adv_inr,
                "avg_close": round(avg_close, 2),
            }
            self._run_sync_cache_set(f"liq:{symbol}", adv_data, _CACHE_TTL)
            log.info(
                "liquidity_checker.adv_computed",
                symbol=symbol,
                adv_shares=avg_volume,
                adv_inr=adv_inr,
            )

        adv_inr = adv_data["adv_inr"]
        position_pct = (
            round(intended_trade_value_inr / adv_inr * 100, 2) if adv_inr > 0 else 0.0
        )

        if position_pct >= _AVOID_THRESHOLD * 100:
            liquidity_flag = "AVOID"
        elif position_pct >= _WARN_THRESHOLD * 100:
            liquidity_flag = "WARN"
        else:
            liquidity_flag = "OK"

        result = {
            "symbol": symbol,
            "adv_shares": adv_data["adv_shares"],
            "adv_inr": adv_inr,
            "avg_close": adv_data["avg_close"],
            "intended_trade_value_inr": round(intended_trade_value_inr, 2),
            "position_pct_of_adv": position_pct,
            "liquidity_flag": liquidity_flag,
        }

        log.info(
            "liquidity_checker.done",
            symbol=symbol,
            position_pct_of_adv=position_pct,
            flag=liquidity_flag,
        )
        return json.dumps(result)

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
