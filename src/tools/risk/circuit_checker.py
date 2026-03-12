"""
src/tools/risk/circuit_checker.py
CircuitBreakerTool — fetches NSE individual stock circuit bands.

NSE applies price bands of 2 %, 5 %, 10 %, or 20 % depending on the scrip.
This tool fetches upperCP / lowerCP from the NSE equity-quote API and:
  * Infers the active circuit band from the price spread.
  * Emits a :class:`CircuitBreakerWarning` if the last price is within 1 %
    of either limit (signal to avoid chasing price into a circuit).

Result is cached for 60 seconds (limits are set once at market open).
"""

from __future__ import annotations

import json
import warnings
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool
from src.tools.market.nse_fetcher import NSESession

log = structlog.get_logger(__name__)

_NSE_QUOTE_URL = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"
_CACHE_TTL = 60            # seconds — circuit limits reset at market open
_WARN_PROXIMITY = 0.01     # warn if within 1 % of a circuit limit


class CircuitBreakerWarning(UserWarning):
    """Emitted when a stock price is within 1 % of its circuit breaker limit."""


class CircuitBreakerInput(BaseModel):
    """Input schema for :class:`CircuitBreakerTool`."""

    symbol: str = Field(
        description="NSE symbol without exchange suffix (e.g. 'RELIANCE')."
    )


def _detect_band(upper_limit: float, last_price: float) -> str:
    """Infer the circuit band from the distance between upper limit and last price."""
    if last_price <= 0:
        return "20%"
    pct = (upper_limit - last_price) / last_price * 100
    if pct <= 2.5:
        return "2%"
    if pct <= 7.5:
        return "5%"
    if pct <= 15.0:
        return "10%"
    return "20%"


class CircuitBreakerTool(BaseTool):
    """Fetch circuit breaker (price band) data from NSE for a single equity.

    Example::

        tool = CircuitBreakerTool()
        result = json.loads(tool._run(symbol="RELIANCE"))
        print(result["band"], result["upper_limit"], result["lower_limit"])
    """

    name: str = "circuit_checker"
    description: str = (
        "Fetches the active NSE circuit breaker band (2/5/10/20%) for a stock "
        "using the NSE equity-quote API. Returns band, upper_limit, lower_limit, "
        "last_price, and near_circuit flag. Issues a CircuitBreakerWarning if the "
        "price is within 1% of either circuit limit."
    )
    args_schema: type[BaseModel] = CircuitBreakerInput

    rate_limit_key: ClassVar[str] = "nse"
    rate_limit_per_minute: ClassVar[int] = 30

    # ------------------------------------------------------------------

    def _run(self, symbol: str) -> str:  # type: ignore[override]
        symbol = symbol.strip().upper()

        cached = self._run_sync_cache_get(f"circuit:{symbol}")
        if cached:
            log.debug("circuit_checker.cache_hit", symbol=symbol)
            return cached if isinstance(cached, str) else json.dumps(cached)

        log.info("circuit_checker.fetching", symbol=symbol)
        session = NSESession()
        try:
            data = session.get(_NSE_QUOTE_URL.format(symbol=symbol))
        finally:
            session.close()

        price_info = data.get("priceInfo", {})
        last_price = float(price_info.get("lastPrice", 0.0))
        upper_limit = float(price_info.get("upperCP", 0.0))
        lower_limit = float(price_info.get("lowerCP", 0.0))

        if upper_limit <= 0 or lower_limit <= 0:
            raise ValueError(
                f"Could not extract circuit limits for '{symbol}'. "
                "Verify the symbol is listed on NSE."
            )

        band = _detect_band(upper_limit, last_price)

        # ── Proximity warning ────────────────────────────────────────────────
        near_circuit = False
        if last_price > 0:
            upper_prox = (upper_limit - last_price) / last_price
            lower_prox = (last_price - lower_limit) / last_price
            if upper_prox <= _WARN_PROXIMITY or lower_prox <= _WARN_PROXIMITY:
                near_circuit = True
                direction = "upper" if upper_prox <= lower_prox else "lower"
                limit_val = upper_limit if direction == "upper" else lower_limit
                warnings.warn(
                    f"{symbol} is within 1% of its {direction} circuit limit "
                    f"({limit_val:.2f}); last_price={last_price:.2f}",
                    CircuitBreakerWarning,
                    stacklevel=2,
                )
                log.warning(
                    "circuit_checker.near_circuit",
                    symbol=symbol,
                    direction=direction,
                    limit=limit_val,
                    last_price=last_price,
                )

        result = {
            "symbol": symbol,
            "band": band,
            "upper_limit": round(upper_limit, 2),
            "lower_limit": round(lower_limit, 2),
            "last_price": round(last_price, 2),
            "near_circuit": near_circuit,
        }

        result_json = json.dumps(result)
        self._run_sync_cache_set(f"circuit:{symbol}", result_json, _CACHE_TTL)
        log.info(
            "circuit_checker.done",
            symbol=symbol,
            band=band,
            upper=upper_limit,
            lower=lower_limit,
        )
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
