"""
src/tools/news/nse_announcements.py
NSEAnnouncementTool — fetches the latest corporate announcements from NSE.

Calls the NSE corporate-announcements API (requires session cookies managed
by :class:`~src.tools.market.nse_fetcher.NSESession`) and returns the
10 most-recent filings for a given symbol.

Typical event types returned
-----------------------------
* Board Meeting / Financial Results
* Investor Presentation
* Corporate Action (dividend, split, buyback)
* Regulation 30 / LODR disclosures
* Change in management / key personnel
* Outcome of AGM/EGM

Announcements are cached for 5 minutes; cache is keyed to the symbol so
different symbols do not collide.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool
from src.tools.market.nse_fetcher import NSESession

log = structlog.get_logger(__name__)

_NSE_ANNOUNCEMENTS_URL = (
    "https://www.nseindia.com/api/corporates-announcements"
    "?index=equities&symbol={symbol}"
)
_CACHE_TTL = 300   # 5 minutes
_MAX_RESULTS = 10


class NSEAnnouncementInput(BaseModel):
    """Input schema for :class:`NSEAnnouncementTool`."""

    symbol: str = Field(
        description="NSE symbol without exchange suffix (e.g. 'INFY', 'RELIANCE')."
    )


class NSEAnnouncementTool(BaseTool):
    """Fetch the 10 most-recent NSE corporate announcements for a symbol.

    Example::

        tool = NSEAnnouncementTool()
        announcements = json.loads(tool._run(symbol="INFY"))
        for a in announcements:
            print(a["date"], a["description"])
    """

    name: str = "nse_announcements"
    description: str = (
        "Fetches the latest NSE corporate announcements (board meetings, "
        "financial results, corporate actions, LODR disclosures) for a given "
        "NSE-listed symbol. Returns up to 10 most-recent announcements as JSON."
    )
    args_schema: type[BaseModel] = NSEAnnouncementInput

    rate_limit_key: ClassVar[str] = "nse"
    rate_limit_per_minute: ClassVar[int] = 30

    # ------------------------------------------------------------------

    def _run(self, symbol: str) -> str:  # type: ignore[override]
        symbol = symbol.strip().upper()

        cached = self._run_sync_cache_get(f"ann:{symbol}")
        if cached:
            log.debug("nse_announcements.cache_hit", symbol=symbol)
            return cached if isinstance(cached, str) else json.dumps(cached)

        log.info("nse_announcements.fetching", symbol=symbol)
        session = NSESession()
        try:
            raw = session.get(_NSE_ANNOUNCEMENTS_URL.format(symbol=symbol))
        finally:
            session.close()

        # The API may return a list directly or wrap it in {"data": [...]}
        if isinstance(raw, dict):
            items = raw.get("data", [])
        elif isinstance(raw, list):
            items = raw
        else:
            items = []

        # Filter to the requested symbol in case the API returns mixed results
        items = [
            r for r in items
            if str(r.get("symbol", "")).upper() == symbol
        ]

        # Sort by announcement date descending and take the most recent N
        items.sort(key=lambda r: str(r.get("sort_date", "")), reverse=True)
        items = items[:_MAX_RESULTS]

        # Normalise to a stable, minimal schema
        announcements = [
            {
                "symbol": symbol,
                "company": r.get("sm_name", ""),
                "date": r.get("sort_date", r.get("bcastdttm", "")),
                "description": r.get("desc", r.get("subject", "")),
                "attachment": r.get("attchmntFile", ""),
            }
            for r in items
        ]

        result_json = json.dumps(announcements)
        self._run_sync_cache_set(f"ann:{symbol}", result_json, _CACHE_TTL)
        log.info(
            "nse_announcements.done",
            symbol=symbol,
            count=len(announcements),
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
