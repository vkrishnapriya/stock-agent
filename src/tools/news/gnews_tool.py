"""
src/tools/news/gnews_tool.py
GNewsAPITool — fetches recent financial news via the gnews Python library.

The ``gnews`` library scrapes Google News RSS feeds; no API key is required.
Rate is capped at **100 requests/day** using the shared ``GNEWS`` Redis
sliding-window limiter from :mod:`src.utils.rate_limiter`.

Results are cached for 1 hour per (symbol, company_name) pair.

Returned fields per article
----------------------------
* ``title``        — headline
* ``description``  — snippet / lede
* ``url``          — full article URL
* ``published_at`` — publication timestamp string
* ``publisher``    — outlet name

Usage::

    tool = GNewsAPITool()
    articles = json.loads(tool._run(symbol="INFY", company_name="Infosys"))
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import gnews
import structlog
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool
from src.utils.rate_limiter import GNEWS as _GNEWS_LIMITER
from src.utils.rate_limiter import RateLimitError

log = structlog.get_logger(__name__)

_CACHE_TTL = 3_600   # 1 hour
_MAX_ARTICLES = 3
_PERIOD = "7d"
_LANGUAGE = "en"
_COUNTRY = "IN"


class GNewsInput(BaseModel):
    """Input schema for :class:`GNewsAPITool`."""

    symbol: str = Field(description="NSE symbol (e.g. 'INFY').")
    company_name: str = Field(
        default="",
        description=(
            "Full or common company name (e.g. 'Infosys'). "
            "Used to build a richer search query. Falls back to symbol if empty."
        ),
    )


class GNewsAPITool(BaseTool):
    """Search Google News for recent financial articles about an NSE stock.

    Rate-limited to 100 requests/day (Google News scraping courtesy limit).

    Example::

        tool = GNewsAPITool()
        articles = json.loads(tool._run(symbol="INFY", company_name="Infosys"))
        for a in articles:
            print(a["published_at"], a["title"])
    """

    name: str = "gnews_search"
    description: str = (
        "Searches Google News for the latest financial articles about an NSE "
        "stock using the gnews library. Returns up to 10 articles from the last "
        "7 days with title, description, url, published_at, and publisher fields."
    )
    args_schema: type[BaseModel] = GNewsInput

    rate_limit_key: ClassVar[str] = "gnews"
    rate_limit_per_minute: ClassVar[int] = 100  # nominal; actual window is 1 day

    # ------------------------------------------------------------------
    # Override: use the 100/day GNEWS limiter, not the default 60/min one
    # ------------------------------------------------------------------

    async def _check_rate_limit(self) -> None:
        allowed = await _GNEWS_LIMITER.acquire()
        if not allowed:
            raise RateLimitError(
                f"GNews daily request limit reached "
                f"(limit={_GNEWS_LIMITER.limit}/day)"
            )

    # ------------------------------------------------------------------

    def _run(  # type: ignore[override]
        self,
        symbol: str,
        company_name: str = "",
    ) -> str:
        symbol = symbol.strip().upper()
        name = company_name.strip() or symbol
        query = f"{name} NSE stock"

        cache_key = f"{symbol}:{name}"
        cached = self._run_sync_cache_get(cache_key)
        if cached:
            log.debug("gnews_search.cache_hit", symbol=symbol)
            return cached if isinstance(cached, str) else json.dumps(cached)

        log.info("gnews_search.fetching", symbol=symbol, query=query)
        client = gnews.GNews(
            language=_LANGUAGE,
            country=_COUNTRY,
            max_results=_MAX_ARTICLES,
            period=_PERIOD,
        )
        raw_articles = client.get_news(query) or []

        articles = [
            {
                "title": a.get("title", "")[:120],
                "description": a.get("description", "")[:150],
                "published_at": a.get("published date", ""),
                "publisher": a.get("publisher", {}).get("title", "")
                if isinstance(a.get("publisher"), dict)
                else str(a.get("publisher", "")),
            }
            for a in raw_articles
        ]

        result_json = json.dumps(articles)
        self._run_sync_cache_set(cache_key, result_json, _CACHE_TTL)
        log.info("gnews_search.done", symbol=symbol, count=len(articles))
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
