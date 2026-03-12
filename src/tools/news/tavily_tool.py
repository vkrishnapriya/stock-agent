"""
src/tools/news/tavily_tool.py
TavilySearchTool — deep web search for financial news via the Tavily AI API.

Tavily's ``search_depth="advanced"`` retrieves full page content and ranks
results by relevance, making it more useful than a plain web search for
surfacing earnings reports, analyst commentary, and regulatory filings.

Rate is capped at **33 requests/day** (free-tier) using the shared ``TAVILY``
Redis sliding-window limiter from :mod:`src.utils.rate_limiter`.

Results are cached for 30 minutes.

Returned fields per result
--------------------------
* ``title``          — page title
* ``content``        — relevant extracted text (up to ~400 chars)
* ``url``            — source URL
* ``published_date`` — ISO date string when available, empty string otherwise
* ``score``          — Tavily relevance score (0–1)

Usage::

    tool = TavilySearchTool()
    results = json.loads(tool._run(symbol="INFY", company_name="Infosys"))
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field
from tavily import TavilyClient

from src.config.settings import get_settings
from src.tools.base_tool import BaseTool
from src.utils.rate_limiter import TAVILY as _TAVILY_LIMITER
from src.utils.rate_limiter import RateLimitError

log = structlog.get_logger(__name__)

_CACHE_TTL = 1_800   # 30 minutes
_MAX_RESULTS = 5
_SEARCH_DEPTH = "advanced"


class TavilyInput(BaseModel):
    """Input schema for :class:`TavilySearchTool`."""

    symbol: str = Field(description="NSE symbol (e.g. 'INFY').")
    company_name: str = Field(
        default="",
        description=(
            "Full or common company name (e.g. 'Infosys'). "
            "Enriches the search query. Falls back to symbol if empty."
        ),
    )


class TavilySearchTool(BaseTool):
    """Deep AI-powered web search for NSE stock news and filings via Tavily.

    Rate-limited to 33 requests/day (free tier).

    Example::

        tool = TavilySearchTool()
        results = json.loads(tool._run(symbol="INFY", company_name="Infosys"))
        for r in results:
            print(r["score"], r["title"])
    """

    name: str = "tavily_search"
    description: str = (
        "Performs an AI-powered deep web search for recent NSE stock news, "
        "earnings reports, analyst commentary, and regulatory filings via "
        "the Tavily API. Returns up to 5 results with title, content, url, "
        "published_date, and relevance score."
    )
    args_schema: type[BaseModel] = TavilyInput

    rate_limit_key: ClassVar[str] = "tavily"
    rate_limit_per_minute: ClassVar[int] = 33  # nominal; actual window is 1 day

    # ------------------------------------------------------------------
    # Override: use the 33/day TAVILY limiter, not the default 60/min one
    # ------------------------------------------------------------------

    async def _check_rate_limit(self) -> None:
        allowed = await _TAVILY_LIMITER.acquire()
        if not allowed:
            raise RateLimitError(
                f"Tavily daily request limit reached "
                f"(limit={_TAVILY_LIMITER.limit}/day)"
            )

    # ------------------------------------------------------------------

    def _run(  # type: ignore[override]
        self,
        symbol: str,
        company_name: str = "",
    ) -> str:
        symbol = symbol.strip().upper()
        name = company_name.strip() or symbol
        query = f"{symbol} {name} stock news earnings"

        cache_key = f"{symbol}:{name}"
        cached = self._run_sync_cache_get(cache_key)
        if cached:
            log.debug("tavily_search.cache_hit", symbol=symbol)
            return cached if isinstance(cached, str) else json.dumps(cached)

        api_key = get_settings().tavily_api_key
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is not configured. "
                "Set it in .env.local or as an environment variable."
            )

        log.info("tavily_search.fetching", symbol=symbol, query=query)
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=_MAX_RESULTS,
            search_depth=_SEARCH_DEPTH,
        )

        raw_results = response.get("results", []) if isinstance(response, dict) else []

        results = [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", ""),
                "published_date": r.get("published_date", ""),
                "score": round(float(r.get("score", 0.0)), 4),
            }
            for r in raw_results
        ]

        result_json = json.dumps(results)
        self._run_sync_cache_set(cache_key, result_json, _CACHE_TTL)
        log.info("tavily_search.done", symbol=symbol, count=len(results))
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
