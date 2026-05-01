"""
src/tools/market/screener_tools.py
ScreenerFinancialsTool — scrapes Screener.in for historical financial statements.
CompetitorMapTool      — scrapes Screener.in for sector peer comparisons.

Fetches the consolidated company page at::

    https://www.screener.in/company/{SYMBOL}/consolidated/

ScreenerFinancialsTool extracts:
  * Key ratios   — P/E, Book Value, ROE, ROCE, Market Cap, Dividend Yield
  * Profit & Loss — annual Sales, Net Profit, EPS (up to 10 years)
  * Balance Sheet — annual Equity, Reserves, Borrowings, Total Assets
  * Cash Flows   — annual Operating, Investing, Financing cash flows
  * Quarterly Results — last 8 quarters: Sales, Net Profit, EPS
  * Shareholding  — latest Promoter / FII / DII holding %

CompetitorMapTool extracts:
  * Peers table — {symbol, company_name, market_cap, pe_ratio} for up to 5 peers

Rate limiting
-------------
A 2-second sleep is applied before every HTTP request to avoid hammering
Screener.in.  Results are cached for 3 600 seconds (1 hour) in Redis.
"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

import httpx
import structlog
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool

log = structlog.get_logger(__name__)

_SCREENER_URL = "https://www.screener.in/company/{symbol}/consolidated/"
_CACHE_TTL = 3_600          # 1 hour
_REQUEST_DELAY = 2.0        # seconds between requests (courtesy rate limit)
_NUM_QUARTERLY = 4          # most-recent quarters to return
_NUM_ANNUAL = 5             # most-recent annual periods to return

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class ScreenerInput(BaseModel):
    """Input schema for :class:`ScreenerFinancialsTool`."""

    symbol: str = Field(
        description="NSE symbol without exchange suffix (e.g. 'INFY', 'RELIANCE')."
    )


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Strip whitespace, NBSP, and the '+' expansion widget suffix."""
    return text.replace("\xa0", " ").replace("+", "").strip()


def _parse_top_ratios(soup: BeautifulSoup) -> dict[str, str]:
    """Extract key ratios from the ``#top-ratios`` list."""
    ul = soup.find("ul", {"id": "top-ratios"})
    if not ul:
        return {}
    ratios: dict[str, str] = {}
    for li in ul.find_all("li"):
        name_el = li.find("span", {"class": "name"})
        val_el = li.find("span", {"class": "number"})
        if name_el and val_el:
            ratios[_clean(name_el.get_text())] = _clean(val_el.get_text())
    return ratios


def _parse_table(section: Tag) -> dict[str, dict[str, str]]:
    """Parse any Screener data table into ``{row_label: {period: value}}``."""
    table = section.find("table")
    if not table:
        return {}

    thead = table.find("thead")
    headers: list[str] = (
        [_clean(th.get_text()) for th in thead.find_all("th")]
        if thead
        else []
    )
    period_headers = headers[1:]  # first header is the row-label column

    result: dict[str, dict[str, str]] = {}
    tbody = table.find("tbody")
    if not tbody:
        return {}
    for tr in tbody.find_all("tr"):
        cells = [_clean(td.get_text()) for td in tr.find_all(["td", "th"])]
        if not cells:
            continue
        row_label = cells[0]
        if not row_label or row_label.lower() in {"raw pdf"}:
            continue
        values = cells[1:]
        result[row_label] = dict(zip(period_headers, values))
    return result


def _parse_shareholding_latest(section: Tag) -> dict[str, str]:
    """Return the most-recent shareholding column as ``{holder: pct}``."""
    table = section.find("table")
    if not table:
        return {}
    thead = table.find("thead")
    if not thead:
        return {}
    col_count = len(thead.find_all("th"))
    # Last column is the most recent period
    result: dict[str, str] = {}
    tbody = table.find("tbody")
    if not tbody:
        return {}
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) >= col_count:
            holder = _clean(cells[0].get_text())
            pct = _clean(cells[-1].get_text())
            if holder:
                result[holder] = pct
    return result


def _last_n_cols(
    table_data: dict[str, dict[str, str]], n: int
) -> dict[str, dict[str, str]]:
    """Keep only the *n* most-recent period columns in a parsed table."""
    all_periods: list[str] = []
    for row_vals in table_data.values():
        for p in row_vals:
            if p not in all_periods:
                all_periods.append(p)
    keep = set(all_periods[-n:])
    return {
        row: {p: v for p, v in vals.items() if p in keep}
        for row, vals in table_data.items()
    }


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ScreenerFinancialsTool(BaseTool):
    """Scrape Screener.in for multi-year financial statements.

    Example::

        tool = ScreenerFinancialsTool()
        data = json.loads(tool._run(symbol="INFY"))
        print(data["key_ratios"]["Stock P/E"])
        print(data["annual_pl"]["Sales"])
    """

    name: str = "screener_financials"
    description: str = (
        "Scrapes Screener.in for a stock's consolidated financials: "
        "P/E, ROE, ROCE, 10-year P&L (Sales, Net Profit, EPS), Balance Sheet, "
        "Cash Flows, last 8 quarterly results, and promoter/FII/DII shareholding. "
        "Requires symbol without exchange suffix (e.g. 'INFY')."
    )
    args_schema: type[BaseModel] = ScreenerInput

    rate_limit_key: ClassVar[str] = "screener"
    rate_limit_per_minute: ClassVar[int] = 20  # well under the 2-second delay floor

    # ------------------------------------------------------------------

    def _run(self, symbol: str) -> str:  # type: ignore[override]
        symbol = symbol.strip().upper()

        cached = self._run_sync_cache_get(f"scr:{symbol}")
        if cached:
            log.debug("screener_financials.cache_hit", symbol=symbol)
            return cached if isinstance(cached, str) else json.dumps(cached)

        log.info("screener_financials.fetching", symbol=symbol)
        time.sleep(_REQUEST_DELAY)   # courtesy rate limit

        url = _SCREENER_URL.format(symbol=symbol)
        resp = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=20.0)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Key ratios ────────────────────────────────────────────────────
        key_ratios = _parse_top_ratios(soup)

        # ── Annual sections (last N years only) ───────────────────────────
        pl_section = soup.find("section", {"id": "profit-loss"})
        annual_pl = _last_n_cols(_parse_table(pl_section) if pl_section else {}, _NUM_ANNUAL)  # type: ignore[arg-type]

        bs_section = soup.find("section", {"id": "balance-sheet"})
        balance_sheet = _last_n_cols(_parse_table(bs_section) if bs_section else {}, _NUM_ANNUAL)  # type: ignore[arg-type]

        cf_section = soup.find("section", {"id": "cash-flow"})
        cash_flows = _last_n_cols(_parse_table(cf_section) if cf_section else {}, _NUM_ANNUAL)  # type: ignore[arg-type]

        # ── Quarterly results (last N quarters) ────────────────────────────
        qr_section = soup.find("section", {"id": "quarters"})
        raw_quarterly = _parse_table(qr_section) if qr_section else {}  # type: ignore[arg-type]
        quarterly_results = _last_n_cols(raw_quarterly, _NUM_QUARTERLY)

        # ── Shareholding ────────────────────────────────────────────────────
        sh_section = soup.find("section", {"id": "shareholding"})
        shareholding = (
            _parse_shareholding_latest(sh_section) if sh_section else {}  # type: ignore[arg-type]
        )

        result = {
            "symbol": symbol,
            "source_url": url,
            "key_ratios": key_ratios,
            "annual_pl": annual_pl,
            "balance_sheet": balance_sheet,
            "cash_flows": cash_flows,
            "quarterly_results": quarterly_results,
            "shareholding_latest": shareholding,
        }

        result_json = json.dumps(result)
        self._run_sync_cache_set(f"scr:{symbol}", result_json, _CACHE_TTL)
        log.info(
            "screener_financials.done",
            symbol=symbol,
            pl_rows=len(annual_pl),
            quarters=len(next(iter(quarterly_results.values()), {})) if quarterly_results else 0,
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


# ---------------------------------------------------------------------------
# Peers parsing helper
# ---------------------------------------------------------------------------


def _parse_peers(
    section: Tag, max_peers: int = 5
) -> list[dict[str, Any]]:
    """Parse the Screener.in peers comparison table.

    Returns a list of ``{symbol, company_name, market_cap, pe_ratio}`` dicts,
    capped at *max_peers* entries.  The NSE symbol is extracted from the
    ``href`` of each company link (e.g. ``/company/TCS/`` → ``"TCS"``).
    """
    table = section.find("table")
    if not table:
        return []

    thead = table.find("thead")
    headers: list[str] = []
    if thead:
        headers = [_clean(th.get_text()).lower() for th in thead.find_all("th")]

    def _find_col(keywords: list[str]) -> int | None:
        for i, h in enumerate(headers):
            if any(kw in h for kw in keywords):
                return i
        return None

    pe_idx = _find_col(["p/e", "pe "])
    mcap_idx = _find_col(["mar cap", "mkt cap", "m.cap", "market cap"])

    tbody = table.find("tbody")
    if not tbody:
        return []

    peers: list[dict[str, Any]] = []
    for tr in tbody.find_all("tr")[:max_peers]:  # type: ignore[union-attr]
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue

        # First cell: company name + href that encodes the NSE symbol
        name_cell = cells[0]
        link = name_cell.find("a")
        company_name = _clean(link.get_text() if link else name_cell.get_text())

        symbol = ""
        if link:
            href = str(link.get("href", ""))
            # e.g. "/company/TCS/" or "/company/TCS/consolidated/"
            parts = [p for p in href.split("/") if p]
            if "company" in parts:
                idx = parts.index("company")
                if idx + 1 < len(parts):
                    symbol = parts[idx + 1].upper()

        if not symbol:
            continue

        def _parse_num(col_idx: int | None) -> float | None:
            if col_idx is None or col_idx >= len(cells):
                return None
            text = _clean(cells[col_idx].get_text()).replace(",", "")
            try:
                return float(text)
            except (ValueError, AttributeError):
                return None

        peers.append({
            "symbol": symbol,
            "company_name": company_name,
            "market_cap": _parse_num(mcap_idx),
            "pe_ratio": _parse_num(pe_idx),
        })

    return peers


# ---------------------------------------------------------------------------
# CompetitorMapTool
# ---------------------------------------------------------------------------


class CompetitorMapInput(BaseModel):
    """Input schema for :class:`CompetitorMapTool`."""

    symbol: str = Field(
        description="NSE symbol without exchange suffix (e.g. 'INFY', 'RELIANCE')."
    )
    max_peers: int = Field(default=5, ge=1, le=10)


class CompetitorMapTool(BaseTool):
    """Fetch sector peers for a stock from Screener.in's Peers table.

    Example::

        tool = CompetitorMapTool()
        peers = json.loads(tool._run(symbol="INFY"))
        for p in peers:
            print(p["symbol"], p["company_name"], p["pe_ratio"])
    """

    name: str = "competitor_map"
    description: str = (
        "Fetches the peer/competitor list from Screener.in for a stock. "
        "Returns a JSON list of {symbol, company_name, market_cap, pe_ratio} "
        "for up to 5 sector peers. Requires symbol without exchange suffix (e.g. 'INFY')."
    )
    args_schema: type[BaseModel] = CompetitorMapInput

    rate_limit_key: ClassVar[str] = "screener"
    rate_limit_per_minute: ClassVar[int] = 20

    # ------------------------------------------------------------------

    def _run(self, symbol: str, max_peers: int = 5) -> str:  # type: ignore[override]
        symbol = symbol.strip().upper()
        cache_key = f"peers:{symbol}"

        cached = self._run_sync_cache_get(cache_key)
        if cached:
            log.debug("competitor_map.cache_hit", symbol=symbol)
            return cached if isinstance(cached, str) else json.dumps(cached)

        log.info("competitor_map.fetching", symbol=symbol)
        time.sleep(_REQUEST_DELAY)  # courtesy rate limit

        url = _SCREENER_URL.format(symbol=symbol)
        resp = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=20.0)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        peers_section = soup.find("section", {"id": "peers"})
        peers = (
            _parse_peers(peers_section, max_peers=max_peers)  # type: ignore[arg-type]
            if peers_section
            else []
        )

        result_json = json.dumps(peers)
        self._run_sync_cache_set(cache_key, result_json, _CACHE_TTL)
        log.info("competitor_map.done", symbol=symbol, peers=len(peers))
        return result_json

    # ------------------------------------------------------------------
    # Sync cache bridge (mirrors ScreenerFinancialsTool)
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
