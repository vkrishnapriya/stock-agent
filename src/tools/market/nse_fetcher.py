"""
src/tools/market/nse_fetcher.py
HTTP session for NSE India website. Handles cookie refresh automatically.
NSE returns 401 after ~5 min of inactivity — this session auto-recovers.

Also provides :class:`NSEResultsTool` which fetches the last 8 consolidated
quarterly financial results (EPS, Revenue, PAT) from the NSE corporates API.

Usage::

    session = NSESession()
    data = session.get("https://www.nseindia.com/api/quote-equity?symbol=RELIANCE")

    tool = NSEResultsTool()
    results = json.loads(tool._run(symbol="INFY"))
"""

from __future__ import annotations

import json
import time
from threading import Lock
from typing import Any, ClassVar

import httpx
import structlog
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool as _BaseTool

log = structlog.get_logger(__name__)

_NSE_HOME = "https://www.nseindia.com"
_SESSION_TTL = 270  # seconds; refresh before NSE's ~5-min timeout

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": _NSE_HOME,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


class NSESession:
    """Thread-safe HTTP session for the NSE website with automatic cookie refresh.

    The session visits the NSE homepage to acquire session cookies before
    making API calls. It refreshes the session automatically after
    :attr:`_SESSION_TTL` seconds of inactivity.
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(
            headers=_DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        self._lock = Lock()
        self._last_refresh: float = 0.0

    def _refresh_session(self) -> None:
        """Hit the NSE homepage to obtain fresh session cookies."""
        log.debug("nse_session.refreshing")
        self._client.get(_NSE_HOME)
        self._last_refresh = time.monotonic()
        log.debug("nse_session.refreshed")

    def _ensure_session(self) -> None:
        elapsed = time.monotonic() - self._last_refresh
        if elapsed >= _SESSION_TTL:
            with self._lock:
                # Double-check after acquiring lock.
                if time.monotonic() - self._last_refresh >= _SESSION_TTL:
                    self._refresh_session()

    def get(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Perform a GET request and return the parsed JSON response.

        Args:
            url:     Full NSE API endpoint URL.
            **kwargs: Extra arguments forwarded to :meth:`httpx.Client.get`.

        Returns:
            Parsed JSON as a dict.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses after cookie refresh.
        """
        self._ensure_session()
        response = self._client.get(url, **kwargs)

        if response.status_code == 401:
            log.warning("nse_session.401_received", url=url)
            with self._lock:
                self._refresh_session()
            response = self._client.get(url, **kwargs)

        response.raise_for_status()
        return response.json()  # type: ignore[return-value]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "NSESession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# NSEResultsTool
# ---------------------------------------------------------------------------

_NSE_RESULTS_LIST_URL = (
    "https://www.nseindia.com/api/corporates-financial-results"
    "?index=equities&symbol={symbol}&period=Quarterly"
)
_NSE_RESULTS_DATA_URL = (
    "https://www.nseindia.com/api/corporates-financial-results-data"
    "?index=equities&params={params}&seq_id={seq_id}"
    "&industry={ind_as}&ind={ind_as}&format={fmt}&bank={bank}"
)
_NUM_QUARTERS = 8
_LAKHS_TO_CRORE = 100  # 1 crore = 100 lakhs


class NSEResultsInput(BaseModel):
    """Input schema for :class:`NSEResultsTool`."""

    symbol: str = Field(
        description="NSE symbol without exchange suffix (e.g. 'INFY', 'RELIANCE')."
    )


class NSEResultsTool(_BaseTool):
    """Fetch the last 8 consolidated quarterly financial results from NSE.

    Returns a JSON list with fields per quarter::

        period       — "Q3 FY25" style label
        from_date    — period start  (YYYY-MM-DD)
        to_date      — period end    (YYYY-MM-DD)
        revenue_cr   — total revenue in INR crore
        pat_cr       — profit after tax in INR crore
        basic_eps    — basic EPS (INR)
        audited      — "Audited" | "Un-Audited"
        filing_date  — date NSE received the filing

    Example::

        tool = NSEResultsTool()
        quarters = json.loads(tool._run(symbol="INFY"))
        for q in quarters:
            print(q["period"], q["revenue_cr"], q["basic_eps"])
    """

    name: str = "nse_quarterly_results"
    description: str = (
        "Fetches the last 8 consolidated quarterly financial results from the "
        "NSE corporates API. Returns EPS, revenue (INR crore), and PAT per quarter."
    )
    args_schema: type[BaseModel] = NSEResultsInput

    rate_limit_key: ClassVar[str] = "nse"
    rate_limit_per_minute: ClassVar[int] = 30

    def _run(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        log.info("nse_quarterly_results.fetching", symbol=symbol)

        session = NSESession()
        try:
            # ── Step 1: get the list of quarterly filings ─────────────────
            list_url = _NSE_RESULTS_LIST_URL.format(symbol=symbol)
            raw = session.get(list_url)
            items: list[dict[str, Any]] = (
                raw if isinstance(raw, list) else raw.get("data", [])
            )

            # Keep only consolidated quarterly filings, sorted newest-first
            consol = [
                i for i in items
                if i.get("consolidated") == "Consolidated"
                and i.get("period") == "Quarterly"
            ]
            consol.sort(key=lambda x: str(x.get("toDate", "")), reverse=True)
            consol = consol[:_NUM_QUARTERS]

            if not consol:
                log.warning("nse_quarterly_results.no_data", symbol=symbol)
                return json.dumps([])

            # ── Step 2: fetch detailed financials for each quarter ────────
            quarters: list[dict[str, Any]] = []
            for item in consol:
                params = item.get("params", "")
                seq_id = item.get("seqNumber", "")
                ind_as = item.get("indAs", "")
                fmt = item.get("format", "")
                bank = item.get("bank", "N")

                data_url = _NSE_RESULTS_DATA_URL.format(
                    params=params, seq_id=seq_id,
                    ind_as=ind_as, fmt=fmt, bank=bank,
                )
                try:
                    detail = session.get(data_url)
                except Exception as exc:
                    log.warning(
                        "nse_quarterly_results.detail_failed",
                        symbol=symbol, params=params, error=str(exc),
                    )
                    detail = {}

                rd2: dict[str, Any] = detail.get("resultsData2") or {}

                # Convert from lakhs to crores
                revenue_lakhs = rd2.get("re_net_sale")
                pat_lakhs = rd2.get("re_net_profit")
                revenue_cr = (
                    round(float(revenue_lakhs) / _LAKHS_TO_CRORE, 2)
                    if revenue_lakhs is not None
                    else None
                )
                pat_cr = (
                    round(float(pat_lakhs) / _LAKHS_TO_CRORE, 2)
                    if pat_lakhs is not None
                    else None
                )

                # EPS — prefer basic; fall back to diluted
                basic_eps = rd2.get("re_basic_eps_for_cont_dic_opr")
                if basic_eps is None:
                    basic_eps = rd2.get("re_basic_eps")
                if basic_eps is None:
                    basic_eps = rd2.get("re_dilut_eps_for_cont_dic_opr")

                # Human-readable period label
                fy_start = item.get("financialYear", "")[:4]  # e.g. "01-Apr-2024" → "2024"
                quarter_map = {
                    "First Quarter": "Q1",
                    "Second Quarter": "Q2",
                    "Third Quarter": "Q3",
                    "Fourth Quarter": "Q4",
                }
                q_label = quarter_map.get(item.get("relatingTo", ""), "Q?")
                # FY label: "FY25" from financialYear "01-Apr-2024 To 31-Mar-2025"
                fy_end = item.get("financialYear", "")[-4:]
                period_label = f"{q_label} FY{fy_end[2:]}"

                quarters.append({
                    "period": period_label,
                    "from_date": item.get("fromDate", ""),
                    "to_date": item.get("toDate", ""),
                    "revenue_cr": revenue_cr,
                    "pat_cr": pat_cr,
                    "basic_eps": float(basic_eps) if basic_eps is not None else None,
                    "audited": item.get("audited", ""),
                    "filing_date": item.get("filingDate", ""),
                })

                time.sleep(0.3)   # be respectful to NSE API

        finally:
            session.close()

        log.info(
            "nse_quarterly_results.done",
            symbol=symbol,
            quarters=len(quarters),
        )
        return json.dumps(quarters)

