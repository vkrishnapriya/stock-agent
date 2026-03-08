"""
src/tools/market/nse_fetcher.py
HTTP session for NSE India website. Handles cookie refresh automatically.
NSE returns 401 after ~5 min of inactivity — this session auto-recovers.

Usage::

    session = NSESession()
    data = session.get("https://www.nseindia.com/api/quote-equity?symbol=RELIANCE")
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

import httpx
import structlog

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
