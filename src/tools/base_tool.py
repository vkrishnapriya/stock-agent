"""
src/tools/base_tool.py
Abstract base class for all trading tools. Provides rate limiting via Redis
and standardised error handling. All tools MUST extend this class.
"""

from __future__ import annotations

import os
import time
from abc import abstractmethod
from typing import Any, ClassVar

import redis
import structlog
from crewai.tools import BaseTool as CrewBaseTool
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(url, decode_responses=True)
    return _redis_client


class BaseTool(CrewBaseTool):
    """Base class for all stock-agent tools.

    Subclasses must declare :attr:`rate_limit_key` and
    :attr:`rate_limit_per_minute`, then implement :meth:`_run`.

    Example::

        class MarketDataTool(BaseTool):
            name: str = "market_data"
            description: str = "Fetch live NSE quotes"
            rate_limit_key: ClassVar[str] = "nse_data"
            rate_limit_per_minute: ClassVar[int] = 10

            def _run(self, symbol: str) -> dict[str, Any]:
                ...
    """

    rate_limit_key: ClassVar[str] = "default"
    rate_limit_per_minute: ClassVar[int] = 60

    def _check_rate_limit(self) -> None:
        """Raise RuntimeError if the per-minute request budget is exhausted."""
        try:
            r = _get_redis()
            redis_key = f"ratelimit:{self.rate_limit_key}"
            pipe = r.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, 60)
            count, _ = pipe.execute()
            if count > self.rate_limit_per_minute:
                wait_secs = r.ttl(redis_key)
                raise RuntimeError(
                    f"Rate limit exceeded for '{self.rate_limit_key}'. "
                    f"Retry in {wait_secs}s."
                )
        except redis.RedisError as exc:
            # Degrade gracefully — log and continue without rate limiting.
            log.warning("rate_limit.redis_error", error=str(exc))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Entry point called by CrewAI. Applies rate limiting then delegates."""
        self._check_rate_limit()
        return self._run(*args, **kwargs)

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Implement the tool's core logic here.

        Use httpx.AsyncClient (not requests) for HTTP calls.
        Cache responses in Redis: market data TTL=60s, fundamentals TTL=3600s.
        """
