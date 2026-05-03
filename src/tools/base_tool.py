"""
src/tools/base_tool.py
Abstract base class for all trading tools.

All tools MUST extend BaseTool and implement :meth:`_run`.
Provides:
  * Async sliding-window rate limiting via :class:`~src.utils.rate_limiter.RateLimiter`.
  * Redis-backed response cache via :meth:`_get_cached` / :meth:`_set_cached`.
  * Structured error logging on every failure.
  * A sync→async bridge in :meth:`run` so CrewAI can call tools transparently.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import os
from abc import abstractmethod
from typing import Any, ClassVar

import redis.asyncio as aioredis
import structlog
from crewai.tools import BaseTool as CrewBaseTool

from src.utils.rate_limiter import RateLimitError, RateLimiter

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level shared Redis client (one connection pool for all tools)
# ---------------------------------------------------------------------------

_async_redis: aioredis.Redis | None = None
_rate_limiters: dict[str, RateLimiter] = {}


def _get_async_redis() -> aioredis.Redis:
    global _async_redis
    if _async_redis is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _async_redis = aioredis.from_url(url, decode_responses=True)
    return _async_redis


def _get_rate_limiter(key: str, limit: int, window_seconds: float) -> RateLimiter:
    """Return a cached RateLimiter instance for the given key+limit combination."""
    cache_key = f"{key}:{limit}:{window_seconds}"
    if cache_key not in _rate_limiters:
        _rate_limiters[cache_key] = RateLimiter(
            key=key, limit=limit, window_seconds=window_seconds
        )
    return _rate_limiters[cache_key]


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------


class BaseTool(CrewBaseTool):
    """Base class for all stock-agent tools.

    Subclasses must declare :attr:`rate_limit_key` and
    :attr:`rate_limit_per_minute`, then implement :meth:`_run`.

    Example::

        class NSEQuoteTool(BaseTool):
            name: str = "nse_quote"
            description: str = "Fetch a live NSE market quote"
            rate_limit_key: ClassVar[str] = "kite_data"
            rate_limit_per_minute: ClassVar[int] = 600   # 10/sec × 60

            def _run(self, symbol: str) -> dict[str, Any]:
                cached = asyncio.run(self._get_cached(symbol))
                if cached:
                    return cached
                data = ...fetch...
                asyncio.run(self._set_cached(symbol, data, ttl=60))
                return data
    """

    rate_limit_key: ClassVar[str] = "default"
    rate_limit_per_minute: ClassVar[int] = 60

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _check_rate_limit(self) -> None:
        """Raise :class:`~src.utils.rate_limiter.RateLimitError` if the
        per-minute request budget for this tool is exhausted.

        Uses a Redis sliding-window counter so the limit is accurate even
        across multiple processes.
        """
        limiter = _get_rate_limiter(
            key=self.rate_limit_key,
            limit=self.rate_limit_per_minute,
            window_seconds=60.0,
        )
        allowed = await limiter.acquire()
        if not allowed:
            raise RateLimitError(
                f"Rate limit exceeded for tool '{self.name}' "
                f"(key='{self.rate_limit_key}', "
                f"limit={self.rate_limit_per_minute}/min)"
            )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def _get_cached(self, key: str) -> Any | None:
        """Return a previously cached value from Redis, or ``None``.

        Args:
            key: Cache key (scoped automatically to this tool's
                 :attr:`rate_limit_key`).

        Returns:
            The deserialised Python value, or ``None`` if absent or expired.
        """
        redis_key = f"cache:{self.rate_limit_key}:{key}"
        try:
            raw = await _get_async_redis().get(redis_key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            log.warning("tool.cache.get_failed", tool=self.name, key=redis_key, error=str(exc))
            return None

    async def _set_cached(self, key: str, value: Any, ttl: int) -> None:
        """Persist *value* to Redis under *key* for *ttl* seconds.

        Args:
            key:   Cache key (scoped to this tool's :attr:`rate_limit_key`).
            value: JSON-serialisable object to cache.
            ttl:   Time-to-live in seconds.
                   Recommended values: market data = 60, fundamentals = 3600.
        """
        redis_key = f"cache:{self.rate_limit_key}:{key}"
        try:
            await _get_async_redis().setex(
                redis_key,
                ttl,
                json.dumps(value, default=str),
            )
        except Exception as exc:
            log.warning("tool.cache.set_failed", tool=self.name, key=redis_key, error=str(exc))

    # ------------------------------------------------------------------
    # Async execution pipeline
    # ------------------------------------------------------------------

    async def _execute(self, *args: Any, **kwargs: Any) -> Any:
        """Async pipeline: rate-limit check → :meth:`_run` → error logging.

        Both sync and async implementations of :meth:`_run` are supported.
        """
        await self._check_rate_limit()
        try:
            result = self._run(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except RateLimitError:
            raise  # already logged by rate limiter
        except Exception as exc:
            log.error(
                "tool.run_failed",
                tool=self.name,
                args=args,
                kwargs=kwargs,
                error=str(exc),
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Sync entry point (called by CrewAI)
    # ------------------------------------------------------------------

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Sync entry point called by CrewAI.

        Bridges into the async :meth:`_execute` pipeline.  Works whether
        or not there is already a running event loop (e.g. Jupyter / tests).

        If a single string argument is passed and ``args_schema`` is defined,
        the string is parsed as JSON and unpacked into keyword arguments so
        that ``_run`` receives individual named parameters.
        """
        # Parse JSON string input through args_schema when available
        if args and isinstance(args[0], str) and not kwargs:
            schema = getattr(self, "args_schema", None)
            if schema is not None:
                try:
                    raw = args[0].strip()
                    payload = json.loads(raw) if raw.startswith("{") else {"symbol": raw}
                    validated = schema(**payload)
                    kwargs = validated.model_dump()
                    args = ()
                except Exception:
                    pass  # fall through with original args

        try:
            asyncio.get_running_loop()
            # Already inside a running loop — delegate to a worker thread
            # that owns its own event loop.
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                future = pool.submit(asyncio.run, self._execute(*args, **kwargs))
                return future.result()
        except RuntimeError:
            # No running loop — safe to call asyncio.run directly.
            return asyncio.run(self._execute(*args, **kwargs))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Implement the tool's core logic.

        May be sync or async — both are handled by :meth:`_execute`.

        Guidelines:
        * Use ``await self._get_cached(key)`` / ``await self._set_cached(key, value, ttl)``
          for response caching.
        * Use ``httpx.AsyncClient`` for HTTP calls (never ``requests``).
        * Cache TTLs: market data = 60 s, fundamentals = 3600 s.
        """
