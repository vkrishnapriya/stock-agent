"""
src/utils/rate_limiter.py
Async Redis sliding-window rate limiter.

Each RateLimiter tracks calls within a rolling time window using a Redis
sorted-set (score = UNIX timestamp, member = unique request ID).

Usage::

    allowed = await KITE_ORDERS.acquire()
    if not allowed:
        raise RateLimitError("Kite order rate limit exceeded")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import redis.asyncio as aioredis
import structlog

log = structlog.get_logger(__name__)


class RateLimitError(Exception):
    """Raised when an acquire() call exceeds the configured limit."""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


@dataclass
class RateLimiter:
    """Async sliding-window rate limiter backed by Redis sorted sets.

    Args:
        key:            Redis key prefix (e.g. ``"kite_orders"``).
        limit:          Maximum number of requests allowed within *window_seconds*.
        window_seconds: Length of the sliding window in seconds.
    """

    key: str
    limit: int
    window_seconds: float
    _client: aioredis.Redis | None = field(default=None, repr=False, compare=False)

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            import os
            url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            self._client = aioredis.from_url(url, decode_responses=True)
        return self._client

    async def acquire(self) -> bool:
        """Attempt to consume one slot in the rate-limit window.

        Returns:
            ``True``  — request is allowed (slot consumed).
            ``False`` — limit exceeded; caller should back off.

        Redis errors are caught and logged; the method returns ``True``
        (allow-through) so a Redis outage never silently blocks all traffic.
        """
        redis_key = f"ratelimit:sliding:{self.key}"
        now = time.time()
        window_start = now - self.window_seconds
        member = str(uuid.uuid4())

        try:
            client = await self._get_client()
            pipe = client.pipeline()
            # Remove requests outside the window
            pipe.zremrangebyscore(redis_key, 0, window_start)
            # Count current window
            pipe.zcard(redis_key)
            # Register this request
            pipe.zadd(redis_key, {member: now})
            # Set expiry slightly beyond the window
            pipe.expire(redis_key, int(self.window_seconds) + 2)
            results = await pipe.execute()

            current_count: int = results[1]  # zcard result (before adding new entry)
            if current_count >= self.limit:
                # Remove the just-added member since we're rejecting
                await client.zrem(redis_key, member)
                log.warning(
                    "rate_limiter.exceeded",
                    key=self.key,
                    count=current_count,
                    limit=self.limit,
                    window_seconds=self.window_seconds,
                )
                return False

            log.debug(
                "rate_limiter.acquired",
                key=self.key,
                count=current_count + 1,
                limit=self.limit,
            )
            return True

        except Exception as exc:
            log.warning("rate_limiter.redis_error", key=self.key, error=str(exc))
            return True  # degrade gracefully

    async def remaining(self) -> int:
        """Return how many slots are still available in the current window."""
        redis_key = f"ratelimit:sliding:{self.key}"
        now = time.time()
        window_start = now - self.window_seconds
        try:
            client = await self._get_client()
            await client.zremrangebyscore(redis_key, 0, window_start)
            used = await client.zcard(redis_key)
            return max(0, self.limit - used)
        except Exception as exc:
            log.warning("rate_limiter.redis_error", key=self.key, error=str(exc))
            return self.limit


# ---------------------------------------------------------------------------
# Predefined limiters (module-level singletons)
# ---------------------------------------------------------------------------

#: Kite data APIs — 10 requests per second
KITE_DATA = RateLimiter(key="kite_data", limit=10, window_seconds=1.0)

#: Kite order APIs — 3 requests per second
KITE_ORDERS = RateLimiter(key="kite_orders", limit=3, window_seconds=1.0)

#: Gemini API — 15 requests per minute
GEMINI = RateLimiter(key="gemini", limit=15, window_seconds=60.0)

#: GNews API — 100 requests per day
GNEWS = RateLimiter(key="gnews", limit=100, window_seconds=86_400.0)

#: Tavily API — 33 requests per day
TAVILY = RateLimiter(key="tavily", limit=33, window_seconds=86_400.0)

__all__ = [
    "RateLimiter",
    "RateLimitError",
    "KITE_DATA",
    "KITE_ORDERS",
    "GEMINI",
    "GNEWS",
    "TAVILY",
]
