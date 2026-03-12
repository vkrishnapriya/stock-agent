"""
src/utils/retry.py
@retry decorator with exponential back-off for httpx and rate-limit errors.

Usage::

    from src.utils.retry import retry, RateLimitError

    @retry(max_attempts=3, backoff_factor=2.0)
    async def fetch_data(url: str) -> dict:
        ...

    @retry(max_attempts=5, backoff_factor=1.5, exceptions=(ValueError,))
    def sync_call() -> str:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
import structlog

log = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Re-export so callers import from one place
from src.utils.rate_limiter import RateLimitError  # noqa: E402

_DEFAULT_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.HTTPError,
    httpx.TimeoutException,
    RateLimitError,
)


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = _DEFAULT_EXCEPTIONS,
    base_delay: float = 1.0,
) -> Callable[[F], F]:
    """Decorator that retries a function on transient errors with exponential back-off.

    Args:
        max_attempts:   Total number of attempts (1 = no retries).
        backoff_factor: Multiplier applied to the delay on each failure.
                        Delay after attempt *n* = ``base_delay * backoff_factor ** (n-1)``.
        exceptions:     Tuple of exception types that trigger a retry.
                        Defaults to ``(httpx.HTTPError, httpx.TimeoutException, RateLimitError)``.
        base_delay:     Delay in seconds before the first retry.

    The decorator transparently supports both **sync** and **async** functions.
    """

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: Exception | None = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await fn(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt == max_attempts:
                            log.error(
                                "retry.exhausted",
                                fn=fn.__qualname__,
                                attempt=attempt,
                                max_attempts=max_attempts,
                                error=str(exc),
                            )
                            raise
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        log.warning(
                            "retry.attempt_failed",
                            fn=fn.__qualname__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay_seconds=round(delay, 2),
                            error=str(exc),
                        )
                        await asyncio.sleep(delay)
                raise RuntimeError("unreachable")  # pragma: no cover

            return async_wrapper  # type: ignore[return-value]

        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: Exception | None = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return fn(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt == max_attempts:
                            log.error(
                                "retry.exhausted",
                                fn=fn.__qualname__,
                                attempt=attempt,
                                max_attempts=max_attempts,
                                error=str(exc),
                            )
                            raise
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        log.warning(
                            "retry.attempt_failed",
                            fn=fn.__qualname__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay_seconds=round(delay, 2),
                            error=str(exc),
                        )
                        time.sleep(delay)
                raise RuntimeError("unreachable")  # pragma: no cover

            return sync_wrapper  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


__all__ = ["retry", "RateLimitError"]
