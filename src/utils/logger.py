"""
src/utils/logger.py
Centralised structlog configuration.

* Development (LOG_LEVEL=DEBUG): colourised human-readable console output.
* Production (any other level): JSON lines to stdout.

Call ``configure_logging()`` once at application startup (e.g. in main.py).
The function is idempotent — safe to call multiple times.

Usage::

    from src.utils.logger import configure_logging, get_logger

    configure_logging()            # once at startup
    log = get_logger(__name__)
    log.info("server.started", port=8080)
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(log_level: str | None = None) -> None:
    """Configure structlog globally.

    Args:
        log_level: Override the ``LOG_LEVEL`` setting. One of
                   ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
                   Defaults to the value in :func:`~src.config.settings.get_settings`.
    """
    global _configured

    if log_level is None:
        try:
            from src.config.settings import get_settings
            log_level = get_settings().log_level
        except Exception:
            log_level = "INFO"

    level = getattr(logging, log_level.upper(), logging.INFO)
    is_debug = level == logging.DEBUG

    # ------------------------------------------------------------------
    # stdlib root logger
    # ------------------------------------------------------------------
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # Shared processors (always applied)
    # ------------------------------------------------------------------
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
    ]

    # ------------------------------------------------------------------
    # Renderer: pretty for dev, JSON for production
    # ------------------------------------------------------------------
    if is_debug:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.filter_by_level,
            structlog.processors.format_exc_info,
            structlog.stdlib.PositionalArgumentsFormatter(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structlog logger bound to *name*.

    Calls :func:`configure_logging` with defaults if it has not been called yet.

    Args:
        name: Typically ``__name__`` of the calling module.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
