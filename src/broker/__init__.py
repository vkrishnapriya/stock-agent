"""
src/broker/__init__.py
Broker factory — returns the correct implementation based on BROKER_MODE setting.

Usage::

    from src.broker import get_broker
    broker = get_broker()
"""

from __future__ import annotations

import structlog

from src.broker.base_broker import BaseBroker, NotMarketHoursError, OrderRejectedError, OrderType

log = structlog.get_logger(__name__)


def get_broker() -> BaseBroker:
    """Return a broker instance appropriate for the current BROKER_MODE.

    * ``paper`` → :class:`~src.broker.paper_broker.PaperBroker`
      (yfinance quotes, PostgreSQL persistence, simulated fills)
    * ``live``  → :class:`~src.broker.kite_broker.KiteBroker`
      (real KiteConnect orders, Redis rate limiting)

    Raises:
        ValueError: If BROKER_MODE is set to an unrecognised value.
    """
    from src.config.settings import get_settings

    mode = get_settings().broker_mode
    log.info("broker.factory", mode=mode)

    if mode == "paper":
        from src.broker.paper_broker import PaperBroker
        return PaperBroker()

    if mode == "live":
        from src.broker.kite_broker import KiteBroker
        return KiteBroker()

    raise ValueError(f"Unknown BROKER_MODE '{mode}'. Expected 'paper' or 'live'.")


__all__ = [
    "get_broker",
    "BaseBroker",
    "OrderType",
    "NotMarketHoursError",
    "OrderRejectedError",
]
