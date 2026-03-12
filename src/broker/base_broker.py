"""
src/broker/base_broker.py
Abstract broker interface and shared exceptions / enums.

All concrete broker implementations MUST extend BaseBroker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import structlog

from src.models.portfolio import Portfolio

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NotMarketHoursError(Exception):
    """Raised when a live-broker method is called outside NSE trading hours."""


class OrderRejectedError(Exception):
    """Raised when the broker rejects an order (e.g. insufficient margin)."""


# ---------------------------------------------------------------------------
# Order type enum
# ---------------------------------------------------------------------------


class OrderType(str, Enum):
    """Encodes both transaction direction and execution type in one value."""

    BUY_MARKET = "BUY_MARKET"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_MARKET = "SELL_MARKET"
    SELL_LIMIT = "SELL_LIMIT"

    @property
    def is_buy(self) -> bool:
        return self.value.startswith("BUY")

    @property
    def is_market(self) -> bool:
        return self.value.endswith("MARKET")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseBroker(ABC):
    """Common interface for all broker implementations.

    Usage::

        broker = get_broker()           # from src.broker
        quote  = broker.get_quote("INFY")
        oid    = broker.place_order("INFY", 10, OrderType.BUY_MARKET, 0.0)
        pf     = broker.get_portfolio()
        pos    = broker.get_positions()
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        """Return a live or simulated market quote for *symbol*.

        Returns a dict with at least::

            {
                "symbol":      str,
                "exchange":    str,
                "last_price":  float,       # INR
                "open":        float,
                "high":        float,
                "low":         float,
                "close":       float,       # previous close
                "volume":      int,
            }
        """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: int,
        order_type: OrderType,
        price: float,
        exchange: str = "NSE",
    ) -> str:
        """Submit an order.

        Args:
            symbol:     NSE/BSE ticker symbol (e.g. "RELIANCE").
            qty:        Number of shares (must be > 0).
            order_type: One of the :class:`OrderType` enum values.
            price:      Limit price in INR; ignored for MARKET orders.
            exchange:   "NSE" (default) or "BSE".

        Returns:
            Broker-assigned order_id string.

        Raises:
            NotMarketHoursError: If the market is closed (live broker only).
            OrderRejectedError:  If the broker rejects the order.
        """

    @abstractmethod
    def get_portfolio(self) -> Portfolio:
        """Return the current portfolio aggregated from all filled orders."""

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Return intraday open positions as a list of raw position dicts."""
