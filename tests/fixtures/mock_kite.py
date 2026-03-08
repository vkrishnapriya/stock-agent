"""
tests/fixtures/mock_kite.py
Mock KiteConnect client for unit tests. No live API calls are made.
Import :func:`mock_kite_connect` as a pytest fixture.

Usage::

    from tests.fixtures.mock_kite import mock_kite_connect

    def test_order_placement(mock_kite_connect):
        # mock_kite_connect is a MagicMock pre-configured with realistic responses
        assert mock_kite_connect.place_order.called
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


def _build_quote(symbol: str, last_price: float) -> dict[str, Any]:
    return {
        "instrument_token": hash(symbol) & 0xFFFF,
        "last_price": last_price,
        "ohlc": {
            "open":  last_price * 0.99,
            "high":  last_price * 1.01,
            "low":   last_price * 0.98,
            "close": last_price * 0.995,
        },
        "volume": 1_000_000,
        "average_price": last_price,
        "buy_quantity": 5_000,
        "sell_quantity": 4_800,
    }


@pytest.fixture
def mock_kite_connect() -> MagicMock:
    """Return a MagicMock that mimics a logged-in KiteConnect instance."""
    kite = MagicMock(name="KiteConnect")

    # Auth
    kite.access_token = "mock_access_token_abc123"
    kite.login_url.return_value = "https://kite.trade/connect/login?api_key=mock"

    # Quotes
    kite.quote.return_value = {
        "NSE:RELIANCE": _build_quote("RELIANCE", 2500.00),
        "NSE:INFY":     _build_quote("INFY",     1850.50),
        "NSE:TCS":      _build_quote("TCS",      4050.75),
    }

    # Orders
    kite.place_order.return_value = "mock_order_id_001"
    kite.cancel_order.return_value = "mock_order_id_001"
    kite.orders.return_value = [
        {
            "order_id":     "mock_order_id_001",
            "status":       "COMPLETE",
            "tradingsymbol": "RELIANCE",
            "exchange":     "NSE",
            "transaction_type": "BUY",
            "quantity":     10,
            "price":        2500.00,
            "filled_quantity": 10,
        }
    ]

    # Positions
    kite.positions.return_value = {
        "net": [],
        "day": [],
    }

    # Holdings
    kite.holdings.return_value = []

    # Instruments
    kite.instruments.return_value = []

    return kite
