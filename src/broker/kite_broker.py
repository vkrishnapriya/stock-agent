"""
src/broker/kite_broker.py
Live broker implementation using KiteConnect (Zerodha).

* Reads credentials from Settings (KITE_API_KEY, KITE_ACCESS_TOKEN).
* Raises NotMarketHoursError if called outside 9:15–15:30 IST.
* Enforces Kite's 3 req/sec limit on order APIs via a Redis sliding window.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pytz
import redis
import structlog
from kiteconnect import KiteConnect  # type: ignore[import-untyped]

from src.broker.base_broker import (
    BaseBroker,
    NotMarketHoursError,
    OrderRejectedError,
    OrderType,
)
from src.config.market_hours import IST, MARKET_CLOSE, MARKET_OPEN
from src.config.settings import get_settings
from src.models.portfolio import Holding, Portfolio

log = structlog.get_logger(__name__)

_KITE_ORDER_RATE_LIMIT = 3   # requests per second
_KITE_DATA_RATE_LIMIT  = 10  # requests per second (not enforced here — tools handle it)

# Kite product / variety constants
_PRODUCT_CNC   = "CNC"    # cash-and-carry (delivery)
_VARIETY_REG   = "regular"
_EXCHANGE_NSE  = "NSE"
_EXCHANGE_BSE  = "BSE"


class KiteBroker(BaseBroker):
    """Live broker backed by Zerodha KiteConnect v5.

    Access token is read from ``KITE_ACCESS_TOKEN`` (refreshed daily via
    ``scripts/setup_kite.py``).  The token is *never* stored in code.
    """

    def __init__(self) -> None:
        super().__init__()
        settings = get_settings()
        self._kite = KiteConnect(api_key=settings.kite_api_key)
        self._kite.set_access_token(settings.kite_access_token)
        self._redis = self._build_redis()
        self._log.info("kite_broker.ready")

    # ------------------------------------------------------------------
    # Redis helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_redis() -> redis.Redis:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        return redis.from_url(url, decode_responses=True)

    # ------------------------------------------------------------------
    # Market hours guard
    # ------------------------------------------------------------------

    def _assert_market_open(self) -> None:
        """Raise NotMarketHoursError if outside 9:15–15:30 IST."""
        now_ist = datetime.now(IST).time()
        if not (MARKET_OPEN <= now_ist < MARKET_CLOSE):
            raise NotMarketHoursError(
                f"Market is closed at {now_ist} IST. "
                f"Trading hours: {MARKET_OPEN}–{MARKET_CLOSE} IST."
            )

    # ------------------------------------------------------------------
    # Order rate limiter (3 req/sec)
    # ------------------------------------------------------------------

    def _check_order_rate_limit(self) -> None:
        """Enforce Kite's 3-requests-per-second limit on order APIs."""
        try:
            key = f"ratelimit:kite_order:{int(time.time())}"
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 2)  # key lives for 2s to cover clock skew
            count, _ = pipe.execute()
            if count > _KITE_ORDER_RATE_LIMIT:
                raise OrderRejectedError(
                    f"Kite order rate limit exceeded ({_KITE_ORDER_RATE_LIMIT} req/sec). "
                    "Retry after 1 second."
                )
        except redis.RedisError as exc:
            # Degrade gracefully — log and continue.
            self._log.warning("kite_broker.rate_limit.redis_error", error=str(exc))

    # ------------------------------------------------------------------
    # Quote
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        """Fetch live quote from Kite for *symbol* on *exchange*."""
        instrument = f"{exchange.upper()}:{symbol.upper()}"
        self._log.debug("kite_broker.get_quote", instrument=instrument)

        raw = self._kite.quote([instrument])
        data = raw.get(instrument)
        if not data:
            raise ValueError(f"No quote data returned for {instrument}")

        ohlc = data.get("ohlc", {})
        return {
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "last_price": float(data["last_price"]),
            "open": float(ohlc.get("open", 0)),
            "high": float(ohlc.get("high", 0)),
            "low": float(ohlc.get("low", 0)),
            "close": float(ohlc.get("close", 0)),
            "volume": int(data.get("volume", 0)),
        }

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        qty: int,
        order_type: OrderType,
        price: float,
        exchange: str = "NSE",
    ) -> str:
        """Place a live order via KiteConnect.

        Raises:
            NotMarketHoursError: If called outside 9:15–15:30 IST.
            OrderRejectedError:  On rate-limit breach or broker rejection.
        """
        self._assert_market_open()
        self._check_order_rate_limit()

        if qty <= 0:
            raise OrderRejectedError(f"qty must be > 0, got {qty}")

        transaction_type = (
            self._kite.TRANSACTION_TYPE_BUY
            if order_type.is_buy
            else self._kite.TRANSACTION_TYPE_SELL
        )
        kite_order_type = (
            self._kite.ORDER_TYPE_MARKET
            if order_type.is_market
            else self._kite.ORDER_TYPE_LIMIT
        )

        self._log.info(
            "kite_broker.place_order",
            symbol=symbol,
            qty=qty,
            order_type=order_type.value,
            price=price,
            exchange=exchange,
        )

        try:
            order_id = self._kite.place_order(
                variety=_VARIETY_REG,
                exchange=exchange.upper(),
                tradingsymbol=symbol.upper(),
                transaction_type=transaction_type,
                quantity=qty,
                product=_PRODUCT_CNC,
                order_type=kite_order_type,
                price=price if not order_type.is_market else None,
            )
        except Exception as exc:
            self._log.error("kite_broker.place_order.failed", error=str(exc))
            raise OrderRejectedError(str(exc)) from exc

        self._log.info("kite_broker.order_placed", order_id=order_id)
        return str(order_id)

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_portfolio(self) -> Portfolio:
        """Build a Portfolio from KiteConnect holdings."""
        raw_holdings = self._kite.holdings()
        holdings: list[Holding] = []

        for h in raw_holdings:
            qty = int(h.get("quantity", 0))
            if qty <= 0:
                continue
            holdings.append(
                Holding(
                    symbol=str(h["tradingsymbol"]),
                    exchange=str(h.get("exchange", "NSE")),
                    isin=str(h.get("isin", "")),
                    quantity=qty,
                    avg_buy_price=float(h.get("average_price", 0.0)),
                    current_price=float(h.get("last_price", 0.0)) or None,
                )
            )

        return Portfolio(
            portfolio_id="kite",
            holdings=holdings,
            available_cash_inr=0.0,  # fetch via kite.margins() if needed
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Return today's net positions from KiteConnect."""
        positions = self._kite.positions()
        return positions.get("net", [])
