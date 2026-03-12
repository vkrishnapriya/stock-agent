"""
src/broker/paper_broker.py
Paper-trading broker backed by yfinance quotes and PostgreSQL persistence.

Order fills are simulated immediately with ±0.1 % slippage.
The DATABASE_URL setting is used; asyncpg is replaced with psycopg2 for the
synchronous SQLAlchemy engine used here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytz
import structlog
import yfinance as yf
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.broker.base_broker import BaseBroker, OrderRejectedError, OrderType
from src.config.market_hours import round_to_tick
from src.config.settings import get_settings
from src.models.portfolio import Holding, Portfolio

log = structlog.get_logger(__name__)

_IST = pytz.timezone("Asia/Kolkata")
_SLIPPAGE = 0.001  # 0.1 %

# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class _PaperOrder(_Base):
    __tablename__ = "paper_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(36), nullable=False, unique=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(5), nullable=False, default="NSE")
    quantity = Column(Integer, nullable=False)
    order_type = Column(String(20), nullable=False)
    requested_price = Column(Float, nullable=True)
    fill_price = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="COMPLETE")
    created_at = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------


class PaperBroker(BaseBroker):
    """Simulated broker for back-testing and safe development.

    * Quotes sourced from yfinance (delayed ~15 min for NSE).
    * Orders fill instantly at market price ± 0.1 % slippage.
    * All fills persisted to PostgreSQL via SQLAlchemy (sync / psycopg2).
    """

    def __init__(self) -> None:
        super().__init__()
        self._engine = self._build_engine()
        self._Session = sessionmaker(bind=self._engine)
        _Base.metadata.create_all(self._engine)
        self._log.info("paper_broker.ready")

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    @staticmethod
    def _build_engine():
        url = get_settings().database_url
        # asyncpg is async-only; swap to psycopg2 for the sync engine.
        url = url.replace("postgresql+asyncpg", "postgresql+psycopg2")
        url = url.replace("postgresql+asyncpg://", "postgresql://")
        return create_engine(url, pool_pre_ping=True, future=True)

    # ------------------------------------------------------------------
    # Quote
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        """Fetch delayed quote from Yahoo Finance (NSE suffix = .NS, BSE = .BO)."""
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
        yf_symbol = f"{symbol.upper()}{suffix}"
        self._log.debug("paper_broker.get_quote", symbol=yf_symbol)

        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                raise ValueError(f"yfinance returned no data for {yf_symbol}")
            row = hist.iloc[-1]
            return {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "last_price": round_to_tick(float(row["Close"])),
                "open": round_to_tick(float(row["Open"])),
                "high": round_to_tick(float(row["High"])),
                "low": round_to_tick(float(row["Low"])),
                "close": round_to_tick(float(hist.iloc[-2]["Close"]) if len(hist) >= 2 else float(row["Close"])),
                "volume": int(row["Volume"]),
            }
        except Exception as exc:
            self._log.error("paper_broker.get_quote.failed", symbol=yf_symbol, error=str(exc))
            raise

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
        if qty <= 0:
            raise OrderRejectedError(f"qty must be > 0, got {qty}")

        # Determine fill price
        if order_type.is_market:
            quote = self.get_quote(symbol, exchange)
            base_price = quote["last_price"]
        else:
            if price <= 0:
                raise OrderRejectedError(f"LIMIT order requires price > 0, got {price}")
            base_price = price

        slippage = _SLIPPAGE if order_type.is_buy else -_SLIPPAGE
        fill_price = round_to_tick(base_price * (1 + slippage))

        order_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        order = _PaperOrder(
            order_id=order_id,
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            quantity=qty,
            order_type=order_type.value,
            requested_price=price if not order_type.is_market else None,
            fill_price=fill_price,
            status="COMPLETE",
            created_at=now,
        )

        with self._Session() as session:
            session.add(order)
            session.commit()

        self._log.info(
            "paper_broker.order_filled",
            order_id=order_id,
            symbol=symbol,
            qty=qty,
            order_type=order_type.value,
            fill_price=fill_price,
        )
        return order_id

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_portfolio(self) -> Portfolio:
        """Reconstruct portfolio by aggregating all completed paper orders."""
        with self._Session() as session:
            rows = session.execute(
                text(
                    "SELECT symbol, exchange, order_type, quantity, fill_price "
                    "FROM paper_orders WHERE status = 'COMPLETE' ORDER BY created_at"
                )
            ).fetchall()

        # Aggregate net quantities and weighted avg buy price per symbol
        positions: dict[str, dict[str, Any]] = {}
        for symbol, exchange, order_type, qty, fill_price in rows:
            key = f"{symbol}:{exchange}"
            if key not in positions:
                positions[key] = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "net_qty": 0,
                    "total_cost": 0.0,
                    "total_buy_qty": 0,
                }
            pos = positions[key]
            if order_type.startswith("BUY"):
                pos["net_qty"] += qty
                pos["total_cost"] += fill_price * qty
                pos["total_buy_qty"] += qty
            else:
                pos["net_qty"] -= qty

        holdings: list[Holding] = []
        for pos in positions.values():
            if pos["net_qty"] <= 0:
                continue
            avg_price = (
                pos["total_cost"] / pos["total_buy_qty"]
                if pos["total_buy_qty"] > 0
                else 0.0
            )
            holdings.append(
                Holding(
                    symbol=pos["symbol"],
                    exchange=pos["exchange"],
                    quantity=pos["net_qty"],
                    avg_buy_price=round(avg_price, 2),
                )
            )

        return Portfolio(
            portfolio_id="paper",
            holdings=holdings,
            available_cash_inr=0.0,  # cash tracking not implemented in paper mode
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Return today's paper orders as open position dicts."""
        with self._Session() as session:
            rows = session.execute(
                text(
                    "SELECT order_id, symbol, exchange, order_type, quantity, fill_price, created_at "
                    "FROM paper_orders WHERE status = 'COMPLETE' "
                    "AND created_at::date = CURRENT_DATE ORDER BY created_at DESC"
                )
            ).fetchall()

        return [
            {
                "order_id": r[0],
                "symbol": r[1],
                "exchange": r[2],
                "order_type": r[3],
                "quantity": r[4],
                "fill_price": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]
