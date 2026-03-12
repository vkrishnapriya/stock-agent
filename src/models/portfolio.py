"""
src/models/portfolio.py
Pydantic data contracts for portfolio and holdings.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Holding(BaseModel):
    """A single stock position in a portfolio."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    isin: str = ""
    quantity: int = Field(ge=0)
    avg_buy_price: float = Field(gt=0.0)
    current_price: float | None = None
    sector: str = ""

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()

    @property
    def market_value_inr(self) -> float | None:
        """Current market value; None when current_price is not set."""
        if self.current_price is None:
            return None
        return self.quantity * self.current_price

    @property
    def unrealised_pnl_inr(self) -> float | None:
        """Unrealised P&L; None when current_price is not set."""
        if self.current_price is None:
            return None
        return (self.current_price - self.avg_buy_price) * self.quantity

    @property
    def unrealised_pnl_pct(self) -> float | None:
        """Unrealised P&L as percentage of cost; None when current_price not set."""
        if self.current_price is None:
            return None
        return ((self.current_price - self.avg_buy_price) / self.avg_buy_price) * 100


class Portfolio(BaseModel):
    """Aggregate view of a portfolio with multiple holdings and cash balance."""

    portfolio_id: str
    holdings: list[Holding] = Field(default_factory=list)
    available_cash_inr: float = Field(default=0.0, ge=0.0)

    @property
    def total_invested_inr(self) -> float:
        return sum(h.quantity * h.avg_buy_price for h in self.holdings)

    @property
    def total_market_value_inr(self) -> float | None:
        """Returns None if any holding is missing current_price."""
        values = [h.market_value_inr for h in self.holdings]
        if any(v is None for v in values):
            return None
        return sum(values)  # type: ignore[arg-type]

    @property
    def total_value_inr(self) -> float | None:
        """Market value + cash; None if market values unavailable."""
        mv = self.total_market_value_inr
        if mv is None:
            return None
        return mv + self.available_cash_inr

    @classmethod
    def load_from_file(cls, path: str | Path) -> "Portfolio":
        """Load a Portfolio from a JSON or CSV file.

        JSON — expects the full portfolio structure (see data/portfolios/sample_portfolio.json).
        CSV  — expects columns: symbol, exchange, quantity, avg_buy_price, sector
               (optional: isin, current_price). portfolio_id defaults to the file stem;
               available_cash_inr defaults to 0.0.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Portfolio file not found: {p}")

        suffix = p.suffix.lower()

        if suffix == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
            # Normalise key differences between JSON sample and model field names
            if "cash_balance_inr" in raw and "available_cash_inr" not in raw:
                raw["available_cash_inr"] = raw.pop("cash_balance_inr")
            # Strip keys that are not on the model (extra="ignore" not set here)
            return cls.model_validate(raw)

        if suffix == ".csv":
            holdings: list[Holding] = []
            with p.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    holdings.append(
                        Holding(
                            symbol=row["symbol"],
                            exchange=row.get("exchange", "NSE"),  # type: ignore[arg-type]
                            isin=row.get("isin", ""),
                            quantity=int(row["quantity"]),
                            avg_buy_price=float(row["avg_buy_price"]),
                            current_price=(
                                float(row["current_price"])
                                if row.get("current_price")
                                else None
                            ),
                            sector=row.get("sector", ""),
                        )
                    )
            return cls(portfolio_id=p.stem, holdings=holdings)

        raise ValueError(f"Unsupported file format '{suffix}'. Use .json or .csv.")
