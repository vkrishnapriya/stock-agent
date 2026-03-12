"""
src/models/scan.py
Pydantic contracts for MarketScannerAgent output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pytz
from pydantic import BaseModel, Field, field_validator

_IST = pytz.timezone("Asia/Kolkata")


def _now_ist() -> datetime:
    return datetime.now(_IST)


class ScanEntry(BaseModel):
    """A single stock returned by the market screener."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    sector: str = ""
    momentum_1m_pct: float = Field(description="1-month price return %")
    volume_ratio: float = Field(ge=0.0, description="Current volume vs 20-day average")
    last_price: float = Field(gt=0.0, description="Latest closing price in INR")
    rank: int = Field(ge=1, description="Rank within the screened universe (1 = best)")

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()


class ScanResults(BaseModel):
    """Output of a :class:`~src.agents.market_scanner.MarketScannerAgent` scan."""

    universe: str = Field(description="Universe name: NIFTY50, NIFTY100, NIFTY500")
    entries: list[ScanEntry] = Field(default_factory=list)
    total_scanned: int = Field(ge=0, description="Number of symbols screened")
    generated_at: datetime = Field(default_factory=_now_ist)

    @property
    def symbols(self) -> list[str]:
        """All symbols in current scan order."""
        return [e.symbol for e in self.entries]

    def shortlist(self, n: int = 15) -> list[ScanEntry]:
        """Return top-*n* entries sorted by descending 1-month momentum."""
        return sorted(self.entries, key=lambda e: e.momentum_1m_pct, reverse=True)[:n]
