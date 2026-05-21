"""
src/models/recommendations.py
Pydantic contracts for agent-generated trading recommendations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pytz
from pydantic import BaseModel, Field, field_validator, model_validator

_IST = pytz.timezone("Asia/Kolkata")


def _now_ist() -> datetime:
    return datetime.now(_IST)


class StockDecision(BaseModel):
    """Hold or Sell decision produced by the Seller workflow for an existing holding."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    action: Literal["SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 = no confidence, 1.0 = certain")
    rationale: str = Field(min_length=10, description="Human-readable rationale for the decision")
    stop_loss_inr: float = Field(gt=0.0, description="Suggested stop-loss price in INR")
    generated_at: datetime = Field(default_factory=_now_ist)

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()


class EntryZone(BaseModel):
    """Price range within which the buy candidate should be entered."""

    lower_inr: float = Field(gt=0.0)
    upper_inr: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _lower_lt_upper(self) -> "EntryZone":
        if self.lower_inr >= self.upper_inr:
            raise ValueError(
                f"lower_inr ({self.lower_inr}) must be less than upper_inr ({self.upper_inr})"
            )
        return self


class BuyCandidate(BaseModel):
    """Buy recommendation produced by the Buyer workflow for a screened symbol."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    score: float = Field(ge=0.0, le=100.0, description="Composite buy score 0-100")
    suggested_allocation_inr: float = Field(gt=0.0, description="Suggested capital to deploy INR")
    entry_zone: EntryZone
    stop_loss_inr: float = Field(gt=0.0, description="Stop-loss price in INR")
    target_inr: float = Field(gt=0.0, description="Price target in INR")
    entry_type: Literal["BREAKOUT", "PULLBACK", "CURRENT_PRICE"] = Field(
        default="CURRENT_PRICE",
        description=(
            "How the entry zone was derived: "
            "BREAKOUT=price at resistance with high volume, "
            "PULLBACK=price near support with RSI<45, "
            "CURRENT_PRICE=default market-price entry"
        ),
    )
    generated_at: datetime = Field(default_factory=_now_ist)

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def _stop_below_entry_below_target(self) -> "BuyCandidate":
        if self.stop_loss_inr >= self.entry_zone.lower_inr:
            raise ValueError(
                f"stop_loss_inr ({self.stop_loss_inr}) must be below entry_zone.lower_inr "
                f"({self.entry_zone.lower_inr})"
            )
        if self.target_inr <= self.entry_zone.upper_inr:
            raise ValueError(
                f"target_inr ({self.target_inr}) must be above entry_zone.upper_inr "
                f"({self.entry_zone.upper_inr})"
            )
        return self
