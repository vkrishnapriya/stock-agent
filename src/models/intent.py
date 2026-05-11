"""
src/models/intent.py
Structured representation of a user's buy intent extracted from a free-text prompt.

Produced by WorkflowDirector.extract_intent() via a single LLM call, then used
by BuyerWorkflow._filter_by_prompt() to narrow the scan shortlist before the
expensive 4-agent batch analysis runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BuyIntent(BaseModel):
    """Structured buy intent parsed from a natural-language prompt.

    All fields are optional — unspecified fields mean "no constraint".

    Example (from "IT sector under Rs 2000, growth focus")::

        BuyIntent(
            sectors=["IT"],
            max_price_inr=2000.0,
            style="growth",
        )
    """

    sectors: list[str] = Field(
        default_factory=list,
        description="Preferred sectors, e.g. ['IT', 'Banking', 'Pharma']. Empty = all sectors.",
    )
    exclude_sectors: list[str] = Field(
        default_factory=list,
        description="Sectors to explicitly avoid, e.g. ['Agri', 'Mining'].",
    )
    max_price_inr: float | None = Field(
        default=None,
        gt=0,
        description="Upper price limit per share in INR. None = no limit.",
    )
    min_price_inr: float | None = Field(
        default=None,
        gt=0,
        description="Lower price limit per share in INR. None = no limit.",
    )
    style: Literal["growth", "value", "dividend", "momentum", "defensive", "any"] = Field(
        default="any",
        description=(
            "Investment style preference:\n"
            "  growth    — high EPS/revenue growth, can be higher P/E\n"
            "  value     — low P/E, high promoter holding, avoid overbought RSI\n"
            "  dividend  — high dividend yield, stable earnings\n"
            "  momentum  — strongest 1M price return (default scanner behaviour)\n"
            "  defensive — low beta, FMCG/Pharma/IT, avoids cyclicals\n"
            "  any       — no style preference"
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Free-form keywords extracted from the prompt for logging.",
    )
