"""
src/models/signals.py
Pydantic output contracts for each analysis agent.

Score conventions
-----------------
TechnicalSignal.score   : -100 (extreme bearish) → +100 (extreme bullish)
SentimentResult.score   : -100 (very negative)   → +100 (very positive)
RiskAssessment.risk_score: 0 (no risk)            → 100 (maximum risk)
FundamentalScore.score  : 0 (poor fundamentals)  → 100 (excellent)
CompetitiveAnalysis.moat_score       : 0 → 100
CompetitiveAnalysis.relative_strength: -100 → +100 vs sector
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pytz
from pydantic import BaseModel, Field, field_validator

_IST = pytz.timezone("Asia/Kolkata")


def _now_ist() -> datetime:
    return datetime.now(_IST)


class TechnicalSignal(BaseModel):
    """Output of the Technical Analysis agent for a single symbol."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    score: float = Field(ge=-100.0, le=100.0, description="Composite technical score")
    trend: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    rsi: float = Field(ge=0.0, le=100.0, description="14-period RSI")
    macd_signal: float = Field(description="MACD histogram value (positive=bullish)")
    support_level_inr: float = Field(gt=0.0, description="Nearest support price in INR")
    resistance_level_inr: float = Field(gt=0.0, description="Nearest resistance price in INR")
    volume_signal: Literal["HIGH", "LOW", "NORMAL"] = "NORMAL"
    generated_at: datetime = Field(default_factory=_now_ist)

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("resistance_level_inr")
    @classmethod
    def _resistance_above_support(cls, v: float, info) -> float:
        support = info.data.get("support_level_inr")
        if support is not None and v <= support:
            raise ValueError(
                f"resistance_level_inr ({v}) must be greater than support_level_inr ({support})"
            )
        return v


class RiskAssessment(BaseModel):
    """Output of the Risk Management agent for a single symbol."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    risk_score: float = Field(ge=0.0, le=100.0, description="0=safe, 100=maximum risk")
    volatility_pct: float = Field(ge=0.0, description="Annualised volatility %")
    beta: float = Field(description="Beta vs Nifty 50")
    max_drawdown_pct: float = Field(ge=0.0, le=100.0, description="Max historical drawdown %")
    circuit_breaker_band: Literal["5%", "10%", "20%"] = "20%"
    suggested_stop_loss_inr: float = Field(gt=0.0, description="Suggested stop-loss price INR")
    position_size_pct: float = Field(
        ge=0.0, le=100.0, description="Suggested position size as % of portfolio"
    )
    generated_at: datetime = Field(default_factory=_now_ist)

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()


class FundamentalScore(BaseModel):
    """Output of the Fundamental Analysis agent for a single symbol."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    score: float = Field(ge=0.0, le=100.0, description="0=poor fundamentals, 100=excellent")
    pe_ratio: float | None = Field(default=None, gt=0.0)
    pb_ratio: float | None = Field(default=None, gt=0.0)
    roe_pct: float | None = None
    debt_to_equity: float | None = Field(default=None, ge=0.0)
    revenue_growth_pct: float | None = None
    earnings_growth_pct: float | None = None
    promoter_holding_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    generated_at: datetime = Field(default_factory=_now_ist)

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()


class SentimentResult(BaseModel):
    """Output of the News/Sentiment agent for a single symbol."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    score: float = Field(ge=-100.0, le=100.0, description="-100=very negative, +100=very positive")
    news_sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    social_sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    whale_signal: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "NEUTRAL"
    whale_activity: list[str] = Field(
        default_factory=list,
        description="Human-readable lines describing each super-investor deal, e.g. "
                    "'BUY: Dolly Khanna bought 80,000 shares @ ₹1,480 on 02-May-2026'.",
    )
    headline_count: int = Field(ge=0)
    key_themes: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now_ist)

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()


class CompetitiveAnalysis(BaseModel):
    """Output of the Competitive/Sector Analysis agent for a single symbol."""

    symbol: str
    exchange: Literal["NSE", "BSE"] = "NSE"
    sector: str
    market_position: Literal["LEADER", "CHALLENGER", "FOLLOWER", "NICHE"]
    moat_score: float = Field(ge=0.0, le=100.0, description="Economic moat strength 0-100")
    relative_strength: float = Field(
        ge=-100.0, le=100.0, description="Performance vs sector peers -100 to +100"
    )
    peers: list[str] = Field(default_factory=list, description="Peer NSE symbols")
    generated_at: datetime = Field(default_factory=_now_ist)

    @field_validator("symbol")
    @classmethod
    def _symbol_upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("peers")
    @classmethod
    def _peers_upper(cls, v: list[str]) -> list[str]:
        return [p.strip().upper() for p in v]


# ---------------------------------------------------------------------------
# Batch wrappers — one LLM call per agent type for all stocks
# ---------------------------------------------------------------------------


class TechnicalSignalBatch(BaseModel):
    """Batch output of TechnicalAnalysisAgent — one signal per symbol."""
    signals: list[TechnicalSignal]


class FundamentalScoreBatch(BaseModel):
    """Batch output of FinancialAnalysisAgent — one score per symbol."""
    scores: list[FundamentalScore]


class RiskAssessmentBatch(BaseModel):
    """Batch output of RiskManagementAgent — one assessment per symbol."""
    assessments: list[RiskAssessment]


class CompetitiveAnalysisBatch(BaseModel):
    """Batch output of CompetitorAnalysisAgent — one analysis per symbol."""
    analyses: list[CompetitiveAnalysis]
