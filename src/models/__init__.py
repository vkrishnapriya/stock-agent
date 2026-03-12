"""src/models — Pydantic data contracts (single source of truth)."""

from src.models.portfolio import Holding, Portfolio
from src.models.recommendations import BuyCandidate, EntryZone, StockDecision
from src.models.reports import FinalReport
from src.models.signals import (
    CompetitiveAnalysis,
    FundamentalScore,
    RiskAssessment,
    SentimentResult,
    TechnicalSignal,
)

__all__ = [
    "Holding",
    "Portfolio",
    "TechnicalSignal",
    "RiskAssessment",
    "FundamentalScore",
    "SentimentResult",
    "CompetitiveAnalysis",
    "StockDecision",
    "EntryZone",
    "BuyCandidate",
    "FinalReport",
]
