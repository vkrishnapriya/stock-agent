"""
src/workflows/seller_workflow.py
SellerWorkflow — analyses each portfolio holding using a 4-agent CrewAI crew
and produces a SELL or HOLD decision per stock.

For each holding the workflow:
  1. Creates tasks for TechnicalAnalysisAgent, RiskManagementAgent,
     FinancialAnalysisAgent, and CompetitorAnalysisAgent.
  2. Runs them as a sequential CrewAI Crew (Process.sequential).
  3. Extracts Pydantic outputs from crew.kickoff().tasks_output.
  4. Aggregates with weights {technical: 0.30, risk: 0.25,
     fundamental: 0.30, competitive: 0.15}.
  5. Returns a StockDecision (SELL if composite < 45, HOLD otherwise).
"""

from __future__ import annotations

from typing import Any

import structlog
from crewai import Crew, Process

from src.agents.competitor_analyst import CompetitorAnalysisAgent
from src.agents.financial_analyst import FinancialAnalysisAgent
from src.agents.risk_manager import RiskManagementAgent
from src.agents.technical_analyst import TechnicalAnalysisAgent
from src.models.portfolio import Holding, Portfolio
from src.models.recommendations import StockDecision
from src.models.reports import FinalReport
from src.models.signals import (
    CompetitiveAnalysis,
    FundamentalScore,
    RiskAssessment,
    TechnicalSignal,
)

log = structlog.get_logger(__name__)

# Scoring weights (must sum to 1.0)
_WEIGHTS = {"technical": 0.30, "risk": 0.25, "fundamental": 0.30, "competitive": 0.15}
# Composite score below this threshold → SELL
_SELL_THRESHOLD = 45.0
# Default stop-loss as fraction of purchase price when RM agent unavailable
_DEFAULT_STOP_PCT = 0.05


# ---------------------------------------------------------------------------
# Score aggregation (module-level for testability)
# ---------------------------------------------------------------------------


def aggregate_to_decision(
    symbol: str,
    avg_buy_price: float,
    technical: TechnicalSignal | None,
    risk: RiskAssessment | None,
    fundamental: FundamentalScore | None,
    competitive: CompetitiveAnalysis | None,
) -> StockDecision:
    """Aggregate four agent signals into a single :class:`StockDecision`.

    All input signals may be ``None`` (e.g. on agent failure); neutral
    defaults (score = 50) are used in that case.

    Args:
        symbol:        NSE ticker.
        avg_buy_price: Cost-basis price used for fallback stop-loss.
        technical:     Output of TechnicalAnalysisAgent (score −100..+100).
        risk:          Output of RiskManagementAgent (risk_score 0..100).
        fundamental:   Output of FinancialAnalysisAgent (score 0..100).
        competitive:   Output of CompetitorAnalysisAgent (moat_score 0..100).

    Returns:
        :class:`StockDecision` with action SELL or HOLD and a rationale.
    """
    # Normalise all signals to a 0–100 "quality / hold-worthiness" scale.
    # technical: −100..+100 → 0..100 (higher = more bullish = HOLD)
    tech_q = ((technical.score + 100) / 2.0) if technical else 50.0
    # risk: 0..100, higher = more risk → invert so high risk lowers quality
    risk_q = (100.0 - risk.risk_score) if risk else 50.0
    # fundamental: 0..100 already; higher = better fundamentals = HOLD
    fund_q = fundamental.score if fundamental else 50.0
    # competitive moat: 0..100; higher = stronger moat = HOLD
    comp_q = competitive.moat_score if competitive else 50.0

    composite = (
        tech_q * _WEIGHTS["technical"]
        + risk_q * _WEIGHTS["risk"]
        + fund_q * _WEIGHTS["fundamental"]
        + comp_q * _WEIGHTS["competitive"]
    )

    action: str = "SELL" if composite < _SELL_THRESHOLD else "HOLD"

    # Confidence: distance from threshold, clamped to [0, 1]
    confidence = min(1.0, abs(composite - _SELL_THRESHOLD) / (_SELL_THRESHOLD))

    # Human-readable rationale
    parts: list[str] = []
    if technical:
        parts.append(f"TA {technical.trend} (score {technical.score:.0f})")
    if risk:
        parts.append(f"Risk {risk.risk_score:.0f}/100")
    if fundamental:
        parts.append(f"FA {fundamental.score:.0f}/100")
    if competitive:
        parts.append(f"Moat {competitive.moat_score:.0f}/100")
    parts.append(f"Composite {composite:.1f}/100 → {action}")
    rationale = "; ".join(parts)

    stop_loss = (
        risk.suggested_stop_loss_inr
        if risk and risk.suggested_stop_loss_inr > 0
        else avg_buy_price * (1.0 - _DEFAULT_STOP_PCT)
    )

    return StockDecision(
        symbol=symbol,
        action=action,  # type: ignore[arg-type]
        confidence=round(confidence, 4),
        rationale=rationale,
        stop_loss_inr=round(stop_loss, 2),
    )


# ---------------------------------------------------------------------------
# SellerWorkflow
# ---------------------------------------------------------------------------


class SellerWorkflow:
    """Run a 4-agent crew for every portfolio holding and produce sell/hold decisions.

    Usage::

        portfolio = Portfolio.load_from_file("data/portfolios/portfolio.json")
        decisions = SellerWorkflow().run(portfolio)
        for d in decisions:
            print(d.symbol, d.action, d.confidence)
    """

    def run(self, portfolio: Portfolio) -> list[StockDecision]:
        """Analyse every holding and return one :class:`StockDecision` each.

        Args:
            portfolio: Loaded portfolio with one or more holdings.

        Returns:
            List of :class:`StockDecision` in the same order as
            ``portfolio.holdings``.
        """
        portfolio_value = (
            portfolio.total_invested_inr + portfolio.available_cash_inr
        )
        decisions: list[StockDecision] = []
        for holding in portfolio.holdings:
            log.info("seller_workflow.analysing", symbol=holding.symbol)
            decision = self._analyse_holding(holding, portfolio_value)
            decisions.append(decision)
        return decisions

    def run_as_report(self, portfolio: Portfolio) -> FinalReport:
        """Run :meth:`run` and wrap results in a :class:`FinalReport`.

        Args:
            portfolio: Loaded portfolio.

        Returns:
            :class:`FinalReport` with ``workflow_type="SELLER"``.
        """
        decisions = self.run(portfolio)
        sell_n = sum(1 for d in decisions if d.action == "SELL")
        hold_n = len(decisions) - sell_n
        summary = (
            f"Analysed {len(decisions)} holdings. "
            f"Recommended: SELL {sell_n}, HOLD {hold_n}."
        )
        return FinalReport(
            workflow_type="SELLER",
            decisions=decisions,  # type: ignore[arg-type]
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse_holding(
        self, holding: Holding, portfolio_value: float
    ) -> StockDecision:
        """Run the 4-agent crew for a single *holding*.

        On any crew failure the method falls back to a neutral HOLD decision
        so that a single bad holding does not abort the entire workflow.
        """
        symbol = holding.symbol
        sector = holding.sector
        entry_price = holding.avg_buy_price
        stop_loss = entry_price * (1.0 - _DEFAULT_STOP_PCT)

        # Build agents and tasks
        ta_ag = TechnicalAnalysisAgent()
        rm_ag = RiskManagementAgent()
        fa_ag = FinancialAnalysisAgent()
        ca_ag = CompetitorAnalysisAgent()

        ta_task = ta_ag.build_task(symbol)
        rm_task = rm_ag.build_task(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            portfolio_value=portfolio_value,
        )
        fa_task = fa_ag.build_task(symbol=symbol, sector=sector)
        ca_task = ca_ag.build_task(symbol=symbol, sector=sector)

        crew = Crew(
            agents=[ta_task.agent, rm_task.agent, fa_task.agent, ca_task.agent],
            tasks=[ta_task, rm_task, fa_task, ca_task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            outputs = getattr(result, "tasks_output", []) or []
            ta_signal: TechnicalSignal | None = _pydantic(outputs, 0)
            risk_sig: RiskAssessment | None = _pydantic(outputs, 1)
            fund_sig: FundamentalScore | None = _pydantic(outputs, 2)
            comp_sig: CompetitiveAnalysis | None = _pydantic(outputs, 3)
        except Exception as exc:
            log.error(
                "seller_workflow.crew_failed",
                symbol=symbol,
                error=str(exc),
                exc_info=True,
            )
            # Graceful fallback — neutral HOLD
            return StockDecision(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                rationale=f"Analysis failed ({exc!s}); defaulting to HOLD.",
                stop_loss_inr=round(stop_loss, 2),
            )

        return aggregate_to_decision(
            symbol=symbol,
            avg_buy_price=entry_price,
            technical=ta_signal,
            risk=risk_sig,
            fundamental=fund_sig,
            competitive=comp_sig,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pydantic(outputs: list[Any], idx: int) -> Any | None:
    """Safely extract ``task_output.pydantic`` at *idx* from *outputs*."""
    if idx < len(outputs):
        out = outputs[idx]
        return getattr(out, "pydantic", None)
    return None
