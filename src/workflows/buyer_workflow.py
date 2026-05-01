"""
src/workflows/buyer_workflow.py
BuyerWorkflow — scans a stock universe, filters by sentiment, runs deep
4-agent analysis in parallel, and returns the top-5 buy candidates.

Step 1: MarketScannerAgent.scan()          → ScanResults  (algorithmic)
Step 2: NewsSentimentAgent per shortlist   → filter stocks with score < -20
Step 3: TechnicalAnalysisAgent +           → per-stock signals (parallel crews)
         RiskManagementAgent +
         FinancialAnalysisAgent +
         CompetitorAnalysisAgent
Step 4: Score with weights {technical:0.30, risk:0.25, fundamental:0.30,
         competitive:0.15} and return top-5 BuyCandidate objects.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import structlog
from crewai import Crew, Process

from src.agents.competitor_analyst import CompetitorAnalysisAgent
from src.agents.financial_analyst import FinancialAnalysisAgent
from src.agents.market_scanner import MarketScannerAgent
from src.agents.news_sentiment import NewsSentimentAgent
from src.agents.risk_manager import RiskManagementAgent
from src.agents.technical_analyst import TechnicalAnalysisAgent
from src.models.recommendations import BuyCandidate, EntryZone
from src.models.reports import FinalReport
from src.models.scan import ScanEntry
from src.models.signals import (
    CompetitiveAnalysis,
    FundamentalScore,
    RiskAssessment,
    SentimentResult,
    TechnicalSignal,
)

log = structlog.get_logger(__name__)

# Tuning knobs
_WEIGHTS = {"technical": 0.30, "risk": 0.25, "fundamental": 0.30, "competitive": 0.15}
_MAX_SHORTLIST = 15        # stocks passed from scanner to sentiment filter
_MAX_DEEP_ANALYSE = 10     # stocks passed from sentiment to full 4-agent analysis
_TOP_N = 5                 # final buy candidates returned
_SENTIMENT_THRESHOLD = -20.0   # stocks below this sentiment score are dropped
_PARALLEL_WORKERS = 1      # sequential to stay within free-tier RPM limits
_DEFAULT_STOP_PCT = 0.05   # 5% below entry if RM agent unavailable


# ---------------------------------------------------------------------------
# BuyerWorkflow
# ---------------------------------------------------------------------------


class BuyerWorkflow:
    """Screen → filter → analyse → rank; return top-5 buy candidates.

    Usage::

        workflow   = BuyerWorkflow()
        candidates = workflow.run(universe="NIFTY50", available_cash=200_000)
        for c in candidates:
            print(c.symbol, c.score, c.suggested_allocation_inr)
    """

    def run(
        self,
        universe: str = "NIFTY50",
        prompt: str = "",
        available_cash: float = 100_000.0,
    ) -> list[BuyCandidate]:
        """Execute the full buy-pipeline and return the ranked candidates.

        Args:
            universe:       Stock universe to scan (NIFTY50/NIFTY100/NIFTY500).
            prompt:         Original user prompt (logged for context; not yet
                            used for semantic filtering in Phase 1).
            available_cash: Capital available for new positions (INR).

        Returns:
            Up to :data:`_TOP_N` :class:`~src.models.recommendations.BuyCandidate`
            objects sorted by composite score descending.
        """
        log.info(
            "buyer_workflow.start",
            universe=universe,
            available_cash=available_cash,
        )

        # ── Step 1: Market scan ───────────────────────────────────────────
        scan_results = MarketScannerAgent().scan(universe, top_n=_MAX_SHORTLIST)
        shortlist = scan_results.shortlist(_MAX_SHORTLIST)
        log.info("buyer_workflow.scan_done", shortlisted=len(shortlist))

        # ── Step 2: Sentiment filter ──────────────────────────────────────
        sentiment_passed = self._filter_by_sentiment(shortlist)
        to_analyse = sentiment_passed[:_MAX_DEEP_ANALYSE]
        log.info(
            "buyer_workflow.sentiment_filter",
            before=len(shortlist),
            after=len(to_analyse),
        )

        if not to_analyse:
            log.warning("buyer_workflow.no_candidates_after_filter")
            return []

        # ── Step 3: Parallel 4-agent deep analysis ────────────────────────
        raw_results = self._run_parallel_analysis(to_analyse, available_cash)

        # ── Step 4: Score, build BuyCandidate objects, and rank ───────────
        candidates: list[BuyCandidate] = []
        for entry, (ta_sig, risk_sig, fund_sig, comp_sig) in raw_results:
            candidate = self._build_buy_candidate(
                entry=entry,
                available_cash=available_cash,
                technical=ta_sig,
                risk=risk_sig,
                fundamental=fund_sig,
                competitive=comp_sig,
            )
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)
        top = candidates[:_TOP_N]
        log.info("buyer_workflow.done", candidates=len(top))
        return top

    def run_as_report(
        self,
        universe: str = "NIFTY50",
        prompt: str = "",
        available_cash: float = 100_000.0,
    ) -> FinalReport:
        """Run :meth:`run` and wrap the candidates in a :class:`FinalReport`.

        Returns:
            :class:`FinalReport` with ``workflow_type="BUYER"``.
        """
        candidates = self.run(universe=universe, prompt=prompt, available_cash=available_cash)
        summary = (
            f"Scanned {universe} universe. "
            f"Identified {len(candidates)} buy candidate(s) after sentiment "
            f"filtering and fundamental analysis."
        )
        if candidates:
            top_sym = ", ".join(c.symbol for c in candidates)
            summary += f" Top picks: {top_sym}."
        return FinalReport(
            workflow_type="BUYER",
            decisions=candidates,  # type: ignore[arg-type]
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Step 2: Sentiment filter
    # ------------------------------------------------------------------

    def _filter_by_sentiment(self, entries: list[ScanEntry]) -> list[ScanEntry]:
        """Run NewsSentimentAgent for each entry; drop those with score < threshold."""
        passed: list[ScanEntry] = []
        for entry in entries:
            sentiment = self._run_sentiment(entry.symbol)
            if sentiment is None or sentiment.score >= _SENTIMENT_THRESHOLD:
                passed.append(entry)
            else:
                log.info(
                    "buyer_workflow.sentiment_dropped",
                    symbol=entry.symbol,
                    score=sentiment.score,
                )
        return passed

    def _run_sentiment(self, symbol: str) -> SentimentResult | None:
        """Run a single-agent NewsSentimentAgent crew for *symbol*."""
        try:
            agent = NewsSentimentAgent()
            task = agent.build_task(symbol=symbol)
            crew = Crew(
                agents=[task.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()
            outputs = getattr(result, "tasks_output", []) or []
            return _pydantic(outputs, 0)
        except Exception as exc:
            log.warning("buyer_workflow.sentiment_failed", symbol=symbol, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Step 3: Parallel deep analysis
    # ------------------------------------------------------------------

    def _run_parallel_analysis(
        self,
        entries: list[ScanEntry],
        available_cash: float,
    ) -> list[tuple[ScanEntry, tuple[
        TechnicalSignal | None,
        RiskAssessment | None,
        FundamentalScore | None,
        CompetitiveAnalysis | None,
    ]]]:
        """Run 4-agent crews for each entry in parallel threads."""
        results: list[tuple[ScanEntry, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as pool:
            future_to_entry = {
                pool.submit(
                    self._run_deep_analysis, entry, available_cash
                ): entry
                for entry in entries
            }
            for future in concurrent.futures.as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    signals = future.result()
                    results.append((entry, signals))
                except Exception as exc:
                    log.error(
                        "buyer_workflow.analysis_failed",
                        symbol=entry.symbol,
                        error=str(exc),
                    )
                    results.append((entry, (None, None, None, None)))

        return results

    def _run_deep_analysis(
        self,
        entry: ScanEntry,
        available_cash: float,
    ) -> tuple[
        TechnicalSignal | None,
        RiskAssessment | None,
        FundamentalScore | None,
        CompetitiveAnalysis | None,
    ]:
        """Run TA + RM + FA + CA crew for a single entry."""
        symbol = entry.symbol
        last_price = entry.last_price
        stop_loss = last_price * (1.0 - _DEFAULT_STOP_PCT)

        ta_ag = TechnicalAnalysisAgent()
        rm_ag = RiskManagementAgent()
        fa_ag = FinancialAnalysisAgent()
        ca_ag = CompetitorAnalysisAgent()

        ta_task = ta_ag.build_task(symbol)
        rm_task = rm_ag.build_task(
            symbol=symbol,
            entry_price=last_price,
            stop_loss=stop_loss,
            portfolio_value=available_cash,
        )
        fa_task = fa_ag.build_task(symbol=symbol, sector=entry.sector)
        ca_task = ca_ag.build_task(symbol=symbol, sector=entry.sector)

        crew = Crew(
            agents=[ta_task.agent, rm_task.agent, fa_task.agent, ca_task.agent],
            tasks=[ta_task, rm_task, fa_task, ca_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        outputs = getattr(result, "tasks_output", []) or []
        return (
            _pydantic(outputs, 0),
            _pydantic(outputs, 1),
            _pydantic(outputs, 2),
            _pydantic(outputs, 3),
        )

    # ------------------------------------------------------------------
    # Step 4: Build BuyCandidate
    # ------------------------------------------------------------------

    def _build_buy_candidate(
        self,
        entry: ScanEntry,
        available_cash: float,
        technical: TechnicalSignal | None,
        risk: RiskAssessment | None,
        fundamental: FundamentalScore | None,
        competitive: CompetitiveAnalysis | None,
    ) -> BuyCandidate | None:
        """Compute composite score and construct a :class:`BuyCandidate`.

        Returns ``None`` if the composite score is below 40 (weak candidate).
        """
        last_price = entry.last_price

        # Normalise signals to 0-100 quality scale
        tech_q = ((technical.score + 100) / 2.0) if technical else 50.0
        risk_q = (100.0 - risk.risk_score) if risk else 50.0
        fund_q = fundamental.score if fundamental else 50.0
        comp_q = competitive.moat_score if competitive else 50.0

        composite = (
            tech_q * _WEIGHTS["technical"]
            + risk_q * _WEIGHTS["risk"]
            + fund_q * _WEIGHTS["fundamental"]
            + comp_q * _WEIGHTS["competitive"]
        )

        if composite < 40.0:
            return None

        # Entry zone: support → support + 2%
        support = technical.support_level_inr if technical else last_price * 0.97
        entry_lower = round(support, 2)
        entry_upper = round(support * 1.02, 2)

        # Stop-loss: 3% below support (must be < entry_lower)
        stop = risk.suggested_stop_loss_inr if risk else round(entry_lower * 0.97, 2)
        stop = round(min(stop, entry_lower * 0.97), 2)

        # Target: resistance, or at least 5% above entry_upper
        resistance = technical.resistance_level_inr if technical else last_price * 1.10
        target = round(max(resistance, entry_upper * 1.05), 2)

        # Suggested allocation: RM position size or proportional to composite
        if risk and risk.position_size_pct > 0:
            allocation = round(available_cash * (risk.position_size_pct / 100.0), 2)
        else:
            allocation = round(available_cash * (composite / 100.0) * 0.20, 2)
        allocation = max(allocation, 1.0)

        try:
            return BuyCandidate(
                symbol=entry.symbol,
                score=round(composite, 2),
                suggested_allocation_inr=allocation,
                entry_zone=EntryZone(lower_inr=entry_lower, upper_inr=entry_upper),
                stop_loss_inr=stop,
                target_inr=target,
            )
        except Exception as exc:
            log.warning(
                "buyer_workflow.invalid_candidate",
                symbol=entry.symbol,
                error=str(exc),
            )
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pydantic(outputs: list[Any], idx: int) -> Any | None:
    if idx < len(outputs):
        return getattr(outputs[idx], "pydantic", None)
    return None
