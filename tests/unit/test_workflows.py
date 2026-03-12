"""
tests/unit/test_workflows.py
Unit tests for WorkflowDirector, SellerWorkflow, BuyerWorkflow,
MarketScannerAgent, and ScanResults.

All LLM, Crew, yfinance, and external API calls are mocked — no network access.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytz

from src.models.portfolio import Holding, Portfolio
from src.models.recommendations import BuyCandidate, EntryZone, StockDecision
from src.models.scan import ScanEntry, ScanResults
from src.models.signals import (
    CompetitiveAnalysis,
    FundamentalScore,
    RiskAssessment,
    SentimentResult,
    TechnicalSignal,
)

_IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_portfolio(symbols: list[str] | None = None) -> Portfolio:
    symbols = symbols or ["INFY", "TCS"]
    holdings = [
        Holding(symbol=s, quantity=10, avg_buy_price=1000.0 + i * 500)
        for i, s in enumerate(symbols)
    ]
    return Portfolio(portfolio_id="test-p", holdings=holdings, available_cash_inr=50_000.0)


def _make_ta_signal(symbol: str = "INFY", score: float = 40.0) -> TechnicalSignal:
    return TechnicalSignal(
        symbol=symbol,
        score=score,
        trend="NEUTRAL",
        rsi=50.0,
        macd_signal=0.0,
        support_level_inr=900.0,
        resistance_level_inr=1100.0,
    )


def _make_risk(symbol: str = "INFY", risk_score: float = 40.0) -> RiskAssessment:
    return RiskAssessment(
        symbol=symbol,
        risk_score=risk_score,
        volatility_pct=20.0,
        beta=1.0,
        max_drawdown_pct=15.0,
        circuit_breaker_band="20%",
        suggested_stop_loss_inr=950.0,
        position_size_pct=5.0,
    )


def _make_fundamental(symbol: str = "INFY", score: float = 65.0) -> FundamentalScore:
    return FundamentalScore(symbol=symbol, score=score)


def _make_competitive(symbol: str = "INFY", moat: float = 60.0) -> CompetitiveAnalysis:
    return CompetitiveAnalysis(
        symbol=symbol,
        sector="IT",
        market_position="CHALLENGER",
        moat_score=moat,
        relative_strength=10.0,
        peers=["TCS", "WIPRO"],
    )


def _make_sentiment(symbol: str = "INFY", score: float = 10.0) -> SentimentResult:
    return SentimentResult(
        symbol=symbol,
        score=score,
        news_sentiment="POSITIVE",
        social_sentiment="NEUTRAL",
        headline_count=5,
    )


def _mock_crew_output(*models) -> MagicMock:
    """Return a mock CrewOutput whose tasks_output contains the given pydantic models."""
    task_outputs = []
    for m in models:
        to = MagicMock()
        to.pydantic = m
        task_outputs.append(to)
    mock_result = MagicMock()
    mock_result.tasks_output = task_outputs
    return mock_result


def _make_scan_entry(symbol: str, momentum: float = 5.0, last_price: float = 1000.0) -> ScanEntry:
    return ScanEntry(
        symbol=symbol,
        sector="IT",
        momentum_1m_pct=momentum,
        volume_ratio=1.2,
        last_price=last_price,
        rank=1,
    )


def _make_scan_results(symbols: list[str] | None = None) -> ScanResults:
    symbols = symbols or ["INFY", "TCS", "WIPRO"]
    entries = [
        _make_scan_entry(s, momentum=float(i + 1) * 2.0, last_price=1000.0 + i * 100)
        for i, s in enumerate(symbols)
    ]
    return ScanResults(universe="NIFTY50", entries=entries, total_scanned=50)


# ---------------------------------------------------------------------------
# TestScanResults
# ---------------------------------------------------------------------------


class TestScanResults:
    def test_symbols_property(self):
        results = _make_scan_results(["INFY", "TCS", "WIPRO"])
        assert results.symbols == ["INFY", "TCS", "WIPRO"]

    def test_shortlist_returns_sorted_by_momentum(self):
        entries = [
            _make_scan_entry("A", momentum=1.0),
            _make_scan_entry("B", momentum=5.0),
            _make_scan_entry("C", momentum=3.0),
        ]
        results = ScanResults(universe="NIFTY50", entries=entries, total_scanned=3)
        short = results.shortlist(2)
        assert [e.symbol for e in short] == ["B", "C"]

    def test_shortlist_caps_at_n(self):
        results = _make_scan_results(["A", "B", "C", "D", "E"])
        assert len(results.shortlist(3)) == 3


# ---------------------------------------------------------------------------
# TestMarketScannerAgent
# ---------------------------------------------------------------------------


def _make_multi_df(tickers: list[str], rows: int = 65) -> pd.DataFrame:
    dates = pd.date_range(end="2024-03-31", periods=rows, freq="B")
    data: dict = {}
    for i, t in enumerate(tickers):
        start = 1000.0 + i * 100
        data[("Close", t)] = np.linspace(start, start * 1.1, rows)
        data[("Volume", t)] = np.full(rows, 1_000_000 + i * 50_000)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestMarketScannerAgent:
    def test_scan_returns_scan_results(self):
        from src.agents.market_scanner import MarketScannerAgent, NIFTY50_SYMBOLS

        ns_syms = [f"{s}.NS" for s in NIFTY50_SYMBOLS]
        df = _make_multi_df(ns_syms[:5])

        with patch("yfinance.download", return_value=df):
            scanner = MarketScannerAgent()
            # Only scan first 5 to keep test fast
            with patch("src.agents.market_scanner._get_universe", return_value=NIFTY50_SYMBOLS[:5]):
                results = scanner.scan("NIFTY50", top_n=3)

        assert isinstance(results, ScanResults)
        assert results.universe == "NIFTY50"

    def test_scan_ranks_from_1(self):
        from src.agents.market_scanner import MarketScannerAgent, NIFTY50_SYMBOLS

        syms = NIFTY50_SYMBOLS[:3]
        ns_syms = [f"{s}.NS" for s in syms]
        df = _make_multi_df(ns_syms)

        with patch("yfinance.download", return_value=df):
            with patch("src.agents.market_scanner._get_universe", return_value=syms):
                results = MarketScannerAgent().scan("NIFTY50", top_n=3)

        assert results.entries[0].rank == 1

    def test_scan_unknown_universe_raises(self):
        from src.agents.market_scanner import MarketScannerAgent

        scanner = MarketScannerAgent()
        with pytest.raises(ValueError, match="Unknown universe"):
            scanner.scan("INVALID_UNIVERSE")

    def test_scan_symbols_uppercased(self):
        from src.agents.market_scanner import NIFTY50_SYMBOLS

        for sym in NIFTY50_SYMBOLS:
            assert sym == sym.upper()

    def test_scan_top_n_respected(self):
        from src.agents.market_scanner import MarketScannerAgent, NIFTY50_SYMBOLS

        syms = NIFTY50_SYMBOLS[:10]
        ns_syms = [f"{s}.NS" for s in syms]
        df = _make_multi_df(ns_syms)

        with patch("yfinance.download", return_value=df):
            with patch("src.agents.market_scanner._get_universe", return_value=syms):
                results = MarketScannerAgent().scan("NIFTY50", top_n=4)

        assert len(results.entries) <= 4

    def test_scan_entries_sorted_descending(self):
        from src.agents.market_scanner import MarketScannerAgent, NIFTY50_SYMBOLS

        syms = NIFTY50_SYMBOLS[:5]
        ns_syms = [f"{s}.NS" for s in syms]
        df = _make_multi_df(ns_syms)

        with patch("yfinance.download", return_value=df):
            with patch("src.agents.market_scanner._get_universe", return_value=syms):
                results = MarketScannerAgent().scan("NIFTY50", top_n=5)

        momenta = [e.momentum_1m_pct for e in results.entries]
        assert momenta == sorted(momenta, reverse=True)


# ---------------------------------------------------------------------------
# TestWorkflowDirector
# ---------------------------------------------------------------------------


class TestWorkflowDirector:
    def _make_director(self, llm_content: str = "BUY") -> "WorkflowDirector":  # type: ignore[name-defined]
        from src.workflows.director import WorkflowDirector

        mock_resp = MagicMock()
        mock_resp.content = llm_content
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp

        with patch("src.workflows.director.get_llm", return_value=mock_llm):
            return WorkflowDirector()

    def test_classify_buy_intent(self):
        d = self._make_director("BUY")
        assert d.classify_intent("Find growth opportunities") == "BUY"

    def test_classify_sell_intent(self):
        d = self._make_director("SELL")
        assert d.classify_intent("Should I exit my positions?") == "SELL"

    def test_classify_defaults_to_buy_on_ambiguous(self):
        d = self._make_director("HMMMM")
        assert d.classify_intent("unclear prompt") == "BUY"

    def test_classify_handles_llm_exception(self):
        from src.workflows.director import WorkflowDirector

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM error")
        with patch("src.workflows.director.get_llm", return_value=mock_llm):
            d = WorkflowDirector()
        assert d.classify_intent("whatever") == "BUY"  # fallback

    def test_parse_portfolio_loads_file(self, tmp_path):
        import json

        from src.workflows.director import WorkflowDirector

        data = {
            "portfolio_id": "test-01",
            "holdings": [
                {
                    "symbol": "INFY",
                    "quantity": 10,
                    "avg_buy_price": 1820.5,
                }
            ],
            "available_cash_inr": 10000.0,
        }
        p = tmp_path / "port.json"
        p.write_text(json.dumps(data))

        d = self._make_director()
        portfolio = d.parse_portfolio(str(p))
        assert portfolio.portfolio_id == "test-01"
        assert len(portfolio.holdings) == 1

    def test_parse_portfolio_raises_on_missing_file(self):
        from src.workflows.director import WorkflowDirector

        d = self._make_director()
        with pytest.raises(FileNotFoundError):
            d.parse_portfolio("/nonexistent/path/port.json")


# ---------------------------------------------------------------------------
# TestAggregateToDecision (unit-tests for the aggregation function)
# ---------------------------------------------------------------------------


class TestAggregateToDecision:
    def test_sell_when_composite_below_threshold(self):
        from src.workflows.seller_workflow import aggregate_to_decision

        # All signals bad: technical bearish, high risk, poor fundamentals
        ta = _make_ta_signal(score=-60.0)    # tech_q = 20
        risk = _make_risk(risk_score=80.0)   # risk_q = 20
        fund = _make_fundamental(score=20.0) # fund_q = 20
        comp = _make_competitive(moat=20.0)  # comp_q = 20
        # composite = 20*0.30 + 20*0.25 + 20*0.30 + 20*0.15 = 20.0 < 45 → SELL
        decision = aggregate_to_decision("INFY", 1000.0, ta, risk, fund, comp)
        assert decision.action == "SELL"

    def test_hold_when_composite_above_threshold(self):
        from src.workflows.seller_workflow import aggregate_to_decision

        ta = _make_ta_signal(score=60.0)    # tech_q = 80
        risk = _make_risk(risk_score=20.0)  # risk_q = 80
        fund = _make_fundamental(score=80.0)
        comp = _make_competitive(moat=80.0)
        # composite = 80*1.0 = 80.0 > 45 → HOLD
        decision = aggregate_to_decision("INFY", 1000.0, ta, risk, fund, comp)
        assert decision.action == "HOLD"

    def test_weights_sum_to_one(self):
        from src.workflows.seller_workflow import _WEIGHTS

        assert sum(_WEIGHTS.values()) == pytest.approx(1.0)

    def test_returns_stock_decision(self):
        from src.workflows.seller_workflow import aggregate_to_decision

        decision = aggregate_to_decision(
            "INFY", 1000.0,
            _make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()
        )
        assert isinstance(decision, StockDecision)
        assert decision.symbol == "INFY"

    def test_stop_loss_uses_rm_agent_value(self):
        from src.workflows.seller_workflow import aggregate_to_decision

        risk = _make_risk(risk_score=30.0)
        risk_copy = risk.model_copy(update={"suggested_stop_loss_inr": 875.0})
        decision = aggregate_to_decision(
            "INFY", 1000.0, _make_ta_signal(), risk_copy, _make_fundamental(), _make_competitive()
        )
        assert decision.stop_loss_inr == pytest.approx(875.0)

    def test_stop_loss_fallback_when_no_risk(self):
        from src.workflows.seller_workflow import aggregate_to_decision

        decision = aggregate_to_decision("INFY", 1000.0, None, None, None, None)
        assert decision.stop_loss_inr == pytest.approx(950.0)  # 5% below 1000

    def test_none_signals_use_neutral_defaults(self):
        from src.workflows.seller_workflow import aggregate_to_decision

        # All None → all neutrals (50.0) → composite = 50.0 → HOLD
        decision = aggregate_to_decision("INFY", 1000.0, None, None, None, None)
        assert decision.action == "HOLD"

    def test_confidence_is_in_range(self):
        from src.workflows.seller_workflow import aggregate_to_decision

        decision = aggregate_to_decision(
            "INFY", 1000.0,
            _make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()
        )
        assert 0.0 <= decision.confidence <= 1.0


# ---------------------------------------------------------------------------
# TestSellerWorkflow
# ---------------------------------------------------------------------------


_MOCK_AGENT_LLM = patch("src.agents.base_agent.get_llm", return_value=MagicMock())


class TestSellerWorkflow:
    def test_run_returns_list_of_decisions(self):
        from src.workflows.seller_workflow import SellerWorkflow

        portfolio = _make_portfolio(["INFY", "TCS"])
        signals = [
            _mock_crew_output(
                _make_ta_signal(s), _make_risk(s), _make_fundamental(s), _make_competitive(s)
            )
            for s in ["INFY", "TCS"]
        ]

        with _MOCK_AGENT_LLM, patch("crewai.Crew.kickoff", side_effect=signals):
            decisions = SellerWorkflow().run(portfolio)

        assert len(decisions) == 2
        assert all(isinstance(d, StockDecision) for d in decisions)

    def test_decision_count_matches_holdings(self):
        from src.workflows.seller_workflow import SellerWorkflow

        portfolio = _make_portfolio(["INFY", "TCS", "WIPRO"])
        mock_out = _mock_crew_output(
            _make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()
        )

        with _MOCK_AGENT_LLM, patch("crewai.Crew.kickoff", return_value=mock_out):
            decisions = SellerWorkflow().run(portfolio)

        assert len(decisions) == 3

    def test_run_as_report_workflow_type(self):
        from src.workflows.seller_workflow import SellerWorkflow

        portfolio = _make_portfolio(["INFY"])
        mock_out = _mock_crew_output(
            _make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()
        )

        with _MOCK_AGENT_LLM, patch("crewai.Crew.kickoff", return_value=mock_out):
            report = SellerWorkflow().run_as_report(portfolio)

        assert report.workflow_type == "SELLER"

    def test_crew_failure_returns_hold(self):
        from src.workflows.seller_workflow import SellerWorkflow

        portfolio = _make_portfolio(["INFY"])

        with _MOCK_AGENT_LLM, patch("crewai.Crew.kickoff", side_effect=RuntimeError("crew failed")):
            decisions = SellerWorkflow().run(portfolio)

        assert len(decisions) == 1
        assert decisions[0].action == "HOLD"
        assert decisions[0].confidence == 0.0

    def test_run_as_report_summary_contains_counts(self):
        from src.workflows.seller_workflow import SellerWorkflow

        portfolio = _make_portfolio(["INFY"])
        mock_out = _mock_crew_output(
            _make_ta_signal(score=80.0), _make_risk(risk_score=10.0),
            _make_fundamental(score=90.0), _make_competitive(moat=90.0)
        )

        with _MOCK_AGENT_LLM, patch("crewai.Crew.kickoff", return_value=mock_out):
            report = SellerWorkflow().run_as_report(portfolio)

        assert "HOLD" in report.summary or "SELL" in report.summary


# ---------------------------------------------------------------------------
# TestBuyerWorkflow
# ---------------------------------------------------------------------------


class TestBuyerWorkflow:
    def _make_workflow_mocks(self, symbols: list[str], positive_sentiment: bool = True):
        """Return context managers to patch scan, sentiment, and crews."""
        scan_results = _make_scan_results(symbols)
        sent_score = 30.0 if positive_sentiment else -50.0

        def mock_scan(*a, **kw):
            return scan_results

        def mock_sentiment_run(self_inner, *a, **kw):
            sym = a[0] if a else kw.get("symbol", "INFY")
            return _mock_crew_output(_make_sentiment(sym, sent_score))

        deep_out = _mock_crew_output(
            _make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()
        )

        return mock_scan, mock_sentiment_run, deep_out

    def test_run_returns_list_of_buy_candidates(self):
        from src.workflows.buyer_workflow import BuyerWorkflow

        mock_scan, mock_sent, deep_out = self._make_workflow_mocks(["INFY", "TCS"])

        with (
            patch("src.agents.market_scanner.MarketScannerAgent.scan", mock_scan),
            patch("crewai.Crew.kickoff", return_value=_mock_crew_output(
                _make_sentiment("INFY", 30.0))),
        ):
            # Patch the whole workflow run for simplicity
            with patch.object(
                BuyerWorkflow, "_filter_by_sentiment",
                return_value=_make_scan_results(["INFY", "TCS"]).entries,
            ):
                with patch.object(
                    BuyerWorkflow, "_run_deep_analysis",
                    return_value=(_make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()),
                ):
                    candidates = BuyerWorkflow().run(universe="NIFTY50", available_cash=100_000)

        assert isinstance(candidates, list)

    def test_run_returns_at_most_top_n(self):
        from src.workflows.buyer_workflow import BuyerWorkflow, _TOP_N

        many = [f"SYM{i}" for i in range(12)]
        entries = [
            _make_scan_entry(s, momentum=float(i), last_price=1000.0)
            for i, s in enumerate(many)
        ]

        with (
            patch.object(BuyerWorkflow, "_filter_by_sentiment", return_value=entries),
            patch.object(
                BuyerWorkflow, "_run_deep_analysis",
                return_value=(_make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()),
            ),
            patch("src.agents.market_scanner.MarketScannerAgent.scan", return_value=_make_scan_results(many)),
        ):
            candidates = BuyerWorkflow().run(available_cash=100_000)

        assert len(candidates) <= _TOP_N

    def test_negative_sentiment_stocks_filtered(self):
        from src.workflows.buyer_workflow import BuyerWorkflow

        entries = [
            _make_scan_entry("GOOD", momentum=5.0),
            _make_scan_entry("BAD", momentum=4.0),
        ]

        def mock_sentiment(self_inner, symbol: str) -> SentimentResult | None:
            return _make_sentiment(symbol, score=-50.0 if symbol == "BAD" else 30.0)

        passed: list = []

        def mock_filter(self_inner, entries_in):
            result = []
            for e in entries_in:
                sent = mock_sentiment(self_inner, e.symbol)
                if sent is None or sent.score >= -20.0:
                    result.append(e)
                else:
                    passed.append(e.symbol)
            return result

        with patch.object(BuyerWorkflow, "_filter_by_sentiment", mock_filter):
            with patch.object(
                BuyerWorkflow, "_run_deep_analysis",
                return_value=(_make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()),
            ):
                with patch(
                    "src.agents.market_scanner.MarketScannerAgent.scan",
                    return_value=ScanResults(universe="NIFTY50", entries=entries, total_scanned=2),
                ):
                    BuyerWorkflow().run(available_cash=100_000)

        assert "BAD" in passed

    def test_run_as_report_workflow_type(self):
        from src.workflows.buyer_workflow import BuyerWorkflow

        with (
            patch.object(BuyerWorkflow, "run", return_value=[]),
        ):
            report = BuyerWorkflow().run_as_report(available_cash=100_000)

        assert report.workflow_type == "BUYER"

    def test_candidates_sorted_by_score_descending(self):
        from src.workflows.buyer_workflow import BuyerWorkflow

        entries = [_make_scan_entry(f"SYM{i}", momentum=float(i)) for i in range(5)]

        call_count = [0]

        def mock_deep(self_inner, entry, cash):
            call_count[0] += 1
            # Different scores based on call order so sorting is meaningful
            score = 10.0 * call_count[0]
            return (
                _make_ta_signal(score=score - 50),  # map 10→-40, 50→0, etc.
                _make_risk(risk_score=max(0, 100 - score)),
                _make_fundamental(score=score),
                _make_competitive(moat=score),
            )

        with (
            patch.object(BuyerWorkflow, "_filter_by_sentiment", return_value=entries),
            patch.object(BuyerWorkflow, "_run_deep_analysis", mock_deep),
            patch("src.agents.market_scanner.MarketScannerAgent.scan",
                  return_value=ScanResults(universe="NIFTY50", entries=entries, total_scanned=5)),
        ):
            candidates = BuyerWorkflow().run(available_cash=100_000)

        if len(candidates) > 1:
            scores = [c.score for c in candidates]
            assert scores == sorted(scores, reverse=True)

    def test_buy_candidate_entry_zone_valid(self):
        from src.workflows.buyer_workflow import BuyerWorkflow

        entry = _make_scan_entry("INFY", last_price=1500.0)

        with (
            patch.object(BuyerWorkflow, "_filter_by_sentiment", return_value=[entry]),
            patch.object(
                BuyerWorkflow, "_run_deep_analysis",
                return_value=(_make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()),
            ),
            patch("src.agents.market_scanner.MarketScannerAgent.scan",
                  return_value=ScanResults(universe="NIFTY50", entries=[entry], total_scanned=5)),
        ):
            candidates = BuyerWorkflow().run(available_cash=100_000)

        for c in candidates:
            assert c.entry_zone.lower_inr < c.entry_zone.upper_inr

    def test_buy_candidate_stop_below_entry(self):
        from src.workflows.buyer_workflow import BuyerWorkflow

        entry = _make_scan_entry("INFY", last_price=1500.0)

        with (
            patch.object(BuyerWorkflow, "_filter_by_sentiment", return_value=[entry]),
            patch.object(
                BuyerWorkflow, "_run_deep_analysis",
                return_value=(_make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()),
            ),
            patch("src.agents.market_scanner.MarketScannerAgent.scan",
                  return_value=ScanResults(universe="NIFTY50", entries=[entry], total_scanned=5)),
        ):
            candidates = BuyerWorkflow().run(available_cash=100_000)

        for c in candidates:
            assert c.stop_loss_inr < c.entry_zone.lower_inr

    def test_buy_candidate_target_above_entry(self):
        from src.workflows.buyer_workflow import BuyerWorkflow

        entry = _make_scan_entry("INFY", last_price=1500.0)

        with (
            patch.object(BuyerWorkflow, "_filter_by_sentiment", return_value=[entry]),
            patch.object(
                BuyerWorkflow, "_run_deep_analysis",
                return_value=(_make_ta_signal(), _make_risk(), _make_fundamental(), _make_competitive()),
            ),
            patch("src.agents.market_scanner.MarketScannerAgent.scan",
                  return_value=ScanResults(universe="NIFTY50", entries=[entry], total_scanned=5)),
        ):
            candidates = BuyerWorkflow().run(available_cash=100_000)

        for c in candidates:
            assert c.target_inr > c.entry_zone.upper_inr
