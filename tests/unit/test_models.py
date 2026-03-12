"""
tests/unit/test_models.py
Instantiation and validation tests for all Pydantic models in src/models/.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

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

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def make_holding(**kw) -> Holding:
    defaults = dict(symbol="RELIANCE", exchange="NSE", quantity=10, avg_buy_price=2500.0)
    return Holding(**{**defaults, **kw})


def make_technical_signal(**kw) -> TechnicalSignal:
    defaults = dict(
        symbol="INFY",
        score=45.0,
        trend="BULLISH",
        rsi=58.0,
        macd_signal=2.3,
        support_level_inr=1800.0,
        resistance_level_inr=1950.0,
    )
    return TechnicalSignal(**{**defaults, **kw})


def make_risk_assessment(**kw) -> RiskAssessment:
    defaults = dict(
        symbol="TCS",
        risk_score=35.0,
        volatility_pct=18.5,
        beta=0.85,
        max_drawdown_pct=22.0,
        circuit_breaker_band="20%",
        suggested_stop_loss_inr=3600.0,
        position_size_pct=5.0,
    )
    return RiskAssessment(**{**defaults, **kw})


def make_fundamental_score(**kw) -> FundamentalScore:
    defaults = dict(
        symbol="HDFCBANK",
        score=72.0,
        pe_ratio=18.5,
        pb_ratio=2.8,
        roe_pct=16.0,
        debt_to_equity=0.9,
        revenue_growth_pct=12.0,
        earnings_growth_pct=15.0,
        promoter_holding_pct=26.0,
    )
    return FundamentalScore(**{**defaults, **kw})


def make_sentiment_result(**kw) -> SentimentResult:
    defaults = dict(
        symbol="WIPRO",
        score=30.0,
        news_sentiment="POSITIVE",
        social_sentiment="NEUTRAL",
        headline_count=12,
        key_themes=["AI adoption", "deal wins"],
    )
    return SentimentResult(**{**defaults, **kw})


def make_competitive_analysis(**kw) -> CompetitiveAnalysis:
    defaults = dict(
        symbol="INFY",
        sector="Information Technology",
        market_position="CHALLENGER",
        moat_score=68.0,
        relative_strength=20.0,
        peers=["TCS", "WIPRO", "HCLTECH"],
    )
    return CompetitiveAnalysis(**{**defaults, **kw})


def make_entry_zone(**kw) -> EntryZone:
    return EntryZone(**{**dict(lower_inr=1850.0, upper_inr=1900.0), **kw})


def make_buy_candidate(**kw) -> BuyCandidate:
    defaults = dict(
        symbol="INFY",
        score=78.0,
        suggested_allocation_inr=50000.0,
        entry_zone=make_entry_zone(),
        stop_loss_inr=1780.0,
        target_inr=2100.0,
    )
    return BuyCandidate(**{**defaults, **kw})


def make_stock_decision(**kw) -> StockDecision:
    defaults = dict(
        symbol="RELIANCE",
        action="HOLD",
        confidence=0.82,
        rationale="Strong support at 2400; RSI not yet oversold.",
        stop_loss_inr=2350.0,
    )
    return StockDecision(**{**defaults, **kw})


def make_final_report(**kw) -> FinalReport:
    defaults = dict(
        workflow_type="SELLER",
        decisions=[make_stock_decision()],
        summary="Portfolio review complete. One HOLD decision generated.",
    )
    return FinalReport(**{**defaults, **kw})


# ===========================================================================
# Holding
# ===========================================================================

class TestHolding:
    def test_basic_instantiation(self):
        h = make_holding()
        assert h.symbol == "RELIANCE"
        assert h.quantity == 10

    def test_symbol_uppercased(self):
        h = make_holding(symbol="reliance")
        assert h.symbol == "RELIANCE"

    def test_quantity_zero_allowed(self):
        h = make_holding(quantity=0)
        assert h.quantity == 0

    def test_negative_quantity_raises(self):
        with pytest.raises(ValidationError):
            make_holding(quantity=-1)

    def test_zero_avg_buy_price_raises(self):
        with pytest.raises(ValidationError):
            make_holding(avg_buy_price=0.0)

    def test_market_value_none_without_current_price(self):
        h = make_holding(current_price=None)
        assert h.market_value_inr is None

    def test_market_value_computed(self):
        h = make_holding(quantity=10, avg_buy_price=2500.0, current_price=2600.0)
        assert h.market_value_inr == pytest.approx(26000.0)

    def test_unrealised_pnl_positive(self):
        h = make_holding(quantity=10, avg_buy_price=2500.0, current_price=2600.0)
        assert h.unrealised_pnl_inr == pytest.approx(1000.0)

    def test_unrealised_pnl_pct(self):
        h = make_holding(quantity=10, avg_buy_price=2500.0, current_price=2750.0)
        assert h.unrealised_pnl_pct == pytest.approx(10.0)

    def test_invalid_exchange_raises(self):
        with pytest.raises(ValidationError):
            make_holding(exchange="NYSE")


# ===========================================================================
# Portfolio
# ===========================================================================

class TestPortfolio:
    def test_basic_instantiation(self):
        p = Portfolio(portfolio_id="p1", holdings=[make_holding()], available_cash_inr=10000.0)
        assert p.portfolio_id == "p1"
        assert len(p.holdings) == 1

    def test_empty_holdings(self):
        p = Portfolio(portfolio_id="empty")
        assert p.holdings == []

    def test_negative_cash_raises(self):
        with pytest.raises(ValidationError):
            Portfolio(portfolio_id="p1", available_cash_inr=-1.0)

    def test_total_invested(self):
        p = Portfolio(
            portfolio_id="p1",
            holdings=[make_holding(quantity=10, avg_buy_price=2500.0)],
        )
        assert p.total_invested_inr == pytest.approx(25000.0)

    def test_total_market_value_none_without_prices(self):
        p = Portfolio(portfolio_id="p1", holdings=[make_holding(current_price=None)])
        assert p.total_market_value_inr is None

    def test_total_value_with_prices(self):
        p = Portfolio(
            portfolio_id="p1",
            holdings=[make_holding(quantity=10, avg_buy_price=2500.0, current_price=3000.0)],
            available_cash_inr=5000.0,
        )
        assert p.total_value_inr == pytest.approx(35000.0)

    def test_load_from_json(self):
        sample = Path("data/portfolios/sample_portfolio.json")
        if not sample.exists():
            pytest.skip("sample_portfolio.json not present")
        p = Portfolio.load_from_file(sample)
        assert p.portfolio_id == "sample-001"
        assert len(p.holdings) > 0

    def test_load_from_json_string_path(self):
        sample = "data/portfolios/sample_portfolio.json"
        if not Path(sample).exists():
            pytest.skip("sample_portfolio.json not present")
        p = Portfolio.load_from_file(sample)
        assert isinstance(p, Portfolio)

    def test_load_from_csv(self, tmp_path):
        csv_file = tmp_path / "holdings.csv"
        csv_file.write_text(
            "symbol,exchange,quantity,avg_buy_price,sector\n"
            "TCS,NSE,5,4000.00,IT\n"
            "INFY,NSE,10,1800.00,IT\n",
            encoding="utf-8",
        )
        p = Portfolio.load_from_file(csv_file)
        assert p.portfolio_id == "holdings"
        assert len(p.holdings) == 2
        assert p.holdings[0].symbol == "TCS"

    def test_load_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_text("dummy", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file format"):
            Portfolio.load_from_file(f)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Portfolio.load_from_file("nonexistent/path.json")


# ===========================================================================
# TechnicalSignal
# ===========================================================================

class TestTechnicalSignal:
    def test_basic_instantiation(self):
        s = make_technical_signal()
        assert s.symbol == "INFY"
        assert s.trend == "BULLISH"

    def test_symbol_uppercased(self):
        s = make_technical_signal(symbol="infy")
        assert s.symbol == "INFY"

    def test_score_bounds(self):
        make_technical_signal(score=-100.0)
        make_technical_signal(score=100.0)

    def test_score_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_technical_signal(score=101.0)
        with pytest.raises(ValidationError):
            make_technical_signal(score=-101.0)

    def test_rsi_bounds(self):
        make_technical_signal(rsi=0.0)
        make_technical_signal(rsi=100.0)

    def test_rsi_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_technical_signal(rsi=101.0)

    def test_resistance_must_exceed_support(self):
        with pytest.raises(ValidationError, match="resistance_level_inr"):
            make_technical_signal(support_level_inr=2000.0, resistance_level_inr=1999.0)

    def test_generated_at_is_datetime(self):
        s = make_technical_signal()
        assert isinstance(s.generated_at, datetime)

    @pytest.mark.parametrize("trend", ["BULLISH", "BEARISH", "NEUTRAL"])
    def test_valid_trends(self, trend):
        s = make_technical_signal(trend=trend)
        assert s.trend == trend

    def test_invalid_trend_raises(self):
        with pytest.raises(ValidationError):
            make_technical_signal(trend="SIDEWAYS")


# ===========================================================================
# RiskAssessment
# ===========================================================================

class TestRiskAssessment:
    def test_basic_instantiation(self):
        r = make_risk_assessment()
        assert r.symbol == "TCS"
        assert r.risk_score == pytest.approx(35.0)

    def test_risk_score_bounds(self):
        make_risk_assessment(risk_score=0.0)
        make_risk_assessment(risk_score=100.0)

    def test_risk_score_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_risk_assessment(risk_score=101.0)

    def test_max_drawdown_bounds(self):
        make_risk_assessment(max_drawdown_pct=0.0)
        make_risk_assessment(max_drawdown_pct=100.0)

    @pytest.mark.parametrize("band", ["5%", "10%", "20%"])
    def test_valid_circuit_breaker_bands(self, band):
        r = make_risk_assessment(circuit_breaker_band=band)
        assert r.circuit_breaker_band == band

    def test_invalid_circuit_breaker_band_raises(self):
        with pytest.raises(ValidationError):
            make_risk_assessment(circuit_breaker_band="15%")

    def test_stop_loss_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_risk_assessment(suggested_stop_loss_inr=0.0)


# ===========================================================================
# FundamentalScore
# ===========================================================================

class TestFundamentalScore:
    def test_basic_instantiation(self):
        f = make_fundamental_score()
        assert f.symbol == "HDFCBANK"
        assert f.score == pytest.approx(72.0)

    def test_optional_fields_can_be_none(self):
        f = FundamentalScore(symbol="XYZ", score=50.0)
        assert f.pe_ratio is None
        assert f.promoter_holding_pct is None

    def test_score_bounds(self):
        make_fundamental_score(score=0.0)
        make_fundamental_score(score=100.0)

    def test_score_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_fundamental_score(score=-1.0)

    def test_promoter_holding_bounds(self):
        make_fundamental_score(promoter_holding_pct=0.0)
        make_fundamental_score(promoter_holding_pct=100.0)

    def test_promoter_holding_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_fundamental_score(promoter_holding_pct=101.0)


# ===========================================================================
# SentimentResult
# ===========================================================================

class TestSentimentResult:
    def test_basic_instantiation(self):
        s = make_sentiment_result()
        assert s.symbol == "WIPRO"
        assert s.score == pytest.approx(30.0)

    def test_score_bounds(self):
        make_sentiment_result(score=-100.0)
        make_sentiment_result(score=100.0)

    def test_score_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_sentiment_result(score=101.0)

    def test_negative_headline_count_raises(self):
        with pytest.raises(ValidationError):
            make_sentiment_result(headline_count=-1)

    @pytest.mark.parametrize("sentiment", ["POSITIVE", "NEGATIVE", "NEUTRAL"])
    def test_valid_news_sentiments(self, sentiment):
        s = make_sentiment_result(news_sentiment=sentiment)
        assert s.news_sentiment == sentiment

    def test_key_themes_default_empty(self):
        s = SentimentResult(
            symbol="ABC",
            score=0.0,
            news_sentiment="NEUTRAL",
            social_sentiment="NEUTRAL",
            headline_count=0,
        )
        assert s.key_themes == []


# ===========================================================================
# CompetitiveAnalysis
# ===========================================================================

class TestCompetitiveAnalysis:
    def test_basic_instantiation(self):
        c = make_competitive_analysis()
        assert c.symbol == "INFY"
        assert c.sector == "Information Technology"

    def test_peers_uppercased(self):
        c = make_competitive_analysis(peers=["tcs", "wipro"])
        assert c.peers == ["TCS", "WIPRO"]

    def test_moat_score_bounds(self):
        make_competitive_analysis(moat_score=0.0)
        make_competitive_analysis(moat_score=100.0)

    def test_moat_score_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_competitive_analysis(moat_score=101.0)

    def test_relative_strength_bounds(self):
        make_competitive_analysis(relative_strength=-100.0)
        make_competitive_analysis(relative_strength=100.0)

    @pytest.mark.parametrize("pos", ["LEADER", "CHALLENGER", "FOLLOWER", "NICHE"])
    def test_valid_market_positions(self, pos):
        c = make_competitive_analysis(market_position=pos)
        assert c.market_position == pos

    def test_invalid_market_position_raises(self):
        with pytest.raises(ValidationError):
            make_competitive_analysis(market_position="DISRUPTOR")


# ===========================================================================
# EntryZone
# ===========================================================================

class TestEntryZone:
    def test_basic_instantiation(self):
        ez = make_entry_zone()
        assert ez.lower_inr == pytest.approx(1850.0)
        assert ez.upper_inr == pytest.approx(1900.0)

    def test_lower_must_be_less_than_upper(self):
        with pytest.raises(ValidationError, match="lower_inr"):
            EntryZone(lower_inr=1900.0, upper_inr=1850.0)

    def test_equal_bounds_raise(self):
        with pytest.raises(ValidationError):
            EntryZone(lower_inr=1900.0, upper_inr=1900.0)

    def test_zero_lower_raises(self):
        with pytest.raises(ValidationError):
            EntryZone(lower_inr=0.0, upper_inr=100.0)


# ===========================================================================
# StockDecision
# ===========================================================================

class TestStockDecision:
    def test_hold_decision(self):
        d = make_stock_decision(action="HOLD")
        assert d.action == "HOLD"

    def test_sell_decision(self):
        d = make_stock_decision(action="SELL")
        assert d.action == "SELL"

    def test_confidence_bounds(self):
        make_stock_decision(confidence=0.0)
        make_stock_decision(confidence=1.0)

    def test_confidence_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_stock_decision(confidence=1.1)

    def test_rationale_too_short_raises(self):
        with pytest.raises(ValidationError):
            make_stock_decision(rationale="short")

    def test_invalid_action_raises(self):
        with pytest.raises(ValidationError):
            make_stock_decision(action="BUY")

    def test_stop_loss_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_stock_decision(stop_loss_inr=0.0)

    def test_symbol_uppercased(self):
        d = make_stock_decision(symbol="reliance")
        assert d.symbol == "RELIANCE"


# ===========================================================================
# BuyCandidate
# ===========================================================================

class TestBuyCandidate:
    def test_basic_instantiation(self):
        b = make_buy_candidate()
        assert b.symbol == "INFY"
        assert b.score == pytest.approx(78.0)

    def test_score_bounds(self):
        make_buy_candidate(score=0.0)
        make_buy_candidate(score=100.0)

    def test_score_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            make_buy_candidate(score=101.0)

    def test_stop_loss_must_be_below_entry(self):
        with pytest.raises(ValidationError, match="stop_loss_inr"):
            make_buy_candidate(
                entry_zone=EntryZone(lower_inr=1850.0, upper_inr=1900.0),
                stop_loss_inr=1860.0,  # inside entry zone — invalid
                target_inr=2100.0,
            )

    def test_target_must_be_above_entry(self):
        with pytest.raises(ValidationError, match="target_inr"):
            make_buy_candidate(
                entry_zone=EntryZone(lower_inr=1850.0, upper_inr=1900.0),
                stop_loss_inr=1780.0,
                target_inr=1890.0,  # inside entry zone — invalid
            )

    def test_suggested_allocation_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_buy_candidate(suggested_allocation_inr=0.0)


# ===========================================================================
# FinalReport
# ===========================================================================

class TestFinalReport:
    def test_basic_instantiation(self):
        r = make_final_report()
        assert r.workflow_type == "SELLER"
        assert r.decision_count == 1

    def test_default_disclaimer_present(self):
        r = make_final_report()
        assert "SEBI" in r.disclaimer
        assert "RECOMMENDATIONS ONLY" in r.disclaimer

    def test_custom_disclaimer(self):
        r = make_final_report(disclaimer="Custom disclaimer text.")
        assert r.disclaimer == "Custom disclaimer text."

    def test_empty_summary_raises(self):
        with pytest.raises(ValidationError):
            make_final_report(summary="")

    @pytest.mark.parametrize("wf", ["BUYER", "SELLER", "PORTFOLIO_REVIEW"])
    def test_valid_workflow_types(self, wf):
        r = make_final_report(workflow_type=wf)
        assert r.workflow_type == wf

    def test_invalid_workflow_type_raises(self):
        with pytest.raises(ValidationError):
            make_final_report(workflow_type="SCREENER")

    def test_sell_decisions_filter(self):
        decisions = [
            make_stock_decision(action="SELL", symbol="INFY"),
            make_stock_decision(action="HOLD", symbol="TCS"),
        ]
        r = make_final_report(decisions=decisions)
        assert len(r.sell_decisions) == 1
        assert r.sell_decisions[0].symbol == "INFY"

    def test_hold_decisions_filter(self):
        decisions = [
            make_stock_decision(action="HOLD", symbol="TCS"),
            make_stock_decision(action="HOLD", symbol="WIPRO"),
        ]
        r = make_final_report(decisions=decisions)
        assert len(r.hold_decisions) == 2

    def test_buy_candidates_filter(self):
        decisions = [make_stock_decision(), make_buy_candidate()]
        r = make_final_report(workflow_type="BUYER", decisions=decisions)
        assert len(r.buy_candidates) == 1
        assert r.buy_candidates[0].symbol == "INFY"

    def test_timestamp_is_datetime(self):
        r = make_final_report()
        assert isinstance(r.timestamp, datetime)

    def test_mixed_decisions(self):
        decisions = [
            make_stock_decision(action="SELL"),
            make_stock_decision(action="HOLD"),
            make_buy_candidate(),
        ]
        r = make_final_report(workflow_type="PORTFOLIO_REVIEW", decisions=decisions)
        assert r.decision_count == 3
        assert len(r.sell_decisions) == 1
        assert len(r.hold_decisions) == 1
        assert len(r.buy_candidates) == 1
