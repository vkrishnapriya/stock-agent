"""
tests/unit/test_competitor_analyst.py
Unit tests for CompetitorMapTool, PeerPricePerformanceTool, and CompetitorAnalysisAgent.

All HTTP, yfinance, Redis, and LLM calls are mocked — no network access required.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Helpers shared across all test classes
# ---------------------------------------------------------------------------


def _patch_rate_limit():
    """Mock BaseTool._check_rate_limit so rate-limit Redis calls are skipped."""
    return patch(
        "src.tools.base_tool.BaseTool._check_rate_limit",
        new_callable=AsyncMock,
    )


@contextmanager
def _patch_cache():
    """Mock _get_cached (returns None) and _set_cached (no-op)."""
    with (
        patch(
            "src.tools.base_tool.BaseTool._get_cached",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "src.tools.base_tool.BaseTool._set_cached",
            new_callable=AsyncMock,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

_PEERS_HTML = """
<html><body>
<section id="peers">
  <table>
    <thead>
      <tr>
        <th>Name</th>
        <th>CMP ₹</th>
        <th>P/E</th>
        <th>Mar Cap ₹ Cr.</th>
        <th>Div Yld %</th>
        <th>NP Qtr ₹ Cr.</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="/company/TCS/">Tata Consultancy Services</a></td>
        <td>3,500.00</td>
        <td>28.5</td>
        <td>12,75,000</td>
        <td>1.5</td>
        <td>12,000</td>
      </tr>
      <tr>
        <td><a href="/company/WIPRO/">Wipro Ltd</a></td>
        <td>450.00</td>
        <td>22.1</td>
        <td>2,35,000</td>
        <td>0.8</td>
        <td>2,800</td>
      </tr>
      <tr>
        <td><a href="/company/HCLTECH/">HCL Technologies</a></td>
        <td>1,200.00</td>
        <td>24.8</td>
        <td>3,26,000</td>
        <td>2.1</td>
        <td>4,100</td>
      </tr>
      <tr>
        <td><a href="/company/TECHM/">Tech Mahindra</a></td>
        <td>1,100.00</td>
        <td>30.0</td>
        <td>1,05,000</td>
        <td>1.2</td>
        <td>1,500</td>
      </tr>
      <tr>
        <td><a href="/company/LTIM/">LTIMindtree</a></td>
        <td>5,500.00</td>
        <td>35.2</td>
        <td>1,63,000</td>
        <td>0.5</td>
        <td>2,100</td>
      </tr>
      <tr>
        <td><a href="/company/MPHASIS/">Mphasis</a></td>
        <td>2,000.00</td>
        <td>27.4</td>
        <td>37,000</td>
        <td>0.3</td>
        <td>380</td>
      </tr>
    </tbody>
  </table>
</section>
</body></html>
"""

_PEERS_HTML_NO_PE = """
<html><body>
<section id="peers">
  <table>
    <thead>
      <tr>
        <th>Name</th>
        <th>CMP ₹</th>
        <th>Mar Cap ₹ Cr.</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="/company/AXISBANK/">Axis Bank</a></td>
        <td>950.00</td>
        <td>2,95,000</td>
      </tr>
    </tbody>
  </table>
</section>
</body></html>
"""

_NO_PEERS_HTML = """
<html><body>
<section id="profit-loss">
  <p>No peers section here</p>
</section>
</body></html>
"""


def _make_mock_response(html: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# yfinance helpers for PeerPricePerformanceTool
# ---------------------------------------------------------------------------


def _make_single_ticker_df(start_price: float = 1000.0, end_price: float = 1100.0, rows: int = 65) -> pd.DataFrame:
    """Flat DataFrame (single ticker) with monotonically increasing closes."""
    dates = pd.date_range(end="2024-03-31", periods=rows, freq="B")
    prices = np.linspace(start_price, end_price, rows)
    return pd.DataFrame({"Close": prices}, index=dates)


def _make_multi_ticker_df(tickers: list[str], rows: int = 65) -> pd.DataFrame:
    """MultiIndex DataFrame for multiple tickers."""
    dates = pd.date_range(end="2024-03-31", periods=rows, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        start = 1000.0 + i * 200
        end = start * 1.1  # +10% over 3M
        data[("Close", t)] = np.linspace(start, end, rows)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ---------------------------------------------------------------------------
# TestCompetitorMapTool
# ---------------------------------------------------------------------------


class TestCompetitorMapTool:
    def test_returns_list_of_dicts(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_PEERS_HTML)
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="INFY"))

        assert isinstance(result, list)
        assert len(result) > 0

    def test_required_keys_per_peer(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_PEERS_HTML)
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="INFY"))

        for peer in result:
            assert "symbol" in peer
            assert "company_name" in peer
            assert "market_cap" in peer
            assert "pe_ratio" in peer

    def test_symbol_extracted_from_href(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_PEERS_HTML)
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="INFY"))

        symbols = [p["symbol"] for p in result]
        assert "TCS" in symbols
        assert "WIPRO" in symbols
        assert "HCLTECH" in symbols

    def test_pe_ratio_is_float(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_PEERS_HTML)
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="INFY"))

        tcs = next(p for p in result if p["symbol"] == "TCS")
        assert tcs["pe_ratio"] == pytest.approx(28.5)

    def test_market_cap_is_float(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_PEERS_HTML)
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="INFY"))

        tcs = next(p for p in result if p["symbol"] == "TCS")
        # "12,75,000" → 1275000.0
        assert tcs["market_cap"] == pytest.approx(1275000.0)

    def test_caps_at_max_peers(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_PEERS_HTML)
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="INFY", max_peers=3))

        # HTML has 6 rows but max_peers=3
        assert len(result) == 3

    def test_symbol_uppercased(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_PEERS_HTML)
            tool = CompetitorMapTool()
            tool._run(symbol="infy")  # lowercase input

        # httpx.get should have been called with the upper-cased symbol URL
        called_url = mock_get.call_args[0][0]
        assert "INFY" in called_url

    def test_2_second_delay_called(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep") as mock_sleep:
            mock_get.return_value = _make_mock_response(_PEERS_HTML)
            tool = CompetitorMapTool()
            tool._run(symbol="INFY")

        mock_sleep.assert_called_once_with(2.0)

    def test_cache_hit_skips_httpx(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        cached_peers = [{"symbol": "TCS", "company_name": "TCS", "market_cap": 1275000.0, "pe_ratio": 28.5}]
        with (
            _patch_rate_limit(),
            patch("src.tools.base_tool.BaseTool._get_cached", new_callable=AsyncMock, return_value=cached_peers),
            patch("src.tools.base_tool.BaseTool._set_cached", new_callable=AsyncMock),
            patch("httpx.get") as mock_get,
        ):
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="INFY"))

        mock_get.assert_not_called()
        assert result[0]["symbol"] == "TCS"

    def test_missing_peers_section_returns_empty(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_NO_PEERS_HTML)
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="INFY"))

        assert result == []

    def test_missing_pe_column_returns_none(self):
        from src.tools.market.screener_tools import CompetitorMapTool

        with _patch_rate_limit(), _patch_cache(), patch("httpx.get") as mock_get, patch("time.sleep"):
            mock_get.return_value = _make_mock_response(_PEERS_HTML_NO_PE)
            tool = CompetitorMapTool()
            result = json.loads(tool._run(symbol="HDFCBANK"))

        assert len(result) == 1
        assert result[0]["symbol"] == "AXISBANK"
        assert result[0]["pe_ratio"] is None


# ---------------------------------------------------------------------------
# TestParsePeers (unit tests for the parsing helper directly)
# ---------------------------------------------------------------------------


class TestParsePeers:
    def test_returns_empty_for_table_less_section(self):
        from src.tools.market.screener_tools import _parse_peers

        soup = BeautifulSoup("<section id='peers'><p>no table</p></section>", "html.parser")
        section = soup.find("section", {"id": "peers"})
        assert _parse_peers(section) == []  # type: ignore[arg-type]

    def test_skips_rows_without_company_link(self):
        from src.tools.market.screener_tools import _parse_peers

        html = """
        <section id="peers">
          <table>
            <thead><tr><th>Name</th><th>P/E</th><th>Mar Cap ₹ Cr.</th></tr></thead>
            <tbody>
              <tr><td>No link here</td><td>20.0</td><td>50000</td></tr>
              <tr><td><a href="/company/WIPRO/">Wipro</a></td><td>22.1</td><td>235000</td></tr>
            </tbody>
          </table>
        </section>"""
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find("section", {"id": "peers"})
        peers = _parse_peers(section)  # type: ignore[arg-type]
        assert len(peers) == 1
        assert peers[0]["symbol"] == "WIPRO"

    def test_handles_consolidated_href(self):
        from src.tools.market.screener_tools import _parse_peers

        html = """
        <section id="peers">
          <table>
            <thead><tr><th>Name</th><th>P/E</th><th>Mar Cap ₹ Cr.</th></tr></thead>
            <tbody>
              <tr>
                <td><a href="/company/HCLTECH/consolidated/">HCL Tech</a></td>
                <td>24.8</td><td>326000</td>
              </tr>
            </tbody>
          </table>
        </section>"""
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find("section", {"id": "peers"})
        peers = _parse_peers(section)  # type: ignore[arg-type]
        assert peers[0]["symbol"] == "HCLTECH"


# ---------------------------------------------------------------------------
# TestPeerPricePerformanceTool
# ---------------------------------------------------------------------------


class TestPeerPricePerformanceTool:
    def test_returns_dict_of_symbols(self):
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        df = _make_multi_ticker_df(["INFY.NS", "TCS.NS"])
        with _patch_rate_limit(), _patch_cache(), patch("yfinance.download", return_value=df):
            tool = PeerPricePerformanceTool()
            result = json.loads(tool._run(symbols_csv="INFY,TCS"))

        assert isinstance(result, dict)
        assert "INFY" in result
        assert "TCS" in result

    def test_required_keys_per_symbol(self):
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        df = _make_multi_ticker_df(["INFY.NS", "TCS.NS"])
        with _patch_rate_limit(), _patch_cache(), patch("yfinance.download", return_value=df):
            tool = PeerPricePerformanceTool()
            result = json.loads(tool._run(symbols_csv="INFY,TCS"))

        for sym in ["INFY", "TCS"]:
            assert "return_1m_pct" in result[sym]
            assert "return_3m_pct" in result[sym]
            assert "current_price" in result[sym]

    def test_3m_return_positive_for_rising_prices(self):
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        # Single ticker: start=1000, end=1100 → +10%
        df = _make_single_ticker_df(start_price=1000.0, end_price=1100.0, rows=65)
        with _patch_rate_limit(), _patch_cache(), patch("yfinance.download", return_value=df):
            tool = PeerPricePerformanceTool()
            result = json.loads(tool._run(symbols_csv="INFY"))

        assert result["INFY"]["return_3m_pct"] == pytest.approx(10.0, abs=0.5)

    def test_1m_return_computed_from_last_21_rows(self):
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        # 65 rows: price at row 44 (idx -21) = 1000 + (44/64)*100 ≈ 1068.75
        # current = 1100; 1M ≈ (1100/1068.75 - 1)*100 ≈ +2.9%
        df = _make_single_ticker_df(start_price=1000.0, end_price=1100.0, rows=65)
        with _patch_rate_limit(), _patch_cache(), patch("yfinance.download", return_value=df):
            tool = PeerPricePerformanceTool()
            result = json.loads(tool._run(symbols_csv="INFY"))

        return_1m = result["INFY"]["return_1m_pct"]
        return_3m = result["INFY"]["return_3m_pct"]
        # 1M return should be smaller than 3M return for a monotonically rising series
        assert return_1m is not None
        assert return_1m < return_3m

    def test_ns_suffix_added_to_download_call(self):
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        df = _make_multi_ticker_df(["INFY.NS", "TCS.NS"])
        with _patch_rate_limit(), _patch_cache(), patch("yfinance.download", return_value=df) as mock_dl:
            tool = PeerPricePerformanceTool()
            tool._run(symbols_csv="INFY,TCS")

        called_symbols = mock_dl.call_args[0][0]
        assert "INFY.NS" in called_symbols
        assert "TCS.NS" in called_symbols

    def test_insufficient_data_falls_back_1m_to_3m(self):
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        # Only 10 rows — fewer than _TRADING_DAYS_1M (21)
        df = _make_single_ticker_df(start_price=500.0, end_price=550.0, rows=10)
        with _patch_rate_limit(), _patch_cache(), patch("yfinance.download", return_value=df):
            tool = PeerPricePerformanceTool()
            result = json.loads(tool._run(symbols_csv="ZOMATO"))

        assert result["ZOMATO"]["return_1m_pct"] == result["ZOMATO"]["return_3m_pct"]

    def test_symbols_uppercased(self):
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        df = _make_multi_ticker_df(["INFY.NS", "TCS.NS"])
        with _patch_rate_limit(), _patch_cache(), patch("yfinance.download", return_value=df) as mock_dl:
            tool = PeerPricePerformanceTool()
            tool._run(symbols_csv="infy,tcs")

        called_symbols = mock_dl.call_args[0][0]
        assert "INFY.NS" in called_symbols
        assert "TCS.NS" in called_symbols

    def test_cache_hit_skips_yfinance(self):
        from src.tools.market.peer_performance import PeerPricePerformanceTool

        cached = {"INFY": {"return_1m_pct": 3.5, "return_3m_pct": 8.2, "current_price": 1500.0}}
        with (
            _patch_rate_limit(),
            patch("src.tools.base_tool.BaseTool._get_cached", new_callable=AsyncMock, return_value=cached),
            patch("src.tools.base_tool.BaseTool._set_cached", new_callable=AsyncMock),
            patch("yfinance.download") as mock_dl,
        ):
            tool = PeerPricePerformanceTool()
            result = json.loads(tool._run(symbols_csv="INFY"))

        mock_dl.assert_not_called()
        assert result["INFY"]["return_1m_pct"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# TestCompetitorAnalysisAgent
# ---------------------------------------------------------------------------


_MOCK_LLM = patch("src.agents.base_agent.get_llm", return_value=MagicMock())


class TestCompetitorAnalysisAgent:
    def test_build_returns_crewai_agent(self):
        from crewai import Agent

        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            crewai_agent = agent.build()
        assert isinstance(crewai_agent, Agent)

    def test_agent_has_three_tools(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        agent = CompetitorAnalysisAgent()
        assert len(agent.tools) == 3

    def test_tool_types(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent
        from src.tools.market.peer_performance import PeerPricePerformanceTool
        from src.tools.market.screener_tools import CompetitorMapTool
        from src.tools.news.gnews_tool import GNewsAPITool

        agent = CompetitorAnalysisAgent()
        tool_types = {type(t) for t in agent.tools}
        assert CompetitorMapTool in tool_types
        assert GNewsAPITool in tool_types
        assert PeerPricePerformanceTool in tool_types

    def test_agent_role_contains_sector(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            crewai_agent = agent.build()
        assert "sector" in crewai_agent.role.lower() or "competitive" in crewai_agent.role.lower()

    def test_agent_defaults_applied(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            crewai_agent = agent.build()
        # BaseAgent sets verbose=True and max_iter=5 by default
        assert crewai_agent.verbose is True
        assert crewai_agent.max_iter == 5

    def test_build_task_output_pydantic(self):
        from src.models.signals import CompetitiveAnalysis
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            task = agent.build_task(symbol="INFY", company_name="Infosys")
        assert task.output_pydantic is CompetitiveAnalysis

    def test_build_task_description_contains_symbol(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            task = agent.build_task(symbol="reliance", company_name="Reliance Industries")
        assert "RELIANCE" in task.description

    def test_build_task_mentions_all_three_tools(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            task = agent.build_task(symbol="HDFCBANK")
        assert "competitor_map" in task.description
        assert "gnews_search" in task.description
        assert "peer_price_performance" in task.description

    def test_build_task_includes_sector_when_provided(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            task = agent.build_task(symbol="INFY", sector="IT Services")
        assert "IT Services" in task.description

    def test_build_task_mentions_relative_strength(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            task = agent.build_task(symbol="TCS")
        assert "relative_strength" in task.description

    def test_build_task_mentions_moat_score(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        with _MOCK_LLM:
            agent = CompetitorAnalysisAgent()
            task = agent.build_task(symbol="TCS")
        assert "moat_score" in task.description

    def test_output_model_class_attribute(self):
        from src.models.signals import CompetitiveAnalysis
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        assert CompetitorAnalysisAgent.output_model is CompetitiveAnalysis

    def test_extra_tools_appended(self):
        from src.agents.competitor_analyst import CompetitorAnalysisAgent

        dummy = MagicMock()
        dummy.name = "dummy_tool"
        dummy.description = "dummy"
        agent = CompetitorAnalysisAgent(extra_tools=[dummy])
        assert len(agent.tools) == 4
        assert agent.tools[-1] is dummy
