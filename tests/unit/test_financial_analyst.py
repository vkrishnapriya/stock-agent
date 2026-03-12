"""
tests/unit/test_financial_analyst.py
Unit tests for Financial Analysis Agent and its tools.

Network calls (httpx to Screener.in, NSESession to NSE API) are all mocked.
Redis is bypassed via patched _get_cached / _set_cached helpers.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_rate_limit():
    return patch(
        "src.tools.base_tool.BaseTool._check_rate_limit",
        new_callable=AsyncMock,
    )


@contextmanager
def _patch_cache():
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
# Screener.in HTML fixtures
# ---------------------------------------------------------------------------

_SCREENER_HTML = """
<html><body>

<ul id="top-ratios">
  <li><span class="name">Stock P/E</span><span class="number">17.8</span></li>
  <li><span class="name">Book Value</span><span class="number">205</span></li>
  <li><span class="name">ROE</span><span class="number">28.8</span></li>
  <li><span class="name">ROCE</span><span class="number">37.5</span></li>
  <li><span class="name">Market Cap</span><span class="number">5,13,311</span></li>
</ul>

<section id="profit-loss">
  <table>
    <thead><tr><th></th><th>Mar 2022</th><th>Mar 2023</th><th>Mar 2024</th></tr></thead>
    <tbody>
      <tr><td>Sales&#160;+</td><td>121641</td><td>146767</td><td>153670</td></tr>
      <tr><td>Net Profit&#160;+</td><td>22110</td><td>24108</td><td>26248</td></tr>
      <tr><td>EPS in Rs</td><td>52.52</td><td>57.40</td><td>62.39</td></tr>
    </tbody>
  </table>
</section>

<section id="balance-sheet">
  <table>
    <thead><tr><th></th><th>Mar 2022</th><th>Mar 2023</th><th>Mar 2024</th></tr></thead>
    <tbody>
      <tr><td>Equity Capital</td><td>572</td><td>572</td><td>2120</td></tr>
      <tr><td>Reserves</td><td>75000</td><td>80000</td><td>88000</td></tr>
      <tr><td>Borrowings</td><td>1000</td><td>1200</td><td>1500</td></tr>
      <tr><td>Total Assets</td><td>160000</td><td>180000</td><td>200000</td></tr>
    </tbody>
  </table>
</section>

<section id="cash-flow">
  <table>
    <thead><tr><th></th><th>Mar 2022</th><th>Mar 2023</th><th>Mar 2024</th></tr></thead>
    <tbody>
      <tr><td>Cash from Operating Activity&#160;+</td><td>28000</td><td>30000</td><td>32000</td></tr>
      <tr><td>Cash from Investing Activity</td><td>-5000</td><td>-6000</td><td>-8000</td></tr>
      <tr><td>Cash from Financing Activity</td><td>-10000</td><td>-12000</td><td>-15000</td></tr>
    </tbody>
  </table>
</section>

<section id="quarters">
  <table>
    <thead><tr><th></th><th>Dec 2022</th><th>Mar 2023</th><th>Jun 2023</th><th>Sep 2023</th>
                       <th>Dec 2023</th><th>Mar 2024</th><th>Jun 2024</th><th>Sep 2024</th></tr></thead>
    <tbody>
      <tr><td>Sales&#160;+</td><td>38318</td><td>37441</td><td>37933</td><td>38994</td>
                               <td>38821</td><td>37923</td><td>39315</td><td>40986</td></tr>
      <tr><td>Net Profit&#160;+</td><td>6586</td><td>6134</td><td>5945</td><td>6215</td>
                                    <td>6113</td><td>7975</td><td>6374</td><td>6506</td></tr>
      <tr><td>EPS in Rs</td><td>15.70</td><td>14.77</td><td>14.32</td><td>14.97</td>
                             <td>14.71</td><td>19.20</td><td>15.34</td><td>15.64</td></tr>
      <tr><td>Raw PDF</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    </tbody>
  </table>
</section>

<section id="shareholding">
  <table>
    <thead><tr><th></th><th>Dec 2023</th><th>Mar 2024</th><th>Jun 2024</th></tr></thead>
    <tbody>
      <tr><td>Promoters&#160;+</td><td>14.78%</td><td>14.60%</td><td>14.50%</td></tr>
      <tr><td>FIIs&#160;+</td><td>33.70%</td><td>34.00%</td><td>33.80%</td></tr>
      <tr><td>DIIs&#160;+</td><td>35.51%</td><td>35.20%</td><td>35.50%</td></tr>
    </tbody>
  </table>
</section>

</body></html>
"""


def _mock_httpx_response(html: str = _SCREENER_HTML) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# NSE quarterly results fixture
# ---------------------------------------------------------------------------

_NSE_RESULTS_LIST = [
    {
        "symbol": "INFY", "consolidated": "Consolidated", "period": "Quarterly",
        "relatingTo": "Third Quarter", "financialYear": "01-Apr-2025 To 31-Mar-2026",
        "fromDate": "01-Oct-2025", "toDate": "31-Dec-2025",
        "params": "01-Oct-202531-Dec-2025Q3ANNCNEINFY",
        "seqNumber": "9999999", "indAs": "Ind-AS New", "format": "New",
        "bank": "N", "audited": "Audited", "filingDate": "15-Jan-2026 19:00",
    },
    {
        "symbol": "INFY", "consolidated": "Consolidated", "period": "Quarterly",
        "relatingTo": "Second Quarter", "financialYear": "01-Apr-2025 To 31-Mar-2026",
        "fromDate": "01-Jul-2025", "toDate": "30-Sep-2025",
        "params": "01-Jul-202530-Sep-2025Q2ANNCNEINFY",
        "seqNumber": "9999998", "indAs": "Ind-AS New", "format": "New",
        "bank": "N", "audited": "Audited", "filingDate": "15-Oct-2025 19:00",
    },
]

_NSE_RESULTS_DETAIL = {
    "resultsData2": {
        "re_net_sale": "4098600",
        "re_net_profit": "650600",
        "re_basic_eps_for_cont_dic_opr": "15.64",
    }
}


# ---------------------------------------------------------------------------
# ScreenerFinancialsTool tests
# ---------------------------------------------------------------------------


class TestScreenerFinancialsTool:
    def test_returns_all_top_level_keys(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=_mock_httpx_response()):
                with patch("src.tools.market.screener_tools.time.sleep"):
                    tool = ScreenerFinancialsTool()
                    result = json.loads(tool._run(symbol="INFY"))

        assert {"symbol", "source_url", "key_ratios", "annual_pl",
                "balance_sheet", "cash_flows", "quarterly_results",
                "shareholding_latest"} == set(result.keys())

    def test_key_ratios_extracted(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=_mock_httpx_response()):
                with patch("src.tools.market.screener_tools.time.sleep"):
                    tool = ScreenerFinancialsTool()
                    result = json.loads(tool._run(symbol="INFY"))

        ratios = result["key_ratios"]
        assert ratios["Stock P/E"] == "17.8"
        assert ratios["ROE"] == "28.8"
        assert ratios["Book Value"] == "205"

    def test_annual_pl_has_sales_and_profit(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=_mock_httpx_response()):
                with patch("src.tools.market.screener_tools.time.sleep"):
                    tool = ScreenerFinancialsTool()
                    result = json.loads(tool._run(symbol="INFY"))

        pl = result["annual_pl"]
        assert "Sales" in pl
        assert "Net Profit" in pl
        assert "EPS in Rs" in pl
        # Verify year mapping
        assert pl["Sales"]["Mar 2024"] == "153670"

    def test_quarterly_results_capped_at_eight(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=_mock_httpx_response()):
                with patch("src.tools.market.screener_tools.time.sleep"):
                    tool = ScreenerFinancialsTool()
                    result = json.loads(tool._run(symbol="INFY"))

        qr = result["quarterly_results"]
        # Each row should have at most 8 columns
        for row_vals in qr.values():
            assert len(row_vals) <= 8

    def test_raw_pdf_row_excluded(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=_mock_httpx_response()):
                with patch("src.tools.market.screener_tools.time.sleep"):
                    tool = ScreenerFinancialsTool()
                    result = json.loads(tool._run(symbol="INFY"))

        # "Raw PDF" is a non-data row and must be excluded
        assert "Raw PDF" not in result["quarterly_results"]

    def test_shareholding_latest_column_only(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=_mock_httpx_response()):
                with patch("src.tools.market.screener_tools.time.sleep"):
                    tool = ScreenerFinancialsTool()
                    result = json.loads(tool._run(symbol="INFY"))

        sh = result["shareholding_latest"]
        # Latest column is "Jun 2024" → should be those values
        assert sh["Promoters"] == "14.50%"
        assert sh["FIIs"] == "33.80%"

    def test_symbol_uppercased(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=_mock_httpx_response()):
                with patch("src.tools.market.screener_tools.time.sleep"):
                    tool = ScreenerFinancialsTool()
                    result = json.loads(tool._run(symbol="infy"))

        assert result["symbol"] == "INFY"

    def test_2_second_delay_called(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=_mock_httpx_response()):
                with patch("src.tools.market.screener_tools.time.sleep") as mock_sleep:
                    tool = ScreenerFinancialsTool()
                    tool._run(symbol="INFY")

        mock_sleep.assert_called_once_with(2.0)

    def test_cache_hit_skips_httpx(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        cached = json.dumps({"symbol": "INFY", "key_ratios": {"Stock P/E": "18"},
                              "annual_pl": {}, "balance_sheet": {}, "cash_flows": {},
                              "quarterly_results": {}, "shareholding_latest": {},
                              "source_url": ""})
        with _patch_rate_limit():
            with (
                patch("src.tools.base_tool.BaseTool._get_cached",
                      new_callable=AsyncMock, return_value=cached),
                patch("src.tools.base_tool.BaseTool._set_cached",
                      new_callable=AsyncMock),
                patch("src.tools.market.screener_tools.httpx.get") as mock_http,
            ):
                tool = ScreenerFinancialsTool()
                result = json.loads(tool._run(symbol="INFY"))

        mock_http.assert_not_called()
        assert result["key_ratios"]["Stock P/E"] == "18"

    def test_http_error_propagates(self):
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = Exception("404 Not Found")
        with _patch_rate_limit(), _patch_cache():
            with patch("src.tools.market.screener_tools.httpx.get",
                       return_value=err_resp):
                with patch("src.tools.market.screener_tools.time.sleep"):
                    tool = ScreenerFinancialsTool()
                    with pytest.raises(Exception):
                        tool._run(symbol="BADSTOCK")


# ---------------------------------------------------------------------------
# Screener parsing helpers (unit tests)
# ---------------------------------------------------------------------------


class TestScreenerParsers:
    def test_clean_strips_nbsp_and_plus(self):
        from src.tools.market.screener_tools import _clean

        assert _clean("Sales\xa0+") == "Sales"
        assert _clean("  Net Profit  ") == "Net Profit"
        assert _clean("Cash from Operating Activity\xa0+") == "Cash from Operating Activity"

    def test_parse_top_ratios_missing_section(self):
        from bs4 import BeautifulSoup
        from src.tools.market.screener_tools import _parse_top_ratios

        soup = BeautifulSoup("<html></html>", "html.parser")
        assert _parse_top_ratios(soup) == {}

    def test_last_n_cols_keeps_only_recent(self):
        from src.tools.market.screener_tools import _last_n_cols

        data = {
            "Sales": {"Mar 2020": "100", "Mar 2021": "110", "Mar 2022": "120",
                      "Mar 2023": "130", "Mar 2024": "140"},
        }
        result = _last_n_cols(data, 3)
        assert set(result["Sales"].keys()) == {"Mar 2022", "Mar 2023", "Mar 2024"}

    def test_parse_table_excludes_empty_row_labels(self):
        from bs4 import BeautifulSoup, Tag
        from src.tools.market.screener_tools import _parse_table

        html = """
        <section id="test">
        <table>
          <thead><tr><th></th><th>2024</th></tr></thead>
          <tbody>
            <tr><td>Sales</td><td>100</td></tr>
            <tr><td></td><td>empty label</td></tr>
          </tbody>
        </table>
        </section>
        """
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find("section")
        result = _parse_table(section)  # type: ignore[arg-type]
        assert "Sales" in result
        assert "" not in result


# ---------------------------------------------------------------------------
# NSEResultsTool tests
# ---------------------------------------------------------------------------


class TestNSEResultsTool:
    def _make_session_mock(
        self,
        list_data: list | None = None,
        detail_data: dict | None = None,
    ) -> MagicMock:
        session = MagicMock()
        list_data = list_data if list_data is not None else _NSE_RESULTS_LIST
        detail_data = detail_data if detail_data is not None else _NSE_RESULTS_DETAIL

        call_count = [0]

        def _get(url, **kwargs):
            call_count[0] += 1
            if "financial-results-data" in url:
                return detail_data
            return list_data

        session.get.side_effect = _get
        session.close = MagicMock()
        return session

    def test_returns_list_of_quarters(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        with patch("src.tools.market.nse_fetcher.NSESession",
                   return_value=self._make_session_mock()):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="INFY"))

        assert isinstance(result, list)
        assert len(result) == 2   # 2 items in our fixture

    def test_required_keys_present(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        with patch("src.tools.market.nse_fetcher.NSESession",
                   return_value=self._make_session_mock()):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="INFY"))

        for q in result:
            assert {"period", "from_date", "to_date", "revenue_cr",
                    "pat_cr", "basic_eps", "audited", "filing_date"} <= set(q.keys())

    def test_revenue_converted_from_lakhs_to_crore(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        with patch("src.tools.market.nse_fetcher.NSESession",
                   return_value=self._make_session_mock()):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="INFY"))

        # 4098600 lakhs / 100 = 40986.0 crore
        assert result[0]["revenue_cr"] == pytest.approx(40986.0, rel=0.01)

    def test_pat_converted_from_lakhs_to_crore(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        with patch("src.tools.market.nse_fetcher.NSESession",
                   return_value=self._make_session_mock()):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="INFY"))

        # 650600 / 100 = 6506 crore
        assert result[0]["pat_cr"] == pytest.approx(6506.0, rel=0.01)

    def test_eps_extracted_correctly(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        with patch("src.tools.market.nse_fetcher.NSESession",
                   return_value=self._make_session_mock()):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="INFY"))

        assert result[0]["basic_eps"] == pytest.approx(15.64)

    def test_period_label_format(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        with patch("src.tools.market.nse_fetcher.NSESession",
                   return_value=self._make_session_mock()):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="INFY"))

        # "Third Quarter" in FY26 → "Q3 FY26"
        assert result[0]["period"] == "Q3 FY26"

    def test_filters_only_consolidated_quarterly(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        mixed_list = _NSE_RESULTS_LIST + [
            {
                **_NSE_RESULTS_LIST[0],
                "consolidated": "Non-Consolidated",
                "seqNumber": "8888888",
            },
            {
                **_NSE_RESULTS_LIST[0],
                "period": "Annual",
                "seqNumber": "7777777",
            },
        ]
        with patch("src.tools.market.nse_fetcher.NSESession",
                   return_value=self._make_session_mock(list_data=mixed_list)):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="INFY"))

        # Only 2 consolidated quarterly items should remain
        assert len(result) == 2

    def test_detail_failure_returns_none_fields(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        session = MagicMock()

        def _get(url, **kwargs):
            if "financial-results-data" in url:
                raise Exception("timeout")
            return _NSE_RESULTS_LIST

        session.get.side_effect = _get
        session.close = MagicMock()

        with patch("src.tools.market.nse_fetcher.NSESession", return_value=session):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="INFY"))

        # revenue_cr/pat_cr/basic_eps should be None on detail failure
        assert result[0]["revenue_cr"] is None
        assert result[0]["pat_cr"] is None
        assert result[0]["basic_eps"] is None

    def test_empty_list_returns_empty(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        with patch("src.tools.market.nse_fetcher.NSESession",
                   return_value=self._make_session_mock(list_data=[])):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                result = json.loads(tool._run(symbol="UNKNOWN"))

        assert result == []

    def test_symbol_uppercased(self):
        from src.tools.market.nse_fetcher import NSEResultsTool

        captured_urls: list[str] = []

        def _get(url, **kwargs):
            captured_urls.append(url)
            if "financial-results-data" in url:
                return _NSE_RESULTS_DETAIL
            return _NSE_RESULTS_LIST

        session = MagicMock()
        session.get.side_effect = _get
        session.close = MagicMock()

        with patch("src.tools.market.nse_fetcher.NSESession", return_value=session):
            with patch("src.tools.market.nse_fetcher.time.sleep"):
                tool = NSEResultsTool()
                tool._run(symbol="infy")

        assert "INFY" in captured_urls[0]


# ---------------------------------------------------------------------------
# FinancialAnalysisAgent tests
# ---------------------------------------------------------------------------


class TestFinancialAnalysisAgent:
    def test_build_returns_crewai_agent(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = FinancialAnalysisAgent()
            crewai_agent = agent.build()

        from crewai import Agent
        assert isinstance(crewai_agent, Agent)

    def test_agent_has_two_tools(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        agent = FinancialAnalysisAgent()
        assert len(agent.tools) == 2

    def test_tool_types(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent
        from src.tools.market.nse_fetcher import NSEResultsTool
        from src.tools.market.screener_tools import ScreenerFinancialsTool

        agent = FinancialAnalysisAgent()
        tool_types = {type(t) for t in agent.tools}
        assert tool_types == {ScreenerFinancialsTool, NSEResultsTool}

    def test_agent_role_contains_fundamental(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = FinancialAnalysisAgent()
            crewai_agent = agent.build()

        assert "Fundamental" in crewai_agent.role or "Financial" in crewai_agent.role

    def test_agent_defaults_applied(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = FinancialAnalysisAgent()
            crewai_agent = agent.build()

        assert crewai_agent.verbose is True
        assert crewai_agent.max_iter == 5

    def test_build_task_output_pydantic(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent
        from src.models.signals import FundamentalScore

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = FinancialAnalysisAgent()
            task = agent.build_task(symbol="INFY", company_name="Infosys")

        assert task.output_pydantic is FundamentalScore

    def test_build_task_description_contains_symbol(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = FinancialAnalysisAgent()
            task = agent.build_task(symbol="RELIANCE", company_name="Reliance Industries")

        assert "RELIANCE" in task.description
        assert "Reliance Industries" in task.description

    def test_build_task_mentions_both_tools(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = FinancialAnalysisAgent()
            task = agent.build_task(symbol="INFY")

        assert "screener_financials" in task.description
        assert "nse_quarterly_results" in task.description

    def test_build_task_mentions_key_metrics(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = FinancialAnalysisAgent()
            task = agent.build_task(symbol="INFY")

        desc_lower = task.description.lower()
        assert "roe" in desc_lower
        assert "debt" in desc_lower
        assert "cagr" in desc_lower
        assert "peg" in desc_lower

    def test_build_task_with_sector(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = FinancialAnalysisAgent()
            task = agent.build_task(symbol="INFY", sector="Information Technology")

        assert "Information Technology" in task.description

    def test_output_model_class_attribute(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent
        from src.models.signals import FundamentalScore

        agent = FinancialAnalysisAgent()
        assert agent.output_model is FundamentalScore

    def test_extra_tools_appended(self):
        from src.agents.financial_analyst import FinancialAnalysisAgent

        extra = MagicMock()
        agent = FinancialAnalysisAgent(extra_tools=[extra])
        assert len(agent.tools) == 3
        assert agent.tools[-1] is extra
