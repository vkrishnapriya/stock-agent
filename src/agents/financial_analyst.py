"""
src/agents/financial_analyst.py
FinancialAnalysisAgent — fundamental analysis for NSE-listed equities.

Wires together:
  * ScreenerFinancialsTool — annual P&L, balance sheet, cash flow, ratios,
                             quarterly results, shareholding (Screener.in)
  * NSEResultsTool         — last 8 consolidated quarterly results (NSE API)

The agent instructs Gemini to:
  1. Collect all financial data from both tools.
  2. Compute or verify: P/E, PEG, ROE, Debt/Equity, Revenue CAGR 3Y.
  3. Identify business headwinds and tailwinds from the data narrative.
  4. Score fundamentals from 0 (poor) to 100 (excellent).
  5. Return a structured :class:`~src.models.signals.FundamentalScore`.

Use :meth:`build_task` to create a Task with
``output_pydantic=FundamentalScore`` already set.
"""

from __future__ import annotations

from typing import Any

from crewai import Agent, Task

from src.agents.base_agent import BaseAgent
from src.models.signals import FundamentalScore
from src.tools.market.nse_fetcher import NSEResultsTool
from src.tools.market.screener_tools import ScreenerFinancialsTool

_ROLE = "Senior Fundamental Analyst for Indian Equity Markets"

_GOAL = (
    "Produce a rigorous, data-driven fundamental analysis of an NSE-listed stock "
    "by computing key financial ratios (P/E, PEG, ROE, Debt/Equity, Revenue CAGR), "
    "assessing earnings quality through quarterly trend analysis, and identifying "
    "the key business headwinds and tailwinds. Synthesise into a FundamentalScore "
    "(0 = poor fundamentals, 100 = excellent) with supporting metrics."
)

_BACKSTORY = (
    "You are a CFA charterholder with 15 years of buy-side equity research experience "
    "at a leading Mumbai-based asset management company. You have covered Indian IT, "
    "FMCG, banking, and industrials sectors in depth. Your framework for scoring "
    "fundamentals is grounded in: (1) earnings quality and consistency, "
    "(2) capital efficiency (ROE, ROCE, ROIC), (3) balance-sheet strength "
    "(D/E ratio, interest-coverage), (4) growth trajectory (revenue and earnings "
    "CAGR), and (5) valuation relative to growth (PEG). "
    "You always flag when accounting policies (Ind-AS vs IGAAP) or one-off items "
    "distort reported numbers, and you clearly distinguish between sustainable "
    "structural tailwinds and cyclical / regulatory headwinds."
)

# Scoring rubric injected into every task description
_SCORING_RUBRIC = (
    "Scoring rubric (0–100):\n"
    "  ROE > 20% & ROCE > 20%        → +20 pts\n"
    "  Revenue CAGR 3Y > 15%         → +20 pts\n"
    "  D/E ratio < 0.5               → +15 pts\n"
    "  Consistent EPS growth (≥ 5/8 quarters positive YoY) → +15 pts\n"
    "  P/E < sector median (from NSE pdSectorPe) → +15 pts\n"
    "  PEG < 1.5                     → +10 pts\n"
    "  Promoter holding > 40%        → +5 pts\n"
    "Deduct points proportionally for each metric that is adverse.\n"
    "If critical data is unavailable, estimate conservatively and note the gap."
)


class FinancialAnalysisAgent(BaseAgent):
    """CrewAI agent that produces a FundamentalScore for a single NSE stock.

    Usage::

        agent = FinancialAnalysisAgent()
        task  = agent.build_task(symbol="INFY", company_name="Infosys")
        crew  = Crew(agents=[agent.build()], tasks=[task])
        result = crew.kickoff()
    """

    #: The Pydantic model this agent is designed to produce.
    output_model: type[FundamentalScore] = FundamentalScore

    def __init__(
        self,
        extra_tools: list[Any] | None = None,
        llm_provider: str | None = None,
    ) -> None:
        tools: list[Any] = [
            ScreenerFinancialsTool(),
            NSEResultsTool(),
            *(extra_tools or []),
        ]
        super().__init__(tools=tools, llm_provider=llm_provider)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def build(self) -> Agent:
        """Return the configured CrewAI Agent."""
        return Agent(
            role=_ROLE,
            goal=_GOAL,
            backstory=_BACKSTORY,
            llm=self._get_llm(),
            tools=self.tools,
            **self._agent_defaults(),
        )

    # ------------------------------------------------------------------
    # Task factory
    # ------------------------------------------------------------------

    def build_task(
        self,
        symbol: str,
        company_name: str = "",
        sector: str = "",
    ) -> Task:
        """Create a CrewAI Task configured to produce a :class:`FundamentalScore`.

        Args:
            symbol:       NSE ticker without suffix (e.g. ``"INFY"``).
            company_name: Full company name for context (e.g. ``"Infosys Ltd"``).
            sector:       Sector name for relative valuation context.

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=FundamentalScore``.
        """
        symbol_upper = symbol.strip().upper()
        name = company_name.strip() or symbol_upper
        sector_line = f"Sector: {sector}\n" if sector else ""

        description = (
            f"Perform a complete fundamental analysis for **{symbol_upper}** ({name}).\n"
            f"{sector_line}\n"
            "Steps:\n"
            f"1. Call `screener_financials` (symbol={symbol_upper}) — fetch annual "
            "   P&L, balance sheet, cash flows, key ratios (P/E, ROE, ROCE, "
            "   Book Value), the last 8 quarterly results, and latest shareholding.\n"
            f"2. Call `nse_quarterly_results` (symbol={symbol_upper}) — fetch the "
            "   last 8 consolidated quarterly results from NSE API to cross-check "
            "   EPS, revenue, and PAT figures.\n"
            "3. Compute the following metrics:\n"
            "   a. **P/E ratio** — use 'Stock P/E' from Screener ratios.\n"
            "   b. **PB ratio** — use 'Book Value' and current price to compute P/B.\n"
            "   c. **ROE** — use 'ROE' from Screener ratios (annualised %).\n"
            "   d. **Debt/Equity** — from latest balance sheet: "
            "      total_borrowings / (equity_capital + reserves).\n"
            "   e. **Revenue CAGR 3Y** — from annual P&L: "
            "      (latest_sales / sales_3y_ago) ^ (1/3) − 1, as a %.\n"
            "   f. **Earnings growth % (TTM)** — latest 4Q EPS vs prior 4Q EPS.\n"
            "   g. **PEG** — P/E divided by earnings growth %.\n"
            "   h. **Promoter holding %** — latest from shareholding section.\n"
            "4. Identify **headwinds** (risks / challenges) and **tailwinds** "
            "   (opportunities / competitive advantages) visible in the data:\n"
            "   - Look at: revenue growth trend, margin trends (OPM%), D/E direction, "
            "     FII/DII holding changes, consistent or volatile EPS.\n"
            "5. Score fundamentals (0–100) using the rubric below and synthesise into "
            "   a FundamentalScore.\n\n"
            f"{_SCORING_RUBRIC}\n"
        )

        expected_output = (
            "A valid FundamentalScore JSON with fields: symbol, exchange, score, "
            "pe_ratio, pb_ratio, roe_pct, debt_to_equity, revenue_growth_pct, "
            "earnings_growth_pct, promoter_holding_pct, generated_at."
        )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=FundamentalScore,
        )
