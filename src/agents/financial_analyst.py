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

_ROLE = "Fundamental Analyst for Indian Equity Markets"

_GOAL = (
    "Compute P/E, PEG, ROE, D/E, Revenue CAGR for an NSE stock and produce "
    "a FundamentalScore (0=poor, 100=excellent) with key headwinds/tailwinds."
)

_BACKSTORY = (
    "CFA with 12 years buy-side equity research in Indian IT, FMCG, banking. "
    "Score fundamentals on: earnings quality, ROE/ROCE, D/E, revenue CAGR, PEG, promoter holding."
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
            f"Fundamental analysis for {symbol_upper} ({name}). {sector_line}"
            f"1. `screener_financials` (symbol={symbol_upper}) — P&L, balance sheet, ratios, shareholding.\n"
            f"2. `nse_quarterly_results` (symbol={symbol_upper}) — last 8 quarters EPS/revenue/PAT.\n"
            "3. Compute: P/E, P/B, ROE, D/E, Revenue CAGR 3Y, earnings growth TTM, PEG, promoter holding.\n"
            "4. Score 0-100: ROE>20%+ROCE>20%(+20), RevCAGR>15%(+20), D/E<0.5(+15), "
            "consistent EPS(+15), P/E<sector(+15), PEG<1.5(+10), promoter>40%(+5). "
            "Deduct for adverse metrics.\n"
            "5. Return FundamentalScore with headwinds and tailwinds."
        )

        expected_output = (
            "FundamentalScore JSON: symbol, exchange, score, pe_ratio, pb_ratio, "
            "roe_pct, debt_to_equity, revenue_growth_pct, earnings_growth_pct, "
            "promoter_holding_pct, generated_at."
        )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=FundamentalScore,
        )
