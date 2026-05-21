"""
src/agents/competitor_analyst.py
CompetitorAnalysisAgent — sector peer comparison and competitive positioning.

Wires together:
  * CompetitorMapTool       — 3-5 sector peers from Screener.in Peers table
  * GNewsAPITool            — recent news for target + top 2 peers (batched)
  * PeerPricePerformanceTool — 1M/3M price returns vs peers (yfinance batch)

The agent instructs Gemini to:
  1. Identify sector peers and compare market cap / P/E positioning.
  2. Contrast news flow: positive catalysts for target vs. peers.
  3. Rank price performance over 1M and 3M.
  4. Score economic moat (0–100) and relative sector strength (−100 to +100).
  5. Return a structured :class:`~src.models.signals.CompetitiveAnalysis`.

Use :meth:`build_task` to create a Task with
``output_pydantic=CompetitiveAnalysis`` already set.
"""

from __future__ import annotations

from typing import Any

from crewai import Agent, Task

from src.agents.base_agent import BaseAgent
from src.models.signals import CompetitiveAnalysis, CompetitiveAnalysisBatch
from src.tools.market.peer_performance import PeerPricePerformanceTool
from src.tools.market.screener_tools import CompetitorMapTool
from src.tools.news.gnews_tool import GNewsAPITool

_ROLE = "Competitive Intelligence Analyst for Indian Equity Markets"

_GOAL = (
    "Compare a stock vs sector peers on price performance, news flow, and market cap. "
    "Produce CompetitiveAnalysis: moat_score (0-100), relative_strength (-100 to +100), "
    "market_position (LEADER/CHALLENGER/FOLLOWER/NICHE)."
)

_BACKSTORY = (
    "Sector analyst with 10 years covering Indian tech, banking, FMCG. "
    "Framework: (1) price performance vs peers 1M/3M, (2) news-flow divergence, "
    "(3) market-cap rank, (4) structural moat from size/margin."
)


class CompetitorAnalysisAgent(BaseAgent):
    """CrewAI agent that produces a CompetitiveAnalysis for a single NSE stock.

    Usage::

        agent = CompetitorAnalysisAgent()
        task  = agent.build_task(symbol="INFY", company_name="Infosys", sector="IT")
        crew  = Crew(agents=[agent.build()], tasks=[task])
        result = crew.kickoff()
    """

    #: The Pydantic model this agent is designed to produce.
    output_model: type[CompetitiveAnalysis] = CompetitiveAnalysis

    def __init__(
        self,
        extra_tools: list[Any] | None = None,
        llm_provider: str | None = None,
    ) -> None:
        tools: list[Any] = [
            CompetitorMapTool(),
            GNewsAPITool(),
            PeerPricePerformanceTool(),
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
        """Create a CrewAI Task configured to produce a :class:`CompetitiveAnalysis`.

        Args:
            symbol:       NSE ticker without suffix (e.g. ``"INFY"``).
            company_name: Full company name for context (e.g. ``"Infosys Ltd"``).
            sector:       Sector hint for the agent (e.g. ``"IT Services"``).

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=CompetitiveAnalysis``.
        """
        symbol_upper = symbol.strip().upper()
        name = company_name.strip() or symbol_upper
        sector_line = f"Sector: {sector}\n" if sector else ""

        description = (
            f"Competitive analysis for {symbol_upper} ({name}). {sector_line}"
            f"1. `competitor_map` (symbol={symbol_upper}) — get 3-5 peers with market cap and P/E.\n"
            f"2. `gnews_search` for {symbol_upper} and top 2 peers — compare news sentiment.\n"
            f"3. `peer_price_performance` (symbols_csv={symbol_upper}+peers) — rank 1M/3M returns.\n"
            "4. Compute: relative_strength [-100,+100] vs peer avg 3M return (±30ppt = ±100); "
            "market_position (LEADER=largest mcap, CHALLENGER=top3, FOLLOWER=below median, NICHE=<3 peers); "
            "moat_score [0-100] (1M rank1=+20, 3M rank1=+20, largest mcap=+15, P/E premium=+15, "
            "positive news=+20, beats Nifty50=+10).\n"
            "5. Return CompetitiveAnalysis."
        )

        expected_output = (
            "CompetitiveAnalysis JSON: symbol, exchange, sector, market_position, "
            "moat_score, relative_strength, peers, generated_at."
        )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=CompetitiveAnalysis,
        )

    def build_batch_task(self, table: str, symbols: list[str]) -> Task:
        """Create a single Task that analyses competitive positioning for all *symbols*.

        Args:
            table: Markdown table with columns: symbol, sector, peers, peer_count,
                   1m_ret_pct, 3m_ret_pct, rank_3m, mcap_rank, mom_1m_pct.
            symbols: Ordered list of NSE symbols present in the table.

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=CompetitiveAnalysisBatch``.
        """
        from crewai import Agent as _Agent

        batch_agent = _Agent(
            role=_ROLE,
            goal=_GOAL,
            backstory=_BACKSTORY,
            llm=self._get_llm(),
            tools=[],
            **self._agent_defaults(),
        )
        n = len(symbols)
        description = (
            f"Assess competitive positioning for {n} NSE stocks using the table below.\n\n"
            f"{table}\n\n"
            "For each stock compute:\n"
            "  relative_strength [-100,+100]: based on rank_3m among peers "
            "(rank 1 = +100, last rank = -100, linear scale).\n"
            "  market_position: LEADER (mcap_rank=1), CHALLENGER (rank 2-3), "
            "FOLLOWER (rank 4+), NICHE (peer_count < 2).\n"
            "  moat_score [0-100]: rank_3m=1 → +20, 3m_ret > 0 → +15, "
            "mcap_rank=1 → +15, mom_1m > 5% → +20, peer_count > 3 → +10, "
            "positive 1m and 3m returns → +20. Deduct for negative returns.\n\n"
            "IMPORTANT — use these EXACT field names in your output JSON (not alternatives):\n"
            "  'moat_score'        ← the computed 0-100 moat score\n"
            "  'relative_strength' ← the computed -100 to +100 value\n"
            "  'market_position'   ← exactly LEADER, CHALLENGER, FOLLOWER, or NICHE\n"
            "  'sector'            ← use the sector column value\n"
            "  'peers'             ← a JSON list of NSE ticker strings "
            "(split the comma-separated peers column, e.g. [\"TCS\", \"INFY\"])\n\n"
            f"Return CompetitiveAnalysisBatch with exactly {n} CompetitiveAnalysis objects "
            "in the same order as the table."
        )
        return Task(
            description=description,
            expected_output=(
                f"CompetitiveAnalysisBatch JSON with an 'analyses' list of {n} CompetitiveAnalysis objects."
            ),
            agent=batch_agent,
            output_pydantic=CompetitiveAnalysisBatch,
        )
