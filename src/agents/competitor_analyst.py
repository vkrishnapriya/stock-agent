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
from src.models.signals import CompetitiveAnalysis
from src.tools.market.peer_performance import PeerPricePerformanceTool
from src.tools.market.screener_tools import CompetitorMapTool
from src.tools.news.gnews_tool import GNewsAPITool

_ROLE = "Senior Sector & Competitive Intelligence Analyst for Indian Equity Markets"

_GOAL = (
    "Identify a stock's key sector competitors, compare their recent price "
    "performance and news flow, and synthesise a CompetitiveAnalysis that scores "
    "the company's economic moat (0–100) and relative sector strength (−100 to +100), "
    "classifying its market position as LEADER, CHALLENGER, FOLLOWER, or NICHE."
)

_BACKSTORY = (
    "You are a sector specialist with 12 years of experience at a leading Indian "
    "equity research desk, covering technology, banking, FMCG, and industrials. "
    "Your competitive-analysis framework rests on four pillars: "
    "(1) relative price performance vs. peers over 1M and 3M windows, "
    "(2) news-flow divergence — positive catalysts for this company vs. headwinds "
    "for peers, (3) market positioning — market-cap rank and P/E premium or discount "
    "vs. peers, and (4) structural moat indicators — brand, distribution, IP, and "
    "regulatory moats inferred from size, margin, and news quality. "
    "You classify market position as: LEADER (largest market cap or dominant on "
    "most metrics), CHALLENGER (strong 2nd–3rd position with narrowing gap), "
    "FOLLOWER (below-average on most metrics), or NICHE (dominant in a narrow "
    "sub-segment with few direct peers). "
    "You always validate peer selection by checking whether the Screener.in peers "
    "table matches the company's actual business lines, and you flag it when fewer "
    "than 3 valid peers are found."
)

_MOAT_RUBRIC = (
    "Moat scoring rubric (0–100):\n"
    "  1M price rank 1st among peers                    → +20 pts\n"
    "  3M price rank 1st among peers                    → +20 pts\n"
    "  Market cap largest among identified peers         → +15 pts\n"
    "  P/E at premium to peer median (pricing power)    → +15 pts\n"
    "  Positive news flow vs. neutral/negative for peers → +20 pts\n"
    "  Outperforms Nifty 50 over 3M (market leader)     → +10 pts\n"
    "  Deduct points proportionally for adverse metrics.\n"
    "  If fewer than 2 peers found, estimate conservatively and note the gap."
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
            f"Perform a comprehensive competitive analysis for **{symbol_upper}** ({name}).\n"
            f"{sector_line}\n"
            "Steps:\n"
            f"1. Call `competitor_map` (symbol={symbol_upper}) — fetch 3–5 sector peers "
            "   from Screener.in. Note each peer's market cap and P/E relative to the "
            "   target to establish size and valuation positioning.\n"
            f"2. Call `gnews_search` (symbol={symbol_upper}, company_name={name!r}) for "
            "   the target. Then call `gnews_search` once each for the top 2 peers by "
            "   market cap — batch carefully to conserve daily rate-limit budget. "
            "   Identify whether the target's news flow is more positive, more negative, "
            "   or neutral compared with its peers.\n"
            f"3. Call `peer_price_performance` with symbols_csv containing {symbol_upper} "
            "   plus all identified peer symbols (comma-separated, no .NS suffix). "
            "   Record each symbol's 1M and 3M return and rank the target.\n"
            "4. Compute the following:\n"
            f"   a. **relative_strength** [−100, +100]: how much better or worse "
            f"      {symbol_upper} performed vs. the unweighted average peer return "
            "      over 3M. Map +30 ppt outperformance to +100, −30 ppt to −100, "
            "      scaling linearly.\n"
            f"   b. **market_position**: LEADER if {symbol_upper} has the largest "
            "      market cap among peers; CHALLENGER if top-3 but not first; "
            "      FOLLOWER if below peer median on market cap and price performance; "
            "      NICHE if fewer than 3 direct peers exist.\n"
            "   c. **moat_score** [0–100]: apply the rubric below.\n"
            "   d. **sector**: use the provided sector if given, otherwise infer from "
            "      the peer names and Screener data.\n"
            "   e. **peers**: list the NSE symbols of all identified peers.\n"
            "5. Synthesise all data into a CompetitiveAnalysis output.\n\n"
            f"{_MOAT_RUBRIC}\n"
        )

        expected_output = (
            "A valid CompetitiveAnalysis JSON with fields: symbol, exchange, sector, "
            "market_position, moat_score, relative_strength, peers, generated_at."
        )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=CompetitiveAnalysis,
        )
