"""
src/agents/news_sentiment.py
NewsSentimentAgent — financial news analysis and sentiment scoring.

Wires together:
  * NSEAnnouncementTool — official corporate filings from NSE
  * GNewsAPITool        — recent news articles (Google News)
  * TavilySearchTool    — deep AI web search (earnings, analyst views)

The agent instructs Gemini to:
  1. Read all source material (announcements + articles).
  2. Identify material events (earnings beat/miss, management change,
     regulatory action, dividend/buyback, credit-rating action, etc.).
  3. Score overall sentiment from -100 (very negative) to +100 (very positive).
  4. Classify news_sentiment and social_sentiment separately.
  5. Return a structured :class:`~src.models.signals.SentimentResult`.

Use :meth:`build_task` to create a Task with
``output_pydantic=SentimentResult`` already set.
"""

from __future__ import annotations

from typing import Any

from crewai import Agent, Task

from src.agents.base_agent import BaseAgent
from src.models.signals import SentimentResult
from src.tools.news.gnews_tool import GNewsAPITool
from src.tools.news.nse_announcements import NSEAnnouncementTool
from src.tools.news.tavily_tool import TavilySearchTool

_ROLE = "Senior Financial News Analyst for Indian Equity Markets"

_GOAL = (
    "Analyse all available news, corporate announcements, and web search results "
    "for an NSE-listed stock to produce an accurate, data-driven sentiment score "
    "in the range -100 (extremely negative) to +100 (extremely positive). "
    "Identify every material event that could affect share price and surface the "
    "dominant themes investors should monitor."
)

_BACKSTORY = (
    "You are a financial journalist turned buy-side analyst with 10 years of "
    "experience covering Indian equities for a Mumbai-based fund. You have "
    "developed a systematic framework for distinguishing material events "
    "(earnings surprises, management changes, regulatory actions, credit-rating "
    "moves, F&O ban-list additions) from noise (routine filings, minor corporate "
    "actions). You are fluent in reading NSE LODR disclosures and BSE exchange "
    "filings. You never let recency bias inflate sentiment — a single negative "
    "earnings miss outweighs ten neutral articles. "
    "You always separate news-based sentiment (driven by official filings and "
    "financial media) from social/retail sentiment (driven by forums, social "
    "media chatter, and retail analyst commentary)."
)

# Material events the agent must watch for
_MATERIAL_EVENTS = (
    "earnings beat or miss vs consensus, management or promoter change, "
    "regulatory investigation or SEBI action, credit-rating upgrade/downgrade, "
    "large block deal or promoter pledge, dividend announcement, buyback, "
    "merger/acquisition/demerger, F&O ban-list addition, order win or "
    "contract cancellation, quarterly guidance revision"
)


class NewsSentimentAgent(BaseAgent):
    """CrewAI agent that analyses financial news and produces a SentimentResult.

    Usage::

        agent = NewsSentimentAgent()
        task  = agent.build_task(symbol="INFY", company_name="Infosys")
        crew  = Crew(agents=[agent.build()], tasks=[task])
        result = crew.kickoff()
    """

    #: The Pydantic model this agent is designed to produce.
    output_model: type[SentimentResult] = SentimentResult

    def __init__(
        self,
        extra_tools: list[Any] | None = None,
        llm_provider: str | None = None,
    ) -> None:
        tools = [
            NSEAnnouncementTool(),
            GNewsAPITool(),
            TavilySearchTool(),
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
    ) -> Task:
        """Create a CrewAI Task configured to produce a :class:`SentimentResult`.

        Args:
            symbol:       NSE ticker without suffix (e.g. ``"INFY"``).
            company_name: Full or common company name (e.g. ``"Infosys"``).
                          Used to build richer GNews and Tavily queries.
                          Falls back to symbol if omitted.

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=SentimentResult``.
        """
        symbol_upper = symbol.strip().upper()
        name = company_name.strip() or symbol_upper

        description = (
            f"Perform a comprehensive financial news and sentiment analysis for "
            f"**{symbol_upper}** ({name}).\n\n"
            "Steps:\n"
            f"1. Call `nse_announcements` (symbol={symbol_upper}) — retrieve the "
            "   latest NSE corporate filings. Note any earnings, management "
            "   changes, corporate actions, or regulatory disclosures.\n"
            f"2. Call `gnews_search` (symbol={symbol_upper}, company_name={name!r}) "
            "   — fetch recent news articles. Identify the dominant narrative "
            "   (positive / negative / mixed) in financial media.\n"
            f"3. Call `tavily_search` (symbol={symbol_upper}, company_name={name!r}) "
            "   — deep-search for earnings commentary, analyst upgrades/downgrades, "
            "   and any web content not covered by the news feed.\n"
            "4. Synthesise all collected material:\n"
            f"   a. Identify all material events from: {_MATERIAL_EVENTS}.\n"
            "   b. Score **news_sentiment** (POSITIVE / NEGATIVE / NEUTRAL) based "
            "      on NSE announcements and financial-media articles.\n"
            "   c. Score **social_sentiment** (POSITIVE / NEGATIVE / NEUTRAL) "
            "      based on retail/forum commentary found via Tavily.\n"
            "   d. Compute an overall **score** in [-100, +100]: weight "
            "      news_sentiment 70% (official sources are more reliable) and "
            "      social_sentiment 30%. A material negative event (earnings miss, "
            "      regulatory action) should push the score below -30 regardless "
            "      of social chatter.\n"
            "   e. List the top 3–5 **key_themes** as short phrases "
            "      (e.g. 'Q3 earnings beat', 'CFO resignation', 'buyback announced').\n"
            "   f. Set **headline_count** to the total number of distinct news "
            "      items collected across all three sources.\n"
        )

        expected_output = (
            "A valid SentimentResult JSON with fields: symbol, exchange, score, "
            "news_sentiment, social_sentiment, headline_count, key_themes, "
            "generated_at."
        )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=SentimentResult,
        )
