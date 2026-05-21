"""
src/agents/news_sentiment.py
NewsSentimentAgent — financial news + whale-activity sentiment scoring.

Two-agent sequential crew:

  Agent 1 — WhaleResearcherAgent
    Tools : WhaleTrackerTool, TavilySearchTool
    Role  : Audits NSE bulk/block deals and web sources for activity by known
            top Indian super investors (Dolly Khanna, Ashish Kacholia, Vijay
            Kedia, Rekha Jhunjhunwala, Sunil Singhania, Radhakishan Damani …).
    Output: Plain-text whale activity report passed as context to Agent 2.

  Agent 2 — NewsSentimentAgent  (this class)
    Tools : NSEAnnouncementTool, GNewsAPITool, TavilySearchTool
    Role  : Reads NSE filings, news, and the whale-activity context to produce
            a final :class:`~src.models.signals.SentimentResult` that includes
            ``whale_signal`` and ``whale_activity`` fields.

Typical usage via :meth:`build_crew`::

    agent  = NewsSentimentAgent()
    crew   = agent.build_crew(symbol="INFY")
    result = crew.kickoff()
"""

from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, Process, Task

from src.agents.base_agent import BaseAgent
from src.models.signals import SentimentResult, SentimentResultBatch
from src.tools.news.gnews_tool import GNewsAPITool
from src.tools.news.nse_announcements import NSEAnnouncementTool
from src.tools.news.tavily_tool import TavilySearchTool
from src.tools.news.whale_tracker import WhaleTrackerTool

# ---------------------------------------------------------------------------
# Agent 2 — News & Sentiment Analyst
# ---------------------------------------------------------------------------

_ROLE = "Senior Financial News Analyst for Indian Equity Markets"

_GOAL = (
    "Analyse all available news, corporate announcements, whale-investor activity, "
    "and web search results for an NSE-listed stock to produce an accurate, "
    "data-driven sentiment score in the range -100 (extremely negative) to "
    "+100 (extremely positive). Identify every material event that could affect "
    "share price and surface the dominant themes investors should monitor."
)

_BACKSTORY = (
    "You are a financial journalist turned buy-side analyst with 10 years of "
    "experience covering Indian equities for a Mumbai-based fund. You distinguish "
    "material events (earnings surprises, management changes, regulatory actions, "
    "credit-rating moves, F&O ban-list additions) from noise. You are fluent in "
    "NSE LODR disclosures and BSE filings. You weight smart-money (super investor) "
    "activity heavily — when recent activity from a known whale like Dolly Khanna or Ashish Kacholia "
    "accumulates, it is a significant bullish signal that can shift the score by "
    "up to +20 points; exits shift it by −20. You never let recency bias inflate "
    "sentiment — a single negative earnings miss outweighs ten neutral articles."
)

# Material events the agent must watch for
_MATERIAL_EVENTS = (
    "earnings beat or miss vs consensus, management or promoter change, "
    "regulatory investigation or SEBI action, credit-rating upgrade/downgrade, "
    "large block deal or promoter pledge, dividend announcement, buyback, "
    "merger/acquisition/demerger, F&O ban-list addition, order win or "
    "contract cancellation, quarterly guidance revision"
)

# ---------------------------------------------------------------------------
# Agent 1 — Whale / Super-Investor Researcher
# ---------------------------------------------------------------------------

_WHALE_ROLE = "Financial News Auditor & Super-Investor Tracker"

_WHALE_GOAL = (
    "Identify official portfolio changes and public statements by top Indian "
    "super investors for a given NSE stock. Separate confirmed regulatory "
    "filings (NSE bulk/block deals) from social-media rumour and speculation."
)

_WHALE_BACKSTORY = (
    "You are an expert in Indian equity markets with deep knowledge of the "
    "disclosure rules under SEBI's LODR regulations. You track bulk/block "
    "deal filings daily and cross-reference them with public statements from "
    "India's most respected super investors:\n"
    "  • Radhakishan Damani — DMart promoter, deep-value buyer\n"
    "  • Dolly Khanna       — known for early entries in small/mid-cap turnarounds\n"
    "  • Vijay Kedia        — runs Kedia Securities; often in capital-goods stocks\n"
    "  • Ashish Kacholia    — Lucky Securities; quality small-cap compounders\n"
    "  • Porinju Veliyath   — Equity Intelligence; high-conviction contrarian bets\n"
    "  • Rekha Jhunjhunwala — Rare Enterprises; carries forward Rakesh's legacy\n"
    "  • Sunil Singhania    — Abakkus Asset Manager; diversified growth focus\n"
    "  • Rajeev Thakkar     — PPFAS Mutual Fund; long-only value discipline\n"
    "  • Saurabh Mukherjea  — Marcellus Investment; consistent compounders\n"
    "You clearly distinguish BUY activity (accumulation signal) from SELL "
    "activity (distribution signal) and flag when multiple whales act in the "
    "same direction simultaneously — that is a high-conviction signal."
)


class NewsSentimentAgent(BaseAgent):
    """Two-agent crew: WhaleResearcher → NewsSentimentAnalyst.

    The preferred entry point is :meth:`build_crew`, which returns a ready-to-run
    sequential :class:`crewai.Crew` that passes whale-activity context from the
    researcher to the analyst.

    Single-agent fallback (:meth:`build_task`) is still available for cases where
    you only need the analyst task (e.g. testing).

    Usage::

        agent  = NewsSentimentAgent()
        crew   = agent.build_crew(symbol="INFY", company_name="Infosys")
        result = crew.kickoff()
        # SentimentResult is in result.tasks_output[1].pydantic
    """

    #: The Pydantic model this agent is designed to produce.
    output_model: type[SentimentResult] = SentimentResult

    def __init__(
        self,
        extra_tools: list[Any] | None = None,
        llm_provider: str | None = None,
    ) -> None:
        # Analyst tools
        tools = [
            NSEAnnouncementTool(),
            GNewsAPITool(),
            TavilySearchTool(),
            *(extra_tools or []),
        ]
        super().__init__(tools=tools, llm_provider=llm_provider)

    # ------------------------------------------------------------------
    # BaseAgent interface — builds the Analyst agent
    # ------------------------------------------------------------------

    def build(self) -> Agent:
        """Return the configured Analyst CrewAI Agent."""
        return Agent(
            role=_ROLE,
            goal=_GOAL,
            backstory=_BACKSTORY,
            llm=self._get_llm(),
            tools=self.tools,
            **self._agent_defaults(),
        )

    # ------------------------------------------------------------------
    # Agent 1 factory — Whale Researcher
    # ------------------------------------------------------------------

    def _build_whale_researcher(self) -> Agent:
        """Return the Whale Researcher agent (Agent 1 in the crew)."""
        return Agent(
            role=_WHALE_ROLE,
            goal=_WHALE_GOAL,
            backstory=_WHALE_BACKSTORY,
            llm=self._get_llm(),
            tools=[WhaleTrackerTool(), TavilySearchTool()],
            **self._agent_defaults(),
        )

    def build_whale_research_task(self, symbol: str, company_name: str = "") -> Task:
        """Create Task 1: audit super-investor activity for *symbol*.

        Args:
            symbol:       NSE ticker without suffix (e.g. ``"INFY"``).
            company_name: Human-readable company name for Tavily queries.

        Returns:
            A :class:`crewai.Task` whose output is plain text — a structured
            whale-activity report consumed as context by the analyst task.
        """
        symbol_upper = symbol.strip().upper()
        name = company_name.strip() or symbol_upper

        description = (
            f"Audit super-investor activity for **{symbol_upper}** ({name}).\n\n"
            "Steps:\n"
            f"1. Call `whale_tracker` (symbol={symbol_upper}) — fetch all NSE bulk "
            "   and block deals. Identify which known super investors bought or sold "
            "   and the quantities/prices involved.\n"
            f"2. Call `tavily_search` with query: "
            f'   "{name} {symbol_upper} Dolly Khanna OR Ashish Kacholia OR Vijay Kedia '
            f'   OR Rekha Jhunjhunwala OR Sunil Singhania OR Radhakishan Damani portfolio 2026"\n'
            "   — find any recent public statements, interviews, or disclosures where "
            "   these investors mention the stock.\n"
            "3. Produce a structured whale-activity report with:\n"
            "   • List every confirmed BUY (name, quantity, price, date)\n"
            "   • List every confirmed SELL (name, quantity, price, date)\n"
            "   • Highlight if multiple whales acted in the same direction\n"
            "   • Note any public quotes or views expressed about the stock\n"
            "   • Set overall whale_signal: BULLISH (more buys), BEARISH (more sells), "
            "     or NEUTRAL (no activity or balanced)\n"
        )

        return Task(
            description=description,
            expected_output=(
                "Plain-text whale activity report: confirmed buys, confirmed sells, "
                "public quotes, and a one-line whale_signal summary "
                "(e.g. 'BULLISH — Dolly Khanna and Ashish Kacholia both accumulated')."
            ),
            agent=self._build_whale_researcher(),
        )

    # ------------------------------------------------------------------
    # Task 2 factory — News Sentiment Analyst
    # ------------------------------------------------------------------

    def build_task(
        self,
        symbol: str,
        company_name: str = "",
        context: list[Task] | None = None,
    ) -> Task:
        """Create Task 2: news + whale-context sentiment analysis.

        Args:
            symbol:       NSE ticker without suffix (e.g. ``"INFY"``).
            company_name: Full or common company name.
            context:      List of upstream Tasks whose output is injected as
                          context (pass the whale research task here).

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=SentimentResult``.
        """
        symbol_upper = symbol.strip().upper()
        name = company_name.strip() or symbol_upper

        description = (
            f"Perform a comprehensive financial news and sentiment analysis for "
            f"**{symbol_upper}** ({name}), incorporating the whale-activity "
            f"context provided by the researcher.\n\n"
            "Steps:\n"
            f"1. Call `nse_announcements` (symbol={symbol_upper}) — retrieve the "
            "   latest NSE corporate filings. Note earnings, management changes, "
            "   corporate actions, or regulatory disclosures.\n"
            f"2. Call `gnews_search` (symbol={symbol_upper}, company_name={name!r}) "
            "   — fetch recent news articles. Identify the dominant narrative.\n"
            f"3. Call `tavily_search` (symbol={symbol_upper}, company_name={name!r}) "
            "   — deep-search for earnings commentary and analyst upgrades/downgrades.\n"
            "4. Synthesise all collected material AND the whale-activity context:\n"
            f"   a. Identify all material events: {_MATERIAL_EVENTS}.\n"
            "   b. Score **news_sentiment** (POSITIVE/NEGATIVE/NEUTRAL) from filings "
            "      and financial-media articles.\n"
            "   c. Score **social_sentiment** (POSITIVE/NEGATIVE/NEUTRAL) from "
            "      retail/forum commentary via Tavily.\n"
            "   d. Set **whale_signal** (BULLISH/BEARISH/NEUTRAL) directly from the "
            "      researcher's whale_signal finding. Copy the whale_activity lines "
            "      from the researcher's report into the whale_activity list.\n"
            "   e. Compute an overall **score** in [-100, +100]:\n"
            "      • Base: news_sentiment 60% + social_sentiment 20%\n"
            "      • Whale adjustment: BULLISH whale_signal → +20 pts; "
            "        BEARISH → -20 pts; NEUTRAL → 0 pts\n"
            "      • A material negative event (earnings miss, regulatory action) "
            "        pushes score below -30 regardless of whale activity.\n"
            "      • Multiple whales accumulating simultaneously adds +10 extra.\n"
            "   f. List the top 3–5 **key_themes** as short phrases.\n"
            "   g. Set **headline_count** to total distinct news items across all sources.\n"
        )

        return Task(
            description=description,
            expected_output=(
                "A valid SentimentResult JSON with fields: symbol, exchange, score, "
                "news_sentiment, social_sentiment, whale_signal, whale_activity, "
                "headline_count, key_themes, generated_at."
            ),
            agent=self.build(),
            output_pydantic=SentimentResult,
            context=context or [],
        )

    # ------------------------------------------------------------------
    # Crew factory — preferred entry point
    # ------------------------------------------------------------------

    def build_crew(self, symbol: str, company_name: str = "") -> Crew:
        """Build a single-agent, tool-free crew using pre-fetched data.

        All data sources (whale tracker, NSE announcements, GNews, Tavily) are
        fetched in parallel Python threads before the LLM call so the agent
        never needs a ReAct tool-use loop.  This avoids the
        ``assistant message prefill`` error on Claude Sonnet 4.6+.

        The SentimentResult is in ``crew.kickoff().tasks_output[0].pydantic``.

        Args:
            symbol:       NSE ticker without suffix.
            company_name: Human-readable name (used in search queries).

        Returns:
            A :class:`crewai.Crew` ready to call ``.kickoff()``.
        """
        from crewai import Agent as _Agent

        symbol_upper = symbol.strip().upper()
        name = company_name.strip() or symbol_upper

        context_block = self._gather_context(symbol_upper, name)

        batch_agent = _Agent(
            role=_ROLE,
            goal=_GOAL,
            backstory=_BACKSTORY,
            llm=self._get_llm(),
            tools=[],
            **self._agent_defaults(),
        )

        description = (
            f"Analyse the pre-fetched data below and produce a SentimentResult "
            f"for **{symbol_upper}** ({name}).\n\n"
            f"{context_block}\n\n"
            "Using only the data above:\n"
            "1. Score **news_sentiment** (POSITIVE/NEGATIVE/NEUTRAL) from NSE "
            "   announcements and news articles.\n"
            "2. Score **social_sentiment** (POSITIVE/NEGATIVE/NEUTRAL) from web "
            "   search results.\n"
            "3. Set **whale_signal** (BULLISH/BEARISH/NEUTRAL) from the whale "
            "   activity section.\n"
            "4. Copy whale activity summary lines into **whale_activity** list.\n"
            "5. Compute overall **score** in [-100, +100]:\n"
            "   • Base: news_sentiment (POSITIVE=+50, NEUTRAL=0, NEGATIVE=-50) × 60%\n"
            "     + social_sentiment (same scale) × 20%\n"
            "   • Whale: BULLISH → +20 pts, BEARISH → -20 pts, NEUTRAL → 0\n"
            "   • Material negative event (earnings miss, regulatory action) → "
            "     push score below -30 regardless of whale activity.\n"
            "   • Multiple whales acting in same direction simultaneously → +10 extra.\n"
            "6. List 3–5 **key_themes** as short phrases.\n"
            "7. Set **headline_count** to total distinct items across all sources.\n"
        )

        task = Task(
            description=description,
            expected_output=(
                "A valid SentimentResult JSON with fields: symbol, exchange, score, "
                "news_sentiment, social_sentiment, whale_signal, whale_activity, "
                "headline_count, key_themes, generated_at."
            ),
            agent=batch_agent,
            output_pydantic=SentimentResult,
        )

        return Crew(
            agents=[batch_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

    def build_batch_crew(self, entries: list[Any]) -> Crew:
        """Single LLM call that scores ALL symbols in *entries* at once.

        Pre-fetches whale, NSE, GNews, and Tavily data for every stock in
        parallel (one ThreadPoolExecutor across all stocks × 4 sources), then
        embeds the results as numbered per-stock sections and asks the LLM to
        return a :class:`~src.models.signals.SentimentResultBatch`.

        This replaces the previous per-stock ``build_crew`` loop, reducing
        LLM calls from N → 1.

        Args:
            entries: List of :class:`~src.models.scan.ScanEntry` (or any object
                     with ``.symbol`` and optionally ``.sector`` attributes).

        Returns:
            A :class:`crewai.Crew` ready to call ``.kickoff()``.
        """
        from crewai import Agent as _Agent

        symbols = [e.symbol.strip().upper() for e in entries]
        context_block = self._gather_all_contexts(symbols)
        n = len(symbols)

        batch_agent = _Agent(
            role=_ROLE,
            goal=_GOAL,
            backstory=_BACKSTORY,
            llm=self._get_llm(),
            tools=[],
            **self._agent_defaults(),
        )

        description = (
            f"Analyse the pre-fetched data below for {n} NSE stocks and produce "
            f"a SentimentResult for each.\n\n"
            f"{context_block}\n\n"
            "For EACH stock section above:\n"
            "1. Score **news_sentiment** (POSITIVE/NEGATIVE/NEUTRAL) from NSE "
            "   announcements and news articles.\n"
            "2. Score **social_sentiment** (POSITIVE/NEGATIVE/NEUTRAL) from web "
            "   search results.\n"
            "3. Set **whale_signal** (BULLISH/BEARISH/NEUTRAL) from the whale "
            "   activity section.\n"
            "4. Copy whale activity summary lines into **whale_activity** list.\n"
            "5. Compute overall **score** in [-100, +100]:\n"
            "   • Base: news_sentiment (POSITIVE=+50, NEUTRAL=0, NEGATIVE=-50) × 60%\n"
            "     + social_sentiment (same scale) × 20%\n"
            "   • Whale: BULLISH → +20 pts, BEARISH → -20 pts, NEUTRAL → 0\n"
            "   • Material negative event (earnings miss, regulatory action) → "
            "     push score below -30 regardless of whale activity.\n"
            "   • Multiple whales acting in same direction simultaneously → +10 extra.\n"
            "6. List 3–5 **key_themes** as short phrases.\n"
            "7. Set **headline_count** to total distinct items across all sources.\n\n"
            f"Return SentimentResultBatch with exactly {n} SentimentResult objects "
            f"in this order: {', '.join(symbols)}."
        )

        task = Task(
            description=description,
            expected_output=(
                f"SentimentResultBatch JSON with a 'results' list of {n} "
                "SentimentResult objects, one per symbol in the order listed."
            ),
            agent=batch_agent,
            output_pydantic=SentimentResultBatch,
        )

        return Crew(
            agents=[batch_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

    def _gather_all_contexts(self, symbols: list[str]) -> str:
        """Pre-fetch all data for all *symbols* in parallel and return formatted sections.

        Fires up to ``len(symbols) × 4`` concurrent threads — one per
        (symbol, source) pair — so total fetch time equals the slowest
        single request rather than the sum of all requests.
        """
        import json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch(symbol: str, source: str) -> tuple[str, str, str]:
            """Return (symbol, source, result_text)."""
            try:
                if source == "whale":
                    return symbol, source, WhaleTrackerTool()._run(symbol=symbol)
                if source == "ann":
                    return symbol, source, NSEAnnouncementTool()._run(symbol=symbol)
                if source == "news":
                    return symbol, source, GNewsAPITool()._run(symbol=symbol, company_name=symbol)
                if source == "tavily":
                    return symbol, source, TavilySearchTool()._run(symbol=symbol, company_name=symbol)
            except Exception as exc:
                fallback = json.dumps({"net_signal": "NEUTRAL", "summary": [str(exc)]}) \
                    if source == "whale" else f"Unavailable: {exc}"
                return symbol, source, fallback
            return symbol, source, "Unavailable"

        sources = ["whale", "ann", "news", "tavily"]
        data: dict[str, dict[str, str]] = {s: {} for s in symbols}

        with ThreadPoolExecutor(max_workers=min(len(symbols) * 4, 20)) as pool:
            futures = [
                pool.submit(_fetch, sym, src)
                for sym in symbols
                for src in sources
            ]
            for fut in as_completed(futures):
                sym, src, text = fut.result()
                data[sym][src] = text

        sections = []
        for i, sym in enumerate(symbols, 1):
            d = data[sym]
            sections.append(
                f"## Stock {i}: {sym}\n"
                f"### Whale / Super-Investor Activity\n{d.get('whale', 'Unavailable')}\n\n"
                f"### NSE Corporate Announcements\n{d.get('ann', 'Unavailable')}\n\n"
                f"### Recent News (GNews)\n{d.get('news', 'Unavailable')}\n\n"
                f"### Web Search (Tavily)\n{d.get('tavily', 'Unavailable')}"
            )

        return "\n\n---\n\n".join(sections)

    def _gather_context(self, symbol: str, name: str) -> str:
        """Pre-fetch all data sources in parallel and return a formatted block."""
        import json
        from concurrent.futures import ThreadPoolExecutor

        def _whale() -> str:
            try:
                return WhaleTrackerTool()._run(symbol=symbol)
            except Exception as exc:
                return json.dumps({
                    "net_signal": "NEUTRAL", "deal_count": 0,
                    "summary": [f"Data unavailable: {exc}"],
                })

        def _announcements() -> str:
            try:
                return NSEAnnouncementTool()._run(symbol=symbol)
            except Exception as exc:
                return f"Unavailable: {exc}"

        def _news() -> str:
            try:
                return GNewsAPITool()._run(symbol=symbol, company_name=name)
            except Exception as exc:
                return f"Unavailable: {exc}"

        def _tavily() -> str:
            try:
                return TavilySearchTool()._run(symbol=symbol, company_name=name)
            except Exception as exc:
                return f"Unavailable: {exc}"

        with ThreadPoolExecutor(max_workers=4) as pool:
            whale_fut = pool.submit(_whale)
            ann_fut   = pool.submit(_announcements)
            news_fut  = pool.submit(_news)
            tav_fut   = pool.submit(_tavily)

        return (
            "### Whale / Super-Investor Activity (NSE Bulk & Block Deals)\n"
            f"{whale_fut.result()}\n\n"
            "### NSE Corporate Announcements\n"
            f"{ann_fut.result()}\n\n"
            "### Recent News (GNews)\n"
            f"{news_fut.result()}\n\n"
            "### Web Search (Tavily)\n"
            f"{tav_fut.result()}\n"
        )
