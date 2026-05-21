"""
src/agents/technical_analyst.py
TechnicalAnalysisAgent — analyses price action and computes TechnicalSignal.

Wires together:
  * OHLCVFetchTool     — raw price/volume data from Yahoo Finance
  * IndicatorEngineTool — RSI, MACD, BB, SMA, OBV → composite score
  * SupportResistanceTool — pivot points, 52w range, MA levels

The agent's task output should be structured as a
:class:`~src.models.signals.TechnicalSignal`.  Use :meth:`build_task` to
create a Task with ``output_pydantic=TechnicalSignal`` already set.
"""

from __future__ import annotations

from typing import Any

from crewai import Agent, Task

from src.agents.base_agent import BaseAgent
from src.models.signals import TechnicalSignal, TechnicalSignalBatch
from src.tools.market.yfinance_tools import OHLCVFetchTool
from src.tools.technical.indicator_engine import IndicatorEngineTool
from src.tools.technical.support_resistance import SupportResistanceTool

_ROLE = "Technical Analyst for Indian Equity Markets"

_GOAL = (
    "Analyse NSE price action and indicators to produce a TechnicalSignal "
    "with score (-100 to +100), trend, support/resistance levels in INR."
)

_BACKSTORY = (
    "CMT with 10 years analysing NSE equities. You combine RSI, MACD, "
    "SMA 50/200, Bollinger Bands, and OBV to form views. Always define "
    "stop-loss and target levels."
)


class TechnicalAnalysisAgent(BaseAgent):
    """CrewAI agent that performs end-to-end technical analysis for a single NSE symbol.

    Usage::

        agent = TechnicalAnalysisAgent()
        task  = agent.build_task(symbol="INFY", period="1y")
        crew  = Crew(agents=[agent.build()], tasks=[task])
        result = crew.kickoff()
    """

    #: The Pydantic model this agent is designed to produce.
    output_model: type[TechnicalSignal] = TechnicalSignal

    def __init__(
        self,
        extra_tools: list[Any] | None = None,
        llm_provider: str | None = None,
    ) -> None:
        tools = [
            OHLCVFetchTool(),
            IndicatorEngineTool(),
            SupportResistanceTool(),
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
        period: str = "3mo",
        interval: str = "1d",
    ) -> Task:
        """Create a CrewAI Task configured to produce a :class:`TechnicalSignal`.

        Args:
            symbol: NSE ticker without exchange suffix (e.g. ``"INFY"``).
            period: yfinance look-back period (e.g. ``"1y"``, ``"6mo"``).
            interval: Bar interval (e.g. ``"1d"`` for daily, ``"15m"`` for intraday).

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=TechnicalSignal``.
        """
        yf_symbol = f"{symbol.upper()}.NS"
        description = (
            f"Technical analysis for {symbol.upper()} ({yf_symbol}), {period} {interval} data.\n"
            f"1. `ohlcv_fetch` (symbol={yf_symbol}, period={period}, interval={interval})\n"
            f"2. `indicator_engine` (symbol={symbol.upper()})\n"
            f"3. `support_resistance` (symbol={yf_symbol})\n"
            "4. Return TechnicalSignal: score [-100,+100], trend BULLISH/BEARISH/NEUTRAL, "
            "support/resistance in INR."
        )
        expected_output = (
            "TechnicalSignal JSON: symbol, exchange, score, trend, rsi, macd_signal, "
            "support_level_inr, resistance_level_inr, volume_signal, generated_at."
        )
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=TechnicalSignal,
        )

    def build_batch_task(self, table: str, symbols: list[str]) -> Task:
        """Create a single Task that analyses all *symbols* from a pre-fetched table.

        No tool calls are made — all indicator data is embedded in *table*.

        Args:
            table: Markdown table with columns: symbol, price, rsi, macd_hist,
                   score, trend, support, resistance, vol_signal, mom_1m_pct, vol_ratio.
            symbols: Ordered list of NSE symbols present in the table.

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=TechnicalSignalBatch``.
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
            f"Review pre-computed technical indicators for {n} NSE stocks in the table below "
            "and produce a final TechnicalSignal for each.\n\n"
            f"{table}\n\n"
            "For each row:\n"
            "1. Accept the pre-computed score (-100 to +100) unless RSI/MACD/trend strongly "
            "   contradict it; adjust by ±10 pts maximum.\n"
            "2. Set trend: BULLISH (score > 20), BEARISH (score < -20), NEUTRAL otherwise.\n"
            "3. Set volume_signal from the vol_signal column (HIGH/LOW/NORMAL).\n"
            "4. Use the support and resistance values as provided.\n\n"
            "IMPORTANT — use these EXACT field names in your output JSON (not alternatives):\n"
            "  'macd_signal'         ← use the value from the macd_hist column\n"
            "  'support_level_inr'   ← use the value from the support column\n"
            "  'resistance_level_inr' ← use the value from the resistance column\n"
            "  'rsi'                 ← use the value from the rsi column\n"
            "  'score'               ← the adjusted score (-100 to +100)\n"
            "  'trend'               ← BULLISH, BEARISH, or NEUTRAL\n"
            "  'volume_signal'       ← HIGH, LOW, or NORMAL\n\n"
            f"Return TechnicalSignalBatch with exactly {n} TechnicalSignal objects, "
            "one per symbol, in the same order as the table."
        )
        return Task(
            description=description,
            expected_output=(
                f"TechnicalSignalBatch JSON with a 'signals' list of {n} TechnicalSignal objects."
            ),
            agent=batch_agent,
            output_pydantic=TechnicalSignalBatch,
        )
