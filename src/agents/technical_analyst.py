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
from src.models.signals import TechnicalSignal
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
