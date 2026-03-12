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

_ROLE = "Senior Technical Analyst for Indian Equity Markets"

_GOAL = (
    "Analyse price action, volume dynamics, and multi-timeframe technical "
    "indicators for NSE-listed equities to identify high-probability entry "
    "and exit signals. Produce a structured TechnicalSignal with a composite "
    "score in the range -100 (extreme bearish) to +100 (extreme bullish), "
    "along with concrete support and resistance levels in INR."
)

_BACKSTORY = (
    "You are a Chartered Market Technician (CMT) with 15 years of experience "
    "analysing Indian equity markets. You specialise in reading NSE price action "
    "across multiple time-frames, combining momentum indicators (RSI, MACD) with "
    "trend filters (SMA 50/200), volatility bands (Bollinger Bands), and volume "
    "confirmation (OBV, VWAP) to form high-conviction views. "
    "You are intimately familiar with NSE-specific dynamics: circuit breakers, "
    "F&O expiry effects on open interest, and how FII/DII flows influence "
    "large-cap technicals. "
    "Your analysis always respects the tick size of ₹0.05 and you never "
    "recommend positions without a clearly defined stop-loss and target."
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
        period: str = "1y",
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
            f"Perform a complete technical analysis for **{symbol.upper()}** "
            f"(Yahoo Finance symbol: {yf_symbol}) using {period} of {interval} data.\n\n"
            "Steps:\n"
            f"1. Fetch OHLCV data using the `ohlcv_fetch` tool "
            f"   (symbol={yf_symbol}, period={period}, interval={interval}).\n"
            "2. Compute technical indicators using the `indicator_engine` tool "
            f"   (pass the OHLCV JSON and symbol={symbol.upper()}).\n"
            "3. Fetch support/resistance levels using the `support_resistance` tool "
            f"   (symbol={yf_symbol}).\n"
            "4. Synthesise all results into a final TechnicalSignal. "
            "   Use the pivot-based support/resistance levels from step 3 when "
            "   they are more precise than the SMA-derived levels from step 2.\n"
            "5. Ensure the final score is in [-100, +100] and the trend label "
            "   (BULLISH/BEARISH/NEUTRAL) is consistent with the score."
        )
        expected_output = (
            "A valid TechnicalSignal JSON with fields: symbol, exchange, score, "
            "trend, rsi, macd_signal, support_level_inr, resistance_level_inr, "
            "volume_signal, generated_at."
        )
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=TechnicalSignal,
        )
