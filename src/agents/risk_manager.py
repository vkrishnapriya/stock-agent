"""
src/agents/risk_manager.py
RiskManagementAgent — position sizing, circuit checking, and liquidity analysis.

Wires together:
  * PositionSizerTool    — Kelly + fixed-fractional sizing, vol/beta/drawdown
  * CircuitBreakerTool   — NSE circuit band and proximity warning
  * LiquidityCheckerTool — 20-day ADV and market-impact flag

Also reads the current PortfolioState from Redis (key: ``portfolio:state``)
and injects it into the task description for concentration-aware sizing.

The agent's task output is structured as a
:class:`~src.models.signals.RiskAssessment`.  Use :meth:`build_task` to
create a Task with ``output_pydantic=RiskAssessment`` already set.
"""

from __future__ import annotations

import json
import os
from typing import Any

import redis
import structlog
from crewai import Agent, Task

from src.agents.base_agent import BaseAgent
from src.models.signals import RiskAssessment
from src.tools.risk.circuit_checker import CircuitBreakerTool
from src.tools.risk.liquidity_checker import LiquidityCheckerTool
from src.tools.risk.position_sizer import PositionSizerTool

log = structlog.get_logger(__name__)

_PORTFOLIO_STATE_KEY = "portfolio:state"

_ROLE = "Senior Risk Manager for Indian Equity Portfolios"

_GOAL = (
    "Quantify all material risks for a proposed NSE equity trade: position sizing "
    "via Kelly criterion, circuit breaker proximity, liquidity / market-impact, "
    "and portfolio concentration. Produce a structured RiskAssessment with a "
    "risk_score (0=safe, 100=maximum risk) and a concrete stop-loss in INR."
)

_BACKSTORY = (
    "You are a CFA charterholder with 12 years of risk management experience at "
    "a SEBI-registered Portfolio Management Service (PMS). You have implemented "
    "Kelly-criterion position sizing, enforced NSE circuit-breaker compliance, and "
    "built liquidity-risk dashboards for mid-cap portfolios. "
    "You always validate that a proposed position respects: (1) NSE circuit limits, "
    "(2) the 10%-of-ADV market-impact threshold, and (3) the 2% portfolio risk rule. "
    "You never approve a trade without a clearly defined stop-loss and you always "
    "check the existing portfolio concentration before adding a new position."
)


# ---------------------------------------------------------------------------
# Portfolio state helpers
# ---------------------------------------------------------------------------


def get_portfolio_state() -> dict[str, Any] | None:
    """Read current portfolio state JSON from Redis.

    Returns:
        Parsed portfolio dict, or None if not set or Redis unavailable.
    """
    try:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(url, decode_responses=True)
        raw = client.get(_PORTFOLIO_STATE_KEY)
        client.close()
        if raw:
            return json.loads(raw)  # type: ignore[return-value]
    except Exception as exc:
        log.warning("portfolio_state.read_failed", error=str(exc))
    return None


def set_portfolio_state(portfolio_dict: dict[str, Any]) -> None:
    """Persist portfolio state to Redis so the risk manager can read it.

    Args:
        portfolio_dict: JSON-serialisable representation of the current portfolio.
                        Typically produced via ``Portfolio.model_dump()``.
    """
    try:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(url, decode_responses=True)
        client.set(_PORTFOLIO_STATE_KEY, json.dumps(portfolio_dict, default=str))
        client.close()
        log.info("portfolio_state.saved")
    except Exception as exc:
        log.warning("portfolio_state.save_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class RiskManagementAgent(BaseAgent):
    """CrewAI agent that performs full risk assessment for a single NSE trade proposal.

    Usage::

        agent = RiskManagementAgent()
        task  = agent.build_task(
            symbol="INFY",
            entry_price=1500.0,
            stop_loss=1430.0,
            portfolio_value=1_000_000.0,
        )
        crew = Crew(agents=[agent.build()], tasks=[task])
        result = crew.kickoff()
    """

    #: The Pydantic model this agent is designed to produce.
    output_model: type[RiskAssessment] = RiskAssessment

    def __init__(
        self,
        extra_tools: list[Any] | None = None,
        llm_provider: str | None = None,
    ) -> None:
        tools = [
            PositionSizerTool(),
            CircuitBreakerTool(),
            LiquidityCheckerTool(),
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
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
        win_rate: float = 0.55,
        reward_risk_ratio: float = 2.0,
    ) -> Task:
        """Create a CrewAI Task configured to produce a :class:`RiskAssessment`.

        Args:
            symbol:            NSE ticker without suffix (e.g. ``"INFY"``).
            entry_price:       Intended entry price in INR.
            stop_loss:         Stop-loss price in INR.
            portfolio_value:   Total portfolio value in INR.
            win_rate:          Historical win rate for Kelly sizing (default 0.55).
            reward_risk_ratio: Expected reward:risk ratio (default 2.0).

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=RiskAssessment``.
        """
        symbol_upper = symbol.strip().upper()
        yf_symbol = f"{symbol_upper}.NS"

        # ── Inject portfolio context from Redis ─────────────────────────────
        portfolio_context = ""
        state = get_portfolio_state()
        if state:
            holdings = state.get("holdings", [])
            existing = next(
                (h for h in holdings if h.get("symbol", "").upper() == symbol_upper),
                None,
            )
            cash = state.get("available_cash_inr", 0)
            portfolio_context = (
                f"\n\nCurrent portfolio context (from Redis):\n"
                f"  Available cash  : {cash:,.0f} INR\n"
                f"  Existing {symbol_upper} position: "
                f"{existing if existing else 'None'}\n"
                f"  Total holdings  : {len(holdings)}\n"
            )

        description = (
            f"Perform a complete risk assessment for a proposed trade in "
            f"**{symbol_upper}** (Yahoo Finance: {yf_symbol}).\n\n"
            f"Trade parameters:\n"
            f"  entry_price       = {entry_price:,.2f} INR\n"
            f"  stop_loss         = {stop_loss:,.2f} INR\n"
            f"  portfolio_value   = {portfolio_value:,.0f} INR\n"
            f"  win_rate          = {win_rate}\n"
            f"  reward_risk_ratio = {reward_risk_ratio}\n"
            f"{portfolio_context}\n"
            "Steps:\n"
            f"1. Call `circuit_checker` (symbol={symbol_upper}) — get the NSE "
            "   circuit band and check if the price is near a circuit limit.\n"
            f"2. Call `position_sizer` (symbol={symbol_upper}, "
            f"   entry_price={entry_price}, stop_loss={stop_loss}, "
            f"   portfolio_value={portfolio_value}, win_rate={win_rate}, "
            f"   reward_risk_ratio={reward_risk_ratio}) — get Kelly + fixed-frac "
            "   position size, quantity, volatility_pct, beta, max_drawdown_pct.\n"
            f"3. Call `liquidity_checker` (symbol={yf_symbol}, "
            "   intended_trade_value_inr=<position_size_inr from step 2>) — "
            "   assess market-impact risk.\n"
            "4. Synthesise into a RiskAssessment:\n"
            "   - risk_score (0–100): weight vol 30%, circuit proximity 20%, "
            "     liquidity flag 20%, beta 15%, drawdown 15%.\n"
            "   - volatility_pct, beta, max_drawdown_pct: from step 2.\n"
            "   - circuit_breaker_band: from step 1.\n"
            "   - suggested_stop_loss_inr: the provided stop_loss — confirm it is "
            "     above the circuit lower_limit from step 1.\n"
            "   - position_size_pct: position_size_pct from step 2.\n"
        )

        expected_output = (
            "A valid RiskAssessment JSON with fields: symbol, exchange, risk_score, "
            "volatility_pct, beta, max_drawdown_pct, circuit_breaker_band, "
            "suggested_stop_loss_inr, position_size_pct, generated_at."
        )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=RiskAssessment,
        )
