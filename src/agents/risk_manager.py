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
from src.models.signals import RiskAssessment, RiskAssessmentBatch
from src.tools.risk.circuit_checker import CircuitBreakerTool
from src.tools.risk.liquidity_checker import LiquidityCheckerTool
from src.tools.risk.position_sizer import PositionSizerTool

log = structlog.get_logger(__name__)

_PORTFOLIO_STATE_KEY = "portfolio:state"

_ROLE = "Risk Manager for Indian Equity Portfolios"

_GOAL = (
    "Assess trade risk: Kelly position sizing, circuit breaker proximity, "
    "liquidity impact. Produce RiskAssessment with risk_score (0-100) and stop-loss."
)

_BACKSTORY = (
    "CFA with 10 years PMS risk management. Validate NSE circuit limits, "
    "10%-ADV market-impact threshold, and 2% portfolio risk rule before approving trades."
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
            f"Risk assessment for {symbol_upper} ({yf_symbol}). "
            f"entry={entry_price:,.2f} stop={stop_loss:,.2f} portfolio={portfolio_value:,.0f} INR."
            f"{portfolio_context}\n"
            f"1. `circuit_checker` (symbol={symbol_upper})\n"
            f"2. `position_sizer` (symbol={symbol_upper}, entry_price={entry_price}, "
            f"stop_loss={stop_loss}, portfolio_value={portfolio_value}, "
            f"win_rate={win_rate}, reward_risk_ratio={reward_risk_ratio})\n"
            f"3. `liquidity_checker` (symbol={yf_symbol}, intended_trade_value_inr=<from step 2>)\n"
            "4. Return RiskAssessment: risk_score 0-100 (vol 30%, circuit 20%, "
            "liquidity 20%, beta 15%, drawdown 15%), stop-loss confirmed above circuit lower limit."
        )

        expected_output = (
            "RiskAssessment JSON: symbol, exchange, risk_score, volatility_pct, beta, "
            "max_drawdown_pct, circuit_breaker_band, suggested_stop_loss_inr, "
            "position_size_pct, generated_at."
        )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.build(),
            output_pydantic=RiskAssessment,
        )

    def build_batch_task(self, table: str, symbols: list[str], portfolio_value: float) -> Task:
        """Create a single Task that assesses risk for all *symbols* from a pre-fetched table.

        Args:
            table: Markdown table with columns: symbol, entry, stop, vol_pct, beta,
                   max_dd_pct, circuit_band, near_circuit, kelly_pct, pos_size_pct, adv_cr.
            symbols: Ordered list of NSE symbols present in the table.
            portfolio_value: Total portfolio value in INR (for context).

        Returns:
            A :class:`crewai.Task` with ``output_pydantic=RiskAssessmentBatch``.
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
            f"Assess trade risk for {n} NSE stocks using the pre-fetched data below.\n"
            f"Portfolio value: ₹{portfolio_value:,.0f}\n\n"
            f"{table}\n\n"
            "Risk score (0–100) per stock — higher = riskier:\n"
            "  Volatility (30%): vol_pct/40 × 30  (cap at 30)\n"
            "  Circuit proximity (20%): near_circuit=True → +20, else band-based\n"
            "  Liquidity (20%): adv_cr<1 → +20, adv_cr<5 → +10, else 0\n"
            "  Beta (15%): |beta-1| × 10, cap at 15\n"
            "  Max drawdown (15%): max_dd_pct/50 × 15, cap at 15\n"
            "  Stop-loss: use the 'stop' column; ensure it is above circuit lower limit.\n"
            "  Position size: use pos_size_pct from table as suggested allocation.\n\n"
            f"Return RiskAssessmentBatch with exactly {n} RiskAssessment objects "
            "in the same order as the table."
        )
        return Task(
            description=description,
            expected_output=(
                f"RiskAssessmentBatch JSON with an 'assessments' list of {n} RiskAssessment objects."
            ),
            agent=batch_agent,
            output_pydantic=RiskAssessmentBatch,
        )
