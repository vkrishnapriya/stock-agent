"""
src/workflows/director.py
WorkflowDirector — top-level router that classifies user intent and delegates
to either SellerWorkflow or BuyerWorkflow.

  classify_intent(prompt) → "BUY" | "SELL"
      Sends a single-turn Gemini prompt to decide intent.

  parse_portfolio(path) → Portfolio
      Thin wrapper around Portfolio.load_from_file().

  execute(prompt, ...) → FinalReport
      Orchestrates the full round-trip: classify → workflow → report.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog

from src.config.llm_config import get_llm
from src.models.intent import BuyIntent
from src.models.portfolio import Portfolio
from src.models.reports import FinalReport

log = structlog.get_logger(__name__)

_CLASSIFY_PROMPT = (
    "You are a trading intent classifier.\n"
    "Classify the following user request as either BUY or SELL.\n"
    "Respond with exactly one word: BUY or SELL.\n\n"
    "Request: {prompt}"
)

_INTENT_PROMPT = """\
You are a stock screening assistant for Indian equity markets (NSE).
Extract structured buy intent from the user's prompt.

Valid sectors (use exact strings):
  IT, Banking, FMCG, Pharma, Auto, NBFC, Insurance, Insurtech, Power,
  Power Finance, Infra Finance, Metals, Steel, Mining, Cement, Chemicals,
  Paints, Consumer Electricals, Textiles, Beverages, Retail, Diversified,
  Capital Goods, Telecom, Infrastructure, Consumer Tech, Fintech, Logistics,
  Healthcare, Oil & Gas, Agri

Valid styles: growth, value, dividend, momentum, defensive, any

Return a JSON object with these fields (omit or use null for unspecified):
  sectors         — list of matching sector strings (empty list if none)
  exclude_sectors — sectors to avoid (empty list if none)
  max_price_inr   — maximum price per share in INR (null if not mentioned)
  min_price_inr   — minimum price per share in INR (null if not mentioned)
  style           — one of the valid styles above (default "any")
  keywords        — 2-5 short phrases summarising the intent

User prompt: {prompt}

Respond with ONLY the JSON object, no explanation."""


class WorkflowDirector:
    """Routes user prompts to the appropriate analysis workflow.

    Usage::

        director = WorkflowDirector()
        intent   = director.classify_intent("I want to exit my losing positions")
        # → "SELL"

        report = director.execute(
            prompt="Find growth stocks under ₹1000",
            universe="NIFTY100",
            available_cash=200_000.0,
        )
    """

    def __init__(self, llm_provider: str | None = None) -> None:
        self._llm: Any = get_llm(llm_provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_intent(self, prompt: str) -> Literal["BUY", "SELL"]:
        """Use Gemini to classify *prompt* as BUY or SELL.

        Args:
            prompt: Free-text user request.

        Returns:
            ``"BUY"`` or ``"SELL"``.  Defaults to ``"BUY"`` when the
            model's response is ambiguous.
        """
        full_prompt = _CLASSIFY_PROMPT.format(prompt=prompt)
        try:
            response = self._llm.invoke(full_prompt)
            text = (response.content if hasattr(response, "content") else str(response)).strip().upper()
            log.debug("workflow_director.classify_intent", prompt=prompt[:80], response=text)
            if "SELL" in text:
                return "SELL"
            return "BUY"
        except Exception as exc:
            log.warning("workflow_director.classify_failed", error=str(exc))
            return "BUY"

    def extract_intent(self, prompt: str) -> BuyIntent:
        """Parse *prompt* into a structured :class:`~src.models.intent.BuyIntent`.

        Uses a single LLM call. Falls back to an unconstrained BuyIntent on
        any failure so the workflow is never blocked.

        Args:
            prompt: Free-text user buy request.

        Returns:
            :class:`~src.models.intent.BuyIntent` with extracted constraints.
        """
        import json

        full_prompt = _INTENT_PROMPT.format(prompt=prompt)
        try:
            response = self._llm.invoke(full_prompt)
            text = (response.content if hasattr(response, "content") else str(response)).strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            intent = BuyIntent(**{k: v for k, v in data.items() if v is not None})
            log.info(
                "workflow_director.intent_extracted",
                sectors=intent.sectors,
                style=intent.style,
                max_price=intent.max_price_inr,
                min_price=intent.min_price_inr,
            )
            return intent
        except Exception as exc:
            log.warning("workflow_director.intent_failed", error=str(exc))
            return BuyIntent()

    def parse_portfolio(self, path: str) -> Portfolio:
        """Load a :class:`~src.models.portfolio.Portfolio` from *path*.

        Args:
            path: Absolute or relative path to a ``.json`` or ``.csv`` file.

        Returns:
            Validated :class:`~src.models.portfolio.Portfolio`.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        return Portfolio.load_from_file(path)

    def execute(
        self,
        prompt: str,
        portfolio_path: str | None = None,
        universe: str = "NIFTY500",
        available_cash: float = 100_000.0,
    ) -> FinalReport:
        """Classify *prompt*, pick the right workflow, and return a report.

        Args:
            prompt:         Free-text user request.
            portfolio_path: Path to portfolio file (required for SELL workflows).
            universe:       Stock universe for BUY scans (NIFTY50/NIFTY100/NIFTY500).
            available_cash: Capital available for new positions (INR).

        Returns:
            :class:`~src.models.reports.FinalReport` from the chosen workflow.
        """
        # Lazy imports avoid circular dependency at module level
        from src.workflows.buyer_workflow import BuyerWorkflow
        from src.workflows.seller_workflow import SellerWorkflow

        intent = self.classify_intent(prompt)
        log.info("workflow_director.route", intent=intent)

        if intent == "SELL" and portfolio_path:
            portfolio = self.parse_portfolio(portfolio_path)
            return SellerWorkflow().run_as_report(portfolio)

        # BUY (or SELL without a portfolio — default to buy)
        return BuyerWorkflow().run_as_report(
            universe=universe,
            prompt=prompt,
            available_cash=available_cash,
        )
