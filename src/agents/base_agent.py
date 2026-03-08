"""
src/agents/base_agent.py
Abstract base class for all trading agents. Provides LLM wiring, retry logic,
and structured logging. All agents MUST extend this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog
from crewai import Agent
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.llm_config import get_llm

log = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all stock-agent trading agents.

    Subclasses must implement :meth:`build` which returns a configured
    :class:`crewai.Agent` instance.

    Example::

        class ResearchAgent(BaseAgent):
            def build(self) -> Agent:
                return Agent(
                    role="Senior Market Analyst",
                    goal="...",
                    backstory="...",
                    llm=self.llm,
                    tools=self.tools,
                )
    """

    def __init__(
        self,
        tools: list[Any] | None = None,
        llm_provider: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.tools: list[Any] = tools or []
        self.llm = get_llm(llm_provider)
        self.verbose = verbose
        self._log = structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    def build(self) -> Agent:
        """Return a fully configured CrewAI Agent."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def run_with_retry(self, task: Any) -> Any:
        """Execute a task with automatic exponential-backoff retry.

        Args:
            task: A :class:`crewai.Task` instance.

        Returns:
            The agent's output string.
        """
        agent = self.build()
        self._log.info("agent.run_start", agent_role=agent.role)
        result = agent.execute_task(task)
        self._log.info("agent.run_complete", agent_role=agent.role)
        return result
