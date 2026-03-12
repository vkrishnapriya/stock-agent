"""
src/agents/base_agent.py
Abstract base class for all trading agents.

All agents MUST extend BaseAgent and implement :meth:`build`.
Common CrewAI settings (verbose, max_iter, memory) are enforced here so
individual agents only need to declare role / goal / backstory / tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog
from crewai import Agent

from src.config.llm_config import get_llm

log = structlog.get_logger(__name__)

# Defaults applied to every Agent built by subclasses
_AGENT_VERBOSE: bool = True
_AGENT_MAX_ITER: int = 5
_AGENT_MEMORY: bool = False


class BaseAgent(ABC):
    """Base class for all stock-agent trading agents.

    Subclasses must implement :meth:`build` and return a configured
    :class:`crewai.Agent`.  Common settings are provided via helpers so
    agent code stays focused on role, goal, and backstory.

    Example::

        class TechnicalAnalysisAgent(BaseAgent):
            def build(self) -> Agent:
                return Agent(
                    role="Technical Analyst",
                    goal="Identify entry and exit signals from price action",
                    backstory="...",
                    llm=self._get_llm(),
                    tools=self.tools,
                    **self._agent_defaults(),
                )
    """

    def __init__(
        self,
        tools: list[Any] | None = None,
        llm_provider: str | None = None,
        verbose: bool = _AGENT_VERBOSE,
        max_iter: int = _AGENT_MAX_ITER,
        memory: bool = _AGENT_MEMORY,
    ) -> None:
        self.tools: list[Any] = tools or []
        self._llm_provider = llm_provider
        self.verbose = verbose
        self.max_iter = max_iter
        self.memory = memory
        self._log = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _get_llm(self, provider: str | None = None) -> Any:
        """Return the LLM configured for this agent.

        Args:
            provider: Override the instance-level provider.  Falls back to
                      the value passed at construction time, then to the
                      ``LLM_PROVIDER`` environment variable.

        Returns:
            A LangChain chat model compatible with CrewAI.
        """
        return get_llm(provider or self._llm_provider)

    # ------------------------------------------------------------------
    # Shared CrewAI keyword arguments
    # ------------------------------------------------------------------

    def _agent_defaults(self) -> dict[str, Any]:
        """Return common keyword arguments to pass to :class:`crewai.Agent`.

        Usage in :meth:`build`::

            return Agent(
                role="...",
                goal="...",
                backstory="...",
                llm=self._get_llm(),
                tools=self.tools,
                **self._agent_defaults(),
            )
        """
        return {
            "verbose": self.verbose,
            "max_iter": self.max_iter,
            "memory": self.memory,
        }

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build(self) -> Agent:
        """Return a fully configured CrewAI Agent.

        Must call :meth:`_get_llm` for the LLM and :meth:`_agent_defaults`
        for common keyword arguments.
        """

    # ------------------------------------------------------------------
    # Execution helper
    # ------------------------------------------------------------------

    def run_with_retry(self, task: Any) -> Any:
        """Execute *task* using the built agent, with retry on failure.

        Uses :func:`src.utils.retry.retry` (3 attempts, ×2 backoff).

        Args:
            task: A :class:`crewai.Task` instance.

        Returns:
            The agent's output string.
        """
        from src.utils.retry import retry  # local import to avoid circular deps

        @retry(max_attempts=3, backoff_factor=2.0, base_delay=2.0)
        def _run() -> Any:
            agent = self.build()
            self._log.info("agent.run_start", agent_role=agent.role)
            result = agent.execute_task(task)
            self._log.info("agent.run_complete", agent_role=agent.role)
            return result

        return _run()
