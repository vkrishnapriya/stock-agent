"""
tests/unit/test_news_sentiment.py
Unit tests for News & Sentiment Agent and its three tools.

All network calls (NSESession, gnews.GNews, TavilyClient) are mocked.
Redis is bypassed via patched _get_cached / _set_cached helpers.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_rate_limit():
    return patch(
        "src.tools.base_tool.BaseTool._check_rate_limit",
        new_callable=AsyncMock,
    )


@contextmanager
def _patch_cache():
    with (
        patch(
            "src.tools.base_tool.BaseTool._get_cached",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "src.tools.base_tool.BaseTool._set_cached",
            new_callable=AsyncMock,
        ),
    ):
        yield


def _sample_nse_response(symbol: str = "INFY") -> dict:
    """Fake NSE corporate-announcements API response."""
    return {
        "data": [
            {
                "symbol": symbol,
                "sm_name": "Infosys Limited",
                "sort_date": "09-Mar-2026 15:30:00",
                "desc": "Board Meeting - Financial Results Q3",
                "attchmntFile": "INFY_Q3_2026.pdf",
                "bcastdttm": "09/03/2026 15:30:00",
            },
            {
                "symbol": symbol,
                "sm_name": "Infosys Limited",
                "sort_date": "08-Mar-2026 10:00:00",
                "desc": "Change in Management",
                "attchmntFile": "INFY_MGMT.pdf",
                "bcastdttm": "08/03/2026 10:00:00",
            },
        ]
    }


def _sample_gnews_response() -> list:
    """Fake gnews article list."""
    return [
        {
            "title": "Infosys beats Q3 estimates on strong deal wins",
            "description": "Infosys reported Q3 revenue of ₹38,000 crore...",
            "url": "https://example.com/infy-q3",
            "published date": "Fri, 07 Mar 2026 09:00:00 GMT",
            "publisher": {"title": "Economic Times"},
        },
        {
            "title": "INFY.NS: FII holding rises to 33%",
            "description": "Foreign institutional investors increased...",
            "url": "https://example.com/infy-fii",
            "published date": "Thu, 06 Mar 2026 12:00:00 GMT",
            "publisher": "Reuters",
        },
    ]


def _sample_tavily_response() -> dict:
    """Fake Tavily search response."""
    return {
        "results": [
            {
                "title": "Infosys Q3 FY26 earnings preview",
                "content": "Analysts expect Infosys to report strong Q3 numbers...",
                "url": "https://example.com/infy-preview",
                "published_date": "2026-03-05",
                "score": 0.92,
            },
            {
                "title": "Infosys raises FY26 guidance",
                "content": "Infosys management upgraded full-year revenue guidance...",
                "url": "https://example.com/infy-guidance",
                "published_date": "2026-03-09",
                "score": 0.88,
            },
        ]
    }


# ---------------------------------------------------------------------------
# NSEAnnouncementTool
# ---------------------------------------------------------------------------


class TestNSEAnnouncementTool:
    def test_returns_list_of_announcements(self):
        from src.tools.news.nse_announcements import NSEAnnouncementTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.nse_announcements.NSESession.get",
                return_value=_sample_nse_response("INFY"),
            ):
                tool = NSEAnnouncementTool()
                result = json.loads(tool._run(symbol="INFY"))

        assert isinstance(result, list)
        assert len(result) == 2

    def test_required_keys_present(self):
        from src.tools.news.nse_announcements import NSEAnnouncementTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.nse_announcements.NSESession.get",
                return_value=_sample_nse_response("INFY"),
            ):
                tool = NSEAnnouncementTool()
                result = json.loads(tool._run(symbol="INFY"))

        for item in result:
            assert {"symbol", "company", "date", "description", "attachment"} <= set(item.keys())

    def test_filters_to_requested_symbol(self):
        from src.tools.news.nse_announcements import NSEAnnouncementTool

        # Response contains both INFY and TCS announcements
        mixed_response = {
            "data": [
                {"symbol": "INFY", "sm_name": "Infosys", "sort_date": "09-Mar-2026", "desc": "INFY news"},
                {"symbol": "TCS", "sm_name": "TCS", "sort_date": "09-Mar-2026", "desc": "TCS news"},
            ]
        }
        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.nse_announcements.NSESession.get",
                return_value=mixed_response,
            ):
                tool = NSEAnnouncementTool()
                result = json.loads(tool._run(symbol="INFY"))

        assert all(r["symbol"] == "INFY" for r in result)
        assert len(result) == 1

    def test_capped_at_ten_results(self):
        from src.tools.news.nse_announcements import NSEAnnouncementTool

        many = {
            "data": [
                {
                    "symbol": "INFY",
                    "sm_name": "Infosys",
                    "sort_date": f"0{i % 9 + 1}-Mar-2026",
                    "desc": f"Ann {i}",
                }
                for i in range(15)
            ]
        }
        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.nse_announcements.NSESession.get",
                return_value=many,
            ):
                tool = NSEAnnouncementTool()
                result = json.loads(tool._run(symbol="INFY"))

        assert len(result) <= 10

    def test_handles_list_response(self):
        """API sometimes returns a bare list instead of {"data": [...]}."""
        from src.tools.news.nse_announcements import NSEAnnouncementTool

        bare_list = [
            {"symbol": "INFY", "sm_name": "Infosys", "sort_date": "09-Mar-2026", "desc": "OK"},
        ]
        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.nse_announcements.NSESession.get",
                return_value=bare_list,
            ):
                tool = NSEAnnouncementTool()
                result = json.loads(tool._run(symbol="INFY"))

        assert len(result) == 1

    def test_symbol_uppercased(self):
        from src.tools.news.nse_announcements import NSEAnnouncementTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.nse_announcements.NSESession.get",
                return_value=_sample_nse_response("INFY"),
            ):
                tool = NSEAnnouncementTool()
                result = json.loads(tool._run(symbol="infy"))

        assert all(r["symbol"] == "INFY" for r in result)

    def test_cache_hit_skips_nse(self):
        from src.tools.news.nse_announcements import NSEAnnouncementTool

        cached = json.dumps([{"symbol": "INFY", "company": "Infosys",
                               "date": "", "description": "cached", "attachment": ""}])
        with _patch_rate_limit():
            with (
                patch(
                    "src.tools.base_tool.BaseTool._get_cached",
                    new_callable=AsyncMock,
                    return_value=cached,
                ),
                patch(
                    "src.tools.base_tool.BaseTool._set_cached",
                    new_callable=AsyncMock,
                ),
                patch("src.tools.news.nse_announcements.NSESession.get") as mock_nse,
            ):
                tool = NSEAnnouncementTool()
                result = json.loads(tool._run(symbol="INFY"))

        mock_nse.assert_not_called()
        assert result[0]["description"] == "cached"

    def test_empty_response_returns_empty_list(self):
        from src.tools.news.nse_announcements import NSEAnnouncementTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.nse_announcements.NSESession.get",
                return_value={"data": []},
            ):
                tool = NSEAnnouncementTool()
                result = json.loads(tool._run(symbol="INFY"))

        assert result == []


# ---------------------------------------------------------------------------
# GNewsAPITool
# ---------------------------------------------------------------------------


class TestGNewsAPITool:
    def test_returns_article_list(self):
        from src.tools.news.gnews_tool import GNewsAPITool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.gnews_tool.gnews.GNews.get_news",
                return_value=_sample_gnews_response(),
            ):
                tool = GNewsAPITool()
                result = json.loads(tool._run(symbol="INFY", company_name="Infosys"))

        assert isinstance(result, list)
        assert len(result) == 2

    def test_required_keys_present(self):
        from src.tools.news.gnews_tool import GNewsAPITool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.gnews_tool.gnews.GNews.get_news",
                return_value=_sample_gnews_response(),
            ):
                tool = GNewsAPITool()
                result = json.loads(tool._run(symbol="INFY", company_name="Infosys"))

        for item in result:
            assert {"title", "description", "url", "published_at", "publisher"} <= set(item.keys())

    def test_published_date_normalised(self):
        from src.tools.news.gnews_tool import GNewsAPITool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.gnews_tool.gnews.GNews.get_news",
                return_value=_sample_gnews_response(),
            ):
                tool = GNewsAPITool()
                result = json.loads(tool._run(symbol="INFY", company_name="Infosys"))

        # The gnews key 'published date' (with space) must be mapped to 'published_at'
        assert result[0]["published_at"] == "Fri, 07 Mar 2026 09:00:00 GMT"

    def test_publisher_dict_extracted(self):
        from src.tools.news.gnews_tool import GNewsAPITool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.gnews_tool.gnews.GNews.get_news",
                return_value=_sample_gnews_response(),
            ):
                tool = GNewsAPITool()
                result = json.loads(tool._run(symbol="INFY", company_name="Infosys"))

        # First article has publisher as dict {"title": "Economic Times"}
        assert result[0]["publisher"] == "Economic Times"
        # Second article has publisher as plain string
        assert result[1]["publisher"] == "Reuters"

    def test_empty_gnews_returns_empty_list(self):
        from src.tools.news.gnews_tool import GNewsAPITool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.gnews_tool.gnews.GNews.get_news",
                return_value=[],
            ):
                tool = GNewsAPITool()
                result = json.loads(tool._run(symbol="ZOMATO"))

        assert result == []

    def test_falls_back_to_symbol_when_company_name_empty(self):
        from src.tools.news.gnews_tool import GNewsAPITool

        mock_get_news = MagicMock(return_value=[])
        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.gnews_tool.gnews.GNews.get_news",
                mock_get_news,
            ):
                tool = GNewsAPITool()
                tool._run(symbol="INFY", company_name="")

        # Should have been called with a query containing the symbol
        args = mock_get_news.call_args[0]
        assert "INFY" in args[0]

    def test_rate_limit_override_uses_gnews_limiter(self):
        """_check_rate_limit must use GNEWS (100/day), not BaseTool default."""
        import asyncio
        from src.tools.news.gnews_tool import GNewsAPITool, _GNEWS_LIMITER

        tool = GNewsAPITool()
        # Verify _check_rate_limit calls the GNEWS limiter, not BaseTool's
        acquire_calls = []

        async def _fake_acquire():
            acquire_calls.append(True)
            return True

        with patch.object(_GNEWS_LIMITER, "acquire", side_effect=_fake_acquire):
            asyncio.run(tool._check_rate_limit())

        assert len(acquire_calls) == 1

    def test_cache_hit_skips_gnews(self):
        from src.tools.news.gnews_tool import GNewsAPITool

        cached = json.dumps([{"title": "cached", "description": "",
                               "url": "", "published_at": "", "publisher": ""}])
        with _patch_rate_limit():
            with (
                patch(
                    "src.tools.base_tool.BaseTool._get_cached",
                    new_callable=AsyncMock,
                    return_value=cached,
                ),
                patch("src.tools.base_tool.BaseTool._set_cached", new_callable=AsyncMock),
                patch("src.tools.news.gnews_tool.gnews.GNews.get_news") as mock_gn,
            ):
                tool = GNewsAPITool()
                result = json.loads(tool._run(symbol="INFY", company_name="Infosys"))

        mock_gn.assert_not_called()
        assert result[0]["title"] == "cached"


# ---------------------------------------------------------------------------
# TavilySearchTool
# ---------------------------------------------------------------------------


class TestTavilySearchTool:
    def test_returns_result_list(self):
        from src.tools.news.tavily_tool import TavilySearchTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.tavily_tool.TavilyClient.search",
                return_value=_sample_tavily_response(),
            ):
                tool = TavilySearchTool()
                result = json.loads(tool._run(symbol="INFY", company_name="Infosys"))

        assert isinstance(result, list)
        assert len(result) == 2

    def test_required_keys_present(self):
        from src.tools.news.tavily_tool import TavilySearchTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.tavily_tool.TavilyClient.search",
                return_value=_sample_tavily_response(),
            ):
                tool = TavilySearchTool()
                result = json.loads(tool._run(symbol="INFY", company_name="Infosys"))

        for item in result:
            assert {"title", "content", "url", "published_date", "score"} <= set(item.keys())

    def test_score_rounded_to_four_decimals(self):
        from src.tools.news.tavily_tool import TavilySearchTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.tavily_tool.TavilyClient.search",
                return_value=_sample_tavily_response(),
            ):
                tool = TavilySearchTool()
                result = json.loads(tool._run(symbol="INFY", company_name="Infosys"))

        assert result[0]["score"] == 0.92
        assert result[1]["score"] == 0.88

    def test_no_api_key_raises(self):
        from src.tools.news.tavily_tool import TavilySearchTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.tavily_tool.get_settings",
                return_value=MagicMock(tavily_api_key=""),
            ):
                tool = TavilySearchTool()
                with pytest.raises(ValueError, match="TAVILY_API_KEY"):
                    tool._run(symbol="INFY")

    def test_empty_results_returns_empty_list(self):
        from src.tools.news.tavily_tool import TavilySearchTool

        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.tavily_tool.TavilyClient.search",
                return_value={"results": []},
            ):
                tool = TavilySearchTool()
                result = json.loads(tool._run(symbol="INFY"))

        assert result == []

    def test_search_called_with_advanced_depth(self):
        from src.tools.news.tavily_tool import TavilySearchTool

        mock_search = MagicMock(return_value={"results": []})
        with _patch_rate_limit(), _patch_cache():
            with patch(
                "src.tools.news.tavily_tool.TavilyClient.search",
                mock_search,
            ):
                tool = TavilySearchTool()
                tool._run(symbol="INFY", company_name="Infosys")

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs.get("search_depth") == "advanced"
        assert call_kwargs.get("max_results") == 5

    def test_rate_limit_override_uses_tavily_limiter(self):
        import asyncio
        from src.tools.news.tavily_tool import TavilySearchTool, _TAVILY_LIMITER

        tool = TavilySearchTool()
        acquire_calls = []

        async def _fake_acquire():
            acquire_calls.append(True)
            return True

        with patch.object(_TAVILY_LIMITER, "acquire", side_effect=_fake_acquire):
            asyncio.run(tool._check_rate_limit())

        assert len(acquire_calls) == 1

    def test_cache_hit_skips_tavily(self):
        from src.tools.news.tavily_tool import TavilySearchTool

        cached = json.dumps([{"title": "cached", "content": "",
                               "url": "", "published_date": "", "score": 0.5}])
        with _patch_rate_limit():
            with (
                patch(
                    "src.tools.base_tool.BaseTool._get_cached",
                    new_callable=AsyncMock,
                    return_value=cached,
                ),
                patch("src.tools.base_tool.BaseTool._set_cached", new_callable=AsyncMock),
                patch("src.tools.news.tavily_tool.TavilyClient.search") as mock_tv,
            ):
                tool = TavilySearchTool()
                result = json.loads(tool._run(symbol="INFY"))

        mock_tv.assert_not_called()
        assert result[0]["title"] == "cached"


# ---------------------------------------------------------------------------
# NewsSentimentAgent
# ---------------------------------------------------------------------------


class TestNewsSentimentAgent:
    def test_build_returns_crewai_agent(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = NewsSentimentAgent()
            crewai_agent = agent.build()

        from crewai import Agent
        assert isinstance(crewai_agent, Agent)

    def test_agent_has_three_tools(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        agent = NewsSentimentAgent()
        assert len(agent.tools) == 3

    def test_tool_types(self):
        from src.agents.news_sentiment import NewsSentimentAgent
        from src.tools.news.gnews_tool import GNewsAPITool
        from src.tools.news.nse_announcements import NSEAnnouncementTool
        from src.tools.news.tavily_tool import TavilySearchTool

        agent = NewsSentimentAgent()
        tool_types = {type(t) for t in agent.tools}
        assert tool_types == {NSEAnnouncementTool, GNewsAPITool, TavilySearchTool}

    def test_agent_role_contains_analyst(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = NewsSentimentAgent()
            crewai_agent = agent.build()

        assert "Analyst" in crewai_agent.role

    def test_agent_defaults_applied(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = NewsSentimentAgent()
            crewai_agent = agent.build()

        assert crewai_agent.verbose is True
        assert crewai_agent.max_iter == 5

    def test_build_task_output_pydantic(self):
        from src.agents.news_sentiment import NewsSentimentAgent
        from src.models.signals import SentimentResult

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = NewsSentimentAgent()
            task = agent.build_task(symbol="INFY", company_name="Infosys")

        assert task.output_pydantic is SentimentResult

    def test_build_task_description_contains_symbol(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = NewsSentimentAgent()
            task = agent.build_task(symbol="RELIANCE", company_name="Reliance Industries")

        assert "RELIANCE" in task.description
        assert "Reliance Industries" in task.description

    def test_build_task_mentions_all_three_tools(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = NewsSentimentAgent()
            task = agent.build_task(symbol="INFY")

        assert "nse_announcements" in task.description
        assert "gnews_search" in task.description
        assert "tavily_search" in task.description

    def test_build_task_falls_back_to_symbol_when_no_company_name(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = NewsSentimentAgent()
            task = agent.build_task(symbol="TCS")

        # company_name defaults to symbol
        assert "TCS" in task.description

    def test_output_model_class_attribute(self):
        from src.agents.news_sentiment import NewsSentimentAgent
        from src.models.signals import SentimentResult

        agent = NewsSentimentAgent()
        assert agent.output_model is SentimentResult

    def test_extra_tools_appended(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        extra = MagicMock()
        agent = NewsSentimentAgent(extra_tools=[extra])
        assert len(agent.tools) == 4
        assert agent.tools[-1] is extra

    def test_build_task_description_mentions_material_events(self):
        from src.agents.news_sentiment import NewsSentimentAgent

        with patch("src.agents.base_agent.get_llm", return_value=MagicMock()):
            agent = NewsSentimentAgent()
            task = agent.build_task(symbol="INFY")

        # Key material events should be in the task description
        assert "earnings" in task.description.lower()
        assert "management" in task.description.lower()
