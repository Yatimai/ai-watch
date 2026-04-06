"""Tests for src/agent/nodes.py — node functions and helpers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.nodes import (
    _format_items_for_prompt,
    _llm_invoke_with_retry,
    _match_tool_call_to_item,
    combine_items,
    enrich_and_brief,
    fetch_sources,
    filter_github,
)
from src.agent.state import AgentState
from src.models.schemas import Item, SourceType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hf_item() -> Item:
    return Item(
        title="Paper A",
        url="https://huggingface.co/papers/123",
        source=SourceType.HUGGINGFACE,
        abstract="Abstract text for paper A",
        upvotes=50,
        authors="Author1",
    )


@pytest.fixture()
def gh_item() -> Item:
    return Item(
        title="owner/repo",
        url="https://github.com/owner/repo",
        source=SourceType.GITHUB,
        description="Desc of the repo",
        stars_today=100,
        language="Python",
        repo_owner="owner",
        repo_name="repo",
    )


@pytest.fixture()
def simon_item() -> Item:
    return Item(
        title="Post A",
        url="https://simonwillison.net/post/1",
        source=SourceType.SIMON,
        tags=["ai", "llm"],
        content_snippet="Snippet text about something interesting in AI land",
    )


@pytest.fixture()
def all_items(hf_item: Item, gh_item: Item, simon_item: Item) -> list[Item]:
    return [hf_item, gh_item, simon_item]


@pytest.fixture()
def item_by_url(all_items: list[Item]) -> dict[str, Item]:
    return {item.url: item for item in all_items}


# ---------------------------------------------------------------------------
# _match_tool_call_to_item
# ---------------------------------------------------------------------------


class TestMatchToolCallToItem:
    """Tests for _match_tool_call_to_item()."""

    def test_fetch_url_exact_match(
        self, hf_item: Item, all_items: list[Item], item_by_url: dict[str, Item]
    ):
        result = _match_tool_call_to_item(
            "fetch_url",
            {"url": "https://huggingface.co/papers/123"},
            item_by_url,
            all_items,
        )
        assert result is hf_item

    def test_fetch_url_partial_match(
        self, hf_item: Item, all_items: list[Item], item_by_url: dict[str, Item]
    ):
        """Partial match: item.url is a substring of the tool arg URL."""
        result = _match_tool_call_to_item(
            "fetch_url",
            {"url": "https://huggingface.co/papers/123?extra=1"},
            item_by_url,
            all_items,
        )
        assert result is hf_item

    def test_get_github_repo_match(
        self, gh_item: Item, all_items: list[Item], item_by_url: dict[str, Item]
    ):
        result = _match_tool_call_to_item(
            "get_github_repo",
            {"owner": "owner", "repo": "repo"},
            item_by_url,
            all_items,
        )
        assert result is gh_item

    def test_search_hf_models_match_by_query_in_title(
        self, hf_item: Item, all_items: list[Item], item_by_url: dict[str, Item]
    ):
        result = _match_tool_call_to_item(
            "search_hf_models",
            {"query": "paper a"},
            item_by_url,
            all_items,
        )
        assert result is hf_item

    def test_no_match_returns_none(self, all_items: list[Item], item_by_url: dict[str, Item]):
        result = _match_tool_call_to_item(
            "fetch_url",
            {"url": "https://example.com/no-such-page"},
            item_by_url,
            all_items,
        )
        assert result is None

    def test_get_github_repo_no_match(self, all_items: list[Item], item_by_url: dict[str, Item]):
        result = _match_tool_call_to_item(
            "get_github_repo",
            {"owner": "unknown", "repo": "nonexistent"},
            item_by_url,
            all_items,
        )
        assert result is None

    def test_search_hf_models_no_match(self, all_items: list[Item], item_by_url: dict[str, Item]):
        result = _match_tool_call_to_item(
            "search_hf_models",
            {"query": "totally unrelated query xyz"},
            item_by_url,
            all_items,
        )
        assert result is None

    def test_unknown_tool_returns_none(self, all_items: list[Item], item_by_url: dict[str, Item]):
        result = _match_tool_call_to_item(
            "unknown_tool",
            {"arg": "value"},
            item_by_url,
            all_items,
        )
        assert result is None


# ---------------------------------------------------------------------------
# _format_items_for_prompt
# ---------------------------------------------------------------------------


class TestFormatItemsForPrompt:
    """Tests for _format_items_for_prompt()."""

    def test_hf_item_format(self, hf_item: Item):
        result = _format_items_for_prompt([hf_item])
        assert "[HF-1]" in result
        assert "Paper A" in result
        assert "upvotes: 50" in result
        assert "auteurs: Author1" in result
        assert "abstract: Abstract text for paper A" in result
        assert hf_item.url in result

    def test_gh_item_format(self, gh_item: Item):
        result = _format_items_for_prompt([gh_item])
        assert "[GH-1]" in result
        assert "owner/repo" in result
        assert "description: Desc of the repo" in result
        assert "stars/jour: 100" in result
        assert "langage: Python" in result
        assert gh_item.url in result

    def test_simon_item_format(self, simon_item: Item):
        result = _format_items_for_prompt([simon_item])
        assert "[SW-1]" in result
        assert "Post A" in result
        assert "tags: ai, llm" in result
        assert "extrait: Snippet text" in result
        assert simon_item.url in result

    def test_mixed_items_indexing(self, hf_item: Item, gh_item: Item, simon_item: Item):
        """Each item gets a sequential index regardless of source."""
        result = _format_items_for_prompt([hf_item, gh_item, simon_item])
        assert "[HF-1]" in result
        assert "[GH-2]" in result
        assert "[SW-3]" in result

    def test_empty_list(self):
        result = _format_items_for_prompt([])
        assert result == ""

    def test_simon_snippet_truncated_at_300(self):
        """content_snippet[:300] is used in the format."""
        long_snippet = "A" * 500
        item = Item(
            title="Long Post",
            url="https://simonwillison.net/post/long",
            source=SourceType.SIMON,
            tags=["test"],
            content_snippet=long_snippet,
        )
        result = _format_items_for_prompt([item])
        # The extrait should be exactly 300 chars of 'A'
        assert f"extrait: {'A' * 300}" in result
        assert "A" * 301 not in result


# ---------------------------------------------------------------------------
# fetch_sources
# ---------------------------------------------------------------------------


class TestFetchSources:
    """Tests for fetch_sources() node."""

    async def test_all_succeed(self, hf_item: Item, gh_item: Item, simon_item: Item):
        with (
            patch(
                "src.agent.nodes.fetch_huggingface_papers",
                new_callable=AsyncMock,
                return_value=[hf_item],
            ),
            patch(
                "src.agent.nodes.fetch_github_trending",
                new_callable=AsyncMock,
                return_value=[gh_item],
            ),
            patch(
                "src.agent.nodes.fetch_simon_willison",
                new_callable=AsyncMock,
                return_value=[simon_item],
            ),
        ):
            state: AgentState = {}
            result = await fetch_sources(state)

        assert result["hf_items"] == [hf_item]
        assert result["gh_items_raw"] == [gh_item]
        assert result["simon_items"] == [simon_item]
        assert result["sources_status"]["huggingface"] == "ok"
        assert result["sources_status"]["github"] == "ok"
        assert result["sources_status"]["simon"] == "ok"

    async def test_one_source_fails(self, hf_item: Item, simon_item: Item):
        with (
            patch(
                "src.agent.nodes.fetch_huggingface_papers",
                new_callable=AsyncMock,
                return_value=[hf_item],
            ),
            patch(
                "src.agent.nodes.fetch_github_trending",
                new_callable=AsyncMock,
                side_effect=RuntimeError("GitHub is down"),
            ),
            patch(
                "src.agent.nodes.fetch_simon_willison",
                new_callable=AsyncMock,
                return_value=[simon_item],
            ),
        ):
            state: AgentState = {}
            result = await fetch_sources(state)

        assert result["hf_items"] == [hf_item]
        assert result["gh_items_raw"] == []
        assert result["simon_items"] == [simon_item]
        assert result["sources_status"]["huggingface"] == "ok"
        assert "error" in result["sources_status"]["github"]
        assert result["sources_status"]["simon"] == "ok"

    async def test_all_sources_fail(self):
        with (
            patch(
                "src.agent.nodes.fetch_huggingface_papers",
                new_callable=AsyncMock,
                side_effect=RuntimeError("HF down"),
            ),
            patch(
                "src.agent.nodes.fetch_github_trending",
                new_callable=AsyncMock,
                side_effect=RuntimeError("GH down"),
            ),
            patch(
                "src.agent.nodes.fetch_simon_willison",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Simon down"),
            ),
        ):
            state: AgentState = {}
            result = await fetch_sources(state)

        assert result["hf_items"] == []
        assert result["gh_items_raw"] == []
        assert result["simon_items"] == []
        assert "error" in result["sources_status"]["huggingface"]
        assert "error" in result["sources_status"]["github"]
        assert "error" in result["sources_status"]["simon"]


# ---------------------------------------------------------------------------
# filter_github
# ---------------------------------------------------------------------------


class TestFilterGithub:
    """Tests for filter_github() node."""

    def _make_gh_items(self) -> list[Item]:
        """Create a list of GitHub items for testing the filter."""
        return [
            Item(
                title="ai-org/llm-framework",
                url="https://github.com/ai-org/llm-framework",
                source=SourceType.GITHUB,
                description="A framework for LLMs",
                stars_today=200,
                language="Python",
                repo_owner="ai-org",
                repo_name="llm-framework",
            ),
            Item(
                title="web-dev/css-lib",
                url="https://github.com/web-dev/css-lib",
                source=SourceType.GITHUB,
                description="A CSS library",
                stars_today=150,
                language="CSS",
                repo_owner="web-dev",
                repo_name="css-lib",
            ),
            Item(
                title="ml-team/diffusion-models",
                url="https://github.com/ml-team/diffusion-models",
                source=SourceType.GITHUB,
                description="Diffusion model training toolkit",
                stars_today=180,
                language="Python",
                repo_owner="ml-team",
                repo_name="diffusion-models",
            ),
        ]

    async def test_valid_json_response_filters_correctly(self):
        gh_items = self._make_gh_items()
        llm_response_json = json.dumps(
            [
                {"repo": "ai-org/llm-framework", "is_ai": True},
                {"repo": "web-dev/css-lib", "is_ai": False},
                {"repo": "ml-team/diffusion-models", "is_ai": True},
            ]
        )
        mock_llm_response = MagicMock()
        mock_llm_response.content = llm_response_json

        mock_llm = MagicMock()
        mock_invoke = AsyncMock(return_value=mock_llm_response)

        state: AgentState = {"gh_items_raw": gh_items}

        with (
            patch("src.agent.nodes._get_llm", return_value=mock_llm),
            patch("src.agent.nodes._llm_invoke_with_retry", mock_invoke),
        ):
            result = await filter_github(state)

        filtered = result["gh_items_filtered"]
        assert len(filtered) == 2
        # Sorted by stars_today descending: llm-framework (200) > diffusion-models (180)
        assert filtered[0].repo_name == "llm-framework"
        assert filtered[1].repo_name == "diffusion-models"
        # css-lib should not be present
        assert all(item.repo_name != "css-lib" for item in filtered)

    async def test_markdown_wrapped_json_response(self):
        """LLM sometimes wraps JSON in markdown code fences."""
        gh_items = self._make_gh_items()
        wrapped_json = (
            "```json\n"
            + json.dumps(
                [
                    {"repo": "ai-org/llm-framework", "is_ai": True},
                    {"repo": "web-dev/css-lib", "is_ai": False},
                    {"repo": "ml-team/diffusion-models", "is_ai": False},
                ]
            )
            + "\n```"
        )
        mock_llm_response = MagicMock()
        mock_llm_response.content = wrapped_json

        mock_llm = MagicMock()
        mock_invoke = AsyncMock(return_value=mock_llm_response)

        state: AgentState = {"gh_items_raw": gh_items}

        with (
            patch("src.agent.nodes._get_llm", return_value=mock_llm),
            patch("src.agent.nodes._llm_invoke_with_retry", mock_invoke),
        ):
            result = await filter_github(state)

        filtered = result["gh_items_filtered"]
        assert len(filtered) == 1
        assert filtered[0].repo_name == "llm-framework"

    async def test_invalid_json_returns_empty_list_bug5_fix(self):
        """BUG-5 fix: invalid JSON should return empty list, not crash."""
        gh_items = self._make_gh_items()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "This is not valid JSON at all!"

        mock_llm = MagicMock()
        mock_invoke = AsyncMock(return_value=mock_llm_response)

        state: AgentState = {"gh_items_raw": gh_items}

        with (
            patch("src.agent.nodes._get_llm", return_value=mock_llm),
            patch("src.agent.nodes._llm_invoke_with_retry", mock_invoke),
        ):
            result = await filter_github(state)

        assert result["gh_items_filtered"] == []

    async def test_empty_gh_items_returns_empty(self):
        state: AgentState = {"gh_items_raw": []}
        result = await filter_github(state)
        assert result["gh_items_filtered"] == []

    async def test_top_3_limit(self):
        """Even if more than 3 repos are AI-related, only top 3 are returned."""
        items = [
            Item(
                title=f"org/repo-{i}",
                url=f"https://github.com/org/repo-{i}",
                source=SourceType.GITHUB,
                description=f"AI tool {i}",
                stars_today=100 + i * 10,
                language="Python",
                repo_owner="org",
                repo_name=f"repo-{i}",
            )
            for i in range(5)
        ]
        llm_json = json.dumps([{"repo": f"org/repo-{i}", "is_ai": True} for i in range(5)])
        mock_llm_response = MagicMock()
        mock_llm_response.content = llm_json

        mock_llm = MagicMock()
        mock_invoke = AsyncMock(return_value=mock_llm_response)

        state: AgentState = {"gh_items_raw": items}

        with (
            patch("src.agent.nodes._get_llm", return_value=mock_llm),
            patch("src.agent.nodes._llm_invoke_with_retry", mock_invoke),
        ):
            result = await filter_github(state)

        assert len(result["gh_items_filtered"]) == 3
        # Top 3 by stars_today: repo-4(140), repo-3(130), repo-2(120)
        assert result["gh_items_filtered"][0].repo_name == "repo-4"
        assert result["gh_items_filtered"][1].repo_name == "repo-3"
        assert result["gh_items_filtered"][2].repo_name == "repo-2"


# ---------------------------------------------------------------------------
# combine_items
# ---------------------------------------------------------------------------


class TestCombineItems:
    """Tests for combine_items() node."""

    async def test_combines_all_three_sources(self, hf_item: Item, gh_item: Item, simon_item: Item):
        state: AgentState = {
            "hf_items": [hf_item],
            "gh_items_filtered": [gh_item],
            "simon_items": [simon_item],
        }
        result = await combine_items(state)
        assert len(result["items_to_enrich"]) == 3
        assert hf_item in result["items_to_enrich"]
        assert gh_item in result["items_to_enrich"]
        assert simon_item in result["items_to_enrich"]

    async def test_handles_missing_sources(self, hf_item: Item):
        """Empty or missing source lists result in partial combination."""
        state: AgentState = {
            "hf_items": [hf_item],
            "gh_items_filtered": [],
            "simon_items": [],
        }
        result = await combine_items(state)
        assert len(result["items_to_enrich"]) == 1
        assert result["items_to_enrich"][0] is hf_item

    async def test_all_empty(self):
        state: AgentState = {
            "hf_items": [],
            "gh_items_filtered": [],
            "simon_items": [],
        }
        result = await combine_items(state)
        assert result["items_to_enrich"] == []

    async def test_missing_keys_default_to_empty(self):
        """State with no source keys should still work (defaults via .get)."""
        state: AgentState = {}
        result = await combine_items(state)
        assert result["items_to_enrich"] == []


# ---------------------------------------------------------------------------
# enrich_and_brief
# ---------------------------------------------------------------------------


class TestEnrichAndBrief:
    """Tests for enrich_and_brief() node."""

    async def test_empty_items_returns_fallback(self):
        state: AgentState = {"items_to_enrich": []}
        result = await enrich_and_brief(state)
        assert "No items available today" in result["briefing_markdown"]
        assert result["enrichment_logs"] == []

    async def test_with_items_no_tool_calls(self, hf_item: Item, gh_item: Item, simon_item: Item):
        """LLM returns text directly without tool calls."""
        mock_response = MagicMock()
        mock_response.content = "# Veille IA\n\nBriefing content here."
        mock_response.tool_calls = []
        mock_response.usage_metadata = {"total_tokens": 500}

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        mock_invoke = AsyncMock(return_value=mock_response)

        state: AgentState = {
            "items_to_enrich": [hf_item, gh_item, simon_item],
            "sources_status": {"huggingface": "ok", "github": "ok", "simon": "ok"},
        }

        with (
            patch("src.agent.nodes._get_llm", return_value=mock_llm),
            patch("src.agent.nodes._llm_invoke_with_retry", mock_invoke),
        ):
            result = await enrich_and_brief(state)

        assert result["briefing_markdown"] == "# Veille IA\n\nBriefing content here."
        assert result["enrichment_logs"] == []
        assert result["llm_calls"] == 1
        assert result["total_tokens"] == 500

    async def test_source_warning_text(self, hf_item: Item):
        """When a source is unavailable, the warning text is included in messages."""
        mock_response = MagicMock()
        mock_response.content = "# Veille IA\n\nPartial briefing."
        mock_response.tool_calls = []
        mock_response.usage_metadata = {"total_tokens": 100}

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        mock_invoke = AsyncMock(return_value=mock_response)

        state: AgentState = {
            "items_to_enrich": [hf_item],
            "sources_status": {
                "huggingface": "ok",
                "github": "error: connection timeout",
                "simon": "ok",
            },
        }

        with (
            patch("src.agent.nodes._get_llm", return_value=mock_llm),
            patch("src.agent.nodes._llm_invoke_with_retry", mock_invoke),
        ):
            result = await enrich_and_brief(state)

        # Verify the warning was passed to the LLM in the messages
        call_args = mock_invoke.call_args_list[0]
        messages = call_args[0][1]  # second positional arg is the messages list
        user_message_content = messages[1].content
        assert "Source indisponible" in user_message_content
        assert "github" in user_message_content
        assert result["briefing_markdown"] == "# Veille IA\n\nPartial briefing."

    async def test_content_as_list_blocks(self, hf_item: Item):
        """Handle case where LLM returns content as a list of blocks."""
        mock_response = MagicMock()
        mock_response.content = [
            {"type": "text", "text": "# Veille IA"},
            {"type": "text", "text": "\n\nBlock 2"},
        ]
        mock_response.tool_calls = []
        mock_response.usage_metadata = {"total_tokens": 200}

        mock_llm = MagicMock()
        mock_llm_with_tools = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools

        mock_invoke = AsyncMock(return_value=mock_response)

        state: AgentState = {
            "items_to_enrich": [hf_item],
            "sources_status": {"huggingface": "ok"},
        }

        with (
            patch("src.agent.nodes._get_llm", return_value=mock_llm),
            patch("src.agent.nodes._llm_invoke_with_retry", mock_invoke),
        ):
            result = await enrich_and_brief(state)

        assert "# Veille IA" in result["briefing_markdown"]
        assert "Block 2" in result["briefing_markdown"]


# ---------------------------------------------------------------------------
# _llm_invoke_with_retry
# ---------------------------------------------------------------------------


class TestLlmInvokeWithRetry:
    """Tests for _llm_invoke_with_retry()."""

    async def test_success_on_first_try(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value="response")
        result = await _llm_invoke_with_retry(mock_llm, ["msg"], max_retries=1)
        assert result == "response"
        assert mock_llm.ainvoke.call_count == 1

    async def test_retry_on_failure_then_success(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[RuntimeError("temporary"), "success"])
        result = await _llm_invoke_with_retry(mock_llm, ["msg"], max_retries=1)
        assert result == "success"
        assert mock_llm.ainvoke.call_count == 2

    async def test_failure_after_all_retries_raises(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[RuntimeError("fail1"), RuntimeError("fail2")])
        with pytest.raises(RuntimeError, match="fail2"):
            await _llm_invoke_with_retry(mock_llm, ["msg"], max_retries=1)
        assert mock_llm.ainvoke.call_count == 2

    async def test_no_retries(self):
        """With max_retries=0, failure on first try raises immediately."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("instant fail"))
        with pytest.raises(RuntimeError, match="instant fail"):
            await _llm_invoke_with_retry(mock_llm, ["msg"], max_retries=0)
        assert mock_llm.ainvoke.call_count == 1

    async def test_success_with_zero_retries(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value="ok")
        result = await _llm_invoke_with_retry(mock_llm, ["msg"], max_retries=0)
        assert result == "ok"
        assert mock_llm.ainvoke.call_count == 1
