"""Comprehensive tests for the 3 source fetchers: HuggingFace, GitHub, Simon."""

import asyncio
from datetime import datetime
from time import struct_time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.models.schemas import Item, SourceType
from src.sources.github import fetch_github_trending
from src.sources.huggingface import fetch_huggingface_papers
from src.sources.simon import fetch_simon_willison

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HUGGINGFACE_API_RESPONSE = [
    {
        "title": "Paper A",
        "paper": {
            "id": "2511.21631",
            "title": "Paper A",
            "summary": "Abstract A about transformers",
            "upvotes": 84,
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
        },
    },
    {
        "title": "Paper B",
        "paper": {
            "id": "2512.03043",
            "title": "Paper B",
            "summary": "Abstract B about diffusion models",
            "upvotes": 19,
            "authors": [{"name": "Charlie"}],
        },
    },
    {
        "title": "Paper C",
        "paper": {
            "id": "2512.99999",
            "title": "Paper C",
            "summary": "Abstract C about RLHF",
            "upvotes": 120,
            "authors": [{"name": "Diana"}, {"name": "Eve"}, {"name": "Frank"}],
        },
    },
    {
        "title": "Paper D",
        "paper": {
            "id": "2512.11111",
            "title": "Paper D",
            "summary": "Abstract D about vision language models",
            "upvotes": 42,
            "authors": [{"name": "Grace"}],
        },
    },
    {
        "title": "Paper E",
        "paper": {
            "id": "2512.22222",
            "title": "Paper E",
            "summary": "Abstract E about reasoning",
            "upvotes": 7,
            "authors": [{"name": "Hank"}],
        },
    },
]

GITHUB_TRENDING_HTML = """
<html>
<body>
<article class="Box-row">
  <h2><a href="/openai/tiktoken">tiktoken</a></h2>
  <p>Fast BPE tokeniser for use with OpenAI models</p>
  <span itemprop="programmingLanguage">Python</span>
  <span class="d-inline-block float-sm-right">456 stars today</span>
</article>
<article class="Box-row">
  <h2><a href="/huggingface/transformers">transformers</a></h2>
  <p>State-of-the-art ML for PyTorch, TF, JAX</p>
  <span itemprop="programmingLanguage">Python</span>
  <span class="d-inline-block float-sm-right">1,234 stars today</span>
</article>
<article class="Box-row">
  <h2><a href="/rust-lang/rust">rust</a></h2>
  <p>The Rust programming language</p>
  <span itemprop="programmingLanguage">Rust</span>
  <span class="d-inline-block float-sm-right">89 stars today</span>
</article>
</body>
</html>
"""

GITHUB_EMPTY_HTML = """
<html><body><div class="Box"></div></body></html>
"""

GITHUB_NO_DESCRIPTION_HTML = """
<html>
<body>
<article class="Box-row">
  <h2><a href="/owner/minimal-repo">minimal-repo</a></h2>
  <span class="d-inline-block float-sm-right">10 stars today</span>
</article>
</body>
</html>
"""


def _make_feed_entry(
    title: str,
    link: str,
    summary: str = "",
    tags: list[dict] | None = None,
    published_parsed: struct_time | None = None,
    content: list[dict] | None = None,
):
    """Build a mock feedparser entry with attribute-style access."""
    entry = MagicMock()
    entry.get = lambda k, default="": {
        "title": title,
        "link": link,
        "tags": tags or [],
    }.get(k, default)
    entry.summary = summary
    entry.content = content or []
    entry.published_parsed = published_parsed
    entry.tags = tags or []
    return entry


# ---------------------------------------------------------------------------
# HuggingFace tests
# ---------------------------------------------------------------------------


class TestFetchHuggingfacePapers:
    """Tests for src.sources.huggingface.fetch_huggingface_papers."""

    async def test_returns_items_sorted_by_upvotes_descending(self):
        mock_response = MagicMock()
        mock_response.json.return_value = HUGGINGFACE_API_RESPONSE
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.huggingface.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_huggingface_papers(top_k=5)

        upvotes = [item.upvotes for item in items]
        assert upvotes == sorted(upvotes, reverse=True)
        # The expected order: 120, 84, 42, 19, 7
        assert upvotes == [120, 84, 42, 19, 7]

    async def test_top_k_limits_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = HUGGINGFACE_API_RESPONSE
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.huggingface.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_huggingface_papers(top_k=3)

        assert len(items) == 3
        # Should be the top 3 by upvotes
        assert items[0].upvotes == 120
        assert items[1].upvotes == 84
        assert items[2].upvotes == 42

    async def test_item_fields_correctly_populated(self):
        # Use a single paper for precise field checking
        single_paper = [
            {
                "title": "My Great Paper",
                "paper": {
                    "id": "2511.21631",
                    "title": "My Great Paper",
                    "summary": "This paper introduces a novel approach",
                    "upvotes": 84,
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                },
            }
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = single_paper
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.huggingface.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_huggingface_papers(top_k=3)

        assert len(items) == 1
        item = items[0]
        assert isinstance(item, Item)
        assert item.title == "My Great Paper"
        assert item.url == "https://huggingface.co/papers/2511.21631"
        assert item.source == SourceType.HUGGINGFACE
        assert item.abstract == "This paper introduces a novel approach"
        assert item.upvotes == 84
        assert item.arxiv_id == "2511.21631"
        assert item.authors == "Alice, Bob"

    async def test_authors_truncated_to_five(self):
        """Authors list is sliced to first 5 entries."""
        paper_with_many_authors = [
            {
                "title": "Collab Paper",
                "paper": {
                    "id": "2512.00000",
                    "title": "Collab Paper",
                    "summary": "Big collaboration",
                    "upvotes": 10,
                    "authors": [
                        {"name": "A1"},
                        {"name": "A2"},
                        {"name": "A3"},
                        {"name": "A4"},
                        {"name": "A5"},
                        {"name": "A6"},
                        {"name": "A7"},
                    ],
                },
            }
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = paper_with_many_authors
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.huggingface.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_huggingface_papers(top_k=1)

        # Only first 5 authors joined
        assert items[0].authors == "A1, A2, A3, A4, A5"

    async def test_http_error_raises_exception(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.sources.huggingface.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await fetch_huggingface_papers()

    async def test_empty_api_response_returns_empty_list(self):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.huggingface.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_huggingface_papers(top_k=3)

        assert items == []

    async def test_missing_paper_fields_use_defaults(self):
        """Papers with missing optional fields should still produce valid Items."""
        sparse_paper = [
            {
                "title": "Sparse Paper",
                "paper": {
                    "id": "2512.55555",
                },
            }
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = sparse_paper
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.huggingface.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_huggingface_papers(top_k=1)

        item = items[0]
        assert item.title == "Sparse Paper"
        assert item.abstract == ""
        assert item.upvotes == 0
        assert item.authors == ""


# ---------------------------------------------------------------------------
# GitHub Trending tests
# ---------------------------------------------------------------------------


class TestFetchGithubTrending:
    """Tests for src.sources.github.fetch_github_trending."""

    async def test_parses_articles_correctly(self):
        mock_response = MagicMock()
        mock_response.text = GITHUB_TRENDING_HTML
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.github.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_github_trending()

        assert len(items) == 3
        assert all(isinstance(i, Item) for i in items)
        assert all(i.source == SourceType.GITHUB for i in items)

    async def test_item_fields_populated(self):
        mock_response = MagicMock()
        mock_response.text = GITHUB_TRENDING_HTML
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.github.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_github_trending()

        # First article: openai/tiktoken
        tiktoken = items[0]
        assert tiktoken.title == "openai/tiktoken"
        assert tiktoken.url == "https://github.com/openai/tiktoken"
        assert tiktoken.repo_owner == "openai"
        assert tiktoken.repo_name == "tiktoken"
        assert tiktoken.description == "Fast BPE tokeniser for use with OpenAI models"
        assert tiktoken.language == "Python"
        assert tiktoken.stars_today == 456

    async def test_stars_today_parsed_from_regex(self):
        mock_response = MagicMock()
        mock_response.text = GITHUB_TRENDING_HTML
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.github.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_github_trending()

        assert items[0].stars_today == 456
        # Comma-separated number: "1,234 stars today"
        assert items[1].stars_today == 1234
        assert items[2].stars_today == 89

    async def test_empty_page_raises_runtime_error(self):
        """BUG-10 fix: an empty trending page must raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.text = GITHUB_EMPTY_HTML
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.sources.github.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(RuntimeError, match="0 repos"),
        ):
            await fetch_github_trending()

    async def test_http_error_raises_exception(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.sources.github.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await fetch_github_trending()

    async def test_missing_description_and_language(self):
        """Articles without <p> or language span should use empty defaults."""
        mock_response = MagicMock()
        mock_response.text = GITHUB_NO_DESCRIPTION_HTML
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.github.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_github_trending()

        assert len(items) == 1
        assert items[0].description == ""
        assert items[0].language == ""
        assert items[0].stars_today == 10

    async def test_skips_articles_without_h2_link(self):
        """Articles without a valid h2 > a element should be skipped."""
        html = """
        <html><body>
        <article class="Box-row">
          <h2></h2>
          <p>No link here</p>
        </article>
        <article class="Box-row">
          <h2><a href="/valid/repo">repo</a></h2>
          <p>Valid repo</p>
          <span class="d-inline-block float-sm-right">50 stars today</span>
        </article>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.github.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_github_trending()

        assert len(items) == 1
        assert items[0].title == "valid/repo"

    async def test_skips_malformed_href_paths(self):
        """href with wrong number of path segments (not owner/repo) should be skipped."""
        html = """
        <html><body>
        <article class="Box-row">
          <h2><a href="/only-one-segment">bad</a></h2>
          <p>Bad path</p>
        </article>
        <article class="Box-row">
          <h2><a href="/a/b/c/too/many">bad</a></h2>
          <p>Too many segments</p>
        </article>
        <article class="Box-row">
          <h2><a href="/good/repo">repo</a></h2>
          <p>Good repo</p>
          <span class="d-inline-block float-sm-right">5 stars today</span>
        </article>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.sources.github.httpx.AsyncClient", return_value=mock_client):
            items = await fetch_github_trending()

        assert len(items) == 1
        assert items[0].title == "good/repo"


# ---------------------------------------------------------------------------
# Simon Willison tests
# ---------------------------------------------------------------------------


class TestFetchSimonWillison:
    """Tests for src.sources.simon.fetch_simon_willison."""

    def _make_feed(self, entries, bozo=False, bozo_exception=None):
        """Build a mock feedparser result."""
        feed = MagicMock()
        feed.bozo = bozo
        feed.bozo_exception = bozo_exception
        feed.entries = entries
        return feed

    async def test_items_correctly_parsed(self):
        published_time = struct_time((2026, 2, 20, 10, 0, 0, 3, 51, 0))
        entries = [
            _make_feed_entry(
                title="LLM round-up",
                link="https://simonwillison.net/2026/Feb/20/llm-roundup/",
                summary="A long summary about LLMs and tools and things" * 5,
                tags=[{"term": "llm"}, {"term": "ai"}],
                published_parsed=published_time,
            ),
            _make_feed_entry(
                title="Weeknotes",
                link="https://simonwillison.net/2026/Feb/19/weeknotes/",
                summary="This week I worked on datasette",
                tags=[{"term": "datasette"}],
                published_parsed=struct_time((2026, 2, 19, 8, 0, 0, 2, 50, 0)),
            ),
        ]
        feed = self._make_feed(entries)

        with patch("src.sources.simon.feedparser") as mock_fp:
            mock_fp.parse.return_value = feed
            items = await fetch_simon_willison(top_k=5)

        assert len(items) == 2
        item = items[0]
        assert isinstance(item, Item)
        assert item.title == "LLM round-up"
        assert item.url == "https://simonwillison.net/2026/Feb/20/llm-roundup/"
        assert item.source == SourceType.SIMON
        assert item.tags == ["llm", "ai"]
        assert item.published_at == datetime(2026, 2, 20, 10, 0, 0)
        assert len(item.content_snippet) <= 500

    async def test_top_k_limits_results(self):
        entries = [
            _make_feed_entry(title=f"Post {i}", link=f"https://example.com/{i}") for i in range(10)
        ]
        feed = self._make_feed(entries)

        with patch("src.sources.simon.feedparser") as mock_fp:
            mock_fp.parse.return_value = feed
            items = await fetch_simon_willison(top_k=3)

        assert len(items) == 3

    async def test_bozo_feed_with_no_entries_raises_runtime_error(self):
        feed = self._make_feed(entries=[], bozo=True, bozo_exception=Exception("XML error"))

        with patch("src.sources.simon.feedparser") as mock_fp:
            mock_fp.parse.return_value = feed
            with pytest.raises(RuntimeError, match="Failed to parse Simon Willison RSS"):
                await fetch_simon_willison()

    async def test_bozo_feed_with_entries_does_not_raise(self):
        """A bozo feed that still has entries should be processed normally."""
        entries = [
            _make_feed_entry(title="Still works", link="https://example.com/ok"),
        ]
        feed = self._make_feed(entries, bozo=True, bozo_exception=Exception("Minor issue"))

        with patch("src.sources.simon.feedparser") as mock_fp:
            mock_fp.parse.return_value = feed
            items = await fetch_simon_willison(top_k=3)

        assert len(items) == 1
        assert items[0].title == "Still works"

    async def test_timeout_raises_asyncio_timeout_error(self):
        with (
            patch("src.sources.simon.feedparser"),
            patch(
                "src.sources.simon.asyncio.wait_for",
                side_effect=asyncio.TimeoutError,
            ),
            pytest.raises(asyncio.TimeoutError),
        ):
            await fetch_simon_willison(timeout=1)

    async def test_content_fallback_to_content_field(self):
        """When summary is empty, content_snippet should come from content[0].value."""
        entry = _make_feed_entry(
            title="Content fallback",
            link="https://example.com/fallback",
            summary="",
            content=[{"value": "This is the content body from Atom feed"}],
        )
        # Make hasattr checks work: summary is empty string, content is a list
        entry.summary = ""
        entry.content = [{"value": "This is the content body from Atom feed"}]

        feed = self._make_feed([entry])

        with patch("src.sources.simon.feedparser") as mock_fp:
            mock_fp.parse.return_value = feed
            items = await fetch_simon_willison(top_k=1)

        assert items[0].content_snippet == "This is the content body from Atom feed"

    async def test_no_published_date_sets_none(self):
        """Entry without published_parsed should have published_at=None."""
        entry = _make_feed_entry(
            title="No date",
            link="https://example.com/nodate",
            summary="No date info",
            published_parsed=None,
        )

        feed = self._make_feed([entry])

        with patch("src.sources.simon.feedparser") as mock_fp:
            mock_fp.parse.return_value = feed
            items = await fetch_simon_willison(top_k=1)

        assert items[0].published_at is None

    async def test_empty_tags_produces_empty_list(self):
        """Entry without tags produces an empty tags list."""
        entry = _make_feed_entry(
            title="No tags",
            link="https://example.com/notags",
            summary="Post without tags",
            tags=[],
        )

        feed = self._make_feed([entry])

        with patch("src.sources.simon.feedparser") as mock_fp:
            mock_fp.parse.return_value = feed
            items = await fetch_simon_willison(top_k=1)

        assert items[0].tags == []

    async def test_content_snippet_truncated_to_500_chars(self):
        """Summaries longer than 500 chars should be truncated."""
        long_summary = "x" * 1000
        entry = _make_feed_entry(
            title="Long post",
            link="https://example.com/long",
            summary=long_summary,
        )

        feed = self._make_feed([entry])

        with patch("src.sources.simon.feedparser") as mock_fp:
            mock_fp.parse.return_value = feed
            items = await fetch_simon_willison(top_k=1)

        assert len(items[0].content_snippet) == 500
