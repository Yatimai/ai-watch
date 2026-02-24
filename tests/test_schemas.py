"""Tests for src/models/schemas.py — Pydantic models."""

from datetime import UTC, datetime

from src.models.schemas import BriefingResult, EnrichmentLog, Item, SourceType


class TestItem:
    """Tests for the Item model."""

    def test_huggingface_item_all_fields(self):
        item = Item(
            title="Attention Is All You Need v2",
            url="https://huggingface.co/papers/1234",
            source=SourceType.HUGGINGFACE,
            abstract="We propose a new transformer architecture.",
            upvotes=42,
            arxiv_id="2401.12345",
            authors="Alice, Bob",
        )
        assert item.title == "Attention Is All You Need v2"
        assert item.source == SourceType.HUGGINGFACE
        assert item.source == "huggingface"
        assert item.abstract == "We propose a new transformer architecture."
        assert item.upvotes == 42
        assert item.arxiv_id == "2401.12345"
        assert item.authors == "Alice, Bob"

    def test_github_item_all_fields(self):
        item = Item(
            title="pytorch/pytorch",
            url="https://github.com/pytorch/pytorch",
            source=SourceType.GITHUB,
            description="Tensor computation with GPU acceleration",
            stars_today=150,
            language="Python",
            repo_owner="pytorch",
            repo_name="pytorch",
        )
        assert item.source == SourceType.GITHUB
        assert item.description == "Tensor computation with GPU acceleration"
        assert item.stars_today == 150
        assert item.language == "Python"
        assert item.repo_owner == "pytorch"
        assert item.repo_name == "pytorch"

    def test_simon_item_all_fields(self):
        now = datetime.now(tz=UTC)
        item = Item(
            title="What I learned about LLMs this week",
            url="https://simonwillison.net/2025/post/",
            source=SourceType.SIMON,
            tags=["llm", "ai", "python"],
            content_snippet="This week I explored...",
            published_at=now,
        )
        assert item.source == SourceType.SIMON
        assert item.tags == ["llm", "ai", "python"]
        assert item.content_snippet == "This week I explored..."
        assert item.published_at == now

    def test_default_values(self):
        item = Item(
            title="Minimal Item",
            url="https://example.com",
            source=SourceType.HUGGINGFACE,
        )
        assert item.abstract == ""
        assert item.upvotes == 0
        assert item.arxiv_id == ""
        assert item.authors == ""
        assert item.description == ""
        assert item.stars_today == 0
        assert item.language == ""
        assert item.repo_owner == ""
        assert item.repo_name == ""
        assert item.tags == []
        assert item.content_snippet == ""
        assert item.published_at is None


class TestEnrichmentLog:
    """Tests for the EnrichmentLog model."""

    def test_all_fields(self):
        log = EnrichmentLog(
            item_title="Some Paper",
            source=SourceType.HUGGINGFACE,
            tools_called=["fetch_url", "search_hf_models"],
            reason="Paper mentions a new model, fetched details.",
        )
        assert log.item_title == "Some Paper"
        assert log.source == SourceType.HUGGINGFACE
        assert log.tools_called == ["fetch_url", "search_hf_models"]
        assert log.reason == "Paper mentions a new model, fetched details."

    def test_default_empty_lists(self):
        log = EnrichmentLog(
            item_title="Another Paper",
            source=SourceType.GITHUB,
        )
        assert log.tools_called == []
        assert log.reason == ""


class TestBriefingResult:
    """Tests for the BriefingResult model."""

    def test_all_fields(self):
        item = Item(
            title="Test Item",
            url="https://example.com",
            source=SourceType.SIMON,
        )
        enrichment = EnrichmentLog(
            item_title="Test Item",
            source=SourceType.SIMON,
            tools_called=["fetch_url"],
            reason="Needed more context",
        )
        result = BriefingResult(
            date="2025-02-24",
            briefing_markdown="# AI Watch\n\nToday's briefing...",
            items=[item],
            enrichment_logs=[enrichment],
            sources_status={"huggingface": "ok", "github": "ok", "simon": "ok"},
            llm_calls=5,
            total_tokens=12000,
            duration_seconds=45.3,
        )
        assert result.date == "2025-02-24"
        assert "AI Watch" in result.briefing_markdown
        assert len(result.items) == 1
        assert len(result.enrichment_logs) == 1
        assert result.sources_status["huggingface"] == "ok"
        assert result.llm_calls == 5
        assert result.total_tokens == 12000
        assert result.duration_seconds == 45.3

    def test_default_optional_fields(self):
        result = BriefingResult(
            date="2025-02-24",
            briefing_markdown="# Briefing",
            items=[],
        )
        assert result.enrichment_logs == []
        assert result.sources_status == {}
        assert result.llm_calls == 0
        assert result.total_tokens == 0
        assert result.duration_seconds == 0.0
