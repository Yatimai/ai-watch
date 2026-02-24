"""Pydantic models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    SIMON = "simon"


class Item(BaseModel):
    """Un item brut récupéré d'une source."""

    title: str
    url: str
    source: SourceType
    # HuggingFace
    abstract: str = ""
    upvotes: int = 0
    arxiv_id: str = ""
    authors: str = ""
    # GitHub
    description: str = ""
    stars_today: int = 0
    language: str = ""
    repo_owner: str = ""
    repo_name: str = ""
    # Simon
    tags: list[str] = Field(default_factory=list)
    content_snippet: str = ""
    published_at: datetime | None = None


class EnrichmentLog(BaseModel):
    """Log d'une décision d'enrichissement."""

    item_title: str
    source: SourceType
    tools_called: list[str] = Field(default_factory=list)
    reason: str = ""


class BriefingResult(BaseModel):
    """Résultat complet d'un run."""

    date: str
    briefing_markdown: str
    items: list[Item]
    enrichment_logs: list[EnrichmentLog] = Field(default_factory=list)
    sources_status: dict[str, str] = Field(default_factory=dict)
    llm_calls: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
