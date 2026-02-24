"""LangGraph agent state."""

from typing import TypedDict

from src.models.schemas import EnrichmentLog, Item


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # Fetch stage
    hf_items: list[Item]
    gh_items_raw: list[Item]  # all trending repos (before AI filter)
    gh_items_filtered: list[Item]  # after AI filter, top 3
    simon_items: list[Item]

    # Combined 9 items for enrichment
    items_to_enrich: list[Item]

    # Enrichment + briefing
    enrichment_logs: list[EnrichmentLog]
    briefing_markdown: str

    # Metadata
    today: str  # YYYY-MM-DD, injected by cli.py
    sources_status: dict[str, str]  # {"huggingface": "ok", "github": "error: ..."}
    llm_calls: int
    total_tokens: int
    error: str | None
