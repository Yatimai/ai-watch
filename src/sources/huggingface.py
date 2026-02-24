"""Fetch HuggingFace Daily Papers API."""

import logging

import httpx

from src.models.schemas import Item, SourceType

logger = logging.getLogger(__name__)


async def fetch_huggingface_papers(top_k: int = 3, timeout: int = 30) -> list[Item]:
    """Fetch top papers from HuggingFace Daily Papers API."""
    url = "https://huggingface.co/api/daily_papers"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()

    papers = response.json()

    # Sort by upvotes descending
    papers.sort(key=lambda p: p.get("paper", {}).get("upvotes", 0), reverse=True)

    items = []
    for p in papers[:top_k]:
        paper = p.get("paper", {})
        items.append(
            Item(
                title=p.get("title", paper.get("title", "")),
                url=f"https://huggingface.co/papers/{paper.get('id', '')}",
                source=SourceType.HUGGINGFACE,
                abstract=paper.get("summary", ""),
                upvotes=paper.get("upvotes", 0),
                arxiv_id=paper.get("id", ""),
                authors=", ".join(a.get("name", "") for a in paper.get("authors", [])[:5]),
            )
        )

    logger.info("HuggingFace: fetched %d papers, returning top %d", len(papers), len(items))
    return items
