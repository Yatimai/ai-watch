"""Fetch Simon Willison's RSS feed."""

import asyncio
import logging
from datetime import datetime

import feedparser

from src.models.schemas import Item, SourceType

logger = logging.getLogger(__name__)


async def fetch_simon_willison(top_k: int = 3, timeout: int = 30) -> list[Item]:
    """Fetch recent posts from Simon Willison's RSS feed."""
    feed_url = "https://simonwillison.net/atom/everything/"

    feed = await asyncio.wait_for(
        asyncio.to_thread(feedparser.parse, feed_url),
        timeout=timeout,
    )

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Failed to parse Simon Willison RSS: {feed.bozo_exception}")

    items = []
    for entry in feed.entries[:top_k]:
        # Extract tags
        tags = [tag.get("term", "") for tag in entry.get("tags", [])]

        # Extract published date
        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6])

        # Content snippet: prefer summary, fall back to content
        snippet = ""
        if hasattr(entry, "summary") and entry.summary:
            snippet = entry.summary[:500]
        elif hasattr(entry, "content") and entry.content:
            snippet = entry.content[0].get("value", "")[:500]

        items.append(
            Item(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source=SourceType.SIMON,
                tags=tags,
                content_snippet=snippet,
                published_at=published,
            )
        )

    logger.info("Simon Willison: fetched %d posts", len(items))
    return items
