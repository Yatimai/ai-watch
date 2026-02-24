"""Scrape GitHub Trending page."""

import logging
import re

import httpx
from bs4 import BeautifulSoup

from src.models.schemas import Item, SourceType

logger = logging.getLogger(__name__)


async def fetch_github_trending(timeout: int = 30) -> list[Item]:
    """Scrape GitHub Trending (daily, all languages) and return all repos."""
    url = "https://github.com/trending?since=daily"

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    items = []

    for article in soup.select("article.Box-row"):
        # Repo name: h2 > a with href like /owner/repo
        h2 = article.select_one("h2 a")
        if not h2:
            continue

        repo_path = str(h2.get("href", "")).strip("/")
        parts = repo_path.split("/")
        if len(parts) != 2:
            continue

        owner, name = parts

        # Description
        p = article.select_one("p")
        description = p.get_text(strip=True) if p else ""

        # Language
        lang_span = article.select_one("[itemprop='programmingLanguage']")
        language = lang_span.get_text(strip=True) if lang_span else ""

        # Stars today
        stars_today = 0
        stars_span = article.select_one("span.d-inline-block.float-sm-right")
        if stars_span:
            match = re.search(r"([\d,]+)\s+stars", stars_span.get_text())
            if match:
                stars_today = int(match.group(1).replace(",", ""))

        items.append(
            Item(
                title=f"{owner}/{name}",
                url=f"https://github.com/{owner}/{name}",
                source=SourceType.GITHUB,
                description=description,
                stars_today=stars_today,
                language=language,
                repo_owner=owner,
                repo_name=name,
            )
        )

    if not items:
        raise RuntimeError(
            "GitHub Trending scraping returned 0 repos — page layout may have changed"
        )

    logger.info("GitHub Trending: scraped %d repos", len(items))
    return items
