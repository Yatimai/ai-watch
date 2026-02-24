"""Tools available to the enrichment agent."""

import ipaddress
import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
]


def _validate_url(url: str) -> None:
    """Validate URL scheme and block private/internal IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme not allowed: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    try:
        addr = ipaddress.ip_address(hostname)
        for network in _BLOCKED_NETWORKS:
            if addr in network:
                raise ValueError(f"URL points to private/internal network: {hostname}")
    except ValueError as e:
        if "private" in str(e) or "not allowed" in str(e) or "no hostname" in str(e):
            raise


@tool
async def fetch_url(url: str) -> str:
    """Fetch and return the text content of a web page.

    Use this when you need to read the full content of a blog post,
    article, or documentation page.
    """
    _validate_url(url)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Truncate to stay within context
    if len(text) > 3000:
        text = text[:3000] + "\n[... tronqué]"

    return text


@tool
async def search_hf_models(query: str) -> str:
    """Search HuggingFace Hub for models matching the query.

    Returns: model name, downloads, tags, last modified.
    Use this when a paper mentions a model and you want concrete details.
    """
    url = "https://huggingface.co/api/models"
    params = {"search": query, "limit": 3, "sort": "downloads", "direction": -1}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()

    models = response.json()
    if not models:
        return f"Aucun modèle trouvé pour '{query}'."

    results = []
    for m in models:
        model_id = m.get("modelId", "?")
        downloads = m.get("downloads", 0)
        tags = ", ".join(m.get("tags", [])[:5])
        last_modified = m.get("lastModified", "?")
        results.append(
            f"- {model_id}: {downloads:,} downloads, tags: [{tags}], modifié: {last_modified}"
        )

    return "\n".join(results)


@tool
async def get_github_repo(owner: str, repo: str) -> str:
    """Get details about a GitHub repository.

    Returns: description, README excerpt, stars, last push date.
    Use this when a repo description is vague and you need more context.
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    async with httpx.AsyncClient(timeout=30) as client:
        # Repo info
        resp = await client.get(base_url)
        resp.raise_for_status()
        data = resp.json()

        # README
        readme_text = ""
        try:
            readme_resp = await client.get(
                f"{base_url}/readme",
                headers={"Accept": "application/vnd.github.raw+json"},
            )
            if readme_resp.status_code == 200:
                readme_text = readme_resp.text[:2000]
        except Exception:
            logger.debug("Could not fetch README for %s/%s", owner, repo)

    description = data.get("description", "") or ""
    stars = data.get("stargazers_count", 0)
    pushed_at = data.get("pushed_at", "?")
    language = data.get("language", "?")

    result = f"""Repo: {owner}/{repo}
Description: {description}
Stars: {stars:,}
Langage: {language}
Dernier push: {pushed_at}

README (extrait):
{readme_text}"""

    if len(result) > 3000:
        result = result[:3000] + "\n[... tronqué]"

    return result
