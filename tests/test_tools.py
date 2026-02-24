"""Tests for src/agent/tools.py — URL validation, fetch, HF search, GitHub repo."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.agent.tools import (
    _validate_url,
    fetch_url,
    get_github_repo,
    search_hf_models,
)

# ---------------------------------------------------------------------------
# _validate_url
# ---------------------------------------------------------------------------


class TestValidateUrl:
    """Tests for the _validate_url helper."""

    def test_valid_http_url(self):
        _validate_url("http://example.com")

    def test_valid_https_url(self):
        _validate_url("https://example.com/path?q=1")

    def test_ftp_scheme_raises(self):
        with pytest.raises(ValueError, match="scheme not allowed"):
            _validate_url("ftp://example.com/file.txt")

    def test_no_scheme_raises(self):
        with pytest.raises(ValueError, match="scheme not allowed"):
            _validate_url("example.com")

    def test_localhost_ip_raises(self):
        with pytest.raises(ValueError, match="private/internal"):
            _validate_url("http://127.0.0.1/admin")

    def test_private_ip_10_raises(self):
        with pytest.raises(ValueError, match="private/internal"):
            _validate_url("http://10.0.0.1/secret")

    def test_private_ip_192_raises(self):
        with pytest.raises(ValueError, match="private/internal"):
            _validate_url("http://192.168.1.1")

    def test_link_local_raises(self):
        with pytest.raises(ValueError, match="private/internal"):
            _validate_url("http://169.254.1.1")

    def test_normal_domain_passes(self):
        # Domain names (not IP literals) should pass without error
        _validate_url("https://example.com")
        _validate_url("https://docs.python.org/3/library/")


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------


class TestFetchUrl:
    """Tests for the fetch_url tool (async, httpx mocked)."""

    async def test_returns_stripped_text(self):
        html = "<html><body><p> Hello World </p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_url.ainvoke({"url": "https://example.com"})

        assert "Hello World" in result

    async def test_script_style_nav_removed(self):
        html = (
            "<html><body>"
            "<script>alert('xss')</script>"
            "<style>.red{color:red}</style>"
            "<nav>Menu</nav>"
            "<p>Real content</p>"
            "</body></html>"
        )
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_url.ainvoke({"url": "https://example.com"})

        assert "alert" not in result
        assert "red" not in result.lower() or "Real content" in result
        assert "Menu" not in result
        assert "Real content" in result

    async def test_text_truncated_at_3000(self):
        long_text = "A" * 5000
        html = f"<html><body><p>{long_text}</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_url.ainvoke({"url": "https://example.com"})

        # 3000 chars + truncation marker
        assert "[... tronqué]" in result
        # The text before the marker should be at most 3000 chars
        idx = result.index("[... tronqué]")
        assert idx <= 3001  # allow newline before marker

    async def test_ssrf_private_ip_blocked_before_http(self):
        """SSRF protection must reject private IPs before any HTTP call."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(ValueError, match="private/internal"),
        ):
            await fetch_url.ainvoke({"url": "http://10.0.0.1/internal"})

        # HTTP client.get must NEVER have been called
        mock_client.get.assert_not_called()


# ---------------------------------------------------------------------------
# search_hf_models
# ---------------------------------------------------------------------------


class TestSearchHfModels:
    """Tests for search_hf_models tool."""

    async def test_returns_formatted_models(self):
        api_response = [
            {
                "modelId": "google/flan-t5-xxl",
                "downloads": 1_234_567,
                "tags": ["text2text-generation", "pytorch", "t5"],
                "lastModified": "2025-01-15T10:00:00Z",
            },
            {
                "modelId": "meta/llama-3",
                "downloads": 999_000,
                "tags": ["text-generation"],
                "lastModified": "2025-02-01T08:00:00Z",
            },
        ]
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=api_response)
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await search_hf_models.ainvoke({"query": "flan"})

        assert "google/flan-t5-xxl" in result
        assert "1,234,567" in result
        assert "meta/llama-3" in result
        assert "text2text-generation" in result

    async def test_empty_response_returns_aucun(self):
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=[])
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await search_hf_models.ainvoke({"query": "nonexistent-xyz"})

        assert "Aucun modèle trouvé" in result

    async def test_formatting_includes_downloads_and_tags(self):
        api_response = [
            {
                "modelId": "test/model",
                "downloads": 42,
                "tags": ["tag-a", "tag-b", "tag-c", "tag-d", "tag-e", "tag-f"],
                "lastModified": "2025-03-01",
            },
        ]
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=api_response)
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await search_hf_models.ainvoke({"query": "test"})

        assert "42 downloads" in result
        # Only first 5 tags should appear
        assert "tag-a" in result
        assert "tag-e" in result
        assert "tag-f" not in result
        assert "modifié:" in result


# ---------------------------------------------------------------------------
# get_github_repo
# ---------------------------------------------------------------------------


class TestGetGithubRepo:
    """Tests for get_github_repo tool."""

    async def test_returns_formatted_repo_info(self):
        repo_data = {
            "description": "An amazing AI library",
            "stargazers_count": 42_000,
            "pushed_at": "2025-02-20T12:00:00Z",
            "language": "Python",
        }
        readme_text = "# My Repo\n\nThis is the README content."

        mock_repo_resp = MagicMock()
        mock_repo_resp.json = MagicMock(return_value=repo_data)
        mock_repo_resp.raise_for_status = MagicMock()

        mock_readme_resp = MagicMock()
        mock_readme_resp.status_code = 200
        mock_readme_resp.text = readme_text

        async def mock_get(url, **kwargs):
            if "/readme" in url:
                return mock_readme_resp
            return mock_repo_resp

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=mock_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await get_github_repo.ainvoke({"owner": "owner", "repo": "myrepo"})

        assert "An amazing AI library" in result
        assert "42,000" in result
        assert "Python" in result
        assert "owner/myrepo" in result
        assert "My Repo" in result

    async def test_truncation_at_3000(self):
        # Description must be long enough so header + 2000-char README > 3000
        repo_data = {
            "description": "D" * 1500,
            "stargazers_count": 10,
            "pushed_at": "2025-01-01",
            "language": "Rust",
        }
        huge_readme = "X" * 5000

        mock_repo_resp = MagicMock()
        mock_repo_resp.json = MagicMock(return_value=repo_data)
        mock_repo_resp.raise_for_status = MagicMock()

        mock_readme_resp = MagicMock()
        mock_readme_resp.status_code = 200
        mock_readme_resp.text = huge_readme

        async def mock_get(url, **kwargs):
            if "/readme" in url:
                return mock_readme_resp
            return mock_repo_resp

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=mock_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await get_github_repo.ainvoke({"owner": "o", "repo": "r"})

        assert "[... tronqué]" in result

    async def test_readme_fetch_failure_handled_gracefully(self):
        repo_data = {
            "description": "Works without README",
            "stargazers_count": 5,
            "pushed_at": "2025-01-01",
            "language": "Go",
        }

        mock_repo_resp = MagicMock()
        mock_repo_resp.json = MagicMock(return_value=repo_data)
        mock_repo_resp.raise_for_status = MagicMock()

        async def mock_get(url, **kwargs):
            if "/readme" in url:
                raise httpx.ConnectError("connection refused")
            return mock_repo_resp

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=mock_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await get_github_repo.ainvoke({"owner": "o", "repo": "r"})

        # Should still return repo info even though README failed
        assert "Works without README" in result
        assert "Go" in result
