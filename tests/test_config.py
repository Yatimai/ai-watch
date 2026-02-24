"""Tests for src/config.py — load_config with env vars and yaml."""

from unittest.mock import mock_open, patch

import pytest

from src.config import load_config

SAMPLE_YAML = """\
llm_model: "claude-sonnet-4-20250514"
sources:
  huggingface:
    enabled: true
    api_url: "https://huggingface.co/api/daily_papers"
    top_k: 3
  github_trending:
    enabled: true
    url: "https://github.com/trending"
    since: "daily"
    top_k: 3
  simon_willison:
    enabled: true
    feed_url: "https://simonwillison.net/atom/everything/"
    top_k: 3
scrape_timeout: 30
briefings_dir: "briefings"
"""


def _patch_yaml(env_vars: dict | None = None, yaml_content: str = SAMPLE_YAML):
    """Return a combined context manager that patches both open() and os.environ."""
    if env_vars is None:
        env_vars = {"ANTHROPIC_API_KEY": "sk-test-key-123"}
    return (
        patch("builtins.open", mock_open(read_data=yaml_content)),
        patch.dict("os.environ", env_vars, clear=True),
    )


class TestLoadConfig:
    """Tests for load_config()."""

    def test_with_api_key_set(self):
        p_open, p_env = _patch_yaml({"ANTHROPIC_API_KEY": "sk-test-key-123"})
        with p_open, p_env:
            config = load_config()

        assert config["anthropic_api_key"] == "sk-test-key-123"
        assert "sources" in config
        assert config["sources"]["huggingface"]["enabled"] is True

    def test_without_api_key_raises_value_error(self):
        """BUG-3 fix: missing ANTHROPIC_API_KEY must raise ValueError."""
        p_open, p_env = _patch_yaml({"ANTHROPIC_API_KEY": ""})
        with p_open, p_env, pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            load_config()

    def test_missing_api_key_env_var_raises(self):
        """No ANTHROPIC_API_KEY in env at all."""
        p_open, p_env = _patch_yaml({})
        with p_open, p_env, pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            load_config()

    def test_whitespace_only_api_key_raises(self):
        p_open, p_env = _patch_yaml({"ANTHROPIC_API_KEY": "   "})
        with p_open, p_env, pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            load_config()

    def test_llm_model_env_override(self):
        env = {
            "ANTHROPIC_API_KEY": "sk-key",
            "LLM_MODEL": "claude-opus-4-20250514",
        }
        p_open, p_env = _patch_yaml(env)
        with p_open, p_env:
            config = load_config()

        assert config["llm_model"] == "claude-opus-4-20250514"

    def test_llm_model_default_from_yaml(self):
        p_open, p_env = _patch_yaml({"ANTHROPIC_API_KEY": "sk-key"})
        with p_open, p_env:
            config = load_config()

        assert config["llm_model"] == "claude-sonnet-4-20250514"

    def test_yaml_values_loaded(self):
        p_open, p_env = _patch_yaml({"ANTHROPIC_API_KEY": "sk-key"})
        with p_open, p_env:
            config = load_config()

        assert config["scrape_timeout"] == 30
        assert config["briefings_dir"] == "briefings"
        assert config["sources"]["github_trending"]["since"] == "daily"
        assert config["sources"]["simon_willison"]["top_k"] == 3
