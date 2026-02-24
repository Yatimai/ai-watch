"""Configuration — env vars + config.yaml."""

import os
from pathlib import Path

import yaml


def load_config() -> dict:
    """Load config from yaml, env vars override."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Env var overrides
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    config["anthropic_api_key"] = api_key
    config["llm_model"] = os.environ.get(
        "LLM_MODEL", config.get("llm_model", "claude-sonnet-4-20250514")
    )

    return config
