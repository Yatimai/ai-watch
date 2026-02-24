"""Structured logging for agent decisions.

Logs each enrichment decision so we can show the agent "thinking"
in the briefing logs.
"""

import json
import logging
from pathlib import Path

from src.models.schemas import BriefingResult

logger = logging.getLogger(__name__)


def save_briefing(result: BriefingResult, output_dir: str = "briefings") -> tuple[Path, Path]:
    """Save briefing markdown and logs JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / f"briefing-{result.date}.md"
    log_path = out / f"logs-{result.date}.json"

    md_path.write_text(result.briefing_markdown, encoding="utf-8")

    log_data = {
        "date": result.date,
        "sources_status": result.sources_status,
        "enrichment": [log.model_dump() for log in result.enrichment_logs],
        "llm_calls": result.llm_calls,
        "total_tokens": result.total_tokens,
        "duration_seconds": result.duration_seconds,
    }
    log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Briefing saved: %s", md_path)
    logger.info("Logs saved: %s", log_path)

    return md_path, log_path
