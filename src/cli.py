"""CLI entry point — python -m src.cli run."""

import asyncio
import logging
import time
from datetime import datetime

from src.agent.graph import compile_graph
from src.models.schemas import BriefingResult
from src.utils.logger import save_briefing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_pipeline() -> None:
    """Run the full pipeline."""
    start = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("Starting pipeline for %s", today)

    graph = compile_graph()
    result = await graph.ainvoke({"today": today})

    duration = time.time() - start

    briefing_result = BriefingResult(
        date=today,
        briefing_markdown=result.get("briefing_markdown", ""),
        items=result.get("items_to_enrich", []),
        enrichment_logs=result.get("enrichment_logs", []),
        sources_status=result.get("sources_status", {}),
        llm_calls=result.get("llm_calls", 0),
        total_tokens=result.get("total_tokens", 0),
        duration_seconds=duration,
    )

    md_path, _log_path = save_briefing(briefing_result)
    logger.info("Pipeline done in %.1fs — %s", duration, md_path)


def main() -> None:
    """Entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(run_pipeline())
    else:
        print("Usage: python -m src.cli run")


if __name__ == "__main__":
    main()
