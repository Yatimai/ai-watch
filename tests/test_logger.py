"""Tests for src/utils/logger.py — save_briefing file output."""

import json

from src.models.schemas import BriefingResult, EnrichmentLog, Item, SourceType
from src.utils.logger import save_briefing


def _make_briefing_result() -> BriefingResult:
    """Create a sample BriefingResult for testing."""
    item = Item(
        title="Test Paper",
        url="https://example.com/paper",
        source=SourceType.HUGGINGFACE,
        abstract="A test abstract.",
        upvotes=10,
    )
    enrichment = EnrichmentLog(
        item_title="Test Paper",
        source=SourceType.HUGGINGFACE,
        tools_called=["fetch_url"],
        reason="Needed abstract details",
    )
    return BriefingResult(
        date="2025-02-24",
        briefing_markdown="# AI Watch Briefing\n\n## Papers\n\n- Test Paper",
        items=[item],
        enrichment_logs=[enrichment],
        sources_status={"huggingface": "ok", "github": "error", "simon": "ok"},
        llm_calls=3,
        total_tokens=8500,
        duration_seconds=22.7,
    )


class TestSaveBriefing:
    """Tests for the save_briefing function."""

    def test_creates_markdown_file(self, tmp_path):
        result = _make_briefing_result()
        md_path, _ = save_briefing(result, output_dir=str(tmp_path))

        assert md_path.exists()
        assert md_path.name == "briefing-2025-02-24.md"

    def test_creates_json_log_file(self, tmp_path):
        result = _make_briefing_result()
        _, log_path = save_briefing(result, output_dir=str(tmp_path))

        assert log_path.exists()
        assert log_path.name == "logs-2025-02-24.json"

    def test_markdown_contents_correct(self, tmp_path):
        result = _make_briefing_result()
        md_path, _ = save_briefing(result, output_dir=str(tmp_path))

        content = md_path.read_text(encoding="utf-8")
        assert content == "# AI Watch Briefing\n\n## Papers\n\n- Test Paper"

    def test_json_log_contents_correct(self, tmp_path):
        result = _make_briefing_result()
        _, log_path = save_briefing(result, output_dir=str(tmp_path))

        log_data = json.loads(log_path.read_text(encoding="utf-8"))

        assert log_data["date"] == "2025-02-24"
        assert log_data["sources_status"]["huggingface"] == "ok"
        assert log_data["sources_status"]["github"] == "error"
        assert log_data["llm_calls"] == 3
        assert log_data["total_tokens"] == 8500
        assert log_data["duration_seconds"] == 22.7

        # Enrichment log entry
        assert len(log_data["enrichment"]) == 1
        entry = log_data["enrichment"][0]
        assert entry["item_title"] == "Test Paper"
        assert entry["source"] == "huggingface"
        assert entry["tools_called"] == ["fetch_url"]
        assert entry["reason"] == "Needed abstract details"

    def test_creates_output_directory_if_missing(self, tmp_path):
        nested_dir = tmp_path / "deep" / "nested" / "output"
        assert not nested_dir.exists()

        result = _make_briefing_result()
        md_path, log_path = save_briefing(result, output_dir=str(nested_dir))

        assert nested_dir.exists()
        assert md_path.exists()
        assert log_path.exists()

    def test_returns_correct_paths(self, tmp_path):
        result = _make_briefing_result()
        md_path, log_path = save_briefing(result, output_dir=str(tmp_path))

        assert md_path.parent == tmp_path
        assert log_path.parent == tmp_path
        assert str(md_path).endswith(".md")
        assert str(log_path).endswith(".json")
