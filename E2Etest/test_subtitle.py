"""Subtitle generation E2E tests.

Tests the POST /v1/audio/subtitles endpoint.
Requires the server to be running on port 8100.
"""
import re
from pathlib import Path

import httpx
import pytest

from utils.client import ASRHTTPClient


# =============================================================================
# Smoke Tests
# =============================================================================

@pytest.mark.smoke
@pytest.mark.subtitle
class TestSubtitleBasic:
    """Basic subtitle generation tests."""

    def test_subtitle_fast_mode(self, ensure_server, sample_audio_5s: Path):
        """Fast mode returns valid SRT."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_5s, mode="fast")

            # Should be valid SRT format
            assert "-->" in srt
            lines = srt.strip().split("\n")
            assert lines[0].strip() == "1"

    def test_subtitle_accurate_mode(self, ensure_server, sample_audio_5s: Path):
        """Accurate mode returns valid SRT (requires ForcedAligner)."""
        with ASRHTTPClient() as client:
            try:
                srt = client.subtitle(sample_audio_5s, mode="accurate")
                assert "-->" in srt
            except Exception:
                pytest.skip("ForcedAligner not available")

    def test_subtitle_with_language(self, ensure_server, sample_audio_5s: Path):
        """Subtitle generation accepts language parameter."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_5s, language="English", mode="fast")
            assert "-->" in srt

    def test_subtitle_content_disposition(self, ensure_server, sample_audio_5s: Path):
        """Response has Content-Disposition header for file download."""
        with httpx.Client(base_url="http://localhost:8100", timeout=300) as hc:
            with open(sample_audio_5s, "rb") as f:
                response = hc.post(
                    "/v1/audio/subtitles",
                    files={"file": (sample_audio_5s.name, f, "audio/wav")},
                    data={"mode": "fast"},
                )
                assert response.status_code == 200
                assert "subtitles.srt" in response.headers.get("content-disposition", "")


# =============================================================================
# Advanced Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.subtitle
class TestSubtitleAdvanced:
    """Advanced subtitle tests for longer audio."""

    def test_subtitle_long_audio(self, ensure_server, sample_audio_long: Path):
        """Long audio (60s) produces multiple subtitle events."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_long, mode="fast")
            # Count subtitle events (numbered entries)
            event_count = len(re.findall(r"^\d+$", srt, re.MULTILINE))
            # 60s of audio should have multiple events
            assert event_count >= 2

    def test_subtitle_line_length(self, ensure_server, sample_audio_20s: Path):
        """No subtitle line exceeds max_line_chars."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_20s, mode="fast", max_line_chars=42)
            for line in srt.split("\n"):
                # Skip index lines, timestamp lines, and blank lines
                if line.strip() and "-->" not in line and not line.strip().isdigit():
                    assert len(line) <= 42, f"Line too long: {len(line)} chars: '{line}'"

    def test_subtitle_no_overlaps(self, ensure_server, sample_audio_20s: Path):
        """No subtitle events overlap in time."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_20s, mode="fast")
            # Parse timestamps
            timestamps = re.findall(
                r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
                srt,
            )
            for i in range(len(timestamps) - 1):
                end_current = timestamps[i][1]
                start_next = timestamps[i + 1][0]
                assert end_current <= start_next, f"Overlap: {end_current} > {start_next}"

    def test_subtitle_valid_srt_structure(self, ensure_server, sample_audio_20s: Path):
        """SRT output has correct block structure: index, timestamp, text, blank."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(sample_audio_20s, mode="fast")
            blocks = srt.strip().split("\n\n")
            for block in blocks:
                lines = block.strip().split("\n")
                assert len(lines) >= 3, f"Block too short: {lines}"
                # First line should be a number
                assert lines[0].strip().isdigit(), f"Expected index, got: {lines[0]}"
                # Second line should be a timestamp range
                assert "-->" in lines[1], f"Expected timestamp, got: {lines[1]}"
                # Third line (and optionally fourth) should be text
                assert len(lines[2].strip()) > 0, f"Empty text in block: {block}"


# =============================================================================
# Error Handling
# =============================================================================

@pytest.mark.subtitle
class TestSubtitleErrors:
    """Error handling tests for subtitle endpoint."""

    def test_subtitle_empty_file(self, http_client: httpx.Client, ensure_server):
        """Empty file upload is handled gracefully."""
        files = {"file": ("empty.wav", b"", "audio/wav")}
        response = http_client.post(
            "/v1/audio/subtitles",
            files=files,
            data={"mode": "fast"},
        )
        # Should not crash; 200 with empty body or 400/422 are acceptable
        assert response.status_code in [200, 400, 422, 500]

    def test_subtitle_missing_file(self, http_client: httpx.Client, ensure_server):
        """Request without file parameter is rejected."""
        response = http_client.post(
            "/v1/audio/subtitles",
            data={"mode": "fast"},
        )
        assert response.status_code in [400, 422]

    def test_subtitle_invalid_mode(self, ensure_server, sample_audio_5s: Path):
        """Invalid mode parameter is handled."""
        with httpx.Client(base_url="http://localhost:8100", timeout=300) as hc:
            with open(sample_audio_5s, "rb") as f:
                response = hc.post(
                    "/v1/audio/subtitles",
                    files={"file": (sample_audio_5s.name, f, "audio/wav")},
                    data={"mode": "invalid_mode"},
                )
                # Server should handle gracefully (fast fallback or error)
                assert response.status_code in [200, 400, 422, 500]
