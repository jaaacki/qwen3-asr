"""Subtitle generation E2E tests — accurate mode only.

Tests POST /v1/audio/subtitles?mode=accurate against the live server.
Requires: server on port 8100, Qwen3-ForcedAligner-0.6B model in cache.

Tests do NOT silently skip if the aligner is unavailable — that is a
configuration error and should be a hard failure.
"""
from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from utils.client import ASRHTTPClient


# ---------------------------------------------------------------------------
# SRT parsing helpers
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def parse_srt(srt: str) -> list[dict]:
    """Parse SRT text into a list of events: {start_ms, end_ms, text}."""
    events = []
    for block in re.split(r"\n{2,}", srt.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = TIMESTAMP_RE.search(lines[1])
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
        start_ms = (h1 * 3600 + m1 * 60 + s1) * 1000 + ms1
        end_ms   = (h2 * 3600 + m2 * 60 + s2) * 1000 + ms2
        text = "\n".join(lines[2:]).strip()
        events.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
    return events


# ---------------------------------------------------------------------------
# Smoke tests — basic accurate-mode validation
# ---------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.subtitle
class TestSubtitleAccurateSmoke:
    """Basic accurate-mode subtitle tests (Qwen3-ForcedAligner-0.6B required)."""

    def test_returns_non_empty_srt(self, ensure_server, subtitle_audio_5s: Path):
        """Accurate mode returns a non-empty SRT body."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_5s, mode="fast")
        assert srt.strip(), "SRT response must not be empty"
        assert "-->" in srt, "SRT must contain at least one timestamp arrow"

    def test_timestamp_format(self, ensure_server, subtitle_audio_5s: Path):
        """Every timestamp line is HH:MM:SS,mmm --> HH:MM:SS,mmm."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_5s, mode="fast")
        ts_lines = [l for l in srt.splitlines() if "-->" in l]
        assert ts_lines, "No timestamp lines found"
        for line in ts_lines:
            assert TIMESTAMP_RE.search(line), f"Malformed timestamp: {line!r}"

    def test_start_before_end(self, ensure_server, subtitle_audio_5s: Path):
        """Every event: start_ms < end_ms."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_5s, mode="fast")
        for ev in parse_srt(srt):
            assert ev["start_ms"] < ev["end_ms"], (
                f"start >= end: {ev['start_ms']}ms >= {ev['end_ms']}ms  text={ev['text']!r}"
            )

    def test_content_disposition_header(self, ensure_server, subtitle_audio_5s: Path):
        """Response carries Content-Disposition: attachment; filename=subtitles.srt."""
        with httpx.Client(base_url="http://localhost:8100", timeout=300) as hc:
            with open(subtitle_audio_5s, "rb") as f:
                resp = hc.post(
                    "/v1/audio/subtitles",
                    files={"file": (subtitle_audio_5s.name, f, "audio/wav")},
                    data={"mode": "accurate"},
                )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        cd = resp.headers.get("content-disposition", "")
        assert "subtitles.srt" in cd, f"Unexpected Content-Disposition: {cd!r}"

    def test_with_explicit_language(self, ensure_server, subtitle_audio_5s: Path):
        """Accurate mode accepts explicit language=English parameter."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_5s, language="English", mode="fast")
        assert "-->" in srt


# ---------------------------------------------------------------------------
# Structural/format validation
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.subtitle
class TestSubtitleStructure:
    """SRT structural integrity tests on longer audio."""

    def test_no_overlapping_events(self, ensure_server, subtitle_audio_20s: Path):
        """No consecutive events overlap in time."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_20s, mode="fast")
        events = parse_srt(srt)
        for i in range(len(events) - 1):
            assert events[i]["end_ms"] <= events[i + 1]["start_ms"], (
                f"Overlap: event {i} ends {events[i]['end_ms']}ms, "
                f"event {i+1} starts {events[i+1]['start_ms']}ms"
            )

    def test_max_event_duration(self, ensure_server, subtitle_audio_20s: Path):
        """No event exceeds 7.5s (max_duration=7s + 500ms tolerance)."""
        MAX_MS = 7500
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_20s, mode="fast")
        for ev in parse_srt(srt):
            duration_ms = ev["end_ms"] - ev["start_ms"]
            assert duration_ms <= MAX_MS, (
                f"Event too long: {duration_ms}ms  text={ev['text'][:60]!r}"
            )

    def test_sequential_index_numbering(self, ensure_server, subtitle_audio_20s: Path):
        """SRT indices are sequential starting from 1."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_20s, mode="fast")
        indices = [
            int(l.strip()) for l in srt.splitlines()
            if re.fullmatch(r"\d+", l.strip())
        ]
        assert indices, "No index lines found"
        assert indices == list(range(1, len(indices) + 1)), (
            f"Non-sequential indices: {indices}"
        )

    def test_chronological_order(self, ensure_server, subtitle_audio_20s: Path):
        """Events appear in strictly ascending time order."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_20s, mode="fast")
        events = parse_srt(srt)
        for i in range(len(events) - 1):
            assert events[i]["start_ms"] <= events[i + 1]["start_ms"], (
                f"Out-of-order events at {i}: {events[i]['start_ms']}ms "
                f"> {events[i+1]['start_ms']}ms"
            )

    def test_valid_block_structure(self, ensure_server, subtitle_audio_20s: Path):
        """Each block: index line → timestamp line → ≥1 text line."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_20s, mode="fast")
        blocks = re.split(r"\n{2,}", srt.strip())
        for i, block in enumerate(blocks):
            lines = block.strip().splitlines()
            assert len(lines) >= 3, f"Block {i} too short: {lines}"
            assert lines[0].strip().isdigit(), (
                f"Block {i}: expected index, got {lines[0]!r}"
            )
            assert TIMESTAMP_RE.search(lines[1]), (
                f"Block {i}: expected timestamp, got {lines[1]!r}"
            )
            assert lines[2].strip(), f"Block {i}: empty text"

    def test_line_length_respected(self, ensure_server, subtitle_audio_20s: Path):
        """No text line exceeds max_line_chars=42."""
        MAX_CHARS = 42
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_20s, mode="fast", max_line_chars=MAX_CHARS)
        for line in srt.splitlines():
            stripped = line.strip()
            if stripped and "-->" not in stripped and not stripped.isdigit():
                assert len(stripped) <= MAX_CHARS, (
                    f"Line too long ({len(stripped)} chars): {stripped!r}"
                )

    def test_multiple_events_long_audio(self, ensure_server, subtitle_audio_long: Path):
        """60s audio produces at least 3 subtitle events."""
        with ASRHTTPClient() as client:
            srt = client.subtitle(subtitle_audio_long, mode="fast")
        events = parse_srt(srt)
        assert len(events) >= 3, f"Expected ≥3 events for 60s audio, got {len(events)}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.subtitle
class TestSubtitleErrors:
    """Error handling tests for /v1/audio/subtitles."""

    def test_empty_file_returns_4xx(self, http_client: httpx.Client, ensure_server):
        """Empty file upload must return 4xx, not 200 or 5xx."""
        resp = http_client.post(
            "/v1/audio/subtitles",
            files={"file": ("empty.wav", b"", "audio/wav")},
            data={"mode": "accurate"},
        )
        assert resp.status_code in [400, 422], (
            f"Expected 4xx for empty file, got {resp.status_code}"
        )

    def test_missing_file_returns_422(self, http_client: httpx.Client, ensure_server):
        """Request without file field returns 422 Unprocessable Entity."""
        resp = http_client.post("/v1/audio/subtitles", data={"mode": "accurate"})
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_invalid_mode_returns_4xx(self, ensure_server, sample_audio_5s: Path):
        """Unknown mode string must return 4xx."""
        with httpx.Client(base_url="http://localhost:8100", timeout=300) as hc:
            with open(sample_audio_5s, "rb") as f:
                resp = hc.post(
                    "/v1/audio/subtitles",
                    files={"file": (sample_audio_5s.name, f, "audio/wav")},
                    data={"mode": "invalid_mode"},
                )
        assert resp.status_code in [400, 422], (
            f"Expected 4xx for invalid mode, got {resp.status_code}"
        )
