"""Pytest configuration and shared fixtures for E2E tests."""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:8100"
DEFAULT_WS_URL = "ws://localhost:8100/ws/transcribe"
HEALTH_TIMEOUT = 30  # seconds to wait for server health check
MODEL_LOAD_TIMEOUT = 120  # seconds to wait for model to load


# =============================================================================
# Markdown Report Generator
# =============================================================================

# Language-specific WER/CER passing thresholds (%).
_ACCURACY_TARGETS: dict[tuple[str, str], float] = {
    ("english", "WER"): 15.0,
    ("chinese", "WER"): 25.0,
    ("japanese", "WER"): 25.0,
    ("cantonese", "WER"): 30.0,
    ("hindi", "WER"): 30.0,
    ("thai", "WER"): 35.0,
    ("english", "CER"): 10.0,
    ("chinese", "CER"): 20.0,
    ("japanese", "CER"): 20.0,
    ("cantonese", "CER"): 25.0,
    ("hindi", "CER"): 25.0,
    ("thai", "CER"): 30.0,
}
_DEFAULT_WER_TARGET = 35.0
_DEFAULT_CER_TARGET = 30.0

# Per-test duration SLA budgets (seconds). Tests exceeding their budget are
# flagged ⚠️ SLOW in the report even if they pass functionally.
_DURATION_SLAS: dict[str, float] = {
    # Health / validation (should be instant)
    "test_health_returns_200": 2.0,
    "test_health_has_expected_fields": 2.0,
    "test_health_status_is_healthy": 2.0,
    "test_health_gpu_info_when_available": 2.0,
    "test_empty_file_upload": 2.0,
    "test_invalid_audio_format": 2.0,
    "test_missing_file_parameter": 2.0,
    "test_very_small_audio": 5.0,
    "test_client_context_manager": 2.0,
    "test_client_health_includes_model_info": 2.0,
    "test_empty_file_returns_4xx": 2.0,
    "test_missing_file_returns_422": 2.0,
    "test_invalid_mode_returns_4xx": 5.0,
    # WebSocket connection / commands
    "test_connect_success": 5.0,
    "test_connection_info_has_required_fields": 5.0,
    "test_connection_sample_rate_is_16000": 5.0,
    "test_connection_format_is_pcm_s16le": 5.0,
    "test_flush_empty_buffer": 5.0,
    "test_reset_command": 5.0,
    "test_config_set_language": 5.0,
    "test_config_set_auto_language": 5.0,
    "test_invalid_json_handled": 5.0,
    "test_unknown_action_handled": 5.0,
    "test_send_before_connect_fails": 2.0,
    "test_flush_before_connect_fails": 2.0,
    "test_invalid_websocket_url": 2.0,
    # Transcription (includes model load on first call)
    "test_transcribe_short_audio": 25.0,
    "test_transcribe_medium_audio": 35.0,
    "test_transcribe_with_language_param": 25.0,
    "test_transcribe_returns_timestamps_when_requested": 25.0,
    "test_client_transcribe_bytes": 25.0,
    # WebSocket streaming
    "test_send_audio_chunk": 15.0,
    "test_flush_with_audio": 15.0,
    "test_disconnect_transcribes_remaining": 15.0,
    # Recovery
    "test_websocket_recovery_after_error": 15.0,
    "test_http_recovery_after_timeout": 15.0,
    # Subtitle smoke (fast mode, 5s audio)
    "test_returns_non_empty_srt": 30.0,
    "test_timestamp_format": 30.0,
    "test_start_before_end": 30.0,
    "test_content_disposition_header": 30.0,
    "test_with_explicit_language": 30.0,
    # Subtitle structure (20s / 60s audio)
    "test_no_overlapping_events": 60.0,
    "test_max_event_duration": 60.0,
    "test_sequential_index_numbering": 60.0,
    "test_chronological_order": 60.0,
    "test_valid_block_structure": 60.0,
    "test_line_length_respected": 60.0,
    "test_multiple_events_long_audio": 90.0,
}
_DEFAULT_SLA = 30.0


def _accuracy_target(language: str, metric: str) -> float:
    return _ACCURACY_TARGETS.get(
        (language.lower(), metric),
        _DEFAULT_CER_TARGET if metric == "CER" else _DEFAULT_WER_TARGET,
    )


def _sla_for(test_name: str) -> float:
    return _DURATION_SLAS.get(test_name, _DEFAULT_SLA)


class MarkdownReportGenerator:
    """Collects test results and generates a rich markdown report."""

    def __init__(self):
        self.results: list[dict] = []
        self.session_start: float = 0.0
        self.session_end: float = 0.0

    def add_result(
        self,
        nodeid: str,
        outcome: str,
        duration: float,
        stdout: str = "",
        longrepr: str = "",
        skip_reason: str = "",
    ):
        self.results.append({
            "nodeid": nodeid,
            "outcome": outcome,
            "duration": duration,
            "stdout": stdout,
            "longrepr": longrepr,
            "skip_reason": skip_reason,
        })

    def _fetch_server_info(self) -> dict:
        try:
            resp = httpx.get(
                f"{os.getenv('E2E_BASE_URL', DEFAULT_BASE_URL)}/health",
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}

    def _categorize(self, nodeid: str) -> str:
        fname = nodeid.split("::")[0].rsplit("/", 1)[-1]
        mapping = {
            "test_api_http": "HTTP API",
            "test_websocket": "WebSocket",
            "test_performance": "Performance",
            "test_integration": "Integration",
            "test_accuracy": "Accuracy",
            "test_subtitle": "Subtitle",
        }
        for key, label in mapping.items():
            if key in fname:
                return label
        return "Other"

    def _parse_performance_metrics(self) -> list[dict]:
        metrics = []
        for r in self.results:
            # Only parse performance test output — avoids false matches from subtitle prints
            if "test_performance" not in r["nodeid"] or not r["stdout"]:
                continue
            for line in r["stdout"].splitlines():
                m = re.search(r"([\w\s]+?):\s*([\d.]+)\s*s(?:econds?)?", line, re.IGNORECASE)
                if m:
                    metrics.append({
                        "test": r["nodeid"].split("::")[-1],
                        "metric": m.group(1).strip(),
                        "value": float(m.group(2)),
                    })
        return metrics

    def _parse_accuracy_metrics(self) -> list[dict]:
        metrics = []
        for r in self.results:
            if not r["stdout"]:
                continue
            entry: dict = {"test": r["nodeid"].split("::")[-1], "status": r["outcome"]}
            for line in r["stdout"].splitlines():
                m = re.match(r"Language:\s*(.+)", line)
                if m:
                    entry["language"] = m.group(1).strip()
                m = re.match(r"(WER|CER):\s*([\d.]+)%", line)
                if m:
                    entry["metric"] = m.group(1)
                    entry["value"] = float(m.group(2))
                m = re.match(r"Reference:\s*(.+)", line)
                if m:
                    entry["reference"] = m.group(1).strip()
                m = re.match(r"Hypothesis:\s*(.+)", line)
                if m:
                    entry["hypothesis"] = m.group(1).strip()
            if "language" in entry and "metric" in entry:
                metrics.append(entry)
        return metrics

    def _parse_transcription_results(self) -> list[dict]:
        """Extract transcription output printed by transcription tests."""
        results = []
        for r in self.results:
            if not r["stdout"]:
                continue
            entry: dict = {"test": r["nodeid"].split("::")[-1], "duration": r["duration"]}
            for line in r["stdout"].splitlines():
                m = re.match(r'Audio:\s*(.+)', line)
                if m:
                    entry["audio"] = m.group(1).strip()
                m = re.match(r'Transcription:\s*(.+)', line)
                if m:
                    entry["transcription"] = m.group(1).strip()
                m = re.match(r'Detected:\s*(.+)', line)
                if m:
                    entry["detected"] = m.group(1).strip()
            if "transcription" in entry:
                results.append(entry)
        return results

    def _parse_subtitle_results(self) -> list[dict]:
        """Extract SRT metrics printed by subtitle tests."""
        results = []
        for r in self.results:
            if not r["stdout"]:
                continue
            entry: dict = {"test": r["nodeid"].split("::")[-1], "duration": r["duration"]}
            for line in r["stdout"].splitlines():
                m = re.match(r'Audio:\s*(.+)', line)
                if m:
                    entry["audio"] = m.group(1).strip()
                m = re.match(r'SRT Events:\s*(\d+)', line)
                if m:
                    entry["events"] = int(m.group(1))
                m = re.match(r'SRT Max Duration:\s*([\d.]+)s', line)
                if m:
                    entry["max_duration"] = float(m.group(1))
                m = re.match(r'SRT Max Line:\s*(\d+)\s*chars', line)
                if m:
                    entry["max_line"] = int(m.group(1))
                m = re.match(r'SRT First:\s*(.+)', line)
                if m:
                    entry["first_event"] = m.group(1).strip()
            if any(k in entry for k in ("events", "first_event", "max_duration")):
                results.append(entry)
        return results

    def generate(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = output_dir / f"{ts}.md"

        info = self._fetch_server_info()
        total_duration = self.session_end - self.session_start

        passed  = sum(1 for r in self.results if r["outcome"] == "passed")
        failed  = sum(1 for r in self.results if r["outcome"] == "failed")
        skipped = sum(1 for r in self.results if r["outcome"] == "skipped")
        errored = sum(1 for r in self.results if r["outcome"] == "error")
        total   = len(self.results)

        # Duration stats
        durations = sorted(r["duration"] for r in self.results if r["outcome"] != "skipped")
        p50 = durations[len(durations) // 2] if durations else 0.0
        slowest = max(self.results, key=lambda r: r["duration"]) if self.results else None

        lines: list[str] = []
        lines.append(f"# E2E Test Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # ── Summary ──────────────────────────────────────────────────────────
        overall_icon = "✅" if failed == 0 else "❌"
        lines.append(f"## {overall_icon} Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total | {total} |")
        lines.append(f"| ✅ Passed | {passed} |")
        if failed:
            lines.append(f"| ❌ Failed | {failed} |")
        if skipped:
            lines.append(f"| ⏭️ Skipped | {skipped} |")
        if errored:
            lines.append(f"| 💥 Errors | {errored} |")
        lines.append(f"| ⏱️ Total Duration | {total_duration:.1f}s |")
        lines.append(f"| ⏱️ p50 Duration | {p50:.2f}s |")
        if slowest:
            sname = slowest["nodeid"].split("::")[-1]
            lines.append(f"| 🐢 Slowest | {sname} ({slowest['duration']:.1f}s) |")
        model = info.get("model_id") or info.get("model") or "N/A"
        lines.append(f"| 🤖 Model | {model} |")
        gpu = info.get("gpu") or info.get("gpu_name") or "N/A"
        lines.append(f"| 🖥️ GPU | {gpu} |")
        lines.append("")

        # ── Failures ─────────────────────────────────────────────────────────
        failures = [r for r in self.results if r["outcome"] in ("failed", "error")]
        if failures:
            lines.append("## ❌ Failures\n")
            lines.append("| Test | Error |")
            lines.append("|------|-------|")
            for r in failures:
                name = r["nodeid"].split("::")[-1]
                # Find the actual exception line: "E   SomeError: ..."
                error_msg = "—"
                for ln in r["longrepr"].splitlines():
                    stripped = ln.strip()
                    if stripped.startswith("E ") and any(
                        kw in stripped for kw in ("Error", "Exception", "Assert", "Failed")
                    ):
                        error_msg = stripped.lstrip("E").strip()[:150]
                        break
                if error_msg == "—":
                    # Fallback: last non-empty line
                    repr_lines = [l.strip() for l in r["longrepr"].splitlines() if l.strip()]
                    error_msg = repr_lines[-1][:150] if repr_lines else "—"
                lines.append(f"| `{name}` | `{error_msg}` |")
            lines.append("")

        # ── Skipped ──────────────────────────────────────────────────────────
        skips = [r for r in self.results if r["outcome"] == "skipped"]
        if skips:
            lines.append("## ⏭️ Skipped\n")
            lines.append("| Test | Reason |")
            lines.append("|------|--------|")
            for r in skips:
                name = r["nodeid"].split("::")[-1]
                reason = r["skip_reason"] or "—"
                lines.append(f"| `{name}` | {reason} |")
            lines.append("")

        # ── Transcription Results ─────────────────────────────────────────────
        txn_results = self._parse_transcription_results()
        if txn_results:
            lines.append("## 🎙️ Transcription Results\n")
            lines.append("| Test | Audio | Transcription | Language | Duration |")
            lines.append("|------|-------|---------------|----------|----------|")
            for t in txn_results:
                name     = t["test"]
                audio    = t.get("audio", "—")
                text     = t.get("transcription", "—")[:80]
                detected = t.get("detected", "—")
                dur      = f"{t['duration']:.1f}s"
                lines.append(f"| `{name}` | {audio} | {text} | {detected} | {dur} |")
            lines.append("")

        # ── Subtitle Results ──────────────────────────────────────────────────
        sub_results = self._parse_subtitle_results()
        if sub_results:
            lines.append("## 📄 Subtitle Results\n")
            lines.append("| Test | Audio | Events | Max Duration | Max Line | First Event |")
            lines.append("|------|-------|--------|--------------|----------|-------------|")
            for s in sub_results:
                name     = s["test"]
                audio    = s.get("audio", "—")
                events   = str(s["events"]) if "events" in s else "—"
                max_dur  = f"{s['max_duration']:.1f}s" if "max_duration" in s else "—"
                max_line = f"{s['max_line']} chars" if "max_line" in s else "—"
                first    = s.get("first_event", "—")[:60]
                lines.append(f"| `{name}` | {audio} | {events} | {max_dur} | {max_line} | {first} |")
            lines.append("")

        # ── Accuracy Breakdown ────────────────────────────────────────────────
        accuracy_metrics = self._parse_accuracy_metrics()
        if accuracy_metrics:
            lines.append("## 📊 Accuracy Breakdown\n")
            lines.append("| Language | Metric | Score | Target | Pass | Reference | Hypothesis |")
            lines.append("|----------|--------|-------|--------|------|-----------|------------|")
            for am in accuracy_metrics:
                lang     = am.get("language", "?")
                metric   = am.get("metric", "?")
                value    = am.get("value", 0)
                target   = _accuracy_target(lang, metric)
                ok       = value <= target
                lines.append(
                    f"| {lang} | {metric} | {value:.1f}% | ≤{target:.0f}% "
                    f"| {'✅' if ok else '❌'} "
                    f"| {am.get('reference','')[:80]} | {am.get('hypothesis','')[:80]} |"
                )
            lines.append("")

        # ── Performance Metrics ───────────────────────────────────────────────
        perf_metrics = self._parse_performance_metrics()
        if perf_metrics:
            lines.append("## ⚡ Performance Metrics\n")
            lines.append("| Test | Metric | Value |")
            lines.append("|------|--------|-------|")
            for pm in perf_metrics:
                lines.append(f"| {pm['test']} | {pm['metric']} | {pm['value']:.2f}s |")
            lines.append("")

        # ── Results by Category ───────────────────────────────────────────────
        lines.append("## Results by Category\n")
        categories: dict[str, list[dict]] = {}
        for r in self.results:
            cat = self._categorize(r["nodeid"])
            categories.setdefault(cat, []).append(r)

        lines.append("| Category | ✅ Passed | ❌ Failed | ⏭️ Skipped |")
        lines.append("|----------|----------|----------|-----------|")
        for cat in sorted(categories):
            cr = categories[cat]
            cp = sum(1 for r in cr if r["outcome"] == "passed")
            cf = sum(1 for r in cr if r["outcome"] == "failed")
            cs = sum(1 for r in cr if r["outcome"] == "skipped")
            icon = "❌" if cf else ("⏭️" if cs == len(cr) else "✅")
            lines.append(f"| {icon} {cat} | {cp} | {cf} | {cs} |")
        lines.append("")

        # ── All Tests ─────────────────────────────────────────────────────────
        lines.append("## All Tests\n")
        lines.append("| Test | Status | Duration | SLA |")
        lines.append("|------|--------|----------|-----|")
        for r in self.results:
            name    = r["nodeid"].split("::")[-1]
            outcome = r["outcome"]
            dur     = r["duration"]
            sla     = _sla_for(name)

            if outcome == "passed":
                status_icon = "✅"
                sla_icon    = "✅" if dur <= sla else f"⚠️ SLOW (≤{sla:.0f}s)"
            elif outcome == "skipped":
                status_icon = "⏭️"
                sla_icon    = "—"
            else:
                status_icon = "❌"
                sla_icon    = "—"

            lines.append(f"| `{name}` | {status_icon} {outcome.upper()} | {dur:.2f}s | {sla_icon} |")
        lines.append("")

        report_path.write_text("\n".join(lines))
        return report_path


# Module-level report generator instance
_report_generator = MarkdownReportGenerator()


# =============================================================================
# Pytest hooks for report generation
# =============================================================================

def pytest_sessionstart(session):
    """Record session start time."""
    _report_generator.session_start = time.time()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture each test result for the markdown report."""
    outcome = yield
    rep = outcome.get_result()
    # Only record the "call" phase (not setup/teardown), or skip from setup
    if rep.when == "call" or (rep.when == "setup" and rep.skipped):
        # Captured stdout
        stdout = ""
        for section_name, section_content in rep.sections:
            if "stdout" in section_name.lower():
                stdout += section_content

        # Failure message (last meaningful line of the traceback)
        longrepr = ""
        if rep.failed and rep.longrepr:
            longrepr = str(rep.longrepr)

        # Skip reason
        skip_reason = ""
        if rep.skipped and rep.longrepr:
            if isinstance(rep.longrepr, tuple) and len(rep.longrepr) >= 3:
                reason = rep.longrepr[2]
                skip_reason = reason[len("Skipped: "):] if reason.startswith("Skipped: ") else reason
            else:
                skip_reason = str(rep.longrepr)

        _report_generator.add_result(
            nodeid=rep.nodeid,
            outcome=rep.outcome,
            duration=rep.duration,
            stdout=stdout,
            longrepr=longrepr,
            skip_reason=skip_reason,
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate the markdown report at the end of the session."""
    _report_generator.session_end = time.time()
    reports_dir = Path(__file__).parent / "reports"
    report_path = _report_generator.generate(reports_dir)
    terminalreporter.write_sep("=", "Markdown Report")
    terminalreporter.write_line(f"Report saved to: {report_path}")


# =============================================================================
# Session-scoped fixtures
# =============================================================================

@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for HTTP API endpoints."""
    return os.getenv("E2E_BASE_URL", DEFAULT_BASE_URL)


@pytest.fixture(scope="session")
def ws_url() -> str:
    """WebSocket URL for streaming transcription."""
    return os.getenv("E2E_WS_URL", DEFAULT_WS_URL)


@pytest.fixture(scope="session")
def api_key() -> str | None:
    """Optional API key for authentication."""
    return os.getenv("E2E_API_KEY")


@pytest.fixture(scope="session")
def server_available(base_url: str) -> bool:
    """Check if server is running without failing."""
    try:
        response = httpx.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def ensure_server(base_url: str):
    """Verify server is running, skip tests if not available."""
    health_url = f"{base_url}/health"
    start_time = time.time()
    last_error = None

    while time.time() - start_time < HEALTH_TIMEOUT:
        try:
            response = httpx.get(health_url, timeout=5)
            if response.status_code == 200:
                return  # Server is ready
        except Exception as e:
            last_error = e
        time.sleep(1)

    # Server not available
    if last_error:
        pytest.skip(f"Server not available at {base_url}: {last_error}")
    else:
        pytest.skip(f"Server not responding at {base_url}")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def audio_dir(data_dir: Path) -> Path:
    """Path to audio test files directory."""
    return data_dir / "audio"


@pytest.fixture(scope="session", autouse=True)
def generate_test_audio(audio_dir: Path):
    """Auto-generate synthetic test audio files if they don't exist."""
    from utils.audio import generate_test_audio_files
    generate_test_audio_files(audio_dir)


@pytest.fixture(scope="session")
def subtitle_audio_dir(data_dir: Path) -> Path:
    """Path to real-speech audio directory for subtitle tests."""
    return data_dir / "audio" / "subtitle"


@pytest.fixture(scope="session", autouse=False)
def download_subtitle_audio(subtitle_audio_dir: Path):
    """Download real-speech subtitle test audio if not already present."""
    needed = ["speech_5s.wav", "speech_20s.wav", "speech_60s.wav"]
    if all((subtitle_audio_dir / f).exists() for f in needed):
        return
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from download_test_audio import download_subtitle_samples
        download_subtitle_samples()
    except Exception as e:
        pytest.skip(f"Could not download subtitle audio samples: {e}")


@pytest.fixture(scope="session")
def subtitle_audio_5s(download_subtitle_audio, subtitle_audio_dir: Path) -> Path:
    """~5s real speech audio for subtitle smoke tests."""
    path = subtitle_audio_dir / "speech_5s.wav"
    if not path.exists():
        pytest.fail(f"Subtitle audio not found: {path}  (run: python E2Etest/download_test_audio.py --subtitle-only)")
    return path


@pytest.fixture(scope="session")
def subtitle_audio_20s(download_subtitle_audio, subtitle_audio_dir: Path) -> Path:
    """~20s real speech audio for subtitle structural tests."""
    path = subtitle_audio_dir / "speech_20s.wav"
    if not path.exists():
        pytest.fail(f"Subtitle audio not found: {path}  (run: python E2Etest/download_test_audio.py --subtitle-only)")
    return path


@pytest.fixture(scope="session")
def subtitle_audio_long(download_subtitle_audio, subtitle_audio_dir: Path) -> Path:
    """~60s real speech audio for subtitle multi-event tests."""
    path = subtitle_audio_dir / "speech_60s.wav"
    if not path.exists():
        pytest.fail(f"Subtitle audio not found: {path}  (run: python E2Etest/download_test_audio.py --subtitle-only)")
    return path


# =============================================================================
# Test audio file fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_audio_5s(audio_dir: Path) -> Path:
    """Path to 5-second test audio file."""
    path = audio_dir / "short_5s.wav"
    if not path.exists():
        pytest.skip(f"Test audio not found: {path}")
    return path


@pytest.fixture(scope="session")
def sample_audio_20s(audio_dir: Path) -> Path:
    """Path to 20-second test audio file (test01_20s.wav)."""
    # First try the existing file in root
    root_path = Path(__file__).parent.parent / "test01_20s.wav"
    if root_path.exists():
        return root_path

    # Then try E2Etest location
    path = audio_dir / "medium_20s.wav"
    if path.exists():
        return path

    pytest.skip(f"Test audio not found: tried {root_path} and {path}")


@pytest.fixture(scope="session")
def sample_audio_long(audio_dir: Path) -> Path:
    """Path to 60-second test audio file for chunking tests."""
    path = audio_dir / "long_60s.wav"
    if not path.exists():
        pytest.skip(f"Test audio not found: {path}")
    return path


@pytest.fixture(scope="session")
def sample_audio_noisy(audio_dir: Path) -> Path:
    """Path to noisy audio for repetition detection test."""
    path = audio_dir / "noisy_sample.wav"
    if not path.exists():
        pytest.skip(f"Test audio not found: {path}")
    return path


# =============================================================================
# HTTP client fixtures
# =============================================================================

@pytest.fixture
def http_client(base_url: str, api_key: str | None) -> Generator[httpx.Client, None, None]:
    """Synchronous HTTP client for API calls."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(base_url=base_url, headers=headers, timeout=300) as client:
        yield client


@pytest.fixture
async def async_http_client(base_url: str, api_key: str | None):
    """Asynchronous HTTP client for API calls."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=300) as client:
        yield client


# =============================================================================
# Model management fixtures
# =============================================================================

@pytest.fixture
def ensure_model_loaded(http_client: httpx.Client, sample_audio_5s: Path):
    """Ensure model is loaded by triggering a transcription if needed."""
    response = http_client.get("/health")
    if response.status_code == 200:
        data = response.json()
        if data.get("model_loaded"):
            return

    # Trigger model load with actual transcription
    with open(sample_audio_5s, "rb") as f:
        files = {"file": (sample_audio_5s.name, f, "audio/wav")}
        http_client.post("/v1/audio/transcriptions", files=files, data={"language": "auto"}, timeout=300)

    # Verify loaded
    start_time = time.time()
    while time.time() - start_time < MODEL_LOAD_TIMEOUT:
        response = http_client.get("/health")
        if response.status_code == 200 and response.json().get("model_loaded"):
            return
        time.sleep(2)

    pytest.skip("Model failed to load within timeout")


@pytest.fixture
def reset_model_state():
    """Reset any cached model state between tests if needed."""
    # Placeholder for state reset logic
    yield
    # Cleanup after test


# =============================================================================
# Pytest configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")
    config.addinivalue_line("markers", "websocket: marks tests as WebSocket tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "requires_aligner: marks tests that require Qwen3-ForcedAligner-0.6B")
    config.addinivalue_line("markers", "subtitle: marks tests as subtitle tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test name."""
    for item in items:
        # Auto-mark slow tests
        if "performance" in item.nodeid or "latency" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        if "smoke" in item.nodeid:
            item.add_marker(pytest.mark.smoke)
        if "websocket" in item.nodeid or "ws" in item.nodeid:
            item.add_marker(pytest.mark.websocket)
        if "subtitle" in item.nodeid:
            item.add_marker(pytest.mark.subtitle)
