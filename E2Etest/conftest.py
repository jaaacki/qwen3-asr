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

class MarkdownReportGenerator:
    """Collects test results and generates a markdown report."""

    def __init__(self):
        self.results: list[dict] = []
        self.session_start: float = 0.0
        self.session_end: float = 0.0
        self.server_info: dict = {}

    def add_result(self, nodeid: str, outcome: str, duration: float, stdout: str = ""):
        self.results.append({
            "nodeid": nodeid,
            "outcome": outcome,
            "duration": duration,
            "stdout": stdout,
        })

    def _fetch_server_info(self) -> dict:
        """Fetch model/GPU info from the health endpoint."""
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
        """Derive a category from the test nodeid."""
        # e.g. test_api_http.py::TestFoo::test_bar -> HTTP API
        fname = nodeid.split("::")[0].rsplit("/", 1)[-1]
        mapping = {
            "test_api_http": "HTTP API",
            "test_websocket": "WebSocket",
            "test_performance": "Performance",
            "test_integration": "Integration",
            "test_accuracy": "Accuracy",
        }
        for key, label in mapping.items():
            if key in fname:
                return label
        return "Other"

    def _parse_performance_metrics(self) -> list[dict]:
        """Extract performance metrics from stdout of performance tests."""
        metrics = []
        for r in self.results:
            if not r["stdout"]:
                continue
            # Look for patterns like "Cold start: 45.2s" or timing lines
            for line in r["stdout"].splitlines():
                # Match "metric_name: Xs" or "metric_name ... Xs"
                m = re.search(
                    r"([\w\s]+?):\s*([\d.]+)\s*s(?:econds?)?",
                    line, re.IGNORECASE,
                )
                if m:
                    metrics.append({
                        "test": r["nodeid"].split("::")[-1],
                        "metric": m.group(1).strip(),
                        "value": float(m.group(2)),
                    })
        return metrics

    def _parse_accuracy_metrics(self) -> list[dict]:
        """Extract accuracy metrics (WER/CER) from stdout of accuracy tests."""
        metrics = []
        for r in self.results:
            if not r["stdout"]:
                continue
            lines = r["stdout"].splitlines()
            entry: dict = {"test": r["nodeid"].split("::")[-1], "status": r["outcome"]}
            for line in lines:
                lang_m = re.match(r"Language:\s*(.+)", line)
                if lang_m:
                    entry["language"] = lang_m.group(1).strip()
                wer_m = re.match(r"(WER|CER):\s*([\d.]+)%", line)
                if wer_m:
                    entry["metric"] = wer_m.group(1)
                    entry["value"] = float(wer_m.group(2))
                ref_m = re.match(r"Reference:\s*(.+)", line)
                if ref_m:
                    entry["reference"] = ref_m.group(1).strip()
                hyp_m = re.match(r"Hypothesis:\s*(.+)", line)
                if hyp_m:
                    entry["hypothesis"] = hyp_m.group(1).strip()
            if "language" in entry and "metric" in entry:
                metrics.append(entry)
        return metrics

    def generate(self, output_dir: Path) -> Path:
        """Write the markdown report and return the file path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = output_dir / f"{ts}.md"

        info = self._fetch_server_info()
        total_duration = self.session_end - self.session_start

        passed = sum(1 for r in self.results if r["outcome"] == "passed")
        failed = sum(1 for r in self.results if r["outcome"] == "failed")
        skipped = sum(1 for r in self.results if r["outcome"] == "skipped")
        errored = sum(1 for r in self.results if r["outcome"] == "error")
        total = len(self.results)

        lines: list[str] = []
        lines.append(f"# E2E Test Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # --- Summary ---
        lines.append("## Summary\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total | {total} |")
        lines.append(f"| Passed | {passed} |")
        lines.append(f"| Failed | {failed} |")
        lines.append(f"| Skipped | {skipped} |")
        if errored:
            lines.append(f"| Errors | {errored} |")
        lines.append(f"| Duration | {total_duration:.1f}s |")
        model = info.get("model_id") or info.get("model") or "N/A"
        lines.append(f"| Model | {model} |")
        gpu = info.get("gpu") or info.get("gpu_name") or "N/A"
        lines.append(f"| GPU | {gpu} |")
        lines.append("")

        # --- Performance Metrics ---
        perf_metrics = self._parse_performance_metrics()
        if perf_metrics:
            lines.append("## Performance Metrics\n")
            lines.append("| Test | Metric | Value |")
            lines.append("|------|--------|-------|")
            for pm in perf_metrics:
                lines.append(f"| {pm['test']} | {pm['metric']} | {pm['value']:.2f}s |")
            lines.append("")

        # --- Accuracy Breakdown ---
        accuracy_metrics = self._parse_accuracy_metrics()
        if accuracy_metrics:
            lines.append("## Accuracy Breakdown\n")
            lines.append("| Language | Metric | Score | Status | Reference | Hypothesis |")
            lines.append("|----------|--------|-------|--------|-----------|------------|")
            for am in accuracy_metrics:
                lang = am.get("language", "?")
                metric = am.get("metric", "?")
                value = f"{am.get('value', 0):.1f}%"
                status = am.get("status", "?").upper()
                ref = am.get("reference", "")[:80]
                hyp = am.get("hypothesis", "")[:80]
                lines.append(f"| {lang} | {metric} | {value} | {status} | {ref} | {hyp} |")
            lines.append("")

        # --- Results by Category ---
        lines.append("## Results by Category\n")
        categories: dict[str, list[dict]] = {}
        for r in self.results:
            cat = self._categorize(r["nodeid"])
            categories.setdefault(cat, []).append(r)

        for cat in sorted(categories):
            cat_results = categories[cat]
            cat_passed = sum(1 for r in cat_results if r["outcome"] == "passed")
            cat_failed = sum(1 for r in cat_results if r["outcome"] == "failed")
            cat_skipped = sum(1 for r in cat_results if r["outcome"] == "skipped")
            cat_total = len(cat_results)

            if cat_failed > 0:
                icon = "x"
            elif cat_skipped == cat_total:
                icon = "-"
            else:
                icon = "check"

            status_parts = []
            if cat_passed:
                status_parts.append(f"{cat_passed} passed")
            if cat_failed:
                status_parts.append(f"{cat_failed} failed")
            if cat_skipped:
                status_parts.append(f"{cat_skipped} skipped")

            lines.append(f"### {'x' if cat_failed else '-' if cat_skipped == cat_total else '✓'} {cat} ({', '.join(status_parts)})\n")
        lines.append("")

        # --- All Tests ---
        lines.append("## All Tests\n")
        lines.append("| Test | Status | Duration |")
        lines.append("|------|--------|----------|")
        for r in self.results:
            name = r["nodeid"].split("::")[-1]
            status = r["outcome"].upper()
            dur = f"{r['duration']:.2f}s"
            lines.append(f"| {name} | {status} | {dur} |")
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
        # Extract captured stdout from rep.sections
        stdout = ""
        for section_name, section_content in rep.sections:
            if "stdout" in section_name.lower():
                stdout += section_content
        _report_generator.add_result(
            nodeid=rep.nodeid,
            outcome=rep.outcome,
            duration=rep.duration,
            stdout=stdout,
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
    """Auto-generate test audio files if they don't exist."""
    from utils.audio import generate_test_audio_files
    generate_test_audio_files(audio_dir)


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
