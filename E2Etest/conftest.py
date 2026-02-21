"""Pytest configuration and shared fixtures for E2E tests."""
from __future__ import annotations

import os
import sys
import time
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
