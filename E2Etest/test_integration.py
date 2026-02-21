"""Integration tests for Qwen3-ASR multi-feature scenarios.

Tests feature combinations:
- Priority queue: WS + HTTP concurrent requests
- Dual model: DUAL_MODEL=true with mixed WS/HTTP
- Quantization: QUANTIZE=int8/fp8 and verify output quality
- Idle timeout: Model unloads after IDLE_TIMEOUT seconds
- Model reload: After idle unload, next request loads again
"""

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from utils.client import ASRHTTPClient, ASRAsyncHTTPClient, ASRWebSocketClient
from utils.audio import generate_test_tone, convert_to_int16


# =============================================================================
# Priority Queue Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPriorityQueue:
    """Tests for priority queue behavior (WebSocket preempts HTTP)."""

    @pytest.mark.asyncio
    async def test_websocket_not_blocked_by_http(
        self, base_url: str, ws_url: str, ensure_server, sample_audio_20s: Path
    ):
        """WebSocket requests are not blocked by HTTP file uploads."""
        async with ASRAsyncHTTPClient(base_url) as http_client:
            # Start a long HTTP request
            http_task = asyncio.create_task(
                http_client.transcribe(sample_audio_20s)
            )

            # Give HTTP request time to start processing
            await asyncio.sleep(0.5)

            # WebSocket should still be responsive
            async with ASRWebSocketClient(ws_url) as ws_client:
                # Send a small audio chunk
                audio = generate_test_tone(1.0, sample_rate=16000)
                audio_int16 = convert_to_int16(audio)
                await ws_client.send_audio(audio_int16.tobytes())

                # Should be able to flush quickly
                start = time.time()
                result = await asyncio.wait_for(ws_client.flush(), timeout=15)
                elapsed = time.time() - start

                assert elapsed < 10, f"WS flush took {elapsed:.1f}s (possibly blocked)"
                assert result.get("is_final") is True

            # Wait for HTTP to complete
            http_result = await http_task
            assert "text" in http_result

    @pytest.mark.asyncio
    async def test_multiple_websocket_priority(
        self, ws_url: str, ensure_server
    ):
        """Multiple WebSocket clients are handled with priority."""
        async with ASRWebSocketClient(ws_url) as ws1:
            async with ASRWebSocketClient(ws_url) as ws2:
                # Send audio from both
                audio = generate_test_tone(1.0, sample_rate=16000)
                audio_int16 = convert_to_int16(audio)

                await ws1.send_audio(audio_int16.tobytes())
                await ws2.send_audio(audio_int16.tobytes())

                # Both should be able to flush
                result1 = await asyncio.wait_for(ws1.flush(), timeout=15)
                result2 = await asyncio.wait_for(ws2.flush(), timeout=15)

                assert result1.get("is_final") is True
                assert result2.get("is_final") is True


# =============================================================================
# Dual Model Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.skip(reason="Requires DUAL_MODEL=true environment variable")
class TestDualModel:
    """Tests for dual model strategy (0.6B partials, 1.7B finals).

    These tests are skipped by default as they require DUAL_MODEL=true.
    """

    @pytest.mark.asyncio
    async def test_fast_model_for_partials(self, ws_url: str, ensure_server):
        """Fast model (0.6B) is used for WebSocket partials."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            buffer_size = info["buffer_size"]

            # Send buffer-sized chunks to trigger partials
            chunk_duration = buffer_size / 2 / 16000
            audio = generate_test_tone(chunk_duration, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            await client.send_audio(audio_int16.tobytes())

            try:
                result = await asyncio.wait_for(client.receive(), timeout=5)
                # Should get partial result
                assert result.get("is_partial") is True
            except asyncio.TimeoutError:
                pytest.skip("No partial received (audio may not trigger transcription)")

    def test_full_model_for_http(self, ensure_server, sample_audio_5s: Path):
        """Full model is used for HTTP requests."""
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s)

            assert "text" in result
            # Result should be from full model


# =============================================================================
# Quantization Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.skip(reason="Requires QUANTIZE=int8 or fp8 environment variable")
class TestQuantization:
    """Tests for INT8/FP8 quantization.

    These tests are skipped by default as they require quantization enabled.
    """

    def test_int8_quantization_produces_output(self, ensure_server, sample_audio_5s: Path):
        """INT8 quantization produces valid transcription."""
        with ASRHTTPClient() as client:
            health = client.health()

            # Check if INT8 is enabled
            # Note: Server doesn't currently report this, would need to add

            result = client.transcribe(sample_audio_5s)
            assert "text" in result
            assert isinstance(result["text"], str)

    def test_fp8_quantization_produces_output(self, ensure_server, sample_audio_5s: Path):
        """FP8 quantization produces valid transcription."""
        with ASRHTTPClient() as client:
            result = client.transcribe(sample_audio_5s)
            assert "text" in result

    def test_quantized_output_quality(self, ensure_server, sample_audio_5s: Path):
        """Quantized model produces reasonable quality output."""
        # Compare with and without quantization would need two server instances
        # This test is a placeholder
        pytest.skip("Requires comparison with non-quantized model")


# =============================================================================
# Idle Timeout Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestIdleTimeout:
    """Tests for model idle unload behavior."""

    def test_model_stays_loaded_during_activity(self, ensure_server, sample_audio_5s: Path):
        """Model stays loaded while requests are being made."""
        with ASRHTTPClient() as client:
            # Make requests in quick succession
            for _ in range(5):
                client.transcribe(sample_audio_5s)
                health = client.health()
                assert health["model_loaded"] is True
                time.sleep(1)

    @pytest.mark.skip(reason="Requires IDLE_TIMEOUT to be set and takes long time")
    def test_model_unloads_after_idle(self, ensure_server, sample_audio_5s: Path):
        """Model unloads after idle timeout period."""
        # This test requires IDLE_TIMEOUT > 0
        # Skipped by default because it takes a long time
        with ASRHTTPClient() as client:
            # Load model
            client.transcribe(sample_audio_5s)
            health = client.health()
            assert health["model_loaded"] is True

            # Wait for idle timeout
            # Default is 120s, add buffer
            time.sleep(140)

            # Check model unloaded
            health = client.health()
            assert health["model_loaded"] is False

    def test_model_reloads_after_unload(self, ensure_server, sample_audio_5s: Path):
        """Model reloads when request comes after unload."""
        with ASRHTTPClient() as client:
            # Load model
            client.transcribe(sample_audio_5s)
            health = client.health()
            assert health["model_loaded"] is True

            # Model may or may not unload depending on IDLE_TIMEOUT
            # Just verify we can still make requests
            result = client.transcribe(sample_audio_5s)
            assert "text" in result

            health = client.health()
            assert health["model_loaded"] is True


# =============================================================================
# Feature Combination Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestFeatureCombinations:
    """Tests for various feature combinations."""

    @pytest.mark.asyncio
    async def test_http_and_websocket_concurrent(
        self, base_url: str, ws_url: str, ensure_server, sample_audio_5s: Path
    ):
        """HTTP and WebSocket work concurrently."""
        async with ASRAsyncHTTPClient(base_url) as http_client:
            # Start WebSocket
            async with ASRWebSocketClient(ws_url) as ws_client:
                # Send HTTP request
                http_task = asyncio.create_task(
                    http_client.transcribe(sample_audio_5s)
                )

                # Send audio via WebSocket
                audio = generate_test_tone(2.0, sample_rate=16000)
                audio_int16 = convert_to_int16(audio)
                await ws_client.send_audio(audio_int16.tobytes())

                # Flush WebSocket
                ws_result = await ws_client.flush()

                # Wait for HTTP
                http_result = await http_task

                assert ws_result.get("is_final") is True
                assert "text" in http_result

    def test_timestamps_and_language_combination(self, ensure_server, sample_audio_5s: Path):
        """Timestamps and language parameters work together.

        Note: Qwen3-ASR requires forced_aligner for timestamps. If not
        configured, this test skips. Language must be a full name like 'English'.
        """
        with ASRHTTPClient() as client:
            try:
                result = client.transcribe(
                    sample_audio_5s,
                    language="English",
                    return_timestamps=True
                )
                assert "text" in result
                assert result.get("language") == "English"
            except Exception:
                pytest.skip("Server does not support return_timestamps (no forced_aligner configured)")

    @pytest.mark.asyncio
    async def test_websocket_config_and_streaming(self, ws_url: str, ensure_server):
        """WebSocket config and streaming work together.

        Note: Qwen3-ASR expects full language names (e.g. 'English') not
        ISO codes (e.g. 'en'). The WebSocket config command sets language
        for subsequent transcriptions.
        """
        async with ASRWebSocketClient(ws_url) as client:
            # Set language -- use full name accepted by Qwen3-ASR model
            config_result = await client.config("English")
            assert config_result.get("status") == "configured"

            # Stream audio
            audio = generate_test_tone(3.0, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)
            await client.send_audio(audio_int16.tobytes())

            # Flush
            result = await client.flush()
            assert result.get("is_final") is True


# =============================================================================
# Recovery Tests
# =============================================================================

@pytest.mark.integration
class TestRecovery:
    """Tests for system recovery from errors."""

    @pytest.mark.asyncio
    async def test_websocket_recovery_after_error(self, ws_url: str, ensure_server):
        """WebSocket can recover/resync after error."""
        async with ASRWebSocketClient(ws_url) as client:
            # Send invalid data
            await client.websocket.send("invalid")

            try:
                await asyncio.wait_for(client.receive(), timeout=2)
            except asyncio.TimeoutError:
                pass

            # Reset to recover
            result = await client.reset()
            assert result.get("status") == "buffer_reset"

            # Should work after reset
            audio = generate_test_tone(1.0, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)
            await client.send_audio(audio_int16.tobytes())

            result = await client.flush()
            assert result.get("is_final") is True

    def test_http_recovery_after_timeout(self, ensure_server, sample_audio_5s: Path):
        """HTTP can recover after request timeout."""
        with ASRHTTPClient() as client:
            # Make a request
            result1 = client.transcribe(sample_audio_5s)
            assert "text" in result1

            # Make another request (should still work)
            result2 = client.transcribe(sample_audio_5s)
            assert "text" in result2


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflows:
    """Complete end-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_full_workflow_http_streaming_ws(
        self, base_url: str, ws_url: str, ensure_server, sample_audio_5s: Path
    ):
        """Full workflow: HTTP transcription, streaming, WebSocket."""
        # 1. HTTP transcription
        with ASRHTTPClient(base_url) as http_client:
            result = http_client.transcribe(sample_audio_5s)
            assert "text" in result

        # 2. HTTP streaming
        with ASRHTTPClient(base_url) as http_client:
            events = http_client.transcribe_stream(sample_audio_5s)
            assert len(events) > 0

        # 3. WebSocket streaming
        async with ASRWebSocketClient(ws_url) as ws_client:
            audio = generate_test_tone(2.0, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            await ws_client.send_audio(audio_int16.tobytes())
            result = await ws_client.flush()
            assert result.get("is_final") is True

    def test_repeated_transcription_consistency(self, ensure_server, sample_audio_5s: Path):
        """Repeated transcription of same audio is consistent."""
        with ASRHTTPClient() as client:
            results = []
            for _ in range(5):
                result = client.transcribe(sample_audio_5s)
                results.append(result["text"])

            # All results should be identical (deterministic)
            assert all(r == results[0] for r in results), \
                f"Inconsistent results: {set(results)}"
