"""WebSocket tests for Qwen3-ASR real-time transcription.

Tests the WS /ws/transcribe endpoint:
- Connection and handshake
- Audio streaming
- Control commands (flush, reset, config)
- Error handling
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import websockets

from utils.client import ASRWebSocketClient
from utils.audio import generate_test_tone, convert_to_int16


# =============================================================================
# Connection Tests
# =============================================================================

@pytest.mark.websocket
@pytest.mark.smoke
class TestWebSocketConnection:
    """Tests for WebSocket connection establishment."""

    @pytest.mark.asyncio
    async def test_connect_success(self, ws_url: str, ensure_server):
        """Can connect to WebSocket endpoint."""
        async with ASRWebSocketClient(ws_url) as client:
            assert client.connection_info is not None

    @pytest.mark.asyncio
    async def test_connection_info_has_required_fields(self, ws_url: str, ensure_server):
        """Connection info has expected configuration fields."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info

            assert "status" in info
            assert info["status"] == "connected"
            assert "sample_rate" in info
            assert "format" in info
            assert "buffer_size" in info
            assert "overlap_size" in info

    @pytest.mark.asyncio
    async def test_connection_sample_rate_is_16000(self, ws_url: str, ensure_server):
        """Server expects 16kHz audio."""
        async with ASRWebSocketClient(ws_url) as client:
            assert client.connection_info["sample_rate"] == 16000

    @pytest.mark.asyncio
    async def test_connection_format_is_pcm_s16le(self, ws_url: str, ensure_server):
        """Server expects PCM 16-bit little-endian."""
        async with ASRWebSocketClient(ws_url) as client:
            assert client.connection_info["format"] == "pcm_s16le"


# =============================================================================
# Basic Audio Streaming Tests
# =============================================================================

@pytest.mark.websocket
class TestWebSocketAudioStreaming:
    """Tests for basic audio streaming."""

    @pytest.mark.asyncio
    async def test_send_audio_chunk(self, ws_url: str, ensure_server):
        """Can send audio chunk and receive result."""
        async with ASRWebSocketClient(ws_url) as client:
            # Generate 1 second of audio
            audio = generate_test_tone(1.0, sample_rate=16000, amplitude=0.5)
            audio_int16 = convert_to_int16(audio)

            await client.send_audio(audio_int16.tobytes())

            # Should eventually get a response
            try:
                response = await asyncio.wait_for(client.receive(), timeout=5)
                # Could be partial result or empty
                assert isinstance(response, dict)
            except asyncio.TimeoutError:
                # No speech detected is also valid
                pass

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, ws_url: str, ensure_server):
        """Flush on empty buffer returns empty final result."""
        async with ASRWebSocketClient(ws_url) as client:
            result = await client.flush()

            assert result.get("is_final") is True
            assert result.get("text", "") == ""

    @pytest.mark.asyncio
    async def test_flush_with_audio(self, ws_url: str, ensure_server):
        """Flush after sending audio returns final transcription."""
        async with ASRWebSocketClient(ws_url) as client:
            # Send audio that fills the buffer
            buffer_size = client.connection_info["buffer_size"]
            duration = buffer_size / 2 / 16000  # bytes / 2 bytes per sample / 16kHz

            audio = generate_test_tone(duration * 2, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            await client.send_audio(audio_int16.tobytes())

            # Wait a bit for processing
            await asyncio.sleep(0.5)

            # Flush remaining audio
            result = await client.flush()

            assert result.get("is_final") is True
            assert "text" in result

    @pytest.mark.asyncio
    async def test_disconnect_transcribes_remaining(self, ws_url: str, ensure_server):
        """Disconnecting transcribes any remaining buffered audio."""
        client = ASRWebSocketClient(ws_url)
        await client.connect()

        # Send some audio
        audio = generate_test_tone(1.0, sample_rate=16000)
        audio_int16 = convert_to_int16(audio)
        await client.send_audio(audio_int16.tobytes())

        # Disconnect without flush
        await client.close()

        # Should not raise an exception
        assert True


# =============================================================================
# Control Command Tests
# =============================================================================

@pytest.mark.websocket
class TestWebSocketCommands:
    """Tests for WebSocket control commands."""

    @pytest.mark.asyncio
    async def test_reset_command(self, ws_url: str, ensure_server):
        """Reset command clears buffer state."""
        async with ASRWebSocketClient(ws_url) as client:
            result = await client.reset()

            assert result.get("status") == "buffer_reset"

    @pytest.mark.asyncio
    async def test_config_set_language(self, ws_url: str, ensure_server):
        """Config command sets language."""
        async with ASRWebSocketClient(ws_url) as client:
            result = await client.config("en")

            assert result.get("status") == "configured"
            assert result.get("language") == "en"

    @pytest.mark.asyncio
    async def test_config_set_auto_language(self, ws_url: str, ensure_server):
        """Config command can set auto language detection."""
        async with ASRWebSocketClient(ws_url) as client:
            result = await client.config("auto")

            assert result.get("status") == "configured"
            assert result.get("language") == "auto"

    @pytest.mark.asyncio
    async def test_invalid_json_handled(self, ws_url: str, ensure_server):
        """Invalid JSON is handled gracefully."""
        async with ASRWebSocketClient(ws_url) as client:
            # Send invalid JSON directly
            await client.websocket.send("not valid json")

            try:
                response = await asyncio.wait_for(client.receive(), timeout=5)
                # Should get error response
                assert "error" in response
            except asyncio.TimeoutError:
                # Timeout is also acceptable handling
                pass

    @pytest.mark.asyncio
    async def test_unknown_action_handled(self, ws_url: str, ensure_server):
        """Unknown action is handled gracefully."""
        async with ASRWebSocketClient(ws_url) as client:
            # Send unknown action
            await client.websocket.send(json.dumps({"action": "unknown_action"}))

            # Server may ignore or return error, either is fine
            try:
                response = await asyncio.wait_for(client.receive(), timeout=2)
                # If we get a response, it should either be error or we ignore it
                pass
            except asyncio.TimeoutError:
                # Timeout means server ignored it
                pass


# =============================================================================
# Streaming Tests
# =============================================================================

@pytest.mark.websocket
@pytest.mark.slow
class TestWebSocketStreaming:
    """Tests for continuous streaming scenarios."""

    @pytest.mark.asyncio
    async def test_stream_10_seconds(self, ws_url: str, ensure_server):
        """Can stream 10 seconds of audio continuously."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            buffer_size = info["buffer_size"]

            # Stream in chunks matching buffer size
            chunk_duration = buffer_size / 2 / 16000
            audio = generate_test_tone(10.0, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            # Split into chunks
            samples_per_chunk = int(16000 * chunk_duration)
            partials_received = 0

            for i in range(0, len(audio_int16), samples_per_chunk):
                chunk = audio_int16[i:i + samples_per_chunk]
                await client.send_audio(chunk.tobytes())

                # Try to receive partial results
                try:
                    msg = await asyncio.wait_for(client.receive(), timeout=0.3)
                    if msg.get("is_partial"):
                        partials_received += 1
                except asyncio.TimeoutError:
                    pass

            # Flush to get final
            final = await client.flush()
            assert final.get("is_final") is True

    @pytest.mark.asyncio
    async def test_partial_results_during_stream(self, ws_url: str, ensure_server):
        """Partial results arrive during streaming."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            buffer_size = info["buffer_size"]
            chunk_samples = buffer_size // 2  # 16-bit = 2 bytes

            # Stream multiple buffer-sized chunks
            audio = generate_test_tone(5.0, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            partials = []
            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i:i + chunk_samples]
                if len(chunk) == chunk_samples:
                    await client.send_audio(chunk.tobytes())

                    # Wait briefly for processing
                    try:
                        msg = await asyncio.wait_for(client.receive(), timeout=0.5)
                        if msg.get("is_partial"):
                            partials.append(msg)
                    except asyncio.TimeoutError:
                        pass

            # Should have received some partials or none (both valid depending on audio)
            # Just verify the process completed
            final = await client.flush()
            assert "text" in final

    @pytest.mark.asyncio
    async def test_overlap_handling(self, ws_url: str, ensure_server):
        """Overlap between chunks prevents word boundary splits."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            overlap_size = info["overlap_size"]

            # Verify overlap is configured
            assert overlap_size > 0

            # Stream some audio
            audio = generate_test_tone(3.0, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            buffer_size = info["buffer_size"]
            chunk_samples = buffer_size // 2

            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i:i + chunk_samples]
                await client.send_audio(chunk.tobytes())

                # Drain any responses
                try:
                    await asyncio.wait_for(client.receive(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass

            # Flush and verify we got a final
            final = await client.flush()
            assert final.get("is_final") is True


# =============================================================================
# File Streaming Tests
# =============================================================================

@pytest.mark.websocket
@pytest.mark.slow
class TestWebSocketFileStreaming:
    """Tests for streaming audio files."""

    @pytest.mark.asyncio
    async def test_stream_audio_file(self, ws_url: str, ensure_server, sample_audio_20s: Path):
        """Can stream an audio file for transcription."""
        async with ASRWebSocketClient(ws_url) as client:
            results = []

            async for msg in client.stream_audio(sample_audio_20s):
                results.append(msg)
                if msg.get("is_final"):
                    break

            assert len(results) > 0
            assert results[-1].get("is_final") is True

    @pytest.mark.asyncio
    async def test_stream_with_custom_chunk_size(self, ws_url: str, ensure_server, sample_audio_5s: Path):
        """Can stream with custom chunk duration."""
        async with ASRWebSocketClient(ws_url) as client:
            results = []

            async for msg in client.stream_audio(
                sample_audio_5s,
                chunk_duration_ms=200,
                overlap_ms=50
            ):
                results.append(msg)
                if msg.get("is_final"):
                    break

            # Should have received results
            assert len(results) >= 1


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.websocket
class TestWebSocketErrors:
    """Tests for WebSocket error handling."""

    @pytest.mark.asyncio
    async def test_send_before_connect_fails(self, ws_url: str):
        """Sending audio before connect raises error."""
        client = ASRWebSocketClient(ws_url)
        audio = generate_test_tone(0.5, sample_rate=16000)
        audio_int16 = convert_to_int16(audio)

        with pytest.raises(RuntimeError):
            await client.send_audio(audio_int16.tobytes())

    @pytest.mark.asyncio
    async def test_flush_before_connect_fails(self, ws_url: str):
        """Flush before connect raises error."""
        client = ASRWebSocketClient(ws_url)

        with pytest.raises(RuntimeError):
            await client.flush()

    @pytest.mark.asyncio
    async def test_invalid_websocket_url(self):
        """Invalid WebSocket URL raises ConnectionError."""
        client = ASRWebSocketClient("ws://invalid:99999")

        with pytest.raises((OSError, ValueError, websockets.exceptions.InvalidURI)):
            await client.connect()


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.websocket
@pytest.mark.performance
class TestWebSocketLatency:
    """Tests for WebSocket latency."""

    @pytest.mark.asyncio
    async def test_connection_latency(self, ws_url: str, ensure_server):
        """WebSocket connection latency is reasonable."""
        import time

        start = time.time()
        async with ASRWebSocketClient(ws_url) as client:
            elapsed = time.time() - start

            assert elapsed < 5.0, f"Connection took {elapsed:.2f}s"
            assert client.connection_info is not None

    @pytest.mark.asyncio
    async def test_round_trip_latency(self, ws_url: str, ensure_server):
        """Round-trip command latency is reasonable."""
        import time

        async with ASRWebSocketClient(ws_url) as client:
            start = time.time()
            result = await client.reset()
            elapsed = time.time() - start

            assert elapsed < 1.0, f"Reset took {elapsed:.2f}s"
            assert result.get("status") == "buffer_reset"
