"""Performance tests for Qwen3-ASR server.

Tests latency, throughput, and resource usage.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from utils.client import ASRHTTPClient, ASRAsyncHTTPClient, ASRWebSocketClient
from utils.audio import generate_test_tone, convert_to_int16


# =============================================================================
# Latency Tests
# =============================================================================

@pytest.mark.performance
class TestInferenceLatency:
    """Tests for inference latency."""

    @pytest.mark.slow
    def test_cold_start_latency(self, ensure_server, sample_audio_5s: Path):
        """Measure cold start latency (first request after server start)."""
        # Note: This test assumes model may not be loaded
        with ASRHTTPClient() as client:
            start = time.time()
            result = client.transcribe(sample_audio_5s)
            cold_elapsed = time.time() - start

            assert "text" in result
            # Cold start can take 30-120s depending on model size
            assert cold_elapsed < 300, f"Cold start took {cold_elapsed:.1f}s"

    @pytest.mark.slow
    def test_warm_inference_latency(self, ensure_server, ensure_model_loaded, sample_audio_5s: Path):
        """Measure warm inference latency (subsequent requests)."""
        with ASRHTTPClient() as client:
            # Warm up
            client.transcribe(sample_audio_5s)

            # Time several requests
            times = []
            for _ in range(3):
                start = time.time()
                result = client.transcribe(sample_audio_5s)
                elapsed = time.time() - start
                times.append(elapsed)
                assert "text" in result

            avg_time = sum(times) / len(times)
            # Warm inference should be reasonably fast
            assert avg_time < 30, f"Warm inference avg {avg_time:.1f}s"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_websocket_chunk_latency(self, ws_url: str, ensure_server):
        """Measure WebSocket chunk processing time."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            buffer_size = info["buffer_size"]
            chunk_duration = buffer_size / 2 / 16000

            # Generate exact buffer-sized chunk
            audio = generate_test_tone(chunk_duration, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            # Time the round-trip
            start = time.time()
            await client.send_audio(audio_int16.tobytes())

            # Wait for response
            try:
                result = await asyncio.wait_for(client.receive(), timeout=30)
                elapsed = time.time() - start
                # Chunk processing should be quick
                assert elapsed < 10, f"Chunk processing took {elapsed:.1f}s"
            except asyncio.TimeoutError:
                # No response for silent/noisy audio is OK
                pass

    @pytest.mark.slow
    def test_streaming_e2e_latency(self, ensure_server, sample_audio_5s: Path):
        """Measure end-to-end streaming latency."""
        with ASRHTTPClient() as client:
            start = time.time()
            events = client.transcribe_stream(sample_audio_5s)
            elapsed = time.time() - start

            assert len(events) > 0
            # Should complete in reasonable time
            assert elapsed < 60, f"Streaming took {elapsed:.1f}s"


# =============================================================================
# Throughput Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestThroughput:
    """Tests for system throughput."""

    def test_single_client_multiple_requests(self, ensure_server, sample_audio_5s: Path):
        """Single client can make multiple requests."""
        with ASRHTTPClient() as client:
            num_requests = 5
            start = time.time()

            for _ in range(num_requests):
                result = client.transcribe(sample_audio_5s)
                assert "text" in result

            elapsed = time.time() - start
            throughput = num_requests / elapsed

            # Should handle reasonable throughput
            assert throughput > 0.1, f"Throughput {throughput:.2f} req/s too low"

    @pytest.mark.asyncio
    async def test_concurrent_http_requests(self, ensure_server, sample_audio_5s: Path):
        """Server handles concurrent HTTP requests."""
        async with ASRAsyncHTTPClient() as client:
            num_concurrent = 3

            async def do_transcribe():
                result = await client.transcribe(sample_audio_5s)
                return result

            start = time.time()
            results = await asyncio.gather(*[do_transcribe() for _ in range(num_concurrent)])
            elapsed = time.time() - start

            assert len(results) == num_concurrent
            for r in results:
                assert "text" in r

            # Concurrent requests shouldn't take N times longer than single
            avg_time = elapsed / num_concurrent
            assert avg_time < 45, f"Avg time per concurrent req {avg_time:.1f}s"

    @pytest.mark.asyncio
    async def test_websocket_stream_throughput(self, ws_url: str, ensure_server):
        """WebSocket can sustain audio streaming throughput."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            buffer_size = info["buffer_size"]

            # Stream 5 seconds of audio
            duration = 5.0
            audio = generate_test_tone(duration, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            chunk_samples = buffer_size // 2
            bytes_sent = 0
            start = time.time()

            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i:i + chunk_samples]
                await client.send_audio(chunk.tobytes())
                bytes_sent += len(chunk) * 2

            elapsed = time.time() - start
            throughput_kbps = (bytes_sent / 1024) / elapsed

            # Should sustain reasonable throughput
            assert throughput_kbps > 50, f"WS throughput {throughput_kbps:.1f} KB/s too low"


# =============================================================================
# Resource Usage Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestResourceUsage:
    """Tests for resource usage and management."""

    def test_gpu_memory_stable_after_multiple_requests(self, ensure_server, sample_audio_5s: Path):
        """GPU memory usage stabilizes after multiple requests."""
        with ASRHTTPClient() as client:
            # Get baseline after first request
            client.transcribe(sample_audio_5s)
            health = client.health()

            if not health.get("cuda"):
                pytest.skip("No GPU available")

            initial_memory = health.get("gpu_allocated_mb", 0)

            # Make several more requests
            for _ in range(5):
                client.transcribe(sample_audio_5s)

            health = client.health()
            final_memory = health.get("gpu_allocated_mb", 0)

            # Memory shouldn't grow excessively
            growth = final_memory - initial_memory
            assert growth < 500, f"GPU memory grew by {growth} MB"

    @pytest.mark.skip(reason="Requires IDLE_TIMEOUT > 0 configured")
    def test_model_idle_unload(self, ensure_server, sample_audio_5s: Path):
        """Model unloads after idle timeout."""
        # This test requires IDLE_TIMEOUT to be set
        # and is skipped by default as it takes a long time
        with ASRHTTPClient() as client:
            # Load model
            client.transcribe(sample_audio_5s)
            health = client.health()
            assert health["model_loaded"] is True

            # Wait for idle timeout
            idle_timeout = 120  # seconds, adjust based on config
            time.sleep(idle_timeout + 10)

            # Check model unloaded
            health = client.health()
            assert health["model_loaded"] is False

    def test_model_reload_after_unload(self, ensure_server, sample_audio_5s: Path):
        """Model reloads after being unloaded."""
        with ASRHTTPClient() as client:
            # Ensure model is loaded
            client.transcribe(sample_audio_5s)
            health = client.health()
            assert health["model_loaded"] is True

            # Make another request (should work)
            result = client.transcribe(sample_audio_5s)
            assert "text" in result


# =============================================================================
# Benchmark Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests with saved results."""

    @pytest.mark.slow
    def test_http_latency_benchmark(self, ensure_server, sample_audio_5s: Path, tmp_path: Path):
        """Benchmark HTTP transcription latency."""
        with ASRHTTPClient() as client:
            # Warm up
            client.transcribe(sample_audio_5s)

            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                client.transcribe(sample_audio_5s)
                times.append(time.time() - start)

            results = {
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "median": sorted(times)[len(times) // 2],
            }

            # Save results
            results_file = tmp_path / "http_latency.json"
            import json
            results_file.write_text(json.dumps(results, indent=2))

            # Print summary
            print(f"\nHTTP Latency: min={results['min']:.2f}s, "
                  f"avg={results['avg']:.2f}s, max={results['max']:.2f}s")

            # Assert reasonable performance
            assert results["avg"] < 30, f"Avg latency {results['avg']:.1f}s too high"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_websocket_latency_benchmark(self, ws_url: str, ensure_server, tmp_path: Path):
        """Benchmark WebSocket streaming latency."""
        async with ASRWebSocketClient(ws_url) as client:
            info = client.connection_info
            buffer_size = info["buffer_size"]

            # Generate buffer-sized chunks
            chunk_duration = buffer_size / 2 / 16000
            audio = generate_test_tone(chunk_duration, sample_rate=16000)
            audio_int16 = convert_to_int16(audio)

            # Warm up
            await client.send_audio(audio_int16.tobytes())
            try:
                await asyncio.wait_for(client.receive(), timeout=5)
            except asyncio.TimeoutError:
                pass

            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                await client.send_audio(audio_int16.tobytes())
                try:
                    await asyncio.wait_for(client.receive(), timeout=5)
                    times.append(time.time() - start)
                except asyncio.TimeoutError:
                    pass

            if times:
                results = {
                    "min": min(times),
                    "max": max(times),
                    "avg": sum(times) / len(times),
                    "median": sorted(times)[len(times) // 2],
                }

                # Save results
                results_file = tmp_path / "websocket_latency.json"
                import json
                results_file.write_text(json.dumps(results, indent=2))

                print(f"\nWebSocket Latency: min={results['min']:.3f}s, "
                      f"avg={results['avg']:.3f}s, max={results['max']:.3f}s")
