"""Real-time WebSocket latency and accuracy benchmark.

Streams a ~10s FLEURS audio clip to the live WebSocket endpoint in
real-time and measures:
  - Per-chunk input-to-output latency (min / median / p95 / max)
  - Flush latency (last audio -> final transcript)
  - Real-Time Factor (RTF): inference time / audio duration
  - WER and CER against the known FLEURS reference transcript

Run:
    pytest E2Etest/test_realtime_accuracy.py -v
    pytest E2Etest/ -m realtime -v

Requires FLEURS audio to be downloaded first:
    python E2Etest/download_test_audio.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from utils.client import ASRWebSocketClient
from test_accuracy import calculate_wer, calculate_cer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 450  # matches default WS_BUFFER_SIZE (~450ms)
_RESPONSE_DRAIN_TIMEOUT = 2.0  # seconds; wait for sliding window inference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fleurs_sample(
    audio_dir: Path,
    expected_dir: Path,
    lang: str,
    idx: int = 1,
) -> tuple[Path, str]:
    """Return (wav_path, reference_text) for a FLEURS sample."""
    wav = audio_dir / "real" / f"{lang}_{idx:02d}.wav"
    txt = expected_dir / f"{lang}_{idx:02d}.txt"
    return wav, txt.read_text().strip() if txt.exists() else ""


async def _stream_and_time(ws_url: str, audio: np.ndarray) -> dict:
    """Stream int16 PCM audio to the WebSocket and return timing + transcript data.

    Returns a dict with keys:
        chunk_latencies_ms: list[float]  -- per-partial latency in ms
        flush_latency_ms: float          -- flush -> final response
        rtf: float                       -- sum(infer_time) / audio_duration
        final_text: str                  -- text from the final (flush) response
        partials: list[dict]             -- [{t_audio, t_recv, latency_ms, text}, ...]
        audio_duration: float
        num_chunks: int
    """
    samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
    chunks = [
        audio[i: i + samples_per_chunk]
        for i in range(0, len(audio), samples_per_chunk)
    ]
    audio_duration = len(audio) / SAMPLE_RATE

    chunk_latencies_ms: list[float] = []
    partials: list[dict] = []
    infer_times: list[float] = []
    flush_latency_ms = 0.0

    async with ASRWebSocketClient(ws_url) as client:
        t_start = time.perf_counter()

        for i, chunk in enumerate(chunks):
            # Wall-clock position of the end of this chunk in the audio timeline
            t_audio_pos = (i + 1) * (CHUNK_DURATION_MS / 1000)

            await client.send_audio(chunk.tobytes())
            t_sent = time.perf_counter()

            # Real-time pacing: sleep until we've reached the audio timeline position
            elapsed = t_sent - t_start
            sleep_s = t_audio_pos - elapsed
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            # Drain any partial responses that arrived during this chunk window
            while True:
                try:
                    msg = await asyncio.wait_for(
                        client.receive(), timeout=_RESPONSE_DRAIN_TIMEOUT
                    )
                    t_recv = time.perf_counter()
                    # Latency = how long after the audio was "spoken" did the result arrive
                    latency_ms = (t_recv - t_start - t_audio_pos) * 1000
                    chunk_latencies_ms.append(latency_ms)
                    infer_times.append(t_recv - t_sent)
                    if msg.get("text"):
                        partials.append({
                            "t_audio": round(t_audio_pos, 3),
                            "t_recv": round(t_recv - t_start, 3),
                            "latency_ms": round(latency_ms, 1),
                            "text": msg["text"],
                        })
                except asyncio.TimeoutError:
                    break

        # Flush -- collects trailing audio and pads silence
        t_flush = time.perf_counter()
        await client.websocket.send(json.dumps({"action": "flush"}))

        final_text = ""
        while True:
            msg = await asyncio.wait_for(client.receive(), timeout=60)
            if msg.get("is_final"):
                flush_latency_ms = (time.perf_counter() - t_flush) * 1000
                final_text = msg.get("text", "")
                break

    rtf = sum(infer_times) / audio_duration if infer_times else 0.0

    # Use last partial if flush returned empty (window already drained)
    if not final_text and partials:
        final_text = partials[-1]["text"]

    return {
        "chunk_latencies_ms": chunk_latencies_ms,
        "flush_latency_ms": flush_latency_ms,
        "rtf": rtf,
        "final_text": final_text,
        "partials": partials,
        "audio_duration": audio_duration,
        "num_chunks": len(chunks),
    }


def _latency_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(latencies_ms)
    p95_idx = max(0, int(len(s) * 0.95) - 1)
    return {
        "min": s[0],
        "median": statistics.median(s),
        "p95": s[p95_idx],
        "max": s[-1],
    }


def _print_report(
    wav: Path,
    lang: str,
    reference: str,
    final_text: str,
    wer: float,
    cer: float,
    stats: dict,
    result: dict,
):
    """Print metrics in the format parsed by MarkdownReportGenerator."""
    num_chunks = result["num_chunks"]
    duration = result["audio_duration"]
    print(
        f"\nAudio: {wav.name} ({duration:.1f}s, {SAMPLE_RATE}Hz mono, "
        f"{num_chunks} chunks x {CHUNK_DURATION_MS}ms)"
    )
    print(f"Language: {lang}")
    print(f"Reference: {reference[:120]}")
    print(f"Hypothesis: {final_text[:120]}")
    print(f"WER: {wer:.1f}%")
    print(f"CER: {cer:.1f}%")
    print(f"Realtime Chunk Min: {stats['min']:.0f}ms")
    print(f"Realtime Chunk Median: {stats['median']:.0f}ms")
    print(f"Realtime Chunk P95: {stats['p95']:.0f}ms")
    print(f"Realtime Chunk Max: {stats['max']:.0f}ms")
    print(f"Realtime Flush Latency: {result['flush_latency_ms']:.0f}ms")
    print(f"Realtime RTF: {result['rtf']:.2f}x")

    # Timeline: first 5 partials
    if result["partials"]:
        print("\n-- Timeline (first 5 partials) --")
        for p in result["partials"][:5]:
            print(
                f"  t={p['t_audio']:.2f}s -> +{p['latency_ms']:.0f}ms  "
                f"\"{p['text'][:60]}\""
            )
        if len(result["partials"]) > 5:
            p = result["partials"][-1]
            print(f"  ... ({len(result['partials'])} total)")
            print(
                f"  t={p['t_audio']:.2f}s -> +{p['latency_ms']:.0f}ms  "
                f"\"{p['text'][:60]}\""
            )


def _save_json(result: dict, lang: str, wer: float, cer: float, stats: dict):
    """Save full results to E2Etest/reports/realtime_latest.json."""
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "language": lang,
        "wer_pct": round(wer, 2),
        "cer_pct": round(cer, 2),
        "latency": {
            "chunk_min_ms": round(stats["min"], 1),
            "chunk_median_ms": round(stats["median"], 1),
            "chunk_p95_ms": round(stats["p95"], 1),
            "chunk_max_ms": round(stats["max"], 1),
            "flush_ms": round(result["flush_latency_ms"], 1),
            "rtf": round(result["rtf"], 3),
        },
        "audio_duration_s": round(result["audio_duration"], 2),
        "num_chunks": result["num_chunks"],
        "chunk_duration_ms": CHUNK_DURATION_MS,
        "partials": result["partials"],
    }
    out_path = reports_dir / "realtime_latest.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved -> {out_path}")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.websocket
@pytest.mark.accuracy
@pytest.mark.realtime
@pytest.mark.slow
class TestRealtimeLatencyAndAccuracy:
    """Stream known FLEURS clips and measure input-to-output latency + WER/CER."""

    @pytest.mark.asyncio
    async def test_english_realtime_benchmark(
        self,
        ws_url: str,
        ensure_server,
        audio_dir: Path,
        data_dir: Path,
    ):
        """English FLEURS clip: real-time latency + WER/CER benchmark."""
        expected_dir = data_dir / "expected"
        wav, reference = _load_fleurs_sample(audio_dir, expected_dir, "english")

        if not wav.exists():
            pytest.skip(
                f"FLEURS audio not found: {wav}\n"
                "Run: python E2Etest/download_test_audio.py"
            )
        if not reference:
            pytest.skip(f"Reference transcript missing: {expected_dir / 'english_01.txt'}")

        audio, sr = sf.read(str(wav), dtype="int16")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)
        if sr != SAMPLE_RATE:
            new_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.int16)

        result = await _stream_and_time(ws_url, audio)

        wer = calculate_wer(reference, result["final_text"]) * 100
        cer = calculate_cer(reference, result["final_text"]) * 100
        stats = _latency_stats(result["chunk_latencies_ms"])

        _print_report(wav, "english", reference, result["final_text"], wer, cer, stats, result)
        _save_json(result, "english", wer, cer, stats)

        assert result["final_text"], "Expected non-empty transcription"
        assert wer <= 55.0, f"WER {wer:.1f}% exceeds 55% streaming threshold"
        assert stats["median"] < 30000, (
            f"Median chunk latency {stats['median']:.0f}ms exceeds 30000ms"
        )

    @pytest.mark.asyncio
    async def test_chinese_realtime_benchmark(
        self,
        ws_url: str,
        ensure_server,
        audio_dir: Path,
        data_dir: Path,
    ):
        """Chinese FLEURS clip: real-time latency + WER/CER benchmark."""
        expected_dir = data_dir / "expected"
        wav, reference = _load_fleurs_sample(audio_dir, expected_dir, "chinese")

        if not wav.exists():
            pytest.skip(
                f"FLEURS audio not found: {wav}\n"
                "Run: python E2Etest/download_test_audio.py"
            )
        if not reference:
            pytest.skip(f"Reference transcript missing: {expected_dir / 'chinese_01.txt'}")

        audio, sr = sf.read(str(wav), dtype="int16")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)
        if sr != SAMPLE_RATE:
            new_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.int16)

        result = await _stream_and_time(ws_url, audio)

        wer = calculate_wer(reference, result["final_text"]) * 100
        cer = calculate_cer(reference, result["final_text"]) * 100
        stats = _latency_stats(result["chunk_latencies_ms"])

        _print_report(wav, "chinese", reference, result["final_text"], wer, cer, stats, result)
        _save_json(result, "chinese", wer, cer, stats)

        assert result["final_text"], "Expected non-empty transcription"
        assert cer <= 45.0, f"CER {cer:.1f}% exceeds 45% streaming threshold for Chinese"
        assert stats["median"] < 30000, (
            f"Median chunk latency {stats['median']:.0f}ms exceeds 30000ms"
        )
