# Design: Reduce Idle RAM + Remove SDK Redundancies (Issue #82)

**Date:** 2026-02-21
**Issue:** #82 — Reduce idle RAM from 2.4GB to <100MB by deferring heavy imports

## Principle

Our code optimizes; the SDK transcribes. If `model.transcribe()` already does it, we delete our version.

## Problem

1. Container uses 2.4GB RAM when idle because `server.py` imports `torch`, `qwen_asr`, `soundfile` at module top-level
2. `torch.compile` spawns ~20 inductor worker subprocesses (~300MB each)
3. server.py reimplements preprocessing, chunking, and processor loading that the SDK already handles internally

## Changes

### Remove (SDK handles natively)

| Code | Reason |
|---|---|
| `preprocess_audio()` | SDK's `normalize_audios()` inside `transcribe()` does mono/resample/normalize |
| `preprocess_audio_ws()` | WS audio is already 16kHz mono float32 |
| `chunk_audio_at_silence()` | SDK's `split_audio_into_chunks()` inside `transcribe()` is superior (sliding window, +/-5s search, up to 20min) |
| Manual chunking loop in HTTP endpoint | SDK handles chunking internally |
| Separate `processor = AutoProcessor.from_pretrained()` | SDK loads processor internally in `Qwen3ASRModel.from_pretrained()` |
| `from qwen_asr.inference.qwen3_asr import AutoProcessor` | No longer needed |
| `_ATTN_IMPL` module-level evaluation | Defer to `_load_model_sync()` |

### Keep (not in SDK)

| Code | Reason |
|---|---|
| Priority inference queue | SDK has no WS vs HTTP priority concept |
| WebSocket streaming + overlap | SDK streaming is vLLM-only |
| VAD gating (Silero) | SDK has no VAD |
| Idle model unloading + watchdog | Server lifecycle |
| `release_gpu_memory()` | GPU memory management |
| Pinned memory + CUDA streams | Below-SDK hardware optimization |
| TRT/ONNX encoder routing | Below-SDK hardware optimization |
| torch.compile | Below-SDK hardware optimization |
| Speculative decoding (0.6B/1.7B) | Not in SDK |
| Dual model for WS partials | Not in SDK |
| Word-level `detect_and_fix_repetitions()` | SDK only does character-level (threshold=20) |
| Gateway mode | Server architecture |

### Lazy Imports (Original #82 scope)

- Move `import torch`, `import soundfile`, `from qwen_asr import ...` inside `_load_model_sync()` and functions that use them
- `from __future__ import annotations` for string type hints
- Move `_PINNED_AUDIO_BUFFER`, `_cuda_stream` init into `_load_model_sync()`
- Add `TORCHINDUCTOR_COMPILE_THREADS=1` to Dockerfile ENV

### Simplified HTTP Endpoint

Before (~40 lines): `sf.read` -> `preprocess_audio` -> `chunk_audio_at_silence` -> loop chunks -> `_do_transcribe` per chunk -> join -> `detect_and_fix_repetitions`

After (~5 lines): `sf.read` -> `_do_transcribe(audio, sr, ...)` -> `detect_and_fix_repetitions`

## Expected Outcome

- Idle RAM: ~2.4GB -> ~50-100MB
- Lines removed: ~80-100 lines of redundant preprocessing/chunking
- Behavior: identical or better (SDK's chunking is superior)
- Cold start: +3-5s on first request (one-time import cost)
