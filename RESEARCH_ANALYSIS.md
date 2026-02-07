# Qwen3-ASR Server — Research Analysis

## Date: 2026-02-08

## Table of Contents

1. [Current Architecture Review](#1-current-architecture-review)
2. [Official Server Analyzed](#2-official-server-analyzed)
3. [Head-to-Head Comparison](#3-head-to-head-comparison)
4. [Model Landscape](#4-model-landscape)
5. [Gap Analysis](#5-gap-analysis)
6. [RAM Idle Problem](#6-ram-idle-problem)
7. [Recommendations (Prioritized)](#7-recommendations-prioritized)

---

## 1. Current Architecture Review

### What We Have

A single-file FastAPI server (`server.py`, ~544 lines) providing:
- OpenAI-compatible `/v1/audio/transcriptions` endpoint (file upload)
- SSE streaming endpoint `/v1/audio/transcriptions/stream`
- Real-time WebSocket endpoint `/ws/transcribe` with overlap buffering
- GPU idle unloading with configurable timeout
- Health check endpoint
- Audio preprocessing (mono, float32, 16kHz resample, peak normalization)

### Strengths of Current Implementation

| Feature | Implementation | Assessment |
|---------|---------------|------------|
| WebSocket real-time transcription | Overlap buffering (~300ms), configurable chunk size (~800ms), flush/reset/config commands | Well-designed for phone calls. Overlap prevents word splits at boundaries. |
| Concurrency safety | `asyncio.Semaphore(1)` + `run_in_executor` | Correct. Event loop never blocked. GPU access serialized. |
| Idle VRAM unload | Background watchdog task, double-checked locking pattern | Correct. Critical for shared GPU on Synology NAS. |
| GPU memory cleanup | `gc.collect()` + `torch.cuda.empty_cache()` + `ipc_collect()` after each transcription + on unload | Correct. Per-request cleanup prevents VRAM accumulation. |
| Request timeout | `asyncio.wait_for()` with configurable `REQUEST_TIMEOUT` (300s default) | Correct. Prevents hung requests. |
| Model load/unload locking | `asyncio.Lock()` with double-checked locking | Correct. Prevents race between load and unload. |
| `torch.inference_mode()` | Applied during transcription | Correct. Disables gradient tracking. |
| Audio preprocessing | Mono conversion, float32, 16kHz resample via librosa, peak normalization | Good. Handles various input formats gracefully. |
| WebSocket protocol | Binary audio + JSON control commands, connection confirmation with config | Well-designed. Client knows expected format on connect. |
| Docker config | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Good. Reduces CUDA memory fragmentation. |

### Weaknesses Identified

| Issue | Severity | Details |
|-------|----------|---------|
| No long audio chunking | **High** | For subtitle transcription, files >30s may degrade or OOM. No intelligent splitting at silence boundaries. |
| No repetition detection | **High** | ASR models hallucinate repeated text on noisy audio (common in phone calls). No post-processing to detect/fix this. |
| No `torch.compile` | Medium | Missing 20-30% inference speedup on repeated requests. |
| `sdpa` instead of `flash_attention_2` | Medium | ~20% slower attention computation. |
| 0.6B model | Medium | 1.7B is 35% more accurate on multilingual, significantly better on Chinese. |
| WebSocket uses semaphore per chunk | Medium | Each 800ms chunk acquires `_infer_semaphore`. During a phone call, an HTTP `/v1/audio/transcriptions` request must wait for the WebSocket chunk to finish. Consider separate semaphores or priority. |
| Python process RAM when idle | **High** | ~1.9GB resident RAM even with model unloaded, due to PyTorch + CUDA runtime. |
| `asyncio.get_event_loop()` deprecated | Low | Should use `asyncio.get_running_loop()` (already used in TTS server). |
| No `lifespan` context manager | Low | Uses deprecated `@app.on_event("startup")`. |
| SSE streaming is fake | Low | `sse_transcribe_generator` checks for `transcribe_stream`/`stream_transcribe` methods that don't exist on the model, then falls back to full transcription. Always falls through to non-streaming path. |
| Librosa import inside function | Low | `import librosa` inside `preprocess_audio()` on every call. Should be top-level with graceful fallback. |
| No Dockerfile thread limiting | Low | Unlike TTS Dockerfile, ASR Dockerfile doesn't set `OMP_NUM_THREADS` or `MKL_NUM_THREADS`. Results in more thread spawning. |

---

## 2. Official Server Analyzed

### QwenLM/Qwen3-ASR (Official)

**Architecture**: The official "server" is a **45-line wrapper** around `vllm serve`. It:
1. Registers the custom Qwen3-ASR model into HuggingFace's AutoConfig/AutoModel/AutoProcessor
2. Registers the custom vLLM model class into vLLM's ModelRegistry
3. Calls `vllm serve` (injects `"serve"` into `sys.argv`)

**All serving is delegated to vLLM**. Zero custom HTTP routing.

Usage:
```bash
pip install qwen-asr[vllm]
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

### What vLLM Provides (That We Don't Have)

| Feature | vLLM Implementation |
|---------|-------------------|
| Continuous batching | Automatic request scheduling, multiple concurrent requests processed together |
| PagedAttention | Efficient KV cache management, configurable `--gpu-memory-utilization` |
| OpenAI-compatible API | `/v1/chat/completions` + `/v1/audio/transcriptions` |
| SSE streaming | Real streaming via token-by-token generation |
| Tensor parallelism | Multi-GPU support via ColumnParallel/RowParallel layers |
| Pipeline parallelism | Cross-GPU pipeline support |
| Automatic batching | Sub-batch processing with configurable `max_inference_batch_size` |

### What the Official SDK Provides (That We Don't Have)

| Feature | Implementation |
|---------|---------------|
| Long audio chunking | `split_audio_into_chunks()` — splits at low-energy boundaries using sliding window energy estimation. Guarantees exact sample-level reconstruction. Pads too-short tail chunks. |
| Repetition detection | `detect_and_fix_repetitions()` — detects and removes hallucinated repeated characters/patterns in model output. |
| 30 language validation | Validates language codes against supported list, auto-detect fallback. |
| Audio input normalization | Accepts path, URL, base64, numpy array. Auto-resample to 16kHz mono. |
| Forced alignment | Optional `Qwen3ForcedAligner` for word-level timestamps (separate model). |
| Streaming transcription | Prefix rollback strategy — last K tokens rolled back to reduce boundary jitter. Configurable `unfixed_chunk_num` and `unfixed_token_num`. |
| Max audio length | Configurable limits: 1200s (20 min) for ASR, 180s (3 min) for forced alignment. |

### Official Streaming Demo (Flask-Based)

A separate Flask app (`demo_streaming.py`, ~506 lines) provides real-time microphone transcription:
- Session management with UUID-based IDs, 10-minute TTL, garbage collection
- `/api/start`, `/api/chunk`, `/api/finish` HTTP polling (not WebSocket)
- Uses vLLM offline mode (not the vLLM server)
- 500ms audio chunks, 16kHz PCM

**Our WebSocket approach is architecturally superior** to their HTTP polling approach for real-time phone calls. WebSocket has lower overhead and true bidirectional communication.

---

## 3. Head-to-Head Comparison

| Feature | Our Server | Official (vLLM) |
|---------|-----------|-----------------|
| **Real-time streaming** | WebSocket with overlap buffering | HTTP polling (Flask demo) or SSE (vLLM server) |
| **Phone call suitability** | Excellent (WebSocket, low latency) | Poor (HTTP polling) or OK (SSE, higher overhead) |
| **Concurrency** | Semaphore(1) — one at a time | vLLM continuous batching — many concurrent |
| **GPU memory** | Manual cleanup per request + idle unload | vLLM PagedAttention — automatic |
| **Idle VRAM unload** | Yes (watchdog) | No — model always loaded |
| **Long audio** | No chunking — caller must split | Auto-chunk at silence boundaries, up to 20 min |
| **Repetition handling** | None | `detect_and_fix_repetitions()` post-processing |
| **Batch processing** | One file at a time | Auto-batching with sub-batch size control |
| **Forced alignment** | No | Yes (separate Qwen3-ForcedAligner model) |
| **Language support** | Pass-through, no validation | 30 languages validated |
| **Audio input** | File upload only | File, URL, base64, numpy |
| **Model sizes** | 0.6B only | 0.6B and 1.7B |
| **Dependencies** | Lightweight (FastAPI + qwen-asr) | Heavy (vLLM ~2GB install) |
| **Tensor parallelism** | No | Yes (multi-GPU) |
| **Request timeout** | Yes (300s) | Configurable via vLLM |
| **Event loop safety** | Correct (`run_in_executor`) | Handled by vLLM async engine |
| **Custom deployment** | Single file, easy to modify | Black box (vLLM internals) |

### Verdict

**For phone calls (WebSocket real-time)**: Our server wins. The official server has no WebSocket support. Their streaming demo uses HTTP polling which has higher latency.

**For subtitle transcription (batch files)**: The official server wins. Long audio chunking, repetition detection, batch processing, and concurrent request handling are all critical for processing many files.

**We need to combine both strengths**: Keep our WebSocket architecture for phone calls, add the official SDK's utilities (chunking, repetition detection) for subtitle transcription.

---

## 4. Model Landscape

### Qwen3-ASR Family (Released 2026-01-29, Apache 2.0)

| Model | Parameters | Purpose |
|-------|-----------|---------|
| **Qwen3-ASR-1.7B** | 1.7B | Flagship. SOTA among open-source ASR. |
| **Qwen3-ASR-0.6B** | 0.6B | **Currently deployed.** Best accuracy-efficiency tradeoff. |
| Qwen3-ForcedAligner-0.6B | 0.6B | Timestamp prediction for 11 languages, up to 5 min. |
| Qwen3-ASR-Flash | API-only | Context-aware transcription, hotword biasing. No public weights. |

### 0.6B vs 1.7B Performance (WER % — Lower is Better)

| Benchmark | 0.6B | 1.7B | Improvement |
|-----------|------|------|-------------|
| LibriSpeech clean | 2.11 | 1.63 | 23% better |
| LibriSpeech other | 4.55 | 3.38 | 26% better |
| CommonVoice English | 9.92 | 7.39 | 25% better |
| WenetSpeech (Chinese) | 5.97 / 6.88 | 4.97 / 5.88 | 15-17% better |
| AISHELL-2 (Chinese) | 3.15 | 2.71 | 14% better |
| Fleurs (30 languages avg) | 7.57 | 4.90 | **35% better** |
| Language ID accuracy | 96.8% | 97.9% | +1.1 points |

**The 1.7B is dramatically better on multilingual content** (35% improvement). Critical for subtitle transcription of non-English content.

### Competing Open-Source ASR Models

| Model | Params | Strengths | Weaknesses |
|-------|--------|-----------|------------|
| **Qwen3-ASR-1.7B** | 1.7B | SOTA multilingual, 30 languages, singing voice | Newer, less ecosystem |
| **Whisper large-v3** | 1.55B | Mature ecosystem, well-tested, broad language support | Lower accuracy than Qwen3-ASR-1.7B on most benchmarks |
| **NVIDIA Canary-Qwen 2.5B** | 2.5B | Best English-only ASR (5.63% WER on HF leaderboard) | English only |
| **faster-whisper** | 1.55B | CTranslate2 optimized, 4x faster than Whisper | Same accuracy as Whisper |
| **FunASR-MLT-Nano** | Small | Lightweight, fast | Lower accuracy |

### Recommendation

Upgrade to **Qwen3-ASR-1.7B** for both phone calls and subtitle transcription. The 35% multilingual improvement is significant.

### Vs. Our Whisper Engine

We also run a separate faster-whisper container (whisper-engine) on port 8001. With Qwen3-ASR-1.7B, the Whisper engine becomes **redundant** for most use cases — Qwen3-ASR-1.7B matches or beats Whisper large-v3 on nearly every benchmark. Consider retiring the Whisper engine to free resources.

---

## 5. Gap Analysis

### Critical Gaps (For Phone Call + Subtitle Use Case)

#### Gap 1: No Long Audio Chunking

**Impact**: Subtitle transcription of TV episodes, movies, or long recordings (>30s) will either OOM, produce degraded results, or require the caller to pre-split audio.

**What's needed**: The official SDK's `split_audio_into_chunks()` approach — split at low-energy boundaries using sliding window energy estimation. Guarantees exact sample-level reconstruction.

**Key parameters from official implementation**:
- `MAX_ASR_INPUT_SECONDS = 1200` (20 minutes max)
- `MIN_ASR_INPUT_SECONDS = 0.5` (minimum chunk length)
- Sliding window energy convolution for boundary detection
- Pad too-short tail chunks

**Complexity**: Medium (~50-80 lines to port from official SDK).

#### Gap 2: No Repetition Detection

**Impact**: ASR models hallucinate repeated text, especially on:
- Noisy phone call audio
- Music/background noise in videos
- Long silence periods

Without detection, subtitles contain garbage like "the the the the the the" or repeated sentences.

**What's needed**: The official SDK's `detect_and_fix_repetitions()` — character-level and pattern-level repetition detection with configurable thresholds.

**Complexity**: Low (~30 lines to port from official SDK).

#### Gap 3: Python Process RAM When Idle

**Impact**: ~1.9GB resident RAM per container even with model unloaded. With ASR + TTS + Whisper containers, that's ~5.7GB wasted on a NAS.

**Root cause**: `import torch` + CUDA context = ~1-1.2GB CPU RAM permanently allocated. Cannot be freed without killing the process.

**What's needed**: Gateway + Worker architecture (see Section 6).

**Complexity**: Medium.

### Medium Gaps

#### Gap 4: No `torch.compile`

**Impact**: Missing 20-30% inference speedup on repeated requests. Matters for both phone call latency and subtitle batch throughput.

**Fix**: Add `torch.compile(model.model, mode="reduce-overhead", fullgraph=False)` after model loading.

**Complexity**: Low (3 lines), needs validation.

#### Gap 5: `sdpa` Instead of `flash_attention_2`

**Impact**: ~20% slower attention computation.

**Fix**: Change `attn_implementation="sdpa"` to `attn_implementation="flash_attention_2"`. Install `flash-attn` in Docker image.

**Complexity**: Low.

#### Gap 6: WebSocket Semaphore Contention

**Impact**: During a phone call (WebSocket), each ~800ms chunk acquires `_infer_semaphore`. If an HTTP file transcription request arrives simultaneously, it must wait for the current WebSocket chunk to finish. And vice versa — a long file transcription blocks WebSocket chunks.

**What's needed**: Separate handling — either priority-based scheduling (WebSocket gets priority) or separate semaphores for real-time vs batch workloads.

**Complexity**: Medium.

#### Gap 7: SSE Streaming is Fake

**Impact**: The `/v1/audio/transcriptions/stream` endpoint always falls through to non-streaming path because the model doesn't have `transcribe_stream` or `stream_transcribe` methods. The endpoint works but provides no streaming benefit.

**Options**:
1. Remove the endpoint (it's misleading)
2. Implement real streaming using the official SDK's `streaming_transcribe()` + `finish_streaming_transcribe()` methods (vLLM backend only)
3. Implement chunked streaming — split audio into segments, transcribe each, yield results progressively

**Complexity**: Option 1 is trivial. Option 3 is medium.

### Low Gaps

#### Gap 8: Model Size

**Fix**: Change `MODEL_ID` env var. Verify VRAM.

#### Gap 9: Missing Dockerfile Thread Limits

**Fix**: Add `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` to Dockerfile (already present in TTS Dockerfile).

#### Gap 10: Deprecated APIs

**Fix**: Replace `asyncio.get_event_loop()` with `asyncio.get_running_loop()`. Replace `@app.on_event("startup")` with `lifespan` context manager.

---

## 6. RAM Idle Problem

### Current State

```
Container running, model loaded:    ~1.9GB RAM + ~1GB VRAM
Container running, model unloaded:  ~1.9GB RAM + ~0GB VRAM   ← RAM not freed
Container stopped:                  ~0GB RAM   + ~0GB VRAM
```

### Breakdown of ~1.9GB Idle RAM

| Component | Approx RAM | Can Free? |
|-----------|-----------|-----------|
| CUDA runtime + driver context | 800MB-1.2GB | Only by killing process |
| PyTorch library (libtorch, etc.) | 300-400MB | Only by killing process |
| Python interpreter + all imports | 100-200MB | Only by killing process |
| uvicorn + FastAPI framework | 20-30MB | Keep this |

### Proposed Solution: Gateway + Worker Architecture

```
                    ┌─────────────────────────────────┐
                    │  Lightweight Gateway (30MB RAM)  │
                    │  - FastAPI + uvicorn             │
                    │  - Health check endpoint         │
                    │  - WebSocket proxy               │
                    │  - Spawns/kills worker process   │
                    └───────────────┬──────────────────┘
                                    │
                          (spawn on first request,
                           kill after idle timeout)
                                    │
                    ┌───────────────▼──────────────────┐
                    │  Heavy Worker (1.9GB RAM)         │
                    │  - import torch, load model      │
                    │  - GPU inference                  │
                    │  - Communicates via IPC/socket    │
                    └──────────────────────────────────┘
```

**Idle state**: Only gateway running → ~30MB RAM, ~0 VRAM
**Active state**: Gateway + worker → ~1.9GB RAM, ~1GB VRAM
**Cold start penalty**: ~5-10s to spawn worker + load model

### WebSocket Considerations for Gateway

For phone calls, the gateway must handle WebSocket connections directly and proxy audio to the worker. When the worker is not running:
1. Accept WebSocket connection
2. Send "warming up" status to client
3. Spawn worker process
4. Wait for worker ready signal
5. Begin proxying audio chunks to worker, results back to client

When the worker idle times out during an active WebSocket connection, the watchdog must NOT kill it.

---

## 7. Recommendations (Prioritized)

Priority is ordered by impact for the primary use case: **2-way phone calls via WebSocket + subtitle transcription**.

### P0 — Critical

| # | Improvement | Impact | Effort | Details |
|---|------------|--------|--------|---------|
| 1 | **Add long audio chunking** | Enables subtitle transcription of full episodes/movies without OOM or degradation | Medium | Port `split_audio_into_chunks()` from official SDK. Split at low-energy silence boundaries. |
| 2 | **Add repetition detection** | Clean subtitles and phone call transcriptions. Removes hallucinated loops. | Low | Port `detect_and_fix_repetitions()` from official SDK. Apply as post-processing on all transcription results. |
| 3 | **Gateway + Worker architecture** | Reclaim ~5.6GB idle RAM across services on Synology NAS | Medium | Split into lightweight gateway + heavy worker subprocess. Kill worker after idle, respawn on demand. Proxy WebSocket connections. |

### P1 — High

| # | Improvement | Impact | Effort | Details |
|---|------------|--------|--------|---------|
| 4 | **Upgrade to 1.7B** | 35% better multilingual accuracy. Critical for non-English subtitles. | Config change | Change `MODEL_ID` env var. Verify VRAM. |
| 5 | **Add `torch.compile`** | 20-30% faster inference for both phone calls and subtitle batch processing | Low (3 lines) | `torch.compile(model.model, mode="reduce-overhead")`. Validate. |
| 6 | **Switch to `flash_attention_2`** | ~20% faster attention | Low (1 line + Docker) | Requires `flash-attn` package. |
| 7 | **Add Dockerfile thread limits** | Reduce idle thread count and CPU contention | Trivial | Add `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` to Dockerfile (match TTS). |

### P2 — Medium

| # | Improvement | Impact | Effort | Details |
|---|------------|--------|--------|---------|
| 8 | **Fix WebSocket semaphore contention** | Prevents phone call audio chunks from being blocked by simultaneous file transcription | Medium | Priority scheduling or separate semaphore pools. |
| 9 | **Fix or remove fake SSE endpoint** | Remove misleading endpoint or implement real chunked streaming | Low-Medium | Either delete `/v1/audio/transcriptions/stream` or implement chunked progressive transcription. |
| 10 | **Use official SDK streaming for WebSocket** | Better accuracy at chunk boundaries via prefix rollback strategy | Medium | Replace current overlap buffering with official `streaming_transcribe()` + `finish_streaming_transcribe()`. Uses accumulated audio + prefix rollback instead of fixed overlap window. |

### P3 — Low

| # | Improvement | Impact | Effort | Details |
|---|------------|--------|--------|---------|
| 11 | **Fix deprecated APIs** | Future-proofing | Low | `get_event_loop()` → `get_running_loop()`, `@app.on_event` → `lifespan`. |
| 12 | **Add process title** | Easier identification in htop | Trivial | `setproctitle("qwen3-asr")` |
| 13 | **Consider retiring whisper-engine** | Free NAS resources | Decision | Qwen3-ASR-1.7B matches or beats Whisper large-v3 on most benchmarks. One less container to maintain. |

---

## Appendix A: Official SDK Key Utilities Worth Porting

### `split_audio_into_chunks()` (from `qwen_asr/inference/utils.py`)

- Splits long audio at low-energy boundaries
- Uses sliding window energy estimation via convolution
- Guarantees exact sample-level reconstruction (no gaps, no overlaps)
- Configurable target chunk length
- Pads too-short tail chunks to minimum duration
- Constants: `MAX_ASR_INPUT_SECONDS = 1200`, `MIN_ASR_INPUT_SECONDS = 0.5`

### `detect_and_fix_repetitions()` (from `qwen_asr/inference/utils.py`)

- Detects hallucinated character repetitions
- Detects hallucinated pattern/phrase repetitions
- Configurable thresholds
- Applied as post-processing on raw model output

### `streaming_transcribe()` (from `qwen_asr/inference/qwen3_asr.py`)

- Prefix rollback strategy for streaming
- For first N chunks (`unfixed_chunk_num`), no prefix prepended
- After that, last K tokens (`unfixed_token_num`) are rolled back each chunk
- Audio accumulated from start and re-fed entirely each chunk
- Configurable chunk size in seconds (default 2.0s)

## Appendix B: Community Projects

### predict-woo/qwen3-asr.cpp
- GitHub: https://github.com/predict-woo/qwen3-asr.cpp
- Pure C++ GGML implementation of Qwen3-ASR-0.6B
- Q8_0 quantization: 1.8GB → 1.3GB
- ~18s for 30s audio on 4 CPU threads
- No Python runtime needed
- Potentially useful for edge deployment

### Official References
- Qwen3-ASR Technical Report: https://arxiv.org/abs/2601.21337
- HuggingFace Collection: https://huggingface.co/collections/Qwen/qwen3-asr
- Official SDK: https://github.com/QwenLM/Qwen3-ASR
- vLLM Serving Docs: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html
