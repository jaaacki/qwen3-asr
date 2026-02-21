# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dockerized FastAPI server wrapping the Qwen3-ASR speech-to-text model with OpenAI-compatible REST endpoints and real-time WebSocket transcription. GPU-accelerated with on-demand model loading and idle auto-unload.

## Common Commands

```bash
# Start the server (builds if needed)
docker compose up -d

# Rebuild after code changes
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down

# Health check
curl http://localhost:8100/health

# Test transcription
curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"

# SSE streaming transcription
curl -X POST http://localhost:8100/v1/audio/transcriptions/stream -F "file=@audio.wav"

# Debug audio processing locally
python src/debug_audio.py
```

### E2E Tests

Tests run against the live container (must be running on port 8100). Install deps first: `pip install -r E2Etest/requirements.txt`

```bash
# Run all tests (server must be running)
pytest E2Etest/ -v

# Via helper script (can auto-start server)
bash E2Etest/run_tests.sh --with-server

# Filter by category
pytest E2Etest/ -k http          # HTTP API tests only
pytest E2Etest/ -k websocket     # WebSocket tests only
pytest E2Etest/ -m performance   # Performance benchmarks
pytest E2Etest/ -m "not slow"    # Skip slow tests

# Single test file
pytest E2Etest/test_api_http.py -v

# Single test
pytest E2Etest/test_api_http.py::TestHealthEndpoint::test_health_returns_ok -v
```

Test markers: `smoke`, `slow`, `performance`, `websocket`, `integration`, `accuracy`, `requires_gpu`. Timeout is 300s per test. Async mode is auto (pytest-asyncio).

## Architecture

### File Organization

- `src/server.py` — Core FastAPI server with inference logic, priority queue, WebSocket handling (~1170 lines)
- `src/gateway.py` — Gateway proxy mode (GATEWAY_MODE=true); routes to worker subprocess
- `src/worker.py` — Inference worker for gateway mode; imports logic from server.py
- `src/export_onnx.py` — Export encoder to ONNX for ORT acceleration
- `src/build_trt.py` — Build TensorRT engine for encoder
- `E2Etest/` — pytest-based E2E test suite (test_api_http, test_websocket, test_performance, test_integration, test_accuracy)

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check, returns model status and config |
| `/v1/audio/transcriptions` | POST | File upload transcription (OpenAI-compatible) |
| `/v1/audio/transcriptions/stream` | POST | SSE streaming transcription (chunks long audio at silence boundaries) |
| `/ws/transcribe` | WebSocket | Real-time streaming with raw PCM input |

### Concurrency Model

**PriorityInferQueue** (replaces simple semaphore):
- Min-heap priority queue with `priority: int` (lower = higher priority)
- WebSocket requests use priority=0, HTTP uploads use priority=1
- This prevents long file uploads from blocking real-time WebSocket transcription
- Single dedicated ThreadPoolExecutor (`max_workers=1`) for all GPU inference
- All inference runs via `run_in_executor()` to keep async event loop unblocked

### Model Lifecycle

- Model loads on first request (not at startup in standalone mode)
- `_idle_watchdog()` background task unloads after `IDLE_TIMEOUT` seconds (default 120s, 0 = disabled)
- `asyncio.Lock()` prevents load/unload race conditions
- GPU memory explicitly released via `release_gpu_memory()` after operations

**Gateway Mode** (`GATEWAY_MODE=true`):
- Splits into gateway (port 8000) + worker subprocess (port 8001)
- Gateway proxies all requests to worker via HTTP/WebSocket
- Killing worker process reclaims ALL RAM/VRAM (useful for memory leak scenarios)

### WebSocket Real-Time Transcription (`/ws/transcribe`)

- Accepts raw PCM: 16-bit little-endian, 16kHz, mono
- Buffers ~450ms of audio (`WS_BUFFER_SIZE=14400` bytes)
- **Overlap**: Last 150ms of each chunk prepended to next chunk (`WS_OVERLAP_SIZE=4800`)
- **Silence padding**: 600ms silence appended on `flush` command to commit trailing words (`WS_FLUSH_SILENCE_MS`)
- **VAD gating**: Silero VAD skips inference for silent frames (no GPU usage for silence)
- **Dual-model**: If `DUAL_MODEL=true`, uses 0.6B for partials, 1.7B for final transcription
- Control messages: `flush`, `reset`, `config` (set language)
- Buffer transcribed on disconnect (no audio loss)

### Audio Preprocessing Pipeline

Input audio → mono conversion → float32 normalization → resample to 16kHz (torchaudio) → peak normalize to [-1, 1]

WebSocket fast path skips resampling (audio already at 16kHz).

### Optimizations (Opt-in)

All Phase 3 features are gated behind environment variables — safe to experiment without breaking defaults.

| Feature | Env Var | Description |
|---------|---------|-------------|
| Flash Attention 2 | auto-detected | Falls back to SDPA if unavailable |
| torch.compile | always on | `mode="reduce-overhead"` for repeated inference |
| Pinned memory | auto | Pre-allocated 30s buffer for fast CPU→GPU transfer |
| CUDA streams | auto | Async DMA pipeline for transfer/compute overlap |
| INT8 quantization | `QUANTIZE=int8` | bitsandbytes W8A8 (~50% VRAM reduction) |
| FP8 quantization | `QUANTIZE=fp8` | torchao (requires sm_89+ Hopper/Ada) |
| Speculative decoding | `USE_SPECULATIVE=true` | 0.6B draft + 1.7B verifier (~2x speed) |
| ONNX encoder | `ONNX_ENCODER_PATH` | ORT-accelerated encoder forward pass |
| TensorRT encoder | `TRT_ENCODER_PATH` | Compiled TRT engine for encoder |
| vLLM backend | `USE_VLLM=true` | PagedAttention serving engine |
| CUDA Graphs warmup | `USE_CUDA_GRAPHS=true` | 3 extra warmup passes for kernel caching |
| NUMA CPU pinning | `NUMA_NODE=0` | Pin to GPU-collocated NUMA node |
| Granian ASGI | `USE_GRANIAN=true` | Rust-based ASGI server (alternative to uvicorn) |

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model (0.6B for speed, 1.7B for accuracy) |
| `FAST_MODEL_ID` | `Qwen/Qwen3-ASR-0.6B` | Draft/partial model for speculative/DUAL_MODE |
| `IDLE_TIMEOUT` | `120` | Seconds before model unloads (0 = keep loaded) |
| `REQUEST_TIMEOUT` | `300` | Max inference time per request |
| `WS_BUFFER_SIZE` | `14400` | WebSocket audio buffer (~450ms at 16kHz) |
| `WS_OVERLAP_SIZE` | `4800` | Overlap between chunks (~150ms) |
| `WS_FLUSH_SILENCE_MS` | `600` | Silence padding on flush (ms) |
| `GATEWAY_MODE` | `false` | Run as gateway+worker split |
| `DUAL_MODEL` | `false` | Load both 0.6B and 1.7B models |
| `QUANTIZE` | `""` | `int8` or `fp8` |

Port mapping: container 8000 → host 8100.

### Docker Build

Base image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`. The `devel` variant is required because `flash-attn` builds from source and needs `nvcc`. HuggingFace model cache is persisted via `./models` volume mount.

## Docs

- `docs/WEBSOCKET_USAGE.md` — WebSocket protocol, connection format, example Python client
- `docs/GRANIAN_BENCHMARK.md` — Performance comparison of ASGI servers
- `RESEARCH_ANALYSIS.md` — Architecture comparison with official Qwen3-ASR SDK and vLLM backend
- `improvements.md` — Prioritized optimization recommendations (includes WebSocket critical path latency analysis)
- `ROADMAP.md` — Milestone planning (3 phases, all completed)
- `CHANGELOG.md` — Version history
- `LEARNING_LOG.md` — Technical learnings and decisions
