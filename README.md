# Qwen3-ASR Docker Server

A Dockerized REST API server for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Automatic Speech Recognition) with GPU acceleration and OpenAI-compatible endpoints.

Wraps Qwen3-ASR-1.7B in a production-ready FastAPI server with real-time WebSocket transcription, priority scheduling, and opt-in hardware acceleration.

## Features

### Core
- OpenAI-compatible `/v1/audio/transcriptions` endpoint (WAV, MP3, FLAC, …)
- Server-Sent Events streaming via `/v1/audio/transcriptions/stream`
- Real-time WebSocket transcription via `/ws/transcribe` with overlap and silence padding
- SRT subtitle generation via `/v1/audio/subtitles` with accurate (ForcedAligner) and fast (heuristic) modes
- Multi-language support with auto-detection
- On-demand model loading with idle auto-unload (0 VRAM when idle)

### Performance (v0.6.0)
- **Priority scheduling** — WS requests (priority 0) preempt HTTP uploads via min-heap queue
- **Silero VAD gating** — silent frames skipped, GPU never invoked for silence
- **Long audio chunking** — files >25s split at silence boundaries for progressive SSE output
- **Dual-model strategy** — 0.6B for WS partials, 1.7B for final transcription (`DUAL_MODEL=true`)
- **KV-cache reuse** across WebSocket chunks
- **Flash Attention 2** with graceful SDPA fallback
- **torch.compile** inference with reduce-overhead mode
- **Pinned memory + CUDA stream** — async DMA pipeline for transfer/compute overlap

### Quantization & Hardware Acceleration (opt-in)
- **INT8 W8A8** via bitsandbytes (`QUANTIZE=int8`) — ~50% VRAM reduction
- **FP8** via torchao (`QUANTIZE=fp8`) — requires sm_89+ (Hopper/Ada Lovelace)
- **ONNX Runtime encoder** (`ONNX_ENCODER_PATH`) — ORT-accelerated encoder with CUDA EP
- **TensorRT encoder** (`TRT_ENCODER_PATH`) — compiled TRT engine for encoder forward pass
- **CUDA Graphs warming** (`USE_CUDA_GRAPHS=true`) — 3 extra warmup passes for kernel caching
- **Speculative decoding** (`USE_SPECULATIVE=true`) — 0.6B draft + 1.7B verifier (~2x speed)
- **vLLM backend** (`USE_VLLM=true`) — PagedAttention serving engine
- **Causal attention encoder** (`USE_CAUSAL_ENCODER=true`) — EXPERIMENTAL incremental encoding

### Infrastructure (opt-in)
- **Gateway + Worker mode** (`GATEWAY_MODE=true`) — splits into proxy + model worker; kill worker to reclaim all VRAM
- **NUMA-aware CPU pinning** (`NUMA_NODE=0`) — pins to GPU-collocated NUMA node
- **Granian ASGI** (`USE_GRANIAN=true`) — Rust-based alternative to uvicorn

## Quick Start

```bash
# Build and start
docker compose up -d

# Check health
curl http://localhost:8100/health

# Transcribe audio file
curl -X POST http://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav"

# Streaming transcription (SSE)
curl -X POST http://localhost:8100/v1/audio/transcriptions/stream \
  -F "file=@audio.wav" -N
```

For WebSocket real-time transcription, see [docs/WEBSOCKET_USAGE.md](docs/WEBSOCKET_USAGE.md).

## Requirements

- Docker with NVIDIA GPU support (`nvidia-docker2` or Docker 19.03+)
- NVIDIA GPU with CUDA 12.4 compatible driver
- ~4GB disk for model download (cached in `./models/`)

## API Endpoints

### `GET /health`

Returns server status, model info, and GPU memory.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "Qwen/Qwen3-ASR-1.7B",
  "cuda": true
}
```

### `POST /v1/audio/transcriptions`

Transcribe an audio file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (WAV, MP3, FLAC, …) |
| `language` | string | `"auto"` | Language code or `"auto"` |
| `return_timestamps` | bool | `false` | Word-level timestamps |

```bash
curl -X POST http://localhost:8100/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "language=en" \
  -F "return_timestamps=true"
```

### `POST /v1/audio/transcriptions/stream`

Same parameters, returns chunked SSE stream. Long audio (>25s) is split at silence boundaries for progressive output.

### `POST /v1/audio/subtitles`

Generate SRT subtitle file from audio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (WAV, MP3, FLAC, ...) |
| `language` | string | `"auto"` | Language code or `"auto"` |
| `mode` | string | `"accurate"` | `"accurate"` (ForcedAligner) or `"fast"` (heuristic) |
| `max_line_chars` | int | `42` | Maximum characters per subtitle line |

```bash
# Fast mode (no aligner needed, instant)
curl -X POST http://localhost:8100/v1/audio/subtitles \
  -F "file=@recording.wav" \
  -F "mode=fast" \
  -o subtitles.srt

# Accurate mode (loads ForcedAligner on first call, ~33ms word accuracy)
curl -X POST http://localhost:8100/v1/audio/subtitles \
  -F "file=@recording.wav" \
  -F "mode=accurate" \
  -o subtitles.srt
```

### `WS /ws/transcribe`

Real-time WebSocket transcription. See [docs/WEBSOCKET_USAGE.md](docs/WEBSOCKET_USAGE.md).

## Configuration

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID |
| `IDLE_TIMEOUT` | `120` | Seconds before unloading model (0 = keep loaded) |
| `REQUEST_TIMEOUT` | `300` | Max seconds per inference request |
| `WS_BUFFER_SIZE` | `14400` | WebSocket buffer bytes (~450ms at 16kHz) |
| `WS_OVERLAP_SIZE` | `4800` | Overlap bytes between chunks (~150ms) |
| `WS_FLUSH_SILENCE_MS` | `600` | Silence padding on flush (ms) |
| `FORCED_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | HuggingFace model for subtitle alignment |

### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `DUAL_MODEL` | `false` | Load 0.6B for WS partials alongside main model |
| `FAST_MODEL_ID` | `Qwen/Qwen3-ASR-0.6B` | Draft/partial model ID |
| `QUANTIZE` | `""` | `int8` or `fp8` quantization |
| `USE_CUDA_GRAPHS` | `false` | Extra warmup passes for CUDA kernel caching |
| `ONNX_ENCODER_PATH` | `""` | Path to exported ONNX encoder |
| `TRT_ENCODER_PATH` | `""` | Path to compiled TRT engine |
| `USE_SPECULATIVE` | `false` | Speculative decoding (0.6B draft + 1.7B verifier) |
| `USE_VLLM` | `false` | Use vLLM engine (requires vllm package) |
| `USE_CAUSAL_ENCODER` | `false` | EXPERIMENTAL causal attention patching |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_MODE` | `false` | Run as gateway proxy (requires worker on port 8001) |
| `NUMA_NODE` | `0` | NUMA node for CPU affinity (0 = first half of CPUs) |
| `USE_GRANIAN` | `false` | Use Granian ASGI instead of uvicorn |

## Optional Tools

```bash
# Export encoder to ONNX (then set ONNX_ENCODER_PATH)
python src/export_onnx.py --output models/encoder.onnx

# Build TensorRT engine (then set TRT_ENCODER_PATH)
python src/build_trt.py --output models/encoder.trt
```

## Project Structure

```
qwen3-asr/
├── compose.yaml          # Docker Compose configuration
├── Dockerfile            # Container build definition
├── src/
│   ├── server.py         # FastAPI server (~1100 lines)
│   ├── subtitle.py       # Subtitle generation (aligner, segmentation, SRT)
│   ├── gateway.py        # Gateway proxy (GATEWAY_MODE=true)
│   ├── worker.py         # Inference worker subprocess
│   ├── export_onnx.py    # Export encoder to ONNX Runtime
│   ├── build_trt.py      # Build TensorRT encoder engine
│   └── server_test.py    # Manual test notes
├── docs/
│   ├── WEBSOCKET_USAGE.md
│   └── GRANIAN_BENCHMARK.md
├── models/               # HuggingFace model cache (auto-populated)
├── CHANGELOG.md
├── ROADMAP.md
└── LEARNING_LOG.md
```

## License

Apache 2.0
