# Qwen3-ASR Docker Server

A Dockerized REST API server for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Automatic Speech Recognition) with GPU acceleration and OpenAI-compatible endpoints.

Wraps Qwen3-ASR-1.7B in a production-ready FastAPI server with real-time WebSocket transcription, SRT subtitle generation, translation, priority scheduling, and opt-in hardware acceleration.

## Features

### Speech-to-Text
- **OpenAI-compatible transcription** — `POST /v1/audio/transcriptions` accepts WAV, MP3, FLAC, OGG, and more
- **SSE streaming** — `POST /v1/audio/transcriptions/stream` returns chunked results as Server-Sent Events; long audio (>5s) is split at silence boundaries for progressive output
- **Real-time WebSocket** — `WS /ws/transcribe` accepts raw PCM audio for live transcription with a sliding window (up to 6s context) and cumulative partials
- **Multi-language** — auto-detects English, Chinese, Japanese, Cantonese, Hindi, Thai, and more

### Subtitles
- **SRT subtitle generation** — `POST /v1/audio/subtitles` produces `.srt` subtitle files from audio
- **Fast mode** — heuristic word-timing estimation from segment boundaries (no extra model, instant)
- **Accurate mode** — word-level alignment via ForcedAligner (~33ms accuracy, requires ~5.8GB VRAM)
- **CJK-aware** — proper line breaking for Chinese, Japanese, and Korean text
- **Configurable** — max line length, subtitle duration, pause thresholds, minimum gaps

### Translation
- **Audio-to-text translation** — `POST /v1/audio/translations` transcribes then translates via external LLM
- **Multiple backends** — works with Ollama Cloud, local Ollama, vLLM, or OpenAI
- **SRT translation** — preserves subtitle timing while translating content (`response_format=srt`)
- **Target languages** — English (`en`) and Chinese (`zh`)

### Real-Time WebSocket
- **Sliding window** — re-transcribes up to 6s of accumulated audio each trigger for full context
- **Server-side VAD** — Silero Voice Activity Detection auto-flushes on speech→silence transitions; skips inference for silence (`ASR_USE_SERVER_VAD=true` by default)
- **Per-connection VAD toggle** — clients can disable VAD via query param (`?use_server_vad=false`) or mid-session config action
- **Silence padding** — 600ms silence appended on flush to commit trailing words
- **Dual-model partials** — optionally use 0.6B for fast partials, 1.7B for final (`DUAL_MODEL=true`)
- **Control commands** — `flush`, `reset`, `config` (set language, toggle VAD)

### Performance
- **Priority scheduling** — WebSocket requests (priority 0) preempt HTTP uploads (priority 1) via min-heap queue
- **On-demand model loading** — model loads on first request, unloads after idle timeout (0 VRAM when idle)
- **Gateway + Worker mode** — proxy + subprocess split; killing worker reclaims all RAM/VRAM
- **Flash Attention 2** with graceful SDPA fallback
- **Pinned memory + CUDA stream** — async DMA pipeline for transfer/compute overlap

### Quantization & Acceleration (opt-in)
- **INT8 W8A8** via bitsandbytes (`QUANTIZE=int8`) — ~50% VRAM reduction
- **FP8** via torchao (`QUANTIZE=fp8`) — requires sm_89+ (Hopper/Ada Lovelace)
- **ONNX Runtime encoder** (`ONNX_ENCODER_PATH`) — ORT-accelerated encoder
- **TensorRT encoder** (`TRT_ENCODER_PATH`) — compiled TRT engine
- **CUDA Graphs warming** (`USE_CUDA_GRAPHS=true`) — kernel caching warmup passes
- **Speculative decoding** (`USE_SPECULATIVE=true`) — 0.6B draft + 1.7B verifier

### Observability
- **Structured logging** — loguru-based with configurable `LOG_LEVEL`
- **Request tracing** — `X-Request-ID` header propagated through gateway→worker→server
- **Atomic request logging** — every endpoint logs entry (params, size) and exit (duration, result)
- **Swagger UI** — interactive API docs at `/docs` with typed Pydantic schemas

## Quick Start

```bash
# Clone and start
git clone https://github.com/jaaacki/qwen3-asr.git
cd qwen3-asr
cp .env.example .env   # Adjust settings as needed
docker compose up -d

# Check health
curl http://localhost:8100/health

# Transcribe audio file
curl -X POST http://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav"

# Streaming transcription (SSE)
curl -X POST http://localhost:8100/v1/audio/transcriptions/stream \
  -F "file=@audio.wav" -N

# Generate subtitles
curl -X POST http://localhost:8100/v1/audio/subtitles \
  -F "file=@audio.wav" -F "mode=fast" -o subtitles.srt

# Translate audio to Chinese
curl -X POST http://localhost:8100/v1/audio/translations \
  -F "file=@audio.wav" -F "language=zh"
```

For WebSocket real-time transcription, see [docs/WEBSOCKET_USAGE.md](docs/WEBSOCKET_USAGE.md).

## Requirements

- Docker with NVIDIA GPU support (`nvidia-docker2` or Docker 19.03+)
- NVIDIA GPU with CUDA 12.4 compatible driver
- ~4GB disk for model download (cached in `./models/`)
- ~3.5GB VRAM for ASR-1.7B model at runtime

## API Endpoints

### `GET /health`

Returns server status, model info, and GPU memory.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "Qwen/Qwen3-ASR-1.7B",
  "cuda": true,
  "gpu_name": "NVIDIA GeForce RTX 4060",
  "gpu_allocated_mb": 3584,
  "gpu_reserved_mb": 3712
}
```

### `POST /v1/audio/transcriptions`

Transcribe an audio file. OpenAI-compatible.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (WAV, MP3, FLAC, OGG, AIFF, etc.) |
| `language` | string | `"auto"` | Language code (e.g. `en`, `zh`, `ja`) or `"auto"` |
| `return_timestamps` | bool | `false` | Include word-level timestamps |

```bash
curl -X POST http://localhost:8100/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "language=en" \
  -F "return_timestamps=true"
```

Response:
```json
{"text": "Hello world.", "language": "en"}
```

### `POST /v1/audio/transcriptions/stream`

Same parameters as transcription. Returns chunked SSE stream. Long audio (>5s) is split for progressive output.

```bash
curl -X POST http://localhost:8100/v1/audio/transcriptions/stream \
  -F "file=@recording.wav" -N
```

Each SSE event:
```
data: {"text": "Hello", "language": "en", "is_final": false, "chunk_index": 0}
data: {"text": "world.", "language": "en", "is_final": true, "chunk_index": 1}
data: {"done": true}
```

### `POST /v1/audio/subtitles`

Generate SRT subtitle file from audio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file |
| `language` | string | `"auto"` | Language code or `"auto"` |
| `mode` | string | `"accurate"` | `"accurate"` (ForcedAligner) or `"fast"` (heuristic) |
| `max_line_chars` | int | `42` | Maximum characters per subtitle line |

```bash
# Fast mode — no aligner needed, works on any GPU
curl -X POST http://localhost:8100/v1/audio/subtitles \
  -F "file=@recording.wav" \
  -F "mode=fast" \
  -o subtitles.srt

# Accurate mode — loads ForcedAligner (~5.8GB VRAM, needs >8GB GPU)
curl -X POST http://localhost:8100/v1/audio/subtitles \
  -F "file=@recording.wav" \
  -F "mode=accurate" \
  -o subtitles.srt
```

Output is standard `.srt` format:
```srt
1
00:00:00,000 --> 00:00:02,340
Hello, this is a test recording.

2
00:00:02,500 --> 00:00:05,120
The subtitles are generated automatically.
```

### `POST /v1/audio/translations`

Transcribe audio then translate via an external OpenAI-compatible LLM API.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file |
| `language` | string | `"en"` | Target language: `"en"` or `"zh"` |
| `response_format` | string | `"json"` | `"json"` (text) or `"srt"` (translated subtitles with timing) |

Requires `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `TRANSLATE_MODEL` configured in `.env`.

```bash
# Text translation
curl -X POST http://localhost:8100/v1/audio/translations \
  -F "file=@recording.wav" \
  -F "language=zh"

# Translated SRT subtitles
curl -X POST http://localhost:8100/v1/audio/translations \
  -F "file=@recording.wav" \
  -F "language=en" \
  -F "response_format=srt" -o translated.srt
```

### `WS /ws/transcribe`

Real-time WebSocket transcription. Accepts raw PCM 16-bit LE, 16kHz mono.

| Query Parameter | Default | Description |
|-----------------|---------|-------------|
| `use_server_vad` | `true` | Enable server-side VAD (auto-flush on silence, skip inference for silence) |

See [docs/WEBSOCKET_USAGE.md](docs/WEBSOCKET_USAGE.md) for full protocol details.

### `GET /docs`

Interactive Swagger UI with all endpoints and Pydantic request/response schemas.

## Configuration

All settings are configured via environment variables in `.env`. See [`.env.example`](.env.example) for a documented template.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8100` | Host port mapped to container port 8000 |
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID (`Qwen/Qwen3-ASR-0.6B` for speed) |
| `IDLE_TIMEOUT` | `120` | Seconds before unloading model from GPU (0 = keep loaded forever) |
| `REQUEST_TIMEOUT` | `300` | Max seconds per inference request |
| `LOG_LEVEL` | `INFO` | Log verbosity: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

### WebSocket Streaming

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_BUFFER_SIZE` | `14400` | Bytes to accumulate before triggering transcription (~450ms at 16kHz) |
| `WS_WINDOW_MAX_S` | `6.0` | Max seconds of audio in the sliding window (higher = better accuracy, more GPU) |
| `WS_FLUSH_SILENCE_MS` | `600` | Silence padding appended on flush/disconnect (ms) |
| `ASR_USE_SERVER_VAD` | `true` | Server-side VAD: auto-flush on speech→silence, skip inference for silence |

### Subtitles

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCED_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | HuggingFace model for accurate mode alignment |
| `SUBTITLE_MAX_DURATION` | `7.0` | Max duration per subtitle block (seconds) |
| `SUBTITLE_PAUSE_THRESHOLD` | `0.5` | Pause duration that triggers subtitle break (seconds) |
| `SUBTITLE_MIN_DURATION` | `0.833` | Minimum subtitle display time (seconds) |
| `SUBTITLE_MIN_GAP` | `0.083` | Minimum gap between subtitles (seconds) |

### Translation

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | — | API base URL (Ollama Cloud: `https://ollama.com/api`, OpenAI: `https://api.openai.com/v1`) |
| `OPENAI_API_KEY` | — | API key for the translation backend |
| `TRANSLATE_MODEL` | `gemma3:12b` | LLM model used for translation |
| `TRANSLATE_TEMPERATURE` | `0.3` | Temperature for text translation |
| `TRANSLATE_SRT_TEMPERATURE` | `0.1` | Temperature for SRT translation (lower = more faithful) |

### SSE Streaming

| Variable | Default | Description |
|----------|---------|-------------|
| `SSE_CHUNK_SECONDS` | `5` | Duration of each SSE chunk for long audio (seconds) |
| `SSE_OVERLAP_SECONDS` | `1` | Overlap between SSE chunks (seconds) |

### Performance & Quantization

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANTIZE` | `""` | `int8` (bitsandbytes W8A8) or `fp8` (torchao, sm_89+) |
| `USE_CUDA_GRAPHS` | `false` | Extra warmup passes for CUDA kernel caching |
| `DUAL_MODEL` | `false` | Load 0.6B model for WebSocket partials alongside 1.7B |
| `FAST_MODEL_ID` | `Qwen/Qwen3-ASR-0.6B` | Model used for partials/draft in dual/speculative mode |
| `USE_SPECULATIVE` | `false` | Speculative decoding (0.6B draft + 1.7B verifier) |
| `ONNX_ENCODER_PATH` | `""` | Path to exported ONNX encoder |
| `TRT_ENCODER_PATH` | `""` | Path to compiled TensorRT engine |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_MODE` | `false` | Run as gateway proxy + worker subprocess (recommended for production) |
| `WORKER_HOST` | `127.0.0.1` | Worker host (used by gateway) |
| `WORKER_PORT` | `8001` | Worker port (used by gateway) |
| `NUMA_NODE` | `0` | NUMA node for CPU affinity (pins to GPU-collocated CPUs) |
| `USE_GRANIAN` | `false` | Use Granian (Rust-based ASGI) instead of uvicorn |

### Docker / CUDA

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch CUDA memory allocator tuning |

## Supported Audio Formats

HTTP endpoints accept any format supported by libsndfile:
**WAV, FLAC, MP3, OGG, AIFF, CAF, AU, W64, RF64**, and more.

Not supported: MP4, AAC, M4A (soundfile limitation).

WebSocket accepts **raw PCM only**: 16-bit little-endian, 16kHz, mono.

## Project Structure

```
qwen3-asr/
├── compose.yaml          # Docker Compose configuration
├── Dockerfile            # Container build (all deps pinned)
├── .env.example          # Documented env var template
├── src/
│   ├── server.py         # FastAPI server — inference, WebSocket, SSE
│   ├── gateway.py        # Gateway proxy (GATEWAY_MODE=true)
│   ├── worker.py         # Inference worker subprocess
│   ├── subtitle.py       # Subtitle generation (aligner, segmentation, SRT)
│   ├── translator.py     # Translation via external OpenAI-compatible APIs
│   ├── logger.py         # Loguru structured logging + uvicorn interception
│   ├── schemas.py        # Pydantic models for Swagger UI documentation
│   ├── config.py         # Config validation and extracted constants
│   ├── errors.py         # Standardized error response helper
│   ├── export_onnx.py    # Export encoder to ONNX Runtime
│   ├── build_trt.py      # Build TensorRT encoder engine
│   └── debug_audio.py    # Debug audio preprocessing locally
├── E2Etest/              # pytest E2E test suite
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
