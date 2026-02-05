# Qwen3-ASR Docker Server

A Dockerized REST API server for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Automatic Speech Recognition) with GPU acceleration.

Wraps the Qwen3-ASR-0.6B model in a FastAPI server with an OpenAI-compatible transcription endpoint.

## Features

- OpenAI-compatible `/v1/audio/transcriptions` endpoint
- Server-Sent Events streaming via `/v1/audio/transcriptions/stream`
- Multi-language support with auto-detection
- Optional timestamp output
- GPU-accelerated inference (NVIDIA CUDA)
- HuggingFace model caching via volume mount

## Quick Start

```bash
# Build and start
docker compose up -d

# Check health
curl http://localhost:8100/health

# Transcribe audio
curl -X POST http://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav"
```

## Requirements

- Docker with NVIDIA GPU support (`nvidia-docker2` or Docker 19.03+)
- NVIDIA GPU with CUDA 12.4 compatible driver
- ~2GB disk for model download (cached in `./models/`)

## API Endpoints

### `GET /health`

Returns server status, model info, and CUDA availability.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "Qwen/Qwen3-ASR-0.6B",
  "cuda": true
}
```

### `POST /v1/audio/transcriptions`

Transcribe an audio file. Accepts WAV, MP3, FLAC, and other formats supported by soundfile.

**Parameters** (multipart form):
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file to transcribe |
| `language` | string | `"auto"` | Language code or `"auto"` for detection |
| `return_timestamps` | bool | `false` | Include word-level timestamps |

**Example:**

```bash
# Basic transcription
curl -X POST http://localhost:8100/v1/audio/transcriptions \
  -F "file=@recording.wav"

# With language hint and timestamps
curl -X POST http://localhost:8100/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "language=en" \
  -F "return_timestamps=true"
```

**Response:**

```json
{
  "text": "Hello, how are you today?",
  "language": "en"
}
```

### `POST /v1/audio/transcriptions/stream`

Streaming transcription via Server-Sent Events. Same parameters as above.

```bash
curl -X POST http://localhost:8100/v1/audio/transcriptions/stream \
  -F "file=@recording.wav" \
  -N
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_ID` | `Qwen/Qwen3-ASR-0.6B` | HuggingFace model ID |

## Port Mapping

The container runs on port 8000 internally, mapped to **8100** on the host (configurable in `compose.yaml`).

## Project Structure

```
qwen3-asr/
├── compose.yaml      # Docker Compose configuration
├── Dockerfile        # Container build definition
├── server.py         # FastAPI server
├── models/           # HuggingFace model cache (auto-populated)
└── README.md
```

## License

Apache 2.0
