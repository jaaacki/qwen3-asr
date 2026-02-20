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

# Debug audio processing locally
python src/debug_audio.py
```

No test suite exists — validation is done via curl/manual testing against the running container.

## Architecture

All server logic lives in `src/server.py` (~545 lines). Key subsystems:

### Model Lifecycle
- Model is loaded on first request, not at startup
- `_idle_watchdog()` background task unloads the model after `IDLE_TIMEOUT` seconds of inactivity (default 120s, 0 = disabled)
- `asyncio.Lock()` prevents load/unload race conditions
- GPU memory explicitly released after every inference via `release_gpu_memory()`

### Concurrency
- `asyncio.Semaphore(1)` serializes all GPU inference — one request at a time
- All inference runs in `run_in_executor()` to keep the async event loop non-blocking
- WebSocket and HTTP requests share the same semaphore (file uploads can block real-time WS calls)

### WebSocket Real-Time Transcription (`/ws/transcribe`)
- Accepts raw PCM: 16-bit little-endian, 16kHz, mono
- Buffers ~450ms of audio before transcribing (`WS_BUFFER_SIZE`)
- **Overlap**: Last 150ms of each chunk is prepended to the next chunk to prevent word splits at boundaries (`WS_OVERLAP_SIZE`)
- **Flush silence padding**: 600ms of silence appended before final transcription on `flush` command to help the model commit trailing words (`WS_FLUSH_SILENCE_MS`)
- Control messages: `flush` (process remaining buffer), `reset` (clear state), `config` (set language)
- Buffer is transcribed on client disconnect (no audio loss)
- Responses: `{"text": "...", "is_partial": false, "is_final": false}`

### Audio Preprocessing Pipeline
Input audio → mono conversion → float32 normalization → resample to 16kHz (librosa) → peak normalization to [-1, 1]

### Known Limitations
- No long-audio chunking — files >30s may degrade quality
- No repetition/hallucination detection for noisy audio
- Uses `sdpa` attention (not Flash Attention 2, ~20% slower)
- No `torch.compile` optimization

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-ASR-0.6B` | HuggingFace model (1.7B gives better multilingual accuracy) |
| `IDLE_TIMEOUT` | `120` | Seconds before model unloads from GPU (0 = keep loaded) |
| `REQUEST_TIMEOUT` | `300` | Max inference time per request |
| `WS_BUFFER_SIZE` | `14400` | WebSocket audio buffer (~450ms at 16kHz) |
| `WS_OVERLAP_SIZE` | `4800` | Overlap between chunks (~150ms) |
| `WS_FLUSH_SILENCE_MS` | `600` | Silence padding on flush |

Port mapping: container 8000 → host 8100.

## Agent Rules

Operating rules live in `.agent-rules/` — read and follow them at all times:

- **`prompt_agent-team-rules.md`** — Three roles: Architect (1), Builder (N), Critic (1+). Scale Builders to independent issues. Coordinate through Architect.
- **`prompt_docs-versioning-rules.md`** — Four living docs updated after every issue: ROADMAP.md, LEARNING_LOG.md, CHANGELOG.md, README.md. Versioning: patch=issue, minor=milestone, major=vision.
- **`prompt_git-workflow-rules.md`** — No code without an issue. Issue title format: `[Enhancement/Bug/Fix/Docs/Refactor/Chore] Description`. Branch: `milestone/{phase} → issue-{N}-{desc}`. Bugs branch from main. Squash issue PRs into milestone; merge commit milestone into main.
- **`prompt_testing-rules.md`** — Test files live next to source (`server_test.py`). Tests ship in same PR as code. Always report `Tests: X passed, Y failed, Z skipped`.

## Docs

- `docs/WEBSOCKET_USAGE.md` — WebSocket protocol, connection format, example Python client
- `RESEARCH_ANALYSIS.md` — Architecture comparison with official Qwen3-ASR SDK and vLLM backend
- `improvements.md` — Prioritized optimization recommendations (Flash Attention, torch.compile, chunking, etc.)
