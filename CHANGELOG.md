# Changelog

## v0.3.0 — 2026-02-07

### Changed
- **Reduced transcription buffer from ~1.5s to ~800ms** — lower latency between speech and transcription results
- **Added 300ms overlap between consecutive chunks** — tail of previous chunk is prepended to next, preventing words from being split at buffer boundaries (#1, #2)
- **Added 600ms silence padding on flush/disconnect** — gives the model trailing acoustic context to commit the last word (#2)
- **Transcribe remaining buffer on WebSocket disconnect** — no audio is lost when client disconnects mid-stream
- **Empty flush now returns empty final response** — instead of silently ignoring
- **Bare `except:` replaced with `except Exception:`** — better exception hygiene

### Configuration
- `WS_BUFFER_SIZE` default changed: 48000 → 25600 bytes (~800ms)
- New env var: `WS_OVERLAP_SIZE` — overlap bytes between chunks (default: 9600 = ~300ms)
- New env var: `WS_FLUSH_SILENCE_MS` — silence padding in ms on flush (default: 600)

## v0.2.0 — 2026-02-06

### Added
- **WebSocket endpoint `/ws/transcribe`** — real-time audio transcription via WebSocket
  - Accepts binary audio frames (PCM 16-bit, 16kHz mono)
  - Automatic buffering and chunking (~1.5 seconds by default, configurable via `WS_BUFFER_SIZE`)
  - Returns JSON responses: `{text: ..., is_partial: bool, is_final: bool}`
  - Control commands: `{action: flush}` and `{action: reset}`
  - Proper connection lifecycle handling (connect, disconnect, errors)
- **On-demand model loading** — model loads on first request instead of at startup (0 VRAM when idle)
- **Idle auto-unload** — model automatically unloads after `IDLE_TIMEOUT` seconds of inactivity (default: 120s), freeing GPU VRAM for other services
- **GPU inference semaphore** — serializes concurrent requests to prevent OOM on shared GPU
- **Request timeout** — configurable via `REQUEST_TIMEOUT` env var (default: 300s)
- **Audio preprocessing** — automatic mono conversion, 16kHz resampling, and peak normalization
- **GPU warmup** — runs a dummy inference on first load to pre-cache CUDA kernels
- **Health endpoint improvements** — reports GPU memory usage, device name
- **Docker healthcheck** in compose.yaml

### Changed
- SDPA attention implementation (`attn_implementation=sdpa`) for better memory efficiency
- `low_cpu_mem_usage=True` for reduced peak memory during model loading
- `torch.inference_mode()` context for all inference calls
- GPU memory explicitly released after every inference via `torch.cuda.empty_cache()`
- Thread pool execution for inference (non-blocking async server)
- Dockerfile: added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--no-install-recommends`, apt cache cleanup

### Performance
- VRAM reduced from ~3,952 MB to ~1,696 MB (57% reduction)
- Inference speed unchanged at ~1.27s for 20s audio
- 0 MB VRAM when idle (model unloaded)

## v0.1.0 — 2026-02-01

- Initial release with Qwen3-ASR-0.6B model
- OpenAI-compatible `/v1/audio/transcriptions` endpoint
- SSE streaming via `/v1/audio/transcriptions/stream`
- Multi-language support with auto-detection
