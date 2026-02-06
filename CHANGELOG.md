# Changelog

## v0.2.0 — 2026-02-06

### Added
- **On-demand model loading** — model loads on first request instead of at startup (0 VRAM when idle)
- **Idle auto-unload** — model automatically unloads after `IDLE_TIMEOUT` seconds of inactivity (default: 120s), freeing GPU VRAM for other services
- **GPU inference semaphore** — serializes concurrent requests to prevent OOM on shared GPU
- **Request timeout** — configurable via `REQUEST_TIMEOUT` env var (default: 300s)
- **Audio preprocessing** — automatic mono conversion, 16kHz resampling, and peak normalization
- **GPU warmup** — runs a dummy inference on first load to pre-cache CUDA kernels
- **Health endpoint improvements** — reports GPU memory usage, device name
- **Docker healthcheck** in compose.yaml

### Changed
- SDPA attention implementation (`attn_implementation="sdpa"`) for better memory efficiency
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
