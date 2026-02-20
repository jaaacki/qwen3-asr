# Changelog

## v0.5.0 — 2026-02-20

### Streaming & Audio Processing
- **Real SSE chunked streaming** — 5s chunks with 1s overlap for progressive transcription (#28)
- **Long audio chunking at silence boundaries** — files >25s split at silences (#24)
- **Silero VAD gating** — skips GPU inference for silent WS frames (#25)
- **Reduced WS buffer** from 800ms to 450ms, overlap from 300ms to 150ms (#26)

### Scheduling & Concurrency
- **Priority scheduling** — WS requests (priority=0) preempt HTTP (priority=1) via min-heap queue (#27)
- **KV-cache reuse** across WebSocket chunks for reduced re-computation (#32)
- **Dual-model strategy** — optional 0.6B for fast WS partials, 1.7B for finals (#33)

### Quantization & Acceleration
- **INT8 W8A8 quantization** via bitsandbytes (opt-in: QUANTIZE=int8) — ~50% VRAM reduction (#29)
- **FP8 quantization** via torchao (opt-in: QUANTIZE=fp8) — requires sm_89+ GPU (#34)
- **CUDA Graphs kernel warming** — 3 extra warmup passes (opt-in: USE_CUDA_GRAPHS=true) (#30)
- **ONNX Runtime encoder** — optional ORT-accelerated encoder (opt-in: ONNX_ENCODER_PATH) (#31)

### Docker
- Added **silero-vad**, **bitsandbytes**, **onnxruntime-gpu**, **torchao** dependencies
- New script: **src/export_onnx.py** for encoder export

## v0.4.0 — 2026-02-20

### Model & Configuration
- **Default model upgraded to Qwen3-ASR-1.7B** — better multilingual accuracy (#7)
- **model.eval() after loading** — disables dropout for deterministic inference (#18)
- **cudnn.benchmark enabled** — auto-selects fastest cuDNN algorithm for fixed-size inputs (#17)
- **TF32 matmul precision enabled** — ~3x throughput on Ampere+ GPUs (#11)
- **Flash Attention 2** — with graceful SDPA fallback when unavailable (#10)
- **torch.compile(mode="reduce-overhead")** — compiled inference with fallback (#9)

### Inference Hot Path
- **Removed per-request release_gpu_memory()** — saves 5-10ms per request (#14)
- **Eliminated bytes() copy in WS handler** — bytearray passed directly to numpy (#16)
- **Fast-path preprocess_audio_ws()** — skips redundant mono/resample for WebSocket audio (#15)
- **Disabled WS per-message-deflate compression** — saves ~1ms CPU per frame (#20)

### Audio Pipeline
- **Replaced librosa with torchaudio** for GPU-native resampling (#12)
- **Warmup with representative noise** instead of silence — better CUDA kernel priming (#19)
- **Repetition detection** — detect_and_fix_repetitions() post-processing on all outputs (#8)

### Threading & Memory
- **Dedicated single-thread ThreadPoolExecutor** for GPU inference with thread affinity (#22)
- **ASGI lifespan handler** — replaces deprecated @app.on_event("startup") (#22)
- **Pre-allocated pinned memory buffer** (1.92 MB) for fast CPU→GPU DMA transfer (#21)
- **Dedicated CUDA stream** for inference with transfer/compute overlap (#23)

### Docker
- **OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1** — prevents CPU thread contention (#13)
- **--ws websockets** added to uvicorn CMD (#20)
- **flash-attn** pip install added (#10)
- **torchaudio** replaces librosa dependency (#12)

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
