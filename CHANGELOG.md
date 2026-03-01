# Changelog

## [v0.14.0] — 2026-03-01

Server-side VAD toggle and comprehensive documentation rewrite. VAD is now configurable per-connection and stale references cleaned across all docs.

### Added
- **`ASR_USE_SERVER_VAD` env var** — server-wide default for VAD (default `true`); controls auto-flush on speech→silence transitions and inference skipping for silent frames
- **Per-connection VAD override** — WebSocket clients can disable/enable VAD via query param (`?use_server_vad=false`) or mid-session config action (`{"action":"config","use_server_vad":false}`)
- **VAD state in handshake** — WebSocket connection confirmation now includes `use_server_vad` field
- **VAD state in config response** — config action response now includes `use_server_vad` field
- **Gateway VAD forwarding** — gateway proxies `use_server_vad` query param to worker

### Changed
- **README.md** — full rewrite: organized features into sections (Speech-to-Text, Subtitles, Translation, WebSocket, Performance, Quantization, Observability); complete env var reference by category; added endpoint examples with response formats; added supported audio formats section; removed stale vLLM/causal encoder references
- **`.env`** — removed stale `USE_VLLM`, `USE_CAUSAL_ENCODER`, `WS_OVERLAP_SIZE`; added `ASR_USE_SERVER_VAD=true`; renamed sections for clarity
- **`.env.example`** — same stale removals; added `ASR_USE_SERVER_VAD` with inline docs explaining per-connection override
- **`docs/WEBSOCKET_USAGE.md`** — added query parameters table, Server-Side VAD section, config action with `use_server_vad` toggle, updated Python example with VAD awareness
- **`CLAUDE.md`** — added `/v1/audio/translations` to endpoints table; updated VAD description; added `ASR_USE_SERVER_VAD` to env vars table; removed torch.compile from optimizations

### Removed
- **Stale env vars from docs** — `USE_VLLM`, `USE_CAUSAL_ENCODER`, `WS_OVERLAP_SIZE` removed from `.env`, `.env.example`, README configuration tables
- **torch.compile from CLAUDE.md** optimizations table (was investigated and abandoned)

## [v0.13.1] — 2026-03-01

Conservative codebase cleanup: dead code removal, helper extraction, stale artifact cleanup, and documentation fixes. No behavioral changes.

### Removed
- **vLLM dead code** — `_load_vllm_engine()`, `_do_transcribe_vllm()`, `USE_VLLM` flag, and `_vllm_engine` global (vllm was never installed in the Docker image)
- **Causal encoder experiment** — `_patch_encoder_causal()` function and `_encoder_state_cache` dict (gated behind `USE_CAUSAL_ENCODER`, never used)
- **`WS_OVERLAP_SIZE`** — dead since v0.12.0 sliding window rewrite; removed from server.py, worker.py, .env, .env.example
- **Duplicate `_fast_model = None`** — module-level shadow removed (kept the one with descriptive comment)
- **Stale files** — `build.log`, `test.txt`, `src/server_test.py` (trivial assert-import tests), `.agent-rules/` (agent orchestration rules)
- **8 completed plan documents** from `docs/plans/` (3,866 lines of historical artifacts)
- **Unused test fixtures** — `async_http_client`, `reset_model_state` from conftest.py

### Changed
- **`_decode_audio()` helper** — extracted in server.py, replaces 4 inline `sf.read()` blocks; worker.py imports and uses it (replaces 4 more)
- **`_proxy_error_or_raise()` helper** — extracted in gateway.py, replaces 3 inline proxy error handling blocks
- **Dockerfile** — replaced 10 individual `COPY src/xxx.py` lines with `COPY src/*.py /app/` (prevents runtime crashes when new files are added)
- **`debug_audio.py`** — replaced removed `librosa` import with `torchaudio` for resampling

### Fixed
- **`model.eval()` not called** on main ASR model after creation (correctness regression — disables dropout)
- **`/transcribe/stream` in worker.py** — missing try/except around audio decode (unhandled 500 on corrupt input)
- **`accuracy` marker** not registered in conftest.py `pytest_configure`
- **CLAUDE.md** — torch.compile claimed "always on" (was investigated and abandoned); server.py line count said ~1100 (actually ~1300); references to non-existent `improvements.md` and `RESEARCH_ANALYSIS.md` removed; vLLM row removed from optimization table
- **`sample_audio_20s` fixture** — removed fragile fallback to root `test01_20s.wav`

## [v0.13.0] — 2026-03-01

Production standards compliance: structured error responses, request tracing, startup validation, and externalized configuration. Every log entry now carries a requestId, every error response is machine-parseable, and every configurable value lives in an env var.

### Added
- **Standard error response shape** — all error responses use `{code, message, statusCode, context}` with machine-readable error codes (`AUDIO_DECODE_FAILED`, `TRANSCRIPTION_TIMEOUT`, `TRANSLATION_FAILED`, `SUBTITLE_TIMEOUT`, `EMPTY_AUDIO`, `INVALID_MODE`, `WORKER_ERROR`) (#107)
- **`ErrorResponse` Pydantic model** in `schemas.py` for Swagger documentation of error responses (#107)
- **`error_response()` helper** in new `src/errors.py` — auto-injects `requestId` into error context (#107)
- **Request ID middleware** — HTTP middleware in server.py, gateway.py, worker.py generates/forwards `X-Request-ID` header for cross-service log correlation (#105)
- **WebSocket session IDs** — each WebSocket session gets a UUID for log correlation, forwarded via query parameter in gateway mode (#105)
- **Startup env var validation** — `validate_env()` in new `src/config.py` validates MODEL_ID, REQUEST_TIMEOUT, IDLE_TIMEOUT, LOG_LEVEL, QUANTIZE, WORKER_PORT, WS_WINDOW_MAX_S at startup; exits with clear error on invalid config (#106)
- **8 extracted config constants** — formerly hardcoded values now configurable via env vars with safe parsing: `TRANSLATE_TEMPERATURE`, `TRANSLATE_SRT_TEMPERATURE`, `SSE_CHUNK_SECONDS`, `SSE_OVERLAP_SECONDS`, `SUBTITLE_MAX_DURATION`, `SUBTITLE_PAUSE_THRESHOLD`, `SUBTITLE_MIN_DURATION`, `SUBTITLE_MIN_GAP` (#106)

### Changed
- **`.env.example`** — 3 new sections: Translation Tuning, SSE Streaming, Subtitle Timing (#106)
- **Gateway error proxying** — parses structured error from worker and forwards it; falls back to `WORKER_ERROR` wrapper for unparseable responses (#107)

### Fixed
- **Missing `sf.read()` error handling** on 3 server.py endpoints (translations, subtitles, stream) — corrupt audio now returns `AUDIO_DECODE_FAILED` instead of unhandled 500 (#107)
- **SSE stream error shape** — uses standard `{code, message, statusCode}` instead of ad-hoc `{"error": ...}` (#107)
- **Log level aliases** — `WARN` and `FATAL` now accepted in `LOG_LEVEL` env var alongside canonical names (#106)

### New Files
- `src/errors.py` — error response helper
- `src/config.py` — centralized config validation and extracted constants

## [v0.12.0] — 2026-02-25

Sliding window WebSocket streaming: model sees up to 6 seconds of context instead of 450ms, bringing streaming accuracy close to batch quality.

### Changed
- **WebSocket streaming**: Replaced per-chunk transcription with expanding sliding window (#101)
  - Model now sees up to 6 seconds of context (was 450ms per chunk)
  - Partials are cumulative transcripts (industry standard: client replaces, never appends)
  - New env var: `WS_WINDOW_MAX_S` (default 6.0) controls window size (#100)
  - Removed `overlap_size` from client handshake, added `window_max_s`
- **WebSocket tests**: Updated for cumulative partial verification (#102)
- **Realtime benchmark**: Simplified — dedup logic removed since cumulative partials eliminate overlap artifacts (#103)

## v0.11.0 — 2026-02-24

Comprehensive observability: every request in, every response out, every error caught — atomic loguru logging across the entire request chain.

### Added
- **Atomic request-level logging** — every HTTP endpoint and WebSocket handler logs entry (method, path, file size, params), exit (duration, result size), and errors with full context across server.py, worker.py, gateway.py, translator.py, and subtitle.py (#98)
- **Configurable LOG_LEVEL** — `LOG_LEVEL` env var controls verbosity (DEBUG/INFO/WARNING/ERROR, default INFO) for both loguru and stdlib logging (#96)
- **Ollama Cloud translation defaults** — `.env.example` ships with `OPENAI_BASE_URL=https://ollama.com/api` and `TRANSLATE_MODEL=gemma3:12b` for out-of-the-box translation support (#97)

### Changed
- **`.env.example` reorganized** — clearer section structure, inline documentation for WebSocket defaults, removed stale "feature branch" comment from translation section (#97)

## v0.10.1 — 2026-02-24

### Changed
- **Pin all Dockerfile dependencies** — 15 packages pinned to latest versions as of 2026-02-24 (e.g. fastapi==0.133.0, uvicorn==0.41.0, websockets==16.0, flash-attn==2.8.3, onnxruntime-gpu==1.24.2) for reproducible builds
- **Pin E2E test dependencies** — 9 packages pinned to latest Python 3.8-compatible versions (pytest==8.3.5, httpx==0.28.1, numpy==1.24.4, etc.)

### Fixed
- **Missing Dockerfile COPY** — `logger.py`, `schemas.py`, `translator.py` were imported at runtime but never copied into the container image, causing crashes on fresh builds
- **`__future__` import order in subtitle.py** — `from __future__ import annotations` must precede all other imports; was after `from logger import log`

## v0.10.0 — 2026-02-24

### Added
- **Real-time WebSocket benchmark** — new `E2Etest/test_realtime_accuracy.py` streams a FLEURS audio clip to the live WebSocket in real-time (450ms chunks with wall-clock pacing) and reports per-chunk input-to-output latency (min/median/p95/max), flush latency, RTF, WER and CER against the reference transcript. (#94)
- **Realtime pytest marker** — `realtime` marker added to `E2Etest/pytest.ini` for filtering (`pytest E2Etest/ -m realtime`). (#93)
- **Real-Time Benchmark report section** — `MarkdownReportGenerator` in `conftest.py` gains `_parse_realtime_metrics()` and a new ⚡ Real-Time Benchmark table in the generated markdown report, showing latency and accuracy per run. (#95)

## v0.9.0 — 2026-02-22

### Added
- **Unified Structured Logging** — Replaced all built-in `print()` and standard `logging` instances with `loguru`. (#87)
- **FastAPI / Uvicorn Log Interception** — Custom `InterceptHandler` automatically maps uvicorn access and error logs to seamlessly conform to `loguru`'s structured JSON architecture to prevent blind spots. (#88)

## v0.8.0 — 2026-02-22

### Added
- **Swagger UI Documentation** — exposed at `/docs` with detailed Pydantic schemas for improved interactive testing and API spec sharing (#85)
- **Audio Translation Endpoint** — new `POST /v1/audio/translations` and internal worker `/translate` endpoint using OpenAI-compatible APIs (like Ollama or vLLM) for translation (#86)
  - Supports English (`en`) and Chinese (`zh`) target translations.
  - Supports `response_format` for `json` (raw text) or `srt` (translates full subtitle file while strictly preserving timestamps).
  - Designed as an isolated `src/translator.py` module to maintain separation of concerns without altering upstream inference hot-paths.

## v0.6.2 — 2026-02-21

### Added
- **SRT subtitle generation** — new `POST /v1/audio/subtitles` endpoint with two modes (#83):
  - **Accurate mode**: Qwen3-ForcedAligner-0.6B for word-level timestamps (~33ms accuracy), lazy-loaded on first request (~2GB VRAM)
  - **Fast mode**: heuristic word timestamps estimated from segment duration proportional to character count (no aligner, no extra VRAM)
- **Subtitle segmentation engine** — groups words into subtitle blocks respecting max line length (42 chars), max duration (7s), sentence boundaries, and long pauses (>500ms) (#83)
- **CJK tokenization** — character-level splitting for Chinese, Japanese, Korean text with mixed CJK/Latin support (#83)
- **Two-line subtitle splitting** — prefers clause boundaries, conjunctions/prepositions, and bottom-heavy layout (#83)
- **Timing enforcement** — minimum subtitle duration (833ms), minimum gap (83ms), overlap fixing (#83)
- **ForcedAligner lifecycle** — lazy-loaded on first accurate-mode request, unloaded alongside main model on idle timeout (#83)
- **Gateway/worker proxy** — subtitle endpoint forwarded in GATEWAY_MODE (#83)
- **E2E subtitle tests** — 10 tests covering fast/accurate modes, SRT structure, line length, overlaps, error handling (#83)

### New Files
- `src/subtitle.py` — subtitle generation module (~500 lines)
- `src/subtitle_test.py` — 41 unit tests
- `E2Etest/test_subtitle.py` — 10 E2E tests

## v0.6.1 — 2026-02-21

### Memory & Performance
- **Lazy imports** — `torch`, `soundfile`, `qwen_asr` deferred to first request; idle container RAM reduced from ~2.4GB to ~50-100MB (#82)
- **Inductor thread limit** — `TORCHINDUCTOR_COMPILE_THREADS=1` prevents ~20 worker subprocesses (~800MB saved) (#82)
- **Health endpoint** — no longer imports torch; uses `sys.modules` check to avoid 2.4GB spike from load balancer polling (#82)

### Removed (SDK handles natively)
- `preprocess_audio()`, `preprocess_audio_ws()`, `chunk_audio_at_silence()` — SDK's `model.transcribe()` handles audio normalization and chunking internally with superior algorithms (#82)
- Separate `AutoProcessor` loading — SDK loads processor inside `Qwen3ASRModel.from_pretrained()` (#82)
- Manual chunking loop in HTTP endpoint — single `_do_transcribe()` call replaces 40-line chunk-and-join logic (#82)

### Fixed
- worker.py: `_infer_semaphore` (non-existent) replaced with `_infer_queue` (#82)

## v0.6.0 — 2026-02-20

### Architecture
- **Gateway + Worker mode** — `GATEWAY_MODE=true` splits into gateway proxy (port 8000) + worker (port 8001); killing worker reclaims all model RAM (#35)
- **vLLM engine backend** — opt-in via `USE_VLLM=true` for production-grade serving with PagedAttention (#36)
- **Speculative decoding (SpecASR)** — opt-in via `USE_SPECULATIVE=true`; drafts with 0.6B, verifies complex outputs with 1.7B for ~2x speed (#38)

### Acceleration
- **TensorRT encoder** — opt-in via `TRT_ENCODER_PATH`; monkey-patches encoder forward pass with TRT-compiled engine (#37)
- **Cache-aware causal encoder** — EXPERIMENTAL via `USE_CAUSAL_ENCODER=true`; patches encoder attention to causal masks for incremental encoding (#39)

### Infrastructure
- **NUMA-aware CPU pinning** — pins process to GPU-collocated NUMA node via `NUMA_NODE` env var (#40)
- **Granian ASGI server** — opt-in via `USE_GRANIAN=true`; Rust-based alternative to uvicorn (#41)

### Docker
- Added **psutil**, **granian** dependencies
- New scripts: **src/gateway.py**, **src/worker.py**, **src/build_trt.py**
- CMD supports GATEWAY_MODE, USE_GRANIAN, and default uvicorn modes
- New doc: **docs/GRANIAN_BENCHMARK.md** with evaluation criteria

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
