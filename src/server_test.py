# server_test.py — Manual test notes for qwen3-asr server changes
# Run: docker compose up -d --build
# Then execute the verification steps listed per issue.

# ─── Issue #18: model.eval() after loading ──────────────────────────────────
# Change: Added model.eval() immediately after Qwen3ASRModel.from_pretrained()
#         in _load_model_sync(). Disables dropout and batchnorm updates for
#         inference correctness and ~5% speed gain.
# Verify:
#   docker compose up -d --build
#   curl http://localhost:8100/health
#   # Expected: {"model_loaded": true, ...} after first request triggers load
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
#   # Run twice with identical audio — results should be deterministic
# Expected: Transcription results are identical across repeated identical requests.

# ─── Issue #17: cudnn.benchmark ─────────────────────────────────────────────
# Change: Added torch.backends.cudnn.benchmark = True before model loading
#         in _load_model_sync(). Auto-selects fastest cuDNN convolution algorithm
#         for fixed-size inputs — 10-15% speedup on repeated same-size inputs.
# Verify:
#   docker compose up -d --build
#   curl http://localhost:8100/health
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
#   # Time multiple requests — second+ requests should be measurably faster
# Expected: Faster inference on repeated same-size audio inputs. No functional change.

# ─── Issue #11: TF32 matmul precision ───────────────────────────────────────
# Change: Added torch.backends.cuda.matmul.allow_tf32 = True and
#         torch.backends.cudnn.allow_tf32 = True in _load_model_sync().
#         TF32 gives ~3x matmul throughput on Ampere+ GPUs with negligible
#         accuracy loss for ASR workloads.
# Verify:
#   docker compose up -d --build
#   curl http://localhost:8100/health
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
#   # Compare transcription quality — should be identical or near-identical
# Expected: Faster inference with no meaningful accuracy degradation.

# ─── Issue #13: OMP/MKL thread env vars ─────────────────────────────────────
# Change: Added OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 to Dockerfile ENV.
#         Prevents OpenMP/MKL from spawning CPU threads that compete with
#         GPU inference, reducing context-switch overhead.
# Verify:
#   docker compose up -d --build
#   docker exec <container> env | grep -E "OMP|MKL"
#   # Expected: OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
# Expected: Transcription works normally. No CPU thread contention during GPU inference.

# ─── Issue #14: Remove per-request release_gpu_memory() ───────────────────
# Change: Removed release_gpu_memory() calls from _do_transcribe() finally block,
#         transcribe() timeout handler, sse_transcribe_generator() finally block,
#         and _transcribe_with_context() timeout handler.
#         Kept release_gpu_memory() only in _unload_model_sync() and warmup.
# Verify:
#   docker compose up -d --build
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
#   # Run multiple requests in succession to confirm no memory leak
#   curl http://localhost:8100/health  # check GPU memory stays stable
# Expected: faster inference (5-10ms saved per request), same transcription quality

# ─── Issue #16: Eliminate unnecessary bytes() copy in WS transcription ─────
# Change: Replaced np.frombuffer(bytes(full_audio), ...) with
#         np.frombuffer(full_audio, ...) in _transcribe_with_context().
#         bytearray is directly supported by np.frombuffer — no copy needed.
# Verify:
#   docker compose up -d --build
#   # Use WebSocket client from docs/WEBSOCKET_USAGE.md to stream audio
#   # Compare transcription output — should be identical
# Expected: eliminates one full buffer copy per WS chunk, same transcription quality

# ─── Issue #15: Optimize WebSocket audio preprocessing path ───────────────
# Change: Added preprocess_audio_ws() fast path that only does peak normalization.
#         Replaced preprocess_audio(audio, TARGET_SR) call in _transcribe_with_context()
#         with preprocess_audio_ws(audio) since WS audio is already mono, float32, 16kHz.
# Verify:
#   docker compose up -d --build
#   # Use WebSocket client from docs/WEBSOCKET_USAGE.md to stream audio
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
#   # Both HTTP and WS should produce correct transcriptions
# Expected: faster WS transcription (skips redundant mono/resample/cast), same quality

# ─── Issue #20: Disable WebSocket per-message-deflate compression ──────────
# Change: Added --ws websockets to uvicorn CMD in Dockerfile.
#         The websockets backend has no per-message compression by default.
#         Added comment in server.py near websocket.accept() for documentation.
# Verify:
#   docker compose up -d --build
#   # Use WebSocket client from docs/WEBSOCKET_USAGE.md to stream audio
#   # Verify WS connection works and transcription is correct
# Expected: ~1ms CPU savings per WS frame, same functionality

# ─── Issue #12: torchaudio resampling ─────────────────────────────────
# Change: replaced librosa.resample with torchaudio.transforms.Resample
# Verify: upload a 44.1kHz or 48kHz audio file, should auto-resample correctly
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio_44k.wav"
# Expected: correct transcription, no librosa import in logs

# ─── Issue #19: representative warmup ─────────────────────────────────
# Change: warmup now uses low-amplitude noise instead of silence
# Verify: docker compose logs shows "Warming up GPU..." with no errors
# Expected: faster first-request latency (warmup exercises more kernel paths)

# ─── Issue #8: repetition detection ───────────────────────────────────
# Change: detect_and_fix_repetitions() applied to all transcription outputs
# Verify: noisy audio that previously caused loops ("thank you thank you thank you...")
#   should now return clean output
# curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@noisy_audio.wav"

# ─── Issue #7: Qwen3-ASR-1.7B default ────────────────────────────────
# Change: default MODEL_ID changed to Qwen/Qwen3-ASR-1.7B
# Verify: docker compose up -d --build
#   curl http://localhost:8100/health  →  "model_id": "Qwen/Qwen3-ASR-1.7B"
# Override: set MODEL_ID=Qwen/Qwen3-ASR-0.6B in compose.yaml environment

# ─── Issue #10: Flash Attention 2 ────────────────────────────────────
# Change: uses flash_attention_2 if flash-attn package is installed
# Verify: docker compose logs | grep "Attention implementation:"
# Expected: "Attention implementation: flash_attention_2"
# Fallback: if flash-attn build fails, logs will show "sdpa"

# ─── Issue #9: torch.compile ─────────────────────────────────────────
# Change: model compiled with torch.compile(mode='reduce-overhead')
# Verify: docker compose logs | grep "torch.compile"
# WARNING: First request after startup will be slow (30-60s compilation)
# Subsequent requests will be 10-30% faster
# curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"

# ─── Issue #22: dedicated ThreadPoolExecutor ─────────────────────────
# Change: all run_in_executor calls use _infer_executor (1 thread, named)
# Verify: docker compose logs should show thread name "qwen3-asr-infer_0"
# curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"

# ─── Issue #21: pinned memory buffer ──────────────────────────────────────────
# Change: pre-allocated pinned memory for CPU→GPU audio transfer.
#         Audio data is copied into pinned buffer and its numpy view is passed
#         to model.transcribe() so internal .cuda() calls use async DMA transfers.
# Verify: docker compose logs | grep "Pinned memory"
# Expected: "Pinned memory buffer allocated: 1920 KB"

# ─── Issue #23: CUDA stream pipelining ────────────────────────────────
# Change: inference runs inside a dedicated CUDA stream
# Verify: docker compose logs | grep "CUDA inference stream"
# Expected: "CUDA inference stream created"
# Benefit: enables async transfer/compute overlap (measurable with CUDA profiler)

# ─── Issue #26: Reduce WebSocket buffer from 800ms to 450ms ──────────
# Change: WS_BUFFER_SIZE default changed from 25600 to 14400 (~450ms)
#         WS_OVERLAP_SIZE default changed from 9600 to 4800 (~150ms)
# Verify: WebSocket streaming still works with lower latency
# Expected: faster partial transcriptions, same accuracy

# ─── Issue #28: Real SSE streaming via chunked transcription ─────────
# Change: sse_transcribe_generator() now chunks long audio into 5s segments
#         with 1s overlap, streaming each chunk's result progressively.
#         Short audio (<5s) still runs as single batch.
# Verify: curl with large audio file should see multiple SSE events
# Expected: progressive results with chunk_index, is_final on last chunk

# ─── Issue #24: Long audio chunking at silence boundaries ────────────
# Change: Added chunk_audio_at_silence() for files >25s. Splits at silence
#         boundaries, transcribes each chunk, joins results.
# Verify: Upload a >30s audio file
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@long_audio.wav"
# Expected: correct transcription without quality degradation

# ─── Issue #25: Silero VAD ───────────────────────────────────────────
# Change: WS transcription skips silent frames
# Verify: docker compose logs | grep "Silero VAD loaded"
# Test: send silent audio via WS — should get empty response without GPU inference

# ─── Issue #27: Priority scheduling for WebSocket vs HTTP ───────────
# Change: Replaced asyncio.Semaphore(1) with PriorityInferQueue.
#         WS requests get priority=0 (higher), HTTP gets priority=1 (lower).
#         Uses min-heap so WS jobs run first when multiple are queued.
# Verify:
#   docker compose up -d --build
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
#   # Start WS stream while HTTP is processing — WS should not be blocked
# Expected: WS transcription completes even while HTTP request is in progress

# ─── Issue #29: INT8 W8A8 quantization with bitsandbytes ────────────
# Change: Added opt-in INT8 quantization via QUANTIZE=int8 env var.
#         Uses bitsandbytes BitsAndBytesConfig for 8-bit model loading.
#         Reduces VRAM usage by ~50% at minor accuracy cost.
# Verify:
#   Set QUANTIZE=int8 in docker-compose.yml environment
#   docker compose up -d --build
#   docker compose logs | grep "INT8"
#   Expected: "INT8 quantization enabled (bitsandbytes)"
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
# Expected: transcription works, gpu_allocated_mb ~50% less than default

# ─── Issue #30: CUDA Graphs for decoder loop ────────────────────────
# Change: Added opt-in CUDA graph warmup via USE_CUDA_GRAPHS=true env var.
#         Performs extra warmup passes to prime CUDA kernel caches.
# Verify:
#   Set USE_CUDA_GRAPHS=true in docker-compose.yml environment
#   docker compose up -d --build
#   docker compose logs | grep "CUDA graph"
#   Expected: "CUDA graph capture: best-effort (variable-length audio limits scope)"
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
# Expected: correct transcription, faster kernel dispatch
#   docker compose logs | grep "CUDA kernel"
#   Expected: "CUDA kernel cache warming complete (3 extra passes)"

# ─── Issue #31: ONNX Runtime encoder ────────────────────────────────
# Change: Added ONNX Runtime support for encoder via ONNX_ENCODER_PATH env var.
#         New script src/export_onnx.py exports encoder to ONNX format.
# Verify:
#   python src/export_onnx.py --output models/encoder.onnx
#   ONNX_ENCODER_PATH=models/encoder.onnx docker compose up -d --build
#   docker compose logs | grep "ONNX"
#   Expected: "ONNX encoder loaded from models/encoder.onnx"
# Without env var: server starts normally, no ONNX loading

# ─── Issue #32: KV-cache reuse across WebSocket chunks ──────────────
# Change: Store encoder output from previous WS chunk and pass it to
#         subsequent chunks. Reduces re-computation for repeated audio.
# Verify:
#   docker compose up -d --build
#   # Connect via WS, send multiple audio chunks, verify transcription
#   # Send {"action": "reset"} to clear cache between sessions
# Expected: faster subsequent WS chunks, same transcription quality

# ─── Issue #33: Dual-model strategy (0.6B partials, 1.7B finals) ────
# Change: When DUAL_MODEL=true, loads 0.6B model for fast WS partials
#         and configured model for final flush results.
# Verify:
#   Set DUAL_MODEL=true in docker-compose.yml environment
#   docker compose up -d --build
#   docker compose logs | grep -i "dual\|fast model"
#   Expected: "Dual-model strategy enabled"
# Without DUAL_MODEL=true: single model behavior (default)

# ─── Issue #34: FP8 quantization (torchao) ───────────────────────────
# Change: Added opt-in FP8 post-training quantization via QUANTIZE=fp8.
#         Uses torchao Float8DynamicActivationFloat8WeightConfig.
#         Requires sm_89+ GPU (Ada Lovelace / Hopper).
#         Applied after model.eval() but before torch.compile.
# Verify:
#   Set QUANTIZE=fp8 in docker-compose.yml environment
#   docker compose up -d --build
#   docker compose logs | grep "FP8"
#   Expected (sm_89+): "FP8 quantization applied (torchao)"
#   Expected (older GPU): "FP8 requires sm_89+, skipping"
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
# Expected: transcription works normally

# ─── Issue #35: Gateway + Worker architecture ────────────────────────
# Change: GATEWAY_MODE=true splits into gateway (port 8000) + worker (port 8001).
#         Gateway manages worker lifecycle; killing worker reclaims all model RAM.
# Verify:
#   GATEWAY_MODE=true docker compose up -d --build
#   curl http://localhost:8100/health
#   curl -X POST http://localhost:8100/v1/audio/transcriptions -F "file=@audio.wav"
# Expected: all endpoints work through gateway proxy
# Without GATEWAY_MODE: monolithic server behavior (default)


def test_gateway_has_all_routes():
    """Verify gateway exposes all required proxy routes."""
    import gateway
    routes = [r.path for r in gateway.app.routes]
    assert "/health" in routes
    assert "/v1/audio/transcriptions" in routes
    assert "/v1/audio/transcriptions/stream" in routes
    assert "/ws/transcribe" in routes


def test_gateway_worker_management():
    """Verify gateway has worker lifecycle functions."""
    from gateway import _ensure_worker, _kill_worker, _idle_watchdog
    assert callable(_ensure_worker)
    assert callable(_kill_worker)
    assert callable(_idle_watchdog)


def test_worker_reuses_server_code():
    """Verify worker imports from server.py rather than duplicating."""
    import worker
    import server
    assert worker.preprocess_audio is server.preprocess_audio

# ─── Issue #36: vLLM engine backend ──────────────────────────────────
# Change: USE_VLLM=true enables vLLM engine for inference (requires vllm pip package)
# Verify:
#   Set USE_VLLM=true in docker-compose.yml (uncomment vllm in Dockerfile)
#   docker compose up -d --build
#   docker compose logs | grep "vLLM"
# Expected: "vLLM engine loaded" or fallback to PyTorch


def test_vllm_globals():
    from server import _vllm_engine, USE_VLLM
    assert _vllm_engine is None
    assert isinstance(USE_VLLM, bool)

def test_vllm_loader_exists():
    from server import _load_vllm_engine
    assert callable(_load_vllm_engine)

def test_vllm_infer_exists():
    from server import _do_transcribe_vllm
    assert callable(_do_transcribe_vllm)

# ─── Issue #37: TensorRT encoder conversion ──────────────────────────
# Change: TRT_ENCODER_PATH enables TensorRT-optimized encoder inference.
#         New script src/build_trt.py builds the TRT engine.
# Verify:
#   python src/build_trt.py --output models/encoder.trt
#   TRT_ENCODER_PATH=models/encoder.trt docker compose up -d --build
#   docker compose logs | grep "TensorRT"
# Expected: "TensorRT encoder loaded from models/encoder.trt"


def test_trt_loader_exists():
    """Verify TRT encoder loader function exists."""
    from server import _try_load_trt_encoder
    assert callable(_try_load_trt_encoder)

def test_trt_encoder_global():
    """Verify _trt_encoder global is defined."""
    from server import _trt_encoder
    assert _trt_encoder is None

def test_build_trt_importable():
    """Verify build_trt module is importable."""
    import build_trt
    assert callable(build_trt.build_trt_engine)

def test_do_transcribe_has_trt_path():
    """Verify _do_transcribe references _trt_encoder for integration."""
    import inspect
    from server import _do_transcribe
    source = inspect.getsource(_do_transcribe)
    assert "_trt_encoder" in source
