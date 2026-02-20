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
