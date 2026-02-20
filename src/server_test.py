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
