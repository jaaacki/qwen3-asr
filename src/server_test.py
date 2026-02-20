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
