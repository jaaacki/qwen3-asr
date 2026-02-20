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
