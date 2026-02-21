# Roadmap

## Phase 1 — Foundation & Quick Wins → v0.4.0
_The system achieves sub-100ms inference with proper model configuration and zero-waste hot path._

Issues ordered by dependency:
- [x] #18 Ensure model.eval() is called after loading
- [x] #14 Remove per-request release_gpu_memory() calls
- [x] #16 Eliminate unnecessary bytes() copy in WS transcription
- [x] #15 Optimize WebSocket audio preprocessing path
- [x] #17 Enable cudnn.benchmark for conv layer auto-tuning
- [x] #19 Warmup with representative audio instead of silence
- [x] #20 Disable WebSocket per-message-deflate compression
- [x] #13 Add OMP_NUM_THREADS and MKL_NUM_THREADS to Dockerfile
- [x] #11 Enable TensorFloat-32 (TF32) matmul precision
- [x] #9 Enable torch.compile for inference speedup
- [x] #10 Switch to Flash Attention 2
- [x] #12 Replace librosa with torchaudio for audio resampling
- [x] #7 Upgrade to Qwen3-ASR-1.7B model
- [x] #8 Implement repetition detection and fixing
- [x] #22 Use dedicated inference ThreadPoolExecutor
- [x] #21 Pre-allocate pinned memory buffers for GPU transfer
- [x] #23 Implement CUDA stream pipelining for transfer/compute overlap

## Phase 2 — Deep Optimization → v0.5.0
_The system handles production workloads with sub-50ms inference, VAD-gated processing, and priority-based scheduling._

Issues ordered by dependency:
- [x] #24 Add long audio chunking at silence boundaries
- [x] #25 Integrate Silero VAD for voice activity detection
- [x] #26 Reduce WebSocket buffer from 800ms to 400-500ms
- [x] #27 Fix WebSocket/HTTP semaphore contention with priority scheduling
- [x] #28 Implement real SSE streaming via chunked progressive transcription
- [x] #29 INT8 W8A8 quantization with SmoothQuant
- [x] #30 CUDA Graphs for decoder loop
- [x] #31 Export encoder to ONNX Runtime
- [x] #32 Implement KV-cache reuse across WebSocket chunks
- [x] #33 Dual-model strategy: 0.6B for partials, 1.7B for finals
- [x] #34 FP8 quantization for Hopper/Ada GPUs

## Phase 3 — Architecture & Maximum Performance → v0.6.0
_The system scales to concurrent streams with sub-25ms inference via hardware-optimized engines and architectural redesign._

Issues ordered by dependency:
- [x] #35 Gateway + Worker architecture for idle RAM reclamation
- [x] #36 Adopt vLLM engine backend
- [x] #37 TensorRT conversion for encoder
- [x] #38 Speculative decoding for ASR (SpecASR)
- [x] #39 Cache-aware streaming encoder with causal attention
- [x] #40 NUMA-aware CPU pinning
- [x] #41 Evaluate Granian as ASGI server replacement

## Phase 4 — Subtitle Generation → v0.7.0
_The system generates SRT subtitle files with word-level timestamp accuracy via ForcedAligner or lightweight heuristic mode._

- [x] #83 SRT subtitle generation with accurate (ForcedAligner) and fast (heuristic) modes

## Phase 5 — Translation & API Documentation → v0.8.0
_The system supports translation to English and Chinese with a fully documented Swagger UI._

Issues ordered by dependency:
- [ ] #85 [Docs] Expose and document FastAPI Swagger UI
- [ ] #86 [Enhancement] Add `/v1/audio/translations` endpoint

## Backlog
_Unplaced items or future considerations._
- Retire whisper-engine container (Qwen3-ASR-1.7B matches/beats Whisper large-v3)
- Triton Inference Server ensemble pipeline for 100+ concurrent streams
- Decoder distillation (FastWhisper-style) — 5x faster with 1% WER loss
- Head pruning + layer merging (BaldWhisper-style) — 48% smaller, 2x faster
