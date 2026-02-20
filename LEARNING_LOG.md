# Learning Log

Running narrative of decisions, patterns, and lessons.

---

## 2026-02-20 — Why this design: Three-phase optimization roadmap

**Type**: Why this design
**Related**: Phase 1-3 planning

### Context
The qwen3-asr server needed a structured approach to real-time optimization. After a deep analysis of the WebSocket critical path (documented in improvements.md), we identified that the current ~150ms per-chunk latency has multiple independent sources of waste — from Python garbage collection after every inference (5-15ms) to redundant memory copies and missing hardware-level optimizations.

### Decision
Organized improvements into three phases by effort/risk/impact:
- **Phase 1 (Quick Wins)**: Zero-risk changes that recover wasted latency — removing per-request gc.collect(), fixing redundant copies, enabling hardware features (TF32, cudnn.benchmark, Flash Attention). Target: sub-100ms.
- **Phase 2 (Deep Optimization)**: Medium-effort changes requiring new dependencies or architecture adjustments — VAD, quantization, CUDA Graphs, ONNX Runtime. Target: sub-50ms.
- **Phase 3 (Architecture)**: High-effort changes that fundamentally reshape the system — TensorRT, speculative decoding, Gateway+Worker, vLLM. Target: sub-25ms.

### Why this ordering
Phase 1 items are ordered so that the easiest, lowest-risk changes land first and immediately improve the baseline. model.eval() and removing gc.collect() are pure corrections that should never have been missing. TF32 and cudnn.benchmark are one-line hardware unlocks. torch.compile and Flash Attention 2 require more validation but are well-understood.

Phase 2 builds on the clean baseline from Phase 1 — quantization and CUDA Graphs require a stable inference path to profile against. VAD is placed before buffer reduction because it changes the effective workload.

Phase 3 is deliberately last because these changes are partially mutually exclusive (vLLM replaces much of the manual optimization from Phase 1-2) and require the most validation.

### What could go wrong
- torch.compile may not be compatible with the Qwen3-ASR model's generate() method (dynamic control flow)
- Flash Attention 2 installation in Docker may conflict with the CUDA 12.4 base image
- INT8 quantization may degrade accuracy on edge cases (accented speech, code-switching)
- The dual-model strategy (Phase 2) may not fit in VRAM alongside INT8 — needs careful memory planning

---

## 2026-02-20 — What just happened: Critical path analysis of WebSocket hot path

**Type**: What just happened
**Related**: improvements.md Section 4

### Pattern
Traced every step of a WebSocket audio chunk from recv to response. Discovered that non-inference overhead accounts for ~15-25% of total latency:
- release_gpu_memory() per request: 5-15ms (the worst offender)
- Thread pool dispatch: ~0.3ms
- Redundant preprocessing: ~0.15ms
- Unnecessary bytes() copy: ~0.05ms

### Aha moment
The gc.collect() + torch.cuda.empty_cache() pattern — commonly seen in tutorials and Stack Overflow answers — is actively harmful for real-time workloads. It's appropriate for notebook-style single-shot inference but defeats PyTorch's caching allocator in a server context. The allocator caches freed blocks specifically to avoid cudaMalloc/cudaFree overhead on the next request. empty_cache() throws away that cache.

### What could go wrong
Without periodic empty_cache(), PyTorch's reserved-but-unused memory will show as "allocated" in nvidia-smi even though it's available to PyTorch. This may look like a memory leak but isn't. Only matters if another process on the same GPU needs VRAM — in which case the idle unload watchdog already handles this by unloading the entire model.

---

## 2026-02-20 — What just happened: Phase 1 complete (15 issues merged)

**Type**: What just happened
**Related**: Phase 1, milestone/phase-1 → main merge

### Pattern
All 15 Phase 1 issues were implemented in parallel by 5 builders, then merged sequentially into milestone/phase-1 in strict dependency order to prevent server.py conflicts. The merge order grouped changes by the area of server.py they touched:
1. Model config (top of _load_model_sync)
2. Hot path (_do_transcribe, _transcribe_with_context, WS handler)
3. Audio pipeline (preprocess_audio, warmup, new functions)
4. Major model changes (model creation, attn_implementation)
5. Threading (executor, pinned memory, CUDA stream globals)

### Lesson learned
Every builder created src/server_test.py from scratch in their first PR, causing add/add conflicts on every subsequent merge. Future phases should have builders append to the existing file instead of creating it new. The architect ended up resolving most of these conflicts manually to keep the pipeline moving.

### Aha moment
The pinned memory + CUDA stream combo in _do_transcribe() creates a complete async DMA pipeline: audio data is copied into page-locked memory, then the inference runs on a dedicated CUDA stream. This should enable transfer/compute overlap when profiled with nsys, though the benefit is harder to measure without a GPU profiling setup.

---

## 2026-02-20 — What just happened: Phase 2 complete (11 issues + 2 fixes merged)

**Type**: What just happened
**Related**: Phase 2, milestone/phase-2 -> main merge

### Pattern
Phase 2 issues were split into two builder groups: "basic" (issues #26, #28, #24, #25) and "advanced" (issues #27, #29, #30, #31, #32, #33, #34). The basic group touched the main transcription paths while the advanced group added new opt-in features. This separation worked well — the basic PRs had predictable conflicts in _do_transcribe and the WS handler, while advanced PRs mostly added new code paths gated by environment variables.

### Key architecture change: PriorityInferQueue
The biggest structural change was replacing `asyncio.Semaphore(1)` with a `PriorityInferQueue` backed by a min-heap. This required updating all inference call sites (HTTP transcribe, SSE streaming, WS transcription) to use `_infer_queue.submit(fn, priority=N)` instead of `async with _infer_semaphore:`. WS gets priority 0 (higher), HTTP/SSE get priority 1. The queue worker runs on the same `_infer_executor` ThreadPoolExecutor.

### Lesson learned: _do_transcribe grew complex
By the end of Phase 2, `_do_transcribe()` handles: pinned memory buffer, fast model selection (dual-model), ONNX encoder monkey-patching, CUDA stream routing, and fallback paths. Each Phase 2 PR added one concern, but the final function has 6 conditional branches. The `_run_transcribe()` inner function (from the ONNX fix) helped reduce duplication. Future refactoring could extract model dispatch to a separate strategy.

### Lesson learned: fix PRs from critic
The critic caught two real issues after merges: a duplicate `_infer_executor` definition (from the original Phase 1 code surviving alongside the priority queue's executor) and the ONNX session being loaded but never wired into inference. Both were fixed with small follow-up PRs merged into milestone/phase-2 before the milestone PR to main.
