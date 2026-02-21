# Learning Log

Running narrative of decisions, patterns, and lessons.

---

## 2026-02-22 — What just happened: Replacing `print` and `logging` with `loguru` (Issues #87, #88)

**Type**: What just happened
**Related**: Issue #87, #88, v0.9.0

### Preamble
The system originally used generic `print()` statements spanning 5 files, mostly to track loading and timing events. While fast for prototyping, standard stdout prevents automated ingestion and makes it impossible to separate info from errors in dense production logs. Standard python `logging` from Uvicorn and FastAPI also bypassed stdout entirely.

### Action
Added a central `src/logger.py` declaring `loguru` with customized colored console formatting. We built an `InterceptHandler` to consume `uvicorn.access` and `uvicorn.error` logs natively into the new structured format. Standard `print`s in the codebase were cleanly replaced with `log.info`, `log.error`, etc. 

### Why this over alternatives
Standardizing python `logging` implies verbose `logger.getLogger(__name__)` everywhere and manually building `FileHandler` and `StreamHandler` configuration maps. `loguru` works exactly like a generic `print()`, wraps exceptions cleanly asynchronously, handles colored multithreading formats natively, and allows direct JSON structured emission whenever we choose to add a `serialize=True` flag in the future.

### What could go wrong
If additional 3rd-party libraries bypass python runtime standard handlers, they might still spawn rogue output into stderr. 

---

## 2026-02-22 — Why this design: Isolated Translation Module (Issue #86)

**Type**: Why this design
**Related**: Issue #86, v0.8.0

### Context
Users requested the ASR model to automatically translate audio text into English or Chinese directly via the API. The `Qwen3ASRModel` is phenomenal at transcription but we recognized that shoehorning translation directly via pipeline parameters or hacking inference kwargs wouldn't scale well and could destabilize the finely-tuned ASR hot-path. 

### Decision: Dedicated `src/translator.py` wrapper using external APIs
Instead of loading an additional multi-gigabyte local LLM explicitly for translation or mutating the ASR model's internals, we implemented an isolated Python module `src/translator.py`.
1. **Separation of Concerns** — The primary endpoints naturally run `_do_transcribe()` exactly as before with zero structural changes. The returned text/SRT is then subsequently passed into standard translation logic asynchronously.
2. **Standard Interfaces** — Using the `openai` python library ensures we can connect the translation pipeline to any OpenAI-compatible backend, such as a local lightweight `Ollama` instance or a powerful external `vLLM` server, entirely controlled via external `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables.
3. **Dual Formats (JSON & SRT)** — The endpoint was structured so users can supply `response_format=srt`. When enabled, the transcription pipeline runs in *accurate* subtitle mode first, translates the raw `SRT` format while strictly maintaining indices and timing tags via an LLM instruction prompt, and returns a perfectly encoded translated SRT file.

### What could go wrong
- **LLM Prompt Misfires (SRT)**: Language models are notoriously tricky at rigidly following formats. If the `TRANSLATE_MODEL` decides to output markdown formatting (e.g., ` ```srt `) or alters timestamp lines ("00:00:01,000" -> "00:00:01.000"), the resulting SRT will be malformed. We deployed stripping logic for the markdown blocks, but robust SRT integrity validation might be needed later.

---
## 2026-02-21 — Why this design: Subtitle generation as separate module (Issue #83)

**Type**: Why this design
**Related**: Issue #83, v0.6.2

### Context
Users need SRT subtitle files from audio transcriptions. The ASR model produces segment-level text, but subtitles require word-level timestamps, line-length constraints, and proper timing. Two accuracy levels are needed: production-grade (ForcedAligner) and lightweight (heuristic).

### Decision: Separate subtitle.py module
All subtitle logic lives in `src/subtitle.py` rather than being embedded in `server.py`. Reasons:
1. **Testability** — 41 unit tests run without FastAPI, GPU, or model dependencies. Pure functions with clear inputs/outputs.
2. **Independence** — subtitle segmentation, timing enforcement, and SRT formatting have no coupling to the HTTP server or inference queue.
3. **ForcedAligner lifecycle** — the aligner is a separate 2GB model with its own load/unload cycle. Keeping it in a module-level global with `load_aligner()`/`unload_aligner()` functions lets server.py manage it alongside the main model's idle timeout.

### Decision: Two-mode design (accurate vs fast)
- **Accurate mode**: Uses Qwen3-ForcedAligner-0.6B for ~33ms word-level timestamps. Lazy-loaded on first request. Handles the 5-minute aligner limit by chunking audio at boundaries and falling back to heuristic estimation per-chunk if alignment fails.
- **Fast mode**: Distributes segment duration across words proportionally by character count. No aligner loaded, no extra VRAM, faster processing. ~200ms accuracy is sufficient for most subtitle use cases.

The mode is a per-request parameter, not a server-wide config. Users can mix fast and accurate requests.

### Decision: CJK tokenization
CJK text (Chinese, Japanese, Korean) requires character-level tokenization for subtitle segmentation, unlike English which uses whitespace. The `_tokenize()` function detects CJK characters and splits them individually while keeping Latin words intact for mixed-language text. The subtitle joiner is empty string for CJK vs space for Latin. This was identified by the critic review and added in the corrections round.

### What could go wrong
- **ForcedAligner version mismatch**: The aligner expects the same text format as the ASR model output. If the ASR model's tokenization changes, alignment could silently degrade.
- **Long audio chunk stitching**: When audio exceeds 5 minutes, each chunk is aligned independently. Word timestamps at chunk boundaries may have discontinuities. The per-chunk fallback to heuristic estimation prevents crashes but reduces accuracy.
- **CJK subtitle length**: CJK characters are typically 2 display columns wide but counted as 1 char. The 42-char max_line_chars may produce visually long lines for CJK. A future enhancement could use display width instead of character count.

---

## 2026-02-21 — Why this design: Lazy imports + SDK cleanup (Issue #82)

**Type**: Why this design
**Related**: Issue #82, v0.6.1

### Context
Container used 2.4GB RAM at idle (model not loaded) because `server.py` imported torch, transformers, and qwen_asr at module top-level. Additionally, server.py reimplemented audio preprocessing, long-audio chunking, and processor loading that the SDK already handles inside `model.transcribe()`.

### Decision
Two changes combined:
1. **Lazy imports** — move all heavy imports into the functions that use them. Idle RAM drops to ~50-100MB (just FastAPI + uvicorn + numpy). Cold start adds ~3-5s to first request (acceptable since model loading already takes ~15s).
2. **Remove SDK redundancies** — delete `preprocess_audio()`, `preprocess_audio_ws()`, `chunk_audio_at_silence()`, and separate `AutoProcessor` loading. The SDK's `model.transcribe()` handles normalization, resampling, and chunking internally.

### Key insight
The SDK's `split_audio_into_chunks()` is measurably superior to our `chunk_audio_at_silence()`: sliding window convolution with +/-5s search range vs simple RMS threshold. Our code was double-processing audio — preprocessing before passing to `transcribe()`, which preprocessed again internally.

**Principle adopted:** Our code optimizes; the SDK transcribes. If `model.transcribe()` does it, delete our version.

### What could go wrong
- **Health endpoint torch import:** Solved by using `sys.modules.get("torch")` — health checks don't trigger the 2.4GB import.
- **SDK behavior changes:** If a future SDK version changes normalization or chunking, transcription quality could degrade without any code change on our side. Mitigation: pin SDK version, E2E accuracy tests.
- **WebSocket normalization:** Old `preprocess_audio_ws()` did peak normalization. SDK handles this internally, but if the SDK's approach differs, quiet audio might transcribe differently.
- **Worker import ordering:** worker.py imports from server.py which now does lazy imports. If worker-side code touches torch symbols before `_ensure_model_loaded()` runs, it will NameError. Currently safe but fragile.

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

---

## 2026-02-20 — What just happened: Phase 3 complete (7 issues merged)

**Type**: What just happened
**Related**: Phase 3, milestone/phase-3 -> main merge

### Pattern
Phase 3 issues were merged sequentially in dependency order: gateway/worker (#35) first (foundational architecture), then vLLM (#36) and TensorRT (#37) which add alternative inference backends, then speculative decoding (#38) which builds on dual-model from Phase 2, causal encoder (#39), NUMA pinning (#40), and finally Granian (#41) which only touches the Dockerfile CMD.

### Key architecture change: _do_transcribe dispatch chain
By the end of Phase 3, `_do_transcribe()` has a layered dispatch chain at the top:
1. vLLM check — if `USE_VLLM=true` and engine loaded, delegates to `_do_transcribe_vllm()`
2. Speculative check — if `USE_SPECULATIVE=true` and fast model loaded, delegates to `_do_transcribe_speculative()`
3. Standard path — pinned memory, dual model selection, TRT/ONNX encoder monkey-patching, CUDA stream

Each dispatch is mutually exclusive: vLLM replaces the entire inference path, speculative uses both models directly, and the standard path uses the encoder acceleration stack.

### Lesson learned: milestone branch push required
The previous session merged PRs #72, #70, #67 into milestone/phase-3 locally but didn't push to origin, causing the worktree-based rebase for PR #65 to target the wrong base (main instead of milestone/phase-3 with all three PRs). The fix was to verify `origin/milestone/phase-3` matches the local branch after each merge. Always `git push origin milestone/phase-3` after squash-merging a PR.

### Lesson learned: opt-in via environment variables
All Phase 3 features are gated by environment variables (USE_VLLM, USE_SPECULATIVE, USE_CAUSAL_ENCODER, TRT_ENCODER_PATH, NUMA_NODE, USE_GRANIAN, GATEWAY_MODE). This pattern kept the default behavior unchanged and avoided breaking changes. Each feature can be independently enabled for testing.
