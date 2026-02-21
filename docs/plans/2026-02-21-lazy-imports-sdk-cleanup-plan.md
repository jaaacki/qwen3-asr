# Lazy Imports + SDK Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce idle container RAM from 2.4GB to <100MB by deferring heavy imports AND remove ~100 lines of redundant code that duplicates SDK functionality.

**Architecture:** Move all heavy imports (`torch`, `soundfile`, `qwen_asr`) into functions that need them (primarily `_load_model_sync()`). Delete preprocessing/chunking code that the SDK already handles inside `model.transcribe()`. Add `TORCHINDUCTOR_COMPILE_THREADS=1` to limit inductor workers.

**Tech Stack:** Python (lazy imports, `from __future__ import annotations`), Docker (env vars), FastAPI

**Design:** See `docs/plans/2026-02-21-lazy-imports-sdk-cleanup-design.md`

---

## Task 1: Remove Redundant Functions

Remove `preprocess_audio()`, `preprocess_audio_ws()`, and `chunk_audio_at_silence()` from `server.py`. The SDK's `model.transcribe()` handles all of this internally.

**Files:**
- Modify: `src/server.py` — lines 209-312 (three functions)

**Step 1: Delete `preprocess_audio()`**

Remove the entire function at lines 209-237. This function does mono conversion, float32 cast, resampling via torchaudio, and peak normalization — all of which `model.transcribe()` does internally via its `normalize_audios()` pipeline.

**Step 2: Delete `preprocess_audio_ws()`**

Remove the entire function at lines 240-245. WebSocket audio is already 16kHz mono float32; the only thing this did was peak normalize, which the SDK also handles.

**Step 3: Delete `chunk_audio_at_silence()`**

Remove the entire function at lines 248-312. The SDK's `split_audio_into_chunks()` (called inside `transcribe()`) uses a superior sliding-window energy convolution algorithm with +/-5s search range and handles up to 20 minutes of audio.

**Step 4: Verify no remaining references**

Search for `preprocess_audio`, `preprocess_audio_ws`, and `chunk_audio_at_silence` in the codebase. Any references found will be fixed in subsequent tasks.

Run: `grep -rn "preprocess_audio\|chunk_audio_at_silence" src/`

**Step 5: Commit**

```bash
git add src/server.py
git commit -m "refactor: remove preprocess_audio, preprocess_audio_ws, chunk_audio_at_silence (#82)

SDK's model.transcribe() handles audio normalization and chunking
internally with a superior algorithm. Our versions were redundant."
```

---

## Task 2: Simplify HTTP Transcription Endpoint

Remove the manual chunking loop and preprocessing from the `/v1/audio/transcriptions` endpoint. Pass raw audio directly to `_do_transcribe()` — let the SDK handle chunking.

**Files:**
- Modify: `src/server.py` — the `transcribe()` endpoint (lines 682-720)

**Step 1: Rewrite the endpoint**

Replace the current body of the `transcribe()` function. The current version calls `sf.read()`, `preprocess_audio()`, conditionally calls `chunk_audio_at_silence()`, loops over chunks calling `_do_transcribe()` for each, then joins results. The new version should:

1. Read audio bytes from upload
2. Decode with `sf.read()` (still needed — model accepts `(ndarray, sr)` tuple)
3. Single call to `_do_transcribe(audio, sr, lang_code, return_timestamps)` via the priority queue
4. Apply word-level `detect_and_fix_repetitions()` on the result

```python
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False)
):
    await _ensure_model_loaded()

    audio_bytes = await file.read()
    import soundfile as sf_mod
    audio, sr = sf_mod.read(io.BytesIO(audio_bytes))

    lang_code = None if language == "auto" else language

    try:
        results = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, return_timestamps),
                priority=1,
            ),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

    if results and len(results) > 0:
        text = detect_and_fix_repetitions(results[0].text)
        language_code = results[0].language
    else:
        text = ""
        language_code = lang_code or language

    return {"text": text, "language": language_code}
```

**Step 2: Simplify SSE streaming endpoint similarly**

The `/v1/audio/transcriptions/stream` endpoint at lines 914-937 also calls `preprocess_audio()`. Remove that call — just `sf.read()` and pass through.

```python
@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False)
):
    """Streaming transcription endpoint using Server-Sent Events (SSE)."""
    await _ensure_model_loaded()

    audio_bytes = await file.read()
    import soundfile as sf_mod
    audio, sr = sf_mod.read(io.BytesIO(audio_bytes))

    lang_code = None if language == "auto" else language

    return StreamingResponse(
        sse_transcribe_generator(audio, sr, lang_code, return_timestamps),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
```

**Step 3: Remove `preprocess_audio_ws()` call from `_transcribe_with_context()`**

In `_transcribe_with_context()` (line 1132), remove the call to `preprocess_audio_ws(audio)`. The audio is already float32 from the int16 conversion on line 1129, and the SDK handles normalization.

Change line 1131-1133 from:
```python
        # Fast path: WS audio is already mono, float32, at 16kHz
        audio = preprocess_audio_ws(audio)
        sr = TARGET_SR
```
To:
```python
        sr = TARGET_SR
```

**Step 4: Commit**

```bash
git add src/server.py
git commit -m "refactor: simplify HTTP/SSE endpoints, remove redundant preprocessing (#82)

Let SDK handle audio normalization and chunking via model.transcribe().
HTTP endpoint no longer manually chunks audio or preprocesses."
```

---

## Task 3: Remove Separate Processor Loading

The `processor` global and `AutoProcessor` import are redundant — `Qwen3ASRModel.from_pretrained()` loads the processor internally. Remove the separate loading.

**Files:**
- Modify: `src/server.py`

**Step 1: Remove the `AutoProcessor` import**

Delete line 18: `from qwen_asr.inference.qwen3_asr import AutoProcessor`

**Step 2: Remove `processor` from globals and `_load_model_sync()`**

Remove `processor = None` from the module-level globals (line 32).

In `_load_model_sync()`, remove line 442:
```python
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

**Step 3: Remove `processor` from `_unload_model_sync()`**

In `_unload_model_sync()` (lines 603-623), remove the lines that delete `processor`:
```python
    del processor
    ...
    processor = None
```

And remove `processor` from the `global` statement on line 605.

**Step 4: Search for remaining `processor` references**

Run: `grep -n "processor" src/server.py`

Verify no code uses `processor` directly. The SDK's model object has its own internal processor.

**Step 5: Commit**

```bash
git add src/server.py
git commit -m "refactor: remove separate processor loading (#82)

Qwen3ASRModel.from_pretrained() loads the processor internally.
The standalone processor global was never used by inference code."
```

---

## Task 4: Lazy Imports — Move Heavy Imports into Functions

Move `torch`, `soundfile`, and `qwen_asr` imports from module top-level into the functions that use them. This is the core RAM reduction.

**Files:**
- Modify: `src/server.py`

**Step 1: Add `from __future__ import annotations` at top of file**

This must be the very first import (line 1, before everything else). It makes all type annotations strings by default, so `torch.Tensor` in type hints won't require torch to be imported.

**Step 2: Remove heavy top-level imports**

Remove these lines from the top of server.py:
```python
import torch                  # line 4
import soundfile as sf        # line 5
from qwen_asr import Qwen3ASRModel, parse_asr_output   # line 17
```

**Step 3: Remove `_ATTN_IMPL` module-level evaluation**

Remove the `_get_attn_implementation()` function (lines 20-26) and its call `_ATTN_IMPL = _get_attn_implementation()` (line 28). Move the logic into `_load_model_sync()` as a local call, since attention implementation is only needed when loading the model.

**Step 4: Change module-level type hints to strings**

Line 124: `_PINNED_AUDIO_BUFFER: torch.Tensor | None = None`
→ `_PINNED_AUDIO_BUFFER: "torch.Tensor | None" = None`

With `from __future__ import annotations`, this becomes automatic. But the actual initialization of `_PINNED_AUDIO_BUFFER` (which calls `torch.zeros(...)`) already happens inside `_load_model_sync()`, so the module-level declaration is just `None`. Same for `_cuda_stream`.

However, since `from __future__ import annotations` is added, these type hints are automatically deferred. No string quoting needed — just make sure the variable is initialized to `None` (not a torch call).

Confirm lines 124 and 128 only set `= None` at module level (they do).

**Step 5: Add lazy imports to every function that uses torch/sf/qwen_asr**

Functions that need `import torch`:
- `release_gpu_memory()` — uses `torch.cuda`
- `is_speech()` — uses `torch.from_numpy`, `torch.no_grad`
- `_try_load_trt_encoder()` — uses `torch.jit.load`
- `_load_model_sync()` — uses torch extensively
- `_unload_model_sync()` — uses `torch.cuda`
- `_do_transcribe()` — uses `torch.inference_mode`, `torch.from_numpy`, `torch.cuda`
- `_do_transcribe_speculative()` — uses `torch.inference_mode`
- `_try_build_cuda_graph()` — uses `torch.cuda`
- `health()` — uses `torch.cuda`
- `_transcribe_with_context()` — no direct torch usage (delegates to `_do_transcribe`)
- `_patch_encoder_causal()` — no torch usage (just attribute access)

Functions that need `import soundfile`:
- `transcribe()` endpoint — uses `sf.read()`
- `transcribe_stream()` endpoint — uses `sf.read()`

Functions that need `from qwen_asr import ...`:
- `_load_model_sync()` — uses `Qwen3ASRModel.from_pretrained()`
- `_do_transcribe()` — no direct usage (calls `model.transcribe()` via the global)
- Result parsing uses `parse_asr_output` — but this is called by the SDK internally. Check if server.py calls it directly.

Run: `grep -n "parse_asr_output" src/server.py`

If server.py doesn't call `parse_asr_output` directly, the import can be removed entirely.

Add `import torch` at the top of each function listed above. Pattern:
```python
def release_gpu_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        ...
```

For `soundfile`, use local import in each endpoint:
```python
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))
```

For `qwen_asr`, add to `_load_model_sync()`:
```python
    from qwen_asr import Qwen3ASRModel
```

**Step 6: Commit**

```bash
git add src/server.py
git commit -m "feat: lazy-import torch/soundfile/qwen_asr for idle RAM reduction (#82)

Heavy ML libraries are now imported on first use instead of at module
load time. Idle container RAM drops from ~2.4GB to ~50-100MB.
Adds from __future__ import annotations for deferred type hints."
```

---

## Task 5: Update worker.py

`worker.py` imports from `server.py` and will be affected by our changes. It also has a pre-existing bug: it imports `_infer_semaphore` which no longer exists (replaced by `_infer_queue`). It also imports `preprocess_audio` which we deleted.

**Files:**
- Modify: `src/worker.py`

**Step 1: Fix the imports**

Replace the import block at lines 8-22:

```python
from server import (
    release_gpu_memory,
    _load_model_sync,
    _do_transcribe,
    _idle_watchdog,
    _ensure_model_loaded,
    _infer_queue,
    _transcribe_with_context,
    detect_and_fix_repetitions,
    TARGET_SR,
    REQUEST_TIMEOUT,
    WS_BUFFER_SIZE,
    WS_OVERLAP_SIZE,
)
import server as _srv
```

Key changes:
- Remove `preprocess_audio` (deleted)
- Remove `_infer_semaphore` (doesn't exist, was `_infer_queue`)
- Add `_infer_queue` (the actual replacement)
- Remove `model` (access via `_srv.model`)
- Add `detect_and_fix_repetitions` (for word-level post-processing)

**Step 2: Remove `import torch` from top-level**

Line 30: `import torch` — remove. Worker.py's top-level code doesn't need torch. When it imports from server.py, torch gets loaded lazily via the server functions.

Similarly, remove `import soundfile as sf` from top-level. Add it locally in the two endpoints that use it.

**Step 3: Update the `/transcribe` endpoint**

Remove the `preprocess_audio()` call. The current code does:
```python
audio, sr = sf.read(io.BytesIO(audio_bytes))
audio, sr = preprocess_audio(audio, sr)  # DELETE THIS
```

Replace `_infer_semaphore` with `_infer_queue`:
```python
    try:
        results = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code, return_timestamps),
                priority=1,
            ),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        ...
```

**Step 4: Update the `/transcribe/stream` endpoint**

Same pattern — remove `preprocess_audio()` call.

**Step 5: Commit**

```bash
git add src/worker.py
git commit -m "fix: update worker.py imports after server.py cleanup (#82)

Remove deleted preprocess_audio import, fix _infer_semaphore -> _infer_queue,
remove top-level torch/soundfile imports."
```

---

## Task 6: Dockerfile — Inductor Thread Limit

Add `TORCHINDUCTOR_COMPILE_THREADS=1` to limit torch.compile from spawning 20 worker subprocesses (~800MB savings).

**Files:**
- Modify: `Dockerfile`

**Step 1: Add env var**

After the existing `MKL_NUM_THREADS=1` line (line 13), add:

```dockerfile
# Limit torch.compile inductor workers (default 20 spawns ~800MB of subprocesses)
ENV TORCHINDUCTOR_COMPILE_THREADS=1
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "perf: limit torch inductor compile threads to 1 (#82)

Prevents torch.compile from spawning ~20 worker subprocesses at import
time, saving ~800MB of shared memory at idle."
```

---

## Task 7: Update CLAUDE.md and Documentation

Update project docs to reflect the simplified architecture.

**Files:**
- Modify: `CLAUDE.md` — update architecture section
- Modify: `CHANGELOG.md` — add entry
- Modify: `LEARNING_LOG.md` — add decision entry

**Step 1: Update CLAUDE.md**

In the architecture section, remove references to:
- `preprocess_audio()` and `chunk_audio_at_silence()`
- Manual chunking in HTTP endpoint
- `processor` global

Add a note in the architecture section:
```
Audio preprocessing and long-audio chunking are handled natively by the SDK's
`model.transcribe()`. server.py only adds server-level concerns (priority queue,
WebSocket streaming, GPU optimizations, idle lifecycle).
```

Update the "Audio Preprocessing Pipeline" subsection to note that it's handled by the SDK.

Update the file organization to note the reduced line count.

**Step 2: Add CHANGELOG entry**

Add under the next version:
```markdown
### Changed
- Deferred heavy imports (torch, soundfile, qwen_asr) to first request — idle RAM reduced from ~2.4GB to ~50-100MB
- Removed redundant `preprocess_audio()`, `preprocess_audio_ws()`, `chunk_audio_at_silence()` — SDK handles these internally
- Removed separate `AutoProcessor` loading — SDK loads processor inside `Qwen3ASRModel.from_pretrained()`
- Simplified HTTP transcription endpoint (removed manual chunking loop)
- Limited torch inductor compile threads to 1 (saves ~800MB idle)

### Fixed
- worker.py: fixed import of non-existent `_infer_semaphore` (now `_infer_queue`)
```

**Step 3: Add LEARNING_LOG entry**

Entry type: "why this design"
```
## Lazy Imports + SDK Cleanup (Issue #82)

**Decision:** Defer all heavy ML imports and delete ~100 lines of redundant preprocessing/chunking code.

**Why:** The qwen_asr SDK's `model.transcribe()` already handles audio normalization (mono, resample, float32), long-audio chunking (superior sliding-window algorithm, up to 20min), and character-level repetition detection internally. Our server.py was reimplementing all of this, then passing already-processed audio to `transcribe()` which would process it again. The SDK's chunking algorithm is measurably better (sliding window convolution with +/-5s search vs our simple RMS threshold).

**Principle:** Our code optimizes; the SDK transcribes. If `model.transcribe()` does it, we delete our version.

**Trade-off:** Cold start adds ~3-5s (one-time import cost on top of ~15s model load). Acceptable because it only affects the first request after idle timeout, and model loading already dominates.
```

**Step 4: Commit**

```bash
git add CLAUDE.md CHANGELOG.md LEARNING_LOG.md
git commit -m "docs: update docs for lazy imports + SDK cleanup (#82)"
```

---

## Task 8: Rebuild and Run E2E Tests

Rebuild the container and run the E2E test suite to verify nothing broke.

**Step 1: Rebuild container**

```bash
docker compose up -d --build
```

Wait for container to be healthy:
```bash
curl http://localhost:8100/health
```

Expected: `{"status": "ok", "model_loaded": false, ...}`

**Step 2: Run smoke tests**

```bash
pytest E2Etest/ -m smoke -v
```

Expected: All smoke tests pass (health endpoint works, basic connectivity).

**Step 3: Run HTTP transcription tests**

```bash
pytest E2Etest/test_api_http.py -v
```

Expected: All HTTP tests pass. Transcription results should be identical or better (SDK chunking is superior).

**Step 4: Run WebSocket tests**

```bash
pytest E2Etest/test_websocket.py -v
```

Expected: WebSocket streaming tests pass. The WS path was minimally changed (only removed `preprocess_audio_ws()` call).

**Step 5: Run full suite**

```bash
pytest E2Etest/ -v
```

Report: `Tests: X passed, Y failed, Z skipped` with any failure details.

**Step 6: Verify idle RAM reduction**

After tests complete, wait for idle timeout (120s) and check container memory:

```bash
docker stats --no-stream qwen3_asr
```

Expected: MEM USAGE should be significantly lower than the previous ~2.4GB (target: <100MB after model unloads, but import-time overhead means ~50-100MB).

---

## Summary

| Task | What | Lines Changed |
|------|------|---------------|
| 1 | Delete redundant functions | -104 lines |
| 2 | Simplify HTTP/SSE endpoints | -30 lines |
| 3 | Remove processor loading | -5 lines |
| 4 | Lazy imports | ~+30 local imports, -5 top-level |
| 5 | Fix worker.py | ~10 lines |
| 6 | Dockerfile inductor threads | +2 lines |
| 7 | Docs | Update 3 files |
| 8 | E2E validation | No code changes |

Net: ~-100 lines of code, idle RAM from ~2.4GB to ~50-100MB.
