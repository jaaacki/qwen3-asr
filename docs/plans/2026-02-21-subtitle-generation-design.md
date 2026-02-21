# Subtitle Generation Design

**Date:** 2026-02-21
**Status:** Approved

## Goal

Add SRT subtitle generation to the Qwen3-ASR server without compromising real-time WebSocket transcription. Two modes: accurate (with ForcedAligner, ~33ms word-level timestamps) and fast (segment-level heuristics, no aligner).

## Requirements

- **Output:** SRT format only
- **API:** New `POST /v1/audio/subtitles` endpoint
- **Aligner loading:** Lazy on first accurate-mode request (~2GB VRAM)
- **No diarization** in v1
- **Zero impact** on WebSocket real-time path

## Architecture

### Data Flow

```
POST /v1/audio/subtitles
  |
  v
server.py — endpoint handler, audio decode, queue submission
  |  priority=1 (same as HTTP transcription)
  v
_do_transcribe() — existing ASR inference (text + optional timestamps)
  |
  v
subtitle.py:generate_srt()
  ├── _load_aligner()        — lazy-load ForcedAligner on first call
  ├── _align_chunks()        — 5-min chunked forced alignment with offset stitching
  ├── _segment_subtitles()   — group words into subtitle blocks
  ├── _enforce_timing()      — duration/gap constraints
  └── _format_srt()          — valid SRT string output
```

### New Files

| File | Purpose | ~Lines |
|------|---------|--------|
| `src/subtitle.py` | Aligner loading, segmentation, SRT formatting | ~300 |
| `src/subtitle_test.py` | Unit tests for segmentation and formatting | ~150 |
| `E2Etest/test_subtitle.py` | E2E tests against running container | ~100 |

### Modifications to Existing Files

| File | Change |
|------|--------|
| `src/server.py` | Add `/v1/audio/subtitles` endpoint (~40 lines). Add aligner unload call in `_unload_model_sync()`. Apply VAD gating to subtitle path. |
| `src/gateway.py` | Proxy the new endpoint to worker |
| `src/worker.py` | Forward the new endpoint |

## API Design

```
POST /v1/audio/subtitles

Form fields:
  file: UploadFile          (required) — audio file
  language: str = "auto"    — language code or "auto"
  mode: str = "accurate"    — "accurate" (ForcedAligner) or "fast" (heuristic)
  max_cps: int = 20         — max characters per second (reading speed)
  max_line_chars: int = 42  — max characters per subtitle line

Response (200):
  Content-Type: text/plain; charset=utf-8
  Content-Disposition: attachment; filename="subtitles.srt"
  Body: raw SRT content

Response (504):
  {"error": "Subtitle generation timed out"}
```

## subtitle.py Module Design

### ForcedAligner Lifecycle

- Module-level global `_aligner = None`
- `_load_aligner()`: lazy-loads `Qwen3-ForcedAligner-0.6B` on first accurate-mode call
- Model ID configurable via `FORCED_ALIGNER_ID` env var
- `unload_aligner()`: called by `server.py:_unload_model_sync()` during idle unload
- Aligner runs on same device as ASR model (CUDA if available)

### 5-Minute Chunk Alignment

The ForcedAligner has a 5-minute limit per call. For longer audio:

1. Identify silence boundaries near 5-minute marks using `is_speech()` VAD
2. Split audio at those boundaries
3. Align each chunk independently via `aligner.align(audio, text, language)`
4. Add cumulative time offset to each chunk's word timestamps
5. Concatenate results

### Subtitle Segmentation (`_segment_subtitles()`)

Input: list of `(word, start_time, end_time)` tuples

Algorithm:
1. Accumulate words into a current subtitle block
2. Break to a new subtitle when any of:
   - Adding the next word would exceed `max_line_chars * 2` (two lines)
   - A sentence-ending punctuation mark is reached (`. ? ! ;`)
   - A pause > 500ms between consecutive words
   - Current block duration would exceed 7 seconds
3. Within a block, split into two lines at the best break point:
   - After punctuation (comma, semicolon, colon)
   - Before conjunctions/prepositions
   - Fallback: split near the midpoint
4. Prefer bottom-heavy layout (shorter top line)

### Timing Enforcement (`_enforce_timing()`)

Post-process subtitle events:
- **Min duration:** Extend end_time to ensure >= 833ms display time
- **Max duration:** Already enforced by segmentation (7s cap)
- **Min gap:** If gap between consecutive subtitles < 83ms, snap the out-cue of the previous subtitle to create an 83ms gap
- **No overlaps:** Truncate previous subtitle's end_time if it overlaps the next subtitle's start_time

### SRT Formatting (`_format_srt()`)

```
1
00:00:01,200 --> 00:00:04,500
First line of subtitle text
Second line if needed

2
00:00:05,000 --> 00:00:08,300
Next subtitle block
```

### Fast Mode

When `mode=fast`:
- Skip ForcedAligner entirely — no aligner loaded
- Use segment-level boundaries from ASR output
- Estimate word timing by distributing segment duration proportionally across words by character count
- Apply same segmentation and timing enforcement
- Less precise (~200ms accuracy) but no extra VRAM, faster processing

## VAD Gating for Subtitle HTTP Path

Apply `is_speech()` from existing Silero VAD before subtitle inference. For the subtitle endpoint specifically:
- Run VAD on each segment before aligner processing
- Skip alignment on silent segments (prevents hallucinated subtitle entries)
- This extends existing WebSocket-only VAD usage to the subtitle pipeline

## What Stays Untouched

- WebSocket endpoint and its entire path
- Existing `/v1/audio/transcriptions` endpoint
- PriorityInferQueue logic
- Model loading (except adding aligner lazy-load hook)
- All existing optimizations (torch.compile, pinned memory, CUDA streams, etc.)

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FORCED_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | HuggingFace model ID for the aligner |

No other new env vars. `max_cps` and `max_line_chars` are per-request parameters.

## Success Criteria

1. `POST /v1/audio/subtitles` returns valid SRT for any supported audio file
2. Word-level timestamp accuracy < 50ms in accurate mode (aligner benchmarks show ~33ms)
3. All subtitle blocks respect CPS, line length, and duration constraints
4. WebSocket latency is unaffected (priority queue isolation)
5. Aligner loads only when needed, unloads with main model on idle
6. Fast mode works without loading aligner
7. No hallucinated subtitles on silent segments (VAD gating)
