# Sliding Window WebSocket Streaming — Design

## Problem

The current WebSocket `/ws/transcribe` transcribes each ~450ms chunk independently
with only 150ms overlap. This gives ultra-low latency (2ms) but poor accuracy
(WER ~42% English, ~26% Chinese) because the model has almost no context per
inference — just 600ms of audio.

## Goal

Implement an expanding sliding window so the model always sees up to N seconds
of accumulated audio. This should bring streaming WER close to batch quality
(~15-25%) while keeping per-update latency in the 200-500ms range.

## Architecture

### Expanding Window

Instead of transcribing each 450ms chunk in isolation, accumulate all received
audio into a growing window (up to `WS_WINDOW_MAX_S`, default 6 seconds).
Each time the buffer triggers (~450ms of new audio), re-transcribe the
**entire window**.

```
Trigger 1:  [===0.5s===]           → "However"
Trigger 2:  [=====1.0s=====]       → "However due to"
Trigger 3:  [========1.5s========] → "However due to the slow"
...
Trigger 12: [===============6.0s (cap)===============] → full sentence
Trigger 13: [shift→][===============6.0s===============] → drop oldest 0.5s
```

### Window Shift

When accumulated audio exceeds `WS_WINDOW_MAX_S`:
- Drop `WS_BUFFER_SIZE` bytes (~450ms) from the front
- Keep 150ms overlap at the shift boundary for acoustic continuity
- Words from before the window were already emitted as partials

### New Environment Variable

| Variable | Default | Purpose |
|----------|---------|---------|
| `WS_WINDOW_MAX_S` | `6.0` | Max seconds of audio the sliding window retains |

### Server State Changes

Replace the current per-chunk state:

**Remove:**
- `overlap_buffer` — no longer needed; the window provides context

**Add:**
- `audio_window: bytearray` — all received audio up to cap
- `prev_transcript: str` — last emitted transcript (for change detection)

### Trigger Logic

Same trigger point as now: when incoming data pushes the buffer past
`WS_BUFFER_SIZE`. Instead of transcribing just that chunk:

1. Append new audio to `audio_window`
2. If `len(audio_window)` exceeds byte cap, trim from front
3. Transcribe entire `audio_window`
4. Emit full cumulative transcript as partial

### Flush Behavior

- Transcribe whatever is in `audio_window` with silence padding
- Return as `is_final: true` — the accurate final transcript
- Clear `audio_window`

## Client Protocol

**Best practice (Google/Deepgram/AWS standard):** Every partial is the full
running transcript. Client always replaces, never appends.

```json
← {"text": "However",              "is_final": false}
← {"text": "However due to",       "is_final": false}
← {"text": "However due to the slow communication", "is_final": false}
← {"text": "...full sentence...",   "is_final": true}
```

Consumer code:
```javascript
ws.onmessage = (msg) => {
  const data = JSON.parse(msg.data);
  display.textContent = data.text;  // always replace
};
```

No backward compatibility concerns — no consumers in production yet.

### Connection Handshake

Add `window_max_s` to the connection info so clients know the window size:
```json
{"status": "connected", "sample_rate": 16000, "format": "pcm_s16le",
 "buffer_size": 14400, "window_max_s": 6.0}
```

Remove `overlap_size` from handshake (no longer relevant to clients).

## Testing

- **`test_realtime_accuracy.py`**: Simplify — remove dedup logic. Last partial
  or flush text is the hypothesis. Tighten WER thresholds (25-30% English,
  30-35% Chinese).
- **`test_websocket.py`**: Update expectations — partials are cumulative text,
  not per-chunk fragments.
- **Gateway proxy**: No changes needed (transparent pass-through).

## Constraints

- GPU cost per inference grows with window size. At 6s cap this is ~0.5-1s
  inference time on RTX 4060 — acceptable for balanced latency/accuracy.
- VAD gating still applies — if the window is all silence, skip inference.
