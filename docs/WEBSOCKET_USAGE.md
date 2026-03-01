# WebSocket Endpoint Usage

## Endpoint
`ws://<host>:8100/ws/transcribe`

### Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_server_vad` | `true` | Enable server-side VAD (auto-flush on speech→silence, skip inference for silence) |

Example: `ws://localhost:8100/ws/transcribe?use_server_vad=false`

## Protocol

### Connection
Client connects and receives confirmation:
```json
{
  "status": "connected",
  "sample_rate": 16000,
  "format": "pcm_s16le",
  "buffer_size": 14400,
  "window_max_s": 6.0,
  "use_server_vad": true
}
```

### Audio Streaming
Send binary audio frames (PCM 16-bit little-endian, 16kHz mono):
- Server accumulates audio in a sliding window (up to `WS_WINDOW_MAX_S` seconds, default 6s)
- Every ~450ms (configurable via `WS_BUFFER_SIZE`), the server re-transcribes the entire accumulated window
- **Cumulative partials**: Each partial result contains the full running transcript — client should replace its display text on each message, never append

### Response Format
```json
{
  "text": "transcribed text",
  "is_partial": true,
  "is_final": false
}
```

- `is_partial: true` — interim result from a streaming chunk
- `is_final: true` — result from a flush, VAD auto-flush, or disconnect (includes silence padding for last-word accuracy)

### Server-Side VAD

When `use_server_vad` is enabled (default), the server uses Silero VAD to:

1. **Auto-flush**: When speech→silence transition is detected, the server automatically flushes the window with silence padding and sends an `is_final: true` result, then clears the window for the next utterance
2. **Skip silence**: If a chunk contains no speech, inference is skipped entirely (saves GPU)

When disabled, the server transcribes every buffer trigger regardless of speech content, and never auto-flushes.

### Control Commands
Send JSON text messages:

**Flush** — Transcribe remaining buffered audio with silence padding:
```json
{"action": "flush"}
```
The server appends 600ms of silence before transcribing, which helps the model commit trailing words.

**Reset** — Clear audio buffer and sliding window state:
```json
{"action": "reset"}
```

**Config** — Update session settings mid-stream:
```json
{"action": "config", "language": "en", "use_server_vad": false}
```
Response:
```json
{"status": "configured", "language": "en", "use_server_vad": false}
```

Both `language` and `use_server_vad` are optional — include only what you want to change.

### Disconnect Behavior
When the WebSocket disconnects, the server transcribes any remaining buffered audio (with silence padding) and logs the result. No audio is lost on disconnect.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WS_BUFFER_SIZE` | `14400` | Buffer size in bytes before triggering transcription (~450ms at 16kHz) |
| `WS_WINDOW_MAX_S` | `6.0` | Max seconds of audio in the sliding window |
| `WS_FLUSH_SILENCE_MS` | `600` | Silence padding in ms appended on flush/disconnect |
| `ASR_USE_SERVER_VAD` | `true` | Server-wide default for VAD (clients can override per-connection) |
| `REQUEST_TIMEOUT` | `300` | Max seconds per inference request |
| `IDLE_TIMEOUT` | `120` | Seconds before unloading model from GPU |

## Example (Python)
```python
import asyncio
import json
import websockets

async def transcribe_stream(audio_bytes: bytes):
    uri = "ws://localhost:8100/ws/transcribe"
    # uri = "ws://localhost:8100/ws/transcribe?use_server_vad=false"  # disable VAD

    async with websockets.connect(uri) as ws:
        # Receive connection confirmation
        init = json.loads(await ws.recv())
        print(f"Connected: window_max_s={init['window_max_s']}, vad={init['use_server_vad']}")

        # Stream audio in chunks (raw PCM, 16-bit LE, 16kHz mono)
        chunk_size = init["buffer_size"]
        for i in range(0, len(audio_bytes), chunk_size):
            await ws.send(audio_bytes[i:i + chunk_size])

            # Check for partial results (non-blocking)
            try:
                response = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.1))
                prefix = "FINAL" if response.get("is_final") else "partial"
                print(f"[{prefix}] {response['text']}")
            except asyncio.TimeoutError:
                pass

        # Flush remaining audio
        await ws.send(json.dumps({"action": "flush"}))
        final = json.loads(await ws.recv())
        print(f"[FINAL] {final['text']}")

asyncio.run(transcribe_stream(open("audio.pcm", "rb").read()))
```

## Notes
- Model loads on-demand (first request triggers load)
- GPU inference serialized via priority queue (one request at a time)
- WebSocket connections keep model loaded (reset idle timer on each request)
- Sliding window re-transcribes accumulated audio for full context (up to 6s default)
- Silence padding on flush prevents the last word from being cut off
- VAD can be toggled per-connection at connect time or mid-session
