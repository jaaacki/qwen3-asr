# WebSocket Endpoint Usage

## Endpoint
`ws://<host>:8100/ws/transcribe`

## Protocol

### Connection
Client connects and receives confirmation:
```json
{
  "status": "connected",
  "sample_rate": 16000,
  "format": "pcm_s16le",
  "buffer_size": 14400,
  "window_max_s": 6.0
}
```

### Audio Streaming
Send binary audio frames (PCM 16-bit little-endian, 16kHz mono):
- Server accumulates audio in a sliding window (up to `WS_WINDOW_MAX_S` seconds, default 6s)
- Every ~450ms (configurable via `WS_BUFFER_SIZE`), the server re-transcribes the entire accumulated window
- **Cumulative partials**: Each partial result contains the full running transcript — client should replace its display text on each message, never append
- Returns partial results as audio streams in

### Response Format
```json
{
  "text": "transcribed text",
  "is_partial": true,
  "is_final": false
}
```

- `is_partial: true` — interim result from a streaming chunk
- `is_final: true` — result from a flush or disconnect (includes silence padding for last-word accuracy)

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

### Disconnect Behavior
When the WebSocket disconnects, the server transcribes any remaining buffered audio (with silence padding) and logs the result. No audio is lost on disconnect.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WS_BUFFER_SIZE` | `14400` | Buffer size in bytes before triggering transcription (~450ms at 16kHz) |
| `WS_WINDOW_MAX_S` | `6.0` | Max seconds of audio in the sliding window |
| `WS_FLUSH_SILENCE_MS` | `600` | Silence padding in ms appended on flush/disconnect |
| `REQUEST_TIMEOUT` | `300` | Max seconds per inference request |
| `IDLE_TIMEOUT` | `120` | Seconds before unloading model from GPU |

## Example (Python)
```python
import asyncio
import websockets
import struct

async def transcribe_stream():
    uri = "ws://localhost:8100/ws/transcribe"

    async with websockets.connect(uri) as ws:
        # Receive connection confirmation
        msg = await ws.recv()
        print(msg)

        # Stream audio (16-bit PCM samples)
        for sample in audio_samples:
            await ws.send(struct.pack('<h', sample))

            # Check for responses
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                print(response)
            except asyncio.TimeoutError:
                pass

        # Flush remaining audio
        await ws.send('{"action": "flush"}')
        final = await ws.recv()
        print(final)

asyncio.run(transcribe_stream())
```

## Notes
- Model loads on-demand (first request)
- GPU inference serialized (one request at a time)
- Audio preprocessing (mono conversion, normalization) applied automatically
- WebSocket connections keep model loaded (reset idle timer)
- Sliding window re-transcribes accumulated audio for full context (up to 6s by default)
- Silence padding on flush prevents the last word from being cut off
