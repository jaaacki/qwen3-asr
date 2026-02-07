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
  "buffer_size": 48000
}
```

### Audio Streaming
Send binary audio frames (PCM 16-bit little-endian, 16kHz mono):
- Server buffers incoming chunks
- Automatically transcribes when buffer reaches ~1.5 seconds (configurable via `WS_BUFFER_SIZE` env var)
- Returns partial results as audio streams in

### Response Format
```json
{
  "text": "transcribed text",
  "is_partial": true,   // false when buffer fully processed
  "is_final": true      // true when all buffered audio consumed
}
```

### Control Commands
Send JSON text messages:

**Flush** - Force transcription of remaining buffered audio:
```json
{"action": "flush"}
```

**Reset** - Clear audio buffer:
```json
{"action": "reset"}
```

## Configuration
- `WS_BUFFER_SIZE` - Buffer size in bytes (default: 48000 = ~1.5 seconds at 16kHz)
- Same `REQUEST_TIMEOUT` and `IDLE_TIMEOUT` settings apply as HTTP endpoints

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
