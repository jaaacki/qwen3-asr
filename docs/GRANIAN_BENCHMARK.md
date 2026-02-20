# Granian vs Uvicorn Benchmark

## Enable Granian

Set `USE_GRANIAN=true` in `compose.yaml` environment:

```yaml
environment:
  - USE_GRANIAN=true
```

Or run directly:

```bash
USE_GRANIAN=true docker compose up -d
```

## Benchmark methodology

### HTTP throughput

Use `hey` for HTTP POST throughput testing:

```bash
hey -n 100 -c 5 -m POST \
  -F "file=@audio.wav" \
  http://localhost:8100/v1/audio/transcriptions
```

### WebSocket latency

Use the client in `docs/WEBSOCKET_USAGE.md`, measure round-trip time per chunk:

```python
import time
t0 = time.monotonic()
ws.send(audio_chunk)
result = ws.recv()
rtt = time.monotonic() - t0
```

### Health endpoint (baseline overhead)

```bash
hey -n 10000 -c 50 http://localhost:8100/health
```

## Expected gains

- HTTP throughput: +15-25% (Rust event loop vs Python asyncio)
- WebSocket frame overhead: -20% (Granian's native WS implementation)
- Latency p99: -10ms typical
- Health endpoint: +30-50% RPS (pure I/O bound, no GPU)

## Caveats

- Granian `workers=1` is required (single GPU, shared model state)
- Lifespan events use the ASGI lifespan protocol (`asynccontextmanager` on FastAPI app)
- WebSocket behavior should be tested manually -- Granian's WS implementation differs from uvicorn+websockets
- Granian does not support `--ws` flag; WebSocket is handled natively
