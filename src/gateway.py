"""
Gateway process: handles HTTP/WS routing and worker lifecycle.
Routes inference requests to the worker via HTTP on an internal port.

Usage: GATEWAY_MODE=true in compose.yaml environment.
The gateway starts on port 8000 (external) and spawns a worker on port 8001 (internal).
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

WORKER_HOST = os.getenv("WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8001"))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

_worker_proc: subprocess.Popen | None = None
_last_used = 0.0
_worker_lock = asyncio.Lock()


async def _ensure_worker():
    """Start worker process if not running."""
    global _worker_proc, _last_used
    async with _worker_lock:
        if _worker_proc is None or _worker_proc.poll() is not None:
            print("Starting worker process...")
            _worker_proc = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "worker:app",
                "--host", WORKER_HOST, "--port", str(WORKER_PORT),
                "--ws", "websockets",
            ])
            # Wait for worker to be ready
            for _ in range(30):
                await asyncio.sleep(1)
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://{WORKER_HOST}:{WORKER_PORT}/health"
                        ) as resp:
                            if resp.status == 200:
                                print("Worker process ready")
                                break
                except Exception:
                    continue
        _last_used = time.time()


async def _kill_worker():
    """Kill worker process to free RAM/VRAM."""
    global _worker_proc
    async with _worker_lock:
        if _worker_proc is not None and _worker_proc.poll() is None:
            print("Killing worker process (idle timeout)...")
            _worker_proc.terminate()
            try:
                _worker_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _worker_proc.kill()
            _worker_proc = None
            print("Worker process killed -- RAM reclaimed")


async def _idle_watchdog():
    """Kill worker after idle timeout to reclaim RAM."""
    while True:
        await asyncio.sleep(30)
        if IDLE_TIMEOUT <= 0:
            continue
        if _worker_proc is not None and _worker_proc.poll() is None:
            if time.time() - _last_used > IDLE_TIMEOUT:
                await _kill_worker()


@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(_idle_watchdog())
    yield
    await _kill_worker()

app = FastAPI(title="Qwen3-ASR Gateway", lifespan=lifespan)


async def _proxy_transcribe(audio_bytes: bytes, language: str, return_timestamps: bool) -> dict:
    """Forward transcription request to worker via HTTP."""
    global _last_used
    await _ensure_worker()
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/transcribe"
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("return_timestamps", str(return_timestamps).lower())
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            _last_used = time.time()
            return await resp.json()


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
):
    audio_bytes = await file.read()
    return await _proxy_transcribe(audio_bytes, language, return_timestamps)


@app.post("/v1/audio/subtitles")
async def generate_subtitles(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    mode: str = Form("accurate"),
    max_line_chars: int = Form(42),
):
    """Proxy subtitle generation to worker."""
    from fastapi.responses import Response

    global _last_used
    await _ensure_worker()
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/subtitles"
    form = aiohttp.FormData()
    form.add_field("file", await file.read(), filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("mode", mode)
    form.add_field("max_line_chars", str(max_line_chars))
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            _last_used = time.time()
            srt_content = await resp.text()
            return Response(
                content=srt_content,
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="subtitles.srt"'},
            )


@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
):
    """Proxy SSE streaming transcription to worker."""
    global _last_used
    await _ensure_worker()
    audio_bytes = await file.read()
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/transcribe/stream"
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("return_timestamps", str(return_timestamps).lower())

    async def stream_from_worker():
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                async for line in resp.content:
                    _last_used = time.time()
                    yield line

    return StreamingResponse(
        stream_from_worker(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/transcribe")
async def websocket_proxy(websocket: WebSocket):
    """Proxy WebSocket transcription to worker."""
    global _last_used
    await websocket.accept()

    try:
        await _ensure_worker()
    except Exception as e:
        await websocket.send_json({"error": f"Worker startup failed: {e}"})
        await websocket.close()
        return

    ws_url = f"ws://{WORKER_HOST}:{WORKER_PORT}/ws/transcribe"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as worker_ws:
                # Read the worker connection confirmation and forward it
                init_msg = await worker_ws.receive_json()
                await websocket.send_json(init_msg)

                async def client_to_worker():
                    """Forward client messages to worker."""
                    try:
                        while True:
                            data = await websocket.receive()
                            _last_used = time.time()
                            if "text" in data:
                                await worker_ws.send_str(data["text"])
                            elif "bytes" in data:
                                await worker_ws.send_bytes(data["bytes"])
                    except WebSocketDisconnect:
                        await worker_ws.close()
                    except Exception:
                        pass

                async def worker_to_client():
                    """Forward worker messages to client."""
                    try:
                        async for msg in worker_ws:
                            _last_used = time.time()
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await websocket.send_text(msg.data)
                            elif msg.type == aiohttp.WSMsgType.BINARY:
                                await websocket.send_bytes(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                    except Exception:
                        pass

                await asyncio.gather(client_to_worker(), worker_to_client(), return_exceptions=True)

    except Exception as e:
        try:
            await websocket.send_json({"error": f"Worker connection failed: {e}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
async def health():
    worker_alive = _worker_proc is not None and _worker_proc.poll() is None
    return {"status": "ok", "mode": "gateway", "worker_alive": worker_alive}
