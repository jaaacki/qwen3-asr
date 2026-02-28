"""
Gateway process: handles HTTP/WS routing and worker lifecycle.
Routes inference requests to the worker via HTTP on an internal port.

Usage: GATEWAY_MODE=true in compose.yaml environment.
The gateway starts on port 8000 (external) and spawns a worker on port 8001 (internal).
"""
from logger import log
from errors import error_response

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
_last_used = time.time()
_worker_lock = asyncio.Lock()


def _check_vram_available(min_free_mb: int = 3500) -> tuple[bool, int]:
    """Return (ok, free_mb) — check GPU has enough free VRAM to load the model."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        free_mb = int(result.stdout.strip())
        return free_mb >= min_free_mb, free_mb
    except Exception:
        return True, -1  # can't check — optimistically proceed


async def _ensure_worker():
    """Start worker process if not running."""
    global _worker_proc, _last_used
    async with _worker_lock:
        if _worker_proc is None or _worker_proc.poll() is not None:
            ok, free_mb = _check_vram_available()
            if not ok:
                log.error("Not enough VRAM to start worker: {}MB free, need ~3500MB", free_mb)
                raise RuntimeError(f"Insufficient VRAM: {free_mb}MB free")
            log.info("Starting worker process... (VRAM free: {}MB)", free_mb)
            _worker_proc = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "worker:app",
                "--host", WORKER_HOST, "--port", str(WORKER_PORT),
                "--ws", "websockets",
            ])
            # Wait for worker to be ready
            for attempt in range(30):
                await asyncio.sleep(1)
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://{WORKER_HOST}:{WORKER_PORT}/health"
                        ) as resp:
                            if resp.status == 200:
                                log.info("Worker process ready")
                                break
                except Exception:
                    continue
            else:
                log.error("Worker process failed to become ready after 30s")
        _last_used = time.time()


async def _kill_worker():
    """Kill worker process to free RAM/VRAM."""
    global _worker_proc
    async with _worker_lock:
        if _worker_proc is not None and _worker_proc.poll() is None:
            log.info("Killing worker process (idle timeout)...")
            _worker_proc.terminate()
            try:
                _worker_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _worker_proc.kill()
            _worker_proc = None
            log.info("Worker process killed -- RAM reclaimed")


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
    from config import validate_env
    validate_env()
    asyncio.create_task(_idle_watchdog())
    if IDLE_TIMEOUT == 0:
        log.info("Always-on mode: pre-spawning worker at startup")
        try:
            await _ensure_worker()
        except Exception as e:
            log.warning("Worker pre-spawn failed (will retry on first request): {}", e)
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


from schemas import HealthResponse, TranscriptionResponse, TranslationResponse

@app.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe Audio",
    description="Upload an audio file to transcribe its speech into text."
)
async def transcribe(
    file: UploadFile = File(..., description="The audio file to transcribe"),
    language: str = Form("auto", description="The language code (e.g. 'en', 'zh'). 'auto' detects automatically."),
    return_timestamps: bool = Form(False, description="Whether to include word-level timestamps in the response text"),
):
    audio_bytes = await file.read()
    log.info("Gateway POST /v1/audio/transcriptions | size={} language={}", len(audio_bytes), language)
    t0 = time.time()
    result = await _proxy_transcribe(audio_bytes, language, return_timestamps)
    log.info("Gateway POST /v1/audio/transcriptions | proxied in {:.2f}s", time.time() - t0)
    return result


@app.post(
    "/v1/audio/translations",
    # Cannot strictly bound response_model to TranslationResponse if it returns SRT plain text, so omitted for raw text
    summary="Translate Audio",
    description="Transcribe and translate an audio file into English ('en') or Chinese ('zh'). Supports returning 'json' or 'srt'."
)
async def translate(
    file: UploadFile = File(..., description="The audio file to translate"),
    language: str = Form("en", description="The target language code for translation ('en' or 'zh')."),
    response_format: str = Form("json", description="The format of the response ('json' or 'srt').")
):
    """Proxy translation request to worker."""
    from fastapi.responses import Response
    global _last_used
    await _ensure_worker()

    content = await file.read()
    log.info("Gateway POST /v1/audio/translations | size={} target={} format={}", len(content), language, response_format)
    t0 = time.time()

    # Restrict to en/zh only
    target_lang = "en" if language.lower() not in ["en", "zh"] else language.lower()

    url = f"http://{WORKER_HOST}:{WORKER_PORT}/translate"
    form = aiohttp.FormData()
    form.add_field("file", content, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", target_lang)
    form.add_field("response_format", response_format)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            _last_used = time.time()
            if resp.status != 200:
                body = await resp.text()
                log.error("Gateway proxy error | url={} status={}", url, resp.status)
                try:
                    worker_error = json.loads(body)
                    if "code" in worker_error:
                        return JSONResponse(status_code=resp.status, content=worker_error)
                except (json.JSONDecodeError, KeyError):
                    pass
                return error_response("WORKER_ERROR", body, resp.status)

            log.info("Gateway POST /v1/audio/translations | proxied in {:.2f}s", time.time() - t0)
            if response_format.lower() == "srt":
                srt_content = await resp.text()
                return Response(
                    content=srt_content,
                    media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="translated_subtitles.srt"'},
                )
            else:
                return await resp.json()


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
    audio_bytes = await file.read()
    log.info("Gateway POST /v1/audio/subtitles | size={} language={} mode={}", len(audio_bytes), language, mode)
    t0 = time.time()
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/subtitles"
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("mode", mode)
    form.add_field("max_line_chars", str(max_line_chars))
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            _last_used = time.time()
            if resp.status != 200:
                body = await resp.text()
                log.error("Gateway proxy error | url={} status={}", url, resp.status)
                try:
                    worker_error = json.loads(body)
                    if "code" in worker_error:
                        return JSONResponse(status_code=resp.status, content=worker_error)
                except (json.JSONDecodeError, KeyError):
                    pass
                return error_response("WORKER_ERROR", body, resp.status)
            srt_content = await resp.text()
            log.info("Gateway POST /v1/audio/subtitles | proxied in {:.2f}s", time.time() - t0)
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
    log.info("Gateway POST /v1/audio/transcriptions/stream | size={} language={}", len(audio_bytes), language)
    url = f"http://{WORKER_HOST}:{WORKER_PORT}/transcribe/stream"
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
    form.add_field("language", language)
    form.add_field("return_timestamps", str(return_timestamps).lower())

    t0 = time.time()

    async def stream_from_worker():
        chunk_count = 0
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
                    async for line in resp.content:
                        _last_used = time.time()
                        chunk_count += 1
                        yield line
            log.info("Gateway POST /v1/audio/transcriptions/stream | done chunks={} elapsed={:.2f}s", chunk_count, time.time() - t0)
        except Exception as e:
            log.error("Gateway POST /v1/audio/transcriptions/stream | error after {:.2f}s: {}", time.time() - t0, e)

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
    log.info("[GW-WS] Client connected, proxying to worker")

    try:
        await _ensure_worker()
    except Exception as e:
        await websocket.send_json({"code": "WORKER_STARTUP_FAILED", "message": f"Worker startup failed: {e}", "statusCode": 503})
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
                            if data.get("type") == "websocket.disconnect":
                                await worker_ws.close()
                                break
                            _last_used = time.time()
                            if "text" in data:
                                await worker_ws.send_str(data["text"])
                            elif "bytes" in data:
                                await worker_ws.send_bytes(data["bytes"])
                    except WebSocketDisconnect:
                        await worker_ws.close()
                    except Exception as e:
                        log.warning("[GW-WS] client_to_worker error: {}", e)

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
                    except Exception as e:
                        log.warning("[GW-WS] worker_to_client error: {}", e)

                await asyncio.gather(client_to_worker(), worker_to_client(), return_exceptions=True)

    except Exception as e:
        try:
            await websocket.send_json({"code": "WORKER_CONNECTION_FAILED", "message": f"Worker connection failed: {e}", "statusCode": 502})
        except Exception:
            pass
    finally:
        log.info("[GW-WS] Proxy session ended")
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
async def health():
    worker_alive = _worker_proc is not None and _worker_proc.poll() is None
    log.debug("Gateway GET /health | worker_alive={}", worker_alive)
    info = {"status": "ok", "mode": "gateway", "worker_alive": worker_alive, "model_loaded": False, "model_id": None}
    if worker_alive:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{WORKER_HOST}:{WORKER_PORT}/health",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        info.update(await resp.json())
        except Exception:
            pass
    return info
