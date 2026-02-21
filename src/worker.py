"""
Worker process: inference-only, can be killed/restarted to free RAM.
Exposes a simple internal HTTP API on port 8001.

Not intended to be run directly -- started by gateway.py.
Imports core inference logic from server.py to stay in sync with optimizations.
"""
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

import asyncio
import io
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Qwen3-ASR Worker")


@app.on_event("startup")
async def startup():
    """Load model eagerly on worker startup."""
    await _ensure_model_loaded()


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
):
    await _ensure_model_loaded()
    audio_bytes = await file.read()
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))
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
        release_gpu_memory()
        return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

    if results and len(results) > 0:
        text = detect_and_fix_repetitions(results[0].text)
        language_code = results[0].language
        if return_timestamps and hasattr(results[0], "timestamps") and results[0].timestamps:
            return {"text": text, "language": language_code, "timestamps": results[0].timestamps}
    else:
        text = ""
        language_code = language

    return {"text": text, "language": language_code}


@app.post("/transcribe/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False),
):
    """SSE streaming transcription -- worker-side endpoint."""
    await _ensure_model_loaded()
    audio_bytes = await file.read()
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    lang_code = None if language == "auto" else language

    from server import sse_transcribe_generator

    return StreamingResponse(
        sse_transcribe_generator(audio, sr, lang_code, return_timestamps),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket transcription -- worker-side, reuses server.py logic."""
    from server import websocket_transcribe as _ws_handler
    await _ws_handler(websocket)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "worker", "model_loaded": _srv.model is not None}
