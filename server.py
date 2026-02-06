from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import torch
import soundfile as sf
import io
import os
import gc
import json
import asyncio
import time
import numpy as np
from qwen_asr import Qwen3ASRModel, parse_asr_output
from qwen_asr.inference.qwen3_asr import AutoProcessor

app = FastAPI(title="Qwen3-ASR API")

model = None
processor = None
loaded_model_id = None

# Semaphore to serialize GPU inference — prevents OOM with concurrent requests
_infer_semaphore = asyncio.Semaphore(1)

# Lock to prevent concurrent load/unload
_model_lock = asyncio.Lock()

# Request timeout in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

# Idle unload timeout in seconds (0 = disabled)
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))

# Track last request time
_last_used = 0.0

# Target sample rate expected by the model
TARGET_SR = 16000


def release_gpu_memory():
    """Force release of unused GPU memory back to the system."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def preprocess_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """
    Preprocess audio for optimal inference:
    - Convert to mono
    - Convert to float32
    - Resample to 16kHz if needed
    - Normalize peak amplitude
    """
    # Convert to mono if stereo/multi-channel
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Convert to float32
    audio = audio.astype(np.float32)

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        except ImportError:
            pass  # librosa not available, pass raw sample rate

    # Peak normalize to [-1, 1] — improves feature extraction on quiet audio
    peak = np.abs(audio).max()
    if peak > 0 and peak != 1.0:
        audio = audio / peak

    return audio, sr


def _load_model_sync():
    """Load model into GPU (blocking). Called from async context via lock."""
    global model, processor, loaded_model_id, _last_used

    if model is not None:
        return

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-0.6B")
    loaded_model_id = model_id

    print(f"Loading {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model = Qwen3ASRModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    # Warmup inference to trigger CUDA kernel caching
    if torch.cuda.is_available():
        print("Warming up GPU...")
        dummy = np.zeros(TARGET_SR, dtype=np.float32)  # 1 second of silence
        try:
            model.transcribe((dummy, TARGET_SR))
        except Exception:
            pass
        release_gpu_memory()

    _last_used = time.time()
    print(f"Model loaded! GPU memory after load:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")


def _unload_model_sync():
    """Unload model from GPU to free VRAM."""
    global model, processor

    if model is None:
        return

    print("Unloading model (idle timeout)...")
    del model
    del processor
    model = None
    processor = None
    release_gpu_memory()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"Model unloaded. GPU: Allocated: {allocated:.0f} MB, Reserved: {reserved:.0f} MB")


async def _ensure_model_loaded():
    """Load model if not already loaded. Thread-safe via lock."""
    global _last_used
    if model is not None:
        _last_used = time.time()
        return
    async with _model_lock:
        if model is not None:
            _last_used = time.time()
            return
        await asyncio.get_event_loop().run_in_executor(None, _load_model_sync)
        _last_used = time.time()


async def _idle_watchdog():
    """Background task that unloads model after IDLE_TIMEOUT seconds of inactivity."""
    while True:
        await asyncio.sleep(30)
        if IDLE_TIMEOUT <= 0 or model is None:
            continue
        if time.time() - _last_used > IDLE_TIMEOUT:
            async with _model_lock:
                if model is not None and time.time() - _last_used > IDLE_TIMEOUT:
                    await asyncio.get_event_loop().run_in_executor(None, _unload_model_sync)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_idle_watchdog())


@app.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2),
        }
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": loaded_model_id,
        "cuda": torch.cuda.is_available(),
        **gpu_info,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False)
):
    await _ensure_model_loaded()

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    audio, sr = preprocess_audio(audio, sr)

    lang_code = None if language == "auto" else language

    try:
        async with _infer_semaphore:
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _do_transcribe(audio, sr, lang_code, return_timestamps)
                ),
                timeout=REQUEST_TIMEOUT
            )
    except asyncio.TimeoutError:
        release_gpu_memory()
        return JSONResponse(status_code=504, content={"error": "Transcription timed out"})

    if results and len(results) > 0:
        text = results[0].text
        language_code = results[0].language
        if return_timestamps and hasattr(results[0], 'timestamps') and results[0].timestamps:
            return {"text": text, "language": language_code, "timestamps": results[0].timestamps}
    else:
        text = ""
        language_code = language

    return {"text": text, "language": language_code}


def _do_transcribe(audio, sr, lang_code, return_timestamps):
    """Run inference in a thread pool, with memory cleanup after."""
    try:
        with torch.inference_mode():
            results = model.transcribe(
                (audio, sr),
                language=lang_code,
                return_time_stamps=return_timestamps
            )
        return results
    finally:
        release_gpu_memory()


async def sse_transcribe_generator(audio, sr, lang_code, return_timestamps):
    """Generator for Server-Sent Events streaming transcription."""
    try:
        if hasattr(model, 'transcribe_stream') or hasattr(model, 'stream_transcribe'):
            stream_method = getattr(model, 'transcribe_stream', None) or getattr(model, 'stream_transcribe', None)
            try:
                for partial_result in stream_method(
                    (audio, sr),
                    language=lang_code,
                    return_time_stamps=return_timestamps
                ):
                    if hasattr(partial_result, 'text'):
                        data = {
                            "text": partial_result.text,
                            "language": getattr(partial_result, 'language', lang_code or 'auto'),
                            "is_final": getattr(partial_result, 'is_final', False)
                        }
                        if return_timestamps and hasattr(partial_result, 'timestamps') and partial_result.timestamps:
                            data["timestamps"] = partial_result.timestamps
                        yield f"data: {json.dumps(data)}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            except (TypeError, AttributeError, NotImplementedError):
                pass

        # Fallback: run full transcription via thread pool
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _do_transcribe(audio, sr, lang_code, return_timestamps)
        )

        if results and len(results) > 0:
            text = results[0].text
            language_code = results[0].language
            data = {
                "text": text,
                "language": language_code,
                "is_final": True
            }
            if return_timestamps and hasattr(results[0], 'timestamps') and results[0].timestamps:
                data["timestamps"] = results[0].timestamps
        else:
            data = {
                "text": "",
                "language": lang_code or "auto",
                "is_final": True
            }

        yield f"data: {json.dumps(data)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        release_gpu_memory()


@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False)
):
    """Streaming transcription endpoint using Server-Sent Events (SSE)."""
    await _ensure_model_loaded()

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    audio, sr = preprocess_audio(audio, sr)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
