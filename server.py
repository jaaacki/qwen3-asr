from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import torch
import soundfile as sf
import io
import os
import json
import numpy as np
from qwen_asr import Qwen3ASRModel, parse_asr_output
from qwen_asr.inference.qwen3_asr import AutoProcessor

app = FastAPI(title="Qwen3-ASR API")

model = None
processor = None
loaded_model_id = None

@app.on_event("startup")
async def load_model():
    global model, processor, loaded_model_id

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-0.6B")
    loaded_model_id = model_id

    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3ASRModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    print(f"Model loaded!")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": loaded_model_id,
        "cuda": torch.cuda.is_available()
    }

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False)
):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Pass audio as tuple (audio_array, sample_rate)
    lang_code = None if language == "auto" else language
    results = model.transcribe(
        (audio, sr),
        language=lang_code,
        return_time_stamps=return_timestamps
    )

    # Extract text from first result
    if results and len(results) > 0:
        text = results[0].text
        language_code = results[0].language
        # Include timestamps if requested
        if return_timestamps and hasattr(results[0], 'timestamps') and results[0].timestamps:
            return {"text": text, "language": language_code, "timestamps": results[0].timestamps}
    else:
        text = ""
        language_code = language

    return {"text": text, "language": language_code}


async def sse_transcribe_generator(audio, sr, lang_code, return_timestamps):
    """
    Generator for Server-Sent Events streaming transcription.
    Attempts to use streaming if available, otherwise returns full result as single event.
    """
    try:
        # Check if model supports streaming transcription
        if hasattr(model, 'transcribe_stream') or hasattr(model, 'stream_transcribe'):
            # Try streaming method if available
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
                # Send done event
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            except (TypeError, AttributeError, NotImplementedError):
                # Streaming not properly implemented, fall back to regular transcription
                pass

        # Fallback: use regular transcription and return as single SSE event
        results = model.transcribe(
            (audio, sr),
            language=lang_code,
            return_time_stamps=return_timestamps
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


@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    return_timestamps: bool = Form(False)
):
    """
    Streaming transcription endpoint using Server-Sent Events (SSE).
    Returns partial transcription results as they become available.
    If streaming is not supported by the model, returns the full result as a single event.
    """
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

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
