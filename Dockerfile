FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git libsndfile1 ffmpeg

# Install python dependencies
# Remove explicit transformers install to allow qwen-asr to resolve its dependency
RUN pip install --no-cache-dir \
    accelerate \
    soundfile \
    librosa \
    fastapi \
    uvicorn \
    python-multipart \
    "git+https://github.com/QwenLM/Qwen3-ASR.git"

COPY server.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
