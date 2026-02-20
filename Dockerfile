FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# CUDA memory allocator tuning — reduce fragmentation for shared GPU
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Disable tokenizer parallelism warnings in forked workers
ENV TOKENIZERS_PARALLELISM=false
# Prevent OpenMP/MKL from spawning CPU threads that compete with GPU inference
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    accelerate \
    soundfile \
    torchaudio==2.5.1 \
    fastapi \
    uvicorn \
    python-multipart \
    websockets \
    silero-vad \
    bitsandbytes \
    onnxruntime-gpu \
    "git+https://github.com/QwenLM/Qwen3-ASR.git"

# torchao for FP8 quantization (opt-in via QUANTIZE=fp8)
RUN pip install --no-cache-dir torchao

# Flash Attention 2 (built from source for CUDA 12.4 compatibility)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

COPY src/server.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]
