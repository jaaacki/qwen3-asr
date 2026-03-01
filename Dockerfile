FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

# CUDA toolkit path — required for flash-attn to locate nvcc in devel image
ENV CUDA_HOME=/usr/local/cuda
# CUDA memory allocator tuning — reduce fragmentation for shared GPU
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Disable tokenizer parallelism warnings in forked workers
ENV TOKENIZERS_PARALLELISM=false
# Prevent OpenMP/MKL from spawning CPU threads that compete with GPU inference
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
# Limit torch.compile inductor workers (default 20 spawns ~800MB of subprocesses)
ENV TORCHINDUCTOR_COMPILE_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (pinned to latest as of 2026-02-26)
RUN pip install --no-cache-dir \
    accelerate==1.12.0 \
    soundfile==0.13.1 \
    torchaudio==2.6.0 \
    fastapi==0.133.1 \
    uvicorn==0.41.0 \
    python-multipart==0.0.22 \
    websockets==16.0 \
    silero-vad==6.2.1 \
    bitsandbytes==0.49.2 \
    onnxruntime-gpu==1.24.2 \
    aiohttp==3.13.3 \
    psutil==7.2.2 \
    granian==2.7.2 \
    loguru==0.7.3 \
    openai==2.24.0 \
    "git+https://github.com/QwenLM/Qwen3-ASR.git"

# torchao for FP8 quantization (opt-in via QUANTIZE=fp8)
RUN pip install --no-cache-dir torchao==0.16.0

# Flash Attention 2 (built from source for CUDA 12.4 compatibility)
RUN pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

# Optional: install torch-tensorrt for TRT encoder support
# RUN pip install --no-cache-dir torch-tensorrt

COPY src/*.py /app/

EXPOSE 8000

# GATEWAY_MODE=true: run gateway+worker split; USE_GRANIAN=true: Rust-based ASGI server
CMD ["sh", "-c", "if [ \"$GATEWAY_MODE\" = 'true' ]; then uvicorn gateway:app --host 0.0.0.0 --port 8000; elif [ \"$USE_GRANIAN\" = 'true' ]; then granian --interface asgi --host 0.0.0.0 --port 8000 --workers 1 server:app; else uvicorn server:app --host 0.0.0.0 --port 8000 --ws websockets; fi"]
