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
    aiohttp \
    psutil \
    granian \
    # vllm \  # Uncomment for USE_VLLM=true support (large dependency)
    "git+https://github.com/QwenLM/Qwen3-ASR.git"

# torchao for FP8 quantization (opt-in via QUANTIZE=fp8)
RUN pip install --no-cache-dir torchao

# Flash Attention 2 (built from source for CUDA 12.4 compatibility)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Optional: install torch-tensorrt for TRT encoder support
# RUN pip install --no-cache-dir torch-tensorrt

COPY src/server.py /app/server.py
COPY src/gateway.py /app/gateway.py
COPY src/worker.py /app/worker.py
COPY src/build_trt.py /app/build_trt.py

EXPOSE 8000

# GATEWAY_MODE=true: run gateway+worker split; USE_GRANIAN=true: Rust-based ASGI server
CMD ["sh", "-c", "if [ \"$GATEWAY_MODE\" = 'true' ]; then uvicorn gateway:app --host 0.0.0.0 --port 8000; elif [ \"$USE_GRANIAN\" = 'true' ]; then granian --interface asgi --host 0.0.0.0 --port 8000 --workers 1 server:app; else uvicorn server:app --host 0.0.0.0 --port 8000 --ws websockets; fi"]
