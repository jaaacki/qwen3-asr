from logger import log
#!/usr/bin/env python3
"""
Build TensorRT engine for Qwen3-ASR encoder.
Usage: python src/build_trt.py --output models/encoder.trt
Requires: torch-tensorrt
"""
import argparse
import os
import torch
import numpy as np


def build_trt_engine(model_id: str, output_path: str):
    try:
        import torch_tensorrt
    except ImportError:
        log.info("torch-tensorrt not installed. Run: pip install torch-tensorrt")
        return

    from qwen_asr import Qwen3ASRModel
    log.info(f"Loading {model_id}...")
    model = Qwen3ASRModel.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    encoder = getattr(model, 'encoder', None) or getattr(
        getattr(model, 'model', None), 'encoder', None
    )
    if encoder is None:
        log.info("Cannot find encoder submodule")
        return

    dummy = torch.randn(1, 80, 3000, dtype=torch.float16, device="cuda")

    trt_model = torch_tensorrt.compile(
        encoder,
        inputs=[torch_tensorrt.Input(
            min_shape=(1, 80, 500),
            opt_shape=(1, 80, 1500),
            max_shape=(1, 80, 3000),
            dtype=torch.float16,
        )],
        enabled_precisions={torch.float16},
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.jit.save(torch.jit.trace(trt_model, dummy), output_path)
    log.info(f"TensorRT engine saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B"))
    parser.add_argument("--output", default="models/encoder.trt")
    args = parser.parse_args()
    build_trt_engine(args.model_id, args.output)
