from logger import log
#!/usr/bin/env python3
"""
Export Qwen3-ASR encoder to ONNX Runtime.
Usage: python src/export_onnx.py --output models/encoder.onnx
"""
import argparse
import torch
import numpy as np
import os


def export_encoder(model_id: str, output_path: str):
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.qwen3_asr import AutoProcessor

    log.info(f"Loading {model_id}...")
    model = Qwen3ASRModel.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    # Extract encoder submodule
    encoder = getattr(model, 'encoder', None) or getattr(model.model, 'encoder', None)
    if encoder is None:
        log.info("Could not find encoder submodule. Model architecture may not support ONNX export.")
        return

    # Dummy input: log-mel features [batch, n_mels, time]
    dummy_input = torch.randn(1, 80, 3000)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["log_mel"],
        output_names=["encoder_out"],
        dynamic_axes={"log_mel": {2: "time"}, "encoder_out": {1: "time"}},
    )
    log.info(f"Encoder exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B"))
    parser.add_argument("--output", default="models/encoder.onnx")
    args = parser.parse_args()
    export_encoder(args.model_id, args.output)
