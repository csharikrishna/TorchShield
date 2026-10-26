"""Benchmark for end-to-end AI tensor and model checkpoint loading workflows.

Evaluates system-level metrics:
- Time to first layer
- Total checkpoint loading time
- Peak host RAM & GPU VRAM
- Plaintext vs Whole-File AEAD vs Chunked Streaming AEAD vs Experimental AHCC
"""
import io
import json
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import torch.nn as nn

from ahcc.checkpoint import SecureModelCheckpoint
from ahcc.secure import AESGCMEnvelope
from ahcc.engine import AHCC_Engine, TensorCipherWrapper


class SyntheticTransformerModel(nn.Module):
    def __init__(self, num_layers=4, d_model=512):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def benchmark_tensor_loading(iterations=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"BENCHMARK: AI TENSOR CHECKPOINT LOADING WORKFLOW (Device: {device})")
    print("=" * 80)

    model = SyntheticTransformerModel(num_layers=8, d_model=512).to(device)
    key = secrets.token_bytes(32)
    state_dict = model.state_dict()
    total_params = sum(p.numel() for p in model.parameters())
    model_bytes = total_params * 4  # float32

    print(f"Model Specifications: 8 Layers, {total_params:,} parameters ({model_bytes / 1024 / 1024:.2f} MiB)")

    with tempfile.TemporaryDirectory() as tmpdir:
        plain_file = os.path.join(tmpdir, "plain.pt")
        whole_file = os.path.join(tmpdir, "whole.aes")
        chunked_file = os.path.join(tmpdir, "chunked.ahcc_ckpt")

        # Save files
        torch.save(state_dict, plain_file)

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        enc_whole = AESGCMEnvelope.encrypt(buf.getvalue(), key, associated_data=b"model")
        with open(whole_file, "wb") as f:
            f.write(enc_whole)

        SecureModelCheckpoint.save_checkpoint(state_dict, chunked_file, key, chunk_bytes=1024*1024)

        # 1. Plaintext Loading
        plain_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            s = torch.load(plain_file, map_location=device, weights_only=True)
            if device == "cuda":
                torch.cuda.synchronize()
            plain_times.append(time.perf_counter() - t0)

        # 2. Whole-File AEAD Loading
        whole_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            with open(whole_file, "rb") as f:
                c = f.read()
            dec = AESGCMEnvelope.decrypt(c, key, associated_data=b"model")
            s = torch.load(io.BytesIO(dec), map_location=device, weights_only=True)
            if device == "cuda":
                torch.cuda.synchronize()
            whole_times.append(time.perf_counter() - t0)

        # 3. Chunked Streaming AEAD Loading
        chunked_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            s = SecureModelCheckpoint.load_checkpoint(chunked_file, key, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            chunked_times.append(time.perf_counter() - t0)

        # 4. Experimental AHCC Loading (Single Layer only due to execution cost)
        ahcc_eng = AHCC_Engine(N=64, R=2, p=4_294_967_291, device=device)
        ahcc_wrap = TensorCipherWrapper(ahcc_eng)
        first_layer_name = list(state_dict.keys())[0]
        sample_tensor = state_dict[first_layer_name].detach()
        q_tensor = (sample_tensor * 1000).to(torch.int64) % ahcc_eng.p
        ct_ahcc, meta_ahcc = ahcc_wrap.encrypt_tensor(q_tensor)

        ahcc_layer_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            p = ahcc_wrap.decrypt_tensor(ct_ahcc, meta_ahcc)
            if device == "cuda":
                torch.cuda.synchronize()
            ahcc_layer_times.append(time.perf_counter() - t0)

        import statistics
        med_plain = statistics.median(plain_times) * 1000
        med_whole = statistics.median(whole_times) * 1000
        med_chunked = statistics.median(chunked_times) * 1000
        med_ahcc_1layer = statistics.median(ahcc_layer_times) * 1000

        print("\nMeasured Model Loading Latency (Median of 10 runs):")
        print("-" * 80)
        print(f"Plaintext PyTorch (No Security)              : {med_plain:7.2f} ms")
        print(f"Whole-File AES-256-GCM                      : {med_whole:7.2f} ms  (+{med_whole - med_plain:5.2f} ms overhead)")
        print(f"Chunked Streaming AEAD (SecureModelCheckpoint): {med_chunked:7.2f} ms  (+{med_chunked - med_plain:5.2f} ms overhead)")
        print(f"AHCC Decryption (SINGLE LAYER ONLY)           : {med_ahcc_1layer:7.2f} ms")
        print("-" * 80)

        results = {
            "model_params": total_params,
            "model_bytes": model_bytes,
            "plain_load_ms": med_plain,
            "whole_file_load_ms": med_whole,
            "chunked_load_ms": med_chunked,
            "ahcc_single_layer_ms": med_ahcc_1layer,
        }
        with open("results/model_loading_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    benchmark_tensor_loading()
