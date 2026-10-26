"""Phase 4 Comprehensive Checkpoint Pipeline Profiling and Optimization Benchmark.

Measures end-to-end latency and fine-grained component timings:
1. Container header read & key derivation
2. Manifest decryption, JSON parsing, & specification digest verification
3. Chunk I/O, AES-GCM decryption, & tag authentication
4. Direct buffer reassembly (preallocated bytearray)
5. Tensor reconstruction from buffer
6. Pinned host-to-device (CPU -> CUDA) DMA transfer
7. Target model parameter/buffer in-place restoration
8. Time to first inference
Across multiple representative model structures and payload sizes.
"""
from __future__ import annotations

import csv
import json
import math
import os
import secrets
import struct
import sys
import tempfile
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ahcc.checkpoint import SecureModelCheckpoint, ResourceLimits, _canonical_json, _strict_json_loads
from ahcc.secure import AESGCMEnvelope


class ConvNet(nn.Module):
    """Small CNN (~4.5 MiB)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TransformerModel(nn.Module):
    """Transformer (~16.8 MiB, 4.4M parameters)."""
    def __init__(self, num_layers: int = 8, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(1000, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        h = self.encoder(h)
        return self.fc(h[:, 0, :])


def get_model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters()) + \
           sum(b.numel() * b.element_size() for b in model.buffers())


def profile_fine_grained_loading(
    checkpoint_path: str,
    master_key: bytes,
    device: str,
    target_model: nn.Module,
) -> Dict[str, float]:
    """Measure fine-grained step-by-step timings of the authenticated loading pipeline."""
    timings = {}

    t0 = time.perf_counter()
    with open(checkpoint_path, "rb") as f:
        # Step 1: Read Header & Derive Keys
        magic = f.read(8)
        assert magic == SecureModelCheckpoint.MAGIC_V4
        ver = f.read(1)[0]
        salt = f.read(SecureModelCheckpoint.SALT_SIZE_V4)
        art_id_len = f.read(1)[0]
        art_id = f.read(art_id_len)
        key_id_len = f.read(1)[0]
        key_id = f.read(key_id_len)
        m_len = struct.unpack(">I", f.read(4))[0]
        enc_manifest = f.read(m_len)

        manifest_key = SecureModelCheckpoint.derive_manifest_key_v4(master_key, salt, art_id)
        chunk_key = SecureModelCheckpoint.derive_chunk_key_v4(master_key, salt, art_id)
        timings["header_and_kdf_ms"] = (time.perf_counter() - t0) * 1000

        # Step 2: Manifest Decryption & Specification Verification
        t_m0 = time.perf_counter()
        manifest_aad = art_id + b"::MANIFEST::v4"
        manifest_bytes = AESGCMEnvelope.decrypt(enc_manifest, manifest_key, associated_data=manifest_aad)
        manifest = _strict_json_loads(manifest_bytes)

        # Reconstruct and verify specification digest
        reconstructed_spec = {
            "version": manifest["version"],
            "artifact_id": manifest["artifact_id"],
            "key_id": manifest["key_id"],
            "total_tensors": manifest["total_tensors"],
            "total_chunks": manifest["total_chunks"],
            "total_raw_bytes": manifest["total_raw_bytes"],
            "tensors": manifest["tensors"],
        }
        computed_digest = torch.tensor([])  # placeholder
        import hashlib
        import secrets as pysecrets
        comp_digest = hashlib.sha256(_canonical_json(reconstructed_spec)).digest()
        decl_digest = bytes.fromhex(manifest["manifest_digest"])
        assert pysecrets.compare_digest(comp_digest, decl_digest)
        timings["manifest_auth_and_verify_ms"] = (time.perf_counter() - t_m0) * 1000

        # Step 3: Stream chunks and reconstruct tensors
        chunk_iter = iter(manifest["chunks"])
        chunk_read_and_decrypt_time = 0.0
        buffer_reconstruction_time = 0.0
        device_transfer_time = 0.0
        model_restore_time = 0.0

        for tensor_info in manifest["tensors"]:
            name = tensor_info["name"]
            dtype_str = tensor_info["dtype"]
            dtype = getattr(torch, dtype_str)
            shape = tuple(tensor_info["shape"])
            total_raw = tensor_info["raw_len"]
            num_chunks = tensor_info["num_chunks"]

            t_buf0 = time.perf_counter()
            layer_buf = bytearray(total_raw) if total_raw > 0 else None
            buffer_reconstruction_time += (time.perf_counter() - t_buf0)

            for chunk_idx in range(num_chunks):
                chunk_info = next(chunk_iter)
                chunk_seq = chunk_info["chunk_seq"]
                offset = chunk_info["offset"]
                expected_raw_len = chunk_info["raw_len"]
                c_len = chunk_info["ciphertext_len"]

                t_dec0 = time.perf_counter()
                c_encrypted = f.read(c_len)
                aad = (
                    art_id
                    + b"::" + decl_digest
                    + b"::tensor=" + name.encode("utf-8")
                    + b"::seq=" + struct.pack(">Q", chunk_seq)
                    + b"::idx=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::offset=" + struct.pack(">Q", offset)
                    + b"::raw_len=" + struct.pack(">Q", expected_raw_len)
                )
                decrypted_slice = AESGCMEnvelope.decrypt(c_encrypted, chunk_key, associated_data=aad)
                chunk_read_and_decrypt_time += (time.perf_counter() - t_dec0)

                t_copy0 = time.perf_counter()
                if layer_buf is not None:
                    layer_buf[offset : offset + len(decrypted_slice)] = decrypted_slice
                buffer_reconstruction_time += (time.perf_counter() - t_copy0)

            t_t0 = time.perf_counter()
            if total_raw > 0:
                tensor = torch.frombuffer(layer_buf, dtype=dtype).reshape(shape)
            else:
                tensor = torch.empty(shape, dtype=dtype)
            buffer_reconstruction_time += (time.perf_counter() - t_t0)

            t_dev0 = time.perf_counter()
            if device != "cpu" and torch.cuda.is_available() and str(device).startswith("cuda"):
                tensor = tensor.pin_memory().to(device, non_blocking=True)
            else:
                tensor = tensor.to(device)
            device_transfer_time += (time.perf_counter() - t_dev0)

            t_res0 = time.perf_counter()
            try:
                curr_param = target_model.get_parameter(name)
                curr_param.data.copy_(tensor)
            except (AttributeError, KeyError):
                try:
                    curr_buf = target_model.get_buffer(name)
                    curr_buf.copy_(tensor)
                except (AttributeError, KeyError):
                    pass
            model_restore_time += (time.perf_counter() - t_res0)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    timings["chunk_read_and_decrypt_ms"] = round(chunk_read_and_decrypt_time * 1000, 2)
    timings["buffer_and_tensor_reconstruction_ms"] = round(buffer_reconstruction_time * 1000, 2)
    timings["host_to_device_transfer_ms"] = round(device_transfer_time * 1000, 2)
    timings["model_parameter_restore_ms"] = round(model_restore_time * 1000, 2)
    timings["total_end_to_end_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return timings


def run_benchmarks(iterations: int = 15, device: str = "cuda") -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"\n========================================================")
    print(f" AHCC Phase 4 Checkpoint Benchmark Suite (Device: {device})")
    print(f"========================================================")

    models = {
        "ConvNet_4.5MB": (ConvNet(), torch.randn(4, 3, 32, 32)),
        "Transformer_16.8MB": (TransformerModel(num_layers=8, d_model=256), torch.randint(0, 1000, (4, 32))),
    }

    benchmark_records = {}

    for name, (model, dummy_input) in models.items():
        model = model.eval()
        size_bytes = get_model_size_bytes(model)
        size_mib = round(size_bytes / (1024**2), 2)
        print(f"\nEvaluating {name} ({size_mib} MiB, {sum(p.numel() for p in model.parameters()):,} params)...")

        key = secrets.token_bytes(32)
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, f"{name}.ahcc")
            plain_path = os.path.join(td, f"{name}.pt")

            # Warmup save
            torch.save(model.state_dict(), plain_path)
            SecureModelCheckpoint.save_checkpoint(model.state_dict(), ckpt_path, key, chunk_bytes=1024 * 1024)

            # 1. Plaintext baseline
            plain_load_times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                st = torch.load(plain_path, weights_only=True)
                if device != "cpu":
                    for k in st:
                        st[k] = st[k].to(device)
                    torch.cuda.synchronize()
                plain_load_times.append((time.perf_counter() - t0) * 1000)

            # 2. Phase 4 End-to-End Loading + First Inference
            target_inst = type(model)().to(device).eval()
            phase4_load_times = []
            time_to_first_inf = []

            for _ in range(iterations):
                t0 = time.perf_counter()
                SecureModelCheckpoint.load_checkpoint(ckpt_path, key, device=device, target_model=target_inst)
                if device != "cpu":
                    torch.cuda.synchronize()
                t_load = (time.perf_counter() - t0) * 1000
                phase4_load_times.append(t_load)

                # First inference
                t_inf0 = time.perf_counter()
                with torch.no_grad():
                    _ = target_inst(dummy_input.to(device))
                if device != "cpu":
                    torch.cuda.synchronize()
                t_inf = (time.perf_counter() - t_inf0) * 1000
                time_to_first_inf.append(t_load + t_inf)

            # 3. Fine-grained Component Timings (3 runs averaged)
            comp_samples = []
            for _ in range(5):
                comp_samples.append(profile_fine_grained_loading(ckpt_path, key, device, target_inst))

            def avg_key(k):
                return round(sum(s[k] for s in comp_samples) / len(comp_samples), 2)

            comp_avg = {k: avg_key(k) for k in comp_samples[0]}

            def calc_stats(arr):
                s = sorted(arr)
                n = len(s)
                return {
                    "median_ms": round(s[n // 2], 2),
                    "mean_ms": round(sum(s) / n, 2),
                    "p95_ms": round(s[int(0.95 * n)], 2),
                    "min_ms": round(s[0], 2),
                    "max_ms": round(s[-1], 2),
                }

            benchmark_records[name] = {
                "size_mib": size_mib,
                "plaintext_load": calc_stats(plain_load_times),
                "phase4_authenticated_load": calc_stats(phase4_load_times),
                "time_to_first_inference": calc_stats(time_to_first_inf),
                "component_breakdown": comp_avg,
            }

            print(f"  Plaintext Load Median:           {benchmark_records[name]['plaintext_load']['median_ms']} ms")
            print(f"  Phase 4 Authenticated Load Med:  {benchmark_records[name]['phase4_authenticated_load']['median_ms']} ms")
            print(f"  Time to First Inference Median:  {benchmark_records[name]['time_to_first_inference']['median_ms']} ms")
            print(f"  Component Breakdown:")
            for ck, cv in comp_avg.items():
                print(f"    - {ck:35s}: {cv} ms")

    return benchmark_records


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_benchmarks(iterations=15, device=dev)

    # Save to results/phase4/
    res_dir = os.path.join(REPO_ROOT, "results", "phase4")
    raw_dir = os.path.join(res_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    json_path = os.path.join(res_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write summary CSV
    csv_path = os.path.join(res_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Size_MiB", "Plaintext_Median_ms", "Phase4_Load_Median_ms", "TimeToFirstInf_Median_ms", "ChunkDecrypt_ms", "DeviceTransfer_ms"])
        for m, data in results.items():
            writer.writerow([
                m,
                data["size_mib"],
                data["plaintext_load"]["median_ms"],
                data["phase4_authenticated_load"]["median_ms"],
                data["time_to_first_inference"]["median_ms"],
                data["component_breakdown"]["chunk_read_and_decrypt_ms"],
                data["component_breakdown"]["host_to_device_transfer_ms"],
            ])

    print(f"\nResults successfully written to:\n- {json_path}\n- {csv_path}")
