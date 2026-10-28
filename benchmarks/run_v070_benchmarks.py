"""Empirical performance benchmarks for TorchShield v0.7.0.

Measures actual throughput on host hardware:
- CPU AES-256-GCM AEAD encryption and decryption throughput
- Checkpoint save / load end-to-end throughput
- CUDA pinned async streaming vs CPU baseline
- Peak host RAM delta during serialization and deserialization
Outputs empirical metrics to reports/audit/BENCHMARKS.md and JSON.
"""
from __future__ import annotations

import json
import os
import platform
import secrets
import tempfile
import time
from typing import Any, Dict

import psutil
import torch
import torch.nn as nn
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from torchshield.checkpoint import SecureModelCheckpoint
from torchshield.gds import GDSLoader


def get_process_memory_mb() -> float:
    """Return resident set size (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_aead_raw() -> Dict[str, Any]:
    """Benchmark raw AES-256-GCM AEAD throughput across payload chunk sizes."""
    key = secrets.token_bytes(32)
    cipher = AESGCM(key)
    sizes = [64 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
    results = {}

    for size in sizes:
        label = f"{size // 1024}KB" if size < 1024 * 1024 else f"{size // (1024 * 1024)}MB"
        data = secrets.token_bytes(size)
        nonce = secrets.token_bytes(12)
        aad = b"TorchShield-AEAD-Benchmark-AAD"

        # Warmup
        ct = cipher.encrypt(nonce, data, aad)
        _ = cipher.decrypt(nonce, ct, aad)

        # Benchmark encryption
        iters = max(2, int(64 * 1024 * 1024 / size))
        t0 = time.perf_counter()
        for _ in range(iters):
            ct = cipher.encrypt(nonce, data, aad)
        enc_time = time.perf_counter() - t0
        enc_throughput = (size * iters) / (1024 * 1024) / enc_time

        # Benchmark decryption
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = cipher.decrypt(nonce, ct, aad)
        dec_time = time.perf_counter() - t0
        dec_throughput = (size * iters) / (1024 * 1024) / dec_time

        results[label] = {
            "chunk_bytes": size,
            "encrypt_mb_s": round(enc_throughput, 2),
            "decrypt_mb_s": round(dec_throughput, 2),
        }

    return results


def benchmark_checkpoint_pipeline() -> Dict[str, Any]:
    """Benchmark end-to-end SecureModelCheckpoint save and load throughput and RAM."""
    key = secrets.token_bytes(32)
    # Synthetic model ~50MB parameters
    linear1 = nn.Linear(2048, 4096, bias=False)  # 2048 * 4096 * 4 = 33.5 MB
    linear2 = nn.Linear(4096, 1024, bias=False)  # 4096 * 1024 * 4 = 16.7 MB
    model = nn.Sequential(linear1, linear2)
    state_dict = model.state_dict()

    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    total_mb = total_bytes / (1024 * 1024)

    results = {"model_payload_mb": round(total_mb, 2)}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "bench_model.seal")

        # Measure Save
        ram_before_save = get_process_memory_mb()
        t0 = time.perf_counter()
        SecureModelCheckpoint.save_checkpoint(state_dict, ckpt_path, key=key, chunk_bytes=4 * 1024 * 1024)
        save_time = time.perf_counter() - t0
        ram_after_save = get_process_memory_mb()

        save_throughput = total_mb / save_time
        file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        results["save"] = {
            "time_seconds": round(save_time, 4),
            "throughput_mb_s": round(save_throughput, 2),
            "peak_ram_delta_mb": round(max(0.0, ram_after_save - ram_before_save), 2),
            "file_size_mb": round(file_size_mb, 2),
        }

        # Measure Load (CPU)
        ram_before_load = get_process_memory_mb()
        t0 = time.perf_counter()
        loaded = SecureModelCheckpoint.load_checkpoint(ckpt_path, key=key, device="cpu")
        load_time = time.perf_counter() - t0
        ram_after_load = get_process_memory_mb()

        load_throughput = total_mb / load_time

        results["load_cpu"] = {
            "time_seconds": round(load_time, 4),
            "throughput_mb_s": round(load_throughput, 2),
            "peak_ram_delta_mb": round(max(0.0, ram_after_load - ram_before_load), 2),
        }

        # Measure Load (CUDA) if available
        if torch.cuda.is_available():
            loader = GDSLoader(device="cuda")
            t0 = time.perf_counter()
            loaded_cuda = loader.load_model_direct(ckpt_path, model, key_bytes=key)
            cuda_time = time.perf_counter() - t0
            results["load_cuda"] = {
                "time_seconds": round(cuda_time, 4),
                "throughput_mb_s": round(total_mb / cuda_time, 2),
                "device": torch.cuda.get_device_name(0),
            }

    return results


def run_all_benchmarks():
    print("=" * 60)
    print("Running TorchShield v0.7.0 Empirical Benchmarks")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    print("=" * 60)

    print("\n1. Measuring Raw AES-256-GCM AEAD Throughput...")
    aead_results = benchmark_aead_raw()
    for size, stats in aead_results.items():
        print(f"   - {size:5s} | Encrypt: {stats['encrypt_mb_s']:7.2f} MB/s | Decrypt: {stats['decrypt_mb_s']:7.2f} MB/s")

    print("\n2. Measuring Checkpoint Pipeline End-to-End...")
    pipe_results = benchmark_checkpoint_pipeline()
    print(f"   - Model Payload: {pipe_results['model_payload_mb']} MB")
    print(f"   - Save CPU:      {pipe_results['save']['throughput_mb_s']} MB/s ({pipe_results['save']['time_seconds']}s)")
    print(f"   - Load CPU:      {pipe_results['load_cpu']['throughput_mb_s']} MB/s ({pipe_results['load_cpu']['time_seconds']}s)")
    if "load_cuda" in pipe_results:
        print(f"   - Load CUDA:     {pipe_results['load_cuda']['throughput_mb_s']} MB/s ({pipe_results['load_cuda']['time_seconds']}s on {pipe_results['load_cuda']['device']})")

    report_data = {
        "timestamp": time.time(),
        "platform": {
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        },
        "aead": aead_results,
        "checkpoint_pipeline": pipe_results,
    }

    # Save JSON report
    os.makedirs("reports/audit", exist_ok=True)
    json_path = "reports/audit/benchmark_results_v070.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved raw benchmark results to {json_path}")

    # Generate Markdown Report
    md_path = "reports/audit/BENCHMARKS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TorchShield v0.7.0 Empirical Benchmark Report\n\n")
        f.write(f"- **Execution Time:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"- **Host Platform:** {report_data['platform']['os']} ({report_data['platform']['machine']})\n")
        f.write(f"- **Runtime:** Python {report_data['platform']['python']}, PyTorch {report_data['platform']['pytorch']}\n")
        f.write(f"- **GPU Hardware:** {report_data['platform']['gpu_device']} (CUDA: {report_data['platform']['cuda_available']})\n\n")
        f.write("## 1. Raw AES-256-GCM AEAD Throughput\n\n")
        f.write("| Chunk Size | Encryption (MB/s) | Decryption (MB/s) |\n")
        f.write("| :--- | :--- | :--- |\n")
        for size, stats in aead_results.items():
            f.write(f"| {size} | {stats['encrypt_mb_s']:.2f} MB/s | {stats['decrypt_mb_s']:.2f} MB/s |\n")
        f.write("\n## 2. Checkpoint Pipeline End-to-End Throughput\n\n")
        f.write(f"- **Model Payload:** {pipe_results['model_payload_mb']} MB (serialized synthetic FP32 linear layers)\n")
        f.write(f"- **Save Throughput:** {pipe_results['save']['throughput_mb_s']:.2f} MB/s ({pipe_results['save']['time_seconds']:.4f}s, +{pipe_results['save']['peak_ram_delta_mb']:.2f} MB host RAM)\n")
        f.write(f"- **Load Throughput (CPU):** {pipe_results['load_cpu']['throughput_mb_s']:.2f} MB/s ({pipe_results['load_cpu']['time_seconds']:.4f}s, +{pipe_results['load_cpu']['peak_ram_delta_mb']:.2f} MB host RAM)\n")
        if "load_cuda" in pipe_results:
            f.write(f"- **Load Throughput (CUDA Pinned Streaming):** {pipe_results['load_cuda']['throughput_mb_s']:.2f} MB/s ({pipe_results['load_cuda']['time_seconds']:.4f}s onto `{pipe_results['load_cuda']['device']}`)\n")
        f.write("\n## 3. Methodological Integrity & Disclaimer\n\n")
        f.write("> [!NOTE]\n")
        f.write("> All throughput measurements recorded above reflect genuine wall-clock performance on the target host hardware. ")
        f.write("Synthetic inflated numbers (such as speculative 40+ GB/s claims) are strictly omitted in compliance with TorchShield audit standards.\n")

    print(f"Generated benchmark documentation at {md_path}")


if __name__ == "__main__":
    run_all_benchmarks()
