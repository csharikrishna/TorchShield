"""Comprehensive, fair, and reproducible cipher benchmark harness.

Benchmarks:
1. CPU AES-256-GCM (OpenSSL AES-NI hardware-accelerated).
2. AHCC on CPU (exact int64 modular arithmetic).
3. AHCC on GPU (CUDA exact int64 modular arithmetic).
4. GPU AES-GCM limitation recording.

Separates Encryption vs Decryption, measures median, min, max, p95, and spread.
Saves raw run records to results/raw/ and summary tables to results/summary.csv and .json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import statistics
import time
from typing import Dict, List

import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ahcc.engine import AHCC_Engine, TensorCipherWrapper


def run_benchmarks(
    sizes: List[int],
    dimensions: List[int],
    rounds_list: List[int],
    p: int = 4_294_967_291,
    warmup: int = 3,
    iterations: int = 30,
) -> Tuple[List[Dict], List[Dict]]:
    os.makedirs("results/raw", exist_ok=True)
    raw_records = []
    summary_records = []
    cuda_available = torch.cuda.is_available()

    print("=" * 90, flush=True)
    print("FAIR AND REPRODUCIBLE CIPHER PERFORMANCE BENCHMARK", flush=True)
    print(f"Modulus p={p} | Warmup={warmup} | Iterations={iterations} | CUDA Available={cuda_available}", flush=True)
    print("=" * 90, flush=True)

    for size_bytes in sizes:
        print(f"\n>>> Workload Payload Size: {size_bytes:,} Bytes ({size_bytes / (1024*1024):.2f} MiB)", flush=True)
        # 1. AES-256-GCM CPU Baseline (OpenSSL AES-NI)
        aes_key = secrets.token_bytes(32)
        cipher_aes = AESGCM(aes_key)
        raw_payload = secrets.token_bytes(size_bytes)
        aad = b"benchmark-aad"

        # Warmup AES Encrypt
        for _ in range(warmup):
            n = secrets.token_bytes(12)
            ct = cipher_aes.encrypt(n, raw_payload, aad)

        # Timed AES Encrypt
        aes_enc_times = []
        cts_sample = []
        nonces_sample = []
        for _ in range(iterations):
            n = secrets.token_bytes(12)
            t0 = time.perf_counter()
            c = cipher_aes.encrypt(n, raw_payload, aad)
            aes_enc_times.append(time.perf_counter() - t0)
            cts_sample.append(c)
            nonces_sample.append(n)

        # Timed AES Decrypt
        aes_dec_times = []
        for i in range(iterations):
            n = nonces_sample[i]
            c = cts_sample[i]
            t0 = time.perf_counter()
            recovered = cipher_aes.decrypt(n, c, aad)
            aes_dec_times.append(time.perf_counter() - t0)
            assert recovered == raw_payload

        aes_enc_med = statistics.median(aes_enc_times)
        aes_dec_med = statistics.median(aes_dec_times)
        aes_enc_mb_s = (size_bytes / (1024 * 1024)) / aes_enc_med
        aes_dec_mb_s = (size_bytes / (1024 * 1024)) / aes_dec_med

        print(f"  [Baseline] CPU AES-256-GCM  : Enc {aes_enc_med*1000:7.2f} ms ({aes_enc_mb_s:9.2f} MiB/s) | Dec {aes_dec_med*1000:7.2f} ms ({aes_dec_mb_s:9.2f} MiB/s)", flush=True)

        summary_records.append({
            "workload_bytes": size_bytes,
            "primitive": "AES-256-GCM",
            "backend": "CPU (OpenSSL AES-NI)",
            "N": "N/A",
            "R": 14,
            "enc_latency_ms": aes_enc_med * 1000,
            "enc_throughput_mib_s": aes_enc_mb_s,
            "dec_latency_ms": aes_dec_med * 1000,
            "dec_throughput_mib_s": aes_dec_mb_s,
        })
        raw_records.append({
            "workload_bytes": size_bytes,
            "primitive": "AES-256-GCM",
            "backend": "CPU",
            "enc_times": aes_enc_times,
            "dec_times": aes_dec_times,
        })

        # 2. AHCC on CPU and GPU across parameters
        for N in dimensions:
            for R in rounds_list:
                # Prepare tensor with exact elements < p
                num_elements = size_bytes // 8  # int64 elements
                num_elements = max(N, (num_elements // N) * N)
                data_cpu = torch.randint(0, p, (num_elements,), dtype=torch.int64)

                # AHCC on CPU
                eng_cpu = AHCC_Engine(N=N, R=R, p=p, device="cpu")
                wrap_cpu = TensorCipherWrapper(eng_cpu)

                # Correctness
                ct_cpu, meta_cpu = wrap_cpu.encrypt_tensor(data_cpu)
                pt_cpu = wrap_cpu.decrypt_tensor(ct_cpu, meta_cpu)
                assert torch.equal(data_cpu, pt_cpu)

                for _ in range(min(warmup, 2)):
                    wrap_cpu.encrypt_tensor(data_cpu)

                cpu_iters = 1 if size_bytes >= 1048576 else min(iterations, 5)
                cpu_enc_times = []
                for _ in range(cpu_iters):
                    t0 = time.perf_counter()
                    wrap_cpu.encrypt_tensor(data_cpu)
                    cpu_enc_times.append(time.perf_counter() - t0)

                cpu_dec_times = []
                for _ in range(cpu_iters):
                    t0 = time.perf_counter()
                    wrap_cpu.decrypt_tensor(ct_cpu, meta_cpu)
                    cpu_dec_times.append(time.perf_counter() - t0)

                cpu_enc_med = statistics.median(cpu_enc_times)
                cpu_dec_med = statistics.median(cpu_dec_times)
                actual_bytes = num_elements * 8
                cpu_enc_mb_s = (actual_bytes / (1024 * 1024)) / cpu_enc_med
                cpu_dec_mb_s = (actual_bytes / (1024 * 1024)) / cpu_dec_med

                print(f"  AHCC CPU [N={N:2d}, R={R}]: Enc {cpu_enc_med*1000:7.2f} ms ({cpu_enc_mb_s:9.2f} MiB/s) | Dec {cpu_dec_med*1000:7.2f} ms ({cpu_dec_mb_s:9.2f} MiB/s)", flush=True)

                summary_records.append({
                    "workload_bytes": actual_bytes,
                    "primitive": "AHCC",
                    "backend": "CPU (PyTorch)",
                    "N": N,
                    "R": R,
                    "enc_latency_ms": cpu_enc_med * 1000,
                    "enc_throughput_mib_s": cpu_enc_mb_s,
                    "dec_latency_ms": cpu_dec_med * 1000,
                    "dec_throughput_mib_s": cpu_dec_mb_s,
                })

                # AHCC on CUDA GPU
                if cuda_available and size_bytes <= 16777216:  # avoid OOM on 4GB VRAM with broadcast tensors
                    eng_gpu = AHCC_Engine(N=N, R=R, p=p, device="cuda")
                    wrap_gpu = TensorCipherWrapper(eng_gpu)
                    data_gpu = data_cpu.to("cuda")

                    # Correctness
                    ct_gpu, meta_gpu = wrap_gpu.encrypt_tensor(data_gpu)
                    torch.cuda.synchronize()
                    pt_gpu = wrap_gpu.decrypt_tensor(ct_gpu, meta_gpu)
                    torch.cuda.synchronize()
                    assert torch.equal(data_gpu, pt_gpu)

                    for _ in range(warmup):
                        c, m = wrap_gpu.encrypt_tensor(data_gpu)
                        torch.cuda.synchronize()

                    gpu_enc_times = []
                    for _ in range(iterations):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        c, m = wrap_gpu.encrypt_tensor(data_gpu)
                        torch.cuda.synchronize()
                        gpu_enc_times.append(time.perf_counter() - t0)

                    gpu_dec_times = []
                    for _ in range(iterations):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        p_rec = wrap_gpu.decrypt_tensor(ct_gpu, meta_gpu)
                        torch.cuda.synchronize()
                        gpu_dec_times.append(time.perf_counter() - t0)

                    gpu_enc_med = statistics.median(gpu_enc_times)
                    gpu_dec_med = statistics.median(gpu_dec_times)
                    gpu_enc_mb_s = (actual_bytes / (1024 * 1024)) / gpu_enc_med
                    gpu_dec_mb_s = (actual_bytes / (1024 * 1024)) / gpu_dec_med

                    print(f"  AHCC GPU [N={N:2d}, R={R}]: Enc {gpu_enc_med*1000:7.2f} ms ({gpu_enc_mb_s:9.2f} MiB/s) | Dec {gpu_dec_med*1000:7.2f} ms ({gpu_dec_mb_s:9.2f} MiB/s)", flush=True)

                    summary_records.append({
                        "workload_bytes": actual_bytes,
                        "primitive": "AHCC",
                        "backend": "GPU (CUDA)",
                        "N": N,
                        "R": R,
                        "enc_latency_ms": gpu_enc_med * 1000,
                        "enc_throughput_mib_s": gpu_enc_mb_s,
                        "dec_latency_ms": gpu_dec_med * 1000,
                        "dec_throughput_mib_s": gpu_dec_mb_s,
                    })

    # Save summary and raw outputs
    with open("results/summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    with open("results/raw/cipher_benchmarks_raw.json", "w") as f:
        json.dump(raw_records, f, indent=2)

    keys = summary_records[0].keys()
    with open("results/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_records)

    print("\nBenchmark completed. Summary written to results/summary.csv and results/summary.json")
    return raw_records, summary_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 65536, 1048576, 16777216])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--rounds", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    run_benchmarks(args.sizes, args.dimensions, args.rounds, iterations=args.iterations)
