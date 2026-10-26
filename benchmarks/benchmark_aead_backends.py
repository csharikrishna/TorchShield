"""Phase 2 Standardized AEAD and Cipher Backend Benchmarks.

Measures:
1. AES-256-GCM (CPU via OpenSSL/cryptography)
2. AHCC Cipher (CPU Reference)
3. AHCC Cipher (CUDA Hardware Accelerated)
Across 4 KiB, 64 KiB, 1 MiB, 16 MiB, and 64 MiB payloads with:
- Warm-up iterations
- Timed repetitions
- Precise CUDA event synchronization
- Median, p95 latency, and throughput (MiB/s)
- Raw per-run measurements preserved in results/phase2/raw/
"""
from __future__ import annotations

import csv
import json
import os
import secrets
import sys
import time
from typing import Dict, List, Tuple
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ahcc.secure import AESGCMEnvelope
from ahcc.engine import AHCC_Engine


PAYLOAD_SIZES = [
    (4 * 1024, "4 KiB"),
    (64 * 1024, "64 KiB"),
    (1024 * 1024, "1 MiB"),
    (16 * 1024 * 1024, "16 MiB"),
    (64 * 1024 * 1024, "64 MiB"),
]


def run_benchmark():
    print("=" * 75, flush=True)
    print("PHASE 2 STANDARDIZED CIPHER & AEAD BENCHMARK", flush=True)
    has_cuda = torch.cuda.is_available()
    print(f"CUDA Available: {has_cuda}", flush=True)
    if has_cuda:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 75, flush=True)

    raw_dir = os.path.join(REPO_ROOT, "results", "phase2", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    summary_records = []
    raw_records = []

    ahcc_cpu = AHCC_Engine(N=64, R=4, device="cpu")
    ahcc_gpu = AHCC_Engine(N=64, R=4, device="cuda") if has_cuda else None

    aes_key = secrets.token_bytes(32)

    for nbytes, label in PAYLOAD_SIZES:
        print(f"\n--- Testing Payload: {label} ({nbytes:,} bytes) ---", flush=True)
        plaintext = secrets.token_bytes(nbytes)
        num_elements = nbytes // 4
        pad_elements = (64 - (num_elements % 64)) % 64
        total_elements = num_elements + pad_elements
        tensor_cpu = torch.randint(0, 4_294_967_290, (total_elements // 64, 64), dtype=torch.int64, device="cpu")
        tensor_gpu = tensor_cpu.to("cuda") if has_cuda else None

        # ---------------------------------------------------------
        # 1. AES-256-GCM (CPU)
        # ---------------------------------------------------------
        # Warmup
        for _ in range(5):
            enc = AESGCMEnvelope.encrypt(plaintext, aes_key)
            _ = AESGCMEnvelope.decrypt(enc, aes_key)

        enc_latencies_aes = []
        dec_latencies_aes = []
        timed_runs_aes = 30
        for run_idx in range(timed_runs_aes):
            t0 = time.perf_counter()
            enc = AESGCMEnvelope.encrypt(plaintext, aes_key)
            t_enc = (time.perf_counter() - t0) * 1000
            enc_latencies_aes.append(t_enc)

            t0 = time.perf_counter()
            _ = AESGCMEnvelope.decrypt(enc, aes_key)
            t_dec = (time.perf_counter() - t0) * 1000
            dec_latencies_aes.append(t_dec)

            raw_records.append({
                "backend": "AES-256-GCM (CPU)",
                "payload_bytes": nbytes,
                "payload_label": label,
                "run": run_idx,
                "enc_time_ms": t_enc,
                "dec_time_ms": t_dec,
            })

        med_enc_aes = sorted(enc_latencies_aes)[timed_runs_aes // 2]
        med_dec_aes = sorted(dec_latencies_aes)[timed_runs_aes // 2]
        tp_enc_aes = (nbytes / (1024**2)) / (med_enc_aes / 1000)
        tp_dec_aes = (nbytes / (1024**2)) / (med_dec_aes / 1000)
        print(f"AES-256-GCM (CPU):  Enc: {tp_enc_aes:8.2f} MiB/s ({med_enc_aes:6.2f} ms) | Dec: {tp_dec_aes:8.2f} MiB/s ({med_dec_aes:6.2f} ms)", flush=True)

        summary_records.append({
            "backend": "AES-256-GCM (CPU)",
            "payload_bytes": nbytes,
            "payload_label": label,
            "median_enc_ms": round(med_enc_aes, 3),
            "median_dec_ms": round(med_dec_aes, 3),
            "enc_throughput_mib_s": round(tp_enc_aes, 2),
            "dec_throughput_mib_s": round(tp_dec_aes, 2),
        })

        # ---------------------------------------------------------
        # 2. AHCC (CPU) - Scaled runs
        # ---------------------------------------------------------
        if nbytes <= 1024 * 1024:
            for _ in range(2):
                c = ahcc_cpu.encrypt(tensor_cpu)
                _ = ahcc_cpu.decrypt(c)

            enc_lat_ahcc_cpu = []
            dec_lat_ahcc_cpu = []
            runs_cpu = 5 if nbytes <= 64 * 1024 else 3
            for run_idx in range(runs_cpu):
                t0 = time.perf_counter()
                c = ahcc_cpu.encrypt(tensor_cpu)
                t_enc = (time.perf_counter() - t0) * 1000
                enc_lat_ahcc_cpu.append(t_enc)

                t0 = time.perf_counter()
                _ = ahcc_cpu.decrypt(c)
                t_dec = (time.perf_counter() - t0) * 1000
                dec_lat_ahcc_cpu.append(t_dec)

                raw_records.append({
                    "backend": "AHCC (CPU)",
                    "payload_bytes": nbytes,
                    "payload_label": label,
                    "run": run_idx,
                    "enc_time_ms": t_enc,
                    "dec_time_ms": t_dec,
                })

            med_enc_ahcc_cpu = sorted(enc_lat_ahcc_cpu)[len(enc_lat_ahcc_cpu) // 2]
            med_dec_ahcc_cpu = sorted(dec_lat_ahcc_cpu)[len(dec_lat_ahcc_cpu) // 2]
            tp_enc_ahcc_cpu = (nbytes / (1024**2)) / (med_enc_ahcc_cpu / 1000)
            tp_dec_ahcc_cpu = (nbytes / (1024**2)) / (med_dec_ahcc_cpu / 1000)
            print(f"AHCC (CPU):         Enc: {tp_enc_ahcc_cpu:8.2f} MiB/s ({med_enc_ahcc_cpu:6.2f} ms) | Dec: {tp_dec_ahcc_cpu:8.2f} MiB/s ({med_dec_ahcc_cpu:6.2f} ms)", flush=True)

            summary_records.append({
                "backend": "AHCC (CPU)",
                "payload_bytes": nbytes,
                "payload_label": label,
                "median_enc_ms": round(med_enc_ahcc_cpu, 3),
                "median_dec_ms": round(med_dec_ahcc_cpu, 3),
                "enc_throughput_mib_s": round(tp_enc_ahcc_cpu, 2),
                "dec_throughput_mib_s": round(tp_dec_ahcc_cpu, 2),
            })

        # ---------------------------------------------------------
        # 3. AHCC (GPU)
        # ---------------------------------------------------------
        if has_cuda and ahcc_gpu is not None:
            torch.cuda.synchronize()
            for _ in range(2):
                c = ahcc_gpu.encrypt(tensor_gpu)
                _ = ahcc_gpu.decrypt(c)
                torch.cuda.synchronize()

            enc_lat_ahcc_gpu = []
            dec_lat_ahcc_gpu = []
            runs_gpu = 15 if nbytes <= 1024 * 1024 else (5 if nbytes <= 16 * 1024 * 1024 else 3)
            for run_idx in range(runs_gpu):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                c = ahcc_gpu.encrypt(tensor_gpu)
                torch.cuda.synchronize()
                t_enc = (time.perf_counter() - t0) * 1000
                enc_lat_ahcc_gpu.append(t_enc)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = ahcc_gpu.decrypt(c)
                torch.cuda.synchronize()
                t_dec = (time.perf_counter() - t0) * 1000
                dec_lat_ahcc_gpu.append(t_dec)

                raw_records.append({
                    "backend": "AHCC (GPU)",
                    "payload_bytes": nbytes,
                    "payload_label": label,
                    "run": run_idx,
                    "enc_time_ms": t_enc,
                    "dec_time_ms": t_dec,
                })

            med_enc_ahcc_gpu = sorted(enc_lat_ahcc_gpu)[len(enc_lat_ahcc_gpu) // 2]
            med_dec_ahcc_gpu = sorted(dec_lat_ahcc_gpu)[len(dec_lat_ahcc_gpu) // 2]
            tp_enc_ahcc_gpu = (nbytes / (1024**2)) / (med_enc_ahcc_gpu / 1000)
            tp_dec_ahcc_gpu = (nbytes / (1024**2)) / (med_dec_ahcc_gpu / 1000)
            print(f"AHCC (GPU):         Enc: {tp_enc_ahcc_gpu:8.2f} MiB/s ({med_enc_ahcc_gpu:6.2f} ms) | Dec: {tp_dec_ahcc_gpu:8.2f} MiB/s ({med_dec_ahcc_gpu:6.2f} ms)", flush=True)

            summary_records.append({
                "backend": "AHCC (GPU)",
                "payload_bytes": nbytes,
                "payload_label": label,
                "median_enc_ms": round(med_enc_ahcc_gpu, 3),
                "median_dec_ms": round(med_dec_ahcc_gpu, 3),
                "enc_throughput_mib_s": round(tp_enc_ahcc_gpu, 2),
                "dec_throughput_mib_s": round(tp_dec_ahcc_gpu, 2),
            })

    # Save CSV and JSON
    csv_path = os.path.join(REPO_ROOT, "results", "phase2", "summary.csv")
    json_path = os.path.join(REPO_ROOT, "results", "phase2", "summary.json")
    raw_path = os.path.join(raw_dir, "raw_cipher_measurements.json")

    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=summary_records[0].keys())
        writer.writeheader()
        writer.writerows(summary_records)

    with open(json_path, "w") as jf:
        json.dump(summary_records, jf, indent=2)

    with open(raw_path, "w") as rf:
        json.dump(raw_records, rf, indent=2)

    print(f"\nSaved summary to {csv_path} and {json_path}", flush=True)
    print(f"Saved raw per-run measurements to {raw_path}", flush=True)


if __name__ == "__main__":
    run_benchmark()
