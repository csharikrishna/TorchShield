# Phase 3 Performance Optimization & Profiling Results

**Date:** 2026-10-26  
**Hardware Environment:** NVIDIA GeForce GTX 1650 (Compute Capability 7.5 Turing, 4,095.69 MiB VRAM), Intel Core i7 CPU, Windows 11 AMD64, PyTorch 2.5.1+cu121  
**Workload:** 25.09 MiB Synthetic Transformer (8 Layers, 6.57M Parameters)  
**Timed Repetitions:** 20 iterations  
**Raw Data Reference:** `results/phase3/pipeline_profile.json`, `results/phase3/summary.csv`, `results/phase3/summary.json`  

---

## 1. Checkpoint Pipeline Latency: Phase 2 vs. Phase 3 Optimization

In Phase 3, we eliminated 3 redundant memory copies during layer reconstruction by preallocating a single contiguous `bytearray` per layer and assigning decrypted chunk slices directly into place, coupled with pinned-memory DMA host-to-device transfers (`pin_memory().to(device, non_blocking=True)`).

| Pipeline Stage / Configuration | Phase 2 Baseline (Median) | Phase 3 Optimized (Median) | Improvement / Latency Reduction |
|---|---|---|---|
| **Chunked Streaming Load (`SecureModelCheckpoint`)** | 292.59 ms (p95: 614.18 ms) | **156.17 ms** (p95: 176.42 ms) | **-46.6% latency (1.87x speedup)** |
| **Time to First Inference** | 317.35 ms (p95: 635.22 ms) | **170.24 ms** (p95: 285.93 ms) | **-46.4% latency (1.86x speedup)** |
| **Whole-File AES-256-GCM Load** | 101.56 ms (p95: 168.93 ms) | **58.34 ms** (p95: 76.18 ms) | **-42.6% latency** |
| **Plaintext Load (`torch.load`)** | 51.49 ms (p95: 272.28 ms) | **35.21 ms** (p95: 141.70 ms) | Reference floor |

---

## 2. Standardized Cipher Throughput (MiB/s)

| Workload Payload | AES-256-GCM Enc (CPU) | AES-256-GCM Dec (CPU) | AHCC Enc (GPU) | AHCC Dec (GPU) | Speedup of AES-GCM over AHCC GPU |
|---|---|---|---|---|---|
| **4 KiB** | 476.37 | 482.25 | 0.86 | 0.18 | **553.9x Enc / 2,679.2x Dec** |
| **64 KiB** | 2,705.63 | 2,705.63 | 13.76 | 2.72 | **196.6x Enc / 994.7x Dec** |
| **1 MiB** | 1,236.25 | 1,255.34 | 32.07 | 19.00 | **38.5x Enc / 66.1x Dec** |
| **16 MiB** | 1,350.88 | 1,330.20 | 32.65 | 20.06 | **41.4x Enc / 66.3x Dec** |
| **64 MiB** | **1,286.20** | **1,352.91** | **32.78** | **20.29** | **39.2x Enc / 66.7x Dec** |

---

## 3. GPU AES-GCM Status

- **Evaluation:** **INCOMPLETE / NOT RUN**
- **Rationale:** No native GPU AES-GCM library is available in this local Python/PyTorch runtime environment. Per research integrity rules, we report this limitation clearly. The measured performance of OpenSSL CPU AES-256-GCM ($1.28 - 2.70\text{ GiB/s}$) remains vastly superior to AHCC GPU ($0.03\text{ GiB/s}$), rendering custom cipher deployment unjustified.
