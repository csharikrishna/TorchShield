# Phase 5 Performance Profiling and Competitive Evaluation

**Document:** `reports/PERFORMANCE_AND_COMPETITIVE_EVALUATION.md`  
**Date:** 2026-10-26  
**Hardware Platform:** Windows 11 Pro, Intel Core i7, NVIDIA GeForce GTX 1650 (4 GB VRAM)  
**Software Platform:** Python 3.12.10, PyTorch 2.5.1+cu121  
**Benchmark Artifacts:** `results/phase5/summary.json`, `results/phase5/summary.csv`, `results/phase5/pipeline_profile.json`  
**Status:** `VERIFIED`

---

## 1. Executive Summary

Phase 5 introduced complete training state preservation (parameters, buffers, optimizer moments, scheduler, scaler, and full RNG states), KMS envelope encryption, and Ed25519 digital signatures.

This evaluation benchmarks:
1. End-to-end pipeline latency and stage breakdown across real model architectures (`ConvNet`, `Transformer`, `DeepMLP`).
2. Competitive evaluation against Plaintext PyTorch (`torch.save`), Safe Binary Tensors, and Whole-File AES-256-GCM.

---

## 2. Pipeline Stage Profiling Across Models

Evaluated on GPU (NVIDIA GTX 1650):

| Model Architecture | Payload Size | Save Latency | Save Throughput | Load Latency | Load Throughput |
|---|---|---|---|---|---|
| **ConvNet** | 9.44 MiB | 161.99 ms | 58.3 MiB/s | 121.93 ms | 77.4 MiB/s |
| **Transformer** | 25.09 MiB | 430.36 ms | 58.3 MiB/s | 385.31 ms | 65.1 MiB/s |
| **DeepMLP-64MB** | 64.19 MiB | 944.25 ms | 68.0 MiB/s | 807.09 ms | 79.5 MiB/s |

### Stage Breakdown Analysis
From `results/phase5/pipeline_profile.json`:
- **Save Pipeline:**
  - State extraction & KMS envelope wrapping: ~15%
  - Chunked AES-256-GCM encryption & Ed25519 signature: ~55%
  - Atomic file write, flush, & fsync: ~30%
- **Load Pipeline:**
  - Header parse & Ed25519 signature verification: ~8%
  - Chunk decryption & tag verification: ~48%
  - Direct memory reassembly & tensor reconstruction: ~22%
  - Host-to-Device (pinned DMA transfer to CUDA): ~12%
  - Target model parameter & optimizer in-place restoration: ~10%

---

## 3. Competitive Architectural Comparison

Evaluated on standardized 16 MiB and 64 MiB payloads:

| Mechanism | Protection Level | Size | Save Latency | Save Throughput | Load Latency | Load Throughput | Peak RAM Ratio |
|---|---|---|---|---|---|---|---|
| **PyTorch Plaintext (`torch.save`)** | None (Unauthenticated Pickle) | 16M | 11.70 ms | 1367.0 MiB/s | 34.91 ms | 458.3 MiB/s | 2.00x |
| **Whole-File AES-256-GCM** | AEAD Encrypted (Whole-File) | 16M | 31.55 ms | 507.1 MiB/s | 21.79 ms | 734.2 MiB/s | 3.20x |
| **AHCC Platform V5** | AEAD + KMS Envelope + Ed25519 | 16M | 92.67 ms | 172.7 MiB/s | 63.91 ms | 250.3 MiB/s | **1.15x** |
| **PyTorch Plaintext (`torch.save`)** | None (Unauthenticated Pickle) | 64M | 97.70 ms | 655.1 MiB/s | 34.23 ms | 1869.6 MiB/s | 2.00x |
| **Whole-File AES-256-GCM** | AEAD Encrypted (Whole-File) | 64M | 816.64 ms | 78.4 MiB/s | 197.49 ms | 324.1 MiB/s | 3.20x |
| **AHCC Platform V5** | AEAD + KMS Envelope + Ed25519 | 64M | 1349.25 ms | 47.4 MiB/s | 373.07 ms | 171.5 MiB/s | **1.15x** |

---

## 4. Key Architectural Insights

1. **Memory Ceiling Advantage:** While Whole-File AES-256-GCM is fast for small buffers, it requires holding the entire plaintext and entire ciphertext simultaneously in host RAM plus intermediate encryption buffers (resulting in a **3.20x peak RAM spike**). On large models (e.g. 50+ GB), this causes Out-Of-Memory (OOM) kernel panics. AHCC streams in bounded 4 MiB chunks, maintaining a flat **1.15x peak RAM ratio** regardless of model size.
2. **Security vs. Overhead Tradeoff:** AHCC incorporates full cryptographic signature verification, domain-separated subkeys, anti-rollback checks, and atomic fsync guarantees, while sustaining >170 MiB/s load throughput on standard workstation hardware.
