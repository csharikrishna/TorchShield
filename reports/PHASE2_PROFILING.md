# Phase 2 Checkpoint Pipeline Profiling & Bottleneck Analysis

**Date:** 2026-10-26  
**Hardware:** NVIDIA GeForce GTX 1650 (Compute Capability 7.5, 4 GB VRAM), Intel Core i7 Host CPU, Windows 11  
**Target Workload:** Synthetic Transformer Model (6,576,650 float32 parameters, 25.09 MiB)  
**Iterations:** 20 timed repetitions  
**Raw Data Reference:** `results/phase2/checkpoint_pipeline_profile.json`  

---

## 1. Measured Performance Breakdown

| Metric | Plaintext (`torch.save` / `torch.load`) | Whole-File AES-256-GCM | Chunked Streaming AEAD (`SecureModelCheckpoint` v2) |
|---|---|---|---|
| **Save / Encryption Latency (Median)** | 42.18 ms | 88.45 ms | 185.32 ms |
| **Save / Encryption Latency (p95)** | 112.54 ms | 154.21 ms | 310.45 ms |
| **Load / Decryption Latency (Median)** | **51.49 ms** | **101.56 ms** | **292.59 ms** |
| **Load / Decryption Latency (p95)** | 272.28 ms | 168.93 ms | 614.18 ms |
| **Time to First Inference (Median)** | — | — | **317.35 ms** |
| **Time to First Inference (p95)** | — | — | **635.22 ms** |

---

## 2. Profiling Insights & Bottleneck Identification

1. **Authentication and Chunking Overhead:**
   - Whole-file AES-256-GCM adds only $+50.07\text{ ms}$ over plaintext loading for a 25 MiB model.
   - Chunked streaming introduces additional Python-level chunk loop overhead, manifest digest verification, and per-chunk AES-GCM envelope decoding ($+191.03\text{ ms}$ over whole-file).
   - In exchange for this latency difference, chunked streaming provides:
     - Immediate rejection of tampered layers before memory allocation.
     - Direct in-place streaming into `target_model` without buffering whole uncompressed checkpoints in system RAM.
     - Resilience against memory exhaustion on large models.
2. **Device Transfer Overlap:**
   - On the local GTX 1650, transferring 25 MiB over PCIe 3.0 requires $\sim 6.5\text{ ms}$.
   - Time to first inference ($317.35\text{ ms}$) is dominated by initial model compilation and stream verification ($292.59\text{ ms}$), with only $24.76\text{ ms}$ required to execute the first forward pass.
