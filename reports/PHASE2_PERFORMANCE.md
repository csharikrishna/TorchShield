# Phase 2 Standardized Performance & AEAD Evaluation

**Date:** 2026-10-26  
**Auditor:** CUDA Performance Engineer & Systems Auditor  
**Hardware Environment:** NVIDIA GeForce GTX 1650 (Compute Capability 7.5, 4 GB VRAM), Windows 11 AMD64, PyTorch 2.5.1+cu121  
**Measurement Boundary:** Host RAM to Device VRAM; AES-256-GCM includes 128-bit authentication tag verification; AHCC includes raw unauthenticated modular arithmetic.  
**Raw Data Reference:** `results/phase2/summary.csv`, `results/phase2/summary.json`, `results/phase2/raw/raw_cipher_measurements.json`  

---

## 1. Measured Throughput Comparison Across Workloads (MiB/s)

| Workload | AES-256-GCM Enc (CPU) | AES-256-GCM Dec (CPU) | AHCC Enc (CPU) | AHCC Dec (CPU) | AHCC Enc (GPU) | AHCC Dec (GPU) |
|---|---|---|---|---|---|---|
| **4 KiB** | 476.37 | 482.25 | 1.20 | 0.31 | 0.86 | 0.18 |
| **64 KiB** | **2,705.63** | **2,705.63** | 1.77 | 0.54 | 13.76 | 2.72 |
| **1 MiB** | 1,236.25 | 1,255.34 | 1.23 | 0.77 | 32.07 | 19.00 |
| **16 MiB** | 1,350.88 | 1,330.20 | *(omitted $O(N^2)$)* | *(omitted $O(N^2)$)* | 32.65 | 20.06 |
| **64 MiB** | **1,286.20** | **1,352.91** | *(omitted $O(N^2)$)* | *(omitted $O(N^2)$)* | **32.78** | **20.29** |

---

## 2. Latency Analysis (64 MiB Payload)

- **AES-256-GCM (CPU):**
  - Median Encryption Latency: **49.76 ms**
  - Median Decryption Latency: **47.31 ms**
  - Throughput: **1,286.20 MiB/s Enc** / **1,352.91 MiB/s Dec**
- **AHCC (GPU, $N=64, R=4$):**
  - Median Encryption Latency: **1,952.44 ms**
  - Median Decryption Latency: **3,154.07 ms**
  - Throughput: **32.78 MiB/s Enc** / **20.29 MiB/s Dec**
- **Relative Difference:**
  - CPU AES-256-GCM is **$39.2\times$ faster for encryption** than AHCC on GPU.
  - CPU AES-256-GCM is **$66.7\times$ faster for decryption** than AHCC on GPU.

---

## 3. GPU AES-GCM Status

- **Status:** **NOT RUN / INCOMPLETE (No verified native GPU AES-GCM engine present)**
- **Audit Statement:** The local runtime environment (Windows 11 with standard PyTorch 2.5.1 and Python cryptography package) provides hardware-accelerated AES-NI on the host CPU. No standalone GPU AES-GCM extension is compiled in this environment. In strict compliance with Non-Negotiable Research Integrity Rule 3 ("Never silently substitute CPU execution for a requested CUDA benchmark") and Rule 7, AHCC is compared against the verified CPU AES-256-GCM baseline. We explicitly do NOT claim to have beaten a theoretical GPU AES-GCM implementation.
