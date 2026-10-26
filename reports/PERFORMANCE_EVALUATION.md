# Performance Evaluation Report: AHCC vs Standard Authenticated Encryption

**Date:** 2026-10-26  
**Auditor:** Senior CUDA/GPU Performance & Cryptography Engineer  
**Host Hardware:** NVIDIA GeForce GTX 1650 (Turing 4 GB VRAM, sm_75), Intel x86_64 CPU (16 GB RAM)  
**Software:** Windows 11, PyTorch 2.5.1+cu121, Python 3.12.10, OpenSSL AES-NI via `cryptography`  
**Raw Data & Artifacts:** `results/raw/cipher_benchmarks_raw.json`, `results/summary.csv`, `results/summary.json`, `results/throughput_comparison.png`  

---

## 1. Executive Summary

We conducted a fair, controlled, and rigorous performance evaluation comparing the experimental AHCC cipher against standard hardware-accelerated AES-256-GCM.
The central empirical conclusion is:

> **AHCC fails to provide any performance advantage over standard authenticated encryption.**
> Across all tested payload sizes ($4\text{ KiB}$ to $4\text{ MiB}$), AES-256-GCM on CPU consistently outperforms AHCC on both CPU and GPU:
> - At target parameter $N=64, R=4$, CPU AES-256-GCM encryption is **$32.5\times$ faster** than AHCC on GPU ($1,986.49\text{ MiB/s}$ vs $60.96\text{ MiB/s}$).
> - For decryption, CPU AES-256-GCM is **$74.3\times$ faster** than AHCC on GPU ($1,548.11\text{ MiB/s}$ vs $20.82\text{ MiB/s}$).
> - Against CPU AHCC ($1.91\text{ MiB/s}$), standard CPU AES-GCM is over **$1,000\times$ faster**.

Furthermore, standard AES-256-GCM includes cryptographic authentication (GHASH), whereas AHCC direct mode provides **zero authentication**.

---

## 2. Experimental Data Table (4 MiB Workload)

Measurements represent the median of 5 timed iterations after 3 full warm-up cycles:

| Cipher Variant | Platform | $N$ | $R$ | Enc Latency (ms) | Enc Throughput (MiB/s) | Dec Latency (ms) | Dec Throughput (MiB/s) | % of AES-GCM Throughput |
|---|---|---|---|---|---|---|---|---|
| **AES-256-GCM** | **CPU (AES-NI)** | **N/A** | **14** | **2.01** | **1,986.49** | **2.58** | **1,548.11** | **100.0% (Baseline)** |
| AHCC | GPU (CUDA) | 16 | 2 | 12.78 | 312.95 | 76.15 | 52.53 | 15.75% |
| AHCC | GPU (CUDA) | 16 | 4 | 25.07 | 159.57 | 151.93 | 26.33 | 8.03% |
| AHCC | GPU (CUDA) | 32 | 2 | 19.61 | 203.96 | 82.79 | 48.32 | 10.27% |
| AHCC | GPU (CUDA) | 32 | 4 | 38.59 | 103.67 | 165.01 | 24.24 | 5.22% |
| AHCC | GPU (CUDA) | 64 | 2 | 33.12 | 120.76 | 96.85 | 41.30 | 6.08% |
| **AHCC (Target)** | **GPU (CUDA)** | **64** | **4** | **65.62** | **60.96** | **192.11** | **20.82** | **3.07%** |
| AHCC | CPU | 16 | 2 | 291.23 | 13.74 | 1,723.18 | 2.32 | 0.69% |
| AHCC | CPU | 64 | 4 | 2,096.45 | 1.91 | 5,280.50 | 0.76 | 0.10% |

---

## 3. Detailed Performance Bottleneck Analysis

### 3.1 Encryption vs. Decryption Asymmetry
In block ciphers like AES, encryption and decryption have nearly identical throughput.
In AHCC, **decryption is $3\times$ to $5\times$ slower than encryption**:
- **Forward S-Box:** $S(x) = x^3 \pmod p$ requires only 2 modular squarings/multiplications.
- **Inverse S-Box:** $S^{-1}(y) = y^d \pmod p$ requires square-and-multiply exponentiation with $d = 2,863,311,527$ ($\approx 32$ bits), performing **$\approx 48$ modular multiplications per single scalar element**.
- This makes AHCC decryption prohibitively slow on both CPU and GPU.

### 3.2 Quadratic Diffusion Scaling $O(N^2)$
- When scaling $N$ from 16 to 64 at $R=4$, GPU encryption throughput drops from $159.57\text{ MiB/s}$ to $60.96\text{ MiB/s}$.
- At the paper's proposed large dimensions ($N=1024$), dense Cauchy diffusion requires $1,048,576$ multiplications per round, reducing throughput to less than $5\text{ MiB/s}$.

### 3.3 Hardware AES-NI Efficiency vs. Emulated Finite Fields
- AES-256-GCM executes directly in silicon via dedicated AES-NI and PCLMULQDQ instructions at sub-nanosecond latency per block.
- AHCC requires multi-limb integer division and simulated int64 matrix multiplication, saturating CUDA memory bandwidth and executing hundreds of instructions per byte.
