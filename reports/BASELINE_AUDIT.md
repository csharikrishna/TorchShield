# Phase Zero Baseline Audit Report

**Date:** 2026-10-26  
**Auditor:** Senior Cryptography & CUDA Systems Research Engineer  
**Baseline Git Commit:** `02156fd` (Master initial snapshot)  
**Host Environment:** Windows 11 (build 10.0.26220), AMD64, 16 GB RAM  
**GPU:** NVIDIA GeForce GTX 1650 (Compute Capability 7.5 Turing, 4,096 MiB VRAM), Driver 617.42, CUDA UMD 13.4  
**Python Runtime:** Python 3.12.10, PyTorch 2.5.1+cu121 (CUDA active and detected)  

---

## 1. Inventory & Project Architecture

The AHCC (Adaptive Hyperdimensional Chaotic Cryptography) codebase consists of:
- `ahcc/engine.py`: Core `AHCC_Engine` and `TensorCipherWrapper`. Implements the Substitution-Permutation Network (SPN) over prime fields $\mathbb{F}_p$ ($p=2^{31}-19$ and $p=2^{32}-5$), with cubic S-box $S(x) = x^3 \pmod p$, dense Cauchy MDS diffusion matrix $M_{i,j} = (x_i + y_j)^{-1} \pmod p$, and two key schedule modes (default HKDF-SHA256 and legacy chaotic polynomial recurrence).
- `ahcc/secure.py`: `AESGCMEnvelope` and `TensorAESGCMEnvelope`. Provides standard AES-256-GCM AEAD encryption with 96-bit random nonces and encrypted/authenticated JSON metadata. Completely independent of AHCC.
- `benchmarks/`: `benchmark_crypto.py`, `benchmark_phase3.py`, `check_cuda.py`.
- `demos/`: `ai_security_use_cases.py` (toy federated learning and toy parameter encryption).
- `tests/`: 24 tests covering invertibility, basic avalanche diagnostic, key schedule validation, and AES-GCM envelope tampering.

---

## 2. Baseline Test Execution Results

Command executed:
```bash
python -m pytest -v
python run_all.py
```
Exit code: `0` (All 24 pytest tests passed; all smoke tests in `run_all.py` completed).  
Log saved to: `reports/baseline_test_log.txt`

### Detailed Breakdown
| Test Group | Checks | Status | Note |
|---|---|---|---|
| `test_phase1_invertibility_and_algebra` | Bijectivity $gcd(3, p-1)=1$, $M \cdot M^{-1} \equiv I \pmod p$, round-trip on 4D tensor | **VERIFIED** | Both primes $p=2147483579$ and $p=4294967291$ verified exact. |
| `test_phase2_avalanche_effect` | Empirical single-bit flip on 100 random blocks | **PARTIAL** | Ratio ~50.27%. This is a diagnostic metric, NOT a proof of SAC or cryptographic security. |
| `test_key_sensitivity_and_statistical_uniformity` | Seed $50000$ vs $50001$ bit sensitivity; mean/variance | **PARTIAL** | Normalized mean 0.4984, variance 0.0843. Diagnostic only; does not substitute for NIST SP 800-22 or cryptanalysis. |
| `test_engine_hardening` | Types, shapes, boundaries, float quantization | **VERIFIED** | Edge cases reject invalid shapes, non-integer dtypes, non-finite floats. |
| `test_key_schedule` | HKDF determinism, rejection sampling, legacy seed mode | **VERIFIED** | HKDF mode handles 256-bit keys; legacy seed mode explicitly warns. |
| `test_secure_envelope` | AES-256-GCM envelope round trips, nonce freshness, tampering | **VERIFIED** | Standard AEAD path functions correctly on CPU. |

---

## 3. Discrepancies, Unsupported Claims & Reality Gaps

1. **Throughput Gap:**
   - The research paper claims AHCC will achieve high throughput outperforming AES on GPUs.
   - Actual baseline measurement on CUDA (`benchmark_phase3.py` in `run_all.py`):
     - ResNet tensor ($150,528$ elements): **63.02 MB/s** (18.22 ms)
     - Batch Vision tensor ($602,112$ elements): **64.81 MB/s** (70.89 ms)
     - Transformer Layer ($262,144$ elements): **62.55 MB/s** (31.97 ms)
   - Contrast: CPU AES-256-GCM on the same host achieves over **5,000 MB/s** (5 GB/s). AHCC on CUDA is nearly **80x to 100x slower** than standard AES-GCM on CPU.
2. **CUDA Implementation Bottleneck:**
   - PyTorch on CUDA does not support `int64` matrix multiplication (`"addmm_cuda" not implemented for 'Long'`).
   - Consequently, `AHCC_Engine._int64_mm` performs an outer-product expansion `(A.unsqueeze(2) * B.unsqueeze(0)).sum(dim=1)` in small tiles, creating severe memory bandwidth bottlenecks.
   - Tensor Cores cannot perform exact modular arithmetic modulo $2^{32}-5$; claims that AHCC maps naturally to Tensor Cores are mathematically false.
3. **No Authentication in AHCC:**
   - `AHCC_Engine.encrypt()` and `decrypt()` produce unauthenticated ciphertexts. Tampering with ciphertext produces garbage plaintext without raising errors.
4. **Legacy Key Schedule Security Failure:**
   - The paper's chaotic key schedule as implemented in `master_seed` mode has an internal state space of exactly $p \approx 2^{32}$, providing at most **32 bits of security** regardless of seed size.
5. **Federated Learning Claim Clarification:**
   - The FL demo in `demos/ai_security_use_cases.py` decrypts client gradients before aggregating them. It provides **in-transit encryption**, NOT secure multi-party aggregation or homomorphic encryption.
