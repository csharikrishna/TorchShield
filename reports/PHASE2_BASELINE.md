# Phase 2 Baseline Verification & Ground Truth Audit

**Date:** 2026-10-26  
**Auditor:** Senior Application Security & Cryptographic Engineer  
**Baseline Git Revision:** `ec12c2d` (Tagged `phase1-baseline`)  
**Operating System:** Windows 11 Pro (10.0.26220)  
**Python Runtime:** 3.12.10 (64-bit AMD64)  
**PyTorch Version:** 2.5.1+cu121  
**Hardware Accelerator:** NVIDIA GeForce GTX 1650 (Compute Capability 7.5, 4 GB VRAM)  
**Host Memory:** 15.84 GiB RAM  

---

## 1. Ground Truth Inspection & Repository State

### 1.1 Git History & Revision Tracking
Prior audit reports referenced initial commit `02156fd` and subsequent Phase 1 work `4a01810`. Inspection of `git log` confirms the exact commit sequence:
1. `02156fd`: baseline initial untouched repository state.
2. `4a01810`: feat: complete master research & engineering mission on AHCC across all 14 phases.
3. `ec12c2d`: fix(checkpoint): support 0-dim scalars, empty tensors, and target_model buffer streaming; add test_attack_experiments and packaging archive.

Prior to introducing any Phase 2 changes, an immutable Git tag was established:
```bash
git tag phase1-baseline
```
Git working tree is verified clean.

### 1.2 Verification of Existing Test Suite
The full existing test suite was executed:
- Command: `pytest -v`
- Result: **133 passed, 0 failed, 0 skipped** in 13.65s.
- Detailed per-test log saved to: `reports/PHASE2_BASELINE_TESTS.txt`.

### 1.3 Verification of Baseline Benchmarks
Baseline benchmarks were executed against actual hardware:
- `python -m benchmarks.benchmark_tensor_loading`
  - Synthetic Transformer model (8 layers, 4,202,496 float32 parameters, 16.03 MiB):
    - Plaintext PyTorch loading: 14.88 ms
    - Whole-File AES-256-GCM loading: 32.46 ms (+17.59 ms overhead)
    - Chunked Streaming AEAD (SecureModelCheckpoint v1): 40.61 ms (+25.74 ms overhead)
    - AHCC Decryption (Single layer only): 51.98 ms
- `python -m benchmarks.benchmark_crypto --device cuda`
  - AHCC CUDA 64 KiB: 5.569 ms (11.22 MiB/s)
  - AES-GCM CPU (OpenSSL) 64 KiB: 0.033 ms (1,923.08 MiB/s)

---

## 2. Identified Discrepancies and Audit Corrections

1. **Unrealistic Algebraic Degree Attack Claim:**
   The Phase 1 report claimed:
   > *"AHCC specifies R = 4 rounds. With R=4, the global algebraic degree is <= 81, which is completely breakable with fewer than 100 known plaintext-ciphertext queries."*
   **Correction:** This claim treated an $N$-dimensional multivariate Substitution-Permutation Network as a 1-dimensional univariate polynomial. With $N=64$ and dense Cauchy mixing across 4 rounds, each coordinate polynomial contains up to $\approx \binom{73}{9} \approx 6 \times 10^{10}$ monomials. A univariate interpolation attack cannot succeed in $<100$ queries. While the low degree of $x^3$ is a severe design vulnerability compared to standard primitives, claim of a $<100$-query break was mathematically unfounded without a concrete multivariate solving algorithm.

2. **Nonce Derivation in Checkpoint v1:**
   In `ahcc/checkpoint.py` (v1), `AESGCMEnvelope.encrypt()` generates a fresh 96-bit random nonce via `secrets.token_bytes(12)` for each chunk. While random nonces in AES-GCM are acceptable for small numbers of messages, NIST SP 800-38D explicitly warns against pure random nonces under the same key across large chunk counts due to birthday-bound collision probabilities ($2^{-32}$ collision risk after $2^{32}$ nonces). In addition, re-encrypting or resuming interrupted chunk writes without per-artifact key isolation creates nonce reuse hazards.
   **Remediation:** Implement deterministic, collision-free chunk nonce derivation bound to an artifact-isolated key derived via HKDF-SHA256.

3. **Atomic File Replacement & Crash Resilience:**
   `ahcc/checkpoint.py` directly opened destination files with `open(output_path, "wb")`. A process crash, out-of-disk condition, or power failure mid-write leaves a corrupt or truncated file at `output_path`, potentially destroying the previous valid checkpoint.
   **Remediation:** Implement strict atomic write semantics: temporary staging (`.tmp.<uuid>`), explicit disk flush (`flush() + os.fsync()`), and atomic replacement (`os.replace()`).

4. **Resource Exhaustion & Allocation Safety:**
   Tensor deserialization currently reconstructs tensors based on metadata without bounding maximum allowed dimensions, tensor counts, or total memory allocation. A malicious file claiming a shape of `[1000000, 1000000]` could trigger host OOM denial-of-service before or during tensor buffer allocation.
   **Remediation:** Implement defensive allocation ceilings and strict tensor header constraints.

---

## 3. Preserved Artifacts
All Phase 1 results in `results/raw/`, `results/summary.csv`, and `results/summary.json` remain completely preserved. All Phase 2 benchmarks will be recorded in `results/phase2/`.
