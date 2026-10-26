# CUDA and GPU Compatibility Validation Report

**Date:** 2026-10-26  
**Auditor:** Senior CUDA/GPU Performance Engineer  
**Hardware Tested:** NVIDIA GeForce GTX 1650 (Compute Capability 7.5, 4,096 MiB VRAM)  
**Driver & Tooling:** NVIDIA Display Driver 617.42, CUDA UMD 13.4, PyTorch CUDA build 12.1 (`torch.version.cuda = 12.1`)  

---

## 1. Hardware & Execution Environment

- **Device Name:** NVIDIA GeForce GTX 1650 (Turing TU117 architecture)
- **Compute Capability:** 7.5 (sm_75)
- **VRAM Total:** 4,095.7 MiB (4.00 GB)
- **Streaming Multiprocessors (SMs):** 14
- **Device Residency:** Verified. Output tensors from GPU `encrypt` and `decrypt` reside strictly on `cuda:0` without implicit host copies.

---

## 2. Correctness & Parity Matrix Results

Execution command:
```bash
python -m pytest tests/test_cuda_correctness.py -v
python -m benchmarks.run_gpu_validation
```
**Outcome:** 56 tests passed, 0 failures.

### Verified Behaviors
1. **CPU/GPU Bit-Exact Equivalence:** Tested across 54 combinations of:
   - Moduli $p \in \{2^{31}-19, 2^{32}-5\}$
   - Vector dimensions $N \in \{4, 16, 64\}$
   - Round counts $R \in \{1, 2, 4\}$
   - Batch sizes $B \in \{1, 13, 64\}$
   - High-boundary inputs $\{0, 1, p-1, (p-1)//2\}$
   All outputs matched between CPU and CUDA bit-for-bit.
2. **Non-Contiguous & Strided Tensors:**
   Transposed tensors ($(B, N)$ vs $(N, B)$) and sliced tensors (`[:, ::2]`) correctly handled without corrupted indexing.
3. **Stream Determinism:**
   10 consecutive synchronized CUDA kernel launches produced identical output. Zero non-deterministic race conditions observed.

---

## 3. Critical GPU Architecture Findings & Bottlenecks

1. **Absence of Int64 GEMM on CUDA:**
   - PyTorch's CUDA cuBLAS backend does not support signed 64-bit matrix multiplication (`"addmm_cuda" not implemented for 'Long'`).
   - AHCC's modular matrix multiplication must therefore execute via outer-product expansion and reduction:
     `out[:, b_start:b_end] = (A.unsqueeze(2) * B[:, b_start:b_end].unsqueeze(0)).sum(dim=1)`
   - This creates massive memory traffic and tile stalls.
2. **Refutation of Tensor Core Acceleration Claim:**
   - The research paper hypothesizes that AHCC's matrix diffusion will exploit GPU Tensor Cores.
   - **Reality:** Tensor Cores perform floating-point (FP16, BF16, TF32) and small integer (INT8, INT4) matrix multiply-accumulate with standard binary wraparound. They **cannot** perform exact finite-field arithmetic modulo $p = 2^{32}-5$.
   - Any attempt to use Tensor Cores would introduce rounding errors or modular truncation, destroying the cipher's exact algebraic invertibility.
3. **Throughput Ceiling:**
   - Because AHCC requires exact limb-based modular operations and dense $O(N^2)$ matrix multiplications, its CUDA throughput is bounded at $\approx 63\text{--}65\text{ MB/s}$ on this hardware.
