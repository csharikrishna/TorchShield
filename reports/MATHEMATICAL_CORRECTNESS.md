# Mathematical Correctness & Algorithmic Audit of AHCC

**Date:** 2026-10-26  
**Auditor:** Senior Cryptography & Systems Research Engineer  
**Scope:** Line-by-line verification of the 19-page research paper *Verifying the Novelty of Adaptive Hyperdimensional Chaotic Cryptography for GPU-Native AI Security* against `ahcc/engine.py`.

---

## 1. Specification Traceability Matrix

| Claim / Equation from Paper | Source File & Function | Required Mathematical Property | Test / Verification Method | Actual Result | Status |
|---|---|---|---|---|---|
| Finite field bedrock $\mathbb{F}_p$, $p = 2^{32} - 5 = 4294967291$ (Page 1) | `ahcc/engine.py:31`, `SUPPORTED_PRIMES` | Primality, closure of $+,-,\times, \div$ over $\mathbb{F}_p$ | Arbitrary-precision Python math boundary checks (`test_modular_arithmetic_boundaries_against_arbitrary_precision`) | Prime verified; field operations closed; int64 intermediate overflow prevented via 16-bit limb splitting. | **VERIFIED** |
| Power map S-box $S(x) = x^3 \pmod p$ (Page 2) | `ahcc/engine.py:306`, `sbox_forward` | Permutation over $\mathbb{F}_p \iff \gcd(3, p-1) = 1$ | Extended Euclidean check: $\gcd(3, 4294967290)=1$, inverse exponent $d=2863311527$. | Analytically proven. Exhaustive on small fields, boundary and random testing on $p=2^{32}-5$. | **VERIFIED** |
| Power map inverse $S^{-1}(y) = y^d \pmod p$ (Page 2) | `ahcc/engine.py:310`, `sbox_inverse` | $S^{-1}(S(x)) \equiv x \pmod p$ for all $x \in \mathbb{F}_p$ | Square-and-multiply modular exponentiation verified against Python `pow(x, d, p)`. | Matches independent Python reference; round trip exact. | **VERIFIED** |
| Cauchy MDS Diffusion Matrix $M_{i,j} = (x_i + y_j)^{-1} \pmod p$ (Page 3) | `ahcc/engine.py:148`, `_build_cauchy_mds_matrices` | $x_i, y_j$ distinct, $x_i + y_j \not\equiv 0 \pmod p$; $M$ is MDS and invertible | Tested $M \cdot M^{-1} \equiv I \pmod p$ for $N \in \{4, 8, 16, 32, 64, 128\}$. | Analytical inverse formula verified exact for all tested dimensions. | **VERIFIED** |
| Round Function $v_{r+1} = (M \cdot v_{\text{temp}} + k_r) \pmod p$ (Page 3) | `ahcc/engine.py:388`, `encrypt` / `decrypt` | Decryption inverts encryption identically | Independent pure-Python reference implementation tested across multiple $N, R, p$. | Round trip exact on CPU and CUDA. | **VERIFIED** |
| Chaotic Key Schedule from Seed (Page 3) | `ahcc/engine.py:236`, `_generate_chaotic_round_keys` | 128/256-bit entropy resistance to key recovery | Cycle length, state space analysis: state is reduced to single field element in $[0, p-1]$. | **Effective key entropy capped at $\le 32$ bits**. Seed expansion is trivial 1D quadratic recurrence. | **FAILED** (Security Failure) |
| Tensor-Aware GPU Alignment & Tensor Core acceleration (Pages 2, 13) | `ahcc/engine.py:356`, `_int64_mm` | Accelerated modular arithmetic on GPU Tensor Cores | Hardware architecture inspection: Turing/Ampere Tensor Cores do NOT support modulo arithmetic over prime fields. PyTorch fails with `addmm_cuda not implemented for Long`. | Dense $O(N^2)$ matrix-vector math must be simulated via sliced broadcast products; Tensor Cores cannot be used. | **FAILED** (Architecture Hypothesis Refuted) |

---

## 2. Detailed Mathematical Analysis

### 2.1 Finite-Field Arithmetic & Overflow Boundaries
- Modulus $p = 2^{32}-5 = 4,294,967,291$.
- In computing $a \cdot b \pmod p$, if $a, b \approx p-1$, the intermediate product is:
  $$(p-1)^2 = (2^{32}-6)^2 \approx 1.84467 \times 10^{19} > 2^{63}-1 \approx 9.22337 \times 10^{18}$$
  In standard signed 64-bit integer arithmetic (`torch.int64`), $(p-1)^2$ causes signed integer overflow.
- **Engine Implementation:** `safe_mod_mul` decomposes operand $b$ into two 16-bit limbs:
  $$b = (b_{\text{hi}} \ll 16) + b_{\text{lo}}$$
  Each sub-product $a \cdot b_{\text{lo}}$ and $a \cdot b_{\text{hi}}$ is bounded by $(2^{32}-6) \times (2^{16}-1) < 2^{48}$, which fits safely in signed 64-bit integers.
- **Matrix-Vector Multiplication:** `safe_mod_matmul` decomposes $V$ into three radix-$2^{11}$ limbs ($V_0, V_1, V_2$). Since $N \le 1024$, the dot product accumulation is at most $1024 \times (2^{32}-6) \times (2^{11}-1) < 2^{53} < 2^{63}-1$.
- **Verification:** Verified against arbitrary-precision Python `int` across boundary values $\{0, 1, 2, p-3, p-2, p-1, (p-1)//2\}$ and 50,000 randomized inputs. Zero discrepancies found.

### 2.2 S-Box Properties: Algebraic Degree vs. Differential Uniformity
- **Differential Uniformity:** For any nonzero difference $a \in \mathbb{F}_p^*$:
  $$(x+a)^3 - x^3 = 3ax^2 + 3a^2x + a^3 = b$$
  Since $\text{char}(\mathbb{F}_p) = p > 3$, the coefficient $3a \ne 0$. This is a degree-2 polynomial in $x$, which has at most 2 roots in $\mathbb{F}_p$.
  Therefore, the differential uniformity is $\delta \le 2$ (almost perfect nonlinear, APN).
- **Algebraic Vulnerability (Low Degree):**
  While differential uniformity is low, the algebraic degree of $S(x)$ is only 3.
  In an $R=4$ round SPN, because linear diffusion layers do not increase degree, the total algebraic degree of the cipher output as a polynomial in the plaintext and key variables is at most:
  $$\deg = 3^4 = 81$$
  Over a large finite field $\mathbb{F}_p$ ($p \approx 2^{32}$), a polynomial of degree 81 requires only 82 known plaintext-ciphertext pairs to reconstruct via Lagrange interpolation or univariate polynomial solving!
  In modern algebraic cryptosystems that use power maps (e.g., MiMC, Poseidon, Hades), reaching security against algebraic attacks requires **at least 40 to 100 rounds**.
  AHCC's choice of $R=4$ rounds is **catastrophically vulnerable to algebraic polynomial interpolation attacks**.

### 2.3 Cauchy MDS Diffusion & Computational Complexity
- The Cauchy matrix construction:
  $$M_{i,j} = \frac{1}{i + (N + j)} \pmod p, \quad 0 \le i, j < N$$
  Since all $x_i = i \in [0, N-1]$ and $y_j = N + j \in [N, 2N-1]$ are mutually disjoint and non-zero, every submatrix has a non-zero determinant. Hence $M$ is an MDS matrix with optimal branch number $\mathcal{B} = N + 1$.
- **Complexity Reality Check:**
  AES MixColumns operates on 4-byte columns using a $4 \times 4$ MDS matrix over $\mathbb{F}_{2^8}$ ($O(1)$ operations per byte).
  AHCC uses a dense $N \times N$ matrix over $\mathbb{F}_p$. For an input vector of $N$ elements:
  - Multiplication by $M$ requires $N^2$ modular multiplications per round.
  - Across $R$ rounds, work scales as $O(R \cdot N^2)$ per $N$ elements, or **$O(R \cdot N)$ operations per word**.
  - For $N=64, R=4$, each 32-bit element requires 256 modular multiplications.
  - For $N=1024, R=4$, each element requires 4,096 modular multiplications!
  - This fundamental $O(N)$ computational blowup explains why AHCC is inherently unable to compete with block ciphers like AES-GCM for high-throughput streaming.
