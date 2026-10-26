# Rigorous Reassessment of the AHCC Cipher & Cryptanalysis

**Date:** 2026-10-26  
**Author:** Senior Applied Cryptographer & Security Evaluator  
**Status:** Approved Technical Report  

---

## 1. Executive Summary & Objective

In Phase 1, an initial audit evaluated the custom Adaptive Hyperdimensional Chaotic Cryptography (AHCC) cipher. While the audit correctly highlighted several fundamental structural flaws, one specific cryptanalytic assertion was technically flawed:
> *"With $R=4$, the global algebraic degree is $\le 81$, which is completely breakable with fewer than 100 known plaintext-ciphertext queries."*

This reassessment corrects that claim with a rigorous mathematical derivation, separates empirically proven vulnerabilities from theoretical research hypotheses, and details the structural security limits of the AHCC construction.

**Conclusion:** AHCC remains **insecure for production use and must remain strictly research-only**. However, this conclusion rests on proven entropy bottlenecks, lack of authentication, deterministic ECB leakage, and algebraic invertibility of the key schedule—not on an invalid univariate interpolation claim.

---

## 2. Mathematical Correction: Univariate vs. Multivariate Algebraic Degree

### 2.1 The Erroneous Claim
The Phase 1 audit observed that the S-box $S(x) = x^3 \pmod p$ has degree 3. Over $R=4$ rounds, composing degree-3 operations yields an algebraic degree bound of $3^4 = 81$. From this, the report asserted that the cipher could be recovered using fewer than 100 known plaintext-ciphertext pairs via polynomial interpolation.

### 2.2 Why the Claim is Mathematically Invalid
Polynomial interpolation of a **univariate** polynomial of degree $d$ requires $d + 1$ evaluations. For $d = 81$, 82 pairs would indeed suffice to recover the coefficients of a single-variable polynomial $P(x) = \sum_{i=0}^{81} c_i x^i$.

However, AHCC is an **$N$-dimensional Substitution-Permutation Network (SPN)** over $\mathbb{F}_p^N$:
1. **Round 0:** Input vector $\mathbf{x} = (x_0, x_1, \dots, x_{N-1}) \in \mathbb{F}_p^N$ is combined with round key $\mathbf{k}^{(0)}$:
   $$u_i^{(0)} = x_i + k_i^{(0)} \pmod p$$
2. **S-Box Layer:**
   $$v_i^{(0)} = S(u_i^{(0)}) = (x_i + k_i^{(0)})^3 \pmod p$$
   Each $v_i^{(0)}$ is a univariate cubic polynomial in $x_i$.
3. **Cauchy MDS Diffusion Layer:**
   $$z_i^{(1)} = \sum_{j=0}^{N-1} M_{i,j} v_j^{(0)} \pmod p$$
   Because the Cauchy matrix $M$ is dense and MDS, **every coordinate $z_i^{(1)}$ is now a multivariate polynomial in all $N$ variables** $(x_0, \dots, x_{N-1})$ of degree 3.
4. **Round 1:** The next S-box layer computes:
   $$v_i^{(1)} = (z_i^{(1)} + k_i^{(1)})^3 \pmod p$$
   Cubing a sum of $N$ cubic terms produces a multivariate polynomial of degree $3 \times 3 = 9$.
   The number of possible monomials of degree $\le 9$ in $N=64$ variables is:
   $$\binom{N + d}{d} = \binom{64 + 9}{9} = \frac{73!}{9! \, 64!} \approx 5.92 \times 10^{10} \text{ monomials}$$
5. **Round 3 & 4:** After $R=4$ rounds, the algebraic degree of each coordinate polynomial is at most $3^4 = 81$. The number of multivariate monomials of degree $\le 81$ in $N=64$ variables is:
   $$\binom{64 + 81}{81} = \binom{145}{81} \approx 1.25 \times 10^{41} \text{ monomials}$$

### 2.3 Complexity of Multivariate Solving
To recover the cipher via algebraic interpolation or linearization, an attacker must solve for the unknown coefficients or round keys:
- 100 known plaintext-ciphertext pairs provide only $100 \times N = 6,400$ linear equations.
- The number of unknown monomial coefficients is $\approx 1.25 \times 10^{41}$.
- The linear system is underdetermined by a factor of $> 10^{37}$.
- Gröbner basis algorithms (e.g., F4, F5) or XL (eXtended Linearization) algorithms require degree of regularity calculations; at degree 81 with $N=64$, the matrix sizes exceed all feasible physical memory and computation limits.

**Distinction:**
- *Proven Fact:* The algebraic degree of AHCC ($3^4 = 81$) is very low compared to standard symmetric ciphers (e.g., AES has algebraic degree 254 over $\mathbb{GF}(2^8)$). In arithmetization-oriented ciphers over large prime fields (such as MiMC, Poseidon, Rescue), low degree necessitates dozens of rounds (e.g., MiMC uses $\lceil \log_3(p) \rceil \approx 65$ rounds for 32-bit primes). AHCC's $R=4$ rounds provide an inadequate algebraic security margin against higher-order differential and Gröbner-basis attacks.
- *Corrected Fallacy:* AHCC **cannot** be broken with $<100$ pairs via trivial interpolation. The attack model requires multivariate polynomial system solving, which remains an open research question for $N=64, R=4$.

---

## 3. Proven Vulnerabilities of AHCC

While the univariate interpolation claim was erroneous, the following vulnerabilities are **empirically proven and fatal** for production security:

### 3.1 Legacy Key Schedule: Entropy Collapse to $2^{32}$
In legacy mode (`master_seed`):
$$x_0 = ((S \pmod{p - 3}) + 3) \pmod p$$
Any 128-bit, 256-bit, or arbitrary-length seed $S$ is reduced modulo $p - 3$. For $p = 4,294,967,291$, there are at most $4,294,967,288$ distinct starting states ($\approx 32$ bits of entropy).
- **GPU Brute Force:** Modern GPUs can evaluate $>2 \times 10^9$ key trials per second. Exhausting $2^{32}$ states takes $<2$ seconds.
- **Equivalent Keys:** Any two seeds $S_1, S_2$ satisfying $S_1 \equiv S_2 \pmod{p-3}$ generate bit-for-bit identical keystreams.

### 3.2 Key Schedule Inversion in $O(1)$ Arithmetic Steps
The chaotic recurrence:
$$x_{m+1} = a x_m^2 + b x_m + c \pmod p$$
is a degree-2 polynomial over $\mathbb{F}_p$. Given any leaked or known round-key element $x_{m+1}$:
$$\Delta = b^2 - 4a(c - x_{m+1}) \pmod p$$
$$x_m = (-b \pm \sqrt{\Delta}) \cdot (2a)^{-1} \pmod p$$
Because $p \equiv 3 \pmod 4$, $\sqrt{\Delta} \equiv \Delta^{(p+1)/4} \pmod p$ is computed in $O(\log p)$ time ($\approx 32$ multiplications). This permits bidirectional state reconstruction.

### 3.3 Zero Native Message Authentication (Malleability)
AHCC encrypts data as raw $\mathbb{F}_p$ elements without an authentication tag or MAC:
- An attacker can modify arbitrary ciphertext elements.
- Decryption succeeds without error, releasing corrupted tensors into the host application or AI model.
- INT-CTXT and INT-PTXT are completely violated.

### 3.4 Deterministic ECB-like Permutation
Without an initialization vector (IV) or nonce:
- Identical plaintext blocks $P_1 = P_2$ encrypt to identical ciphertext blocks $C_1 = C_2$.
- In AI weight matrices, zero-padded regions, repeated bias values, and structured embeddings leak exact patterns directly through the ciphertext.

### 3.5 Computational Bottleneck of Dense Cauchy Diffusion
The MDS diffusion matrix requires $N^2$ multiplications per round:
- For $N=64$, each round requires 4,096 field multiplications.
- Standard GPU Tensor Cores accelerate FP16/BF16/INT8 matrix multiplications, but lack native hardware instructions for 64-bit integer modular arithmetic ($(a \times b) \pmod p$).
- As measured, AHCC CUDA throughput is capped at $\sim 11 - 80$ MiB/s, compared to $> 2,000$ MiB/s for CPU AES-256-GCM using hardware AES-NI instructions.

---

## 4. Remediation and Policy Directive

1. **Experimental Status:** The `ahcc.engine` module is designated strictly **EXPERIMENTAL AND RESEARCH-ONLY**.
2. **Explicit Warnings:** Instantiating `AHCC_Engine` or calling `TensorCipherWrapper` emits an explicit `InsecureCipherWarning` and logs a clear security notice.
3. **Fail-Safe Checkpoint Engine:** The production checkpoint API (`SecureModelCheckpoint`) uses AES-256-GCM exclusively and is structurally incapable of falling back to raw AHCC.
