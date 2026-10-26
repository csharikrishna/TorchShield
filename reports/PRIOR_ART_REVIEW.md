# Prior-Art and Research-Novelty Audit

**Date:** 2026-10-26  
**Auditor:** Senior Research Cryptographer & Systems Auditor  

---

## 1. Prior-Art Analysis Matrix

| Work / Citation | Venue & Date | Core Technique / Architecture | Security Properties & Claims | What the Work Already Solves | What AHCC Contributes Beyond It | Evaluation & Gap Analysis |
|---|---|---|---|---|---|---|
| **Alvarez & Li** (*Some Basic Cryptographic Requirements for Chaos-Based Cryptosystems*) | Int. J. Bifurcation and Chaos (2006) | Survey of chaos-based ciphers and recurring pitfalls | Formalized 10 rules for chaotic ciphers (key space, degradation, diffusion) | Explains why finite-precision chaos collapses into short cycles and why key sensitivity is not security. | AHCC uses finite field $\mathbb{F}_p$ instead of float chaos to avoid precision loss. | AHCC avoids floating-point degradation, but its 1D quadratic key recurrence still collapses into $\le 32$ bits of entropy. |
| **MiMC: Efficient Cryptographic Primitives for ZK & MPC** (Albrecht et al.) | ASIACRYPT 2016 | $S(x) = x^3$ over $\mathbb{F}_p$ or $\mathbb{F}_{2^n}$ with round constants | Proved resistance to algebraic and differential attacks with $R \ge 65$ rounds | Proves how power maps over large prime fields can form secure permutations. | AHCC claims $R=4$ rounds with Cauchy diffusion is sufficient. | **AHCC claim fails.** MiMC requires 65+ rounds because $x^3$ degree grows slowly ($3^R$). At $R=4$, AHCC degree is only 81, making it breakable by polynomial interpolation. |
| **Poseidon: A New Hash Function for Zero-Knowledge** (Grassi et al.) | USENIX Security 2021 | Hades design strategy, power map $x^5$, Cauchy-like MDS matrices over $\mathbb{F}_p$ | IND-CPA/CR, resistant to algebraic, differential, Gröbner basis attacks | Established MDS Cauchy matrices over prime fields and partial S-box rounds. | AHCC proposes dense Cauchy MDS for GPU data encryption. | Poseidon uses Cauchy MDS over $\mathbb{F}_p$ for hashing/SNARKs where $O(N^2)$ work is acceptable for small $N \le 16$. In bulk tensor encryption, $O(N^2)$ is fatal. |
| **CryptoTensors / Safetensors AES-GCM** | Open-Source / HuggingFace (2022-2024) | AES-256-GCM authenticated encryption for safetensors headers and buffers | IND-CCA2, INT-CTXT authenticated encryption, zero-copy deserialization | Fast, standard-compliant model checkpoint encryption at rest and in transit. | AHCC proposed tensor-native finite field encryption. | CryptoTensors with AES-GCM operates at 5–10 GB/s on modern CPUs with AES-NI. AHCC runs at ~63 MB/s on GPU and lacks authentication. |
| **GPU-Accelerated AES-GCM** (Nishikawa et al., GPUSec, cuCrypto) | IEEE / ACM (2018-2023) | Bitsliced and CTR-mode AES-GCM on NVIDIA CUDA | Hardware-level AES-GCM achieving 40–100 GB/s on NVIDIA Turing/Ampere | High-throughput bulk data encryption exploiting CUDA thread parallelism. | AHCC claims AES-GCM is unsuitable for GPUs due to sequential GMAC. | **Refuted.** While GMAC polynomial multiplication has sequential dependency across blocks, parallel tree-reduction algorithms in GF($2^{128}$) achieve $>50$ GB/s on GPUs. AHCC's claim that AES cannot be parallelized on GPUs is incorrect. |
| **Inverting Gradients in Federated Learning** (Geiping et al., Zhu et al. DLG) | NeurIPS 2020 / USENIX Security 2023 | Gradient reconstruction optimization attacks | Shows that accessing unencrypted or decrypted gradients allows exact training data recovery | Established why intermediate gradient protection requires secure aggregation or DP, not server decryption. | AHCC claimed gradient obfuscation. | AHCC decrypts gradients at the server prior to aggregation, leaving gradients completely vulnerable to DLG and Geiping attacks. |

---

## 2. Research Novelty Assessment

### Is the core AHCC cipher novel?
- **No.**
- Power-map S-boxes ($x^3 \pmod p$) combined with round key addition and Cauchy MDS diffusion over prime fields $\mathbb{F}_p$ have been thoroughly investigated in the literature since at least 2016 (MiMC, Hades, Poseidon, GMiMC).
- The idea of using discretized chaotic maps for key schedules has been repeatedly proposed and repeatedly broken over the past 25 years.
- The specific parameterization of AHCC ($R=4$ rounds, dense Cauchy matrix up to $N=1024$) is not a breakthrough; rather, it combines the worst of both worlds:
  1. Low round count ($R=4$) leaves the cipher vulnerable to low-degree polynomial interpolation.
  2. Dense matrix diffusion ($O(N^2)$) makes the cipher 100x slower than standard AES-GCM.
