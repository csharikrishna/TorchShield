# Formal Algorithm Specification: Adaptive Hyperdimensional Chaotic Cryptography (AHCC)

**Specification Version:** 1.0-Formal  
**Target Modulus:** $p = 2^{32} - 5 = 4,294,967,291$ (primary), $p = 2^{31} - 19 = 2,147,483,579$ (secondary)  
**Classification:** Experimental Finite-Field Substitution-Permutation Network (SPN)  

---

## 1. Mathematical Domain and State Representation

AHCC operates on vectors of dimension $N$ over the Galois field $\mathbb{F}_p$.
- Modulus $p$ is a 32-bit prime satisfying $p \equiv 2 \pmod 3$ and $p \equiv 3 \pmod 4$.
- The state $V \in \mathbb{F}_p^N$ is an $N$-element column vector:
  $$V = [v_0, v_1, \dots, v_{N-1}]^T, \quad v_i \in \{0, 1, \dots, p-1\}$$
- For batch processing, state is represented as an $N \times B$ matrix $V \in \mathbb{F}_p^{N \times B}$, where $B$ is the batch count (number of vector blocks).

---

## 2. Parameter Constraints

- **Dimension $N$:** Positive integer $1 \le N \le 1024$. The condition $3N - 2 < p$ must hold so all Cauchy denominators are non-zero.
- **Rounds $R$:** Positive integer $1 \le R \le 64$ (default $R = 4$).
- **Modulus $p$:** Restricted to validated primes where $\gcd(3, p-1) = 1$.

---

## 3. Core Transformation Layers

### 3.1 Non-Linear Confusion Layer: Power-Map S-Box
For each state coordinate $x \in \mathbb{F}_p$:
$$S(x) = x^3 \pmod p$$
Because $\gcd(3, p-1) = 1$, the map $S: \mathbb{F}_p \to \mathbb{F}_p$ is a bijection.
The inverse map is defined by:
$$S^{-1}(y) = y^d \pmod p, \quad \text{where } d \equiv 3^{-1} \pmod{p-1}$$
For $p = 4,294,967,291$, $d = 2,863,311,527$.

### 3.2 Linear Diffusion Layer: Cauchy MDS Matrix
Diffusion is performed by matrix multiplication $M \cdot V \pmod p$, where $M \in \mathbb{F}_p^{N \times N}$ is a Cauchy matrix:
$$M_{i,j} = (x_i + y_j)^{-1} \pmod p$$
with parameter sequences:
$$x_i = i, \quad y_j = N + j, \quad \text{for } 0 \le i, j < N$$
Since $\{x_i\} \cap \{y_j\} = \emptyset$ and $x_i + y_j < 3N - 2 < p$, all elements are well-defined.
By the Cauchy determinant formula, all square submatrices of $M$ are non-singular.
Thus $M$ is a Maximum Distance Separable (MDS) matrix with branch number $\mathcal{B} = N + 1$.

The exact analytical inverse $M^{-1} \in \mathbb{F}_p^{N \times N}$ is given by:
$$(M^{-1})_{j,i} = \beta_j \cdot \alpha_i \cdot M_{i,j} \pmod p$$
where:
$$\alpha_i = \left( \prod_{k=0}^{N-1} (x_i + y_k) \right) \cdot \left( \prod_{k \ne i} (x_i - x_k) \right)^{-1} \pmod p$$
$$\beta_j = \left( \prod_{k=0}^{N-1} (x_k + y_j) \right) \cdot \left( \prod_{k \ne j} (y_j - y_k) \right)^{-1} \pmod p$$

### 3.3 Key Schedule Modes

#### Mode 1: Legacy Chaotic Polynomial Recurrence (Research Prototype)
Initialized from an integer seed $S$:
$$x_0 = ((S \pmod{p - 3}) + 3) \pmod p$$
Iterated using the quadratic map:
$$x_{m+1} = (a \cdot x_m^2 + b \cdot x_m + c) \pmod p$$
Constants:
$$a = 2,654,435,761 \pmod p, \quad b = 1,664,525 \pmod p, \quad c = 1,013,904,223 \pmod p$$
A warm-up of 64 iterations is executed, followed by $(R+1) \cdot N$ iterations generating the round key vectors $K_0, K_1, \dots, K_R \in \mathbb{F}_p^N$.

#### Mode 2: Modern HKDF-SHA256 Derivation (Default)
Input: 256-bit uniform master key $K \in \{0, 1\}^{32}$.
Salt: `b"AHCC-HKDF-SHA256-key-schedule-v1"`
Info string: `b"AHCC experimental round key;N=<N>;R=<R>;p=<p>;round=<r>;batch=<b>"`
Generates 32-bit words via HKDF-Expand with rejection sampling ($v < p$) to produce uniform elements in $\mathbb{F}_p$.

---

## 4. Encryption & Decryption Algorithms

### 4.1 Encryption
Input: Plaintext vector $V^{(0)} \in \mathbb{F}_p^N$, round keys $\{K_r\}_{r=0}^R$, matrix $M$.
For $r = 0, 1, \dots, R-1$:
1. **Pre-whitening / Key Addition:**
   $$W^{(r)} = (V^{(r)} + K_r) \pmod p$$
2. **S-Box Substitution:**
   $$U^{(r)} = S(W^{(r)}) = (W^{(r)})^3 \pmod p$$
3. **MDS Diffusion:**
   $$Z^{(r)} = (M \cdot U^{(r)}) \pmod p$$
4. **Post-diffusion Key Addition:**
   $$V^{(r+1)} = (Z^{(r)} + K_{r+1}) \pmod p$$
Output: Ciphertext vector $C = V^{(R)}$.

### 4.2 Decryption
Input: Ciphertext vector $C \in \mathbb{F}_p^N$, round keys $\{K_r\}_{r=0}^R$, inverse matrix $M^{-1}$.
Initialize $V^{(R)} = C$.
For $r = R-1, R-2, \dots, 0$:
1. **Subtract Post-diffusion Key:**
   $$Z^{(r)} = (V^{(r+1)} - K_{r+1}) \pmod p$$
2. **Inverse MDS Diffusion:**
   $$U^{(r)} = (M^{-1} \cdot Z^{(r)}) \pmod p$$
3. **Inverse S-Box Substitution:**
   $$W^{(r)} = S^{-1}(U^{(r)}) = (U^{(r)})^d \pmod p$$
4. **Subtract Pre-whitening Key:**
   $$V^{(r)} = (W^{(r)} - K_r) \pmod p$$
Output: Recovered Plaintext $V^{(0)}$.
