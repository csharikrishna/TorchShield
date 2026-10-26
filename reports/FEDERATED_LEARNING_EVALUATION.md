# Federated Learning & Secure Gradient Aggregation Audit

**Date:** 2026-10-26  
**Auditor:** Senior Distributed Systems & Privacy Researcher  
**Scope:** Evaluation of AHCC Claims for Federated Learning and Privacy-Preserving Machine Learning.

---

## 1. Distinction of Security & Privacy Primitives

The AHCC proposal claims that AHCC provides "mathematically verifiable confidentiality against data reconstruction attacks" in federated learning.
In cryptographic systems research, it is vital to distinguish between fundamentally different primitives:

| Mechanism | Trust Model | What the Aggregation Server Sees | Does it Protect Against Malicious Aggregators? |
|---|---|---|---|
| **In-Transit Symmetric Encryption (AHCC / TLS / AES-GCM)** | Aggregator is trusted; communication channel is untrusted. | Plaintext individual client gradients (after decrypting on the server). | **NO.** Server decrypts each client's gradient individually; server can perform gradient inversion. |
| **Homomorphic Encryption (FHE / Paillier / CKKS)** | Server is untrusted. | Ciphertexts only. Server adds ciphertexts directly without decrypting. Decryption occurs only under collective client key. | **YES.** Server never sees individual gradients. High compute overhead. |
| **Secure Multi-Party Aggregation (Bonawitz et al. / SecAgg)** | Server is untrusted; clients pair-wise mask updates with secret keys. | Aggregated sum only; individual masks cancel out algebraically ($\sum s_{i,j} = 0$). | **YES.** Information-theoretically secure against honest-but-curious server. |
| **Differential Privacy (DP-SGD)** | Server can be trusted or semi-trusted. | Gradients perturbed with calibrated Gaussian/Laplacian noise. | **YES.** Statistical bound on privacy leakage ($\epsilon, \delta$), at the expense of model utility. |

---

## 2. Technical Evaluation of AHCC in Federated Learning

1. **Non-Homomorphic Property:**
   - AHCC round function is:
     $$v_{r+1} = M \cdot S(v_r + k_r) + k_{r+1} \pmod p$$
   - The S-box $S(x) = x^3 \pmod p$ is non-linear:
     $$(x + y)^3 = x^3 + 3x^2y + 3xy^2 + y^3 \ne x^3 + y^3 \pmod p$$
   - Therefore, summing two AHCC ciphertexts $C_A + C_B \pmod p$ does NOT equal the encryption of $(V_A + V_B)$.
   - An untrusted server **cannot** perform gradient aggregation in the ciphertext domain.
2. **Server Must Decrypt Plaintexts:**
   - In `demos/ai_security_use_cases.py:89-95`:
     ```python
     # The server decrypts each payload before aggregating plaintext gradients.
     for idx, (ct_grad, shape_meta, q_meta) in enumerate(encrypted_payloads):
         rec_q_grad = wrapper.decrypt_tensor(ct_grad, shape_meta)
         rec_fp32 = dequantize_fp_to_fp32(rec_q_grad.cpu(), q_meta)
         decrypted_grads.append(rec_fp32)
     ```
   - The server possesses the decryption key and decrypts every client's gradient to plaintext in memory.
   - Any malicious or curious server can immediately apply Deep Leakage from Gradients (DLG, Zhu et al.) or Inverting Gradients (Geiping et al.) to fully reconstruct the client's private training images or text!
3. **Communication & Quantization Overhead:**
   - Floating-point gradients must be quantized to $\mathbb{F}_p$, expanding 32-bit floats into 64-bit tensors in PyTorch, doubling transmission bandwidth compared to float16 or int8 quantization.
   - Dequantization introduces quantization error ($\approx 4.7 \times 10^{-7}$).
   - Standard TLS 1.3 / AES-GCM already provides in-transit confidentiality and authentication for gradient transport with hardware acceleration, without requiring custom finite-field quantization.

---

## 3. Conclusion on Federated Learning

AHCC contributes **NO privacy-preserving secure aggregation capability**.
Describing AHCC as securing federated learning against gradient reconstruction is **technically inaccurate and misleading**:
- If the threat is network eavesdropping, standard TLS / AES-GCM is faster, authenticated, and standard-compliant.
- If the threat is a curious or compromised aggregation server, AHCC provides **zero protection** because the server decrypts all client gradients before aggregation.
