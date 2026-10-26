# Standard AEAD Validation Report: AES-256-GCM & HKDF-SHA256

**Date:** 2026-10-26  
**Auditor:** Senior Cryptography Engineer  
**Reference Standards:** NIST SP 800-38D (GCM Mode), RFC 5869 (HKDF)  
**Implementation Modules:** `ahcc/secure.py` (`AESGCMEnvelope`, `TensorAESGCMEnvelope`)  

---

## 1. Compliance and Architecture Overview

To provide a secure, trustworthy encryption baseline independent of the experimental AHCC cipher, the codebase integrates standard AES-256-GCM and HKDF-SHA256:

1. **Primitive Specification:**
   - AES-256 in Galois/Counter Mode (GCM).
   - 256-bit symmetric keys.
   - 96-bit (12-byte) initialization vectors / nonces, generated fresh using OS CSPRNG (`secrets.token_bytes`).
   - 128-bit (16-byte) GHASH authentication tag.
2. **Envelope Layout:**
   $$\text{Magic (8 B: } \texttt{b"AHCCSEAL"}\text{)} \parallel \text{Version (1 B: } \texttt{0x01}\text{)} \parallel \text{Nonce (12 B)} \parallel \text{Ciphertext (N B)} \parallel \text{Tag (16 B)}$$
3. **Associated Authenticated Data (AAD):**
   The fixed envelope header ($\texttt{Magic} \parallel \texttt{Version}$) and caller-supplied AAD (e.g., layer IDs, tensor shapes, chunk indices) are bound into the GHASH authentication computation:
   $$\text{AAD}_{\text{authenticated}} = \text{Header} \parallel \text{Nonce} \parallel \text{AAD}_{\text{caller}}$$
4. **Fail-Closed Semantics:**
   No plaintext is released prior to full verification of the 128-bit authentication tag. Any modification of ciphertext, nonce, tag, or AAD raises `AuthenticationError`.

---

## 2. Test Verification Matrix

| Test Suite | Specification / Reference | Test Count | Result | Status |
|---|---|---|---|---|
| RFC 5869 HKDF-SHA256 KAT | RFC 5869 Appendix A (Test Cases 1 & 2) | 2 | Matched official PRK & OKM byte-for-byte | **VERIFIED** |
| NIST SP 800-38D AES-GCM KAT | NIST SP 800-38D Appendix C (Cases 13, 14, 16) | 3 | Matched official ciphertext & authentication tags exactly | **VERIFIED** |
| Full-Payload Tampering Matrix | Bit flips across Header, Nonce, Ciphertext, Tag | 50+ | 100% caught; zero unauthenticated release | **VERIFIED** |
| Associated Data Tampering | Modified AAD, swapped model metadata, omitted AAD | 3 | 100% caught | **VERIFIED** |
| Truncation Matrix | Truncated envelopes at every single byte offset | All lengths | 100% caught | **VERIFIED** |
| Nonce Freshness Check | 500 successive encryptions under same key | 500 trials | Zero duplicate nonces | **VERIFIED** |

All tests passed with zero failures in `tests/test_aead_known_answer.py` and `tests/test_aead_tampering.py`.
