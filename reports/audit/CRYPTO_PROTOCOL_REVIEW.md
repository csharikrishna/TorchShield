# TorchShield Cryptographic Protocol & Container Format Review

**Audit Date:** 2026-10-28  
**Scope:** Cryptographic Core (`torchshield/checkpoint.py`, `torchshield/secure.py`)  

---

## 1. Cryptographic Primitives & Standards Compliance

TorchShield relies exclusively on standardized, high-assurance cryptographic primitives from the OpenSSL/hazmat engine via Python `cryptography`:

- **Bulk Encryption:** AES-256-GCM (NIST SP 800-38D) and ChaCha20-Poly1305 (RFC 8439) with 128-bit authentication tags.
- **Key Derivation:** HKDF-SHA256 (RFC 5869) with 32-byte public header salt and domain-separated `info` labels.
- **Digital Signatures:** Ed25519 (RFC 8032) asymmetric digital signatures over canonical JSON manifests.
- **Envelope Wrapping:** Standard AWS/GCP/Azure/Vault KMS 256-bit AES-GCM wrapping.

---

## 2. Key Derivation & Domain Separation

For every encrypted checkpoint, subkeys are derived from the master secret (IKM) and a fresh 32-byte header salt using HKDF-SHA256:

$$\text{PRK} = \text{HKDF-Extract}(\text{Salt}, \text{IKM})$$

$$\text{ChunkKey} = \text{HKDF-Expand}(\text{PRK}, \text{"ts-chunk-key-v5"}, 32)$$

$$\text{ManifestKey} = \text{HKDF-Expand}(\text{PRK}, \text{"ts-manifest-key-v5"}, 32)$$

- **Domain Separation Guarantee:** Chunk payloads and container manifests never share the same derived cryptographic key, preventing cross-domain ciphertext substitution.
- **Salt Freshness:** A new cryptographically random 32-byte salt (`secrets.token_bytes(32)`) is generated for every save operation, guaranteeing independent key spaces even if the same master key is reused.

---

## 3. Nonce Construction & Collision Invariance

The AEAD nonce space is constructed as a 12-byte deterministic prefix:

$$\text{Nonce}_{\text{chunk}}(i) = \text{b"CHNK"} \;\|\; \text{uint64\_be}(i)$$

$$\text{Nonce}_{\text{manifest}}(j) = \text{b"MANI"} \;\|\; \text{uint64\_be}(j)$$

- **Collision Probability:** Within a single encryption session, monotonic counter incrementation guarantees **zero collision probability ($P=0$)**.
- **Cross-Session Collisions:** Because each session derives distinct ephemeral subkeys via a fresh 32-byte random salt, cross-session key collision probability is bounded at $P \approx 2^{-128}$.
- **Elimination of Birthday Attacks:** Random 96-bit nonces suffer from birthday-paradox collision risks after $\approx 2^{32}$ chunks. Deterministic counter nonces completely eliminate this vulnerability.

---

## 4. Container v5 Binary Wire Format Layout

```
+-----------------------------------------------------------------------------------------+
| OFFSET | FIELD NAME       | TYPE / SIZE | DESCRIPTION                                   |
+-----------------------------------------------------------------------------------------+
| 00..07 | Magic Header     | Bytes[8]    | ASCII "TSHD\x00CK5"                            |
| 08..08 | Version          | Uint8 (1B)  | Format version 0x05                           |
| 09..40 | Public Salt      | Bytes[32]   | Cryptographic HKDF-SHA256 salt                |
| 41..44 | Artifact ID Len  | Uint32 (4B) | Big-endian length of artifact UUID string     |
| 45..X  | Artifact ID      | String[N]   | UTF-8 canonical artifact UUID                 |
| X..Y   | Key ID Length    | Uint32 (4B) | Big-endian length of KMS Key ID / ARN         |
| Y..Z   | Key ID String    | String[M]   | UTF-8 Key ID / ARN                            |
| Z..A   | Wrapped DEK Len  | Uint32 (4B) | Big-endian length of wrapped DEK blob         |
| A..B   | Wrapped DEK Blob | Bytes[W]    | KMS-wrapped 32-byte DEK ciphertext            |
| B..C   | Chunk Count      | Uint64 (8B) | Total number of encrypted chunk records       |
| C..End | Chunk Stream     | Variable    | Sequence of authenticated chunk envelopes     |
| End-M  | Manifest Envelope| Variable    | Authenticated canonical JSON + Ed25519 sig    |
+-----------------------------------------------------------------------------------------+
```

---

## 5. Known Answer Tests (KAT) & Tamper Verification

- **NIST Test Vectors:** Verified in `tests/test_aead_known_answer.py` (5 tests pass).
- **Tampering Invariants:** Injected bit flips across container headers, salt, nonces, chunk ciphertext, and authentication tags in `tests/test_aead_tampering.py` (4 tests pass) confirm that the engine fails closed instantly with `CheckpointError`.
