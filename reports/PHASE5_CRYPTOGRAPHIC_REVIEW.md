# Phase 5 Cryptographic Review: Authenticated Checkpoint Platform

**Document:** `reports/PHASE5_CRYPTOGRAPHIC_REVIEW.md`  
**Date:** 2026-10-26  
**Auditor:** Principal Cryptography & Systems Infrastructure Engineer  
**Status:** `VERIFIED`

---

## 1. Executive Summary

This review provides a formal cryptographic audit of the AHCC Phase 5 Secure Checkpoint Platform. 

In Phase 5, the cryptographic architecture was expanded from single-model weight protection to a complete AI training platform incorporating client-side KMS envelope encryption, digital signature publisher provenance, monotonic freshness ledgers, and distributed inventory manifests.

### Formal Status Assessment
- **Production Checkpoint System:** `VERIFIED` (Production-Ready). Built strictly on NIST SP 800-38D AES-256-GCM, RFC 5869 HKDF-SHA256 domain-separated key derivation, RFC 8032 Ed25519 digital signatures, and AWS KMS / HSM client-side envelope encryption.
- **Experimental AHCC Cipher:** `REQUIRES INDEPENDENT REVIEW` (Quarantined). The custom algebraic power-map and Cauchy MDS cipher remains strictly isolated from all production checkpoint paths.

---

## 2. Key Derivation and Domain Separation Audit

### Multi-Key Hierarchy
Under Version 5 (`b"AHCCMOD5"`), key derivation utilizes RFC 5869 HKDF-SHA256 with 256-bit entropy salts and strict context strings:

$$\text{ManifestKey} = \text{HKDF-Expand}(\text{HKDF-Extract}(\text{salt}_{32}, \text{DEK}_{32}), \text{"AHCC-v5-Manifest-Key::"} \parallel \text{artifact\_id}_{16}, 32)$$

$$\text{ChunkKey} = \text{HKDF-Expand}(\text{HKDF-Extract}(\text{salt}_{32}, \text{DEK}_{32}), \text{"AHCC-v5-Chunk-Key::"} \parallel \text{artifact\_id}_{16}, 32)$$

### Cryptographic Independence
- **Independence:** Even if an adversary were to obtain the manifest key (e.g. through a side channel), knowledge of $\text{ManifestKey}$ yields zero information about $\text{ChunkKey}$ due to the pseudorandom function (PRF) security of HMAC-SHA256.
- **Salt Freshness:** A fresh 32-byte cryptographically secure random salt (`secrets.token_bytes(32)`) is generated for every checkpoint operation, ensuring no cross-artifact subkey collisions.

---

## 3. Nonce Policy and SP 800-38D Compliance

### Monotonic Nonce Construction
- **Manifest Nonce:** Fixed deterministic 96-bit nonce: `b"MANI\x00\x00\x00\x00\x00\x00\x00\x00"` under the dedicated $\text{ManifestKey}$. Exactly one manifest is encrypted under this key.
- **Chunk Nonces:** Monotonically constructed 96-bit nonces:
  $$\text{Nonce}_i = \text{b"CHNK"} \parallel \text{uint64\_be}(i), \quad i \in [0, \text{total\_chunks}-1]$$
  under the dedicated $\text{ChunkKey}$.

### Collision Probability
- By using a deterministic 64-bit monotonic sequence counter, the probability of a nonce collision under the same derived key is **identically zero** ($P(\text{collision}) = 0$) for up to $2^{64}-1$ chunks per artifact.
- The maximum chunk limit enforced by `ResourceLimits` is $1,000,000$ chunks ($< 2^{20}$), well below the NIST SP 800-38D invocation threshold of $2^{32}$.

---

## 4. Associated Authenticated Data (AAD) Cryptographic Binding

To prevent chunk reordering, splicing, or cross-model substitution attacks, each chunk is encrypted with structured AAD:

$$\text{AAD}_i = \text{artifact\_id}_{16} \parallel \text{b"::"} \parallel \text{manifest\_digest}_{32} \parallel \text{b"::tensor="} \parallel \text{name} \parallel \text{b"::seq="} \parallel \text{seq}_{64} \parallel \text{b"::idx="} \parallel \text{idx}_{32} \parallel \text{total}_{32} \parallel \text{b"::offset="} \parallel \text{offset}_{64} \parallel \text{b"::raw\_len="} \parallel \text{len}_{64}$$

This construction cryptographically binds:
1. The container artifact ID.
2. The manifest specification digest (and transitively all metadata and all other tensors).
3. The exact tensor name and its logical chunk sequence number.
4. The exact byte offset within the tensor buffer.

---

## 5. Provenance and Anti-Rollback Review

1. **Digital Signatures (Ed25519):** Applied over the SHA-256 digest of the canonical RFC 8785 manifest specification. Fails closed if the public key verification fails.
2. **Freshness Ledger:** Tracks monotonic high-water marks `(run_id, max_step, max_epoch)` with atomic staging and fsync, completely closing the model downgrade / rollback attack vector.
3. **Pickle Elimination:** All serialization paths use raw memory buffers and canonical JSON. Deserialization involves zero executable Python code.
