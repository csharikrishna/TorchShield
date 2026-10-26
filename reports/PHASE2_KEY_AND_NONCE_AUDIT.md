# Phase 2 Key Management & Nonce Policy Audit

**Date:** 2026-10-26  
**Auditor:** Senior Applied Cryptographer & Security Systems Engineer  
**Status:** Complete & Verified  

---

## 1. Executive Summary

A cryptographic audit was conducted on AES-256-GCM key management and chunk nonce generation across `ahcc.secure` and `ahcc.checkpoint`. 

### Key Findings
1. **Previous Vulnerability (v1):** Checkpoint v1 generated independent random 96-bit nonces for each chunk under a static key. Under NIST SP 800-38D, random nonces carry birthday-bound collision probabilities ($2^{-32}$ collision probability after $2^{32}$ nonces). More critically, resuming an interrupted write or re-encrypting an existing model without per-artifact key isolation created severe nonce reuse risks.
2. **Remediation (v2 Hardened Architecture):**
   - **Per-Artifact HKDF Subkey Isolation:** Each new checkpoint generation samples a fresh 128-bit cryptographically secure random salt ($S \leftarrow \text{secrets.token\_bytes}(16)$) and a unique UUIDv4 artifact ID. An isolated 256-bit AES-GCM subkey is derived using RFC 5869 HKDF-SHA256 with domain-separated info `AHCC-SecureCheckpoint-v2-Key:<artifact_id>`.
   - **Strict Deterministic 96-bit Nonces:** Under the per-artifact subkey, nonces are strictly derived from the monotonic chunk sequence counter:
     $$\text{nonce}(\text{chunk\_seq}) = \text{struct.pack}(">Q", \text{chunk\_seq}) + \text{struct.pack}(">I", 0)$$
   - **Manifest Nonce Reservation:** Nonce `0x000000000000000000000000` ($\text{chunk\_seq} = 0$) is strictly reserved for the authenticated manifest envelope.
   - **Payload Chunk Nonces:** Payload chunks monotonically increment starting at $\text{chunk\_seq} = 1, 2, \dots$.
   - **Collision Assumption:** Nonce collisions are mathematically impossible within a checkpoint because chunk sequence counters are strictly monotonic. Across independent checkpoints, nonce collisions are completely mitigated because each checkpoint operates under a fresh HKDF-derived artifact key.

---

## 2. NIST SP 800-38D Compliance Evaluation

| Requirement | Implementation Detail | Compliance Status |
|---|---|---|
| **Key Length** | Strictly 256-bit (32 bytes) enforced; invalid lengths rejected with `ValueError`. | **COMPLIANT** |
| **Nonce Uniqueness** | Deterministic monotonic counter per-artifact derived key; nonce 0 reserved for manifest. | **COMPLIANT** |
| **Tag Size** | Full 128-bit (16 bytes) authentication tag enforced. Truncated tags rejected. | **COMPLIANT** |
| **Fail-Closed Behavior** | Any tag mismatch, ciphertext bit-flip, or AAD alteration raises `AuthenticationError` / `CheckpointError`. | **COMPLIANT** |
| **No Plaintext Release** | In-place or streamed tensors are deserialized only after chunk tag authentication succeeds. | **COMPLIANT** |
| **Associated Data (AAD)** | Binds file artifact ID, manifest digest, layer name, chunk sequence, chunk index, offset, and shape. | **COMPLIANT** |

---

## 3. Key Lifecycle and Rotation

- **Key Identifiers:** Checkpoint format v2 includes a dedicated `key_id` metadata field in the unencrypted container header and authenticated manifest. This allows enterprise key management systems (e.g. AWS KMS, HashiCorp Vault) to identify which master key is required without exposing key material.
- **Key Rotation:** A model checkpoint can be safely rotated to a new key by loading and re-saving under the new master key. Because fresh salts and artifact IDs are sampled, the new file uses fresh nonces and subkeys with zero risk of nonce collision with the old file.

---

## 4. Test Verification
All key management and nonce policy invariants are verified under automated pytest suites:
- `tests/test_phase2_aead_security.py` (5 tests passed)
- `tests/test_phase2_nonce_policy.py` (3 tests passed)
