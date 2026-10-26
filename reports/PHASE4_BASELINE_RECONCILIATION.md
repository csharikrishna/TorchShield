# Phase 4 Baseline Reconciliation & Discrepancy Audit

**Audit Date:** 2026-10-26  
**Auditor:** Senior Applied Cryptographer & Systems Security Auditor  
**Baseline Git Revision:** `5a1bfcd` (`v0.4.0-phase3-final`)  
**Target Environment:** Windows 11 Pro, Python 3.12.10, PyTorch 2.5.1+cu121, NVIDIA GeForce GTX 1650 (CC 7.5, 4 GB VRAM)  

---

## 1. Executive Summary

This reconciliation report independently inspects and audits the baseline established at the close of Phase 3 (`v0.4.0-phase3-final`). Every claim made in the Phase 3 reports was cross-examined against:
1. The executable repository source code in `ahcc/`.
2. The installed wheel (`ahcc_gpu_engine-0.4.0-py3-none-any.whl`) and source archive (`dist/ahcc-secure-checkpoint-phase3.zip`).
3. The cryptographic digests recorded in `dist/SHA256SUMS`.
4. The test execution logs and raw benchmark records in `results/phase3/`.

### Baseline Verification Matrix

| Verification Target | Claimed State (Phase 3 Report) | Actual Verified State | Reconciliation Status |
| :--- | :--- | :--- | :---: |
| **Git Revision & Tree** | Clean working tree on `master` at `5a1bfcd` | Clean working tree; commit matches `5a1bfcd` | **VERIFIED** |
| **Release Archive Hash** | `5df3efdc...223c3bb` (`ahcc-secure-checkpoint-phase3.zip`) | Matches exactly: `5df3efdc9999a8bd8e1eb73208a8487edf13ef3628404b5ae1b432f27223c3bb` | **VERIFIED** |
| **Archive Source Equivalence** | Distributed ZIP matches repository source | Extracted ZIP critical files match repository byte-for-byte | **VERIFIED** |
| **Unit Test Suite** | 162 passing tests in 10.90s | 162 passing tests in 12.74s (0 failures, 167 deprecation/insecure warnings) | **VERIFIED** |
| **Fuzzing Suite** | 200+ property mutations rejected fail-closed | 200+ mutations pass cleanly via `tests/fuzz_phase2_envelope.py` in 6.2s | **VERIFIED** |
| **Nonce Construction Claim** | Report claimed: `BLAKE2b-96(salt \|\| tensor_name \|\| chunk_idx_64BE)` | Actual code used: `struct.pack(">Q", chunk_seq) + struct.pack(">I", 0)` | **DISCREPANCY (IDENTIFIED)** |
| **Collision Impossibility Claim** | Report claimed mathematical impossibility of collision | Truncated hashes have non-zero collision probability; counter uniqueness requires strict key domain separation | **DISCREPANCY (IDENTIFIED)** |
| **Key Domain Separation** | Single `artifact_key` used for both manifest and chunks | Code derived only one `artifact_key`; manifest and chunks shared key under sequence separation | **DISCREPANCY (IDENTIFIED)** |
| **Manifest Authentication Binding** | Report claimed manifest digest cryptographically binds chunks | Code computed `pre_manifest` digest, but loader never recomputed/verified `pre_manifest` | **DISCREPANCY (IDENTIFIED)** |
| **Resource Limits & DoS** | `MAX_DIM_SIZE = 2**31 - 1` | Allowed theoretically valid but practically absurd single dimensions up to 2 billion elements | **DISCREPANCY (IDENTIFIED)** |

---

## 2. In-Depth Discrepancy Analysis

### Discrepancy 1: Nonce Derivation Claim vs. Actual Code Implementation
* **The Claim:** The Phase 3 report described chunk nonces as:
  $$\text{Nonce}_{\text{chunk}} = \text{BLAKE2b-96}(\text{salt} \parallel \text{tensor\_name} \parallel \text{chunk\_idx}_{64\text{BE}})$$
  and asserted that this construction mathematically guarantees collision resistance.
* **The Reality in Code:** In `ahcc/checkpoint.py` (lines 88–96):
  ```python
  @staticmethod
  def derive_chunk_nonce(chunk_seq: int) -> bytes:
      if chunk_seq < 0 or chunk_seq >= 2**64:
          raise ValueError(f"Invalid chunk sequence number: {chunk_seq}")
      return struct.pack(">Q", chunk_seq) + struct.pack(">I", 0)
  ```
  The code actually used a deterministic 64-bit integer counter padded with 4 zero bytes!
* **Cryptographic Assessment:**
  1. If BLAKE2b-96 *were* used, truncated hash outputs to 96 bits have a collision probability dictated by the birthday paradox: after $2^{48}$ invocations, collision probability approaches 50%. Claiming a truncated hash provides "mathematical impossibility of collisions" violates fundamental probability theory.
  2. While the actual counter code (`struct.pack(">Q", chunk_seq) + struct.pack(">I", 0)`) is structurally superior to a truncated hash, both manifest and payload chunks were encrypted under the *same* `artifact_key`. Manifest used sequence 0, while chunks used sequence $1, 2, \dots$.
* **Required Remediation (Phase 4):**
  - Implement strict HKDF domain separation for `manifest_key` vs `chunk_key`.
  - Retain the deterministic counter construction with documented 96-bit prefix encoding (`b"CHNK" + struct.pack(">Q", chunk_seq)`).
  - Explicitly document that fresh salts and UUIDs provide probabilistic uniqueness ($1 - \epsilon$), while internal counters guarantee deterministic uniqueness under that specific derived key.

---

### Discrepancy 2: Manifest Authentication Circularity and Verification Gap
* **The Reality in Code:**
  In Phase 3 `save_checkpoint`:
  1. A preliminary manifest dictionary `pre_manifest_dict` was created with layer shapes and dtypes.
  2. `manifest_digest = hashlib.sha256(pre_manifest_bytes).digest()` was computed.
  3. Chunks were encrypted with AAD containing `manifest_digest`.
  4. The final manifest included chunk ciphertext lengths and hashes, plus `manifest_digest.hex()`.
  5. The final manifest was encrypted with `artifact_key`.
  In Phase 3 `load_checkpoint`:
  1. The final manifest was decrypted and authenticated.
  2. `manifest_digest = bytes.fromhex(manifest["manifest_digest"])` was extracted.
  3. Chunks were verified against this `manifest_digest`.
* **The Defect:**
  The loader **never reconstructed `pre_manifest_dict`** to verify that `sha256(reconstructed_pre_manifest)` equaled `manifest_digest`! 
  Because the final manifest was authenticated by AES-GCM, an external attacker could not tamper with `manifest_digest`. However, having a dangling unverified intermediate hash introduces architectural ambiguity and makes formal security proofs unnecessarily fragile.
* **Required Remediation (Phase 4):**
  - Define a canonical Manifest Specification ($\mathcal{M}_{\text{spec}}$).
  - Explicitly verify during load that $\text{SHA-256}(\text{canonical}(\mathcal{M}_{\text{spec}})) == \text{manifest\_digest}$.
  - Ensure zero circular dependencies exist between chunk encryption and manifest serialization.

---

### Discrepancy 3: Denial-of-Service Bounds and Allocation Limits
* **The Reality in Code:**
  `MAX_DIM_SIZE = 2**31 - 1` (2,147,483,647 elements).
  While multiplication overflow was checked via `expected_numel *= dim`, setting an individual dimension limit of 2 billion elements is completely impractical for real AI accelerators with 4 GB – 80 GB VRAM. A malicious or corrupted shape such as `(2000000000,)` of `float32` would declare an 8 GB tensor, which would cause an unhandled `MemoryError` or system thrashing before reading ciphertext.
* **Required Remediation (Phase 4):**
  - Introduce configurable `ResourceLimits` dataclass.
  - Set default maximum dimension to $100,000,000$ elements.
  - Set default single-tensor allocation ceiling to $16\text{ GiB}$.
  - Validate declared byte length against `numel * itemsize` and against actual file slice lengths *before* allocating `bytearray`.

---

## 3. Discrepancy Summary Table

| Item | Phase 3 Claim | Baseline Truth | Phase 4 Action |
| :--- | :--- | :--- | :--- |
| **Nonce Algorithm** | BLAKE2b-96 truncated hash | Counter `>Q` + `>I` in code | Standardize counter `b"CHNK" + >Q` |
| **Key Derivation** | Single key for all operations | Single key for all operations | Split into `manifest_key` & `chunk_key` |
| **Manifest Hash** | Binds chunks to manifest | Unverified intermediate digest | Verify canonical $\mathcal{M}_{\text{spec}}$ on load |
| **Allocation Bounds** | $2^{31}-1$ per dimension | Unrealistic for physical RAM | Cap at 100M dim, configurable limits |
| **Crash Consistency** | Power-loss durability claimed | Atomic file replace only | Document OS/fsync durability limits |

This baseline reconciliation confirms that while the core AES-GCM encryption is mathematically sound and fast, the protocol requires the specific cryptographic and architectural corrections detailed above.
