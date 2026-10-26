# Phase 2 Authenticated Checkpoint Wire Format Specification

**Date:** 2026-10-26  
**Status:** Approved Specification (Version 2 Hardened)  

---

## 1. Container Wire Layout

A Phase 2 AHCC Secure Model Checkpoint file (`.ahcc`) consists of:
1. **Container Header** (Variable length, 63 + key_id bytes)
2. **Encrypted Manifest Envelope**
3. **Encrypted Chunk Envelopes**

```
+-----------------------------------------------------------------------------------------------+
| CONTAINER HEADER                                                                              |
| - Magic:            8 bytes   (ASCII "AHCCMOD2")                                              |
| - Version:          1 byte    (0x02)                                                          |
| - Salt:            16 bytes   (Cryptographically random per-artifact salt)                    |
| - Artifact ID Len:  1 byte    (0x24 = 36 bytes)                                               |
| - Artifact ID:     36 bytes   (UUIDv4 ASCII string)                                           |
| - Key ID Len:       1 byte    (0x00 to 0xFF)                                                  |
| - Key ID:           N bytes   (Optional key identifier ASCII string)                          |
| - Manifest Len:     4 bytes   (Big-endian uint32, length of encrypted manifest envelope)      |
+-----------------------------------------------------------------------------------------------+
| ENCRYPTED MANIFEST ENVELOPE (AESGCMEnvelope with Nonce = 0x00 * 12)                           |
| - Magic:            8 bytes   (ASCII "AHCCSEAL")                                              |
| - Version:          1 byte    (0x01)                                                          |
| - Nonce:           12 bytes   (0x000000000000000000000000)                                    |
| - Ciphertext:       Variable  (AES-256-GCM encrypted canonical JSON manifest)                  |
| - Auth Tag:        16 bytes   (128-bit authentication tag)                                    |
+-----------------------------------------------------------------------------------------------+
| ENCRYPTED CHUNK ENVELOPES (Chunk Seq = 1, 2, ..., total_chunks)                               |
| Each chunk is an independent AESGCMEnvelope:                                                  |
| - Magic:            8 bytes   (ASCII "AHCCSEAL")                                              |
| - Version:          1 byte    (0x01)                                                          |
| - Nonce:           12 bytes   (Big-endian uint64 chunk_seq || uint32 0)                       |
| - Ciphertext:       Variable  (AES-256-GCM encrypted raw tensor slice)                        |
| - Auth Tag:        16 bytes   (128-bit authentication tag)                                    |
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Cryptographic Bindings & Associated Data (AAD)

To guarantee INT-CTXT (Ciphertext Integrity) and prevent cross-checkpoint chunk splicing, layer reordering, or chunk injection:

### 2.1 Manifest Envelope AAD
$$\text{AAD}_{\text{manifest}} = \text{artifact\_id} \parallel \text{"::MANIFEST::v2"}$$

### 2.2 Payload Chunk AAD
Every chunk envelope cryptographically authenticates:
$$\text{AAD}_{\text{chunk}} = \text{artifact\_id} \parallel \text{"::"} \parallel \text{manifest\_digest} \parallel \text{"::layer="} \parallel \text{name} \parallel \text{"::chunk\_seq="} \parallel [i]_8 \parallel \text{"::chunk\_idx="} \parallel [j]_4 \parallel [M]_4 \parallel \text{"::offset="} \parallel [\text{off}]_8 \parallel \text{"::shape="} \parallel \text{canonical\_shape\_json}$$
where:
- `manifest_digest`: SHA-256 digest of the pre-manifest JSON structure.
- `[i]_8`: 8-byte big-endian global chunk sequence number ($1, 2, \dots$).
- `[j]_4`: 4-byte big-endian chunk index within the current layer ($0, \dots, M-1$).
- `[M]_4`: 4-byte big-endian total chunk count for this layer.
- `[\text{off}]_8`: 8-byte big-endian byte offset of the slice in the uncompressed tensor buffer.

---

## 3. Atomic File Operations & Crash Resilience

To prevent corrupt or truncated checkpoint files during power loss or disk exhaustion:
1. All writes stage to `<output_path>.tmp.<uuid>`.
2. The complete header, manifest, and all chunks are written.
3. Explicit `file.flush()` followed by `os.fsync(file.fileno())` flushes OS disk buffers.
4. File is closed.
5. `os.replace(staging_path, output_path)` performs an atomic destination replacement.
6. If an exception occurs at any point, the temporary staging file is deleted immediately, leaving the original destination file untouched.

---

## 4. Replay and Freshness Limitations

> [!IMPORTANT]
> **Integrity vs. Freshness Boundary:** AES-256-GCM provides absolute cryptographic confidentiality and integrity against unauthorized tampering. However, standard symmetric cryptography cannot prevent an adversary who has local disk write access from replacing `model_v2.ahcc` with an older, previously valid `model_v1.ahcc` (Rollback Attack). If rollback prevention is required, the host environment must enforce freshness using an external monotonic counter, trusted hardware ledger, or cloud metadata store.
