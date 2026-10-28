# TorchShield Formal Threat Model Reassessment

**Audit Date:** 2026-10-28  
**Scope:** Architecture & Protocol Threat Boundaries (v0.7.0)  

---

## 1. Adversary Profiles & Trust Boundaries

The TorchShield security model assumes storage backends (NFS, S3, cloud scratch disks, shared Ceph volumes) and communication channels are untrusted. Adversaries may attempt to inspect, modify, truncate, substitute, or replay model checkpoints.

```
+-----------------------------------------------------------------------------------------+
|                                    TRUST BOUNDARY MAP                                   |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|   [ UNTRUSTED ZONE ]                                          [ TRUSTED ZONE ]          |
|   * Cloud Object Storage (S3 / GCS / Azure Blob)              * Model Execution RAM     |
|   * Shared Cluster Filesystems (NFS, Lustre, Ceph)            * CPU / Host Process      |
|   * Network Interconnect & Middleboxes                        * Dedicated GPU VRAM      |
|   * Public Hubs / Untrusted Registries                        * Secure Key Management   |
|   * Snapshot Recovery / Backup Systems                          (AWS/GCP/Azure/Vault)   |
|                                                                                         |
|          |                                                          ^                   |
|          | Checkpoint Storage (.shield)                             | Decrypted Tensors |
|          +----------------------------------------------------------+                   |
|                                        |                                                |
|                             [ CRYPTOGRAPHIC ENGINE ]                                    |
|                             * AES-256-GCM / Poly1305                                    |
|                             * Monotonic Counter Nonces (P=0)                            |
|                             * Ed25519 Publisher Signatures                              |
|                             * Two-Phase Strict Validation                               |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Attacker Capabilities & Defensive Guarantees

| Attack Vector | Attacker Capability | TorchShield Defense | Residual Risk / Assumption |
|---|---|---|---|
| **Arbitrary Code Execution (Pickle)** | Supply malicious `__reduce__` exploit in checkpoint | **Zero-Pickle Deserialization:** Weights serialized as dense numeric buffers; metadata strictly allowlisted JSON. | None in verified paths. Unsafe `torch.load` is completely eliminated. |
| **Weight Tampering & Backdooring** | Alter weights or inject backdoor trigger weights in storage | **Authenticated Streaming AEAD:** Poly1305 / AES-256-GCM authentication tags fail closed before tensor allocation. | Requires secure key management (attacker cannot access KMS DEK/KEK). |
| **Manifest / Header Modification** | Alter tensor shapes, dtypes, or metadata | **Associated Authenticated Data (AAD):** Container header and canonical JSON manifest are cryptographically bound. | Attacker cannot tamper with header without triggering `AuthenticationError`. |
| **Nonce Collision (Key Replay)** | Force birthday-bound nonce collisions | **Monotonic Counter Nonces:** Nonces formatted as `b"CHNK" + uint64_be` guarantee strictly zero ($P=0$) collision risk. | Key must not be reused across independent sessions with reset counters. |
| **Directory Traversal in Shards** | Inject `../../etc/shadow` in distributed manifest | **Strict Path Sanitization:** Rejects path separators, directory traversal sequences, and symlinks. | Directory containing shards must have standard POSIX permissions. |
| **Rollback / Replay Attacks** | Replace latest checkpoint with an earlier valid checkpoint | **Monotonic Freshness Ledger:** Verifies run generation counter; external etcd backend provides quorum-backed CAS checks. | Local SQLite ledger cannot prevent full VM snapshot restorations; requires external etcd or cloud ledger. |
| **Identity Impersonation** | Substitute model with weights from an unauthorized trainer | **Ed25519 Publisher Signatures:** Canonical JSON manifest is signed with private key; verified against trusted public key. | Caller must configure `expected_signer_public_key`. |

---

## 3. Operational Boundaries & Non-Goals

1. **Host Memory Snoopers (Without Confidential Computing):** An attacker with `root` or `CAP_SYS_PTRACE` on the host operating system who can read process memory can read decrypted weights in RAM. Full mitigation requires Hardware Enclaves (NVIDIA H100 Confidential Computing / AMD SEV-SNP).
2. **Side-Channel Attacks on Memory Buses:** Physical interposers attached to PCIe or HBM3 buses are outside the software threat boundary.
3. **Hypervisor Snapshot Restorations:** If an adversary rolls back an entire virtual machine snapshot, restoring both the checkpoint and local SQLite disk files simultaneously, a local ledger cannot detect the reversion. Production deployments requiring rollback immunity must use `EtcdFreshnessBackend` or an out-of-band external store.
