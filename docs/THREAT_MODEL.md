# AHCC Secure AI Checkpoint Platform: Threat Model

**Document:** `docs/THREAT_MODEL.md`  
**Date:** 2026-10-26  
**Scope:** AI Model Checkpoints, Training State, KMS Integration, Provenance, and Distributed Clusters  
**Status:** `VERIFIED`

---

## 1. System Context and Trust Boundaries

The AHCC Secure Checkpoint Platform operates across three primary security perimeters:

```
+---------------------------------------------------------------------------------------------------+
|                                       SECURITY PERIMETERS                                         |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ TRUSTED LOCAL COMPUTATION ]                [ UNTRUSTED TRANSIT & STORAGE ]                     |
|  +-------------------------------------+      +-----------------------------------------+         |
|  | GPU Workers / Training Nodes        |      | Cloud Object Stores (S3, GCS, Azure)    |         |
|  | - PyTorch runtime                   |      | Shared NFS / Lustre Cluster Storage     |         |
|  | - GPU VRAM / Host RAM               |      | Network File Transfer / Model Hubs      |         |
|  | - Decrypted weight tensors          |      +-----------------------------------------+         |
|  | - Transient DEKs (wiped on exit)    |                                                          |
|  +-------------------------------------+                                                          |
|                    ^                                                                              |
|                    | TLS 1.3 / mTLS                                                               |
|                    v                                                                              |
|  [ TRUSTED KEY INFRASTRUCTURE ]                                                                   |
|  +-------------------------------------+                                                          |
|  | Hardware-Backed Cloud KMS (HSM)     |                                                          |
|  | - Master KEK (Never leaves HSM)     |                                                          |
|  | - IAM & Role-Based Access Control   |                                                          |
|  +-------------------------------------+                                                          |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Threat Actors & Attacker Capabilities

| Threat Actor | Capabilities | Objectives |
|---|---|---|
| **Storage Attacker / Untrusted Cloud** | Full control over persistent storage (S3/disk); can read, modify, delete, reorder, or inject files. | Exfiltrate proprietary model weights; inject malicious weight backdoors (e.g. Sleeper Agents). |
| **Network Adversary (Man-in-the-Middle)** | Intercept, modify, replay, or inject traffic between training nodes and storage. | Modify optimizer states or weights in transit to sabotage training runs. |
| **Malicious Training Contributor** | Can provide checkpoints for fine-tuning or evaluation. | Execute arbitrary code on evaluator machines via deserialization exploits (e.g., pickle RCE). |
| **Rogue Rank / Byzantine Worker** | Corrupted or compromised node in a multi-node distributed cluster. | Submit invalid or mismatched rank shards to desynchronize the distributed training run. |
| **Rollback Adversary** | Can replace current checkpoint with a validly signed earlier checkpoint. | Revert training progress or rollback a safety-aligned model to an unaligned/vulnerable version. |

---

## 3. Threat Mitigation Matrix

| Threat | Attack Vector | AHCC Platform Mitigation |
|---|---|---|
| **Remote Code Execution (RCE)** | Crafting malicious Python pickle payloads in model weights or optimizer states. | **Strictly Pickle-Free Architecture:** All weights are dense numerical buffers; all metadata is canonical JSON; zero deserialization hooks exist. |
| **Weight Exfiltration** | Snooping on checkpoint files stored in public or compromised object storage. | **NIST SP 800-38D AES-256-GCM:** 256-bit symmetric encryption with fresh per-artifact DEK derived via RFC 5869 HKDF-SHA256. |
| **Silent Weight Poisoning / Backdoors** | Bit-flipping weights or biases to create backdoor triggers without invalidating format. | **Authenticated Encryption (AEAD):** Every 4 MiB chunk possesses a 128-bit Poly1305/GMAC authentication tag. Any tampering triggers immediate `CheckpointError`. |
| **Chunk Reordering / Truncation** | Reordering chunk sequence, deleting tail chunks, or swapping chunks between layers. | **Monotonic Nonce & Cryptographic AAD Binding:** Nonce contains `b"CHNK" + seq_64BE`; AAD commits to `artifact_id`, `manifest_digest`, `tensor_name`, `seq`, `offset`, and `raw_len`. |
| **Publisher Impersonation / Supply Chain Attack** | Distributing untrusted checkpoints signed by unauthorized parties. | **Ed25519 Publisher Provenance:** Canonical manifest specification is cryptographically signed; public keys are enforced at load time. |
| **Rollback / Downgrade Attack** | Presenting an older, valid checkpoint to revert safety alignment or training steps. | **Monotonic Freshness Ledger:** `FreshnessLedger` tracks `(run_id, max_step, max_epoch)`; rejects any checkpoint with step < current ledger step. |
| **Partial / Incomplete Distributed Restore** | Restoring a sharded checkpoint where rank 3 is from step 100 and rank 0 is from step 200. | **Global Inventory Manifest:** Cryptographically binds all `world_size` shard SHA-256 hashes and step numbers into a single signed document. |
| **Key Compromise via Static Storage** | Storing long-term encryption keys on disk in plaintext. | **KMS Envelope Encryption:** KEKs reside inside hardware HSMs; per-artifact DEKs are wrapped with authenticated encryption contexts. |

---

## 4. Quarantined Components & Out-of-Scope Items

1. **Experimental AHCC Cipher:** The custom algebraic S-box and Cauchy diffusion cipher is strictly quarantined as research code and **must never be used in production**.
2. **Physical GPU Memory Snooping:** Physical cold-boot attacks against host DRAM or GPU VRAM are out of scope (requires confidential computing / AMD SEV-SNP / Intel TDX).
