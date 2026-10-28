# TorchShield Distributed Checkpoints & Rollback Protection Report

**Audit Date:** 2026-10-28  
**Scope:** Distributed Checkpoints (`torchshield/distributed.py`) & Freshness Ledgers (`torchshield/provenance.py`)  

---

## 1. Distributed Sharded Architecture

For large distributed training jobs (e.g. FSDP, DeepSpeed ZeRO-3, Megatron-LM), each rank saves an independent authenticated checkpoint shard:

$$\text{Shard Path} = \text{rank\_}X\text{\_of\_}Y\text{.shield}$$

Once all ranks complete local checkpointing, Rank 0 assembles and cryptographically signs a `global_manifest.json` cataloging:
- World size and rank index bounds.
- Cryptographic SHA-256 digests and file sizes for every rank shard.
- Monotonic run generation counter and step number.
- Ed25519 digital signature binding the entire distributed state.

---

## 2. Distributed Manifest Security Hardening

In response to the Phase 5 and Phase 7 audit requirements, `torchshield/distributed.py` was hardened against cluster-level supply-chain attacks:

1. **Path Traversal Rejection:** Any shard record containing path navigation sequences (`../`, `..\`) or directory delimiters (`/`, `\`) is rejected immediately.
2. **Symlink Attack Mitigation:** The loader invokes `os.path.islink()` on both `global_manifest.json` and every individual shard file. Symbolic links are strictly rejected.
3. **Rank Geometry & Uniqueness Invariants:** The manifest is verified to contain **exactly one shard per declared rank** in the continuous sequence $0 \dots (\text{world\_size} - 1)$. Duplicate ranks or missing ranks fail closed.
4. **Strict JSON Parsing:** Manifest parsing utilizes `_strict_json_loads()` to detect and reject duplicate keys or malformed structures.

Verified in `tests/distributed/test_distributed_hardening.py` (3 tests pass).

---

## 3. Anti-Rollback Freshness Ledgers

Rollback attacks involve an adversary replacing a newer, valid checkpoint (Step 10,000) with an older, valid checkpoint (Step 1,000) from the same training run. Because both checkpoints possess valid cryptographic tags and publisher signatures, authenticated encryption alone cannot detect the regression.

TorchShield enforces monotonic freshness via a tiered ledger architecture:

### 3.1 Local SQLite & File Ledgers
- **Mechanism:** Stores monotonic generation records (`(run_id, max_step, max_epoch, artifact_id, version)`) in an ACID SQLite database or atomic local file.
- **Defensive Boundary:** Protects against local accidental reversion and process rollback.
- **Operational Limitation:** Cannot prevent full VM/disk snapshot restorations where the disk containing both the checkpoint and SQLite file is reverted simultaneously.

### 3.2 Quorum-Backed Etcd Freshness Backend (`EtcdFreshnessBackend`)
- **Mechanism:** Leverages an external consensus cluster (etcd) with transactional Compare-And-Swap (CAS) primitives and monotonic cluster revision numbers.
- **Defensive Boundary:** Guarantees atomic monotonic progression even if individual worker nodes crash, restart, or have their local disks restored from snapshot images.
- **Fail-Closed Policy:** If the external etcd cluster is unreachable or partitioned, the engine fails closed with `FreshnessServiceUnavailableError` rather than silently degrading to weaker local checks.

Verified in `tests/test_etcd_freshness_backend.py` (3 tests pass).
