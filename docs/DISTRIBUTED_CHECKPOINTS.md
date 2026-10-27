# Secure Distributed and Sharded Checkpoint Architecture

**Document:** `docs/DISTRIBUTED_CHECKPOINTS.md`  
**Date:** 2026-10-26  
**Implementation:** `ahcc.distributed`  
**Status:** `VERIFIED`

---

## 1. Challenges in Distributed Checkpointing

In modern distributed training (Fully Sharded Data Parallel / FSDP, DeepSpeed ZeRO-3, Megatron-LM):
- Each rank $r \in [0, W-1]$ holds only a shard of the model weights and optimizer moments.
- Saving to a single monolithic file requires gathering all weights to Rank 0, which exhausts GPU memory and causes severe cluster stalls.
- However, naive distributed saving (each rank saving independently) creates major security vulnerabilities:
  1. **Partial Restores:** If Rank 2 crashes during save, the checkpoint directory contains an incomplete set of shards. Loading 3 out of 4 shards will crash or silently desynchronize the cluster.
  2. **Skewed Steps:** Rank 0 might save step 10,001 while Rank 1 saves step 10,000.
  3. **Shard Tampering:** An attacker can replace a single worker's shard with malicious weights.

---

## 2. The Two-Phase Sharded Checkpoint Protocol

AHCC implements a coordinated, two-phase distributed checkpointing protocol:

```
[ PHASE 1: PARALLEL WORKER SAVES ]
Rank 0 ------------> shard_00000_of_00004.ahcc (AES-256-GCM encrypted)
Rank 1 ------------> shard_00001_of_00004.ahcc (AES-256-GCM encrypted)
Rank 2 ------------> shard_00002_of_00004.ahcc (AES-256-GCM encrypted)
Rank 3 ------------> shard_00003_of_00004.ahcc (AES-256-GCM encrypted)
                            |
                     (Cluster Barrier)
                            v
[ PHASE 2: COORDINATOR GLOBAL INVENTORY SEAL ]
Coordinator / Rank 0:
  1. Inspects all W shards.
  2. Calculates individual SHA-256 digests and file sizes.
  3. Compiles global inventory manifest:
     {
       "run_id": "llama3-run-01",
       "global_step": 10000,
       "world_size": 4,
       "shards": [ {rank, filename, sha256, size}... ]
     }
  4. Signs canonical inventory digest with Ed25519 publisher key.
  5. Atomically writes 'global_manifest.json'.
```

---

## 3. Strict Verification Invariants at Load Time

When `load_sharded_checkpoint` is called on any rank:
1. **Inventory Completeness:** Confirms that all $W$ shards declared in `global_manifest.json` exist.
2. **Digest Verification:** Recomputes the SHA-256 hash of each shard on disk; fails closed on any bit modification.
3. **World Size Match:** Ensures the executing cluster's `world_size` exactly matches the checkpoint's `world_size`.
4. **Signature Verification:** Verifies the Ed25519 digital signature over the entire inventory.
5. **Anti-Rollback:** Consults `FreshnessLedger` to prevent loading an older iteration.

---

## 4. Architectural Boundaries and Limitations

1. **Custom Multi-Rank Sharding:** This module implements custom sharded file management with cryptographic inventory sealing. It is **NOT** an implementation of PyTorch's native `torch.distributed.checkpoint` (DCP) plugin interface.
2. **No Dynamic Resharding:** A checkpoint saved with world size $W$ must be loaded with world size $W$. Loading a 4-shard checkpoint onto 2 ranks or 8 ranks is strictly rejected.
3. **Shared Storage Dependency:** Multi-node clusters require shards to be written to a shared distributed filesystem (NFS, Lustre, GPFS, CephFS, or cloud-mounted object storage).
