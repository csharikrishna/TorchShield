# Monotonic Anti-Rollback Freshness Ledger

**Document:** `docs/ROLLBACK_PROTECTION.md`  
**Date:** 2026-10-26  
**Implementation:** `ahcc.provenance.FreshnessLedger`  
**Status:** `VERIFIED`

---

## 1. The Rollback Vulnerability in Machine Learning

While digital signatures and AEAD encryption guarantee authenticity and integrity, they do not inherently prevent **replay and rollback attacks**:
- **Scenario:** A safety-critical model is fine-tuned to remove dangerous capabilities or backdoors. At Step 50,000, it is aligned and secure. An adversary with storage write access replaces the Step 50,000 checkpoint with a validly signed Step 1,000 checkpoint from the same training run.
- **Consequence:** The model loads successfully, passes AEAD decryption and signature verification, but silently runs an older, vulnerable, or unaligned version of the model.

---

## 2. Monotonic Freshness Architecture

AHCC implements the `FreshnessLedger` to enforce monotonic step progression per model lineage / training run:

```
[ Incoming Checkpoint ] 
  (run_id="run-01", step=12000)
             |
             v
+--------------------------+
|  FreshnessLedger.verify  | <--- Reads Current State from Ledger
+--------------------------+      (run-01: max_step=15000)
             |
             +---> 12000 < 15000 ? ---> YES ---> RAISE RollbackError (Attack Blocked!)
             |
             +---> 16000 >= 15000 ? --> PASS --> Update max_step=16000 (Atomic fsync)
```

### Ledger Invariants
For any checkpoint restoration or promotion:
1. `incoming_step >= recorded_max_step`: Step count must strictly advance monotonically.
2. If `incoming_step == recorded_max_step`, then `incoming_epoch >= recorded_max_epoch`.
3. Independent training runs (`run_id`) maintain separate monotonic high-water marks.

### Atomic Concurrency Protection
The ledger state is persisted using atomic staging:
`temporary_file -> flush -> fsync -> os.replace`
This guarantees no corrupt or half-written ledger state in the event of an abrupt process crash or host power interruption.

---

## 3. Integration Example

```python
from ahcc.provenance import FreshnessLedger
from ahcc.distributed import load_sharded_checkpoint

ledger = FreshnessLedger("/var/log/ahcc/freshness_ledger.json")

# Fails with RollbackError if step < recorded step in ledger
tensors, meta = load_sharded_checkpoint(
    checkpoint_dir="/shared/checkpoints/step_10000",
    rank=0,
    world_size=4,
    key=encryption_key,
    ledger=ledger,
)
```
