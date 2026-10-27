# Monotonic Anti-Rollback Freshness Ledger & Threat Architecture

**Document:** `docs/ROLLBACK_PROTECTION.md`  
**Date:** 2026-10-26  
**Implementation:** `ahcc.provenance.FreshnessLedger`, `ahcc.provenance.FreshnessBackend`  
**Status:** `VERIFIED`

---

## 1. The Rollback Vulnerability in Machine Learning

While digital signatures and AEAD encryption guarantee authenticity and integrity, they do not inherently prevent **replay and rollback attacks**:
- **Scenario:** A safety-critical model is fine-tuned to remove dangerous capabilities or backdoors. At Step 50,000, it is aligned and secure. An adversary with storage write access replaces the Step 50,000 checkpoint with a validly signed Step 1,000 checkpoint from the same training run.
- **Consequence:** The model loads successfully, passes AEAD decryption and signature verification, but silently runs an older, vulnerable, or unaligned version of the model.

---

## 2. Attacker Model & Trust Boundaries

The AHCC rollback mitigation framework distinguishes 6 threat scenarios:

| Scenario | Attacker Capability | Local File Backend | Trusted External Backend |
|---|---|---|---|
| **Scenario 1** | Attacker replaces checkpoint files; trusted ledger intact | `PROTECTED` | `PROTECTED` |
| **Scenario 2** | Attacker restores both checkpoint AND local ledger (e.g. disk snapshot) | `VULNERABLE` | `PROTECTED` |
| **Scenario 3** | Attacker replays valid earlier signed checkpoint | `PROTECTED` | `PROTECTED` |
| **Scenario 4** | Concurrent writers across multi-worker training cluster | `RACE RISK` | `PROTECTED (CAS / ACID)` |
| **Scenario 5** | Freshness service unavailable / network partition | `FAILS CLOSED` | `FAILS CLOSED` |
| **Scenario 6** | Write failure recovery in legitimate run | `PROTECTED` | `PROTECTED (Recovery Tokens)` |

### Critical Security Boundary Notice
`LocalFileFreshnessBackend` stores state in a local JSON file. Atomic staging and `os.fsync` prevent corrupted files across crashes, but **CANNOT** protect against Scenario 2 (an attacker who controls the local disk or can restore a local VM/filesystem snapshot). In production multi-node deployments, an external or transactional backend (`SQLiteFreshnessBackend` on network storage or remote KV service) must be used.

---

## 3. Pluggable Backend Architecture

```python
from ahcc.provenance import (
    FreshnessLedger,
    LocalFileFreshnessBackend,
    TransactionalMemoryFreshnessBackend,
    SQLiteFreshnessBackend,
)

# 1. Local development / crash-consistency testing (Scenario 1 & 3 protected)
local_ledger = FreshnessLedger(ledger_path="/tmp/freshness.json")

# 2. ACID transactional SQL backend (Concurrency & crash protected)
sql_backend = SQLiteFreshnessBackend("/mnt/secure/freshness.db", external_trusted=True)
sql_ledger = FreshnessLedger(backend=sql_backend)

# 3. High-throughput distributed in-memory service with CAS optimistic locking
memory_backend = TransactionalMemoryFreshnessBackend(external_trusted=True)
trusted_ledger = FreshnessLedger(backend=memory_backend)
```

---

## 4. Ledger Invariants & API

For any checkpoint restoration or promotion:
1. `incoming_step >= recorded_max_step`: Step count must strictly advance monotonically.
2. If `incoming_step == recorded_max_step`, then `incoming_epoch >= recorded_max_epoch`.
3. Independent training runs (`run_id`) maintain separate monotonic high-water marks.
4. Fail-closed: if the freshness service is unreachable, restoration raises `FreshnessServiceUnavailableError`.

```python
from ahcc.provenance import FreshnessLedger, RollbackError
from ahcc.distributed import load_sharded_checkpoint

ledger = FreshnessLedger(backend=sql_backend)

try:
    tensors, meta = load_sharded_checkpoint(
        checkpoint_dir="/shared/checkpoints/step_10000",
        rank=0,
        world_size=4,
        key=encryption_key,
        ledger=ledger,
    )
except RollbackError as e:
    # Stale or replayed checkpoint detected!
    raise SystemExit(f"Refusing to load rolled-back checkpoint: {e}")
```
