# AHCC v0.7.0 Release Closeout and Source Reproducibility Report

**Document:** `reports/RELEASE_CLOSEOUT_V070.md`  
**Evaluation Date:** October 27, 2026  
**Candidate Release:** AHCC Secure Checkpoint Platform v0.7.0  
**Audit Evaluation:** `VERIFIED (PROVENANCE & REPRODUCIBILITY RESOLVED)`  
**Release Gate Recommendation:** `RESTRICTED PILOT`  

---

## 1. Executive Summary

This report completes the release closeout and establishes cryptographic source-to-artifact reproducibility for the **AHCC Secure Checkpoint Platform v0.7.0**.

The Phase 10 audit (`reports/PHASE10_RELEASE_PROVENANCE.md`) identified a critical provenance gap: Git HEAD was sitting at commit `85c513a` (`v0.6.0-phase5-final`), while the v0.7.0 production changes and core test suites existed only as uncommitted modifications in the local working tree. The previous v0.7.0 wheel was built directly from that uncommitted tree.

In this closeout:
1. **Source Freeze:** All 13 tracked modified production files and 13 core test suites were reviewed, separated from untracked benchmark/evaluation artifacts, and committed into dedicated release commit **`0e0ad864de3485ce3f78d60b4df30d5aefaec73f`** (`0e0ad86`) on branch `master`.
2. **Clean-Room Build:** From an exact `git archive` export of commit `0e0ad86` into an isolated clean environment, the release wheel (`ahcc_gpu_engine-0.7.0-py3-none-any.whl`) and source distribution (`ahcc_gpu_engine-0.7.0.tar.gz`) were built without dirty working-tree contamination.
3. **Artifact Integrity:** The new artifacts were archived under `dist/release_v070/` preserving all historical artifacts in `dist/` intact. SHA-256 digests were computed and confirmed immutable.
4. **Clean-Install Verification:** The wheel was installed into a clean, isolated virtual environment outside the source tree. Import isolation and the documented public-API training checkpoint workflow executed with 100% pass rate.
5. **Full Regression Validation:** All 304 production test cases in `AHCC_GPU_Engine` and all 27 integration tests in the independent consumer repository `ahcc_test_codebase` were executed against the candidate and passed with zero failures.

---

## 2. Git Repository & Working Tree Status

### 2.1 Git Release Commit Details
- **Release Commit Hash:** `0e0ad864de3485ce3f78d60b4df30d5aefaec73f`
- **Short Hash:** `0e0ad86`
- **Commit Subject:** `release(v0.7.0): freeze source and establish release provenance`
- **Parent Commit:** `85c513ac96c8bb45d745594766cf05f4c8984e03` (`v0.6.0-phase5-final`, `phase6-baseline-start`)
- **Author / Committer:** `Researcher <researcher@example.com>`
- **Committer Date:** Mon Oct 27 2026
- **Branch:** `master`

### 2.2 Working Tree Status
- **Tracked Files:** Completely clean (`nothing added to commit, working tree clean` for tracked files). No uncommitted modifications exist in tracked repository files.
- **Untracked Artifacts Preserved:**
  - `benchmarks/`: `benchmark_phase6_streaming.py`, `benchmark_phase7_closeout.py`, `run_phase9_benchmarks.py`, `run_recovery_experiment.py`
  - `pilot/`: Developer trial guides and trial logs
  - `reports/`: Phase 6–10 security, benchmark, and audit reports
  - `results/`: Raw CSV and JSON benchmark results
  - `scripts/`: Development and audit helper scripts
- **Git Tags:** Preserved without alteration (7 historical tags intact: `phase1-baseline`, `phase2-baseline-start`, `v0.3.0-phase2-rc1`, `v0.4.0-phase3-final`, `v0.5.0-phase4-final`, `phase6-baseline-start`, `v0.6.0-phase5-final`). In accordance with policy, no new tag was created and no commits were pushed to remote.

---

## 3. Reviewed Production Changes (Commit `0e0ad86`)

The dedicated release commit encompasses exactly **26 files** (13 tracked modified files and 13 core test suites; 3,889 insertions, 114 deletions):

### 3.1 Production Source Files (9 files)
1. `ahcc/__init__.py`: Version bumped to `"0.7.0"`. Exposed top-level public APIs: `save_training_checkpoint`, `load_training_checkpoint`, `inspect_training_checkpoint`, `TrainingProgress`, `LedgerStorageBackend`, and `LocalFileLedgerBackend`.
2. `ahcc/checkpoint.py`: Integrated Ed25519 digital signature signing and verification (`sign_checkpoint`, `verify_checkpoint_signature`), enhanced manifest authentication, and chunk-bounds validation.
3. `ahcc/key_providers/base.py`: Formalized `KeyProvider` abstract interface with `generate_key`, `wrap_key`, and `unwrap_key` methods.
4. `ahcc/key_providers/kms.py`: Production hardening for AWS KMS: bounded exponential backoff with jitter retries, sanitized error messages (preventing secret leakage), and AWS KMS context manager support.
5. `ahcc/key_providers/local.py`: Local key wrapping envelope support (`generate_key`, `wrap_key`, `unwrap_key`).
6. `ahcc/provenance.py`: Comprehensive digital provenance: multi-backend `FreshnessLedger` (in-memory, local file, distributed), monotonic counter validation for anti-rollback protection, and Ed25519 signing/verifying.
7. `ahcc/training_state.py`: Core high-level safe training state orchestration: `save_training_checkpoint`, `load_training_checkpoint`, `inspect_training_checkpoint`, `TrainingProgress`. Implements atomic staging (`.tmp` write followed by atomic rename), canonical allowlisted JSON metadata, zero-pickle enforcement, and RNG state restoration.
8. `README.md`: Quickstart and public API reference updated for v0.7.0 training checkpoint management and KMS key provider usage.
9. `SECURITY.md`: Clarified public HKDF salt (bytes 0x09-0x28), V5 subkey derivation, and NIST SP 800-38D monotonic counter collision bounds ($P=0$ intra-key vs $2^{-128}$ inter-key).

### 3.2 Documentation & Configuration Files (4 files)
10. `docs/DISTRIBUTED_CHECKPOINTS.md`: Updated distributed protocol specifications and v0.7.0 architecture notes.
11. `docs/ROLLBACK_PROTECTION.md`: Anti-rollback ledger and snapshot limitation threat model documentation.
12. `docs/TRAINING_CHECKPOINT_SCHEMA.md`: Clarified public derivation salt at bytes 0x09-0x28.
13. `pyproject.toml`: Package version bumped from `0.6.0` to `0.7.0`.

### 3.3 Core Test Suites (13 new files)
14. `tests/test_phase6_aead_vectors.py`: AEAD known-answer vectors, NIST SP 800-38D counter bounds.
15. `tests/test_phase6_crash_recovery.py`: Atomic write crash recovery and truncated container handling.
16. `tests/test_phase6_distributed_integrity.py`: Distributed sharded checkpoint verification.
17. `tests/test_phase6_freshness.py`: FreshnessLedger anti-rollback and monotonic counter tests.
18. `tests/test_phase6_key_derivation.py`: HKDF key schedule and subkey isolation tests.
19. `tests/test_phase6_kms_provider.py`: KMS key provider, error sanitization, and retry tests.
20. `tests/test_phase6_nonce_protocol.py`: Nonce generation, uniqueness, and exhaustion tests.
21. `tests/test_phase6_pickle_free.py`: Safe unpickling prevention and pure tensor serialization.
22. `tests/test_phase6_resource_limits.py`: Resource exhaustion limits, OOM protection, large file tests.
23. `tests/test_phase6_signatures.py`: Ed25519 signature creation, verification, and tamper detection.
24. `tests/test_phase6_untrusted_input.py`: Adversarial payload rejection, header tampering.
25. `tests/test_phase7_training_recovery.py`: End-to-end training state recovery, optimizer resume, RNG state restore.
26. `tests/test_phase8_public_api.py`: Comprehensive test suite for the v0.7.0 top-level public APIs.

---

## 4. Distribution Artifact Digests & Build Provenance

### 4.1 Build Environment
- **Operating System:** Windows 11 Enterprise (x86_64)
- **Python Version:** 3.12.10 (MSC v.1943 64-bit AMD64)
- **Build Tool:** `build` v1.5.1
- **Packaging Backend:** `setuptools` v84.2.1, `wheel` v0.46.3
- **Source Revision:** Git Commit `0e0ad864de3485ce3f78d60b4df30d5aefaec73f` (`git archive` export)

### 4.2 Clean Release Artifact Digests
The clean artifacts built directly from commit `0e0ad86` are located in `dist/release_v070/`:

| Artifact Filename | Type | Size (Bytes) | SHA-256 Digest | Status |
|---|:---:|:---:|---|:---:|
| `ahcc_gpu_engine-0.7.0-py3-none-any.whl` | Wheel (`bdist_wheel`) | 174,748 | `68cac9e130140af356df925405849f356f91f231526cbf3b741985edad16fe18` | **REPRODUCIBLE CANDIDATE** |
| `ahcc_gpu_engine-0.7.0.tar.gz` | Source Dist (`sdist`) | 140,333 | `b5598b02a1cf4376e1b2ba46815c3b2de3a1b4bdfdb3da0963dd6638c1ad2060` | **REPRODUCIBLE CANDIDATE** |
| `SHA256SUMS` | Verification Manifest | 202 | `2bbc65be606d789ff5134ddf81ddc46cde8a8b7d726a103a8903607a6eb72979` | **SIGNED HASH MANIFEST** |

### 4.3 Provenance Reconciliation with Phase 9 Artifacts
- **Phase 9 Wheel (`dist/ahcc_gpu_engine-0.7.0-py3-none-any.whl`, 185,669 bytes, SHA-256 `0c54c9dd...`):** Built from an uncommitted working tree; setuptools globbed three untracked benchmark scripts (`benchmark_phase6_streaming.py`, `benchmark_phase7_closeout.py`, `run_recovery_experiment.py`).
- **Phase 10 Closeout Wheel (`dist/release_v070/ahcc_gpu_engine-0.7.0-py3-none-any.whl`, 174,748 bytes, SHA-256 `68cac9e1...`):** Built from the exact committed source revision `0e0ad86`. Contains strictly the 192 tracked files from Git HEAD. All internal files match commit `0e0ad86` byte-for-byte.
- Existing historical release artifacts in `dist/` were preserved intact without modification.

---

## 5. Clean-Install and Test Verification Record

### 5.1 Fresh Virtual Environment Installation & Public API Workflow
- **Environment:** `C:\Users\cshar\AppData\Local\Temp\ahcc_v070_clean_test_venv` (outside source checkout)
- **Install Command:**
  ```powershell
  pip install --no-deps c:\Users\cshar\Downloads\AHCC\AHCC_GPU_Engine\dist\release_v070\ahcc_gpu_engine-0.7.0-py3-none-any.whl
  ```
- **Exit Code:** `0`
- **Import Isolation Verification:**
  - Resolved module: `C:\Users\cshar\AppData\Local\Temp\ahcc_v070_clean_test_venv\Lib\site-packages\ahcc\__init__.py`
  - Version: `0.7.0`
  - Working directory isolation: Clean (no leakage from repo source)
- **Public-API Workflow Execution:**
  - Initialized `nn.Sequential` model, AdamW optimizer, and StepLR scheduler.
  - Executed training step, constructed `TrainingProgress(global_step=100, epoch=2, batch_idx=14, loss=...)`.
  - Saved via `save_training_checkpoint()`.
  - Inspected zero-key container header via `inspect_training_checkpoint()` (`AHCC-SafeTrainingCheckpoint-v5`, `authenticated=False`).
  - Inspected authenticated manifest with master key (`global_step=100`, `epoch=2`, `authenticated=True`).
  - Restored training state into fresh model/optimizer/scheduler via `load_training_checkpoint()`.
  - Verified exact parameter equality (`torch.equal(p1, p2)` across all tensors).
  - Workflow Result: **`100% PASSED (EXIT CODE 0)`**

### 5.2 Independent Consumer Integration Tests (`ahcc_test_codebase`)
- **Repository:** `c:\Users\cshar\Downloads\AHCC\ahcc_test_codebase`
- **Environment:** `c:\Users\cshar\Downloads\AHCC\ahcc_test_codebase\.venv`
- **Reinstallation:** Force-reinstalled clean candidate wheel `dist/release_v070/ahcc_gpu_engine-0.7.0-py3-none-any.whl`.
- **Command:**
  ```powershell
  c:\Users\cshar\Downloads\AHCC\ahcc_test_codebase\.venv\Scripts\python.exe -m pytest -v tests
  ```
- **Exit Code:** `0`
- **Results:** **27 passed in 24.55s**, 0 failed, 0 errors, 0 warnings.
- **Key Suites Verified:**
  - `test_checkpoint_manager.py`: CheckpointManager save, zero-key inspect, authenticated inspect, restore, negative wrong-key and tamper fail-closed handling.
  - `test_cli_and_unsupported_inputs.py`: Invalid key length, unsupported dtypes, fail-closed key providers, CLI invocation.
  - `test_crash_recovery.py`: Atomic write crash recovery, stale `.tmp` cleanup.
  - `test_equivalence.py`: Uninterrupted vs resumed training equivalence on CPU and CUDA.
  - `test_kms_mock.py`: KMS mock provider save/load and failure modes.
  - `test_no_pickle.py`: Zero pickle invocations, rejection of raw pickled checkpoints without execution, raw byte decoding.
  - `test_security_failures.py`: Wrong key, tampered chunk, truncated file, corrupted manifest, invalid signature, missing shard, monotonic freshness anti-rollback, disk snapshot limitations.

### 5.3 AHCC Engine Regression Suite (`AHCC_GPU_Engine`)
- **Repository:** `c:\Users\cshar\Downloads\AHCC\AHCC_GPU_Engine`
- **Command:**
  ```powershell
  pytest -v tests
  ```
- **Exit Code:** `0`
- **Results:** **304 passed, 167 warnings in 36.51s**, 0 failed, 0 errors.
- **Warnings:** 167 `InsecureCipherWarning`s emitted exclusively from unit tests targeting the experimental research cipher `AHCC_Engine` (as designed, documenting that raw AHCC is unauthenticated and unsuitable for direct production encryption).

---

## 6. Discrepancies, Constraints & Blockers

1. **Git Release Tag Policy:** No Git tag (`v0.7.0`) was created. Tagging requires explicit authorization from the project owner. Commit `0e0ad86` is prepared and ready for tagging upon authorization.
2. **Remote Push Restriction:** No commits have been pushed to a remote repository.
3. **Storage-Level Rollback Boundary:** As documented in `docs/ROLLBACK_PROTECTION.md` and verified in `tests/test_security_failures.py`, local file-based `FreshnessLedger` cannot detect a hypervisor/VM-level snapshot revert that restores both the checkpoint file and the local ledger file simultaneously. Production anti-rollback requires distributed ledger backends or KMS-managed nonces.
4. **Experimental Core Cipher Boundary:** `AHCC_Engine` remains an unauthenticated research cipher. Production deployments must use `SecureModelCheckpoint` / `save_training_checkpoint` backed by standard `AESGCMEnvelope` (AES-256-GCM).

---

## 7. Developer Reproduction & Quickstart Guide

To verify and exercise the v0.7.0 release candidate in any environment:

### Step 1: Install the Candidate Wheel
```bash
pip install dist/release_v070/ahcc_gpu_engine-0.7.0-py3-none-any.whl
```

### Step 2: Verify Artifact Integrity
```bash
# Windows PowerShell
Get-FileHash dist/release_v070/ahcc_gpu_engine-0.7.0-py3-none-any.whl -Algorithm SHA256
# Expected: 68CAC9E130140AF356DF925405849F356F91F231526CBF3B741985EDAD16FE18

# Linux/macOS
sha256sum dist/release_v070/ahcc_gpu_engine-0.7.0-py3-none-any.whl
```

### Step 3: Exercise Public-API Workflow
```python
import torch
import torch.nn as nn
from ahcc import (
    save_training_checkpoint,
    load_training_checkpoint,
    inspect_training_checkpoint,
    TrainingProgress,
    AESGCMEnvelope,
)

# 1. Initialize
model = nn.Sequential(nn.Linear(10, 4), nn.ReLU(), nn.Linear(4, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
key = AESGCMEnvelope.generate_key()

# 2. Save
progress = TrainingProgress(global_step=1, epoch=0, batch_idx=0, loss=0.5)
save_training_checkpoint("checkpoint.ahcc", model=model, optimizer=optimizer, progress=progress, key=key)

# 3. Inspect
info = inspect_training_checkpoint("checkpoint.ahcc", key=key)
print(f"Verified checkpoint: step={info['global_step']}, format={info['format']}")

# 4. Resume
restored_model = nn.Sequential(nn.Linear(10, 4), nn.ReLU(), nn.Linear(4, 2))
restored_prog = load_training_checkpoint("checkpoint.ahcc", model=restored_model, key=key)
print(f"Successfully resumed at step {restored_prog.global_step}")
```

---

## 8. Final Release Decision

| Evaluation Dimension | Status | Evidence / Result |
|---|:---:|---|
| Source Freeze & Commit Identity | **`VERIFIED`** | Commit `0e0ad864de3485ce3f78d60b4df30d5aefaec73f` |
| Working-Tree Cleanliness (Tracked) | **`VERIFIED`** | 0 modified tracked files remaining |
| Artifact Lineage & Reproducibility | **`VERIFIED`** | Built from clean `git archive` export in isolated env |
| Artifact Digests Computed | **`VERIFIED`** | Wheel SHA-256 `68cac9e1...`, sdist SHA-256 `b5598b02...` |
| Historical Artifact Preservation | **`VERIFIED`** | `dist/` past artifacts untouched; output in `dist/release_v070/` |
| Fresh Virtual Environment Install | **`VERIFIED`** | Installed outside source tree; import isolation confirmed |
| Documented Public-API Workflow | **`VERIFIED`** | End-to-end save, inspect, load roundtrip passed |
| Consumer Test Suite (`ahcc_test_codebase`) | **`VERIFIED`** | 27 / 27 tests passed (exit code 0) |
| Engine Test Suite (`AHCC_GPU_Engine`) | **`VERIFIED`** | 304 / 304 tests passed (exit code 0) |
| Post-Build Immutability Check | **`VERIFIED`** | Artifact and source digests re-verified identical |

### Release Recommendation
**`RESTRICTED PILOT`**

The release provenance and reproducibility gap identified in Phase 10 is **fully resolved**. The source is frozen at commit `0e0ad86`, and the release artifacts are reproducible. Deployment remains strictly bounded at `RESTRICTED PILOT` pending operational telemetry, production KMS setup, and the project owner's authorization for Git release tagging and publishing.
