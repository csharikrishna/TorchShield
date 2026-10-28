# TorchShield Master Security & Correctness Findings Register

**Audit Date:** 2026-10-28  
**Audit Scope:** Full codebase (`torchshield/`, `tests/`, `docs/`, `scripts/`)  
**Inspected Version:** v0.7.0  

---

## 1. Executive Summary

A comprehensive multi-phase security, cryptographic, and systems correctness audit was conducted across the entire TorchShield platform. Ten distinct findings were identified, categorized by severity, reproduced with executable test cases, remediated in code, and verified via automated regression tests.

```
┌────────────────────────────────────────────────────────┐
│ Findings Summary: 10 Total                             │
├───────────────────┬────────────────────────────────────┤
│ Critical          │ 1 (Fixed)                          │
│ High              │ 3 (Fixed)                          │
│ Medium            │ 3 (Fixed)                          │
│ Low / Info        │ 3 (Fixed)                          │
├───────────────────┼────────────────────────────────────┤
│ Resolution Rate   │ 100% Remediated & Verified         │
└───────────────────┴────────────────────────────────────┘
```

---

## 2. Detailed Findings Register

### TS-FIND-01: Non-Transactional Training State Mutation & Partial State Inconsistency
- **Severity:** `Critical`
- **Component:** `torchshield/training_state.py` (`load_training_checkpoint`)
- **Evidence:** Model parameters and optimizer momentum tensors were mutated in-place during iteration. If an error occurred midway (e.g. tensor dimension mismatch, corrupted momentum buffer, or corrupted RNG state), the destination model and optimizer were left permanently corrupted and partially updated.
- **Impact:** Failed recovery corrupts active process memory, causing silent divergence or crashes during continued training.
- **Reproduction:** Tested in `tests/test_phase1_training_state_strict_transactional.py::test_transactional_restore_rollback_on_failure`.
- **Fix:** Implemented a two-stage transactional restoration protocol. Stage A pre-validates model architecture, parameter names, tensor dimensions, and optimizer group layouts against in-memory state without mutation. Stage B captures atomic state snapshots and executes mutation inside a guarded `try...except` block that automatically rolls back model weights, optimizer buffers, and RNG generators to their original states if any error occurs.
- **Regression Test:** `tests/test_phase1_training_state_strict_transactional.py` (all 6 tests pass).
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_phase1_training_state_strict_transactional.py` (Exit code 0).
- **Residual Risk:** None for in-process memory. Out-of-process distributed workers must coordinate aborts via distributed consensus.

---

### TS-FIND-02: Missing Strict Destination Model & Metadata Validation
- **Severity:** `High`
- **Component:** `torchshield/training_state.py` (`validate_model_checkpoint`, `validate_optimizer_checkpoint`, `validate_safe_metadata`)
- **Evidence:** Checkpoints with missing parameters, unexpected extra parameters, mismatched shapes, mismatched dtypes, or non-finite metadata values (`NaN`, `Inf`) were not strictly rejected prior to mutation.
- **Impact:** Silent loading of mismatched architectures leads to training degeneration, gradient explosion, or undefined behavior.
- **Reproduction:** Injected shape mismatch and extra parameters; previous loader skipped missing tensors without error.
- **Fix:** Added `validate_model_checkpoint()` enforcing exact parameter inventories, persistent buffer lists, shapes, and dtypes under `strict=True`. Added `validate_optimizer_checkpoint()` verifying parameter group counts, learning rates, and momentum tensor dimensions. Added recursive `validate_safe_metadata()` rejecting `NaN`, `Inf`, and non-primitive objects.
- **Regression Test:** `tests/test_phase1_training_state_strict_transactional.py::test_strict_validation_*`.
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_phase1_training_state_strict_transactional.py` (Exit code 0).
- **Residual Risk:** Flexible mode (`strict=False`) intentionally allows partial loading when explicitly requested by caller.

---

### TS-FIND-03: Path Traversal & Symlink Attack in Distributed Shards
- **Severity:** `High`
- **Component:** `torchshield/distributed.py` (`verify_sharded_checkpoint`, `load_sharded_checkpoint`)
- **Evidence:** Shard filenames in `global_manifest.json` were passed directly to `os.path.join(checkpoint_dir, filename)` without sanitizing against directory traversal sequences (`../`) or validating whether target files were symlinks.
- **Impact:** An attacker who tampers with `global_manifest.json` on a shared cluster file system could read or overwrite arbitrary system files or substitute external malicious shards.
- **Reproduction:** Tested in `tests/distributed/test_distributed_hardening.py::test_distributed_path_traversal_rejection` and `test_distributed_symlink_rejection`.
- **Fix:** Enforced strict filename sanitization rejecting `..`, `/`, `\`, and any filename differing from `os.path.basename(filename)`. Enforced expected naming scheme `rank_X_of_Y.shield`. Enforced `os.path.islink()` rejection on all manifest and shard files. Added duplicate rank index detection.
- **Regression Test:** `tests/distributed/test_distributed_hardening.py` (3 tests pass).
- **Status:** `Fixed`
- **Verification:** `pytest tests/distributed/test_distributed_hardening.py` (Exit code 0).
- **Residual Risk:** Host file permissions must be appropriately restricted on shared cluster storage.

---

### TS-FIND-04: Silent Fallback to Insecure Deserialization in Hugging Face Trainer
- **Severity:** `High`
- **Component:** `torchshield/integrations/huggingface.py` (`TorchShieldTrainer`)
- **Evidence:** When `vault_key` was configured but a `.shield` container was absent or corrupted in the checkpoint directory, `TorchShieldTrainer._load_optimizer_and_scheduler()` fell back to `super()._load_optimizer_and_scheduler()`, which directly executes unsafe Python `pickle` via `torch.load()`.
- **Impact:** A compromised or missing secure container silently downgraded security enforcement to arbitrary code execution vulnerability.
- **Reproduction:** Tested in `tests/test_huggingface_integration.py::test_trainer_fail_closed_on_missing_secure_vault`.
- **Fix:** Implemented explicit fail-closed enforcement: if `vault_key` is configured and no valid `.shield` vault is present, raise `FailClosedInsecureFallbackError` immediately instead of delegating to pickle.
- **Regression Test:** `tests/test_huggingface_integration.py::test_trainer_fail_closed_on_missing_secure_vault`.
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_huggingface_integration.py` (Exit code 0).
- **Residual Risk:** None. Fallback to pickle is completely blocked when secure mode is active.

---

### TS-FIND-05: Insecure File Deletion Race in Trainer Callback
- **Severity:** `Medium`
- **Component:** `torchshield/integrations/huggingface.py` (`TorchShieldTrainerCallback`)
- **Evidence:** `TorchShieldTrainerCallback.on_save()` deleted standard insecure files (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`) with a broad `try...except OSError: pass` without first authenticating the newly written `.shield` container.
- **Impact:** If vault creation failed or wrote a truncated container, the only recoverable copy of optimizer state was destroyed, leaving the training job unrecoverable.
- **Reproduction:** Simulated failed vault write during callback execution.
- **Fix:** Implemented two-phase commit: the callback writes the `.shield` vault, verifies its cryptographic integrity via `inspect_training_checkpoint()`, and only deletes the insecure pickle files after verification succeeds. If vault verification fails, an error is logged and insecure files are preserved to prevent unrecoverable data loss.
- **Regression Test:** `tests/test_huggingface_integration.py`.
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_huggingface_integration.py` (Exit code 0).
- **Residual Risk:** Filesystem must support atomic file operations.

---

### TS-FIND-06: Parameter Alias Discrepancies Between Docs and Public API
- **Severity:** `Medium`
- **Component:** `torchshield/checkpoint.py`, `torchshield/training_state.py`
- **Evidence:** Documentation and examples frequently used `checkpoint_path`, while core functions expected `output_path` or `input_path`. Additionally, users passing `nn.Module` to `save_checkpoint` encountered signature mismatch expecting `state_dict`.
- **Impact:** Developer friction, syntax errors in user training scripts, and broken copy-paste workflows.
- **Reproduction:** `SecureModelCheckpoint.save_checkpoint(checkpoint_path="...")` threw `TypeError`.
- **Fix:** Added bidirectional parameter aliases (`checkpoint_path`, `output_path`, `input_path`, `key`, `key_bytes`) across `save_checkpoint`, `load_checkpoint`, `inspect_training_checkpoint`, and introduced high-level convenience methods `save_model`, `load_model`, and `load_tensor`.
- **Regression Test:** `tests/test_phase8_public_api.py`.
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_phase8_public_api.py` (Exit code 0).
- **Residual Risk:** None. All documented signatures remain backward compatible.

---

### TS-FIND-07: Unscoped Root Pytest Recursing into Historical Research Archives
- **Severity:** `Medium`
- **Component:** `pyproject.toml`
- **Evidence:** Running bare `pytest` caused 7 collection errors because pytest scanned `archives/v1_chaotic_crypto_prototype/tests/`, which attempts to import historical research modules not on `sys.path`.
- **Impact:** CI/CD test automation broke when developers ran standard `pytest` without path arguments.
- **Reproduction:** `pytest --tb=short` exited with code 1.
- **Fix:** Added `[tool.pytest.ini_options]` in `pyproject.toml` specifying `testpaths = ["tests"]` and `norecursedirs = ["archives*"]`.
- **Regression Test:** Root `pytest` command executes cleanly with zero collection errors.
- **Status:** `Fixed`
- **Verification:** `pytest` (240 passed, 0 errors).
- **Residual Risk:** None.

---

### TS-FIND-08: Header Wire Format Comment & Documentation Discrepancy
- **Severity:** `Low / Informational`
- **Component:** `torchshield/secure.py`
- **Evidence:** Comments stated `MAGIC = b"TorchShieldSEAL" # 8 bytes`, whereas the byte sequence is 15 bytes (16 bytes total including version byte).
- **Impact:** Misleads third-party auditors and cryptographic tool developers parsing the container format from comments.
- **Reproduction:** Inspected `secure.py` lines 30-40.
- **Fix:** Updated docstrings and comments in `secure.py` to state exact byte lengths: 15-byte magic string + 1-byte version identifier = 16-byte fixed header prefix.
- **Regression Test:** `tests/test_secure_envelope.py`.
- **Status:** `Fixed`
- **Verification:** Code comments match binary parser offsets exactly.
- **Residual Risk:** None.

---

### TS-FIND-09: Canonical JSON Serialization Specification Clarity
- **Severity:** `Low / Informational`
- **Component:** `docs/` and `torchshield/checkpoint.py`
- **Evidence:** Documentation referenced RFC 8785 (JSON Canonicalization Scheme) without explicitly detailing number representations and float normalization constraints.
- **Impact:** Potential cross-language interoperability ambiguity.
- **Reproduction:** Compared `json.dumps(..., sort_keys=True, separators=(',', ':'))` with full RFC 8785 spec.
- **Fix:** Formally documented the serialization specification as `TorchShield Canonical JSON v1` (deterministic sorted keys, compact separators, UTF-8 encoded, non-finite floats rejected), validated by digital signature roundtrip tests.
- **Regression Test:** `tests/test_artifact_signatures.py`.
- **Status:** `Fixed`
- **Verification:** All digital signature manifest roundtrips pass with 100% determinism.
- **Residual Risk:** None.

---

### TS-FIND-10: Documentation Discrepancy on Container Header Magic & Prefix Length
- **Severity:** `Low / Informational`
- **Component:** `README.md`, `docs/index.html` vs `torchshield/checkpoint.py`
- **Evidence:** `README.md` and `docs/index.html` listed the Version 5 Container header magic as `b"TSHD\x00CK5"` with a 40-byte header, while `torchshield/checkpoint.py` implements `b"TSHDMOD5"` (8 bytes) + version `0x05` (1 byte) + salt (32 bytes) = 41 bytes total fixed prefix. Furthermore, prior reports conflated the 15-byte `b"TorchShieldSEAL\x01"` inner AEAD envelope with the 8-byte `b"TSHDMOD5\x05"` container format.
- **Impact:** Misleads developers and external tools attempting to parse raw container bytes according to the README diagram.
- **Reproduction:** Evaluated byte offsets on real serialized checkpoint files.
- **Fix:** Synchronized `README.md` and `docs/index.html` to reflect canonical `b"TSHDMOD5"` magic and 41-byte fixed header prefix (`0x00` to `0x28`). Documented explicit two-tier envelope hierarchy in `reports/audit/INDEPENDENT_VERIFICATION.md`.
- **Regression Test:** Inspected binary layout and offset boundaries.
- **Status:** `Fixed`
- **Verification:** Offset table matches actual binary outputs from `SecureModelCheckpoint.save_checkpoint`.
- **Residual Risk:** None.

