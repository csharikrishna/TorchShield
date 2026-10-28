# TorchShield Master Security & Correctness Findings Register

**Audit Date:** 2026-10-28  
**Audit Scope:** Full codebase (`torchshield/`, `tests/`, `docs/`, `scripts/`)  
**Inspected Version:** v0.7.0  

---

## 1. Executive Summary

A comprehensive multi-phase security, cryptographic, and systems correctness audit was conducted across the entire TorchShield platform. Seventeen distinct findings were identified, categorized by severity, reproduced with executable test cases, remediated in code, and verified via automated regression tests.

```
┌────────────────────────────────────────────────────────┐
│ Findings Summary: 17 Total                             │
├───────────────────┬────────────────────────────────────┤
│ Critical (P1)     │ 3 (Fixed)                          │
│ High (P1)         │ 5 (Fixed)                          │
│ Medium            │ 6 (Fixed)                          │
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

---

### TS-FIND-11: Optimizer Rollback Disconnects Parameter Identity from Target Model (P1)
- **Severity:** `Critical (P1)`
- **Component:** `torchshield/training_state.py` (`_snapshot_optimizer`, `_rollback_optimizer`)
- **Evidence:** `_snapshot_optimizer()` performed `copy.deepcopy(optimizer.param_groups)`. In PyTorch, deep-copying `param_groups` also deep-copies the constituent `nn.Parameter` tensors. When rollback restored these deep-copied groups, the optimizer held references to detached cloned parameters instead of the destination model's live parameters.
- **Impact:** Subsequent training iterations executed `optimizer.step()` against detached tensor objects, silently failing to update actual model parameters and leaving momentum states out of sync.
- **Reproduction:** Tested in `tests/test_phase1_training_state_strict_transactional.py::test_optimizer_rollback_preserves_parameter_identity_and_updates_model`.
- **Fix:** Refactored `_snapshot_optimizer` to preserve exact object identity for all `nn.Parameter` references using shallow list copies for `grp["params"]` while deep-copying hyperparameters and detached momentum states. Refactored `_rollback_optimizer` to restore `live_grp["params"]` maintaining identity.
- **Regression Test:** `tests/test_phase1_training_state_strict_transactional.py::test_optimizer_rollback_preserves_parameter_identity_and_updates_model` verifies `opt_p is model_p` for all parameters, executes `loss.backward()`, and verifies `dest_opt.step()` mutates `dest_model` parameters.
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_phase1_training_state_strict_transactional.py` (Exit code 0).
- **Residual Risk:** None.

---

### TS-FIND-12: Attestation Verifier Accepted Unsigned Quotes & Lacked Challenge Binding (P1)
- **Severity:** `Critical (P1)`
- **Component:** `torchshield/attestation.py` (`HardwareAttestationVerifier.verify`, `AttestationGatedKeyProvider`)
- **Evidence:** `HardwareAttestationVerifier.verify()` guarded signature verification inside `if evidence.signature is not None:`. If an attestation quote omitted the cryptographic signature (`evidence.signature is None`), the verification loop fell through and returned `True`. Furthermore, staged evidence in `AttestationGatedKeyProvider` lacked binding to the verifier's fresh challenge nonce, and `is_production_secure` could advertise `True` even under mock modes or unsigned quotes.
- **Impact:** Attackers could forge hardware quotes (e.g. spoofing TDX or SEV-SNP golden hashes) without possessing a valid hardware root-of-trust private signing key, and replay quotes without fresh challenge binding.
- **Reproduction:** Tested in `tests/test_hardware_attestation.py::test_attestation_missing_signature_fails_closed` and `test_attestation_stage_evidence_without_challenge_fails_closed`.
- **Fix:**
  1. Added `require_signature: bool = True` to `AttestationPolicy` by default, failing closed if signatures or signer IDs are absent.
  2. Hardened `AttestationGatedKeyProvider.is_production_secure` to strictly return `False` when mock attestation is active, trusted signing keys are missing, challenge nonces are unbound, or signature validation is disabled.
  3. Implemented `create_challenge()` and `stage_evidence(evidence, challenge)` requiring cryptographic challenge nonce binding under `require_fresh_nonce=True`.
  4. Preserved staged evidence on verification failure so callers can inspect failure diagnostics without premature destruction.
  5. Documented that current verification covers a custom signed evidence structure, not vendor hardware quotes.
- **Regression Tests:** `tests/test_hardware_attestation.py` (12 tests pass).
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_hardware_attestation.py` (Exit code 0).
- **Residual Risk:** Full vendor-native hardware quote verifiers (Hopper/Blackwell CC, AMD SEV-SNP) remain on roadmap.

---

### TS-FIND-13: Incomplete Rollback Coverage for State Providers and Suppressed Errors (P1)
- **Severity:** `High (P1)`
- **Component:** `torchshield/training_state.py` (`load_training_checkpoint`, `_snapshot_state_providers`, `_rollback_state_providers`)
- **Evidence:** `load_training_checkpoint` restored auxiliary `SafeStateProvider` instances in Stage B but did not snapshot or roll them back on error. Furthermore, snapshot/rollback errors were suppressed with `pass`, duplicate state provider keys could corrupt snapshots, and error handlers invoked `get_state_key()` a second time, risking secondary exceptions and unhandled variable scoping errors.
- **Impact:** Partial restoration failure left auxiliary state providers corrupted, duplicate keys caused undefined snapshot states, and secondary exceptions masked original errors.
- **Reproduction:** Tested in `tests/test_phase1_training_state_strict_transactional.py::test_transactional_rollback_on_state_provider_failure`, `test_state_provider_failing_get_state_key`, and `test_state_provider_duplicate_keys_rejected`.
- **Fix:**
  1. Pre-initialized diagnostic variables (`desc`) before all `try` blocks to prevent `UnboundLocalError`.
  2. Enforced duplicate key detection before mutating or applying snapshots.
  3. Never re-invoked `get_state_key()` inside exception handlers; used pre-computed diagnostic descriptors.
  4. Preserved optimizer parameter identity (`p is model_param`) on rollback.
  5. Implemented comprehensive exception collection distinguishing restoration vs rollback errors.
- **Regression Tests:** `tests/test_phase1_training_state_strict_transactional.py` (13 tests pass).
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_phase1_training_state_strict_transactional.py` (Exit code 0).
- **Residual Risk:** None.

---

### TS-FIND-14: GPUDirect Storage Mode Advertising Inaccuracy
- **Severity:** `Medium`
- **Component:** `torchshield/gds.py` (`probe_gds_capability`, `GDSStatus`)
- **Evidence:** `probe_gds_capability()` previously indicated active direct cuFile support when `libcufile` was present, even though direct cuFile DMA read bindings (`cuFileRead`, `cuFileHandleRegister`) are not wired into the loader.
- **Impact:** Misleading diagnostic status advertising an un-wired hardware pathway.
- **Reproduction:** Inspected `GDSLoader.load_model_direct()`, which operates via pinned CUDA streaming.
- **Fix:** Explicitly defined `GDSStatus` fields: `available=False` (since direct cuFile DMA read I/O is not implemented), `cufile_loaded` (bool diagnostic), and `active_mode="pinned_streaming_fallback"`. Added unit test asserting exact field meanings across CUDA and cuFile permutations.
- **Regression Tests:** `tests/test_gds_loading.py` (5 tests pass).
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_gds_loading.py` (Exit code 0).
- **Residual Risk:** None. Pinned streaming fallback is verified and operational.

---

### TS-FIND-15: PQC Signature Test Verifier Checked Only Length & Overclaimed KAT/CAVP Compliance
- **Severity:** `High`
- **Component:** `torchshield/pqc.py` (`MockTestPQCBackend`, `HybridSignatureEngine.verify`)
- **Evidence:** In the prototype test harness (`use_test_backend=True`), `HybridSignatureEngine.verify()` only checked `len(mldsa_sig_bytes) != 64` without checking signature bytes against the expected public key or digest. Furthermore, the mock backend was labeled with KAT/CAVP claims despite using classical primitives.
- **Impact:** Arbitrary 64-byte payloads were accepted as valid signatures in tests, and mock testing primitives were inaccurately labeled as standard-validated lattice cryptography.
- **Reproduction:** Tested in `tests/test_pqc_hybrid_crypto.py::test_hybrid_signatures_tampered_pqc_component_fails`.
- **Fix:**
  1. Renamed `MockAuditedPQCBackend` to `MockTestPQCBackend` (retained backwards-compatibility aliases).
  2. Updated docstrings to explicitly state that the backend is strictly an internal test fixture using classical primitives (X25519/Ed25519) to test framework plumbing, NOT NIST FIPS 203/204 or KAT/CAVP vectors.
  3. Replaced all salts/domains with `b"TorchShield-MockTestPQC-..."`.
  4. Wired `MockTestPQCBackend.dsa_sign` / `dsa_verify` to cryptographically verify signature bytes against the public key and manifest digest, failing closed on tampering.
  5. Enforced that production mode strictly fails closed with `PQCUnavailableError` when an external audited C library (`liboqs`) is not present.
- **Regression Tests:** `tests/test_pqc_hybrid_crypto.py` (5 tests pass).
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_pqc_hybrid_crypto.py` (Exit code 0).
- **Residual Risk:** Production PQC continues to fail closed until audited liboqs C library bindings are linked.

---

### TS-FIND-16: Model Serving In-Memory Semantics & Rollback
- **Severity:** `Medium`
- **Component:** `torchshield/integrations/serving.py` (`SecureInferenceLoader.load_into_model`, `stream_weights`)
- **Evidence:** `load_into_model()` partially updated model parameters before validating the complete key set or stream, leaving destination tensors in a broken state if an error occurred. The docstring also claimed streaming and kernel-bypass disk isolation with an unverified `zero_disk_enforcement` parameter.
- **Impact:** Mid-load failures corrupted inference models in host memory; speculative policy flags conveyed unverified security promises.
- **Reproduction:** Tested in `tests/test_serving_integration.py::test_secure_inference_loader_rollback_on_mid_load_error`.
- **Fix:** Clarified module and class docstrings to state in-memory deserialization without intermediate plaintext disk staging. Removed unused `zero_disk_enforcement` parameter. Added snapshot and rollback to `load_into_model()` to cleanly restore original model parameters if loading encounters any error.
- **Regression Tests:** `tests/test_serving_integration.py` (6 tests pass).
- **Status:** `Fixed`
- **Verification:** `pytest tests/test_serving_integration.py` (Exit code 0).
- **Residual Risk:** None.

---

### TS-FIND-17: AEAD Benchmark Nonce Reuse & Aggregation Methodology
- **Severity:** `Medium`
- **Component:** `benchmarks/run_v070_benchmarks.py` (`benchmark_aead_raw`, `benchmark_checkpoint_pipeline`)
- **Evidence:** Earlier benchmarks reused nonces across iterations, resetting counters per chunk size and using static nonces. Results also reported single-run iterations without multi-trial median aggregation.
- **Impact:** Reusing nonces in AES-256-GCM violates NIST SP 800-38D guidelines. Single-run benchmarks are sensitive to host jitter.
- **Reproduction:** Inspected loop in `benchmarks/run_v070_benchmarks.py`.
- **Fix:** Implemented NIST SP 800-38D Section 8.2.1 deterministic IV construction with a single 4-byte random session prefix and a monotonically increasing 8-byte counter across all sizes, warmups, and iterations under the benchmark key. Added 3-trial median throughput aggregation for both AEAD and checkpoint pipelines. Removed all unused nonce variables.
- **Regression Tests:** Executed `python benchmarks/run_v070_benchmarks.py`.
- **Status:** `Fixed`
- **Verification:** Benchmarks regenerated with 3-trial medians (Exit code 0).
- **Residual Risk:** None.
- **Regression Test:** Executed `python benchmarks/run_v070_benchmarks.py`.
- **Status:** `Fixed`
- **Verification:** Script executes cleanly, benchmarks complete with zero nonce collisions.
- **Residual Risk:** None.

