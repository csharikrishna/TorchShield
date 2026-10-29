# TorchShield v0.7.2 — Security Hardening, Attestation Concurrency & Reconciled Audit Baseline

**Classification:** Internal Staging / Research Candidate (Conditional Approval)
**Target Branch:** `master`
**Release Tag:** `v0.7.2`

## Summary of Security & Architecture Enhancements

### 1. Attestation Cross-Challenge Race Remediation & Bounded Capacity (TS-FIND-19)
- **Cross-Challenge Staging Isolation:** Resolved a critical cross-challenge race in `AttestationGatedKeyProvider.unwrap_key()` where the `finally` block cleared staged evidence unconditionally. Staged evidence is now cleared only if bound to the challenge executing the unwrap operation (`self._staged_evidence[1] == bound_challenge`), preventing operation A from wiping evidence staged by concurrent operation B.
- **Bounded Challenge Lifecycle:** Enforced `DEFAULT_MAX_AUTHORIZED_CHALLENGES = 1024` with fail-closed capacity limits in `create_challenge()`.
- **Opportunistic Cleanup & Active Eviction Protection:** Implemented locked opportunistic cleanup (`_cleanup_expired_challenges_locked`) that purges expired challenges while strictly shielding active in-flight challenges from premature eviction.
- **Attestation Regression Suite:** Added 4 synchronization barrier/event tests covering cross-challenge staging race (`test_attestation_cross_challenge_staging_race`), post-failure evidence preservation and recovery (`test_attestation_verification_failure_preservation_and_recovery`), independent challenge ordering across both completion permutations (`test_attestation_independent_challenge_ordering`), and bounded lifecycle capacity enforcement (`test_attestation_challenge_expiry_and_bounded_lifecycle`). Resolved signature tampering flakiness via bitwise inversion. All 23 tests pass.

### 2. State-Provider Key Lifecycle & Rollback Helper Hardening (TS-FIND-20)
- **Strict Key Length Enforcement:** Enforced strict `provider_keys` length validation (`len(provider_keys) == len(providers)`) in `_snapshot_state_providers` and `_rollback_state_providers`, raising `TrainingStateError` rather than silently re-invoking `get_state_key()` during error recovery.
- **Unified Key Re-use:** Unified validated provider keys across pre-validation, snapshotting, restoration, and rollback in `load_training_checkpoint`.
- **State Provider Regression Tests:** Added 3 tests in `tests/test_phase1_training_state_strict_transactional.py` covering rollback raising, save workflow key contract, and helper length consistency. All 17 tests pass.

### 3. Automated Continuous Integration (CI) Workflow
- Created `.github/workflows/ci.yml` executing on push and PR to `master`.
- Automates full pytest suite with JUnit XML generation, standalone Local Moto KMS verification, sdist/wheel packaging, artifact SHA-256 validation, clean-room isolated wheel installation, and artifact archiving.

### 4. Genuinely Isolated Clean-Room Packaging Validation
- Rewrote `scripts/verify_isolated_wheel.py` to create a pristine virtual environment strictly without `--system-site-packages`.
- Validates full wheel installation with pip dependency resolution, CLI executable version check (`torchshield version`), full checkpoint save/load roundtrip, exact single `get_state_key()` call count, and cryptographic bitflip/wrong-key tamper rejection outside the repository tree.

### 5. Reconciled Test & Verification Baseline (277 Total Checks)
- **270 / 270 Pytest Tests Passed** (0 failures, 0 errors, 0 skips across 49 test files).
- **7 / 7 Standalone Local KMS Checks Passed** via embedded Moto KMS daemon (`scripts/verify_local_kms.py`).
- **Combined Total:** **277 / 277 verified checks** passed with 100% pass rate.

### 6. Explicit Operational Boundaries & Non-GA Declaration
- **Readiness Classification:** **Conditional Approval — Internal Staging / Research Candidate** (Not General Production GA).
- **GPUDirect Storage (GDS):** Pinned host streaming fallback operational; direct cuFile kernel-bypass DMA read APIs remain roadmap items.
- **Hardware Attestation:** Custom signed evidence structure validated; vendor-native TEE hardware quotes (Hopper/Blackwell CC, SEV-SNP) remain roadmap items.
- **Post-Quantum Cryptography (PQC):** Hybrid protocol plumbing and mock testing verified; production deployment requires external audited C library (`liboqs`).
- **Multi-Cloud KMS & Quorum Freshness:** Contract-tested with fail-closed mocks; live services require configured cloud credentials or cluster deployments.
- **Local Ledger Anti-Rollback:** Local SQLite ledger detects local rollback but cannot resist whole-system VM/hypervisor snapshot rollback. High-assurance deployments must utilize `EtcdFreshnessBackend` or cloud consensus.

## Release Assets & Hashes

| Artifact | File Size | SHA-256 Digest |
|---|---|---|
| `torchshield-0.7.2-py3-none-any.whl` | 78,664 bytes | `B2B13F81949D917EDF84DC266FD3B9B534B4BB3DD001F1F5666DBB3FA4BF0544` |
| `torchshield-0.7.2.tar.gz` | 147,993 bytes | `90683158D478211866C8442635A08E3F91F446D815AC934F0C4B96E578109D19` |
| `dist/SHA256SUMS` | 192 bytes | `C2C997F164CEA5D025C226A38A4D7A2D47A6F1DCE2C38F9F84B241EF45AA2B12` |

## Authoritative Audit & Verification Report
For complete mathematical evidence, test logs, finding-by-finding remediation proofs, and build provenance, refer to the canonical report:
- [FINAL_CLOSURE_VERIFICATION_REPORT.md](FINAL_CLOSURE_VERIFICATION_REPORT.md)
