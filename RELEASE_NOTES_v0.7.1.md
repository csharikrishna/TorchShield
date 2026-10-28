# TorchShield v0.7.1 — Security Hardening, Attestation Concurrency & Reconciled Audit Baseline

**Classification:** Internal Staging / Research Candidate (Conditional Approval)
**Target Branch:** `master`
**Authoritative Source Commit:** `9897e1c185180d5cf427e60da8d2b078de2adec1`
**Authoritative Tree SHA:** `e08fb736ee10da7932e66c0eb35c946447a85afc`

## Summary of Security & Architecture Enhancements

### 1. Attestation Concurrency Hardening (TS-FIND-18)
- Synchronized `create_challenge`, `stage_evidence`, `unwrap_key`, and `is_production_secure` under an `RLock` in `torchshield/attestation.py`.
- Enforced single-use challenge consumption under concurrency: tracked `_in_flight_challenges` so concurrent threads racing on the same staged evidence are blocked fail-closed (`AttestationError("Challenge is currently in use by another concurrent unwrap operation")`).
- Non-destructive verification failure handling: verified evidence remains accessible for diagnostics upon failure without being irreversibly cleared.
- Added comprehensive multi-threaded synchronization barrier regression tests in `tests/test_hardware_attestation.py` (all 19 tests passing).

### 2. State Provider Key Lifecycle
- Guaranteed `SafeStateProvider.get_state_key()` is invoked strictly once during restoration setup and cached across snapshots, restoration, and transactional rollbacks.
- Rejection of duplicate provider keys with explicit error reporting preserving original exception contexts.

### 3. Reconciled Test & Verification Baseline (270 Total Checks)
- **263 / 263 Pytest Tests Passed** (0 failures, 0 errors, 0 skips) across all unit, integration, and security regression suites.
- **7 / 7 Standalone Local KMS Checks Passed** via embedded Moto KMS daemon (`scripts/verify_local_kms.py`).
- **Genuinely Isolated Clean-Room Packaging Validation Passed**: Full clean-room dependency-resolved installation and execution outside the repository tree (`scripts/verify_isolated_wheel.py`).
- **Empirical Benchmarks Reconciled**: Accurate 3-trial median performance reported from `reports/audit/BENCHMARKS.md`.

### 4. Explicit Operational Boundaries
- **GPUDirect Storage (GDS):** Pinned host streaming fallback operational; direct cuFile kernel-bypass DMA read APIs classified as roadmap items.
- **Hardware Attestation:** Custom signed evidence structure validated; vendor-native TEE hardware quotes (Hopper CC, SEV-SNP) classified as roadmap items.
- **Post-Quantum Cryptography (PQC):** Hybrid protocol plumbing and mock testing verified; production deployment requires external audited C library (`liboqs`).
- **Multi-Cloud KMS & Quorum Freshness:** Contract-tested with fail-closed mocks; live services require configured cloud credentials or cluster deployments.

## Release Assets & Hashes

| Artifact | File Size | SHA-256 Digest |
|---|---|---|
| `torchshield-0.7.1-py3-none-any.whl` | 77,692 bytes | `F764CD9A23632CF7D2320A6D70C3D28DF98BB5AFA4DFB663DEB9E2E0A95CA722` |
| `torchshield-0.7.1.tar.gz` | 143,303 bytes | `E69DC4087AA50273DF41C564FA556BC64DE26A9A0F2B8A8972A8C08794A0A76E` |
| `SHA256SUMS` | 192 bytes (corrected from typographical 162 bytes; verified against GitHub release asset) | `9760A98DBF79653DC5E08ADC8E5D4F0955BC6D2544DC1A5A1AC6D49A413747B9` |
