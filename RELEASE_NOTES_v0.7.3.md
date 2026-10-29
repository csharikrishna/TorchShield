# TorchShield v0.7.3 — Artifact Provenance, Hardened CI Gates & Environment Accounting

**Classification:** Internal Staging / Research Candidate (Conditional Approval — Strictly Non-GA)
**Target Branch:** `master`
**Release Tag:** `v0.7.3`

## Summary of Enhancements

TorchShield `v0.7.3` delivers full release-integrity and verification gate closure, addressing release-asset provenance, automated continuous integration gating, clean-room isolated wheel verification, committed-diff validation, and transparent multi-platform test accounting.

### 1. Root Cause Resolution of v0.7.2 Artifact Provenance Discrepancy
- **Forensic Source Audit:** Rigorous byte-level and member-level inspection established that the published `v0.7.2` distribution packages (`torchshield-0.7.2-py3-none-any.whl` and `torchshield-0.7.2.tar.gz`) were built from source revision `edc26670ff86b336bada4a05c19d122afa75ea61` (tree `fe3e23c27e2a5396322c41d21d1e66c25579488c`), containing 100% of the attestation stale-evidence purging and state-provider key consistency code.
- **Line-Ending Divergence:** The artifact hash differences between the local Windows build (`B2B13F81...` wheel, `90683158...` sdist) and the Linux CI build (`D7100AE4...` wheel, `A3F4A9FA...` sdist) resulted from Windows CRLF (`\r\n`) versus Linux LF (`\n`) line endings across 44 text files checked out under `core.autocrlf=true`. Normalized source bytes were identical.
- **Historical Report Correction:** Resolved the documentation error in the v0.7.2 closure report that misidentified parent tree `9f707adc5df9b28dd5b49345ed583e3a66733843` as the build source. Preserved `v0.7.2` release assets and tag history untouched.

### 2. Automated CI Hard Artifact Provenance Gate
Upgraded `.github/workflows/ci.yml` to establish a fail-closed release verification pipeline:
- **Clean Checkout & Committed Diff Check:** Replaced no-op `git diff --check` with `git diff --check HEAD^ HEAD` (with fallback to `git show --check HEAD`) to inspect committed revisions for whitespace errors.
- **5-Point Version Agreement Gate:** Asserts exact version agreement across `pyproject.toml`, `torchshield.__version__`, CLI version output (`torchshield.cli version`), wheel metadata (`dist-info/METADATA`), and sdist metadata (`PKG-INFO`).
- **Clean Build Workspace:** Purges build directories prior to building, guaranteeing that exactly one wheel and one source distribution archive are generated per build.
- **Dynamic Manifest Generation & Hard Verification:** Generates `dist/SHA256SUMS` during the build and verifies artifact bytes using `sha256sum -c SHA256SUMS`, which fails closed on any mismatch.
- **Package Content & Source Tree Diff Verification:** In-pipeline static analysis verifies that critical security fixes (attestation staged evidence purge, narrowed lock, provider key length validation) are present in the package files and matches the repository source tree with zero diffs.
- **Deterministic Clean-Room Execution:** Explicitly invokes `scripts/verify_isolated_wheel.py` with `--wheel` and `--expected-version`, preventing silent fallback to older wheels.
- **Metadata Archival:** Generates and uploads machine-readable build environment metadata, JUnit test reports, distribution archives, and SHA-256 manifests.

### 3. Transparent Cross-Platform Test Accounting
- **Local Windows Environment (NVIDIA GTX 1650, CUDA 12.1, Python 3.12.10, `transformers` installed):**
  - **270 / 270 Pytest Tests Passed** (0 failures, 0 errors, 0 skips across 49 test files).
  - **7 / 7 Standalone Local KMS Checks Passed** via embedded Moto KMS daemon.
  - **Total:** **277 / 277 verified checks passed (100% pass rate)**.
- **GitHub Actions Ubuntu Runner (Ubuntu 24.04, Python 3.12, CPU runner, no `transformers` installed):**
  - **261 passed, 9 skipped, 4 warnings** in automated Pytest suite.
  - **7 / 7 Standalone Local KMS Checks Passed**.
  - **Documented 9 Skips:**
    - 5 Hugging Face integration tests skipped because `transformers` is an optional dependency not installed on the CI runner (`test_callback_key_validation`, `test_callback_on_save_workflow`, `test_resume_from_hf_vault`, `test_tamper_detection_in_hf_vault`, `test_trainer_fail_closed_on_missing_vault`).
    - 4 CUDA correctness tests skipped because GitHub-hosted runners lack NVIDIA GPU hardware (`test_restore_optimizer_on_cuda_when_available`, `test_direct_cuda_tensor_loading`, `test_cuda_model_streaming_and_inference`, `test_torch_cuda_rng_exact_reproducibility`).

### 4. Canonical Operational Boundaries (Non-GA Declaration)
- **Deployment Classification:** Conditional Approval — Internal Staging / Research Candidate (`v0.7.3`). Strictly Non-GA.
- **GPUDirect Storage (GDS):** Pinned host streaming fallback operational; direct cuFile kernel-bypass DMA read APIs remain roadmap items.
- **Hardware Attestation:** Custom signed evidence format validated; vendor-native TEE hardware quotes (Hopper/Blackwell CC, SEV-SNP) remain roadmap items.
- **Post-Quantum Cryptography (PQC):** Hybrid protocol plumbing and mock testing verified; production deployment requires external audited C library (`liboqs`).
- **Multi-Cloud KMS & Quorum Freshness:** Contract-tested with fail-closed mocks; live services require configured cloud credentials or cluster deployments.
- **Local Ledger Anti-Rollback:** Local SQLite ledger detects local rollback but cannot resist whole-system VM/hypervisor snapshot rollback. High-assurance deployments must utilize `EtcdFreshnessBackend` or cloud consensus.
- **Git Signature Status:** Git commits and tags are unsigned (no developer GPG key); authenticity is established via immutable commit/tree SHAs, CI provenance, and published SHA-256 checksums.

## Release Assets & Hashes

*(Hashes and byte sizes are populated directly from the artifacts produced and verified by GitHub Actions CI)*

| Artifact | File Size | SHA-256 Digest |
|---|---|---|
| `torchshield-0.7.3-py3-none-any.whl` | TBD | TBD |
| `torchshield-0.7.3.tar.gz` | TBD | TBD |
| `dist/SHA256SUMS` | TBD | TBD |

## Authoritative Audit & Verification Report
For complete mathematical evidence, test logs, finding-by-finding remediation proofs, and build provenance, refer to the canonical report:
- [FINAL_CLOSURE_VERIFICATION_REPORT.md](FINAL_CLOSURE_VERIFICATION_REPORT.md)
