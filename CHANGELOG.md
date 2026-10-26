# Changelog

All notable changes to the AHCC GPU Engine and Validated AI Data Protection project are documented in this file.

## [0.3.0-phase2-rc1] - 2026-10-26

### Security & Cryptographic Remediation
- **Cryptanalysis Correction:** Formally corrected the univariate interpolation claim in `reports/PHASE2_CIPHER_REASSESSMENT.md`. Proved that an $N=64$ dense SPN contains $\approx 1.25 \times 10^{41}$ multivariate monomials, refuting the claim of a $<100$-query break while confirming that $R=4$ rounds remain inadequate for algebraic security.
- **Strict Insecurity Warnings:** Instantiation of `AHCC_Engine` or `TensorCipherWrapper` emits a runtime `InsecureCipherWarning` and flags `is_production_secure = False`. The production checkpoint API cannot invoke raw AHCC under any circumstances.
- **Per-Artifact HKDF-SHA256 Subkey Isolation:** Each checkpoint generation derives an isolated 256-bit AES-GCM subkey using RFC 5869 HKDF-SHA256 with 128-bit random salt and domain separation.
- **Deterministic 96-bit Chunk Nonce Policy:** NIST SP 800-38D compliant monotonic chunk nonces eliminate random nonce collision risks. Nonce 0 is reserved exclusively for the authenticated manifest.
- **Atomic File Operations & Crash Resilience:** Checkpoints write to temporary staging files (`.tmp.<uuid>`), flush with `os.fsync()`, and perform atomic `os.replace()`. Unfinished or crashed writes safely clean up without corrupting prior valid checkpoints.
- **Defensive Tensor Deserialization:** Zero pickle dependency. Enforces strict limits on tensor count ($500,000$), rank ($16$), dimension size ($2^{48}$), and memory allocation ($100\text{ GiB}$) to prevent host OOM denial of service.

### Added Tests & Benchmarks
- **Phase 2 Test Suites:** Added 27 new tests across `test_phase2_aead_security.py`, `test_phase2_nonce_policy.py`, `test_phase2_manifest_integrity.py`, `test_phase2_chunk_integrity.py`, `test_phase2_atomic_write.py`, `test_phase2_tensor_validation.py`, `test_phase2_pytorch_integration.py`, `test_phase2_cuda_correctness.py`, `test_phase2_fault_injection.py`, and `fuzz_phase2_envelope.py` (total test suite expanded to 160 passing tests).
- **Comprehensive Hardware Validation:** Physical CUDA verification script `benchmarks/validate_phase2_cuda.py` verifying VRAM residency and device parity on NVIDIA GeForce GTX 1650.
- **Pipeline Profiling & Benchmarking:** Detailed latency and throughput profiling in `benchmarks/benchmark_checkpoint_pipeline.py` and standardized backend comparison in `benchmarks/benchmark_aead_backends.py`.

## [2.0.0-audited] - 2026-10-26

### Added
- **Production Checkpoint System (`ahcc/checkpoint.py`):** Chunked and layer-wise streaming AEAD model checkpoint serialization backed by AES-256-GCM. Authenticated manifests bind file ID, layer names, chunk sequence, and tensor dimensions.
- **Formal Mathematical Reference Tests (`tests/test_arithmetic_reference.py`, `tests/test_cipher_reference.py`):** 34 tests validating finite-field arithmetic, power-map S-box bijectivity, and Cauchy MDS inversion against independent arbitrary-precision Python math.
- **Cryptanalytic Attack Demonstrations (`tests/attack_experiments.py`):** Empirical demonstrations of legacy chaotic key schedule collapse ($\le 32$ bits), quadratic recurrence reversibility, unauthenticated ciphertext malleability, and deterministic ECB block leakage.
- **Official Cryptographic KATs (`tests/test_aead_known_answer.py`, `tests/test_aead_tampering.py`):** RFC 5869 HKDF-SHA256 and NIST SP 800-38D AES-256-GCM known-answer test vectors and full negative tampering suites.
- **Comprehensive CUDA Parity Matrix (`tests/test_cuda_correctness.py`, `benchmarks/run_gpu_validation.py`):** 56 tests validating bit-exact numerical parity between CPU and CUDA across all supported prime moduli, dimensions ($N=4, 16, 64$), rounds ($R=1, 2, 4$), and batch sizes.
- **Reproducible Benchmarks (`benchmarks/benchmark_cipher.py`, `benchmarks/benchmark_tensor_loading.py`):** Controlled harnesses measuring encryption/decryption latency, throughput, and memory scaling across $4\text{ KiB}$ to $4\text{ MiB}$ payloads.
- **Analysis Pipeline (`analysis/analyze_results.py`, `analysis/plot_results.py`):** Automated statistical ratio analysis and publication-grade throughput chart generation (`results/throughput_comparison.png`).
- **Research Documentation:** Full reports for Phases 0 through 12, including threat model, mathematical correctness, cryptographic audit, CUDA validation, prior art review, federated learning audit, and final verification.

### Changed
- Refactored project classification: AHCC cipher labelled strictly as an uncertified academic research prototype.
- Standard AES-256-GCM and chunked streaming checkpoint system adopted as the deployable path for AI data protection.
