# Changelog

All notable changes to the AHCC GPU Engine and Validated AI Data Protection project are documented in this file.

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
