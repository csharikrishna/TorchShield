# Changelog

All notable changes to the TorchShield GPU Engine and Validated AI Data Protection project are documented in this file.

## [0.7.2] - 2026-10-28

### Security Hardening & Concurrency Defect Remediation
- **Attestation Cross-Challenge Race Remediation (TS-FIND-19):** Resolved critical cross-challenge race condition in `AttestationGatedKeyProvider.unwrap_key()` where the `finally` block cleared staged evidence unconditionally. Staged evidence is now cleared only if bound to the executing challenge (`self._staged_evidence[1] == bound_challenge`), preventing operation A from wiping evidence staged by concurrent operation B.
- **Attestation Bounded Challenge Capacity & Lifecycle (TS-FIND-19):** Enforced `DEFAULT_MAX_AUTHORIZED_CHALLENGES = 1024` with fail-closed capacity limits on `create_challenge()`. Implemented locked opportunistic cleanup (`_cleanup_expired_challenges_locked`) that purges expired challenges while strictly preserving active in-flight challenges from premature eviction.
- **Hardware Attestation Deterministic Regression Suite:** Added 4 synchronization barrier/event tests covering cross-challenge staging race (`test_attestation_cross_challenge_staging_race`), post-failure evidence preservation and recovery (`test_attestation_verification_failure_preservation_and_recovery`), independent challenge ordering across both completion permutations (`test_attestation_independent_challenge_ordering`), and bounded lifecycle capacity enforcement (`test_attestation_challenge_expiry_and_bounded_lifecycle`). Resolved signature bit-tampering flakiness via bitwise inversion. All 23 attestation tests pass.
- **State-Provider Key Lifecycle & Rollback Hardening (TS-FIND-20):** Enforced strict provider key length validation (`len(provider_keys) == len(providers)`) in `_snapshot_state_providers` and `_rollback_state_providers`, raising `TrainingStateError` and guaranteeing callers never trigger repeated or fallback `get_state_key()` calls during failure recovery. Unified validated provider keys across pre-validation, snapshotting, restoration, and rollback in `load_training_checkpoint`. Added 3 regression tests in `tests/test_phase1_training_state_strict_transactional.py`. All 17 training state tests pass.
- **Automated CI Workflow:** Created `.github/workflows/ci.yml` running on push and pull request to `master` with pytest execution, standalone Local Moto KMS verification, wheel and source distribution packaging, artifact SHA-256 verification, clean-room isolated wheel installation, and artifact archiving.
- **Isolated Clean-Room Wheel Verification:** Rewrote `scripts/verify_isolated_wheel.py` to create a pristine virtual environment strictly without `--system-site-packages`, verifying full wheel installation with pip dependency resolution, CLI executable version output, checkpoint save/load roundtrip, exact single `get_state_key()` invocation, and cryptographic bitflip/wrong-key tamper rejection outside the repository root.
- **Reconciled Audit Baseline:** Reconciled test accounting to 270 passed pytest tests across 49 test files + 7 standalone Local KMS checks = 277 total verification checks (100% pass rate).

## [0.7.1] - 2026-10-28

### Security Hardening & Concurrency Protection
- **Thread-Safe Attestation Challenge Lifecycle:** Enforced synchronized challenge creation, staging, and single-use consumption in `AttestationGatedKeyProvider`. Prevented concurrent `unwrap_key` operations from reusing in-flight or consumed staged evidence, verified by synchronization barrier regression tests.
- **Fail-Closed Evidence Preservation:** Preserved staged evidence for post-mortem diagnostics upon verification failures without permitting unauthorized key unwrap.
- **Provider-Key Setup Caching:** Pre-validated `SafeStateProvider.get_state_key()` once during setup, eliminating redundant invocations and ensuring stable key mapping across snapshots, restorations, and transactional rollbacks.
- **Canonical JSON v1 Specification:** Formally designated metadata and manifest serialization format as deterministic `TorchShield Canonical JSON v1`.
- **Reconciled Audit Evidence:** Synchronized benchmark metrics to empirical 3-trial medians, reclassified advanced workstreams with explicit reference/prototype boundaries, and established authoritative build provenance.

## [0.6.0-phase5-final] - 2026-10-26

### Safe, Pickle-Free AI Training State Engine
- **Complete Training State Protection:** Created `torchshield/training_state.py` capturing model parameters, non-trainable buffers (BatchNorm running_mean/var), optimizer moment buffers (`exp_avg`, `exp_avg_sq`), learning rate schedulers, mixed-precision gradient scalers (`GradScaler`), and complete RNG states.
- **Zero-Pickle Serialization Contract:** Eliminated Python `pickle` completely. Numerical weights are stored as dense raw binary buffers; metadata and hyperparameters are stored as allowlisted canonical JSON (`TorchShield Canonical JSON v1`). Deserialization involves zero executable code.
- **Stable Parameter Mapping:** Mapped transient PyTorch integer parameter pointers to stable canonical model parameter names (`model.named_parameters()`), guaranteeing 100% portable state restoration across separate process invocations.
- **Full Deterministic RNG Preservation:** Implemented safe non-pickle serializers/deserializers for Python `random` (625 uint32 keys), NumPy `np.random` (MT19937 state array), PyTorch CPU RNG (`torch.get_rng_state()`), and PyTorch CUDA RNG per GPU device.
- **Bit-for-Bit Deterministic Resume:** Proved exact numerical equivalence between an uninterrupted 30-step reference training run and an interrupted, checkpointed, restarted, and resumed run (`tests/test_training_resume_equivalence.py`, `examples/resumable_training_demo.py`).

### Enterprise Key Management Service (KMS) & Envelope Encryption
- **Pluggable `KeyProvider` Architecture:** Created `torchshield/key_providers/base.py` defining the client-side envelope encryption interface.
- **AWS KMS Provider:** Implemented `AWSKMSProvider` utilizing AWS KMS client-side envelope encryption (`GenerateDataKey`, `Decrypt`, `ReEncrypt`) with authenticated encryption context validation.
- **Local Dev Key Provider:** Implemented `LocalDevKeyProvider` with AES-256-GCM envelope wrapping for unit testing and offline development.
- **Safe KEK Rotation:** Validated zero-payload re-encryption of Data Encryption Keys without re-encrypting multi-gigabyte tensor payloads.

### Publisher Provenance & Monotonic Anti-Rollback
- **Ed25519 Digital Signatures:** Implemented `torchshield/provenance.py` providing RFC 8032 Ed25519 digital signature signing and verification over canonical manifest specifications.
- **Transitive Parameter Protection:** Since the manifest specification digest is bound into every chunk's AEAD Associated Authenticated Data (AAD), modifying even a single float32 weight invalidates both the AEAD tag and the Ed25519 signature.
- **Monotonic Freshness Ledger:** Implemented `FreshnessLedger` tracking `(run_id, max_step, max_epoch)` with atomic staging and fsync persistence. Defensively detects and blocks model rollback / downgrade attacks.

### Secure Distributed & Sharded Checkpointing
- **Multi-Rank Parallel Saving:** Created `torchshield/distributed.py` enabling $W$ ranks to save individual shards (`shard_{rank:05d}_of_{world_size:05d}.shield`) simultaneously with $O(1)$ memory scaling per node.
- **Global Inventory Manifest:** Aggregates all $W$ shard file sizes and SHA-256 digests into a single coordinator-signed `global_manifest.json`.
- **Completeness Enforcement:** Fails closed if any shard is missing, corrupted, or has an unexpected SHA-256 hash before loading weights into memory.

### Container Version 5 Format & CLI Tooling
- **Version 5 Container Header (`b"TSHDMOD5"`):** Extends container header with wrapped DEK length and ciphertext blob, eliminating circular key dependencies.
- **TorchShield Management CLI (`torchshield/cli.py`):** Added commands `torchshield inspect`, `torchshield verify-distributed`, and `torchshield version` for offline inspection without exposing decryption keys.
- **Expanded Test Suite:** 26 new Phase 5 tests (total suite expanded to **212 passing tests**).

## [0.5.0-phase4-final] - 2026-10-26

### Cryptographic Protocol Hardening & Independent Verification
- **Corrected Nonce Design:** Formally replaced truncated hashing claims with a deterministic, auditable 96-bit sequence counter (`b"CHNK" + uint64_be`), eliminating birthday bound collision probabilities within single-key lifetimes under NIST SP 800-38D.
- **RFC 5869 HKDF Domain Separation:** Derived independent 256-bit keys for manifest encryption (`manifest_key`) and payload chunk encryption (`chunk_key`) using distinct context strings.
- **Dedicated Manifest Nonce Domain:** Assigned dedicated manifest nonce (`b"MANI" + 8 zero bytes`) under the isolated manifest key.
- **Canonical Manifest Specification Binding:** Decomposed the manifest into a Canonical Manifest Specification ($\mathcal{M}_{\text{spec}}$) and container manifest, verifying `SHA-256(canonical_json(M_spec))` upon load before decrypting any chunk payload. Eliminates circular dependencies.
- **Strict Canonical JSON & Duplicate Key Rejection:** Added `_strict_json_loads` parser rejecting duplicate JSON keys, non-object roots, and ambiguous encodings.
- **Controlled Crypto Session:** Introduced internal `_ChunkCryptoSession` encapsulating sequence generation and sealing to eliminate explicit nonce handling from public and package APIs.

### Resource Ceilings & DoS Prevention
- **Configurable ResourceLimits:** Introduced immutable `ResourceLimits` configuration dataclass controlling maximum file size (128 GiB), raw bytes (64 GiB), single tensor bytes (16 GiB), manifest length (16 MiB), tensor count (50,000), tensor rank (8), chunk count (1,000,000), chunk bytes (64 MiB), and dimension elements (100,000,000).
- **Pre-Allocation Bounds Validation:** Verified dimension products and element item sizes against bounds *before* allocating buffer memory, preventing crafted header DoS attacks.

### Performance Optimization & Streaming DMA
- **Fine-Grained Latency Profile:** Benchmark suite (`benchmarks/benchmark_phase4_checkpoint.py`) instrumented every pipeline step. Reduced 25.09 MiB Transformer CUDA loading latency to **131.54 ms** (55.0% reduction vs Phase 2 baseline) and time to first inference to **138.06 ms** (56.5% reduction vs Phase 2).
- **Pinned Host-to-Device Streaming:** Validated memory-sharing semantics of `torch.frombuffer` and non-blocking pinned DMA staging.

### Test Suite Expansion & Durability
- **Added Phase 4 Test Suites:** 24 new tests across `test_phase4_nonce_protocol.py`, `test_phase4_manifest_protocol.py`, `test_phase4_resource_limits.py`, `test_phase4_crash_recovery.py`, and `test_phase4_public_api_and_integrity.py` (total test suite expanded to **186 passing tests**).
- **Crash & Durability Verification:** Validated atomic file staging (`.tmp.<uuid>`), filesystem sync (`os.fsync`), atomic replacement (`os.replace`), and automatic cleanup on failure.

## [0.4.0-phase3-final] - 2026-10-26

### Security Hardening & Protocol Verification
- **Eliminated Public `explicit_nonce`:** Removed custom nonce parameter from the public `AESGCMEnvelope.encrypt` API to prevent accidental nonce reuse. Nonce management is now fully automatic (CSPRNG 96-bit). Deterministic chunk nonces are restricted to an internal package-private helper (`_encrypt_with_explicit_nonce`).
- **Defensive Resource Ceilings:** Tightened resource boundaries: max tensor count 50,000, max tensor rank 8, max manifest length 10 MiB, max dimension size $2^{31}-1$. Overflow-safe dimension multiplication prevents integer overflow attacks.
- **Sustained Property-Based Fuzzing:** Expanded fuzz testing (`tests/fuzz_phase2_envelope.py`) to 200+ sustained randomized mutations and semantic JSON vectors. Verified 100% fail-closed rejection without crashes or leaks.

### Performance Optimizations
- **Direct-to-Buffer Reconstruction:** Eliminated 3 redundant buffer copies during model loading by preallocating layer bytearrays and writing decrypted chunks directly into place.
- **Pinned Host-to-Device Streaming:** Enabled non-blocking pinned DMA transfers when loading to CUDA devices.
- **Measured Results:** Reduced model loading latency from 292.59 ms to 156.17 ms (46.6% latency reduction / 1.87x speedup on 25.09 MiB Transformer). Time to first inference reduced from 317.35 ms to 170.24 ms.

## [0.3.0-phase2-rc1] - 2026-10-26

### Security & Cryptographic Remediation
- **Cryptanalysis Correction:** Formally corrected the univariate interpolation claim in `reports/PHASE2_CIPHER_REASSESSMENT.md`. Proved that an $N=64$ dense SPN contains $\approx 1.25 \times 10^{41}$ multivariate monomials, refuting the claim of a $<100$-query break while confirming that $R=4$ rounds remain inadequate for algebraic security.
- **Strict Insecurity Warnings:** Instantiation of `TorchShield_Engine` or `TensorCipherWrapper` emits a runtime `InsecureCipherWarning` and flags `is_production_secure = False`. The production checkpoint API cannot invoke raw TorchShield under any circumstances.
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
- **Production Checkpoint System (`torchshield/checkpoint.py`):** Chunked and layer-wise streaming AEAD model checkpoint serialization backed by AES-256-GCM. Authenticated manifests bind file ID, layer names, chunk sequence, and tensor dimensions.
- **Formal Mathematical Reference Tests (`tests/test_arithmetic_reference.py`, `tests/test_cipher_reference.py`):** 34 tests validating finite-field arithmetic, power-map S-box bijectivity, and Cauchy MDS inversion against independent arbitrary-precision Python math.
- **Cryptanalytic Attack Demonstrations (`tests/attack_experiments.py`):** Empirical demonstrations of legacy chaotic key schedule collapse ($\le 32$ bits), quadratic recurrence reversibility, unauthenticated ciphertext malleability, and deterministic ECB block leakage.
- **Official Cryptographic KATs (`tests/test_aead_known_answer.py`, `tests/test_aead_tampering.py`):** RFC 5869 HKDF-SHA256 and NIST SP 800-38D AES-256-GCM known-answer test vectors and full negative tampering suites.
- **Comprehensive CUDA Parity Matrix (`tests/test_cuda_correctness.py`, `benchmarks/run_gpu_validation.py`):** 56 tests validating bit-exact numerical parity between CPU and CUDA across all supported prime moduli, dimensions ($N=4, 16, 64$), rounds ($R=1, 2, 4$), and batch sizes.
- **Reproducible Benchmarks (`benchmarks/benchmark_cipher.py`, `benchmarks/benchmark_tensor_loading.py`):** Controlled harnesses measuring encryption/decryption latency, throughput, and memory scaling across $4\text{ KiB}$ to $4\text{ MiB}$ payloads.
- **Analysis Pipeline (`analysis/analyze_results.py`, `analysis/plot_results.py`):** Automated statistical ratio analysis and publication-grade throughput chart generation (`results/throughput_comparison.png`).
- **Research Documentation:** Full reports for Phases 0 through 12, including threat model, mathematical correctness, cryptographic audit, CUDA validation, prior art review, federated learning audit, and final verification.

### Changed
- Refactored project classification: TorchShield cipher labelled strictly as an uncertified academic research prototype.
- Standard AES-256-GCM and chunked streaming checkpoint system adopted as the deployable path for AI data protection.
