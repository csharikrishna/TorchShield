# AHCC Codebase Audit Report

**Scope:** Review of the supplied AHCC implementation against the 19-page paper *Verifying the Novelty of Adaptive Hyperdimensional Chaotic Cryptography for GPU-Native AI Security*. This is an implementation/code review, not an independent cryptanalysis or novelty survey.

## Outcome

The codebase now runs as an explicitly labelled research prototype with stronger input validation, edge-case coverage, HKDF-SHA256 key derivation for the default mode, separate AES-256-GCM authenticated byte and tensor-envelope APIs, and benchmark/smoke-test harnesses. Regression tests passed on the available CPU-only PyTorch runtime. The custom AHCC transform itself remains an experimental cipher and is not certified production-secure. Note that the default HKDF-based round-key mode intentionally differs from the paper's stated chaotic-key-schedule idea; the legacy coefficient-based recurrence is retained for reproducibility but the supplied PDF does not fully specify the recurrence needed for exact paper reproduction.

It is **not accurate to call the cryptosystem “perfect” or production-secure**. The paper's mathematical components do not, by themselves, prove the security of the complete cipher. Several design-level security and performance claims remain unverified.

## Corrections and improvements made

1. **Constructor and key safety:** restricted the modulus to the two project-supported primes; validated `N`, bounded `R` to 1–64, checked mutually exclusive/valid key parameters, and retained the dense-matrix cap (`N <= 1024`). The default accepts or generates a 256-bit key and derives round keys with HKDF-SHA256, parameter/round domain separation, and rejection sampling into the finite field. Explicit `master_seed=` selects a clearly labelled legacy recurrence for reproducibility only.
2. **Input contract:** encrypt/decrypt now reject floats, scalars, unsupported ranks, and invalid vector/batch shapes with clear errors rather than failing indirectly or silently truncating floating-point values.
3. **Arithmetic boundaries:** kept the int64 overflow-safe modular multiplication and limb-based matrix multiplication; added boundary tests for the paper's 32-bit prime and inverse-matrix checks at multiple dimensions.
4. **Tensor wrapper:** the wrapper now explicitly accepts only `torch.int64` finite-field elements in `[0, p-1]`; validates dimensions, dtype, metadata version/configuration, padding, ciphertext shape and ciphertext range; and handles empty tensors.
5. **Float quantization:** added validation for non-finite values, positive finite scale, int64 overflow, and signed range ambiguity. Dequantization now validates metadata and field-value bounds.
6. **Test quality:** broadened the single-bit avalanche diagnostic so it does not only flip the least-significant bit, made stochastic checks repeatable, and labelled avalanche, adjacent-seed sensitivity, and mean/variance checks as diagnostics—not proofs of security or randomness.
7. **Demos and benchmarks:** corrected claims that could imply secure aggregation or on-the-fly confidential inference. The demo aggregates after decrypting gradients, and restores all toy model parameters to plaintext before inference. The legacy phase-3 benchmark only checks parity against a small naive Python AHCC reference and reports AHCC throughput; it no longer extrapolates a speedup from that tiny reference. The new benchmark separately compares identical payload bytes against AES-GCM.
8. **Authenticated encryption APIs:** added `AESGCMEnvelope` for bytes and `TensorAESGCMEnvelope` for dense PyTorch tensors. Tensor shape/dtype metadata is encrypted and authenticated; no pickle serialization is used. Both APIs use standard AES-256-GCM and are independent of AHCC.
9. **Runtime/benchmark tools:** added `benchmarks/check_cuda.py` for CPU/CUDA parity and round-trip checks on a target CUDA host, plus `benchmarks/benchmark_crypto.py` for measuring AHCC against AES-GCM. The benchmark supports a user-provided CUDA AES-GCM adapter and verifies its output byte-for-byte against PyCA AES-GCM before timing; without such an adapter AES-GCM runs on CPU.
10. **Packaging and documentation:** updated dependencies, package exports, README, and security notes to state the current limitations and the scope of each test.

## Verification performed

| Check | Result |
|---|---|
| `python -m compileall -q .` | Passed |
| `python -m pytest -q` | **24 passed** |
| `python run_all.py` | Passed all integrated checks and synthetic demos on CPU |
| Invertibility with `p=2147483579` | Passed sample round trips and `M @ M_inv == I mod p` |
| Invertibility with `p=4294967291` | Passed sample round trips and `M @ M_inv == I mod p` |
| High-boundary modular multiplication | Matched Python big-integer reference results |
| Tensor wrapper and quantization edge cases | Passed validation and round-trip checks |
| Editable package import | Passed with `pip install -e . --no-build-isolation --no-deps` |

Environment note: the runtime used Python 3.13 and PyTorch 2.10 CPU-only (`torch.version.cuda is None`; `torch.cuda.is_available() == False`). `python -m benchmarks.check_cuda` reports **SKIP**, so CUDA execution/performance and GPU-specific behavior have not been verified here. The editable install succeeded using locally installed build tooling with build isolation disabled. See `GPU_VALIDATION.md` for the target-hardware checklist.

### Sample CPU benchmark (diagnostic only)

On this CPU-only runtime, with `N=32`, `R=2`, and a 65,536-byte input payload, the median of five timed iterations after one warm-up was approximately **0.923 ms for AHCC (67.69 MiB/s)** and **0.008 ms for PyCA/OpenSSL AES-GCM on CPU (7,839.94 MiB/s)**. This is a single-host CPU comparison, not a GPU-vs-GPU result; timings vary by host and payload size. It does not support a performance advantage claim for the current AHCC implementation. The new benchmark harness is intended to be rerun on the target GPU with a validated GPU AES-GCM adapter.

## Critical remaining limitations

- **Key-schedule design is still a research question:** explicit `master_seed=` selects the old low-entropy recurrence for reproducibility only. The default HKDF-SHA256 mode improves derivation from a 256-bit master key but is not the chaotic schedule described conceptually in the paper. The supplied PDF does not fully specify a recurrence for exact reproduction. Neither mode proves the custom cipher is resistant to cryptanalysis.
- **Direct AHCC mode has no integrity:** `AHCC_Engine.encrypt()`/`decrypt()` has no authentication tag or nonce protocol. Modified ciphertexts are not detected. The separate `AESGCMEnvelope` and `TensorAESGCMEnvelope` helpers provide standard AES-GCM authenticated encryption for bytes/tensors but are not integrated into, and do not validate, the AHCC primitive.
- **No independent cryptanalysis:** no known-plaintext/chosen-plaintext, differential, linear, algebraic, related-key, state-recovery, or implementation side-channel review has been performed. The tests only validate selected correctness properties.
- **No NIST randomness-suite result:** basic mean/variance checks and avalanche ratios are not evidence that ciphertexts are computationally indistinguishable from random data.
- **No measured GPU-AES-GCM comparison in this runtime:** the new harness can run on a target CUDA host and accept a verified adapter; current environment does not expose CUDA and therefore cannot produce a GPU-vs-GPU result. A CPU AES-GCM comparison is not a substitute.
- **No Tensor Core acceleration:** the implementation relies on exact `int64` PyTorch operations and contains no custom fused CUDA kernel. The CUDA smoke test script is included but has not passed on hardware in this environment. CUDA behavior and performance need actual GPU validation.
- **Dense scaling limit:** the dense Cauchy matrix is quadratic in storage and each vector diffusion is quadratic in work. This release caps `N` at 1024, so higher dimensions in the paper are not supported by this implementation.
- **Use-case limits:** the federated-learning demo is ordinary encrypt/decrypt followed by plaintext aggregation; it is not homomorphic or secure aggregation. The model demo is illustrative and does not protect weights while the model runs.

## Recommended next research gate

Before making any production-security or performance claim: freeze a complete threat model and formal specification; commission independent cryptanalysis of the AHCC round function; and benchmark against a validated GPU AES-GCM implementation on named GPU hardware, reporting transfer-inclusive and kernel-only timing separately. The package now provides a standard AES-GCM envelope for practical byte encryption, but this does not validate the custom AHCC cipher. If AHCC is intended to remain a new cipher, independent cryptographic review is a prerequisite—not a post-release optimization.
