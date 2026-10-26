# Security Status and Intended Use

**Status: experimental AHCC research transform plus a separate AES-GCM utility. The AHCC transform is not approved for production use.**

## What changed in this revision

- The default AHCC key schedule now accepts a uniformly random 256-bit master key and derives round material using HKDF-SHA256 with parameter and round domain separation. If no key is supplied, a random key is generated and stored on the engine object as `engine.master_key`; callers must protect and persist it securely if needed to reconstruct the engine. Python memory is not zeroized by this package.
- Supplying `master_seed=` explicitly selects the original finite-state recurrence for reproducible legacy experiments. It remains weak and must not be used for real confidentiality.
- `AESGCMEnvelope` provides a **separate** versioned AES-256-GCM envelope with a random 96-bit nonce, authenticated header and optional associated data. `TensorAESGCMEnvelope` stores dense PyTorch tensor bytes plus authenticated dtype/shape metadata inside that AEAD envelope, without pickle. Both APIs are independent of the AHCC transform and do not make AHCC itself secure. They are the recommended APIs in this repository for protecting bytes/tensors, subject to correct key storage and nonce/key lifecycle management.

## AHCC transform: limitations remain

1. **Custom cipher not independently cryptanalyzed.** HKDF improves derivation of round keys, but it does not establish resistance of the custom multi-round SPN to known-plaintext, chosen-plaintext, differential, linear, algebraic, related-key, state-recovery, or side-channel attacks. The cubic power map and Cauchy MDS layer have useful mathematical properties, but component properties do not prove security of the whole cipher.
2. **No AHCC ciphertext authentication.** Direct `AHCC_Engine.encrypt()`/`decrypt()` has no nonce protocol, MAC, or authentication tag. Tampering is not detected by those methods.
3. **Legacy seed mode is insecure for secrets.** The integer seed has low entropy and is passed through a deterministic recurrence. It is retained only for reproducibility and compatibility.
4. **No implementation hardening claim.** Constant-time behavior, GPU side channels, fault resistance, key zeroization, and secure memory handling have not been established.
5. **No encrypted aggregation.** The federated-learning demo decrypts client updates before aggregation; it is not homomorphic encryption or secure aggregation.
6. **No Tensor Core acceleration.** The current engine uses exact `int64` PyTorch tensor operations. There are no fused custom CUDA kernels, and finite-field int64 operations do not map to ordinary Tensor Core matrix multiply instructions.

## AESGCMEnvelope use and constraints

`AESGCMEnvelope` uses a vetted AES-GCM implementation from the `cryptography` package. The envelope format is versioned and includes a fresh random nonce; the tag covers the envelope header, nonce, and caller-provided associated data. A modified envelope, wrong key, or wrong associated data fails closed. AES-GCM is specified by NIST SP 800-38D: <https://csrc.nist.gov/pubs/sp/800/38/d/final>.

Callers must still use a high-entropy key, keep it separate from the encrypted data, avoid excessive message volume under one key, and follow a key-rotation policy. Random nonces are not a substitute for deliberate key lifecycle management. This helper does not implement a secret manager, key wrapping, access control, or key rotation.

## What the tests do and do not prove

Tests check field arithmetic, invertibility and round trips, input validation, deterministic HKDF output from the same key/configuration, random default keys, AES-GCM interoperability, tensor dtype/shape restoration, and detection of tampering/wrong keys/wrong associated data. These are implementation and API checks, not an independent cryptographic security evaluation. Avalanche ratios and statistical summaries are diagnostics only; no claim is made that NIST SP 800-22 or a comprehensive cryptanalysis suite has passed.

## CUDA and performance

Run `python -m benchmarks.check_cuda` on the target CUDA machine. This runtime's PyTorch build is CPU-only, so CUDA execution cannot be certified from this environment. `python -m benchmarks.benchmark_crypto --device cuda` measures AHCC on CUDA if available; unless a separately verified GPU AES-GCM adapter is supplied with `--aesgcm-module`, the AES-GCM side still runs on CPU and the result is not a GPU-vs-GPU comparison. The included adapter benchmark requires ciphertext/tag bytes to match PyCA AES-GCM exactly before timing, but actual device usage must be verified independently.

**Do not use the AHCC transform itself to protect sensitive data until independent cryptanalysis, a complete threat model/specification, and security review have been completed.** For real data in this package, use `AESGCMEnvelope` or another vetted AEAD implementation.
