# AHCC Mathematical/Security Notes (Preliminary, Not an Independent Cryptanalysis)

This note separates properties that follow from the current equations from security claims that remain unproven. It does not certify AHCC as a secure cipher.

## Properties supported by the implemented equations

### Cubic map is a permutation for the supported primes

The supported moduli satisfy `p ≡ 2 (mod 3)`, so `gcd(3, p - 1) = 1`. Therefore `x -> x^3 mod p` is a permutation of `F_p`, and an exponent `d` satisfying `3*d ≡ 1 (mod p-1)` gives the inverse `y -> y^d mod p`. The implementation and tests check this invertibility.

### Differential equation for the cubic map has at most two solutions

For a nonzero input difference `a`,

`(x + a)^3 - x^3 = 3*a*x^2 + 3*a^2*x + a^3`.

Since the characteristic is neither 2 nor 3 for the supported primes, the right-hand side is a quadratic polynomial in `x`. For a fixed output difference, it has at most two roots in `F_p`. Thus the single-symbol cubic map has differential uniformity at most 2. This is a property of the S-box alone; it is not a proof of the security of the full multi-round cipher.

### The Cauchy layer is invertible under its parameter conditions

The construction chooses distinct `x_i` and `y_j` values with no denominator `x_i + y_j` equal to zero modulo `p`, so it constructs a Cauchy matrix over the field. Tests check the analytical inverse on selected dimensions and both supported primes. This gives an algebraically invertible diffusion layer, not a full cipher security proof.

## Claims that do not follow from those properties

- **Finite-state exact arithmetic is not cryptographic chaos.** Every deterministic map on a finite state space eventually enters a cycle. Moving arithmetic from floating point to a finite field avoids floating-point rounding degradation; it does not establish long periods, unpredictability, or resistance to state recovery.
- **A strong KDF does not make a custom round function secure.** Default HKDF-SHA256 mode uses a high-entropy 256-bit master key for round-key derivation. That addresses the old single-small-integer seed bottleneck, but security of the resulting substitution/permutation network still needs independent analysis. The legacy `master_seed=` mode is retained for reproducibility of the previous code only and is not secure key generation.
- **The PDF's chaotic recurrence is underspecified.** The supplied paper describes using a discretized finite-field chaotic dynamical system for the key schedule, but does not provide a complete recurrence/parameter specification in the described section. The existing code's coefficient-based recurrence is therefore an implementation choice, not evidence of exact paper reproduction. The default HKDF mode intentionally replaces that experimental recurrence; results from the two modes must not be mixed without an explicit specification.
- **MDS diffusion is not a bit-level SAC proof.** The matrix gives strong symbol-level diffusion. The bit avalanche behavior of the complete keyed composition is an empirical diagnostic and needs cryptanalytic analysis.
- **Four rounds have not been justified.** The project has not derived a security bound or enough rounds from differential/linear trail bounds. Do not infer that `R=4` is secure because the avalanche test is near 50%.
- **No authentication is provided by the AHCC engine.** `AHCC_Engine.encrypt()` and `decrypt()` do not use a nonce or authentication tag. Use the separate `AESGCMEnvelope` for authenticated byte encryption, or another vetted AEAD. That does not validate AHCC itself.
- **No GPU performance advantage is established.** Current code uses exact `int64` PyTorch operations and no fused custom CUDA kernel or Tensor Core implementation. Performance claims must be based on a comparable end-to-end benchmark on actual named hardware, including transfer costs and a verified GPU AES-GCM implementation.

## Suggested independent cryptanalysis checklist

1. Freeze a precise specification: state encoding, key/nonce lifecycle, mode, round count, parameter generation and decryption behavior.
2. Prove all inverse operations and field conditions for every supported parameter, including the exact Cauchy inverse formula.
3. Analyze differential and linear trail bounds across the complete rounds; do not rely on random avalanche tests as a substitute.
4. Attempt algebraic/state-recovery attacks against chosen-plaintext data and assess the algebraic degree growth for the composed rounds.
5. Analyze related-key and weak-key behavior, including parameter choices and any key-to-field-element bias.
6. Review all implementation side channels and fault behavior on CPU and GPU.
7. Have independent cryptographers review the design before making any production-security claim.

Until those steps are completed, AHCC remains a research construction rather than a security boundary.
