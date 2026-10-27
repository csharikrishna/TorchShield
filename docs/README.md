# TorchShield Documentation Suite

Welcome to the documentation suite for the **TorchShield Secure AI Checkpoint Platform** (Version 0.7.0).

---

## 1. Documentation Map

The table below catalogs every specification and integration guide:

| Document | Description | Target Audience |
|---|---|---|
| [`INSTALLATION.md`](INSTALLATION.md) | Prerequisites, virtualenv setup, wheel and source installation instructions. | Developers, DevOps, MLOps |
| [`USAGE.md`](USAGE.md) | Quickstart code examples for training checkpoints, inspection, RNG recovery, and CLI. | ML Engineers, Researchers |
| [`HUGGINGFACE_INTEGRATION.md`](HUGGINGFACE_INTEGRATION.md) | Secure, pickle-free Hugging Face Trainer callback and model weight encryption. | NLP & LLM Engineers, MLOps |
| [`TRAINING_CHECKPOINT_SCHEMA.md`](TRAINING_CHECKPOINT_SCHEMA.md) | Wire format specification for v5 encrypted containers, chunk headers, and AAD construction. | Cryptographers, Systems Engineers |
| [`THREAT_MODEL.md`](THREAT_MODEL.md) | Formal threat model, attacker capabilities, defensive boundaries, and non-goals. | Security Architects, Auditors |
| [`ROLLBACK_PROTECTION.md`](ROLLBACK_PROTECTION.md) | Monotonic version sequencing, freshness ledgers, and hypervisor snapshot boundary conditions. | Infrastructure & Security Teams |
| [`KMS_INTEGRATION.md`](KMS_INTEGRATION.md) | Envelope encryption via AWS KMS, Local key providers, and rotation protocols. | Cloud Architects, Security Ops |
| [`DISTRIBUTED_CHECKPOINTS.md`](DISTRIBUTED_CHECKPOINTS.md) | Sharded multi-node and multi-GPU tensor checkpoint coordination (Zero-Redundancy Optimizer / FSDP). | Distributed Training Teams |
| [`ARTIFACT_PROVENANCE.md`](ARTIFACT_PROVENANCE.md) | Ed25519 publisher signatures, RFC 8785 canonical manifests, and transitively signed model parameters. | Model Governance, Compliance |

---

## 2. Core Security Architecture Overview

```
                                    TRAINING STATE
                   [Model Weights | Optimizer | Schedulers | RNG States]
                                           |
                                           v
                          Pickle-Free Safe Tensor Serializer
                                           |
                                           v
                              HKDF-SHA256 Key Derivation
                     [Master Secret + 32-Byte Public Header Salt]
                                           |
                                           v
                         AES-256-GCM Streaming Chunk Encryption
                    [Monotonic Counter Nonces: b"CHNK" + uint64_be]
                                           |
                                           v
                      Canonical JSON Manifest & Ed25519 Signature
                                           |
                                           v
                        Atomic Flush to Disk (.tmp -> .shield)
```

### Key Pillars:
1. **Zero-Pickle Architecture:** Eliminates arbitrary code execution attacks during deserialization.
2. **Authenticated Streaming AEAD:** AES-256-GCM / ChaCha20-Poly1305 with deterministic counter nonces (zero collision risk).
3. **Exact Training Equivalence:** Restores model weights, optimizer momentum, and NumPy/PyTorch/CUDA RNG generators bit-for-bit.
4. **End-to-End Integrity:** Any byte manipulation, truncated chunk, or parameter tampering triggers instant `AuthenticationError`.

---

## 3. Historical Archives

Materials and research prototypes preceding the current secure checkpoint platform have been safely archived under:
- `archives/v1_chaotic_crypto_prototype/`: Original research blueprint, paper cryptanalysis audit, and initial chaotic transform prototype.
- `archives/phase_reports/`: Intermediate verification and audit milestone reports.
- `archives/benchmark_scripts_and_results/`: Historical raw benchmark execution scripts and dataset logs.
