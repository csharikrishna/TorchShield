# AHCC GPU-Native Engine (Research Prototype)

This repository implements an **experimental** Adaptive Hyperdimensional Chaotic Cryptography (AHCC) construction from the accompanying research blueprint. It provides exact integer arithmetic modulo a supported prime, a cubic permutation, a dense Cauchy diffusion matrix, an experimental round-key schedule, and a tensor wrapper for values already represented in the finite field.

> **Security warning:** The AHCC transform is still not a production-ready cipher. The default mode now derives round keys from 256-bit key material using HKDF-SHA256; the explicit `master_seed=` compatibility mode retains a low-entropy experimental recurrence. But neither mode authenticates AHCC ciphertext, and the complete custom construction has not undergone independent cryptanalysis. Tests do **not** establish cryptographic security. Do not use direct AHCC `encrypt()` or `decrypt()` to protect sensitive assets. For real confidentiality and integrity, use the separate `AESGCMEnvelope` API or another vetted AEAD scheme.

## What is implemented

- Finite-field arithmetic using `torch.int64` and exact modular reduction.
- Supported moduli: `2147483579` (31-bit variant) and `4294967291` (`2**32 - 5`, the paper's target).
- Bijective cubic map `S(x) = x^3 mod p` and its modular-exponent inverse.
- Dense Cauchy matrix `M[i,j] = inverse(i + N + j) mod p` and analytical inverse.
- Multi-limb multiplication to avoid signed-int64 overflow for the two supported moduli.
- Default 256-bit master-key handling and HKDF-SHA256 round-key derivation with parameter/round domain separation; legacy integer seed mode is retained only for repeatable research experiments. The default HKDF variant is **not the PDF's chaotic-key-schedule variant**. The PDF describes a discretized chaotic schedule but does not fully specify an exact recurrence, so the legacy recurrence should not be represented as an exact reproduction of the paper without a complete formal specification.
- Separate `AESGCMEnvelope` API backed by AES-256-GCM, with random nonces, a versioned envelope, associated-data authentication, and fail-closed tamper detection. `TensorAESGCMEnvelope` applies the same standard AEAD to dense PyTorch tensors and authenticates dtype/shape metadata inside the encrypted payload, without using pickle. These APIs are independent of the AHCC cipher.
- 1D vector and 2D batch encrypt/decrypt round trips, plus an explicit wrapper for `torch.int64` tensors of finite-field elements.
- Parameter, dtype, shape, metadata, and empty-tensor validation.
- Diagnostic tests, workload timing, and explanatory demonstrations.

## Known limits

- The dense Cauchy matrix and inverse consume `O(N^2)` storage and each state-vector diffusion costs `O(N^2)`. This implementation deliberately caps `N` at 1024 to fail safely instead of risking accidental multi-gigabyte allocations. Large dimensions mentioned as future targets in the paper are **not** implemented efficiently here.
- The implementation uses exact `int64` PyTorch operations; it does **not** use NVIDIA Tensor Cores or a custom fused CUDA kernel for finite-field arithmetic. CUDA behavior and performance must be measured on actual supported hardware.
- `TensorCipherWrapper` accepts only `torch.int64` inputs whose values are already in `[0, p-1]`. Float tensors must be explicitly quantized first. Its metadata is a runtime Python dictionary, not an authenticated or portable ciphertext format.
- The demos use a shared experimental engine to encrypt/decrypt and then aggregate plaintext updates. They do not implement homomorphic encrypted aggregation, secure aggregation among untrusted clients, or a production key-management protocol.
- The model-weight demonstration decrypts parameter tensors sequentially, but ultimately restores all model parameters to plaintext memory before inference. It does not provide a protected execution environment or prove lower model-loading overhead.

## Requirements

- Python 3.10+ recommended
- PyTorch 2.x (`torch>=2.0.0`)
- `cryptography>=42.0` for HKDF-SHA256 derivation and AES-GCM envelopes

CPU mode is supported. If CUDA is installed and available, it is selected by default; use `device="cpu"` to force CPU execution.

## Quick start

From the `AHCC_GPU_Engine` directory:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Run the tests with pytest (optional):

```bash
python -m pip install pytest
python -m pytest -q
```

Run the AHCC-vs-AES-GCM benchmark on CPU (no GPU-AES-GCM claim):

```bash
python -m benchmarks.benchmark_crypto --device cpu
```

On a CUDA-enabled target host, first check CUDA execution:

```bash
python -m benchmarks.check_cuda
```

Then benchmark AHCC on CUDA. Without `--aesgcm-module`, AES-GCM still runs through CPU OpenSSL/PyCA and this is **not** a GPU-vs-GPU comparison. For an apples-to-apples GPU comparison, provide a verified Python adapter module exposing `BACKEND_NAME`, `DEVICE = "cuda"`, and `encrypt(plaintext, key, nonce, associated_data) -> ciphertext_and_tag`; the harness verifies the adapter output against PyCA AES-GCM before timing. The adapter must be independently checked to ensure it really runs on GPU. Timings alone do not establish security.

## Minimal use

```python
import secrets
import torch
from ahcc import AHCC_Engine, TensorCipherWrapper

# Keep this key in a secret store if it must survive process exit.
master_key = secrets.token_bytes(32)
engine = AHCC_Engine(N=64, R=4, p=2_147_483_579, master_key=master_key, device="cpu")
field_values = torch.randint(0, engine.p, (2, 1000), dtype=torch.int64)
wrapper = TensorCipherWrapper(engine)
ciphertext, metadata = wrapper.encrypt_tensor(field_values)
recovered = wrapper.decrypt_tensor(ciphertext, metadata)
assert torch.equal(field_values, recovered)
```

This example demonstrates AHCC round-trip correctness only. Direct AHCC encryption has no authentication and the custom cipher has not been proven secure. For sensitive bytes, use the separate standard AES-GCM envelope:

### Production Model Checkpoint Encryption

For production AI checkpoint protection, use `SecureModelCheckpoint` (`ahcc/checkpoint.py`), which provides chunked authenticated streaming with tamper detection:

```python
from ahcc.checkpoint import SecureModelCheckpoint
from ahcc.secure import AESGCMEnvelope

key = AESGCMEnvelope.generate_key()
# Encrypts model with authenticated manifest and per-chunk AES-256-GCM
SecureModelCheckpoint.save_checkpoint(model.state_dict(), "model.ahcc_ckpt", key)

# Stream decrypts layer-by-layer directly into GPU VRAM
state_dict = SecureModelCheckpoint.load_checkpoint("model.ahcc_ckpt", key, device="cuda")
model.load_state_dict(state_dict)
```

### End-to-End Secure Training State Checkpoint & Recovery (v0.7.0+)

Save, inspect, and resume complete training state (model parameters, BatchNorm buffers, AdamW/SGD optimizer moments, learning rate schedulers, GradScaler, and all RNG states) with zero pickle and authenticated AES-256-GCM encryption directly via top-level imports:

```python
import torch
import torch.nn as nn
from ahcc import (
    save_training_checkpoint,
    load_training_checkpoint,
    inspect_training_checkpoint,
    TrainingProgress,
    AESGCMEnvelope,
)

# 1. Initialize model, optimizer, scheduler
model = nn.Sequential(nn.Linear(10, 4), nn.ReLU(), nn.Linear(4, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
key = AESGCMEnvelope.generate_key()

# 2. Save complete training state mid-epoch
progress = TrainingProgress(global_step=100, epoch=2, batch_idx=14, loss=0.42)
save_training_checkpoint(
    output_path="checkpoint.ahcc",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    progress=progress,
    key=key,
)

# 3. Inspect checkpoint header without keys (zero-key header inspection)
header = inspect_training_checkpoint("checkpoint.ahcc")
print(header["format"], header["file_size"])  # AHCC-SafeTrainingCheckpoint-v5

# 4. Authenticate manifest metadata with key (no tensor weights loaded)
manifest = inspect_training_checkpoint("checkpoint.ahcc", key=key)
print(f"Step: {manifest['global_step']}, Loss: {manifest['loss']}")

# 5. Restore into fresh process
model_resumed = nn.Sequential(nn.Linear(10, 4), nn.ReLU(), nn.Linear(4, 2))
optimizer_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)
scheduler_resumed = torch.optim.lr_scheduler.StepLR(optimizer_resumed, step_size=5)

restored_prog = load_training_checkpoint(
    input_path="checkpoint.ahcc",
    model=model_resumed,
    optimizer=optimizer_resumed,
    scheduler=scheduler_resumed,
    key=key,
)
print(f"Resumed at Step {restored_prog.global_step}, Epoch {restored_prog.epoch}, Batch {restored_prog.batch_idx}")
```

## Research Audit Deliverables & Reports

Full reproducible reports and research audits are available in `reports/`:
- [`BASELINE_AUDIT.md`](reports/BASELINE_AUDIT.md): Initial snapshot and baseline verification.
- [`MATHEMATICAL_CORRECTNESS.md`](reports/MATHEMATICAL_CORRECTNESS.md): Finite field arithmetic, S-box algebra, and Cauchy MDS verification.
- [`THREAT_MODEL.md`](reports/THREAT_MODEL.md): AI data protection threat model and security bounds.
- [`CRYPTOGRAPHIC_SECURITY_AUDIT.md`](reports/CRYPTOGRAPHIC_SECURITY_AUDIT.md): Detailed cryptanalytic findings and attack analysis.
- [`ATTACK_EXPERIMENTS.md`](reports/ATTACK_EXPERIMENTS.md): Empirical verification of key collapse, reversibility, and malleability.
- [`STANDARD_AEAD_VALIDATION.md`](reports/STANDARD_AEAD_VALIDATION.md): NIST SP 800-38D and RFC 5869 KAT verification.
- [`CUDA_VALIDATION.md`](reports/CUDA_VALIDATION.md): Hardware validation on NVIDIA GTX 1650.
- [`PERFORMANCE_EVALUATION.md`](reports/PERFORMANCE_EVALUATION.md): Fair benchmark comparing AHCC against AES-256-GCM.
- [`STATISTICAL_ANALYSIS.md`](reports/STATISTICAL_ANALYSIS.md): Confidence intervals, speedup ratios, and hypothesis tests.
- [`FEDERATED_LEARNING_EVALUATION.md`](reports/FEDERATED_LEARNING_EVALUATION.md): Analysis of secure aggregation vs in-transit encryption.
- [`PRIOR_ART_REVIEW.md`](reports/PRIOR_ART_REVIEW.md): Literature comparison against MiMC, Poseidon, and GPU-AES.
- [`RESEARCH_HYPOTHESES.md`](reports/RESEARCH_HYPOTHESES.md): Pre-registered hypotheses and empirical falsification.
- [`FINAL_VERIFICATION.md`](reports/FINAL_VERIFICATION.md): Master decision matrix and final pivot recommendation.

