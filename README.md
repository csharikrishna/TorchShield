# TorchShield Secure AI Checkpoint Platform

[![Version](https://img.shields.io/badge/version-0.7.0-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-Conditional_Staging_Approved-green.svg)](docs/)
[![Tests](https://img.shields.io/badge/tests-260_passed-brightgreen.svg)](tests/)
[![Security](https://img.shields.io/badge/security-AEAD_AES--256--GCM-blue.svg)](SECURITY.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csharikrishna/TorchShield/blob/master/notebooks/TorchShield_Interactive_Walkthrough.ipynb)

**TorchShield Secure AI Checkpoint Platform** is a cryptographically hardened, high-performance training and model checkpoint platform for PyTorch. It protects model weights, optimizer states, learning rate schedulers, and RNG generators against unauthorized inspection, weight tampering, supply-chain deserialization exploits (`pickle`), and unauthorized rollbacks.

---

## Key Capabilities

- **Zero-Pickle Deserialization:** Immune to arbitrary code execution attacks; tensor tensors and metadata are serialized via explicit binary buffers and canonical JSON.
- **Streaming AEAD Cryptography:** Standard AES-256-GCM / ChaCha20-Poly1305 with cryptographically enforced monotonic sequential counter nonces (`b"CHNK" + uint64_be`), guaranteeing zero nonce reuse collision risk ($P=0$).
- **Exact Bit-for-Bit Training Equivalence:** Restores model weights, AdamW/SGD optimizer momentum states, and Python/NumPy/PyTorch/CUDA RNG states with mathematical bitwise exactness across process interruptions.
- **Enterprise Key Management:** Native support for local 256-bit keys and AWS KMS envelope encryption (`AWSKMSKeyProvider`) with secure unwrap operations and rotation hooks.
- **Publisher Signatures & Provenance:** Ed25519 digital signatures bind deterministic canonical JSON (TorchShield Canonical JSON v1) manifests transitively to all model weights.
- **Anti-Rollback Freshness:** Local freshness ledger tracks monotonic checkpoint generations to block replay attacks.
- **Distributed & Sharded Training:** Seamless integration with multi-GPU and distributed training frameworks (ZeRO / FSDP sharding).

---

## Architecture & Container Wire Format (v5)

```
+------------------------------------------------------------------------------------+
|                                CONTAINER HEADER (41 Bytes)                         |
|  Magic: "TSHDMOD5" (8B) | Version: 0x05 (1B) | Salt: 32B Public HKDF Salt (32B)    |
+------------------------------------------------------------------------------------+
|                                 CHUNK ENVELOPES (1..M)                             |
|  [Chunk Header: Chunk ID (8B) || Payload Size (8B)]                                |
|  [AEAD Nonce: b"CHNK" || uint64_be Counter (12B)]                                  |
|  [Encrypted Raw Tensor Bytes]                                                      |
|  [Poly1305 / GCM Authentication Tag (16B)]                                         |
+------------------------------------------------------------------------------------+
|                                 MANIFEST ENVELOPE                                  |
|  [Manifest Length (8B)]                                                            |
|  [AEAD Nonce: b"MANI" || uint64_be Counter (12B)]                                  |
|  [Encrypted Canonical JSON Manifest (Tensors, RNG, Optimizer, Metadata)]           |
|  [Ed25519 Digital Signature Blob (64B)]                                            |
|  [Authentication Tag (16B)]                                                        |
+------------------------------------------------------------------------------------+
```

---

## Quickstart

### 1. Installation

If cloning the repository from source:
```bash
# Editable development install
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

Or build and install the standalone wheel distribution:
```bash
python -m build
pip install dist/torchshield-0.7.0-py3-none-any.whl
```
*(Pre-built wheels in `dist/` can be installed directly once built via `python -m build`).*

### 2. Saving an Authenticated Training Checkpoint

```python
import secrets
import torch
import torch.nn as nn
from torchshield import save_training_checkpoint, TrainingProgress

# Setup training components
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
progress = TrainingProgress(epoch=10, global_step=5000, loss=0.1425)

# 256-bit master secret key (or AWS KMS KeyProvider)
key = secrets.token_bytes(32)

# Save checkpoint with atomic flush and tamper-proof streaming AEAD
save_training_checkpoint(
    checkpoint_path="checkpoints/run_01_step5000.shield",
    key=key,
    model=model,
    optimizer=optimizer,
    progress=progress,
    extra_metadata={"experiment": "bert_base_finetune", "batch_size": 32},
)
```

### 3. Resuming Training with Exact Equivalence

```python
import torch
import torch.nn as nn
from torchshield import load_training_checkpoint

# Initialize fresh model and optimizer instances
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Load and verify cryptographic authenticity (fails closed if tampered)
metadata = load_training_checkpoint(
    checkpoint_path="checkpoints/run_01_step5000.shield",
    key=key,
    model=model,
    optimizer=optimizer,
    device="cpu",  # or "cuda"
)

print(f"Successfully resumed at epoch {metadata.progress.epoch}, step {metadata.progress.global_step}")
# Subsequent optimizer.step() and loss evaluations match an uninterrupted run bit-for-bit
```

### 4. Inspecting Checkpoints (CLI and Python)

Inspect container headers without decrypting or allocating tensors:

```python
from torchshield import inspect_training_checkpoint

# Zero-key inspect (public container envelope)
info = inspect_training_checkpoint("checkpoints/run_01_step5000.shield")
print(f"Format Version: {info.version}, Salt: {info.salt.hex()[:16]}...")

# Authenticated inspect (reads training progress and parameter inventory)
auth_info = inspect_training_checkpoint("checkpoints/run_01_step5000.shield", key=key)
print(f"Progress: Epoch {auth_info.progress.epoch}, Step {auth_info.progress.global_step}")
print(f"Tensor Count: {auth_info.total_tensors}, Raw Size: {auth_info.total_raw_bytes} B")
```

Or via command-line:
```bash
torchshield info checkpoints/run_01_step5000.shield
torchshield inspect checkpoints/run_01_step5000.shield --key-hex <64_HEX_CHARS>
```

### 5. Hugging Face Trainer Integration

Eliminate insecure unpickling of `optimizer.pt`, `scheduler.pt`, and `rng_state.pth` in Hugging Face `Trainer`:

```python
from transformers import Trainer, TrainingArguments
from torchshield import TorchShieldTrainerCallback

# Attach callback to automatically save encrypted, pickle-free checkpoints
callback = TorchShieldTrainerCallback(
    key=key,
    vault_filename="torchshield_checkpoint.shield",
    remove_insecure_pickle=True,  # Purges unsafe pickle files generated by default Trainer
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./secure_ckpts", save_strategy="steps", save_steps=500),
    train_dataset=train_dataset,
    callbacks=[callback],
)
trainer.train()
```

### 6. Local KMS & Zero-Cost Cloud Emulation (LocalStack / Moto)

Test enterprise AWS KMS envelope encryption locally at **$0 cloud cost** and with **zero AWS credentials**:

```python
from torchshield import save_training_checkpoint, load_training_checkpoint, TrainingProgress
from torchshield.key_providers import AWSKMSProvider

# Point directly to LocalStack or Moto emulator (port 4566)
provider = AWSKMSProvider(
    key_arn="arn:aws:kms:us-east-1:123456789012:key/test-key-uuid",
    endpoint_url="http://127.0.0.1:4566",
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test",
)

# Checkpoint payload is encrypted via AES-256-GCM; DEK is envelope-wrapped by KMS
save_training_checkpoint(
    output_path="checkpoints/local_kms_model.shield",
    model=model,
    optimizer=optimizer,
    progress=TrainingProgress(global_step=100, epoch=1, loss=0.42),
    key_provider=provider,
)

# Automated bit-for-bit restoration with cryptographic authentication
load_training_checkpoint(
    input_path="checkpoints/local_kms_model.shield",
    model=model,
    optimizer=optimizer,
    key_provider=provider,
)
```

Run the included end-to-end Local KMS verification suite (auto-spawns an embedded local KMS server if none is running):
```bash
python scripts/verify_local_kms.py
```

---

## Documentation Index

The complete documentation suite is available under [`docs/`](docs/):

| Guide | Content |
|---|---|
| [**Docs Overview**](docs/README.md) | Master documentation index and cryptographic summary |
| [**Installation Guide**](docs/INSTALLATION.md) | Setup instructions for Windows, Linux, CUDA, and wheel distributions |
| [**Usage & API Guide**](docs/USAGE.md) | Comprehensive code recipes for training, inference, and CLI workflows |
| [**Hugging Face Integration**](docs/HUGGINGFACE_INTEGRATION.md) | Secure, pickle-free Hugging Face Trainer callback and model weight encryption |
| [**Training Checkpoint Schema**](docs/TRAINING_CHECKPOINT_SCHEMA.md) | Complete byte-level v5 container format specification and AAD layout |
| [**Threat Model**](docs/THREAT_MODEL.md) | Adversary capabilities, defensive boundaries, and non-goals |
| [**Rollback Protection**](docs/ROLLBACK_PROTECTION.md) | Anti-rollback freshness ledger and hypervisor snapshot boundary conditions |
| [**KMS Integration**](docs/KMS_INTEGRATION.md) | Envelope encryption with AWS KMS and cloud secret management |
| [**Distributed Checkpoints**](docs/DISTRIBUTED_CHECKPOINTS.md) | Multi-node sharding and distributed training coordination |
| [**Artifact Provenance**](docs/ARTIFACT_PROVENANCE.md) | Ed25519 digital signatures and TorchShield Canonical JSON v1 manifest signing |

---

## Security Model & Operational Boundaries

> [!IMPORTANT]
> **Deployment Status:** Conditional Approval — Internal Staging / Research Candidate (v0.7.0).
>
> 1. **Derivation Salt vs Secret Key:** The 32-byte header salt is a non-secret public salt used for HKDF domain separation. The 32-byte master key is private input keying material (IKM).
> 2. **Nonce Monotonicity Invariant:** Sequential intra-key counter nonces (`b"CHNK" + uint64_be`) guarantee strictly zero collision risk ($P=0$). Cross-key random collisions are bounded at $P \approx 2^{-128}$.
> 3. **Storage-Level Snapshot Boundary:** Local SQLite/file ledgers detect local replay attacks but cannot prevent hypervisor-level snapshot rollbacks that restore both disk and checkpoint files simultaneously. High-assurance deployments require an out-of-band external freshness store.
> 4. **Cloud KMS & Zero-Cost Local Emulation:** Production deployments integrate with AWS KMS hardware security modules (HSM). Local testing and CI/CD pipelines can use LocalStack, Moto, or the built-in `LocalDevKeyProvider` at zero cloud cost.
> 5. **Advanced Hardware, PQC & Serving Boundaries:**
>    - **GDS GPU Loading (`gds.py`):** Reference prototype (`available=False`, `active_mode="pinned_streaming_fallback"`). Direct cuFile DMA kernel-bypass I/O read is a future roadmap item; streaming pinned buffer fallback is verified.
>    - **Hardware Attestation (`attestation.py`):** Prototype signed evidence verifier (`AttestationEvidence`). Verifies custom signed structures with Ed25519 platform signatures over measurement digests and nonces. Vendor-native hardware quotes (NVIDIA Hopper CC, AMD SEV-SNP) remain roadmap items. Strictly fails closed with `is_production_secure=False` when mock or without a usable evidence path.
>    - **Post-Quantum Cryptography (`pqc.py`):** Reference hybrid framework architecture. Internal test backend uses classical primitives strictly for test plumbing; production mode strictly fails closed pending integration with audited native C bindings (`liboqs`).
>    - **Model Serving (`serving.py`):** Reference serving pipeline. In-memory tensor deserialization without disk staging; transactional rollback protects target model against partial corruption; vLLM / Triton hooks are integration reference adapters.
>    - **Quorum Freshness (`freshness.py`):** Reference etcd consensus client (`EtcdFreshnessBackend`). Monotonic revision and transactional CAS anti-rollback client verified; live distributed deployment requires an external etcd cluster.

---

## Verification & Testing

Run the full core test suite:
```bash
pytest tests
```
*Current test suite: **260 passed in ~44s** (0 failed, 0 errors).*

Run the end-to-end Local KMS envelope encryption test suite:
```bash
python scripts/verify_local_kms.py
```
*Local KMS suite: **7/7 verification checks passed [100% SUCCESS]** (DEK wrapping, tamper blocking, bit-for-bit weight recovery, HuggingFace integration). Total verified checks: 267/267.*

Run independent consumer integration tests:
```bash
cd ../torchshield_test_codebase
.venv/Scripts/pytest tests
```
*Consumer test suite: **27 passed in ~21s** (0 failed, 0 errors).*

---

## Historical Archives

Earlier research prototypes, paper cryptanalysis notes, and benchmark execution logs preceding the v0.7.0 platform release have been safely archived under:
- `archives/v1_chaotic_crypto_prototype/`: Original research blueprint, paper cryptanalysis audit, and initial chaotic cipher prototype.
- `archives/phase_reports/`: Intermediate phase evaluation reports and test logs.
- `archives/benchmark_scripts_and_results/`: Raw benchmark scripts and phase performance data.

---

## License & Security Policy

- **Security Disclosures:** Please review [`SECURITY.md`](SECURITY.md) for vulnerability reporting procedures.
- **Changelog:** Detailed version history is recorded in [`CHANGELOG.md`](CHANGELOG.md).
- **Release Audit:** The formal release closeout report is in [`docs/`](docs/).
