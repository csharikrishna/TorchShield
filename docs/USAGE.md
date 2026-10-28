# Usage Guide: TorchShield Secure AI Checkpoint Platform

**Document:** `docs/USAGE.md`
**Platform Version:** `0.7.1`

---

## 1. High-Level Training Checkpoint API (Recommended)

The high-level training state API protects the entire training loop state—model weights, optimizer states, learning-rate schedulers, RNG generators, and iteration progress—without using `pickle`.

### 1.1 Saving an Authenticated Training Checkpoint

```python
import secrets
import torch
import torch.nn as nn
from torchshield import save_training_checkpoint, TrainingProgress

# 1. Initialize training state
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
progress = TrainingProgress(epoch=5, global_step=1250, loss=0.2845)

# 2. Acquire a 256-bit symmetric key (or from KMS)
master_key = secrets.token_bytes(32)

# 3. Save checkpoint securely
save_training_checkpoint(
    checkpoint_path="checkpoints/epoch_5.shield",
    key=master_key,
    model=model,
    optimizer=optimizer,
    progress=progress,
    extra_metadata={"dataset": "imagenet", "batch_size": 64},
)
```

### 1.2 Resuming Training with Exact Bitwise Equivalence

When resuming training in a fresh process, `load_training_checkpoint` restores all parameters and deterministic RNG states:

```python
import torch
import torch.nn as nn
from torchshield import load_training_checkpoint

# 1. Initialize fresh instances
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 2. Load and verify cryptographic integrity
metadata = load_training_checkpoint(
    checkpoint_path="checkpoints/epoch_5.shield",
    key=master_key,
    model=model,
    optimizer=optimizer,
    device="cpu",  # or "cuda"
)

# 3. Inspect restored training state
print(f"Resumed from epoch {metadata.progress.epoch}, step {metadata.progress.global_step}")
print(f"Recorded validation loss: {metadata.progress.loss:.4f}")

# 4. Continue training loop immediately with identical subsequent forward/backward passes
```

---

## 2. Inspecting Checkpoints Without Full Model Allocation

You can inspect checkpoint metadata without loading heavy tensors into memory:

```python
from torchshield import inspect_training_checkpoint

# Zero-key inspect (inspects unencrypted outer container header)
header_info = inspect_training_checkpoint("checkpoints/epoch_5.shield")
print(f"Container Magic: {header_info.magic.hex()}")
print(f"Container Version: {header_info.version}")
print(f"Salt Size: {len(header_info.salt)} bytes")

# Authenticated inspect (decrypts manifest to read training progress & metadata)
manifest_info = inspect_training_checkpoint("checkpoints/epoch_5.shield", key=master_key)
print(f"Epoch: {manifest_info.progress.epoch}")
print(f"Global Step: {manifest_info.progress.global_step}")
print(f"Tensors Cataloged: {manifest_info.total_tensors}")
print(f"Raw Byte Volume: {manifest_info.total_raw_bytes} bytes")
```

---

## 3. Provenance & Digital Signatures (Ed25519)

Sign checkpoints with an organizational private signing key so downstream consumers can verify publishing authority without sharing decryption keys:

```python
from cryptography.hazmat.primitives.asymmetric import ed25519
from torchshield import save_training_checkpoint, load_training_checkpoint

# 1. Publisher generates or loads signing key
private_signing_key = ed25519.Ed25519PrivateKey.generate()
public_verifying_key = private_signing_key.public_key()

# 2. Save with cryptographic publisher signature
save_training_checkpoint(
    checkpoint_path="checkpoints/signed_model.shield",
    key=master_key,
    model=model,
    signing_key=private_signing_key,
)

# 3. Consumer verifies signature during restoration
load_training_checkpoint(
    checkpoint_path="checkpoints/signed_model.shield",
    key=master_key,
    model=model,
    verifying_key=public_verifying_key,
)
```

---

## 4. Key Management Services (KMS) & Local Emulation

TorchShield natively integrates with AWS KMS and enterprise secret managers via `KeyProvider`:

### 4.1 Production AWS KMS
```python
from torchshield import save_training_checkpoint, load_training_checkpoint
from torchshield.key_providers import AWSKMSProvider

kms_provider = AWSKMSProvider(
    key_arn="arn:aws:kms:us-east-1:123456789012:key/my-checkpoint-key",
    region_name="us-east-1",
)

# Pass KMS provider directly; TorchShield handles envelope encryption and unwrap operations
save_training_checkpoint(
    output_path="checkpoints/kms_model.shield",
    model=model,
    optimizer=optimizer,
    key_provider=kms_provider,
)
```

### 4.2 Zero-Cost Local KMS Testing (LocalStack / Moto)
To test envelope encryption locally or in CI/CD without cloud accounts or AWS credentials:
```python
# Connect to LocalStack or Moto emulator on standard port 4566
local_kms = AWSKMSProvider(
    key_arn="arn:aws:kms:us-east-1:123456789012:key/test-kek-uuid",
    endpoint_url="http://127.0.0.1:4566",
    region_name="us-east-1",
    aws_access_key_id="test",
    aws_secret_access_key="test",
)

save_training_checkpoint(
    output_path="checkpoints/local_kms_model.shield",
    model=model,
    optimizer=optimizer,
    key_provider=local_kms,
)
```

Run the self-contained verification suite:
```bash
python scripts/verify_local_kms.py
```

---

## 5. Pure Model Weight Checkpointing (`SecureModelCheckpoint`)

For inference deployments or pipelines saving only model weights (`state_dict`):

```python
from torchshield import SecureModelCheckpoint

# Save state_dict directly
state_dict = model.state_dict()
SecureModelCheckpoint.save_checkpoint(
    state_dict=state_dict,
    output_path="checkpoints/weights.shield",
    key=master_key,
    chunk_bytes=4 * 1024 * 1024,  # 4 MiB chunks
)

# Load state_dict
loaded_state = SecureModelCheckpoint.load_checkpoint(
    input_path="checkpoints/weights.shield",
    key=master_key,
    device="cuda",
)
model.load_state_dict(loaded_state)
```

---

## 6. Command Line Interface (CLI)

The `torchshield` CLI tool allows inspecting and verifying checkpoints from the terminal:

```bash
# Print container summary (no key needed)
torchshield info checkpoints/epoch_5.shield

# Inspect manifest with key
torchshield inspect checkpoints/epoch_5.shield --key-hex <64_HEX_CHARS>

# Verify cryptographic signature
torchshield verify checkpoints/signed_model.shield --public-key <PUB_KEY_HEX>
```
