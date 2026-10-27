# Usage Guide: AHCC Secure AI Checkpoint Platform

**Document:** `docs/USAGE.md`  
**Platform Version:** `0.7.0`  

---

## 1. High-Level Training Checkpoint API (Recommended)

The high-level training state API protects the entire training loop state—model weights, optimizer states, learning-rate schedulers, RNG generators, and iteration progress—without using `pickle`.

### 1.1 Saving an Authenticated Training Checkpoint

```python
import secrets
import torch
import torch.nn as nn
from ahcc import save_training_checkpoint, TrainingProgress

# 1. Initialize training state
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
progress = TrainingProgress(epoch=5, global_step=1250, loss=0.2845)

# 2. Acquire a 256-bit symmetric key (or from KMS)
master_key = secrets.token_bytes(32)

# 3. Save checkpoint securely
save_training_checkpoint(
    checkpoint_path="checkpoints/epoch_5.ahcc",
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
from ahcc import load_training_checkpoint

# 1. Initialize fresh instances
model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 2. Load and verify cryptographic integrity
metadata = load_training_checkpoint(
    checkpoint_path="checkpoints/epoch_5.ahcc",
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
from ahcc import inspect_training_checkpoint

# Zero-key inspect (inspects unencrypted outer container header)
header_info = inspect_training_checkpoint("checkpoints/epoch_5.ahcc")
print(f"Container Magic: {header_info.magic.hex()}")
print(f"Container Version: {header_info.version}")
print(f"Salt Size: {len(header_info.salt)} bytes")

# Authenticated inspect (decrypts manifest to read training progress & metadata)
manifest_info = inspect_training_checkpoint("checkpoints/epoch_5.ahcc", key=master_key)
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
from ahcc import save_training_checkpoint, load_training_checkpoint

# 1. Publisher generates or loads signing key
private_signing_key = ed25519.Ed25519PrivateKey.generate()
public_verifying_key = private_signing_key.public_key()

# 2. Save with cryptographic publisher signature
save_training_checkpoint(
    checkpoint_path="checkpoints/signed_model.ahcc",
    key=master_key,
    model=model,
    signing_key=private_signing_key,
)

# 3. Consumer verifies signature during restoration
load_training_checkpoint(
    checkpoint_path="checkpoints/signed_model.ahcc",
    key=master_key,
    model=model,
    verifying_key=public_verifying_key,
)
```

---

## 4. Key Management Services (KMS)

AHCC natively integrates with AWS KMS and enterprise secret managers via `KeyProvider`:

```python
from ahcc.key_providers.kms import AWSKMSKeyProvider
from ahcc import save_training_checkpoint

kms_provider = AWSKMSKeyProvider(
    kms_key_id="arn:aws:kms:us-east-1:123456789012:key/my-checkpoint-key",
    region_name="us-east-1",
)

# Pass KMS provider directly; AHCC handles envelope encryption and unwrap operations
save_training_checkpoint(
    checkpoint_path="checkpoints/kms_model.ahcc",
    key=kms_provider,
    model=model,
    optimizer=optimizer,
)
```

---

## 5. Pure Model Weight Checkpointing (`SecureModelCheckpoint`)

For inference deployments or pipelines saving only model weights (`state_dict`):

```python
from ahcc import SecureModelCheckpoint

# Save state_dict directly
state_dict = model.state_dict()
SecureModelCheckpoint.save_checkpoint(
    state_dict=state_dict,
    output_path="checkpoints/weights.ahcc",
    key=master_key,
    chunk_bytes=4 * 1024 * 1024,  # 4 MiB chunks
)

# Load state_dict
loaded_state = SecureModelCheckpoint.load_checkpoint(
    input_path="checkpoints/weights.ahcc",
    key=master_key,
    device="cuda",
)
model.load_state_dict(loaded_state)
```

---

## 6. Command Line Interface (CLI)

The `ahcc` CLI tool allows inspecting and verifying checkpoints from the terminal:

```bash
# Print container summary (no key needed)
ahcc info checkpoints/epoch_5.ahcc

# Inspect manifest with key
ahcc inspect checkpoints/epoch_5.ahcc --key-hex <64_HEX_CHARS>

# Verify cryptographic signature
ahcc verify checkpoints/signed_model.ahcc --public-key <PUB_KEY_HEX>
```
