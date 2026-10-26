# Usage Guide: Authenticated AI Data Protection & AHCC Research Prototype

## 1. Production Model Checkpoint Encryption (Recommended AEAD)

For protecting neural network checkpoints and weights in production, use `SecureModelCheckpoint` backed by authenticated AES-256-GCM:

```python
import torch
import torch.nn as nn
from ahcc.checkpoint import SecureModelCheckpoint
from ahcc.secure import AESGCMEnvelope

# 1. Generate or load 256-bit key from KMS/Vault
key = AESGCMEnvelope.generate_key()

# 2. Save model state dict with authenticated chunks and manifest
model = MyTransformerModel()
SecureModelCheckpoint.save_checkpoint(
    state_dict=model.state_dict(),
    output_path="checkpoints/model.ahcc_ckpt",
    key=key,
    chunk_bytes=4 * 1024 * 1024  # 4 MiB chunks
)

# 3. Stream load directly into GPU VRAM with full integrity verification
loaded_state = SecureModelCheckpoint.load_checkpoint(
    input_path="checkpoints/model.ahcc_ckpt",
    key=key,
    device="cuda"
)
model.load_state_dict(loaded_state)
```

Any modification of ciphertext bytes, altered model manifests, or swapped layers raises `AuthenticationError` and halts loading immediately.

---

## 2. Experimental AHCC Research Prototype (Research Only)

```python
import torch
from ahcc.engine import AHCC_Engine, TensorCipherWrapper

# Initialize AHCC Engine over prime field F_p (p = 2^32 - 5)
engine = AHCC_Engine(
    N=64,                      # Vector dimension
    R=4,                       # Round count
    p=4_294_967_291,           # Primary prime modulus
    device="cuda"              # 'cuda' or 'cpu'
)

# NOTE: Direct engine encryption provides NO authentication and NO nonces.
# Do not use for valuable or sensitive data.
plaintext = torch.randint(0, engine.p, (64, 10), dtype=torch.int64, device="cuda")
ciphertext = engine.encrypt(plaintext)
recovered = engine.decrypt(ciphertext)
assert torch.equal(plaintext, recovered)
```
