# Enterprise KMS Integration and Key Lifecycle Guide

**Document:** `docs/KMS_INTEGRATION.md`  
**Date:** 2026-10-26  
**Module:** `torchshield.key_providers`  

---

## 1. Overview and Envelope Encryption Architecture

In enterprise AI infrastructure, model checkpoints range from hundreds of megabytes to hundreds of gigabytes. Transmitting these multi-gigabyte payloads across cloud KMS endpoints is cost-prohibitive, introduces severe network latency, and exceeds KMS single-request payload limits (e.g., 4 KiB limit in AWS KMS).

The TorchShield Secure Checkpoint Engine implements **client-side envelope encryption**:
1. **Key Encryption Keys (KEKs):** Managed securely inside hardware-backed HSMs via the KMS provider (e.g., AWS KMS Customer Managed Keys).
2. **Data Encryption Keys (DEKs):** A fresh, cryptographically random 256-bit AES key is generated for each checkpoint artifact.
3. **Payload Protection:** All tensor chunks, optimizer buffers, and metadata are encrypted and authenticated locally via high-throughput AES-256-GCM.
4. **Wrapped Key Storage:** Only the KMS-wrapped DEK ciphertext blob (typically ~100–250 bytes) and the non-secret Key ID are stored inside the checkpoint container header.

```
+----------------------------------------------------------------------------------------------------+
|                                    ENVELOPE ENCRYPTION WORKFLOW                                    |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    +-------------+          1. GenerateDataKey(KeyId="arn:aws:kms:...")                            |
|    |             | ------------------------------------------------------------+                   |
|    |   AWS KMS   |                                                             |                   |
|    |   / HSM     | <-------------------------------------------------------+   |                   |
|    +-------------+          2. Returns { PlaintextDEK (32B), WrappedDEK }   |   |                   |
|           |                                                                |   |                   |
|           |                                                                v   |                   |
|           |                                                  +---------------+ |                   |
|           |                                                  | Training Node | |                   |
|           |                                                  +---------------+ |                   |
|           |                                                          |         |                   |
|           |                                          Encrypt Tensors |         |                   |
|           |                                          via AES-256-GCM |         |                   |
|           |                                                          v         |                   |
|           |                                    +-----------------------------+ |                   |
|           |                                    | Stored Checkpoint Artifact  | |                   |
|           |                                    |                             | |                   |
|           |                                    | [Header]                    | |                   |
|           |                                    | Key ID: "arn:aws:..."       | |                   |
|           |                                    | Wrapped DEK: <blob> <-------+ |                   |
|           |                                    |                             |                     |
|           |                                    | [Payload]                   |                     |
|           |                                    | Encrypted Chunks & Tag      |                     |
|           |                                    +-----------------------------+                     |
|           |                                                  |                                     |
|           |                                                  |                                     |
|           | 3. Decrypt(WrappedDEK)                           | Restore / Resume                    |
|           +--------------------------------------------------+                                     |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Supported Key Providers

### `AWSKMSProvider` (Production)
- **Class:** `torchshield.key_providers.AWSKMSProvider`
- **Security Rating:** Production-Secure (`is_production_secure = True`)
- **Backing:** AWS KMS API (`generate_data_key`, `decrypt`, `re_encrypt`).
- **Authentication:** Standard AWS IAM credentials (roles, instance profiles, or environment).

```python
from torchshield.key_providers import AWSKMSProvider
from torchshield.training_state import save_training_checkpoint, load_training_checkpoint

# Initialize KMS provider with Key ARN
kms_provider = AWSKMSProvider(
    key_arn="arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
)

# Save training checkpoint using KMS envelope encryption
save_training_checkpoint(
    model=model,
    optimizer=optimizer,
    output_path="checkpoints/epoch_10.shield",
    key_provider=kms_provider,
    encryption_context={"environment": "production", "model": "llama-transformer"},
)

# Load training checkpoint - DEK is unwrapped automatically via KMS Decrypt API
load_training_checkpoint(
    input_path="checkpoints/epoch_10.shield",
    key_provider=kms_provider,
    model=model,
    optimizer=optimizer,
    encryption_context={"environment": "production", "model": "llama-transformer"},
)
```

### `LocalDevKeyProvider` (Development / Test Only)
- **Class:** `torchshield.key_providers.LocalDevKeyProvider`
- **Security Rating:** Development Only (`is_production_secure = False`)
- **Backing:** Software-based AES-256-GCM key wrapping using a 256-bit local KEK.
- **Usage:** Ideal for local unit tests, CI test pipelines, and offline debugging without cloud credentials.

---

## 3. Key Rotation & Migration Without Re-Encryption

When a Key Encryption Key (KEK) is rotated or migrated:
- **No Payload Re-Encryption:** Model weights and tensors (potentially hundreds of gigabytes) do **not** need to be re-encrypted.
- **DEK Re-Wrapping:** Calling `key_provider.rewrap_key(wrapped_dek, new_key_id)` decrypts the 32-byte DEK and re-wraps it under the new KEK in a single sub-second operation.
