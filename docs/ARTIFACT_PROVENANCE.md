# Checkpoint Artifact Provenance and Digital Signatures

**Document:** `docs/ARTIFACT_PROVENANCE.md`  
**Date:** 2026-10-26  
**Implementation:** `torchshield.provenance`  
**Algorithm:** Ed25519 (RFC 8032)  
**Status:** `VERIFIED`

---

## 1. Motivation

In high-consequence AI deployments (e.g. medical diagnostics, autonomous driving, financial systems, or frontier LLMs), authenticated symmetric encryption verifies that the holder of the symmetric key created or modified the checkpoint. However, in multi-team organizations, fine-tuning pipelines, and external model registries:
1. Multiple entities may share read access (holding the symmetric DEK / KMS unwrap permission).
2. Read permission must not imply publisher authority.
3. Consumers must cryptographically verify that a checkpoint was built and published by an authorized training pipeline or organization, without possessing the private publishing key.

---

## 2. Cryptographic Construction

TorchShield uses **Ed25519 digital signatures** over the canonical manifest specification digest:

```
                                    CANONICAL MANIFEST (M_spec)
                                    [artifact_id, tensors, metadata]
                                                  |
                                                  v
                                     TorchShield Canonical JSON v1
                                                  |
                                                  v
                                            SHA-256 Digest
                                                  |
                                                  v
             Publisher Private Key ------> Ed25519 Sign ------> 64-Byte Signature Blob
                                                                      |
                                                                      v
                                                            Stored in Final Manifest
```

### Signature Scope
The Ed25519 signature binds:
1. `artifact_id`: Unique identifier of the container.
2. `version`: Container and schema version numbers.
3. `total_tensors`, `total_chunks`, `total_raw_bytes`: Structural volume invariants.
4. `metadata`: Complete training progress (global step, epoch, loss, optimizer hyperparams, config).
5. `tensors`: Complete catalog of tensor names, shapes, and dtypes.
6. `manifest_digest`: SHA-256 digest of the canonical specification.

Because each chunk envelope cryptographically incorporates the `manifest_digest` in its Associated Authenticated Data (AAD), **the digital signature transitively signs every single weight parameter in the model**. Modifying even a single float32 weight anywhere in the multi-gigabyte file invalidates both the AEAD chunk tag and the Ed25519 publisher signature.

---

## 3. Publisher & Consumer Workflow

### Signing During Save
```python
from torchshield.training_state import save_training_checkpoint
from torchshield.provenance import Ed25519Signer

signer = Ed25519Signer()  # Or load from HSM / secure vault
publisher_pubkey = signer.public_bytes

save_training_checkpoint(
    output_path="model_step_10000.shield",
    model=model,
    optimizer=optimizer,
    key_provider=kms_provider,
    signer_private_key=signer.private_bytes,
)
```

### Enforcing Provenance During Load
```python
from torchshield.training_state import load_training_checkpoint

# Load fails closed if signature is missing or does not match publisher_pubkey
load_training_checkpoint(
    input_path="model_step_10000.shield",
    model=model,
    optimizer=optimizer,
    key_provider=kms_provider,
    expected_signer_public_key=publisher_pubkey,
)
```
