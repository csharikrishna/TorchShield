# TorchShield Framework & Model Serving Integration Report

**Audit Date:** 2026-10-28  
**Scope:** Hugging Face Integration (`torchshield/integrations/huggingface.py`) & Serving Engine Adapters (`torchshield/integrations/serving.py`)  

---

## 1. Hugging Face Transformers & Trainer Integration

Standard Hugging Face `Trainer` checkpoints introduce critical security vulnerabilities:
- Model weights may be saved via Safetensors, but `optimizer.pt`, `scheduler.pt`, and `rng_state.pth` are serialized using standard Python `pickle`.
- If an adversary tampers with these files in storage, resuming training (`trainer.train(resume_from_checkpoint=...)`) executes arbitrary Python code with full cluster credentials.

### 1.1 Remediation & Defensive Hardening
TorchShield provides a hardened integration layer:

1. **`TorchShieldTrainerCallback`:**
   - Intercepts the `on_save` event during training.
   - Captures model weights, optimizer momentum, learning rate schedulers, and RNG generators into an authenticated, encrypted `.shield` vault container.
   - Executes a two-phase commit: verifies the written `.shield` container via `inspect_training_checkpoint()` before deleting the insecure legacy pickle files.
2. **`TorchShieldTrainer`:**
   - Overrides `_load_optimizer_and_scheduler()` and `_load_rng_state()` to restore state directly from the encrypted `.shield` vault without pickle.
   - **Fail-Closed Enforcement:** If `vault_key` is configured and no valid `.shield` vault exists, it raises `FailClosedInsecureFallbackError` instead of delegating to unsafe `torch.load()`.
3. **`save_hf_model` / `load_hf_model`:**
   - Public APIs for exporting and restoring standalone Hugging Face model architectures under AES-256-GCM authenticated encryption.

Verified in `tests/test_huggingface_integration.py` (6 tests pass).

---

## 2. Native Model Serving Integration (`torchshield/integrations/serving.py`)

For inference serving environments (e.g. vLLM, TensorRT-LLM, TGI, or custom inference workers), storing unencrypted model weights on node scratch disks violates data protection boundaries.

### 2.1 Direct In-Memory Decryption Architecture
`torchshield/integrations/serving.py` implements a secure loader pattern:

```
[ Encrypted Model Checkpoint (.shield) ]
                   |
                   v
[ Stream-Decrypt Directly into Host RAM / VRAM ]
  * Authenticates container header, public salt, and AAD
  * Validates publisher digital signature (Ed25519)
  * Verifies chunk authentication tags (AES-256-GCM / Poly1305)
                   |
                   v
[ Direct Ingestion into Serving Model state_dict ]
  * Allocates tensors directly on target device (CPU / CUDA)
  * Zero intermediate plaintext files written to persistent storage
```

### 2.2 Critical Security Boundaries in Serving
- **Zero Plaintext Footprint:** The unencrypted model weights exist only in process RAM/VRAM during execution; no plaintext `.bin` or `.safetensors` file ever touches the disk.
- **KV Cache Separation:** Model parameter weights are loaded strictly into model execution memory; runtime KV cache attention states are maintained separately by the inference engine runtime.
- **Publisher Verification:** Inference runtimes can enforce `expected_signer_public_key`, rejecting untrusted or third-party modified model weights before serving requests.

Verified in `tests/test_serving_integration.py` (5 tests pass).
