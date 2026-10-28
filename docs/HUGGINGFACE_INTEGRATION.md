# Hugging Face Transformers & Trainer Integration

**Document:** `docs/HUGGINGFACE_INTEGRATION.md`
**Platform Version:** `0.7.1`

---

## 1. Why TorchShield for Hugging Face?

Hugging Face `transformers` revolutionized model sharing with `safetensors`. However, during active model training with `Trainer`:
1. **The Insecure Pickle Triad:** While model weights can be saved with `safetensors`, Hugging Face `Trainer` still saves `optimizer.pt`, `scheduler.pt`, and `rng_state.pth` using raw Python `pickle` (`torch.save`), exposing pipelines to deserialization vulnerabilities.
2. **No Built-in Encryption:** Checkpoints stored on disk or shared across team buckets are unencrypted plaintext.
3. **No Tamper Authentication:** If optimizer momentum states or weights are modified, `Trainer` resumes blindly without verification.

TorchShield provides **`TorchShieldTrainerCallback`** and **`TorchShieldTrainer`** to solve all three problems.

---

## 2. Using `TorchShieldTrainerCallback` with Hugging Face `Trainer`

Simply pass the callback into any standard Hugging Face `Trainer`:

```python
import secrets
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from torchshield import TorchShieldTrainerCallback

# 1. 256-bit symmetric encryption key (or AWS KMS provider)
master_key = secrets.token_bytes(32)

# 2. Configure training arguments
training_args = TrainingArguments(
    output_dir="./secure_checkpoints",
    save_strategy="steps",
    save_steps=500,
    per_device_train_batch_size=16,
)

# 3. Attach TorchShield callback
vault_callback = TorchShieldTrainerCallback(
    key=master_key,
    vault_filename="torchshield_checkpoint.shield",
    remove_insecure_pickle=True,  # Purges unsafe optimizer.pt, scheduler.pt, rng_state.pth
)

# 4. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[vault_callback],
)

# 5. Train safely
trainer.train()
```

When `trainer.train()` executes, every saved checkpoint contains an encrypted, authenticated, pickle-free `.shield` container containing:
- Exact model weights
- AdamW/SGD optimizer states mapped to named parameters
- Learning rate scheduler state
- Deterministic Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG generators
- Training progress metadata (`global_step`, `epoch`, `loss`)

---

## 3. Resuming Hugging Face `Trainer` from an Encrypted Vault

Use `TorchShieldTrainer` as a drop-in replacement for `transformers.Trainer`:

```python
from torchshield.integrations.huggingface import TorchShieldTrainer

trainer = TorchShieldTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    vault_key=master_key,  # Handles AEAD decryption and tamper detection
)

# Resume from checkpoint directory
trainer.train(resume_from_checkpoint="./secure_checkpoints/checkpoint-500")
```

If an attacker or corrupted disk flips even a single bit in the checkpoint container, `TorchShieldTrainer` fails closed immediately with `CheckpointError`.

---

## 4. Standalone Model Saving (`save_hf_model` / `load_hf_model`)

For saving only the model weights of an HF `PreTrainedModel`:

```python
from torchshield import save_hf_model, load_hf_model
from transformers import AutoModelForCausalLM

# Save encrypted weights
model = AutoModelForCausalLM.from_pretrained("gpt2")
save_hf_model(model, "weights.vault", key=master_key)

# Load decrypted weights directly into model architecture
fresh_model = AutoModelForCausalLM.from_pretrained("gpt2")
load_hf_model(fresh_model, "weights.vault", key=master_key, device="cuda")
```
