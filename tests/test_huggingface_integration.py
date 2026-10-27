"""Tests for Hugging Face Transformers & Trainer Integration."""
import os
import secrets
import shutil
import pytest
import torch
import torch.nn as nn

from ahcc import (
    TorchVaultTrainerCallback,
    AHCCTrainerCallback,
    save_hf_model,
    load_hf_model,
    inspect_training_checkpoint,
    load_training_checkpoint,
    CheckpointError,
)

try:
    from transformers import TrainingArguments, TrainerState, TrainerControl
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_callback_key_validation():
    """Verify that callback rejects invalid keys."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("transformers not installed")

    with pytest.raises(ValueError, match="exactly 32 bytes"):
        TorchVaultTrainerCallback(key=b"too_short")


def test_callback_on_save_workflow(tmp_path):
    """Verify that TorchVaultTrainerCallback intercepts on_save, writes encrypted container,
    and optionally purges unsafe pickle files."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("transformers not installed")

    key = secrets.token_bytes(32)
    output_dir = str(tmp_path / "trainer_output")
    step_dir = os.path.join(output_dir, "checkpoint-50")
    os.makedirs(step_dir, exist_ok=True)

    # Create dummy insecure pickle files that HF Trainer normally creates
    for fname in ["optimizer.pt", "scheduler.pt", "rng_state.pth"]:
        with open(os.path.join(step_dir, fname), "wb") as f:
            f.write(b"insecure_pickle_bytes")

    callback = TorchVaultTrainerCallback(
        key=key,
        vault_filename="torchvault_checkpoint.ahcc",
        remove_insecure_pickle=True,
    )

    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    # Perform a dummy training step to populate optimizer state
    loss = model(torch.randn(4, 32)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    args = TrainingArguments(output_dir=output_dir, use_cpu=True)
    state = TrainerState(epoch=2.0, global_step=50, log_history=[{"loss": 0.42}])
    control = TrainerControl()

    # Invoke on_save
    callback.on_save(
        args=args,
        state=state,
        control=control,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    vault_file = os.path.join(step_dir, "torchvault_checkpoint.ahcc")
    assert os.path.isfile(vault_file), "Vault checkpoint was not written!"
    assert os.path.getsize(vault_file) > 0

    # Insecure pickle files must have been cleaned up
    assert not os.path.isfile(os.path.join(step_dir, "optimizer.pt"))
    assert not os.path.isfile(os.path.join(step_dir, "scheduler.pt"))
    assert not os.path.isfile(os.path.join(step_dir, "rng_state.pth"))

    # Authenticated inspect must succeed
    meta = inspect_training_checkpoint(vault_file, key=key)
    assert meta["authenticated"] is True
    assert meta["epoch"] == 2
    assert meta["global_step"] == 50
    assert meta["loss"] == pytest.approx(0.42)
    assert meta["config"]["framework"] == "huggingface_transformers"


def test_resume_from_hf_vault(tmp_path):
    """Verify that an interrupted training state written by the callback can be restored."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("transformers not installed")

    key = secrets.token_bytes(32)
    step_dir = str(tmp_path / "checkpoint-100")
    os.makedirs(step_dir, exist_ok=True)

    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # Train 2 steps
    for _ in range(2):
        loss = model(torch.randn(4, 32)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    callback = TorchVaultTrainerCallback(key=key)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True)
    state = TrainerState(epoch=3.0, global_step=100, log_history=[{"loss": 0.15}])
    control = TrainerControl()

    callback.on_save(args=args, state=state, control=control, model=model, optimizer=optimizer)
    vault_file = os.path.join(step_dir, "torchvault_checkpoint.ahcc")

    # Fresh model and optimizer
    fresh_model = SimpleModel()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)

    progress = load_training_checkpoint(
        input_path=vault_file,
        model=fresh_model,
        optimizer=fresh_optimizer,
        key=key,
    )

    assert progress.epoch == 3
    assert progress.global_step == 100
    # Weights must match exactly
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh_model.named_parameters()):
        assert torch.equal(p1, p2), f"Weight mismatch for {n1}"


def test_save_and_load_hf_model(tmp_path):
    """Test the direct save_hf_model and load_hf_model helpers."""
    key = secrets.token_bytes(32)
    model = SimpleModel()
    path = str(tmp_path / "hf_model.vault")

    save_hf_model(model, path, key)
    assert os.path.isfile(path)

    restored_model = SimpleModel()
    load_hf_model(restored_model, path, key)

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), restored_model.named_parameters()):
        assert torch.equal(p1, p2)


def test_tamper_detection_in_hf_vault(tmp_path):
    """Verify that tampering with an HF vault container fails closed."""
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("transformers not installed")

    key = secrets.token_bytes(32)
    step_dir = str(tmp_path / "checkpoint-200")
    os.makedirs(step_dir, exist_ok=True)

    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    callback = TorchVaultTrainerCallback(key=key)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True)
    state = TrainerState(epoch=1.0, global_step=200)
    control = TrainerControl()

    callback.on_save(args=args, state=state, control=control, model=model, optimizer=optimizer)
    vault_file = os.path.join(step_dir, "torchvault_checkpoint.ahcc")

    # Tamper with the container
    with open(vault_file, "r+b") as f:
        f.seek(80)
        byte = f.read(1)[0]
        f.seek(80)
        f.write(bytes([byte ^ 0xFF]))

    fresh_model = SimpleModel()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)

    with pytest.raises(CheckpointError):
        load_training_checkpoint(
            input_path=vault_file,
            model=fresh_model,
            optimizer=fresh_optimizer,
            key=key,
        )
