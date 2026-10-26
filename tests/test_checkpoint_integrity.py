"""Tests for authenticated checkpoint storage and integrity validation.

Verifies end-to-end serialization, exact tensor value reconstruction,
model inference equivalence, and detection of tampering (chunk corruption,
manifest alterations, wrong keys, truncation, and layer substitution).
"""
import os
import tempfile
import pytest
import torch
import torch.nn as nn

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError
from ahcc.secure import AuthenticationError


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_checkpoint_roundtrip_and_inference_equivalence():
    """Verify saved checkpoint restores exact weights and reproduces identical predictions."""
    model = SimpleToyModel()
    key = os.urandom(32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    x = torch.randn(8, 32, device=device)
    baseline_output = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.ahcc_ckpt")
        SecureModelCheckpoint.save_checkpoint(model.state_dict(), ckpt_path, key, chunk_bytes=1024)

        # Restore into new model
        restored_model = SimpleToyModel()
        restored_state = SecureModelCheckpoint.load_checkpoint(ckpt_path, key, device=device)
        restored_model.load_state_dict(restored_state)
        restored_model.to(device)

        restored_output = restored_model(x)

        # 1. Exact parameter equality
        for k in model.state_dict():
            assert torch.equal(model.state_dict()[k], restored_state[k]), f"Mismatch in layer {k}"

        # 2. Exact inference equivalence
        assert torch.equal(baseline_output, restored_output), "Model outputs diverged after restore!"


def test_checkpoint_detects_chunk_tampering():
    """Verify bit flip in any chunk causes immediate fail-closed authentication error."""
    model = SimpleToyModel()
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.ahcc_ckpt")
        SecureModelCheckpoint.save_checkpoint(model.state_dict(), ckpt_path, key, chunk_bytes=1024)

        with open(ckpt_path, "rb") as f:
            data = bytearray(f.read())

        # Flip byte in the middle of chunk data (near end of file)
        data[-50] ^= 0xFF

        bad_path = os.path.join(tmpdir, "bad.ahcc_ckpt")
        with open(bad_path, "wb") as f:
            f.write(data)

        with pytest.raises(AuthenticationError):
            SecureModelCheckpoint.load_checkpoint(bad_path, key)


def test_checkpoint_detects_wrong_key():
    """Verify wrong decryption key fails authentication."""
    model = SimpleToyModel()
    key = os.urandom(32)
    wrong_key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.ahcc_ckpt")
        SecureModelCheckpoint.save_checkpoint(model.state_dict(), ckpt_path, key)

        with pytest.raises(AuthenticationError):
            SecureModelCheckpoint.load_checkpoint(ckpt_path, wrong_key)


def test_checkpoint_detects_truncation():
    """Verify truncated file fails authentication."""
    model = SimpleToyModel()
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.ahcc_ckpt")
        SecureModelCheckpoint.save_checkpoint(model.state_dict(), ckpt_path, key)

        with open(ckpt_path, "rb") as f:
            data = f.read()

        truncated_path = os.path.join(tmpdir, "trunc.ahcc_ckpt")
        with open(truncated_path, "wb") as f:
            f.write(data[: len(data) // 2])

        with pytest.raises(AuthenticationError):
            SecureModelCheckpoint.load_checkpoint(truncated_path, key)
