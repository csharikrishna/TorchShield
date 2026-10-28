"""Unit tests for GPUDirect Storage (GDS) and High-Performance Tensor Loading."""
import os
import secrets
import tempfile
import pytest
import torch
import torch.nn as nn

from torchshield.checkpoint import CheckpointError, SecureModelCheckpoint
from torchshield.gds import GDSLoader, GDSStatus, probe_gds_capability


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_probe_gds_capability():
    status = probe_gds_capability()
    assert isinstance(status, GDSStatus)
    assert status.active_mode in ("direct_cufile", "pinned_streaming_fallback")
    if torch.cuda.is_available():
        assert status.cuda_available is True
        assert len(status.device_name) > 0


def test_gds_loader_model_roundtrip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = secrets.token_bytes(32)
    model = SimpleNet()
    if device == "cuda":
        model = model.to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "gds_model.seal")
        SecureModelCheckpoint.save_model(model, ckpt_path, key)

        loader = GDSLoader(device=device)
        assert isinstance(loader.status, GDSStatus)

        target_model = SimpleNet()
        if device == "cuda":
            target_model = target_model.to(device)

        loaded_model = loader.load_model_direct(
            checkpoint_path=ckpt_path,
            model=target_model,
            key_bytes=key,
        )

        for name, param in model.named_parameters():
            loaded_param = loaded_model.get_parameter(name)
            assert torch.allclose(param.cpu(), loaded_param.cpu())
            if device == "cuda":
                assert loaded_param.device.type == "cuda"


def test_gds_loader_tensor_direct():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = secrets.token_bytes(32)
    model = SimpleNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "gds_tensor.seal")
        SecureModelCheckpoint.save_model(model, ckpt_path, key)

        loader = GDSLoader(device=device)
        tensor = loader.load_tensor_direct(
            checkpoint_path=ckpt_path,
            layer_name="fc1.weight",
            key_bytes=key,
        )

        orig_tensor = model.fc1.weight
        assert torch.allclose(orig_tensor.cpu(), tensor.cpu())
        if device == "cuda":
            assert tensor.device.type == "cuda"


def test_gds_loader_tamper_fails_closed():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = secrets.token_bytes(32)
    model = SimpleNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "gds_tamper.seal")
        SecureModelCheckpoint.save_model(model, ckpt_path, key)

        # Tamper byte in file
        with open(ckpt_path, "r+b") as f:
            data = bytearray(f.read())
            data[-10] ^= 0xFF
            f.seek(0)
            f.write(data)

        loader = GDSLoader(device=device)
        target_model = SimpleNet()
        with pytest.raises(CheckpointError):
            loader.load_model_direct(
                checkpoint_path=ckpt_path,
                model=target_model,
                key_bytes=key,
            )
