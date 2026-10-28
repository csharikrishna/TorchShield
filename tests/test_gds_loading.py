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
    # Available must be False until genuine cuFile DMA read I/O is implemented
    assert status.available is False
    assert status.active_mode == "pinned_streaming_fallback"
    assert isinstance(status.cufile_loaded, bool)
    assert status.cuda_available == torch.cuda.is_available()
    if status.cuda_available:
        assert len(status.device_name) > 0
        assert status.device_name != "None"
    else:
        assert status.device_name == "None"
    assert len(status.reason) > 0


def test_probe_gds_capability_field_semantics_mocked(monkeypatch):
    """Test all capability fields under simulated hardware/driver states."""
    import ctypes
    from torchshield import gds

    # Scenario 1: CUDA available and libcufile detected
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA A100-SXM4-80GB")
    monkeypatch.setattr(ctypes, "CDLL", lambda name: object())

    status_detected = gds.probe_gds_capability()
    # Must NOT report available=True merely because library was found
    assert status_detected.available is False
    assert status_detected.cufile_loaded is True
    assert status_detected.cuda_available is True
    assert status_detected.device_name == "NVIDIA A100-SXM4-80GB"
    assert status_detected.active_mode == "pinned_streaming_fallback"
    assert "direct cuFile DMA read I/O is not implemented" in status_detected.reason

    # Scenario 2: CUDA available but libcufile missing
    def fail_cdll(name):
        raise OSError("Library not found")
    monkeypatch.setattr(ctypes, "CDLL", fail_cdll)

    status_no_lib = gds.probe_gds_capability()
    assert status_no_lib.available is False
    assert status_no_lib.cufile_loaded is False
    assert status_no_lib.cuda_available is True
    assert status_no_lib.active_mode == "pinned_streaming_fallback"

    # Scenario 3: CUDA not available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status_no_cuda = gds.probe_gds_capability()
    assert status_no_cuda.available is False
    assert status_no_cuda.cufile_loaded is False
    assert status_no_cuda.cuda_available is False
    assert status_no_cuda.device_name == "None"
    assert status_no_cuda.active_mode == "pinned_streaming_fallback"
    assert "CUDA is not available" in status_no_cuda.reason


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
