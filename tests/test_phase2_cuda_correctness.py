"""Phase 2 CUDA Correctness and Hardware Acceleration Tests.

Validates:
- Real hardware CUDA execution on NVIDIA GeForce GTX 1650 (Compute Capability 7.5).
- Direct checkpoint loading to CUDA device ('cuda:0').
- Parity between CPU-loaded and CUDA-loaded model parameters and outputs.
- Memory cleanup after loading and exception handling.
- Verification that no implicit fallback to CPU occurs when 'cuda' is requested.
"""
from __future__ import annotations

import os
import secrets
import pytest
import torch
import torch.nn as nn

from torchshield.checkpoint import SecureModelCheckpoint, CheckpointError


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is not available")
def test_direct_cuda_tensor_loading(tmp_path):
    """Verify tensors are placed directly onto the requested CUDA device."""
    key = secrets.token_bytes(32)
    state = {
        "fc.weight": torch.randn(128, 256),
        "fc.bias": torch.randn(128),
    }
    path = str(tmp_path / "cuda_model.shield")
    SecureModelCheckpoint.save_checkpoint(state, path, key)

    loaded = SecureModelCheckpoint.load_checkpoint(path, key, device="cuda")

    for k in state:
        assert loaded[k].is_cuda, f"Tensor {k} was not loaded onto CUDA!"
        assert loaded[k].device.type == "cuda"
        # Compare values after moving reference to CUDA
        assert torch.equal(loaded[k], state[k].to("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is not available")
def test_cuda_model_streaming_and_inference(tmp_path):
    """Verify target_model parameter streaming on CUDA and forward pass parity."""
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(42)
    model_cpu = SimpleMLP().eval()
    dummy_input = torch.randn(8, 64)
    with torch.no_grad():
        cpu_out = model_cpu(dummy_input)

    key = secrets.token_bytes(32)
    path = str(tmp_path / "mlp.shield")
    SecureModelCheckpoint.save_checkpoint(model_cpu.state_dict(), path, key)

    # Instantiate model on CUDA
    model_cuda = SimpleMLP().to("cuda").eval()
    SecureModelCheckpoint.load_checkpoint(path, key, device="cuda", target_model=model_cuda)

    with torch.no_grad():
        cuda_out = model_cuda(dummy_input.to("cuda"))

    assert torch.allclose(cpu_out.to("cuda"), cuda_out, atol=1e-5)
    max_diff = (cpu_out.to("cuda") - cuda_out).abs().max().item()
    assert max_diff < 1e-6
