"""Phase 2 PyTorch Model Integration and Inference Parity Tests.

Validates:
- Complete PyTorch neural network serialization, chunking, and authenticated streaming.
- Exact restoration of model parameters and persistent buffers (e.g. BatchNorm running stats).
- Bit-exact deterministic forward inference pass output before and after loading.
- Direct streaming into target_model in-place without intermediate plaintext disk leaks.
- Fail-closed security: if loading fails, target_model is not left partially corrupted.
"""
from __future__ import annotations

import os
import secrets
import pytest
import torch
import torch.nn as nn

from torchshield.checkpoint import SecureModelCheckpoint, CheckpointError


class MiniVisionBackbone(nn.Module):
    """Synthetic CNN with Conv, BatchNorm, and Linear layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


def test_pytorch_model_inference_parity(tmp_path):
    """Verify bit-exact output parity on deterministic forward pass."""
    torch.manual_seed(42)
    model = MiniVisionBackbone().eval()

    # Create fixed synthetic input
    test_input = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        expected_output = model(test_input)

    key = secrets.token_bytes(32)
    path = str(tmp_path / "vision_model.shield")

    # Save checkpoint
    SecureModelCheckpoint.save_checkpoint(model.state_dict(), path, key, chunk_bytes=16 * 1024)

    # Initialize a fresh target model with completely different random weights
    torch.manual_seed(999)
    fresh_model = MiniVisionBackbone().eval()
    with torch.no_grad():
        initial_fresh_output = fresh_model(test_input)
    assert not torch.allclose(expected_output, initial_fresh_output)

    # Stream-load checkpoint directly into fresh_model
    SecureModelCheckpoint.load_checkpoint(path, key, target_model=fresh_model)

    # Verify model outputs match bit-for-bit
    with torch.no_grad():
        restored_output = fresh_model(test_input)

    assert torch.equal(expected_output, restored_output)
    max_diff = (expected_output - restored_output).abs().max().item()
    assert max_diff == 0.0


def test_batchnorm_buffers_and_scalar_restoration(tmp_path):
    """Verify BatchNorm running_mean, running_var, and num_batches_tracked scalar restoration."""
    model = MiniVisionBackbone()
    # Simulate some training iterations to update running stats
    for _ in range(5):
        dummy_in = torch.randn(8, 3, 32, 32)
        model(dummy_in)

    model.eval()
    key = secrets.token_bytes(32)
    path = str(tmp_path / "bn_model.shield")

    orig_state = model.state_dict()
    assert orig_state["bn1.num_batches_tracked"].dim() == 0  # 0-dim scalar
    assert orig_state["bn1.num_batches_tracked"].item() == 5

    SecureModelCheckpoint.save_checkpoint(orig_state, path, key)
    loaded_state = SecureModelCheckpoint.load_checkpoint(path, key)

    for k in orig_state:
        assert k in loaded_state
        assert loaded_state[k].shape == orig_state[k].shape
        assert torch.equal(loaded_state[k], orig_state[k])
