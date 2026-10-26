"""Phase 4 Resource Limits and Denial-of-Service Defense Suite.

Validates:
- Configurable ResourceLimits dataclass.
- Pre-allocation rejection of excessive tensor ranks, dimensions, and element counts.
- Rejection of oversized manifest declarations.
- Rejection of oversized file lengths.
- Operator customization of limits for constrained or high-memory environments.
"""
from __future__ import annotations

import secrets
import struct
import pytest
import torch

from ahcc.checkpoint import (
    SecureModelCheckpoint,
    CheckpointError,
    ResourceLimits,
    DEFAULT_LIMITS,
)


def test_resource_limits_defaults():
    """Verify that default limits are realistic and protective."""
    limits = DEFAULT_LIMITS
    assert limits.max_tensor_rank == 8
    assert limits.max_dim_size == 100_000_000
    assert limits.max_manifest_bytes == 16 * 1024 * 1024
    assert limits.max_tensor_count == 50_000
    assert limits.max_chunk_count == 1_000_000
    assert limits.max_total_raw_bytes == 64 * 1024 * 1024 * 1024


def test_custom_limits_rejection_on_save(tmp_path):
    """Verify that operator can configure strict custom limits to reject large tensors."""
    strict_limits = ResourceLimits(
        max_tensor_count=2,
        max_dim_size=50,
        max_chunk_bytes=1024 * 1024,
    )
    key = secrets.token_bytes(32)

    # 1. Test exceeding max_tensor_count
    excessive_state = {f"t_{i}": torch.randn(2, 2) for i in range(3)}
    with pytest.raises(ValueError, match="Tensor count 3 exceeds maximum limit 2"):
        SecureModelCheckpoint.save_checkpoint(
            excessive_state,
            str(tmp_path / "excess_tensors.ahcc"),
            key,
            limits=strict_limits,
        )

    # 2. Test exceeding max_dim_size
    large_dim_state = {"large": torch.randn(10, 100)}  # dim 100 > 50
    with pytest.raises(ValueError, match="Invalid dimension size 100 in tensor 'large'"):
        SecureModelCheckpoint.save_checkpoint(
            large_dim_state,
            str(tmp_path / "large_dim.ahcc"),
            key,
            chunk_bytes=512 * 1024,
            limits=strict_limits,
        )


def test_custom_limits_rejection_on_load(tmp_path):
    """Verify that a checkpoint saved with standard limits is rejected when loaded under strict limits."""
    key = secrets.token_bytes(32)
    state = {
        "layer_a": torch.randn(20, 20),
        "layer_b": torch.randn(20, 20),
        "layer_c": torch.randn(20, 20),
    }
    path = str(tmp_path / "model.ahcc")
    SecureModelCheckpoint.save_checkpoint(state, path, key)

    # Try loading with max_tensor_count = 2
    strict_limits = ResourceLimits(max_tensor_count=2)
    with pytest.raises(CheckpointError, match="Tensor count 3 exceeds maximum limit 2"):
        SecureModelCheckpoint.load_checkpoint(path, key, limits=strict_limits)


def test_manifest_length_ceiling_enforcement(tmp_path):
    """Verify that declared manifest length exceeding limits.max_manifest_bytes is rejected before reading."""
    key = secrets.token_bytes(32)
    state = {"w": torch.randn(4, 4)}
    path = str(tmp_path / "model.ahcc")
    SecureModelCheckpoint.save_checkpoint(state, path, key)

    # Load with tiny manifest limit
    tiny_limits = ResourceLimits(max_manifest_bytes=100)
    with pytest.raises(CheckpointError, match="Invalid manifest length"):
        SecureModelCheckpoint.load_checkpoint(path, key, limits=tiny_limits)


def test_file_size_ceiling_enforcement(tmp_path):
    """Verify that file size exceeding limits.max_file_size is rejected before processing."""
    key = secrets.token_bytes(32)
    state = {"w": torch.randn(100, 100)}
    path = str(tmp_path / "model.ahcc")
    SecureModelCheckpoint.save_checkpoint(state, path, key)

    # Set max_file_size below actual file size
    actual_size = len(open(path, "rb").read())
    tiny_limits = ResourceLimits(max_file_size=actual_size // 2)

    with pytest.raises(CheckpointError, match="exceeds limit"):
        SecureModelCheckpoint.load_checkpoint(path, key, limits=tiny_limits)
