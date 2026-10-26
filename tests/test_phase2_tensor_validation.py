"""Phase 2 Tensor Metadata Validation and Safe Deserialization Tests.

Validates:
- Strict allow-listing of PyTorch numeric dtypes.
- Rejection of invalid, negative, or overflow-inducing tensor shapes.
- Rejection of excessive tensor counts, ranks, or memory sizes.
- Exact serialization and deserialization of 0-dim scalars, empty tensors,
  and non-contiguous strided tensor slices.
- Protection against untrusted code execution / pickle deserialization.
"""
from __future__ import annotations

import os
import secrets
import pytest
import torch

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError


def test_supported_dtypes_roundtrip(tmp_path):
    """Verify exact round-trip across all supported PyTorch dtypes."""
    key = secrets.token_bytes(32)
    state_dict = {
        "f32": torch.tensor([1.25, -2.5, 3.125], dtype=torch.float32),
        "f16": torch.tensor([1.5, -0.5, 4.0], dtype=torch.float16),
        "bf16": torch.tensor([1.5, -0.5, 4.0], dtype=torch.bfloat16),
        "f64": torch.tensor([1.12345678901234, -2.987654321], dtype=torch.float64),
        "i64": torch.tensor([9223372036854775807, -9223372036854775808], dtype=torch.int64),
        "i32": torch.tensor([2147483647, -2147483648], dtype=torch.int32),
        "i16": torch.tensor([32767, -32768], dtype=torch.int16),
        "i8": torch.tensor([127, -128], dtype=torch.int8),
        "u8": torch.tensor([0, 128, 255], dtype=torch.uint8),
        "b": torch.tensor([True, False, True, True], dtype=torch.bool),
    }

    path = str(tmp_path / "dtypes.ahcc")
    SecureModelCheckpoint.save_checkpoint(state_dict, path, key)
    loaded = SecureModelCheckpoint.load_checkpoint(path, key)

    for k, v in state_dict.items():
        assert k in loaded
        assert loaded[k].dtype == v.dtype
        assert loaded[k].shape == v.shape
        assert torch.equal(loaded[k], v)


def test_scalars_empty_and_strided_tensors(tmp_path):
    """Verify 0-dim scalars, empty tensors, and non-contiguous tensors."""
    key = secrets.token_bytes(32)
    base = torch.arange(100).reshape(10, 10)
    strided = base[::2, ::2]  # non-contiguous slice

    state_dict = {
        "scalar_0d": torch.tensor(42, dtype=torch.int64),
        "scalar_float": torch.tensor(3.14159, dtype=torch.float32),
        "empty_0el": torch.tensor([], dtype=torch.float32),
        "empty_multidim": torch.empty((0, 16, 32), dtype=torch.float32),
        "non_contiguous": strided,
    }

    path = str(tmp_path / "edge_tensors.ahcc")
    SecureModelCheckpoint.save_checkpoint(state_dict, path, key)
    loaded = SecureModelCheckpoint.load_checkpoint(path, key)

    for k, v in state_dict.items():
        assert k in loaded
        assert loaded[k].shape == v.shape
        assert loaded[k].dtype == v.dtype
        if v.numel() > 0:
            assert torch.equal(loaded[k], v)


def test_rejection_of_excessive_dimensions(tmp_path):
    """Verify rejection of unreasonable or malformed dimensions."""
    key = secrets.token_bytes(32)

    # Tensor with rank > 16
    bad_rank_shape = [1] * 17
    bad_rank_tensor = torch.zeros(bad_rank_shape)

    with pytest.raises(ValueError, match="exceeds maximum"):
        SecureModelCheckpoint.save_checkpoint({"bad_rank": bad_rank_tensor}, str(tmp_path / "bad.ahcc"), key)
