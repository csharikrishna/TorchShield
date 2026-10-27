"""Defensive Deserialization and Untrusted Input Security Test Suite.

Section 6 Audit Requirements:
- Duplicate JSON keys rejection
- Malformed syntax and non-object roots rejection
- Unsupported PyTorch dtypes rejection
- Excessive tensor rank (> max_tensor_rank) rejection
- Negative or oversized dimension bounds rejection
- Element count integer overflow rejection
- Bounded memory limits enforcement
"""
from __future__ import annotations

import os
import struct
import tempfile
import pytest
import torch

from torchshield.checkpoint import (
    SecureModelCheckpoint,
    CheckpointError,
    ResourceLimits,
    _strict_json_loads,
    _canonical_json,
)


def test_strict_json_parser_rejects_duplicate_keys():
    """Verify that duplicate keys in JSON manifests raise CheckpointError."""
    payload_dup = b'{"version": 5, "key": 1, "key": 2}'
    with pytest.raises(CheckpointError, match="Duplicate key in manifest JSON: 'key'"):
        _strict_json_loads(payload_dup)


def test_strict_json_parser_rejects_non_object_roots():
    """Verify that arrays, numbers, or strings at root are rejected."""
    with pytest.raises(CheckpointError, match="Manifest root must be a JSON object"):
        _strict_json_loads(b'[1, 2, 3]')

    with pytest.raises(CheckpointError, match="Manifest root must be a JSON object"):
        _strict_json_loads(b'"hello world"')


def test_strict_json_parser_rejects_malformed_syntax():
    """Verify that malformed JSON payloads fail closed."""
    with pytest.raises(CheckpointError, match="Malformed manifest JSON payload"):
        _strict_json_loads(b'{"key": unquoted_value}')


def test_save_checkpoint_rejects_unsupported_dtypes():
    """Verify rejection of unapproved dtypes (e.g. complex64, qint8)."""
    # PyTorch complex tensor
    t = torch.tensor([1.0 + 2.0j], dtype=torch.complex64)
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "complex.shield")
        with pytest.raises(ValueError, match="has unsupported dtype 'complex64'"):
            SecureModelCheckpoint.save_checkpoint({"c": t}, path, key=key)


def test_save_checkpoint_rejects_excessive_tensor_rank():
    """Verify rejection of tensors with rank > max_tensor_rank (default 8)."""
    # 9-dimensional tensor
    t = torch.ones((1, 1, 1, 1, 1, 1, 1, 1, 1))
    assert len(t.shape) == 9
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "rank9.shield")
        with pytest.raises(ValueError, match="rank 9 exceeds maximum 8"):
            SecureModelCheckpoint.save_checkpoint({"t": t}, path, key=key)


def test_resource_limits_rejection_of_oversized_dimensions():
    """Verify that tensor dimensions exceeding max_dim_size are rejected before allocation."""
    limits = ResourceLimits(max_dim_size=100)
    t = torch.randn(101)  # Exceeds custom limit of 100
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "limit.shield")
        with pytest.raises(ValueError, match="Invalid dimension size 101.*limit: 100"):
            SecureModelCheckpoint.save_checkpoint({"t": t}, path, key=key, limits=limits)


def test_resource_limits_rejection_of_excessive_tensor_count():
    """Verify that state dicts with more than max_tensor_count items are rejected."""
    limits = ResourceLimits(max_tensor_count=5)
    state = {f"t_{i}": torch.randn(1) for i in range(6)}
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "count.shield")
        with pytest.raises(ValueError, match="Tensor count 6 exceeds maximum limit 5"):
            SecureModelCheckpoint.save_checkpoint(state, path, key=key, limits=limits)
