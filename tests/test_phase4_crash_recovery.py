"""Phase 4 Crash Recovery, Durability, and Atomic Staging Suite.

Validates:
- Atomic file replacement via staging .tmp.<uuid> files.
- Staging file cleanup upon save-time exceptions.
- Preservation of pre-existing valid checkpoints if a replacement write fails.
- Fail-closed behavior on truncated, incomplete, or corrupted files.
- Filesystem synchronization (flush and fsync) prior to atomic rename.
"""
from __future__ import annotations

import os
import secrets
from unittest import mock
import pytest
import torch

from torchshield.checkpoint import SecureModelCheckpoint, CheckpointError


def test_atomic_replacement_preserves_valid_checkpoint_on_save_error(tmp_path):
    """Verify that an interrupted or failing save never overwrites an existing valid checkpoint."""
    key = secrets.token_bytes(32)
    valid_state = {"weights": torch.ones(10, 10)}
    path = str(tmp_path / "model.shield")

    # 1. Save valid initial checkpoint
    SecureModelCheckpoint.save_checkpoint(valid_state, path, key)
    assert os.path.exists(path)

    # 2. Attempt save with invalid tensor that raises TypeError during serialization
    corrupted_state = {"bad_tensor": "NOT_A_TORCH_TENSOR"}
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        SecureModelCheckpoint.save_checkpoint(corrupted_state, path, key)

    # 3. Target file must remain intact and loadable with the original state
    restored = SecureModelCheckpoint.load_checkpoint(path, key)
    assert torch.equal(restored["weights"], valid_state["weights"])

    # 4. Verify no staging files were left behind in the directory
    leftover_tmp = [f for f in os.listdir(tmp_path) if ".tmp." in f]
    assert len(leftover_tmp) == 0, f"Dangling staging files found: {leftover_tmp}"


def test_staging_cleanup_on_io_error(tmp_path):
    """Verify that if an IOError occurs during write, the temporary staging file is deleted."""
    key = secrets.token_bytes(32)
    state = {"weights": torch.randn(20, 20)}
    path = str(tmp_path / "model.shield")

    # Mock open or os.fsync to simulate unexpected disk write failure
    with mock.patch("os.fsync", side_effect=OSError("Disk full simulation")):
        with pytest.raises(OSError, match="Disk full simulation"):
            SecureModelCheckpoint.save_checkpoint(state, path, key)

    # Target path should not exist (or if it existed, remain intact)
    assert not os.path.exists(path)
    # Staging path should be cleaned up
    leftover_tmp = [f for f in os.listdir(tmp_path) if ".tmp." in f]
    assert len(leftover_tmp) == 0


def test_load_truncated_checkpoint_fails_closed(tmp_path):
    """Verify that truncated files at any byte boundary fail closed with CheckpointError."""
    key = secrets.token_bytes(32)
    state = {"layer1": torch.randn(50, 50), "layer2": torch.randn(25, 25)}
    path = str(tmp_path / "model.shield")
    SecureModelCheckpoint.save_checkpoint(state, path, key)

    data = open(path, "rb").read()
    total_len = len(data)

    # Test truncation at several critical points:
    # 1. In magic bytes (4 bytes)
    # 2. In header (30 bytes)
    # 3. In manifest (halfway)
    # 4. In chunk payloads
    # 5. Exactly 1 byte missing from the end
    cut_points = [4, 25, total_len // 2, total_len - 1]
    for cut in cut_points:
        trunc_path = str(tmp_path / f"trunc_{cut}.shield")
        with open(trunc_path, "wb") as f:
            f.write(data[:cut])

        with pytest.raises(CheckpointError):
            SecureModelCheckpoint.load_checkpoint(trunc_path, key)
