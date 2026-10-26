"""Phase 2 Atomic Write and Crash Resilience Tests.

Validates:
- Checkpoints write to temporary staging files (.tmp.<uuid>) and atomic os.replace.
- If serialization or encryption fails mid-write, the temporary file is safely cleaned up.
- A previously existing valid checkpoint is never destroyed or truncated by a failed write attempt.
"""
from __future__ import annotations

import os
import secrets
import pytest
import torch

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError


def test_atomic_write_preserves_prior_file_on_failure(tmp_path):
    """Verify that an existing valid checkpoint remains intact if a subsequent save crashes."""
    key = secrets.token_bytes(32)
    valid_state = {"weights": torch.ones(50, 50)}
    path = str(tmp_path / "production_model.ahcc")

    # 1. Successfully write initial checkpoint
    SecureModelCheckpoint.save_checkpoint(valid_state, path, key)
    assert os.path.exists(path)
    initial_mtime = os.path.getmtime(path)

    # 2. Attempt to save corrupted/invalid state dict (e.g. non-tensor item)
    bad_state = {
        "valid_layer": torch.ones(10, 10),
        "corrupted_layer": "not_a_tensor_string_object",  # Will raise TypeError
    }

    with pytest.raises(TypeError):
        SecureModelCheckpoint.save_checkpoint(bad_state, path, key)

    # 3. Checkpoint file must still exist and be completely valid and unmodified
    assert os.path.exists(path)
    reloaded = SecureModelCheckpoint.load_checkpoint(path, key)
    assert torch.equal(reloaded["weights"], valid_state["weights"])

    # 4. No temporary staging files left behind in directory
    files = os.listdir(str(tmp_path))
    assert files == ["production_model.ahcc"]


def test_atomic_cleanup_on_disk_error(tmp_path, monkeypatch):
    """Verify temporary files are safely removed when I/O error occurs."""
    key = secrets.token_bytes(32)
    state = {"weights": torch.ones(20, 20)}
    path = str(tmp_path / "crashed_model.ahcc")

    # Monkeypatch os.fsync to simulate sudden I/O failure
    def failing_fsync(fd):
        raise OSError("Simulated disk full or I/O failure during fsync")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="Simulated disk full"):
        SecureModelCheckpoint.save_checkpoint(state, path, key)

    # Output file should not exist, and no temp files should remain
    assert not os.path.exists(path)
    assert os.listdir(str(tmp_path)) == []
