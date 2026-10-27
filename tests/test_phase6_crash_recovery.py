"""Crash Consistency, Atomic Durability, and Staging Cleanup Tests for Phase 6.

Section 9 Audit Requirements:
- Exceptions during encryption (verify temp file cleanup)
- Simulated disk-full / write error during file output (verify destination intact)
- Atomic replacement guarantee (os.replace + fsync)
- Truncated file detection
- Failed KMS unwrapping recovery
- Staged freshness update crash recovery
"""
from __future__ import annotations

import os
import tempfile
from unittest import mock
import pytest
import torch

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError
from ahcc.provenance import TransactionalMemoryFreshnessBackend, RollbackError


def test_exception_during_chunk_encryption_cleans_staging_file():
    """Verify that an exception mid-encryption removes the temporary staging file."""
    state = {"w": torch.randn(10, 10)}
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "crash_target.ahcc")

        # Mock encrypt_next_chunk to simulate an unhandled exception on the first chunk
        with mock.patch("ahcc.checkpoint._ChunkCryptoSession.encrypt_next_chunk", side_effect=RuntimeError("Simulated encryption hardware fault")):
            with pytest.raises(RuntimeError, match="Simulated encryption hardware fault"):
                SecureModelCheckpoint.save_checkpoint(state, output_path, key=key)

        # Destination must not exist
        assert not os.path.exists(output_path)
        # Staging file must be cleaned up
        remaining_files = os.listdir(tmp_dir)
        assert len(remaining_files) == 0, f"Orphaned staging files found: {remaining_files}"


def test_simulated_disk_full_leaves_prior_checkpoint_intact():
    """Verify that a disk full exception during writing does not corrupt the existing destination file."""
    original_state = {"w": torch.ones(5, 5)}
    new_state = {"w": torch.zeros(5, 5)}
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = os.path.join(tmp_dir, "existing.ahcc")

        # Save valid existing checkpoint
        SecureModelCheckpoint.save_checkpoint(original_state, target_path, key=key)
        initial_size = os.path.getsize(target_path)
        with open(target_path, "rb") as f:
            initial_bytes = f.read()

        # Simulate OSError(ENOSPC, "No space left on device") during write
        with mock.patch("os.fsync", side_effect=OSError(28, "No space left on device")):
            with pytest.raises(OSError, match="No space left on device"):
                SecureModelCheckpoint.save_checkpoint(new_state, target_path, key=key)

        # Existing destination file must remain completely uncorrupted
        assert os.path.exists(target_path)
        assert os.path.getsize(target_path) == initial_size
        with open(target_path, "rb") as f:
            assert f.read() == initial_bytes

        # Checkpoint can still be decrypted and yields original weights
        restored = SecureModelCheckpoint.load_checkpoint(target_path, key=key)
        assert torch.allclose(restored["w"], original_state["w"])


def test_truncated_checkpoint_files_fail_closed():
    """Verify that checkpoint files truncated at various byte offsets fail closed."""
    state = {"layer.weight": torch.randn(50, 50)}
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        valid_path = os.path.join(tmp_dir, "valid.ahcc")
        SecureModelCheckpoint.save_checkpoint(state, valid_path, key=key)

        with open(valid_path, "rb") as f:
            valid_bytes = f.read()

        truncation_points = [
            4,    # Truncated magic
            15,   # Truncated salt
            50,   # Truncated header
            100,  # Truncated manifest
            len(valid_bytes) - 10,  # Truncated chunk / tag
        ]

        for idx, cut in enumerate(truncation_points):
            t_path = os.path.join(tmp_dir, f"trunc_{idx}.ahcc")
            with open(t_path, "wb") as f:
                f.write(valid_bytes[:cut])

            with pytest.raises(CheckpointError):
                SecureModelCheckpoint.load_checkpoint(t_path, key=key)


def test_staged_freshness_update_abort_and_recovery():
    """Verify that an interrupted write stages and safely aborts a freshness update token."""
    backend = TransactionalMemoryFreshnessBackend()
    backend.advance("run-crash", step=10, epoch=1)

    # Worker begins write of step 20
    token = backend.begin_update("run-crash", step=20, epoch=1)

    # Worker crashes: abort update
    backend.abort_update("run-crash", token)

    # Verify run state remains at step 10
    rec = backend.get_record("run-crash")
    assert rec.max_step == 10
    assert rec.pending_recovery_token is None

    # Next attempt succeeds cleanly
    backend.advance("run-crash", step=20, epoch=1)
    assert backend.get_record("run-crash").max_step == 20
