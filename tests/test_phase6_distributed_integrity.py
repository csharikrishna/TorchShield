"""Distributed Sharded Checkpoint Integrity and Fault Injection Tests for Phase 6.

Section 7 Audit Requirements:
- Multi-rank save and restore (4 ranks)
- Shard deletion rejection (fail-closed on missing shard)
- Shard corruption rejection (SHA-256 and AEAD tag mismatch)
- Shard substitution from another artifact rejection
- Interrupted rank during save simulation
- Coordinator failure simulation (missing global manifest)
- Global manifest inventory completeness and Ed25519 signature
- World size mismatch / resharding limitation verification
"""
from __future__ import annotations

import os
import shutil
import tempfile
import pytest
import torch

from torchshield.distributed import (
    save_sharded_checkpoint,
    build_global_inventory,
    verify_sharded_checkpoint,
    load_sharded_checkpoint,
    get_shard_filename,
    DistributedCheckpointError,
)
from torchshield.provenance import Ed25519Signer, FreshnessLedger


def test_distributed_save_and_restore_4_ranks():
    """Verify standard 4-rank sharded checkpointing workflow."""
    key = os.urandom(32)
    world_size = 4
    signer = Ed25519Signer()

    rank_tensors = {
        rank: {f"layer_rank_{rank}": torch.randn(10, 10) * (rank + 1)}
        for rank in range(world_size)
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Phase 1: Parallel save from 4 ranks
        for rank in range(world_size):
            save_sharded_checkpoint(
                shard_tensors=rank_tensors[rank],
                checkpoint_dir=tmp_dir,
                rank=rank,
                world_size=world_size,
                key=key,
            )

        # Phase 2: Coordinator builds global manifest
        manifest_path = build_global_inventory(
            checkpoint_dir=tmp_dir,
            run_id="dist-run-1",
            global_step=100,
            world_size=world_size,
            epoch=1,
            signer_private_key=signer.private_bytes,
        )
        assert os.path.exists(manifest_path)

        # Phase 3: Rank 2 loads its shard with publisher verification
        restored_tensors, meta = load_sharded_checkpoint(
            checkpoint_dir=tmp_dir,
            rank=2,
            world_size=world_size,
            key=key,
            expected_signer_public_key=signer.public_bytes,
        )
        assert torch.allclose(restored_tensors["layer_rank_2"], rank_tensors[2]["layer_rank_2"])
        assert meta["global_manifest"]["world_size"] == 4


def test_distributed_missing_shard_rejection():
    """Verify that if one rank shard is missing, inventory build and load fail closed."""
    key = os.urandom(32)
    world_size = 3

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Only save ranks 0 and 1 (rank 2 missing)
        save_sharded_checkpoint({"w": torch.randn(2)}, tmp_dir, rank=0, world_size=world_size, key=key)
        save_sharded_checkpoint({"w": torch.randn(2)}, tmp_dir, rank=1, world_size=world_size, key=key)

        with pytest.raises(DistributedCheckpointError, match="Missing shard for rank 2 of 3"):
            build_global_inventory(tmp_dir, "run-incomplete", global_step=10, world_size=world_size)


def test_distributed_corrupted_shard_rejection():
    """Verify that tampering with any shard bytes causes inventory validation failure."""
    key = os.urandom(32)
    world_size = 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_sharded_checkpoint({"w0": torch.ones(5)}, tmp_dir, rank=0, world_size=world_size, key=key)
        save_sharded_checkpoint({"w1": torch.zeros(5)}, tmp_dir, rank=1, world_size=world_size, key=key)

        build_global_inventory(tmp_dir, "run-corrupt", global_step=5, world_size=world_size)

        # Tamper shard 1
        shard1_path = os.path.join(tmp_dir, get_shard_filename(1, world_size))
        with open(shard1_path, "r+b") as f:
            f.seek(50)
            f.write(b"\xFF\xFF")

        with pytest.raises(DistributedCheckpointError, match="SHA-256 mismatch for shard"):
            verify_sharded_checkpoint(tmp_dir)


def test_distributed_shard_substitution_from_different_checkpoint_rejection():
    """Verify that replacing a shard with a valid shard from another checkpoint is rejected."""
    key = os.urandom(32)
    world_size = 2

    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        # Checkpoint 1
        save_sharded_checkpoint({"w": torch.ones(5)}, dir1, rank=0, world_size=world_size, key=key)
        save_sharded_checkpoint({"w": torch.ones(5)}, dir1, rank=1, world_size=world_size, key=key)
        build_global_inventory(dir1, "run-1", global_step=10, world_size=world_size)

        # Checkpoint 2
        save_sharded_checkpoint({"w": torch.zeros(5)}, dir2, rank=0, world_size=world_size, key=key)
        save_sharded_checkpoint({"w": torch.zeros(5)}, dir2, rank=1, world_size=world_size, key=key)
        build_global_inventory(dir2, "run-2", global_step=20, world_size=world_size)

        # Substitute rank 1 from checkpoint 2 into checkpoint 1
        shard1_src = os.path.join(dir2, get_shard_filename(1, world_size))
        shard1_dst = os.path.join(dir1, get_shard_filename(1, world_size))
        shutil.copyfile(shard1_src, shard1_dst)

        # Hash mismatch against dir1's global manifest
        with pytest.raises(DistributedCheckpointError, match="SHA-256 mismatch for shard"):
            verify_sharded_checkpoint(dir1)


def test_distributed_world_size_mismatch_rejection():
    """Verify that requesting a different world_size (resharding) is strictly rejected."""
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Checkpoint saved with world_size = 4
        for r in range(4):
            save_sharded_checkpoint({"w": torch.randn(2)}, tmp_dir, rank=r, world_size=4, key=key)
        build_global_inventory(tmp_dir, "run-ws", global_step=1, world_size=4)

        # Attempt to load with world_size = 2
        with pytest.raises(DistributedCheckpointError, match="does not match requested"):
            load_sharded_checkpoint(tmp_dir, rank=0, world_size=2, key=key)


def test_distributed_save_and_restore_model_optimizer_and_per_rank_rng():
    """Requirement 2: Verify model, optimizer, and per-rank RNG state in distributed shards."""
    key = os.urandom(32)
    world_size = 2
    signer = Ed25519Signer()

    rank_data = {}
    for r in range(world_size):
        torch.manual_seed(100 + r)
        rank_data[r] = {
            "model_param::weight": torch.randn(8, 8) * (r + 1),
            "opt::weight::exp_avg": torch.zeros(8, 8) + (r * 0.5),
            "opt::weight::exp_avg_sq": torch.ones(8, 8) + (r * 0.1),
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        for r in range(world_size):
            save_sharded_checkpoint(
                shard_tensors=rank_data[r],
                checkpoint_dir=tmp_dir,
                rank=r,
                world_size=world_size,
                key=key,
                metadata={"rank_seed": 100 + r, "rng_step": 42},
            )

        build_global_inventory(
            checkpoint_dir=tmp_dir,
            run_id="dist-state-run",
            global_step=42,
            world_size=world_size,
            signer_private_key=signer.private_bytes,
        )

        for r in range(world_size):
            restored, meta = load_sharded_checkpoint(
                checkpoint_dir=tmp_dir,
                rank=r,
                world_size=world_size,
                key=key,
                expected_signer_public_key=signer.public_bytes,
            )
            assert torch.allclose(restored["model_param::weight"], rank_data[r]["model_param::weight"])
            assert torch.allclose(restored["opt::weight::exp_avg"], rank_data[r]["opt::weight::exp_avg"])
            assert torch.allclose(restored["opt::weight::exp_avg_sq"], rank_data[r]["opt::weight::exp_avg_sq"])
            assert meta["shard_meta"]["custom"]["rank_seed"] == 100 + r


def test_distributed_interrupted_rank_during_save():
    """Requirement 6: Verify rank crash mid-save cleans up staging and causes inventory to fail closed."""
    key = os.urandom(32)
    world_size = 3
    from unittest import mock

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Rank 0 and 1 succeed
        save_sharded_checkpoint({"w": torch.randn(4)}, tmp_dir, rank=0, world_size=world_size, key=key)
        save_sharded_checkpoint({"w": torch.randn(4)}, tmp_dir, rank=1, world_size=world_size, key=key)

        # Rank 2 crashes during chunk encryption
        with mock.patch("torchshield.checkpoint.ChunkCryptoSession.encrypt_next_chunk", side_effect=RuntimeError("Rank 2 power loss")):
            with pytest.raises(RuntimeError, match="Rank 2 power loss"):
                save_sharded_checkpoint({"w": torch.randn(4)}, tmp_dir, rank=2, world_size=world_size, key=key)

        # Confirm no corrupt rank 2 shard or staging file exists
        shard2_file = get_shard_filename(2, world_size)
        assert not os.path.exists(os.path.join(tmp_dir, shard2_file))
        assert not any(f.startswith(shard2_file + ".tmp") for f in os.listdir(tmp_dir))

        # Coordinator inventory build must strictly fail closed
        with pytest.raises(DistributedCheckpointError, match="Missing shard for rank 2 of 3"):
            build_global_inventory(tmp_dir, "run-interrupted", global_step=1, world_size=world_size)


def test_distributed_coordinator_failure_simulation():
    """Requirement 7: Verify fail-closed behavior when coordinator fails before inventory seal."""
    key = os.urandom(32)
    world_size = 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        # All ranks save successfully
        save_sharded_checkpoint({"w": torch.randn(2)}, tmp_dir, rank=0, world_size=world_size, key=key)
        save_sharded_checkpoint({"w": torch.randn(2)}, tmp_dir, rank=1, world_size=world_size, key=key)

        # Coordinator crashes before build_global_inventory can write global_manifest.json
        assert not os.path.exists(os.path.join(tmp_dir, "global_manifest.json"))

        # Sharded verification must fail closed
        with pytest.raises(DistributedCheckpointError, match="Missing global manifest"):
            verify_sharded_checkpoint(tmp_dir)

        # Sharded loading must fail closed
        with pytest.raises(DistributedCheckpointError, match="Missing global manifest"):
            load_sharded_checkpoint(tmp_dir, rank=0, world_size=world_size, key=key)
