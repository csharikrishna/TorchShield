"""Tests for secure distributed and sharded checkpointing."""
import os
import shutil
import tempfile
import pytest
import torch

from torchshield.distributed import (
    save_sharded_checkpoint,
    build_global_inventory,
    load_sharded_checkpoint,
    verify_sharded_checkpoint,
    DistributedCheckpointError,
)
from torchshield.provenance import Ed25519Signer, FreshnessLedger, RollbackError
from torchshield.key_providers.local import LocalDevKeyProvider


def test_distributed_sharded_checkpoint_save_inventory_and_load():
    tmp_dir = tempfile.mkdtemp()
    try:
        WORLD_SIZE = 4
        key = os.urandom(32)
        signer = Ed25519Signer()

        # Step 1: Save 4 ranks
        for rank in range(WORLD_SIZE):
            shard_tensors = {
                f"layer_{rank}_weight": torch.randn(8, 8),
                f"layer_{rank}_bias": torch.randn(8),
            }
            save_sharded_checkpoint(
                shard_tensors=shard_tensors,
                checkpoint_dir=tmp_dir,
                rank=rank,
                world_size=WORLD_SIZE,
                key=key,
            )

        # Step 2: Build global inventory manifest with signature
        manifest_path = build_global_inventory(
            checkpoint_dir=tmp_dir,
            run_id="fsdp-training-run-001",
            global_step=5000,
            world_size=WORLD_SIZE,
            epoch=2,
            signer_private_key=signer.private_bytes,
        )
        assert os.path.exists(manifest_path)

        # Step 3: Verify manifest independently
        verified = verify_sharded_checkpoint(
            checkpoint_dir=tmp_dir,
            expected_signer_public_key=signer.public_bytes,
        )
        assert verified["world_size"] == WORLD_SIZE
        assert len(verified["shards"]) == WORLD_SIZE

        # Step 4: Load rank 2
        tensors_rank2, meta2 = load_sharded_checkpoint(
            checkpoint_dir=tmp_dir,
            rank=2,
            world_size=WORLD_SIZE,
            key=key,
            expected_signer_public_key=signer.public_bytes,
        )
        assert "layer_2_weight" in tensors_rank2
        assert tensors_rank2["layer_2_weight"].shape == (8, 8)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_distributed_checkpoint_detects_missing_shard():
    tmp_dir = tempfile.mkdtemp()
    try:
        WORLD_SIZE = 2
        key = os.urandom(32)

        # Only save rank 0
        save_sharded_checkpoint(
            shard_tensors={"w": torch.randn(4)},
            checkpoint_dir=tmp_dir,
            rank=0,
            world_size=WORLD_SIZE,
            key=key,
        )

        # Building inventory should fail because rank 1 is missing
        with pytest.raises(DistributedCheckpointError, match="Missing shard for rank 1"):
            build_global_inventory(
                checkpoint_dir=tmp_dir,
                run_id="run-missing",
                global_step=10,
                world_size=WORLD_SIZE,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_distributed_checkpoint_detects_tampered_shard():
    tmp_dir = tempfile.mkdtemp()
    try:
        WORLD_SIZE = 2
        key = os.urandom(32)

        for rank in range(WORLD_SIZE):
            save_sharded_checkpoint(
                shard_tensors={"w": torch.randn(4)},
                checkpoint_dir=tmp_dir,
                rank=rank,
                world_size=WORLD_SIZE,
                key=key,
            )

        build_global_inventory(
            checkpoint_dir=tmp_dir,
            run_id="run-tamper",
            global_step=100,
            world_size=WORLD_SIZE,
        )

        # Tamper with rank 1 file
        shard1_path = os.path.join(tmp_dir, "shard_00001_of_00002.shield")
        with open(shard1_path, "r+b") as f:
            f.seek(50)
            f.write(b"\xFF\xFF")

        # Loading must fail closed
        with pytest.raises(DistributedCheckpointError, match="SHA-256 mismatch"):
            verify_sharded_checkpoint(checkpoint_dir=tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_distributed_checkpoint_enforces_anti_rollback():
    tmp_dir = tempfile.mkdtemp()
    ledger_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    ledger_path = ledger_file.name
    ledger_file.close()

    try:
        WORLD_SIZE = 2
        key = os.urandom(32)
        ledger = FreshnessLedger(ledger_path)

        for rank in range(WORLD_SIZE):
            save_sharded_checkpoint(
                shard_tensors={"w": torch.randn(4)},
                checkpoint_dir=tmp_dir,
                rank=rank,
                world_size=WORLD_SIZE,
                key=key,
            )

        # Checkpoint at step 1000
        build_global_inventory(
            checkpoint_dir=tmp_dir,
            run_id="distributed-run-001",
            global_step=1000,
            world_size=WORLD_SIZE,
        )

        # First load succeeds and records step 1000
        load_sharded_checkpoint(
            checkpoint_dir=tmp_dir,
            rank=0,
            world_size=WORLD_SIZE,
            key=key,
            ledger=ledger,
        )
        assert ledger.get_run_info("distributed-run-001")["max_step"] == 1000

        # Rollback attempt: checkpoint claimed to be step 500
        build_global_inventory(
            checkpoint_dir=tmp_dir,
            run_id="distributed-run-001",
            global_step=500,
            world_size=WORLD_SIZE,
        )

        with pytest.raises(RollbackError, match="Rollback attack detected"):
            load_sharded_checkpoint(
                checkpoint_dir=tmp_dir,
                rank=0,
                world_size=WORLD_SIZE,
                key=key,
                ledger=ledger,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(ledger_path):
            os.remove(ledger_path)
