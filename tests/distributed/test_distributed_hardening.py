"""Tests for distributed checkpoint hardening: path traversal, symlink rejection, rank uniqueness, and resource bounds."""
import json
import os
import secrets
import tempfile
import pytest
import torch

from torchshield.distributed import (
    save_sharded_checkpoint,
    build_global_inventory,
    verify_sharded_checkpoint,
    DistributedCheckpointError,
    get_shard_filename,
)
from torchshield.checkpoint import ResourceLimits


def test_distributed_hardening_rejects_duplicate_ranks(tmp_path):
    checkpoint_dir = str(tmp_path / "ckpt_dup_rank")
    os.makedirs(checkpoint_dir, exist_ok=True)
    key = secrets.token_bytes(32)

    # Save 2 valid shards
    tensors_0 = {"layer1": torch.tensor([1.0, 2.0])}
    tensors_1 = {"layer1": torch.tensor([3.0, 4.0])}
    save_sharded_checkpoint(tensors_0, checkpoint_dir, rank=0, world_size=2, key=key)
    save_sharded_checkpoint(tensors_1, checkpoint_dir, rank=1, world_size=2, key=key)
    build_global_inventory(checkpoint_dir, run_id="run_1", global_step=10, world_size=2)

    # Tamper with global_manifest.json to duplicate rank 0
    manifest_path = os.path.join(checkpoint_dir, "global_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Set both shard records to rank 0
    manifest["shards"][1]["rank"] = 0
    # Recompute specification digest so we test rank validation specifically
    from torchshield.distributed import _canonical_json
    import hashlib
    spec = {
        "version": manifest["version"],
        "format": manifest["format"],
        "run_id": manifest["run_id"],
        "global_step": manifest["global_step"],
        "epoch": manifest["epoch"],
        "world_size": manifest["world_size"],
        "shards": manifest["shards"],
    }
    manifest["inventory_digest"] = hashlib.sha256(_canonical_json(spec)).hexdigest()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    with pytest.raises(DistributedCheckpointError, match="Duplicate rank index detected"):
        verify_sharded_checkpoint(checkpoint_dir)


def test_distributed_hardening_rejects_path_traversal(tmp_path):
    checkpoint_dir = str(tmp_path / "ckpt_traversal")
    os.makedirs(checkpoint_dir, exist_ok=True)
    key = secrets.token_bytes(32)

    save_sharded_checkpoint({"w": torch.tensor([1.0])}, checkpoint_dir, rank=0, world_size=1, key=key)
    build_global_inventory(checkpoint_dir, run_id="run_1", global_step=5, world_size=1)

    manifest_path = os.path.join(checkpoint_dir, "global_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Inject path traversal in filename
    manifest["shards"][0]["filename"] = "../evil_shard.shield"
    from torchshield.distributed import _canonical_json
    import hashlib
    spec = {
        "version": manifest["version"],
        "format": manifest["format"],
        "run_id": manifest["run_id"],
        "global_step": manifest["global_step"],
        "epoch": manifest["epoch"],
        "world_size": manifest["world_size"],
        "shards": manifest["shards"],
    }
    manifest["inventory_digest"] = hashlib.sha256(_canonical_json(spec)).hexdigest()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    with pytest.raises(DistributedCheckpointError, match="Path traversal detected"):
        verify_sharded_checkpoint(checkpoint_dir)


def test_distributed_hardening_enforces_manifest_size_limits(tmp_path):
    checkpoint_dir = str(tmp_path / "ckpt_oversized_manifest")
    os.makedirs(checkpoint_dir, exist_ok=True)
    key = secrets.token_bytes(32)

    save_sharded_checkpoint({"w": torch.tensor([1.0])}, checkpoint_dir, rank=0, world_size=1, key=key)
    build_global_inventory(checkpoint_dir, run_id="run_1", global_step=5, world_size=1)

    limits = ResourceLimits(max_manifest_bytes=100)
    with pytest.raises(DistributedCheckpointError, match="exceeds limit"):
        verify_sharded_checkpoint(checkpoint_dir, limits=limits)
