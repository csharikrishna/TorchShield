"""Secure Distributed and Sharded Checkpoint Coordination Engine.

Supports:
1. Multi-rank parallel sharded checkpoint saving (FSDP / Megatron-style model & optimizer state).
2. Global inventory manifest binding all ranks, SHA-256 file digests, tensor catalogs, and step counts.
3. Cryptographic publisher provenance via Ed25519 signatures over the global inventory.
4. Monotonic anti-rollback enforcement via FreshnessLedger.
5. Strict completeness verification preventing partial or mismatched rank restorations.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from torchshield.checkpoint import SecureModelCheckpoint, CheckpointError, ResourceLimits
from torchshield.key_providers.base import KeyProvider
from torchshield.provenance import Ed25519Signer, Ed25519Verifier, FreshnessLedger, ProvenanceError, RollbackError


class DistributedCheckpointError(CheckpointError):
    """Raised when distributed sharded checkpoint validation, inventory, or restoration fails."""


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def get_shard_filename(rank: int, world_size: int) -> str:
    """Standardized deterministic shard file name."""
    return f"shard_{rank:05d}_of_{world_size:05d}.shield"


def save_sharded_checkpoint(
    shard_tensors: Dict[str, torch.Tensor],
    checkpoint_dir: str,
    rank: int,
    world_size: int,
    key: Optional[bytes] = None,
    key_id: str = "",
    key_provider: Optional[KeyProvider] = None,
    encryption_context: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    signer_private_key: Optional[bytes] = None,
    chunk_bytes: int = 4 * 1024 * 1024,
    limits: Optional[ResourceLimits] = None,
) -> str:
    """Save a rank-specific shard to the distributed checkpoint directory."""
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size {world_size}.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    shard_filename = get_shard_filename(rank, world_size)
    shard_path = os.path.join(checkpoint_dir, shard_filename)

    shard_metadata = {
        "rank": rank,
        "world_size": world_size,
        "custom": metadata or {},
    }

    SecureModelCheckpoint.save_checkpoint(
        state_dict=shard_tensors,
        output_path=shard_path,
        key=key,
        chunk_bytes=chunk_bytes,
        key_id=key_id,
        metadata=shard_metadata,
        key_provider=key_provider,
        encryption_context=encryption_context,
        signer_private_key=signer_private_key,
        limits=limits,
    )

    return shard_path


def build_global_inventory(
    checkpoint_dir: str,
    run_id: str,
    global_step: int,
    world_size: int,
    epoch: int = 0,
    signer_private_key: Optional[bytes] = None,
) -> str:
    """Scan and cryptographically seal the global inventory manifest across all rank shards.

    Fail-closed: all world_size shards must be present on disk.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    shards_catalog: List[Dict[str, Any]] = []

    for rank in range(world_size):
        filename = get_shard_filename(rank, world_size)
        path = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(path):
            raise DistributedCheckpointError(
                f"Missing shard for rank {rank} of {world_size}: expected '{filename}'."
            )

        file_size = os.path.getsize(path)
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha256.update(chunk)

        shards_catalog.append({
            "rank": rank,
            "filename": filename,
            "size_bytes": file_size,
            "sha256": sha256.hexdigest(),
        })

    inventory_spec = {
        "version": 1,
        "format": "TorchShield-DistributedGlobalInventory-v1",
        "run_id": str(run_id),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "world_size": int(world_size),
        "shards": shards_catalog,
    }

    spec_bytes = _canonical_json(inventory_spec)
    inventory_digest = hashlib.sha256(spec_bytes).digest()

    global_manifest = dict(inventory_spec)
    global_manifest["inventory_digest"] = inventory_digest.hex()

    if signer_private_key is not None:
        priv_key = Ed25519PrivateKey.from_private_bytes(signer_private_key)
        signature = priv_key.sign(inventory_digest)
        pub_bytes = priv_key.public_key().public_bytes_raw()
        global_manifest["signature"] = signature.hex()
        global_manifest["signer_pubkey"] = pub_bytes.hex()

    manifest_bytes = json.dumps(global_manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest_path = os.path.join(checkpoint_dir, "global_manifest.json")
    tmp_path = os.path.join(checkpoint_dir, f".global_manifest_tmp_{uuid.uuid4().hex}")

    try:
        with open(tmp_path, "wb") as f:
            f.write(manifest_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, manifest_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    return manifest_path


def verify_sharded_checkpoint(
    checkpoint_dir: str,
    expected_signer_public_key: Optional[bytes] = None,
    ledger: Optional[FreshnessLedger] = None,
) -> Dict[str, Any]:
    """Verify inventory completeness, hashes, signatures, and freshness of a sharded checkpoint."""
    manifest_path = os.path.join(checkpoint_dir, "global_manifest.json")
    if not os.path.exists(manifest_path):
        raise DistributedCheckpointError(f"Missing global manifest: '{manifest_path}'.")

    with open(manifest_path, "rb") as f:
        try:
            manifest = json.loads(f.read().decode("utf-8"))
        except Exception as exc:
            raise DistributedCheckpointError(f"Malformed global manifest JSON: {exc}") from exc

    required_fields = ("run_id", "global_step", "epoch", "world_size", "shards", "inventory_digest")
    for req in required_fields:
        if req not in manifest:
            raise DistributedCheckpointError(f"Missing required field '{req}' in global manifest.")

    declared_digest = bytes.fromhex(manifest["inventory_digest"])
    reconstructed_spec = {
        "version": manifest.get("version", 1),
        "format": manifest.get("format", "TorchShield-DistributedGlobalInventory-v1"),
        "run_id": manifest["run_id"],
        "global_step": manifest["global_step"],
        "epoch": manifest["epoch"],
        "world_size": manifest["world_size"],
        "shards": manifest["shards"],
    }
    computed_digest = hashlib.sha256(_canonical_json(reconstructed_spec)).digest()
    if not secrets.compare_digest(computed_digest, declared_digest):
        raise DistributedCheckpointError("Global inventory specification digest mismatch.")

    # Signature verification
    if expected_signer_public_key is not None:
        if "signature" not in manifest or not manifest["signature"]:
            raise DistributedCheckpointError("Global manifest missing digital signature.")
        sig_bytes = bytes.fromhex(manifest["signature"])
        pub_key = Ed25519PublicKey.from_public_bytes(expected_signer_public_key)
        try:
            pub_key.verify(sig_bytes, declared_digest)
        except Exception as exc:
            raise DistributedCheckpointError(f"Global manifest signature verification failed: {exc}") from exc

    # Freshness ledger anti-rollback check
    if ledger is not None:
        ledger.check_freshness(
            run_id=manifest["run_id"],
            step=manifest["global_step"],
            epoch=manifest["epoch"],
        )

    # Shard completeness and integrity
    world_size = manifest["world_size"]
    shards = manifest["shards"]
    if len(shards) != world_size:
        raise DistributedCheckpointError(
            f"Shard count mismatch: expected {world_size}, found {len(shards)}."
        )

    for shard_record in shards:
        rank = shard_record["rank"]
        filename = shard_record["filename"]
        expected_size = shard_record["size_bytes"]
        expected_sha256 = shard_record["sha256"]

        path = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(path):
            raise DistributedCheckpointError(f"Missing shard file for rank {rank}: '{filename}'.")

        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            raise DistributedCheckpointError(
                f"File size mismatch for shard '{filename}': {actual_size} != {expected_size}."
            )

        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha.update(chunk)
        if sha.hexdigest() != expected_sha256:
            raise DistributedCheckpointError(f"SHA-256 mismatch for shard '{filename}'.")

    return manifest


def load_sharded_checkpoint(
    checkpoint_dir: str,
    rank: int,
    world_size: int,
    key: Optional[bytes] = None,
    key_provider: Optional[KeyProvider] = None,
    encryption_context: Optional[Dict[str, str]] = None,
    expected_signer_public_key: Optional[bytes] = None,
    ledger: Optional[FreshnessLedger] = None,
    device: Union[str, torch.device] = "cpu",
    limits: Optional[ResourceLimits] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Verify inventory and load the rank-specific shard."""
    manifest = verify_sharded_checkpoint(
        checkpoint_dir=checkpoint_dir,
        expected_signer_public_key=expected_signer_public_key,
        ledger=ledger,
    )

    if manifest["world_size"] != world_size:
        raise DistributedCheckpointError(
            f"Checkpoint world_size ({manifest['world_size']}) does not match requested ({world_size})."
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size {world_size}.")

    shard_filename = get_shard_filename(rank, world_size)
    shard_path = os.path.join(checkpoint_dir, shard_filename)

    tensors, meta = SecureModelCheckpoint.load_checkpoint(
        input_path=shard_path,
        key=key,
        device=device,
        limits=limits,
        return_metadata=True,
        key_provider=key_provider,
        encryption_context=encryption_context,
    )

    # Record freshness into ledger if provided
    if ledger is not None:
        ledger.record_and_verify(
            run_id=manifest["run_id"],
            step=manifest["global_step"],
            epoch=manifest["epoch"],
            artifact_id=meta.get("artifact_id", ""),
        )

    return tensors, {
        "global_manifest": manifest,
        "shard_meta": meta,
    }
