"""Secure Distributed Sharded Checkpointing Demonstration.

Demonstrates:
1. Multi-rank parallel shard generation and atomic persistence.
2. Global manifest aggregation, SHA-256 verification, and Ed25519 signing.
3. Strict inventory completeness verification (detecting missing/tampered shards).
4. Anti-rollback freshness ledger tracking monotonic steps.
5. Rank-specific shard restoration.
"""
import os
import shutil
import tempfile
import torch

from ahcc.distributed import (
    save_sharded_checkpoint,
    build_global_inventory,
    load_sharded_checkpoint,
    verify_sharded_checkpoint,
)
from ahcc.provenance import Ed25519Signer, FreshnessLedger


def run_distributed_demo():
    print("================================================================================")
    print("  AHCC SECURE DISTRIBUTED SHARDED CHECKPOINT DEMONSTRATION")
    print("  Phase 5 Multi-Rank Coordination, Inventory Seal & Rollback Protection")
    print("================================================================================\n")

    WORLD_SIZE = 4
    RUN_ID = "fsdp-llama3-finetune-node01"
    STEP = 10000
    EPOCH = 3

    checkpoint_dir = tempfile.mkdtemp(prefix="ahcc_dist_demo_")
    ledger_path = os.path.join(checkpoint_dir, "freshness_ledger.json")

    try:
        key = os.urandom(32)
        signer = Ed25519Signer()
        ledger = FreshnessLedger(ledger_path)

        print(f"[*] Configuration: world_size={WORLD_SIZE}, run_id='{RUN_ID}', step={STEP}, epoch={EPOCH}")
        print(f"[*] Staging Directory: '{checkpoint_dir}'")

        # ----------------------------------------------------------------------
        # 1. Parallel Shard Saving
        # ----------------------------------------------------------------------
        print("\n[Phase 1] Simulating parallel worker ranks saving their shards...")
        for rank in range(WORLD_SIZE):
            shard_data = {
                f"transformer.block_{rank}.attn.qkv": torch.randn(64, 64),
                f"transformer.block_{rank}.attn.proj": torch.randn(64, 64),
                f"transformer.block_{rank}.mlp.fc1": torch.randn(64, 256),
                f"transformer.block_{rank}.mlp.fc2": torch.randn(256, 64),
            }
            path = save_sharded_checkpoint(
                shard_tensors=shard_data,
                checkpoint_dir=checkpoint_dir,
                rank=rank,
                world_size=WORLD_SIZE,
                key=key,
                metadata={"rank": rank, "sharding_type": "FSDP_HYBRID"},
            )
            print(f"  -> Rank {rank}/{WORLD_SIZE} saved: {os.path.basename(path)}")

        # ----------------------------------------------------------------------
        # 2. Coordinator Inventory Manifest Aggregation & Ed25519 Signature
        # ----------------------------------------------------------------------
        print("\n[Phase 2] Coordinator sealing global manifest inventory with Ed25519 signature...")
        manifest_path = build_global_inventory(
            checkpoint_dir=checkpoint_dir,
            run_id=RUN_ID,
            global_step=STEP,
            world_size=WORLD_SIZE,
            epoch=EPOCH,
            signer_private_key=signer.private_bytes,
        )
        print(f"  -> Global manifest written to: {os.path.basename(manifest_path)}")

        # ----------------------------------------------------------------------
        # 3. Independent Verification
        # ----------------------------------------------------------------------
        print("\n[Phase 3] Verifying inventory integrity, completeness, and digital signature...")
        verified_manifest = verify_sharded_checkpoint(
            checkpoint_dir=checkpoint_dir,
            expected_signer_public_key=signer.public_bytes,
            ledger=ledger,
        )
        print("  -> Signature: VALID [OK]")
        print("  -> Inventory Completeness: 4/4 SHARDS PRESENT [OK]")
        print("  -> SHA-256 Digest Verification: ALL PASS [OK]")

        # ----------------------------------------------------------------------
        # 4. Rank Shard Loading & Anti-Rollback
        # ----------------------------------------------------------------------
        print("\n[Phase 4] Rank worker loading shard with anti-rollback ledger...")
        tensors, meta = load_sharded_checkpoint(
            checkpoint_dir=checkpoint_dir,
            rank=2,
            world_size=WORLD_SIZE,
            key=key,
            expected_signer_public_key=signer.public_bytes,
            ledger=ledger,
        )

        print(f"  -> Successfully loaded Rank 2 shard ({len(tensors)} tensor layers)")
        print(f"  -> Ledger updated monotonic step: {ledger.get_run_info(RUN_ID)['max_step']}")

        print("\n================================================================================")
        print("  RESULT: VERIFIED")
        print("  Distributed sharded checkpoint successfully saved, verified, and loaded.")
        print("================================================================================")

    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    run_distributed_demo()
