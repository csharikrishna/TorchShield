"""Phase 7 Rigorous End-to-End Training Recovery Validation Suite.

Validates:
1. Multi-seed, multi-restart bit-for-bit deterministic equivalence:
   - Every single step compared (not just milestones)
   - Model parameters AND buffers (BatchNorm running_mean/var)
   - Optimizer momentum and adaptive moments (AdamW exp_avg, exp_avg_sq)
   - Learning-rate scheduler state (StepLR)
   - Mixed precision GradScaler state
   - Full RNG determinism across Python, NumPy, PyTorch CPU and CUDA
   - Consecutive restarts (restart at step 10 -> resume to 20 -> restart -> resume to 30)
   - Evaluated across multiple random seeds (42, 2026)
2. Fail-closed recovery under adverse conditions:
   - Incomplete / truncated checkpoint container
   - Corrupted ciphertext chunk
   - Corrupted manifest payload
   - Wrong encryption key
   - Missing or tampered shard in distributed checkpoint
"""
from __future__ import annotations

import copy
import hashlib
import os
import random
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from ahcc.training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    TrainingProgress,
    TrainingStateError,
)
from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError
from ahcc.distributed import save_sharded_checkpoint, load_sharded_checkpoint, build_global_inventory, DistributedCheckpointError
from ahcc.key_providers.local import LocalDevKeyProvider
from ahcc.provenance import Ed25519Signer


class ComprehensiveTestModel(nn.Module):
    """Model featuring weights, biases, and persistent buffers (BatchNorm)."""
    def __init__(self, in_features: int = 16, hidden: int = 32, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.bn(self.fc1(x))))


@pytest.mark.parametrize("seed", [42, 2026])
def test_multi_seed_and_consecutive_restart_bit_for_bit_equivalence(seed: int):
    """Validate that interrupted runs with consecutive restarts match continuous runs at EVERY step."""
    TOTAL_STEPS = 30
    CHECKPOINT_STEP_1 = 10
    CHECKPOINT_STEP_2 = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic dataset
    torch.manual_seed(seed)
    dataset = [
        (torch.randn(16, 16, device=device), torch.randint(0, 4, (16,), device=device))
        for _ in range(TOTAL_STEPS)
    ]
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # Run A: Uninterrupted Reference Run (0 -> TOTAL_STEPS)
    # -------------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_ref = ComprehensiveTestModel().to(device)
    opt_ref = AdamW(model_ref.parameters(), lr=1e-2, weight_decay=1e-4)
    sched_ref = StepLR(opt_ref, step_size=8, gamma=0.5)

    ref_losses = []
    ref_preds = []

    for step in range(TOTAL_STEPS):
        opt_ref.zero_grad()
        x, y = dataset[step]
        # Introduce stochastic jitter to verify Python & NumPy RNG tracking
        jitter = random.random() * 1e-7 + np.random.randn() * 1e-7
        x_noisy = x + jitter
        pred = model_ref(x_noisy)
        loss = criterion(pred, y)
        loss.backward()
        opt_ref.step()
        sched_ref.step()
        ref_losses.append(loss.item())
        ref_preds.append(pred.detach().cpu())

    # -------------------------------------------------------------------------
    # Run B: Interrupted at Step 10, Resumed to Step 20, Interrupted, Resumed to 30
    # -------------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_b = ComprehensiveTestModel().to(device)
    opt_b = AdamW(model_b.parameters(), lr=1e-2, weight_decay=1e-4)
    sched_b = StepLR(opt_b, step_size=8, gamma=0.5)

    run_b_losses = []
    run_b_preds = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path_1 = os.path.join(tmp_dir, "ckpt_step_10.ahcc")
        ckpt_path_2 = os.path.join(tmp_dir, "ckpt_step_20.ahcc")
        key = os.urandom(32)
        signer = Ed25519Signer()

        # Step 0 -> 10
        for step in range(CHECKPOINT_STEP_1):
            opt_b.zero_grad()
            x, y = dataset[step]
            jitter = random.random() * 1e-7 + np.random.randn() * 1e-7
            x_noisy = x + jitter
            pred = model_b(x_noisy)
            loss = criterion(pred, y)
            loss.backward()
            opt_b.step()
            sched_b.step()
            run_b_losses.append(loss.item())
            run_b_preds.append(pred.detach().cpu())

        # Save Checkpoint 1 at Step 10
        prog_1 = TrainingProgress(
            global_step=CHECKPOINT_STEP_1,
            epoch=1,
            batch_idx=CHECKPOINT_STEP_1,
            loss=run_b_losses[-1],
            learning_rate=sched_b.get_last_lr()[0],
        )
        save_training_checkpoint(
            output_path=ckpt_path_1,
            model=model_b,
            optimizer=opt_b,
            scheduler=sched_b,
            progress=prog_1,
            key=key,
            signer_private_key=signer.private_bytes,
        )

        # Crash simulation 1: Pollute state and reconstruct fresh instances
        random.seed(888888)
        np.random.seed(777777)
        torch.manual_seed(666666)

        model_resumed_1 = ComprehensiveTestModel().to(device)
        opt_resumed_1 = AdamW(model_resumed_1.parameters(), lr=1e-2, weight_decay=1e-4)
        sched_resumed_1 = StepLR(opt_resumed_1, step_size=8, gamma=0.5)

        load_training_checkpoint(
            input_path=ckpt_path_1,
            model=model_resumed_1,
            optimizer=opt_resumed_1,
            scheduler=sched_resumed_1,
            key=key,
            expected_signer_public_key=signer.public_bytes,
            device=device,
            restore_rng=True,
        )

        # Step 10 -> 20
        for step in range(CHECKPOINT_STEP_1, CHECKPOINT_STEP_2):
            opt_resumed_1.zero_grad()
            x, y = dataset[step]
            jitter = random.random() * 1e-7 + np.random.randn() * 1e-7
            x_noisy = x + jitter
            pred = model_resumed_1(x_noisy)
            loss = criterion(pred, y)
            loss.backward()
            opt_resumed_1.step()
            sched_resumed_1.step()
            run_b_losses.append(loss.item())
            run_b_preds.append(pred.detach().cpu())

        # Save Checkpoint 2 at Step 20
        prog_2 = TrainingProgress(
            global_step=CHECKPOINT_STEP_2,
            epoch=2,
            batch_idx=CHECKPOINT_STEP_2,
            loss=run_b_losses[-1],
            learning_rate=sched_resumed_1.get_last_lr()[0],
        )
        save_training_checkpoint(
            output_path=ckpt_path_2,
            model=model_resumed_1,
            optimizer=opt_resumed_1,
            scheduler=sched_resumed_1,
            progress=prog_2,
            key=key,
            signer_private_key=signer.private_bytes,
        )

        # Crash simulation 2: Pollute state again and reconstruct fresh instances
        random.seed(555555)
        np.random.seed(444444)
        torch.manual_seed(333333)

        model_resumed_2 = ComprehensiveTestModel().to(device)
        opt_resumed_2 = AdamW(model_resumed_2.parameters(), lr=1e-2, weight_decay=1e-4)
        sched_resumed_2 = StepLR(opt_resumed_2, step_size=8, gamma=0.5)

        load_training_checkpoint(
            input_path=ckpt_path_2,
            model=model_resumed_2,
            optimizer=opt_resumed_2,
            scheduler=sched_resumed_2,
            key=key,
            expected_signer_public_key=signer.public_bytes,
            device=device,
            restore_rng=True,
        )

        # Step 20 -> 30
        for step in range(CHECKPOINT_STEP_2, TOTAL_STEPS):
            opt_resumed_2.zero_grad()
            x, y = dataset[step]
            jitter = random.random() * 1e-7 + np.random.randn() * 1e-7
            x_noisy = x + jitter
            pred = model_resumed_2(x_noisy)
            loss = criterion(pred, y)
            loss.backward()
            opt_resumed_2.step()
            sched_resumed_2.step()
            run_b_losses.append(loss.item())
            run_b_preds.append(pred.detach().cpu())

    # -------------------------------------------------------------------------
    # Verification across ALL 30 steps
    # -------------------------------------------------------------------------
    assert len(ref_losses) == TOTAL_STEPS
    assert len(run_b_losses) == TOTAL_STEPS

    for step in range(TOTAL_STEPS):
        l_ref = ref_losses[step]
        l_b = run_b_losses[step]
        assert l_ref == l_b, f"Loss mismatch at step {step}: {l_ref} != {l_b}"
        assert torch.equal(ref_preds[step], run_b_preds[step]), f"Prediction mismatch at step {step}"

    # Verify model parameters and buffers
    for (n_ref, p_ref), (n_res, p_res) in zip(
        model_ref.named_parameters(), model_resumed_2.named_parameters()
    ):
        assert n_ref == n_res
        assert torch.equal(p_ref, p_res), f"Parameter divergence in {n_ref}"

    for (n_ref, b_ref), (n_res, b_res) in zip(
        model_ref.named_buffers(), model_resumed_2.named_buffers()
    ):
        assert n_ref == n_res
        assert torch.equal(b_ref, b_res), f"Buffer divergence in {n_ref}"

    # Verify optimizer moments
    for (p_ref, p_res) in zip(model_ref.parameters(), model_resumed_2.parameters()):
        st_ref = opt_ref.state[p_ref]
        st_res = opt_resumed_2.state[p_res]
        assert st_ref["step"] == st_res["step"]
        assert torch.equal(st_ref["exp_avg"], st_res["exp_avg"])
        assert torch.equal(st_ref["exp_avg_sq"], st_res["exp_avg_sq"])


def test_recovery_fail_closed_on_corrupt_manifest():
    """Verify that any modification to encrypted manifest fails closed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test.ahcc")
        model = ComprehensiveTestModel()
        key = os.urandom(32)

        save_training_checkpoint(output_path=ckpt_path, model=model, key=key)

        with open(ckpt_path, "r+b") as f:
            data = bytearray(f.read())
            # Flip byte in manifest region (after 8 bytes magic + 1 byte ver + 32 salt)
            data[60] ^= 0xFF
            f.seek(0)
            f.write(data)

        fresh_model = ComprehensiveTestModel()
        with pytest.raises(CheckpointError):
            load_training_checkpoint(input_path=ckpt_path, model=fresh_model, key=key)


def test_recovery_fail_closed_on_corrupt_ciphertext_chunk():
    """Verify that tampering with any chunk ciphertext fails closed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test.ahcc")
        model = ComprehensiveTestModel()
        key = os.urandom(32)

        save_training_checkpoint(output_path=ckpt_path, model=model, key=key)

        with open(ckpt_path, "r+b") as f:
            data = bytearray(f.read())
            # Flip byte in the last 100 bytes (chunk ciphertext / tag)
            data[-20] ^= 0xAA
            f.seek(0)
            f.write(data)

        fresh_model = ComprehensiveTestModel()
        with pytest.raises(CheckpointError):
            load_training_checkpoint(input_path=ckpt_path, model=fresh_model, key=key)


def test_recovery_fail_closed_on_truncated_checkpoint():
    """Verify that truncated / interrupted checkpoint file fails closed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test.ahcc")
        model = ComprehensiveTestModel()
        key = os.urandom(32)

        save_training_checkpoint(output_path=ckpt_path, model=model, key=key)

        with open(ckpt_path, "rb") as orig:
            orig_data = orig.read()
        file_len = len(orig_data)

        # Test multiple truncation points (50% and 90%)
        for frac in [0.5, 0.9]:
            with open(ckpt_path, "wb") as f:
                f.write(orig_data[: int(file_len * frac)])

            fresh_model = ComprehensiveTestModel()
            with pytest.raises(CheckpointError):
                load_training_checkpoint(input_path=ckpt_path, model=fresh_model, key=key)


def test_recovery_fail_closed_on_invalid_key():
    """Verify that loading with the wrong key fails closed without leaking state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test.ahcc")
        model = ComprehensiveTestModel()
        key = os.urandom(32)
        wrong_key = os.urandom(32)

        save_training_checkpoint(output_path=ckpt_path, model=model, key=key)

        fresh_model = ComprehensiveTestModel()
        with pytest.raises(CheckpointError):
            load_training_checkpoint(input_path=ckpt_path, model=fresh_model, key=wrong_key)


def test_recovery_fail_closed_on_missing_distributed_shard():
    """Verify that distributed checkpoint loading fails closed if any shard is missing."""
    from ahcc.distributed import build_global_inventory
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = os.path.join(tmp_dir, "dist_ckpt")
        key = os.urandom(32)
        world_size = 4

        # Save 4 shards
        for rank in range(world_size):
            state = {f"weight_rank_{rank}": torch.randn(4, 4)}
            save_sharded_checkpoint(
                shard_tensors=state,
                checkpoint_dir=ckpt_dir,
                rank=rank,
                world_size=world_size,
                key=key,
            )

        build_global_inventory(
            checkpoint_dir=ckpt_dir,
            run_id="run-dist-fail",
            global_step=10,
            world_size=world_size,
        )

        # Delete shard 2
        shard_2 = os.path.join(ckpt_dir, "shard_00002_of_00004.ahcc")
        assert os.path.exists(shard_2)
        os.remove(shard_2)

        # Rank 0 load should fail closed
        with pytest.raises(Exception):
            load_sharded_checkpoint(
                checkpoint_dir=ckpt_dir,
                rank=0,
                world_size=world_size,
                key=key,
            )


def test_mixed_precision_grad_scaler_recovery_bit_for_bit_equivalence():
    """Verify bit-for-bit resumption of mixed-precision training with torch.amp.GradScaler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOTAL_STEPS = 10
    RESUME_STEP = 5
    seed = 42

    # Synthesize dataset
    torch.manual_seed(seed)
    dataset = [
        (torch.randn(8, 16, device=device), torch.randint(0, 4, (8,), device=device))
        for _ in range(TOTAL_STEPS)
    ]
    criterion = nn.CrossEntropyLoss()

    # --- Run A: Uninterrupted Run ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_ref = ComprehensiveTestModel().to(device)
    opt_ref = AdamW(model_ref.parameters(), lr=1e-2)
    scaler_ref = torch.amp.GradScaler(device.type)

    ref_losses = []
    ref_preds = []

    for step in range(TOTAL_STEPS):
        opt_ref.zero_grad()
        x, y = dataset[step]
        with torch.amp.autocast(device.type):
            pred = model_ref(x)
            loss = criterion(pred, y)
        scaler_ref.scale(loss).backward()
        scaler_ref.step(opt_ref)
        scaler_ref.update()
        ref_losses.append(loss.item())
        ref_preds.append(pred.detach().cpu())

    # --- Run B: Interrupted at step 5 and resumed ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_b = ComprehensiveTestModel().to(device)
    opt_b = AdamW(model_b.parameters(), lr=1e-2)
    scaler_b = torch.amp.GradScaler(device.type)

    run_b_losses = []
    run_b_preds = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "scaler_ckpt.ahcc")
        key = os.urandom(32)

        for step in range(RESUME_STEP):
            opt_b.zero_grad()
            x, y = dataset[step]
            with torch.amp.autocast(device.type):
                pred = model_b(x)
                loss = criterion(pred, y)
            scaler_b.scale(loss).backward()
            scaler_b.step(opt_b)
            scaler_b.update()
            run_b_losses.append(loss.item())
            run_b_preds.append(pred.detach().cpu())

        prog = TrainingProgress(
            global_step=RESUME_STEP,
            epoch=1,
            batch_idx=RESUME_STEP,
            loss=run_b_losses[-1],
        )
        save_training_checkpoint(
            output_path=ckpt_path,
            model=model_b,
            optimizer=opt_b,
            scaler=scaler_b,
            progress=prog,
            key=key,
        )

        # Scramble environment
        random.seed(99999)
        np.random.seed(88888)
        torch.manual_seed(77777)

        model_resumed = ComprehensiveTestModel().to(device)
        opt_resumed = AdamW(model_resumed.parameters(), lr=1e-2)
        scaler_resumed = torch.amp.GradScaler(device.type)

        load_training_checkpoint(
            input_path=ckpt_path,
            model=model_resumed,
            optimizer=opt_resumed,
            scaler=scaler_resumed,
            key=key,
            device=device,
            restore_rng=True,
        )

        # Confirm scaler internal state restored bit-for-bit at resumption point
        assert scaler_b.state_dict() == scaler_resumed.state_dict()

        for step in range(RESUME_STEP, TOTAL_STEPS):
            opt_resumed.zero_grad()
            x, y = dataset[step]
            with torch.amp.autocast(device.type):
                pred = model_resumed(x)
                loss = criterion(pred, y)
            scaler_resumed.scale(loss).backward()
            scaler_resumed.step(opt_resumed)
            scaler_resumed.update()
            run_b_losses.append(loss.item())
            run_b_preds.append(pred.detach().cpu())

    # Verify bit-for-bit equivalence across all steps
    for step in range(TOTAL_STEPS):
        assert ref_losses[step] == run_b_losses[step], f"Loss mismatch at step {step}"
        assert torch.equal(ref_preds[step], run_b_preds[step]), f"Pred mismatch at step {step}"

    # Verify final scaler state dict
    assert scaler_ref.state_dict() == scaler_resumed.state_dict()

    # Verify model parameters
    for (n_ref, p_ref), (n_res, p_res) in zip(
        model_ref.named_parameters(), model_resumed.named_parameters()
    ):
        assert torch.equal(p_ref, p_res)


def test_distributed_global_inventory_enospc_failure_fails_closed():
    """Verify that simulated ENOSPC disk-full during build_global_inventory fails closed atomically."""
    from unittest import mock
    from ahcc.distributed import verify_sharded_checkpoint
    world_size = 2
    key = os.urandom(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save valid shards for ranks 0 and 1
        for r in range(world_size):
            save_sharded_checkpoint({"w": torch.randn(4)}, tmp_dir, rank=r, world_size=world_size, key=key)

        # Simulate ENOSPC during temporary manifest file write
        orig_open = open

        def mocked_open(file, mode="r", *args, **kwargs):
            if isinstance(file, str) and ".global_manifest_tmp_" in file and "w" in mode:
                raise OSError(28, "No space left on device")
            return orig_open(file, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=mocked_open):
            with pytest.raises(OSError, match="No space left on device"):
                build_global_inventory(
                    checkpoint_dir=tmp_dir,
                    run_id="run-enospc",
                    global_step=1,
                    world_size=world_size,
                )

        # Verify no .global_manifest_tmp_* files left behind
        tmp_files = [f for f in os.listdir(tmp_dir) if ".global_manifest_tmp_" in f]
        assert len(tmp_files) == 0, f"Temporary manifest leaked: {tmp_files}"

        # Verify no global_manifest.json exists
        assert not os.path.exists(os.path.join(tmp_dir, "global_manifest.json"))

        # Verify subsequent verify and load fail closed
        with pytest.raises(DistributedCheckpointError):
            verify_sharded_checkpoint(tmp_dir)


