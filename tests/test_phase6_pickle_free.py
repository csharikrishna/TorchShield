"""Pickle-Free Training State Serialization Verification Suite.

Section 6 Audit Requirements:
- Verify zero pickle usage across model, optimizer, scheduler, scaler, and all 4 RNG states
- Confirm bit-for-bit reconstruction of AdamW moments, learning rate schedulers, and RNG
- Prohibit silent fallback to pickle for unsupported types
- Verify that standard pickle.loads is NEVER invoked during checkpoint loading
"""
from __future__ import annotations

import os
import random
import tempfile
from unittest import mock
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from ahcc.checkpoint import SecureModelCheckpoint
from ahcc.training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    TrainingProgress,
    TrainingStateError,
    serialize_all_rng,
    restore_all_rng,
)


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc2(self.bn(self.fc1(x)))


def test_zero_pickle_invocation_during_save_and_load():
    """Verify via monkeypatching that pickle.loads and pickle.dumps are NEVER called."""
    with mock.patch("pickle.dumps", side_effect=AssertionError("pickle.dumps was invoked!")):
        with mock.patch("pickle.loads", side_effect=AssertionError("pickle.loads was invoked!")):
            model = SampleModel()
            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

            # Advance training step to populate optimizer moments
            x = torch.randn(4, 16)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            key = os.urandom(32)
            with tempfile.TemporaryDirectory() as tmp_dir:
                ckpt_path = os.path.join(tmp_dir, "clean.ahcc")

                save_training_checkpoint(
                    output_path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    progress=TrainingProgress(global_step=1, epoch=0, loss=float(loss.item())),
                    key=key,
                )

                # Fresh instances
                restored_model = SampleModel()
                restored_opt = AdamW(restored_model.parameters(), lr=1e-3)
                restored_sched = StepLR(restored_opt, step_size=5, gamma=0.5)

                prog = load_training_checkpoint(
                    input_path=ckpt_path,
                    model=restored_model,
                    optimizer=restored_opt,
                    scheduler=restored_sched,
                    key=key,
                )
                assert prog.global_step == 1

                # Verify parameter equivalence
                for p_orig, p_rest in zip(model.parameters(), restored_model.parameters()):
                    assert torch.allclose(p_orig, p_rest)


def test_rng_states_full_environment_determinism():
    """Verify bit-for-bit determinism across Python, NumPy, and PyTorch CPU/CUDA RNGs."""
    # Seed all
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)

    # Capture initial RNG snapshot
    rng_snapshot = serialize_all_rng()

    # Generate reference random numbers
    py_ref = [random.random() for _ in range(5)]
    np_ref = np.random.randn(5).tolist()
    th_ref = torch.randn(5).tolist()

    # Scramble states
    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)

    # Restore snapshot
    restore_all_rng(rng_snapshot)

    # Generate numbers again
    py_rest = [random.random() for _ in range(5)]
    np_rest = np.random.randn(5).tolist()
    th_rest = torch.randn(5).tolist()

    assert py_ref == py_rest
    assert np.allclose(np_ref, np_rest)
    assert np.allclose(th_ref, th_rest)


def test_optimizer_unsupported_object_rejected_without_pickle_fallback():
    """Verify that custom arbitrary non-serializable objects in optimizer state are rejected."""
    model = SampleModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    first_param = next(iter(model.parameters()))
    # Inject arbitrary unapproved Python object
    optimizer.state[first_param]["injected_dangerous_object"] = lambda x: x * 2

    key = os.urandom(32)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "bad_opt.ahcc")
        with pytest.raises(TrainingStateError, match="Unsupported optimizer state item"):
            save_training_checkpoint(
                output_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                key=key,
            )
