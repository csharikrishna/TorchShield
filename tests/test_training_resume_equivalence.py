"""Deterministic Bit-for-Bit Training Resume Equivalence Validation.

Compares:
Run A: Continuous training for N total steps (e.g. 20 steps).
Run B: Interrupted training at step K (e.g. 10 steps), saved as TorchShield checkpoint,
       loaded into fresh model/optimizer/RNG, and trained for the remaining steps.

Asserts exact bit-for-bit equivalence of:
- Loss value at every subsequent step
- Model weights and biases at final step
- Optimizer moment tensors (exp_avg, exp_avg_sq)
- Next random number generation values
"""
import copy
import os
import random
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from torchshield.training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    TrainingProgress,
)


class RegressionNet(nn.Module):
    def __init__(self, in_features=16, hidden=32, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def train_single_step(model, optimizer, scheduler, criterion, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()


def test_bit_for_bit_training_resume_equivalence():
    TOTAL_STEPS = 20
    INTERRUPT_STEP = 10
    SEED = 42

    # -------------------------------------------------------------------------
    # Generate Synthetic Dataset with fixed initial seed
    # -------------------------------------------------------------------------
    torch.manual_seed(SEED)
    dataset = [
        (torch.randn(8, 16), torch.randn(8, 4))
        for _ in range(TOTAL_STEPS)
    ]

    # =========================================================================
    # Run A: Uninterrupted Reference Run (0 -> TOTAL_STEPS)
    # =========================================================================
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model_a = RegressionNet()
    optimizer_a = AdamW(model_a.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler_a = StepLR(optimizer_a, step_size=5, gamma=0.8)
    criterion = nn.MSELoss()

    losses_a = []
    for step in range(TOTAL_STEPS):
        # Sample stochastic batch perturbation using Python & NumPy RNG to test full RNG sync
        _noise_py = random.random() * 1e-6
        _noise_np = np.random.randn() * 1e-6
        x, y = dataset[step]
        x_noisy = x + _noise_py + _noise_np
        loss_val = train_single_step(model_a, optimizer_a, scheduler_a, criterion, x_noisy, y)
        losses_a.append(loss_val)

    # Next RNG outputs after Run A
    next_py_a = random.random()
    next_np_a = np.random.randn()
    next_th_a = torch.randn(5)

    # =========================================================================
    # Run B: Interrupted, Checkpointed & Resumed Run
    # =========================================================================
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model_b = RegressionNet()
    optimizer_b = AdamW(model_b.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler_b = StepLR(optimizer_b, step_size=5, gamma=0.8)

    losses_b = []
    for step in range(INTERRUPT_STEP):
        _noise_py = random.random() * 1e-6
        _noise_np = np.random.randn() * 1e-6
        x, y = dataset[step]
        x_noisy = x + _noise_py + _noise_np
        loss_val = train_single_step(model_b, optimizer_b, scheduler_b, criterion, x_noisy, y)
        losses_b.append(loss_val)

    # Save checkpoint at INTERRUPT_STEP
    checkpoint_key = os.urandom(32)
    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        progress = TrainingProgress(
            global_step=INTERRUPT_STEP,
            epoch=0,
            batch_idx=INTERRUPT_STEP,
            loss=losses_b[-1],
        )
        save_training_checkpoint(
            output_path=ckpt_path,
            model=model_b,
            optimizer=optimizer_b,
            scheduler=scheduler_b,
            progress=progress,
            key=checkpoint_key,
        )

        # Deliberately pollute RNG and create brand-new instances to simulate process restart
        random.seed(99999)
        np.random.seed(88888)
        torch.manual_seed(77777)

        resumed_model = RegressionNet()
        resumed_opt = AdamW(resumed_model.parameters(), lr=1e-2, weight_decay=1e-4)
        resumed_sched = StepLR(resumed_opt, step_size=5, gamma=0.8)

        restored_prog = load_training_checkpoint(
            input_path=ckpt_path,
            model=resumed_model,
            optimizer=resumed_opt,
            scheduler=resumed_sched,
            key=checkpoint_key,
            restore_rng=True,
        )

        assert restored_prog.global_step == INTERRUPT_STEP

        # Continue training resumed model from INTERRUPT_STEP to TOTAL_STEPS
        for step in range(INTERRUPT_STEP, TOTAL_STEPS):
            _noise_py = random.random() * 1e-6
            _noise_np = np.random.randn() * 1e-6
            x, y = dataset[step]
            x_noisy = x + _noise_py + _noise_np
            loss_val = train_single_step(resumed_model, resumed_opt, resumed_sched, criterion, x_noisy, y)
            losses_b.append(loss_val)

        # Next RNG outputs after Run B
        next_py_b = random.random()
        next_np_b = np.random.randn()
        next_th_b = torch.randn(5)

        # =====================================================================
        # RIGOROUS EQUIVALENCE ASSERTIONS
        # =====================================================================
        # 1. Losses at every step match bit-for-bit
        assert len(losses_a) == len(losses_b) == TOTAL_STEPS
        for step_idx, (l_a, l_b) in enumerate(zip(losses_a, losses_b)):
            assert l_a == l_b, f"Loss mismatch at step {step_idx}: {l_a} != {l_b}"

        # 2. Final model parameters match bit-for-bit
        for (name_a, p_a), (name_b, p_b) in zip(
            model_a.named_parameters(), resumed_model.named_parameters()
        ):
            assert name_a == name_b
            assert torch.equal(p_a, p_b), f"Parameter mismatch for '{name_a}'"

        # 3. Final optimizer moments match bit-for-bit
        for p_a, p_b in zip(model_a.parameters(), resumed_model.parameters()):
            st_a = optimizer_a.state[p_a]
            st_b = resumed_opt.state[p_b]
            for key_st in st_a:
                val_a = st_a[key_st]
                val_b = st_b[key_st]
                if isinstance(val_a, torch.Tensor):
                    assert torch.equal(val_a, val_b), f"Optimizer state mismatch for key {key_st}"
                else:
                    assert val_a == val_b

        # 4. Post-training RNG states match bit-for-bit
        assert next_py_a == next_py_b
        assert next_np_a == next_np_b
        assert torch.equal(next_th_a, next_th_b)

    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
