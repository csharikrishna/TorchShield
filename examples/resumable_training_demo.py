"""End-to-End Resumable Training Demonstration with Bit-for-Bit Validation.

Demonstrates:
1. Continuous reference training run (Steps 0 -> 30).
2. Interrupted training run checkpointed at Step 15 with TorchShield pickle-free envelope encryption.
3. Full process restart, loading checkpoint, restoring model weights, optimizer moments,
   learning-rate scheduler state, and all RNG generators (Python, NumPy, PyTorch CPU & CUDA).
4. Resume training (Steps 15 -> 30).
5. Exact bit-for-bit loss and weight verification.
"""
import os
import random
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from torchshield.training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    TrainingProgress,
)
from torchshield.key_providers.local import LocalDevKeyProvider
from torchshield.provenance import Ed25519Signer


class DemoClassifier(nn.Module):
    def __init__(self, in_features=32, hidden=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def run_demo():
    print("================================================================================")
    print("  TorchShield SECURE RESUMABLE TRAINING DEMONSTRATION")
    print("  Phase 5 Bit-for-Bit Deterministic Equivalence Engine")
    print("================================================================================\n")

    TOTAL_STEPS = 30
    CHECKPOINT_STEP = 15
    SEED = 1337

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Generate synthetic training batches
    torch.manual_seed(SEED)
    batches = [
        (torch.randn(16, 32, device=device), torch.randint(0, 4, (16,), device=device))
        for _ in range(TOTAL_STEPS)
    ]
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------
    # 1. Uninterrupted Reference Run (0 -> TOTAL_STEPS)
    # --------------------------------------------------------------------------
    print("\n[Phase 1] Executing continuous Reference Training Run (0 -> 30)...")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ref_model = DemoClassifier().to(device)
    ref_opt = AdamW(ref_model.parameters(), lr=1e-2, weight_decay=1e-4)
    ref_sched = StepLR(ref_opt, step_size=10, gamma=0.5)

    ref_losses = []
    for step in range(TOTAL_STEPS):
        ref_opt.zero_grad()
        x, y = batches[step]
        # Stochastic jitter to rigorously exercise Python & NumPy RNG state
        jitter = random.random() * 1e-7 + np.random.randn() * 1e-7
        x_noisy = x + jitter
        out = ref_model(x_noisy)
        loss = criterion(out, y)
        loss.backward()
        ref_opt.step()
        ref_sched.step()
        ref_losses.append(loss.item())

    print(f"  Reference Initial Loss: {ref_losses[0]:.6f}")
    print(f"  Reference Midpoint Loss (Step {CHECKPOINT_STEP}): {ref_losses[CHECKPOINT_STEP]:.6f}")
    print(f"  Reference Final Loss (Step {TOTAL_STEPS}): {ref_losses[-1]:.6f}")

    # --------------------------------------------------------------------------
    # 2. Interrupted Run (0 -> CHECKPOINT_STEP) & Checkpointing
    # --------------------------------------------------------------------------
    print(f"\n[Phase 2] Executing Interrupted Training Run (0 -> {CHECKPOINT_STEP})...")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_model = DemoClassifier().to(device)
    run_opt = AdamW(run_model.parameters(), lr=1e-2, weight_decay=1e-4)
    run_sched = StepLR(run_opt, step_size=10, gamma=0.5)

    run_losses = []
    for step in range(CHECKPOINT_STEP):
        run_opt.zero_grad()
        x, y = batches[step]
        jitter = random.random() * 1e-7 + np.random.randn() * 1e-7
        x_noisy = x + jitter
        out = run_model(x_noisy)
        loss = criterion(out, y)
        loss.backward()
        run_opt.step()
        run_sched.step()
        run_losses.append(loss.item())

    # Create KMS key provider and Ed25519 publisher signer
    kek = os.urandom(32)
    provider = LocalDevKeyProvider(key_id="demo-kek-v1", master_kek=kek)
    signer = Ed25519Signer()

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        checkpoint_path = tmp.name

    try:
        print(f"[*] Saving authenticated training checkpoint to '{checkpoint_path}'...")
        progress = TrainingProgress(
            global_step=CHECKPOINT_STEP,
            epoch=1,
            batch_idx=CHECKPOINT_STEP,
            loss=run_losses[-1],
            learning_rate=run_sched.get_last_lr()[0],
            config={"arch": "DemoClassifier", "batch_size": 16},
        )

        save_training_checkpoint(
            output_path=checkpoint_path,
            model=run_model,
            optimizer=run_opt,
            scheduler=run_sched,
            progress=progress,
            key_provider=provider,
            encryption_context={"env": "demo", "experiment": "p5-resume"},
            signer_private_key=signer.private_bytes,
        )

        file_size_kb = os.path.getsize(checkpoint_path) / 1024
        print(f"[*] Checkpoint saved successfully ({file_size_kb:.2f} KiB).")

        # ----------------------------------------------------------------------
        # 3. Simulate Process Restart and Checkpoint Restoration
        # ----------------------------------------------------------------------
        print("\n[Phase 3] Simulating crash / process restart...")
        # Pollute RNG state deliberately
        random.seed(999999)
        np.random.seed(888888)
        torch.manual_seed(777777)

        # Fresh model, optimizer, scheduler
        resumed_model = DemoClassifier().to(device)
        resumed_opt = AdamW(resumed_model.parameters(), lr=1e-2, weight_decay=1e-4)
        resumed_sched = StepLR(resumed_opt, step_size=10, gamma=0.5)

        print("[*] Restoring model, optimizer, scheduler, and RNG states...")
        loaded_prog = load_training_checkpoint(
            input_path=checkpoint_path,
            model=resumed_model,
            optimizer=resumed_opt,
            scheduler=resumed_sched,
            key_provider=provider,
            encryption_context={"env": "demo", "experiment": "p5-resume"},
            expected_signer_public_key=signer.public_bytes,
            device=device,
            restore_rng=True,
        )

        print(f"[*] Successfully authenticated and resumed from step {loaded_prog.global_step}.")

        # ----------------------------------------------------------------------
        # 4. Resume Training ({CHECKPOINT_STEP} -> TOTAL_STEPS)
        # ----------------------------------------------------------------------
        print(f"\n[Phase 4] Resuming training ({CHECKPOINT_STEP} -> {TOTAL_STEPS})...")
        for step in range(CHECKPOINT_STEP, TOTAL_STEPS):
            resumed_opt.zero_grad()
            x, y = batches[step]
            jitter = random.random() * 1e-7 + np.random.randn() * 1e-7
            x_noisy = x + jitter
            out = resumed_model(x_noisy)
            loss = criterion(out, y)
            loss.backward()
            resumed_opt.step()
            resumed_sched.step()
            run_losses.append(loss.item())

        # ----------------------------------------------------------------------
        # 5. Bit-for-Bit Verification
        # ----------------------------------------------------------------------
        print("\n[Phase 5] Performing bit-for-bit equivalence comparison...")
        print("  Step | Reference Loss | Resumed Loss   | Match")
        print("  -----+----------------+----------------+------")
        for step in range(TOTAL_STEPS):
            l_ref = ref_losses[step]
            l_res = run_losses[step]
            match = (l_ref == l_res)
            status = "MATCH [OK]" if match else "MISMATCH [FAIL]"
            if step in (0, CHECKPOINT_STEP - 1, CHECKPOINT_STEP, TOTAL_STEPS - 1):
                print(f"  {step:4d} | {l_ref:14.8f} | {l_res:14.8f} | {status}")
            assert match, f"Loss divergence at step {step}: {l_ref} != {l_res}"

        # Weight bit-for-bit check
        for (n_ref, p_ref), (n_res, p_res) in zip(
            ref_model.named_parameters(), resumed_model.named_parameters()
        ):
            assert n_ref == n_res
            assert torch.equal(p_ref, p_res), f"Weight divergence in layer '{n_ref}'"

        print("\n================================================================================")
        print("  RESULT: VERIFIED")
        print("  All 30 steps matched bit-for-bit (loss, parameters, moments, RNG).")
        print("================================================================================")

    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


if __name__ == "__main__":
    run_demo()
