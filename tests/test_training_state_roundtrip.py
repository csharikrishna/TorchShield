"""Comprehensive roundtrip tests for pickle-free training checkpoints."""
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from ahcc.training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    TrainingProgress,
)
from ahcc.key_providers.local import LocalDevKeyProvider
from ahcc.provenance import Ed25519Signer
from ahcc.checkpoint import CheckpointError


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x)))
        return self.fc(h.view(h.size(0), -1))


def test_training_checkpoint_full_roundtrip_raw_key():
    torch.manual_seed(101)
    model = ToyModel()
    optimizer = AdamW(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    # Perform a few optimization steps
    for step in range(3):
        optimizer.zero_grad()
        x = torch.randn(2, 1, 4, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

    key = os.urandom(32)
    with tempfile.NamedTemporaryFile(suffix=".ahcc", delete=False) as tmp:
        path = tmp.name

    try:
        progress_in = TrainingProgress(
            global_step=3,
            epoch=1,
            batch_idx=15,
            loss=float(loss.item()),
            learning_rate=scheduler.get_last_lr()[0],
            config={"batch_size": 2, "arch": "ToyModel"},
        )

        save_training_checkpoint(
            output_path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress_in,
            key=key,
        )

        # Recreate fresh model, optimizer, scheduler
        restored_model = ToyModel()
        restored_opt = AdamW(restored_model.parameters(), lr=0.01)
        restored_sched = StepLR(restored_opt, step_size=2, gamma=0.5)

        progress_out = load_training_checkpoint(
            input_path=path,
            model=restored_model,
            optimizer=restored_opt,
            scheduler=restored_sched,
            key=key,
        )

        # Invariant checks
        assert progress_out.global_step == 3
        assert progress_out.epoch == 1
        assert progress_out.batch_idx == 15
        assert progress_out.config["arch"] == "ToyModel"

        # Parameters bit-for-bit check
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), restored_model.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)

        # Buffers check
        for (n1, b1), (n2, b2) in zip(model.named_buffers(), restored_model.named_buffers()):
            assert n1 == n2
            assert torch.equal(b1, b2)

        # Scheduler check
        assert restored_sched.get_last_lr() == scheduler.get_last_lr()

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_training_checkpoint_roundtrip_kms_provider_and_provenance():
    torch.manual_seed(202)
    model = ToyModel()
    optimizer = AdamW(model.parameters(), lr=0.005)

    kek = os.urandom(32)
    provider = LocalDevKeyProvider(key_id="local-kms-key-1", master_kek=kek)
    signer = Ed25519Signer()

    with tempfile.NamedTemporaryFile(suffix=".ahcc", delete=False) as tmp:
        path = tmp.name

    try:
        progress_in = TrainingProgress(global_step=500, epoch=2, loss=0.123)

        save_training_checkpoint(
            output_path=path,
            model=model,
            optimizer=optimizer,
            progress=progress_in,
            key_provider=provider,
            encryption_context={"env": "staging", "project": "ahcc-p5"},
            signer_private_key=signer.private_bytes,
        )

        # Load with KMS provider and verify signature
        restored_model = ToyModel()
        restored_opt = AdamW(restored_model.parameters(), lr=0.005)

        progress_out = load_training_checkpoint(
            input_path=path,
            model=restored_model,
            optimizer=restored_opt,
            key_provider=provider,
            encryption_context={"env": "staging", "project": "ahcc-p5"},
            expected_signer_public_key=signer.public_bytes,
        )

        assert progress_out.global_step == 500
        for p1, p2 in zip(model.parameters(), restored_model.parameters()):
            assert torch.equal(p1, p2)

        # Failure when encryption context is tampered
        with pytest.raises(CheckpointError):
            load_training_checkpoint(
                input_path=path,
                model=ToyModel(),
                key_provider=provider,
                encryption_context={"env": "production"},  # Mismatched context
            )

    finally:
        if os.path.exists(path):
            os.remove(path)
