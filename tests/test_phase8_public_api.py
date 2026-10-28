"""Phase 8 Public API & Checkpoint Inspection Regression Test Suite."""
import os
import secrets
import tempfile
import pytest
import torch
import torch.nn as nn

import torchshield
from torchshield import (
    save_training_checkpoint,
    load_training_checkpoint,
    inspect_training_checkpoint,
    TrainingProgress,
    TrainingStateError,
    SecureModelCheckpoint,
    CheckpointError,
    ResourceLimits,
    AESGCMEnvelope,
    AuthenticationError,
    TensorAESGCMEnvelope,
    serialize_all_rng,
    restore_all_rng,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_top_level_exports():
    """Verify all expected public APIs are directly accessible from `torchshield`."""
    assert torchshield.__version__ == "0.7.2"
    for name in [
        "save_training_checkpoint",
        "load_training_checkpoint",
        "inspect_training_checkpoint",
        "TrainingProgress",
        "TrainingStateError",
        "SecureModelCheckpoint",
        "CheckpointError",
        "ResourceLimits",
        "AESGCMEnvelope",
        "AuthenticationError",
        "TensorAESGCMEnvelope",
        "serialize_all_rng",
        "restore_all_rng",
    ]:
        assert hasattr(torchshield, name), f"Missing public export: {name}"
        assert name in torchshield.__all__, f"{name} not listed in torchshield.__all__"


def test_public_save_load_and_inspect_cycle():
    """Verify that the public API functions perform full save, load, and inspect without sub-package imports."""
    model = SimpleModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5)

    # Perform a dummy training step
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    sched.step()

    key = secrets.token_bytes(32)
    prog = TrainingProgress(
        global_step=42,
        epoch=3,
        batch_idx=7,
        loss=float(loss.item()),
        learning_rate=0.01,
        config={"model_name": "SimpleModel", "phase": 8},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.shield")

        # 1. Save using top-level API
        save_training_checkpoint(
            output_path=ckpt_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            progress=prog,
            key=key,
        )
        assert os.path.exists(ckpt_path)

        # 2. Inspect without key (unauthenticated header inspection)
        header_info = inspect_training_checkpoint(ckpt_path)
        assert header_info["authenticated"] is False
        assert header_info["format"] == "TorchShield-SafeTrainingCheckpoint-v5"
        assert header_info["version"] == 5
        assert header_info["file_size"] > 0
        assert "artifact_id" in header_info
        assert "key_id" in header_info

        # 3. Inspect with key (authenticated manifest inspection)
        manifest_info = inspect_training_checkpoint(ckpt_path, key=key)
        assert manifest_info["authenticated"] is True
        assert manifest_info["global_step"] == 42
        assert manifest_info["epoch"] == 3
        assert manifest_info["batch_idx"] == 7
        assert abs(manifest_info["loss"] - float(loss.item())) < 1e-6
        assert manifest_info["config"]["phase"] == 8
        assert manifest_info["total_tensors"] > 0
        assert manifest_info["has_optimizer"] is True
        assert manifest_info["has_scheduler"] is True
        assert manifest_info["has_rng"] is True

        # 4. Load using top-level API
        model2 = SimpleModel()
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5)

        restored_prog = load_training_checkpoint(
            input_path=ckpt_path,
            model=model2,
            optimizer=opt2,
            scheduler=sched2,
            key=key,
        )

        assert restored_prog.global_step == 42
        assert restored_prog.epoch == 3
        assert restored_prog.batch_idx == 7
        assert restored_prog.config["phase"] == 8

        # Confirm weights match exactly
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter {n1} mismatch"


def test_inspect_training_checkpoint_wrong_key():
    """Verify inspect_training_checkpoint fails with CheckpointError on wrong key."""
    model = SimpleModel()
    key = secrets.token_bytes(32)
    wrong_key = secrets.token_bytes(32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "wrong_key.shield")
        save_training_checkpoint(ckpt_path, model=model, key=key)

        with pytest.raises(CheckpointError, match="authentication failed"):
            inspect_training_checkpoint(ckpt_path, key=wrong_key)


def test_inspect_training_checkpoint_invalid_file():
    """Verify inspect_training_checkpoint handles nonexistent or corrupted files cleanly."""
    with pytest.raises(FileNotFoundError):
        inspect_training_checkpoint("nonexistent_file_path_12345.shield")

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = os.path.join(tmpdir, "corrupt.shield")
        with open(bad_path, "wb") as f:
            f.write(b"CORRUPTED_MAGIC_NOT_TorchShield_HEADER")

        with pytest.raises(CheckpointError, match="Not a valid TorchShield secure model checkpoint"):
            inspect_training_checkpoint(bad_path)


def test_inspect_training_checkpoint_truncated_headers():
    """Verify truncated V5 and V4 headers fail-closed with CheckpointError (no uncaught IndexError)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Truncated V5 header
        p_v5 = os.path.join(tmpdir, "trunc_v5.shield")
        with open(p_v5, "wb") as f:
            f.write(SecureModelCheckpoint.MAGIC_V5 + b"\x05")  # Missing salt & rest
        with pytest.raises(CheckpointError, match="Truncated checkpoint salt"):
            inspect_training_checkpoint(p_v5)

        # 2. Truncated V4 header (must not raise IndexError)
        p_v4 = os.path.join(tmpdir, "trunc_v4.shield")
        with open(p_v4, "wb") as f:
            f.write(SecureModelCheckpoint.MAGIC_V4)  # 8 bytes only
        with pytest.raises(CheckpointError, match="Truncated checkpoint header"):
            inspect_training_checkpoint(p_v4)


def test_inspect_training_checkpoint_resource_limits_and_consistency():
    """Verify inspect_training_checkpoint enforces ResourceLimits and detects truncated chunks."""
    model = SimpleModel()
    key = secrets.token_bytes(32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "limits_test.shield")
        save_training_checkpoint(ckpt_path, model=model, key=key)

        # 1. File size exceeds limit
        strict_limits = ResourceLimits(max_file_size=100)
        with pytest.raises(CheckpointError, match="exceeds maximum limit"):
            inspect_training_checkpoint(ckpt_path, limits=strict_limits)

        # 2. Manifest size limit exceeded
        tiny_manifest_limits = ResourceLimits(max_manifest_bytes=10)
        with pytest.raises(CheckpointError, match="Invalid manifest length"):
            inspect_training_checkpoint(ckpt_path, limits=tiny_manifest_limits)

        # 3. Truncated payload: valid header and manifest, but file truncated after manifest
        raw_size = os.path.getsize(ckpt_path)
        hdr_orig = inspect_training_checkpoint(ckpt_path)
        header_and_manifest_bytes = (
            8 + 1 + 32 + 1 + len(bytes.fromhex(hdr_orig["artifact_id"])) + 1
            + len(hdr_orig["key_id"].encode("utf-8")) + 2 + 0 + 4 + hdr_orig["encrypted_manifest_len"]
        )
        assert raw_size > header_and_manifest_bytes + 100, "Checkpoint should contain chunk data beyond manifest"

        trunc_ckpt_path = os.path.join(tmpdir, "trunc_chunks.shield")
        with open(ckpt_path, "rb") as f_in, open(trunc_ckpt_path, "wb") as f_out:
            # Write header and manifest, but only a fraction of chunk payload
            f_out.write(f_in.read(header_and_manifest_bytes + 10))

        # Inspect without key passes container header check
        hdr = inspect_training_checkpoint(trunc_ckpt_path)
        assert hdr["authenticated"] is False
        assert hdr["encrypted_manifest_len"] == hdr_orig["encrypted_manifest_len"]

        # Authenticated inspection detects chunk truncation and fails closed
        with pytest.raises(CheckpointError, match="Checkpoint payload is truncated"):
            inspect_training_checkpoint(trunc_ckpt_path, key=key)


def test_inspect_training_checkpoint_v4_support():
    """Verify inspect_training_checkpoint correctly parses and authenticates Version 4 checkpoints."""
    state = {"fc.weight": torch.randn(4, 8), "fc.bias": torch.randn(4)}
    key = secrets.token_bytes(32)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "v4_model.shield")
        SecureModelCheckpoint.save_checkpoint(
            state_dict=state,
            output_path=ckpt_path,
            key=key,
            format_version=SecureModelCheckpoint.VERSION_V4,
            key_id="legacy-v4-test",
        )

        # 1. Unauthenticated inspection
        info_unauth = inspect_training_checkpoint(ckpt_path)
        assert info_unauth["format"] == "TorchShield-v4"
        assert info_unauth["version"] == 4
        assert info_unauth["key_id"] == "legacy-v4-test"
        assert info_unauth["authenticated"] is False
        assert info_unauth["encrypted_manifest_len"] > 0

        # 2. Authenticated inspection with key
        info_auth = inspect_training_checkpoint(ckpt_path, key=key)
        assert info_auth["format"] == "TorchShield-v4"
        assert info_auth["version"] == 4
        assert info_auth["authenticated"] is True
        assert info_auth["total_tensors"] == 2
        assert info_auth["total_chunks"] == 2


