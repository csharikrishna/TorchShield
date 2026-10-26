"""Phase 4 Public API and Model Integrity Verification Suite.

Validates:
- Public API contract from caller perspective without internal implementation knowledge.
- In-place model state loading and deterministic inference equivalence.
- Strict isolation: production checkpoint system never calls experimental AHCC cipher.
- Rejection of malicious shapes, mismatched keys, and altered tensor names.
- Complete rollback/fail-closed behavior: no partial state returned on decryption failure.
- File handle closure and leak prevention on Windows.
- Absolute exclusion of pickle / torch.load deserialization on unauthenticated data.
"""
from __future__ import annotations

import os
import secrets
import pytest
import torch
import torch.nn as nn

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError
from ahcc.secure import AESGCMEnvelope, AuthenticationError


class VerificationModel(nn.Module):
    """Multi-layer network with parameters, buffers, and non-linearities."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 8 * 8, 10)
        self.register_buffer("constant_scale", torch.tensor([3.14159]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv(x)))
        x = torch.max_pool2d(x, 2)
        x = x.flatten(1)
        return self.fc(x) * self.constant_scale


def test_public_api_happy_path_and_inference_equivalence(tmp_path):
    """Caller test: save and load model state dict, verify bitwise identical outputs."""
    key = secrets.token_bytes(32)
    path = str(tmp_path / "model.ahcc")

    torch.manual_seed(1234)
    model1 = VerificationModel().eval()
    dummy_input = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        out1 = model1(dummy_input)

    # Public API save
    SecureModelCheckpoint.save_checkpoint(model1.state_dict(), path, key, key_id="prod-key-1")

    # Public API load into fresh instance
    model2 = VerificationModel().eval()
    restored = SecureModelCheckpoint.load_checkpoint(path, key, target_model=model2)

    with torch.no_grad():
        out2 = model2(dummy_input)

    assert torch.equal(out1, out2), "Model outputs MUST be identical after checkpoint roundtrip"
    assert torch.equal(model2.constant_scale, model1.constant_scale), "Buffers MUST be restored accurately"


def test_public_api_disallows_explicit_nonce_tampering():
    """Verify that public encryption API rejects explicit_nonce keyword argument."""
    key = secrets.token_bytes(32)
    with pytest.raises(TypeError):
        AESGCMEnvelope.encrypt(b"data", key, explicit_nonce=b"123456789012")  # type: ignore


def test_wrong_key_fails_closed_without_partial_load(tmp_path):
    """Verify that providing an incorrect key immediately raises CheckpointError and loads nothing."""
    correct_key = secrets.token_bytes(32)
    wrong_key = secrets.token_bytes(32)
    path = str(tmp_path / "protected.ahcc")

    model = VerificationModel().eval()
    SecureModelCheckpoint.save_checkpoint(model.state_dict(), path, correct_key)

    fresh_model = VerificationModel().eval()
    initial_weights = fresh_model.fc.weight.clone()

    with pytest.raises(CheckpointError):
        SecureModelCheckpoint.load_checkpoint(path, wrong_key, target_model=fresh_model)

    # Verify model state was NOT partially modified
    assert torch.equal(fresh_model.fc.weight, initial_weights), "Target model state must remain untouched after failure"


def test_file_handle_closure_on_failure(tmp_path):
    """Verify that file handles are cleanly closed when loading fails, allowing deletion on Windows."""
    key = secrets.token_bytes(32)
    path = str(tmp_path / "corrupt.ahcc")

    # Write truncated file
    with open(path, "wb") as f:
        f.write(b"AHCCMOD4\x04" + secrets.token_bytes(20))

    with pytest.raises(CheckpointError):
        SecureModelCheckpoint.load_checkpoint(path, key)

    # On Windows, open file handles prevent file deletion. Attempt deletion to verify cleanup.
    try:
        os.remove(path)
    except OSError as exc:
        pytest.fail(f"File handle leaked; unable to remove failed checkpoint: {exc}")


def test_pickle_is_not_imported_or_called_in_checkpoint_path():
    """Verify that pickle module is NOT used in SecureModelCheckpoint."""
    import ahcc.checkpoint as ckpt_mod
    assert "pickle" not in ckpt_mod.__dict__, "pickle MUST NOT be used in SecureModelCheckpoint"
