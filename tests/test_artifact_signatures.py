"""Tests for Ed25519 checkpoint publisher provenance and signature verification."""
import os
import tempfile
import pytest
import torch

from torchshield.checkpoint import SecureModelCheckpoint, CheckpointError
from torchshield.provenance import Ed25519Signer, Ed25519Verifier, ProvenanceError


def test_ed25519_signer_and_verifier_roundtrip():
    signer = Ed25519Signer()
    verifier = Ed25519Verifier(signer.public_bytes)

    msg = b"ModelCheckpointDigest::abc123"
    signature = signer.sign(msg)
    assert len(signature) == 64
    assert verifier.verify(signature, msg) is True


def test_ed25519_verifier_detects_tampered_payload():
    signer = Ed25519Signer()
    verifier = Ed25519Verifier(signer.public_bytes)

    msg = b"ModelCheckpointDigest::abc123"
    signature = signer.sign(msg)

    tampered_msg = b"ModelCheckpointDigest::abc124"
    with pytest.raises(ProvenanceError):
        verifier.verify(signature, tampered_msg)


def test_ed25519_verifier_detects_tampered_signature():
    signer = Ed25519Signer()
    verifier = Ed25519Verifier(signer.public_bytes)

    msg = b"ModelCheckpointDigest::abc123"
    signature = bytearray(signer.sign(msg))
    signature[10] ^= 0xFF

    with pytest.raises(ProvenanceError):
        verifier.verify(bytes(signature), msg)


def test_checkpoint_v5_with_ed25519_provenance():
    signer = Ed25519Signer()
    key = os.urandom(32)
    state = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        SecureModelCheckpoint.save_checkpoint(
            state_dict=state,
            output_path=tmp_path,
            key=key,
            signer_private_key=signer.private_bytes,
        )

        # 1. Successful verification with correct public key
        restored = SecureModelCheckpoint.load_checkpoint(
            input_path=tmp_path,
            key=key,
            expected_signer_public_key=signer.public_bytes,
        )
        assert torch.allclose(restored["weight"], state["weight"])
        assert torch.allclose(restored["bias"], state["bias"])

        # 2. Rejection with wrong public key
        wrong_signer = Ed25519Signer()
        with pytest.raises(CheckpointError, match="Checkpoint digital signature verification failed"):
            SecureModelCheckpoint.load_checkpoint(
                input_path=tmp_path,
                key=key,
                expected_signer_public_key=wrong_signer.public_bytes,
            )

        # 3. Checkpoint saved without signature rejected when expected_signer_public_key is required
        unsigned_path = tmp_path + ".unsigned"
        SecureModelCheckpoint.save_checkpoint(
            state_dict=state,
            output_path=unsigned_path,
            key=key,
            format_version=SecureModelCheckpoint.VERSION_V5,
        )
        try:
            with pytest.raises(CheckpointError, match="missing required digital signature"):
                SecureModelCheckpoint.load_checkpoint(
                    input_path=unsigned_path,
                    key=key,
                    expected_signer_public_key=signer.public_bytes,
                )
        finally:
            if os.path.exists(unsigned_path):
                os.remove(unsigned_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
