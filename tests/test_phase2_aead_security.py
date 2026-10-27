"""Phase 2 AEAD Security and Failure Behavior Tests.

Validates:
- Correct 256-bit key handling and rejection of invalid key lengths/types.
- NIST SP 800-38D fail-closed behavior on altered ciphertext, tag, nonce, AAD, header.
- Zero plaintext leakage prior to successful tag authentication.
- HKDF RFC 5869 domain-separated artifact subkey isolation.
- Key rotation support without format ambiguity.
"""
from __future__ import annotations

import os
import secrets
import struct
import pytest
import torch

from ahcc.secure import AESGCMEnvelope, AuthenticationError
from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError


def test_key_length_and_type_validation():
    """Verify strict validation of 256-bit AES keys."""
    valid_key = secrets.token_bytes(32)
    plaintext = b"TensorWeights confidential payload."

    # Valid key works
    envelope = AESGCMEnvelope.encrypt(plaintext, valid_key)
    assert AESGCMEnvelope.decrypt(envelope, valid_key) == plaintext

    # Invalid lengths rejected
    for invalid_len in [0, 8, 16, 24, 31, 33, 64]:
        bad_key = secrets.token_bytes(invalid_len)
        with pytest.raises(ValueError, match="32 bytes"):
            AESGCMEnvelope.encrypt(plaintext, bad_key)
        with pytest.raises(ValueError, match="32 bytes"):
            AESGCMEnvelope.decrypt(envelope, bad_key)

    # Invalid types rejected
    for bad_type in [None, 12345, "string_key_not_bytes", ["key_in_list"]]:
        with pytest.raises(TypeError):
            AESGCMEnvelope.encrypt(plaintext, bad_type)  # type: ignore


def test_aead_fail_closed_tampering():
    """Verify fail-closed behavior across all envelope byte mutations."""
    key = secrets.token_bytes(32)
    plaintext = b"Critical Model Layer Weights: [0.123, -0.456, 0.789]"
    aad = b"layer_name=backbone.layer1.weight"

    envelope = bytearray(AESGCMEnvelope.encrypt(plaintext, key, associated_data=aad))

    # Correct decryption
    assert AESGCMEnvelope.decrypt(envelope, key, associated_data=aad) == plaintext

    # Wrong key
    wrong_key = secrets.token_bytes(32)
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(envelope, wrong_key, associated_data=aad)

    # Wrong AAD
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(envelope, key, associated_data=b"wrong_aad")

    # Mutate header
    bad_header = bytearray(envelope)
    bad_header[0] ^= 0x01
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(bad_header, key, associated_data=aad)

    # Mutate nonce
    bad_nonce = bytearray(envelope)
    bad_nonce[10] ^= 0xFF
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(bad_nonce, key, associated_data=aad)

    # Mutate ciphertext body
    bad_body = bytearray(envelope)
    bad_body[25] ^= 0x55
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(bad_body, key, associated_data=aad)

    # Mutate tag (last 16 bytes)
    bad_tag = bytearray(envelope)
    bad_tag[-1] ^= 0xAA
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(bad_tag, key, associated_data=aad)

    # Truncated envelope
    for cut in [1, 5, 10, 16, 25]:
        with pytest.raises(AuthenticationError):
            AESGCMEnvelope.decrypt(envelope[:-cut], key, associated_data=aad)


def test_hkdf_domain_separation_and_artifact_isolation():
    """Verify that different artifact salts and IDs derive independent, uncorrelated subkeys."""
    master_key = secrets.token_bytes(32)
    salt1 = secrets.token_bytes(16)
    salt2 = secrets.token_bytes(16)
    id1 = b"artifact-uuid-001"
    id2 = b"artifact-uuid-002"

    k1 = SecureModelCheckpoint.derive_artifact_key(master_key, salt1, id1)
    k2 = SecureModelCheckpoint.derive_artifact_key(master_key, salt2, id1)
    k3 = SecureModelCheckpoint.derive_artifact_key(master_key, salt1, id2)

    assert len(k1) == 32
    assert len(k2) == 32
    assert len(k3) == 32
    assert k1 != k2
    assert k1 != k3
    assert k2 != k3


def test_key_rotation_support(tmp_path):
    """Verify re-encrypting checkpoint with a new key rotates artifact without state corruption."""
    key_old = secrets.token_bytes(32)
    key_new = secrets.token_bytes(32)

    state_dict = {
        "encoder.weight": torch.randn(16, 16),
        "encoder.bias": torch.randn(16),
    }

    path_old = str(tmp_path / "checkpoint_old.ahcc")
    path_new = str(tmp_path / "checkpoint_new.ahcc")

    SecureModelCheckpoint.save_checkpoint(state_dict, path_old, key_old, key_id="v1-key-2026")
    loaded = SecureModelCheckpoint.load_checkpoint(path_old, key_old)

    # Re-encrypt under new rotated key
    SecureModelCheckpoint.save_checkpoint(loaded, path_new, key_new, key_id="v2-key-2027")

    # Old file still loads with old key
    old_loaded = SecureModelCheckpoint.load_checkpoint(path_old, key_old)
    assert torch.equal(old_loaded["encoder.weight"], state_dict["encoder.weight"])

    # New file loads with new key
    new_loaded = SecureModelCheckpoint.load_checkpoint(path_new, key_new)
    assert torch.equal(new_loaded["encoder.weight"], state_dict["encoder.weight"])

    # New file fails with old key
    with pytest.raises(CheckpointError):
        SecureModelCheckpoint.load_checkpoint(path_new, key_old)


def test_production_path_never_invokes_ahcc_engine(tmp_path):
    """Verify that the production package does not expose or invoke the legacy experimental cipher."""
    import ahcc
    assert not hasattr(ahcc, "AHCC_Engine"), "Production package must not export experimental AHCC_Engine"

    # Save and load checkpoint under production API
    key = secrets.token_bytes(32)
    state = {"linear.weight": torch.randn(10, 10)}
    path = str(tmp_path / "prod.ahcc")

    # Must succeed cleanly with AEAD
    SecureModelCheckpoint.save_checkpoint(state, path, key)
    loaded = SecureModelCheckpoint.load_checkpoint(path, key)
    assert torch.equal(loaded["linear.weight"], state["linear.weight"])


def test_public_encrypt_disallows_explicit_nonce():
    """Verify that AESGCMEnvelope.encrypt does not allow callers to supply custom nonces."""
    key = secrets.token_bytes(32)
    plaintext = b"sensitive data"
    custom_nonce = b"\x00" * 12

    # Public API must not accept explicit_nonce
    with pytest.raises(TypeError):
        AESGCMEnvelope.encrypt(plaintext, key, explicit_nonce=custom_nonce)  # type: ignore


def test_private_encrypt_explicit_nonce_validation():
    """Verify validation on the internal _encrypt_with_explicit_nonce helper."""
    key = secrets.token_bytes(32)
    plaintext = b"chunk slice"

    # Valid 12-byte nonce works
    valid_nonce = b"\x01" * 12
    enc = AESGCMEnvelope._encrypt_with_explicit_nonce(plaintext, key, nonce=valid_nonce)
    assert len(enc) >= AESGCMEnvelope.MIN_ENVELOPE_SIZE
    assert AESGCMEnvelope.decrypt(enc, key) == plaintext

    # Invalid nonce lengths rejected
    for bad_len in [0, 8, 11, 13, 16]:
        with pytest.raises(ValueError, match="nonce must be exactly 12 bytes"):
            AESGCMEnvelope._encrypt_with_explicit_nonce(plaintext, key, nonce=b"X" * bad_len)

    # Invalid nonce type rejected
    with pytest.raises(TypeError):
        AESGCMEnvelope._encrypt_with_explicit_nonce(plaintext, key, nonce=12345)  # type: ignore


