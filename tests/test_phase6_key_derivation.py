"""Cryptographic Key Derivation and Domain Separation Tests for Phase 6.

Verifies:
1. HKDF-SHA256 (RFC 5869) key derivation across Version 2, Version 4, and Version 5.
2. Cryptographic domain separation between manifest_key and chunk_key.
3. Avalanche effect and strict input validation on master keys, salts, and artifact IDs.
4. Prevention of cross-domain key reuse.
"""
from __future__ import annotations

import os
import secrets
import pytest

from ahcc.checkpoint import SecureModelCheckpoint


def test_v5_key_derivation_domain_separation():
    """Verify that manifest_key and chunk_key derived from identical inputs are strictly distinct."""
    master_key = secrets.token_bytes(32)
    salt = secrets.token_bytes(32)
    artifact_id = secrets.token_bytes(16)

    manifest_key = SecureModelCheckpoint.derive_manifest_key_v5(master_key, salt, artifact_id)
    chunk_key = SecureModelCheckpoint.derive_chunk_key_v5(master_key, salt, artifact_id)

    assert len(manifest_key) == 32
    assert len(chunk_key) == 32
    assert manifest_key != chunk_key

    # Check Hamming distance (should be around ~128 bits for random 256-bit keys)
    hamming_dist = bin(int.from_bytes(manifest_key, "big") ^ int.from_bytes(chunk_key, "big")).count("1")
    assert 90 <= hamming_dist <= 166, f"Hamming distance {hamming_dist} indicates inadequate key independence"


def test_v5_key_derivation_avalanche_effect():
    """Verify that a 1-bit difference in master_key, salt, or artifact_id alters >35% of derived key bits."""
    master_key = secrets.token_bytes(32)
    salt = secrets.token_bytes(32)
    artifact_id = secrets.token_bytes(16)

    k_base = SecureModelCheckpoint.derive_chunk_key_v5(master_key, salt, artifact_id)

    # 1. Flip 1 bit in master_key
    mk_flipped = bytearray(master_key)
    mk_flipped[0] ^= 0x01
    k_mk = SecureModelCheckpoint.derive_chunk_key_v5(bytes(mk_flipped), salt, artifact_id)
    dist_mk = bin(int.from_bytes(k_base, "big") ^ int.from_bytes(k_mk, "big")).count("1")
    assert 90 <= dist_mk <= 166

    # 2. Flip 1 bit in salt
    salt_flipped = bytearray(salt)
    salt_flipped[0] ^= 0x01
    k_salt = SecureModelCheckpoint.derive_chunk_key_v5(master_key, bytes(salt_flipped), artifact_id)
    dist_salt = bin(int.from_bytes(k_base, "big") ^ int.from_bytes(k_salt, "big")).count("1")
    assert 90 <= dist_salt <= 166

    # 3. Flip 1 bit in artifact_id
    art_flipped = bytearray(artifact_id)
    art_flipped[0] ^= 0x01
    k_art = SecureModelCheckpoint.derive_chunk_key_v5(master_key, salt, bytes(art_flipped))
    dist_art = bin(int.from_bytes(k_base, "big") ^ int.from_bytes(k_art, "big")).count("1")
    assert 90 <= dist_art <= 166


def test_key_derivation_rejects_invalid_inputs():
    """Verify strict validation of input lengths for master_key and salt."""
    valid_key = secrets.token_bytes(32)
    valid_salt = secrets.token_bytes(32)
    valid_art = secrets.token_bytes(16)

    # Short key
    with pytest.raises(ValueError, match="master_key must be exactly 32 bytes"):
        SecureModelCheckpoint.derive_manifest_key_v5(valid_key[:31], valid_salt, valid_art)

    # Long key
    with pytest.raises(ValueError, match="master_key must be exactly 32 bytes"):
        SecureModelCheckpoint.derive_chunk_key_v5(valid_key + b"\x00", valid_salt, valid_art)

    # Short salt
    with pytest.raises(ValueError, match="salt must be exactly 32 bytes"):
        SecureModelCheckpoint.derive_manifest_key_v5(valid_key, valid_salt[:16], valid_art)

    # V4 checks
    with pytest.raises(ValueError, match="master_key must be exactly 32 bytes"):
        SecureModelCheckpoint.derive_chunk_key_v4(valid_key[:16], valid_salt, valid_art)
    with pytest.raises(ValueError, match="salt must be exactly 32 bytes"):
        SecureModelCheckpoint.derive_chunk_key_v4(valid_key, valid_salt[:31], valid_art)

    # V2 checks
    with pytest.raises(ValueError, match="salt must be exactly 16 bytes"):
        SecureModelCheckpoint.derive_artifact_key(valid_key, valid_salt, valid_art)


def test_version_isolation_between_v4_and_v5():
    """Verify that V4 and V5 derive distinct keys even given identical master_key, salt, and artifact_id."""
    master_key = secrets.token_bytes(32)
    salt = secrets.token_bytes(32)
    artifact_id = secrets.token_bytes(16)

    v4_chunk = SecureModelCheckpoint.derive_chunk_key_v4(master_key, salt, artifact_id)
    v5_chunk = SecureModelCheckpoint.derive_chunk_key_v5(master_key, salt, artifact_id)
    assert v4_chunk != v5_chunk

    v4_mani = SecureModelCheckpoint.derive_manifest_key_v4(master_key, salt, artifact_id)
    v5_mani = SecureModelCheckpoint.derive_manifest_key_v5(master_key, salt, artifact_id)
    assert v4_mani != v5_mani
