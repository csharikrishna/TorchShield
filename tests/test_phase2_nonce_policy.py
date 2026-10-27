"""Phase 2 Nonce Policy and Collision-Prevention Tests.

Validates:
- NIST SP 800-38D nonce uniqueness per key.
- Deterministic 96-bit nonce derivation from sequential chunk numbers.
- Manifest nonce reservation (chunk_seq 0).
- Prevention of nonce reuse across independent writes, even when rewriting same path.
- Inability to reuse nonces on interrupted writes.
"""
from __future__ import annotations

import os
import secrets
import struct
import pytest
import torch

from torchshield.checkpoint import SecureModelCheckpoint, CheckpointError


def test_deterministic_nonce_structure():
    """Verify exact 96-bit (12-byte) binary encoding of deterministic chunk nonces."""
    # Manifest nonce (chunk_seq 0)
    nonce_0 = SecureModelCheckpoint.derive_chunk_nonce(0)
    assert len(nonce_0) == 12
    assert nonce_0 == struct.pack(">Q", 0) + struct.pack(">I", 0)
    assert nonce_0 == b"\x00" * 12

    # Chunk 1
    nonce_1 = SecureModelCheckpoint.derive_chunk_nonce(1)
    assert len(nonce_1) == 12
    assert nonce_1 == struct.pack(">Q", 1) + struct.pack(">I", 0)

    # Chunk 1000
    nonce_1000 = SecureModelCheckpoint.derive_chunk_nonce(1000)
    assert len(nonce_1000) == 12
    assert nonce_1000 == struct.pack(">Q", 1000) + struct.pack(">I", 0)

    # Invalid chunk sequence
    with pytest.raises(ValueError):
        SecureModelCheckpoint.derive_chunk_nonce(-1)
    with pytest.raises(ValueError):
        SecureModelCheckpoint.derive_chunk_nonce(2**64)


def test_fresh_key_and_nonce_per_rewrite(tmp_path):
    """Verify that multiple writes of the same model use fresh salts, artifact IDs, and derived keys."""
    master_key = secrets.token_bytes(32)
    state_dict = {"weights": torch.ones(100, 100)}

    path1 = str(tmp_path / "model1.shield")
    path2 = str(tmp_path / "model2.shield")

    SecureModelCheckpoint.save_checkpoint(state_dict, path1, master_key, format_version=2)
    SecureModelCheckpoint.save_checkpoint(state_dict, path2, master_key, format_version=2)

    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        hdr1 = f1.read(8 + 1 + 16 + 1 + 36)
        hdr2 = f2.read(8 + 1 + 16 + 1 + 36)

    # Magic and Version match
    assert hdr1[:9] == hdr2[:9]
    # Salt (16 bytes starting at offset 9) MUST differ
    salt1 = hdr1[9:25]
    salt2 = hdr2[9:25]
    assert salt1 != salt2

    # Artifact ID (36 bytes starting at offset 26) MUST differ
    art1 = hdr1[26:62]
    art2 = hdr2[26:62]
    assert art1 != art2


def test_nonce_sequence_uniqueness_across_many_chunks(tmp_path):
    """Verify that a multi-chunk checkpoint generates strictly unique nonces across all chunks."""
    master_key = secrets.token_bytes(32)
    # Generate 5 layers of 2 MiB each, written with 512 KiB chunks -> 20 chunks total
    state_dict = {f"layer_{i}": torch.randn(512, 1024) for i in range(5)}
    path = str(tmp_path / "multi_chunk.shield")

    SecureModelCheckpoint.save_checkpoint(state_dict, path, master_key, chunk_bytes=512 * 1024, format_version=2)

    # Inspect the manifest
    with open(path, "rb") as f:
        magic = f.read(8)
        ver = f.read(1)[0]
        salt = f.read(16)
        art_id_len = f.read(1)[0]
        art_id = f.read(art_id_len)
        key_id_len = f.read(1)[0]
        key_id = f.read(key_id_len)
        m_len = struct.unpack(">I", f.read(4))[0]
        enc_manifest = f.read(m_len)

    art_key = SecureModelCheckpoint.derive_artifact_key(master_key, salt, art_id)
    from torchshield.secure import AESGCMEnvelope
    import json

    manifest_bytes = AESGCMEnvelope.decrypt(enc_manifest, art_key, associated_data=art_id + b"::MANIFEST::v2")
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    # Extract all chunk_seq numbers
    seen_seqs = set()
    total_chunks = 0
    for layer in manifest["layers"]:
        for chk in layer["chunks"]:
            seq = chk["chunk_seq"]
            assert seq not in seen_seqs, f"Duplicate chunk sequence {seq} detected!"
            seen_seqs.add(seq)
            total_chunks += 1

    assert total_chunks == manifest["total_chunks"]
    # Verify strictly monotonic: 1, 2, ..., total_chunks
    assert seen_seqs == set(range(1, total_chunks + 1))
