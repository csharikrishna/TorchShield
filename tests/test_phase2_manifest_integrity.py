"""Phase 2 Manifest Integrity Tests.

Validates:
- Manifest authentication binds format version, artifact ID, key ID, layers, shapes, and dtypes.
- Any modification to manifest metadata (e.g. layer count, layer name, tensor shape, dtype) fails closed.
- Swapping manifest from one checkpoint to another is immediately rejected.
- Extra trailing bytes or unauthenticated container modifications are rejected.
"""
from __future__ import annotations

import json
import os
import secrets
import struct
import pytest
import torch

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError
from ahcc.secure import AESGCMEnvelope


def test_manifest_layer_metadata_tampering(tmp_path):
    """Verify that tampering with manifest plaintext fails during tag verification."""
    key = secrets.token_bytes(32)
    state_dict = {
        "layer_a": torch.randn(10, 10),
        "layer_b": torch.randn(5),
    }
    path = str(tmp_path / "model.ahcc")
    SecureModelCheckpoint.save_checkpoint(state_dict, path, key, format_version=2)

    with open(path, "rb") as f:
        data = bytearray(f.read())

    # Manifest length is at offset 8 (magic) + 1 (ver) + 16 (salt) + 1 (art_id_len) + 36 (art_id) + 1 (key_id_len) + 0 (key_id) = 63
    # Wait, let's parse exact offsets:
    offset = 8 + 1 + 16 + 1 + 36 + 1  # 63
    manifest_len = struct.unpack(">I", data[offset:offset + 4])[0]
    manifest_start = offset + 4
    manifest_end = manifest_start + manifest_len

    # Mutate byte in encrypted manifest
    tampered = bytearray(data)
    tampered[manifest_start + 10] ^= 0xFF
    tampered_path = str(tmp_path / "tampered_manifest.ahcc")
    with open(tampered_path, "wb") as f:
        f.write(tampered)

    with pytest.raises(CheckpointError, match="manifest authentication failed"):
        SecureModelCheckpoint.load_checkpoint(tampered_path, key)


def test_cross_checkpoint_manifest_swap(tmp_path):
    """Verify that replacing a checkpoint's manifest with another valid checkpoint's manifest fails."""
    key = secrets.token_bytes(32)
    state1 = {"layer1": torch.randn(20, 20)}
    state2 = {"layer2": torch.randn(30, 30)}

    path1 = str(tmp_path / "model1.ahcc")
    path2 = str(tmp_path / "model2.ahcc")

    SecureModelCheckpoint.save_checkpoint(state1, path1, key, format_version=2)
    SecureModelCheckpoint.save_checkpoint(state2, path2, key, format_version=2)

    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        d1 = f1.read()
        d2 = f2.read()

    # Extract manifest from model2 and splice into model1
    offset = 8 + 1 + 16 + 1 + 36 + 1
    m1_len = struct.unpack(">I", d1[offset:offset + 4])[0]
    m2_len = struct.unpack(">I", d2[offset:offset + 4])[0]

    spliced = (
        d1[:offset]
        + struct.pack(">I", m2_len)
        + d2[offset + 4:offset + 4 + m2_len]
        + d1[offset + 4 + m1_len:]
    )

    spliced_path = str(tmp_path / "spliced.ahcc")
    with open(spliced_path, "wb") as f:
        f.write(spliced)

    # Must fail because model2's manifest AAD is bound to model2's artifact_id, not model1's
    with pytest.raises(CheckpointError):
        SecureModelCheckpoint.load_checkpoint(spliced_path, key)


def test_trailing_garbage_detection(tmp_path):
    """Verify that extraneous trailing bytes after all chunks are rejected."""
    key = secrets.token_bytes(32)
    state = {"w": torch.randn(10, 10)}
    path = str(tmp_path / "clean.ahcc")
    SecureModelCheckpoint.save_checkpoint(state, path, key)

    with open(path, "rb") as f:
        data = f.read()

    tampered_path = str(tmp_path / "trailing.ahcc")
    with open(tampered_path, "wb") as f:
        f.write(data + b"EXTRA_UNAUTHENTICATED_BYTES")

    with pytest.raises(CheckpointError, match="Extraneous unauthenticated trailing bytes"):
        SecureModelCheckpoint.load_checkpoint(tampered_path, key)
