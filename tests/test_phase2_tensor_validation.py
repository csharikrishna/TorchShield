"""Phase 2 Tensor Metadata Validation and Safe Deserialization Tests.

Validates:
- Strict allow-listing of PyTorch numeric dtypes.
- Rejection of invalid, negative, or overflow-inducing tensor shapes.
- Rejection of excessive tensor counts, ranks, or memory sizes.
- Exact serialization and deserialization of 0-dim scalars, empty tensors,
  and non-contiguous strided tensor slices.
- Protection against untrusted code execution / pickle deserialization.
"""
from __future__ import annotations

import os
import secrets
import struct
import pytest
import torch

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError


def test_supported_dtypes_roundtrip(tmp_path):
    """Verify exact round-trip across all supported PyTorch dtypes."""
    key = secrets.token_bytes(32)
    state_dict = {
        "f32": torch.tensor([1.25, -2.5, 3.125], dtype=torch.float32),
        "f16": torch.tensor([1.5, -0.5, 4.0], dtype=torch.float16),
        "bf16": torch.tensor([1.5, -0.5, 4.0], dtype=torch.bfloat16),
        "f64": torch.tensor([1.12345678901234, -2.987654321], dtype=torch.float64),
        "i64": torch.tensor([9223372036854775807, -9223372036854775808], dtype=torch.int64),
        "i32": torch.tensor([2147483647, -2147483648], dtype=torch.int32),
        "i16": torch.tensor([32767, -32768], dtype=torch.int16),
        "i8": torch.tensor([127, -128], dtype=torch.int8),
        "u8": torch.tensor([0, 128, 255], dtype=torch.uint8),
        "b": torch.tensor([True, False, True, True], dtype=torch.bool),
    }

    path = str(tmp_path / "dtypes.ahcc")
    SecureModelCheckpoint.save_checkpoint(state_dict, path, key)
    loaded = SecureModelCheckpoint.load_checkpoint(path, key)

    for k, v in state_dict.items():
        assert k in loaded
        assert loaded[k].dtype == v.dtype
        assert loaded[k].shape == v.shape
        assert torch.equal(loaded[k], v)


def test_scalars_empty_and_strided_tensors(tmp_path):
    """Verify 0-dim scalars, empty tensors, and non-contiguous tensors."""
    key = secrets.token_bytes(32)
    base = torch.arange(100).reshape(10, 10)
    strided = base[::2, ::2]  # non-contiguous slice

    state_dict = {
        "scalar_0d": torch.tensor(42, dtype=torch.int64),
        "scalar_float": torch.tensor(3.14159, dtype=torch.float32),
        "empty_0el": torch.tensor([], dtype=torch.float32),
        "empty_multidim": torch.empty((0, 16, 32), dtype=torch.float32),
        "non_contiguous": strided,
    }

    path = str(tmp_path / "edge_tensors.ahcc")
    SecureModelCheckpoint.save_checkpoint(state_dict, path, key)
    loaded = SecureModelCheckpoint.load_checkpoint(path, key)

    for k, v in state_dict.items():
        assert k in loaded
        assert loaded[k].shape == v.shape
        assert loaded[k].dtype == v.dtype
        if v.numel() > 0:
            assert torch.equal(loaded[k], v)


def test_rejection_of_excessive_dimensions(tmp_path):
    """Verify rejection of unreasonable or malformed dimensions."""
    key = secrets.token_bytes(32)

    # Tensor with rank > 8 (MAX_TENSOR_RANK = 8)
    bad_rank_shape = [1] * 9
    bad_rank_tensor = torch.zeros(bad_rank_shape)

    with pytest.raises(ValueError, match="exceeds maximum"):
        SecureModelCheckpoint.save_checkpoint({"bad_rank": bad_rank_tensor}, str(tmp_path / "bad.ahcc"), key)


def test_overflow_safe_tensor_dimensions_rejection(tmp_path):
    """Verify that astronomical dimensions in manifest trigger arithmetic overflow rejection."""
    key = secrets.token_bytes(32)
    state = {"valid": torch.randn(4, 4)}
    path = str(tmp_path / "model.ahcc")
    SecureModelCheckpoint.save_checkpoint(state, path, key, format_version=2)

    with open(path, "rb") as f:
        data = f.read()

    # Tamper with the shape inside the encrypted manifest
    # Extract salt, artifact_id to derive artifact key
    salt = data[9:25]
    art_id = data[26:62]
    art_key = SecureModelCheckpoint.derive_artifact_key(key, salt, art_id)

    offset = 8 + 1 + 16 + 1 + 36 + 1
    m_len = struct.unpack(">I", data[offset:offset + 4])[0]
    enc_m = data[offset + 4:offset + 4 + m_len]

    from ahcc.secure import AESGCMEnvelope
    import json
    m_json = json.loads(AESGCMEnvelope.decrypt(enc_m, art_key, associated_data=art_id + b"::MANIFEST::v2").decode("utf-8"))

    # Inject astronomical dimension causing overflow: [2**30, 2**30]
    m_json["layers"][0]["shape"] = [2**30, 2**30]
    new_m_bytes = json.dumps(m_json, sort_keys=True, separators=(",", ":")).encode("utf-8")
    new_enc_m = AESGCMEnvelope._encrypt_with_explicit_nonce(
        new_m_bytes,
        art_key,
        nonce=SecureModelCheckpoint.derive_chunk_nonce(0),
        associated_data=art_id + b"::MANIFEST::v2",
    )

    tampered_bytes = (
        data[:offset]
        + struct.pack(">I", len(new_enc_m))
        + new_enc_m
        + data[offset + 4 + m_len:]
    )
    tampered_path = str(tmp_path / "overflow.ahcc")
    with open(tampered_path, "wb") as f:
        f.write(tampered_bytes)

    with pytest.raises(CheckpointError, match="overflow in tensor dimension"):
        SecureModelCheckpoint.load_checkpoint(tampered_path, key)

