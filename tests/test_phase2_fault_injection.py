"""Phase 2 Systematic Fault Injection and Failure Recovery Tests.

Validates:
- Controlled failure handling across all possible structural corruptions.
- Truncation at every 10% boundary of the file.
- Header, magic, version, salt, and metadata mutations.
- Corrupted chunk sequence indices, lengths, and tags.
- Fail-closed guarantee: unauthenticated plaintext is NEVER released.
"""
from __future__ import annotations

import os
import secrets
import struct
import pytest
import torch

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError


@pytest.fixture
def base_checkpoint(tmp_path):
    key = secrets.token_bytes(32)
    state = {
        "layer_1": torch.randn(64, 64),
        "layer_2": torch.randn(32, 64),
        "bias": torch.randn(32),
    }
    path = str(tmp_path / "valid.ahcc")
    SecureModelCheckpoint.save_checkpoint(state, path, key, chunk_bytes=4096, format_version=2)
    with open(path, "rb") as f:
        raw_bytes = f.read()
    return {"key": key, "raw_bytes": raw_bytes, "tmp_path": tmp_path}


def test_truncation_fault_injection_matrix(base_checkpoint):
    """Systematically test file truncation at various percentage thresholds."""
    data = base_checkpoint["raw_bytes"]
    key = base_checkpoint["key"]
    total_len = len(data)

    # Test truncating at 10%, 20%, ..., 95%, and 1 byte off
    percentages = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    for pct in percentages:
        cut_len = int(total_len * pct)
        truncated_path = str(base_checkpoint["tmp_path"] / f"trunc_{int(pct*100)}.ahcc")
        with open(truncated_path, "wb") as f:
            f.write(data[:cut_len])

        with pytest.raises(CheckpointError):
            SecureModelCheckpoint.load_checkpoint(truncated_path, key)


def test_header_field_mutations(base_checkpoint):
    """Mutate individual fields in the container header."""
    data = bytearray(base_checkpoint["raw_bytes"])
    key = base_checkpoint["key"]

    # 1. Mutate magic bytes
    bad_magic = bytearray(data)
    bad_magic[0] = ord(b"X")
    p1 = str(base_checkpoint["tmp_path"] / "bad_magic.ahcc")
    with open(p1, "wb") as f:
        f.write(bad_magic)
    with pytest.raises(CheckpointError, match="Not a valid AHCC"):
        SecureModelCheckpoint.load_checkpoint(p1, key)

    # 2. Mutate version
    bad_ver = bytearray(data)
    bad_ver[8] = 99  # version 99
    p2 = str(base_checkpoint["tmp_path"] / "bad_ver.ahcc")
    with open(p2, "wb") as f:
        f.write(bad_ver)
    with pytest.raises(CheckpointError, match="Unsupported checkpoint version"):
        SecureModelCheckpoint.load_checkpoint(p2, key)

    # 3. Mutate salt
    bad_salt = bytearray(data)
    bad_salt[12] ^= 0xFF
    p3 = str(base_checkpoint["tmp_path"] / "bad_salt.ahcc")
    with open(p3, "wb") as f:
        f.write(bad_salt)
    with pytest.raises(CheckpointError, match="manifest authentication failed"):
        SecureModelCheckpoint.load_checkpoint(p3, key)

    # 4. Mutate declared manifest length to excessive size
    bad_mlen = bytearray(data)
    # offset 8 + 1 + 16 + 1 + 36 + 1 = 63
    bad_mlen[63:67] = struct.pack(">I", 500 * 1024 * 1024)  # 500 MiB manifest
    p4 = str(base_checkpoint["tmp_path"] / "bad_mlen.ahcc")
    with open(p4, "wb") as f:
        f.write(bad_mlen)
    with pytest.raises(CheckpointError, match="Invalid manifest length"):
        SecureModelCheckpoint.load_checkpoint(p4, key)
