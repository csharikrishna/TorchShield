"""Phase 2 Chunk Integrity and Tampering Resilience Tests.

Validates:
- Detection of chunk reordering, substitution, duplication, deletion, truncation, and corruption.
- Cryptographic binding of chunk sequence numbers and manifest digest into chunk AAD.
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


@pytest.fixture
def multi_chunk_setup(tmp_path):
    """Fixture providing a multi-chunk checkpoint and raw file byte breakdown."""
    key = secrets.token_bytes(32)
    # 3 layers of 512 KiB each, chunked at 256 KiB -> exactly 2 chunks per layer, 6 chunks total
    state_dict = {
        "layer_0": torch.randn(256, 512),
        "layer_1": torch.randn(256, 512),
        "layer_2": torch.randn(256, 512),
    }
    path = str(tmp_path / "model.ahcc")
    SecureModelCheckpoint.save_checkpoint(state_dict, path, key, chunk_bytes=256 * 1024, format_version=2)

    with open(path, "rb") as f:
        file_bytes = f.read()

    # Extract manifest to get exact offsets and lengths of all chunks
    offset = 8 + 1 + 16 + 1 + 36 + 1
    salt = file_bytes[9:25]
    art_id = file_bytes[26:62]
    art_key = SecureModelCheckpoint.derive_artifact_key(key, salt, art_id)
    m_len = struct.unpack(">I", file_bytes[offset:offset + 4])[0]
    enc_manifest = file_bytes[offset + 4:offset + 4 + m_len]
    manifest_bytes = AESGCMEnvelope.decrypt(enc_manifest, art_key, associated_data=art_id + b"::MANIFEST::v2")
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    # Map chunks in file order
    chunk_list = []
    curr_offset = offset + 4 + m_len
    for layer in manifest["layers"]:
        for c in layer["chunks"]:
            c_len = c["ciphertext_len"]
            chunk_slice = file_bytes[curr_offset:curr_offset + c_len]
            chunk_list.append({
                "seq": c["chunk_seq"],
                "layer": layer["name"],
                "offset": curr_offset,
                "len": c_len,
                "bytes": chunk_slice,
            })
            curr_offset += c_len

    header_and_manifest = file_bytes[:offset + 4 + m_len]

    return {
        "key": key,
        "path": path,
        "state_dict": state_dict,
        "header_and_manifest": header_and_manifest,
        "chunks": chunk_list,
        "tmp_path": tmp_path,
    }


def test_chunk_bitflip_tampering(multi_chunk_setup):
    """Verify bit-flipping in any chunk payload triggers CheckpointError."""
    setup = multi_chunk_setup
    key = setup["key"]

    with open(setup["path"], "rb") as f:
        data = bytearray(f.read())

    # Flip a bit in chunk 1 (starts after header and manifest)
    c1_offset = setup["chunks"][0]["offset"]
    data[c1_offset + 30] ^= 0x01

    tampered_path = str(setup["tmp_path"] / "chunk_bitflip.ahcc")
    with open(tampered_path, "wb") as f:
        f.write(data)

    with pytest.raises(CheckpointError, match="Authentication failed on chunk"):
        SecureModelCheckpoint.load_checkpoint(tampered_path, key)


def test_chunk_reordering_attack(multi_chunk_setup):
    """Verify that swapping chunk 1 and chunk 2 triggers sequence anomaly detection."""
    setup = multi_chunk_setup
    key = setup["key"]
    chunks = setup["chunks"]

    # Swap chunk 0 and chunk 1
    tampered_bytes = (
        setup["header_and_manifest"]
        + chunks[1]["bytes"]  # Chunk 2 in place of Chunk 1
        + chunks[0]["bytes"]  # Chunk 1 in place of Chunk 2
        + b"".join(c["bytes"] for c in chunks[2:])
    )

    tampered_path = str(setup["tmp_path"] / "swapped_chunks.ahcc")
    with open(tampered_path, "wb") as f:
        f.write(tampered_bytes)

    # Must fail because chunk 2's AAD is bound to chunk_seq=2, so decrypting it at chunk_seq=1 fails tag authentication
    with pytest.raises(CheckpointError, match="Authentication failed on chunk|Chunk sequence anomaly"):
        SecureModelCheckpoint.load_checkpoint(tampered_path, key)


def test_chunk_deletion_attack(multi_chunk_setup):
    """Verify that omitting a chunk causes truncated or missing chunk detection."""
    setup = multi_chunk_setup
    key = setup["key"]
    chunks = setup["chunks"]

    # Omit the last chunk
    tampered_bytes = setup["header_and_manifest"] + b"".join(c["bytes"] for c in chunks[:-1])

    tampered_path = str(setup["tmp_path"] / "deleted_chunk.ahcc")
    with open(tampered_path, "wb") as f:
        f.write(tampered_bytes)

    with pytest.raises(CheckpointError, match="Truncated chunk"):
        SecureModelCheckpoint.load_checkpoint(tampered_path, key)


def test_cross_checkpoint_chunk_substitution(tmp_path):
    """Verify that replacing a chunk with a valid chunk from another model fails."""
    key = secrets.token_bytes(32)
    state1 = {"layer": torch.randn(100, 100)}
    state2 = {"layer": torch.randn(100, 100)}

    p1 = str(tmp_path / "m1.ahcc")
    p2 = str(tmp_path / "m2.ahcc")

    SecureModelCheckpoint.save_checkpoint(state1, p1, key, chunk_bytes=1024, format_version=2)
    SecureModelCheckpoint.save_checkpoint(state2, p2, key, chunk_bytes=1024, format_version=2)

    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        d1 = bytearray(f1.read())
        d2 = bytearray(f2.read())

    # Replace chunk 1 in model 1 with chunk 1 from model 2
    # Manifest offset in v2: 8 + 1 + 16 + 1 + 36 + 1 = 63
    m1_len = struct.unpack(">I", d1[63:67])[0]
    m2_len = struct.unpack(">I", d2[63:67])[0]

    c1_start = 67 + m1_len
    c2_start = 67 + m2_len

    # Splicing foreign chunk into m1
    spliced = d1[:c1_start] + d2[c2_start:c2_start + 1057] + d1[c1_start + 1057:]
    spliced_path = str(tmp_path / "foreign_chunk.ahcc")
    with open(spliced_path, "wb") as f:
        f.write(spliced)

    # Must fail because chunk's AAD includes the artifact_id and manifest_digest of model 2!
    with pytest.raises(CheckpointError):
        SecureModelCheckpoint.load_checkpoint(spliced_path, key)
