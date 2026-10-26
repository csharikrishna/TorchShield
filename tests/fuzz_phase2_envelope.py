"""Phase 2 Random Envelope Fuzzing Suite.

Performs randomized bit-flipping, byte insertions, deletions, and structural
mutations across checkpoint files to verify that all corruptions fail closed.
"""
from __future__ import annotations

import os
import random
import secrets
import sys
import pytest
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError


def generate_valid_fuzz_checkpoint(path: str, key: bytes):
    """Generate a realistic test checkpoint for fuzzing."""
    state = {
        "encoder.conv": torch.randn(8, 8, 3, 3),
        "encoder.bias": torch.randn(8),
        "fc.weight": torch.randn(16, 72),
    }
    SecureModelCheckpoint.save_checkpoint(state, path, key, chunk_bytes=1024)
    with open(path, "rb") as f:
        return f.read()


def test_random_bit_and_byte_fuzzing(tmp_path):
    """Fuzz 100 random bit and byte corruptions and confirm 100% fail-closed rejection."""
    import pathlib
    tmp_path = pathlib.Path(tmp_path)
    key = secrets.token_bytes(32)
    path = str(tmp_path / "valid.ahcc")
    raw_data = bytearray(generate_valid_fuzz_checkpoint(path, key))

    rng = random.Random(1337)  # Deterministic seed for reproducible fuzzing

    for iteration in range(100):
        fuzzed = bytearray(raw_data)
        mutation_type = rng.choice(["bit_flip", "byte_replace", "byte_insert", "byte_delete", "slice_zero"])

        if mutation_type == "bit_flip":
            pos = rng.randint(0, len(fuzzed) - 1)
            bit = rng.randint(0, 7)
            fuzzed[pos] ^= (1 << bit)

        elif mutation_type == "byte_replace":
            pos = rng.randint(0, len(fuzzed) - 1)
            fuzzed[pos] = rng.randint(0, 255)

        elif mutation_type == "byte_insert":
            pos = rng.randint(0, len(fuzzed))
            fuzzed.insert(pos, rng.randint(0, 255))

        elif mutation_type == "byte_delete":
            if len(fuzzed) > 10:
                pos = rng.randint(0, len(fuzzed) - 1)
                del fuzzed[pos]

        elif mutation_type == "slice_zero":
            pos = rng.randint(0, len(fuzzed) - 10)
            length = rng.randint(1, 16)
            fuzzed[pos:pos + length] = b"\x00" * length

        fuzz_file = str(tmp_path / f"fuzz_{iteration}.ahcc")
        with open(fuzz_file, "wb") as f:
            f.write(fuzzed)

        # Must fail closed with CheckpointError
        with pytest.raises(CheckpointError):
            SecureModelCheckpoint.load_checkpoint(fuzz_file, key)

        try:
            os.remove(fuzz_file)
        except OSError:
            pass


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        print("Running standalone 100-iteration envelope fuzzer...")
        test_random_bit_and_byte_fuzzing(td)
        print("Fuzzing test passed: 100/100 mutations safely rejected with fail-closed CheckpointError.")
