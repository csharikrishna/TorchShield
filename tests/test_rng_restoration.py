"""Tests for safe, pickle-free RNG state serialization and exact restoration."""
import random
import numpy as np
import pytest
import torch

from ahcc.training_state import (
    serialize_python_rng,
    restore_python_rng,
    serialize_numpy_rng,
    restore_numpy_rng,
    serialize_torch_cpu_rng,
    restore_torch_cpu_rng,
    serialize_torch_cuda_rng,
    restore_torch_cuda_rng,
    serialize_all_rng,
    restore_all_rng,
    TrainingStateError,
)


def test_python_rng_exact_reproducibility():
    random.seed(1337)
    _ = [random.random() for _ in range(50)]
    saved = serialize_python_rng()

    expected_next = [random.random() for _ in range(100)]

    # Disrupt RNG
    _ = [random.random() for _ in range(999)]

    # Restore
    restore_python_rng(saved)
    actual_next = [random.random() for _ in range(100)]

    assert actual_next == expected_next


def test_numpy_rng_exact_reproducibility():
    np.random.seed(4242)
    _ = np.random.uniform(size=50)
    saved = serialize_numpy_rng()

    expected_uniform = np.random.uniform(size=100)
    expected_normal = np.random.normal(size=100)

    # Disrupt RNG
    _ = np.random.uniform(size=500)

    # Restore
    restore_numpy_rng(saved)
    actual_uniform = np.random.uniform(size=100)
    actual_normal = np.random.normal(size=100)

    np.testing.assert_array_equal(actual_uniform, expected_uniform)
    np.testing.assert_array_equal(actual_normal, expected_normal)


def test_torch_cpu_rng_exact_reproducibility():
    torch.manual_seed(9876)
    _ = torch.randn(50)
    saved = serialize_torch_cpu_rng()

    expected_floats = torch.randn(100)

    # Disrupt RNG
    _ = torch.randn(500)

    # Restore
    restore_torch_cpu_rng(saved)
    actual_floats = torch.randn(100)

    assert torch.equal(actual_floats, expected_floats)


def test_torch_cuda_rng_exact_reproducibility():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for CUDA RNG test.")

    torch.cuda.manual_seed_all(777)
    _ = torch.randn(50, device="cuda")
    saved = serialize_torch_cuda_rng()

    expected = torch.randn(100, device="cuda")

    # Disrupt RNG
    _ = torch.randn(500, device="cuda")

    # Restore
    restore_torch_cuda_rng(saved)
    actual = torch.randn(100, device="cuda")

    assert torch.equal(actual, expected)


def test_all_rng_bundle_roundtrip():
    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)

    bundle = serialize_all_rng()
    assert "python" in bundle
    assert "numpy" in bundle
    assert "torch_cpu" in bundle

    next_py = random.random()
    next_np = np.random.rand()
    next_th = torch.randn(1).item()

    # Disrupt
    random.seed(999)
    np.random.seed(888)
    torch.manual_seed(777)

    # Restore
    restore_all_rng(bundle)
    assert random.random() == next_py
    assert np.random.rand() == next_np
    assert torch.randn(1).item() == next_th


def test_invalid_rng_deserialization_fails_defensively():
    with pytest.raises(TrainingStateError):
        restore_python_rng({"invalid": "payload"})

    with pytest.raises(TrainingStateError):
        restore_numpy_rng({"keys": [1, 2, 3]})  # missing fields
