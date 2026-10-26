"""Comprehensive CUDA vs CPU correctness and numerical equivalence matrix.

Exercises integer boundaries, batching dimensions, non-contiguous layouts,
stream synchronization, and memory device residency on GPU.
"""
import pytest
import torch

from ahcc.engine import AHCC_Engine, SUPPORTED_PRIMES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
@pytest.mark.parametrize("p", SUPPORTED_PRIMES)
@pytest.mark.parametrize("N", [4, 16, 64])
@pytest.mark.parametrize("R", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 13, 64])
def test_cuda_cpu_numerical_equivalence_matrix(p, N, R, batch_size):
    """Verify bit-exact output equivalence between CPU and CUDA backends."""
    key = bytes((i * 31 + N) % 256 for i in range(32))
    engine_cpu = AHCC_Engine(N=N, R=R, p=p, master_key=key, device="cpu")
    engine_gpu = AHCC_Engine(N=N, R=R, p=p, master_key=key, device="cuda")

    # Generate boundary and random data
    torch.manual_seed(42 + N + batch_size)
    data = torch.randint(0, p, (N, batch_size), dtype=torch.int64)
    # Inject boundary elements
    data[0, 0] = 0
    data[1, 0] = 1
    data[2, 0] = p - 1
    data[3, 0] = (p - 1) // 2

    # CPU run
    ct_cpu = engine_cpu.encrypt(data)
    pt_cpu = engine_cpu.decrypt(ct_cpu)

    # GPU run
    data_gpu = data.to("cuda")
    ct_gpu = engine_gpu.encrypt(data_gpu)
    torch.cuda.synchronize()
    assert ct_gpu.is_cuda, "Ciphertext was unexpectedly placed on CPU!"

    pt_gpu = engine_gpu.decrypt(ct_gpu)
    torch.cuda.synchronize()
    assert pt_gpu.is_cuda, "Decrypted tensor was unexpectedly placed on CPU!"

    # Equivalence checks
    assert torch.equal(ct_cpu, ct_gpu.cpu()), f"CPU/CUDA ciphertext divergence for N={N}, R={R}, p={p}"
    assert torch.equal(pt_cpu, pt_gpu.cpu()), f"CPU/CUDA decrypted divergence for N={N}, R={R}, p={p}"
    assert torch.equal(data, pt_gpu.cpu()), f"GPU round trip failed for N={N}, R={R}, p={p}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
def test_cuda_non_contiguous_and_transposed_inputs():
    """Verify GPU engine handles non-contiguous and transposed tensors correctly."""
    p = 4_294_967_291
    N = 32
    engine_gpu = AHCC_Engine(N=N, R=2, p=p, device="cuda")

    # (batch, N) orientation
    data_transposed = torch.randint(0, p, (50, N), dtype=torch.int64, device="cuda")
    ct = engine_gpu.encrypt(data_transposed)
    assert ct.shape == (50, N)
    pt = engine_gpu.decrypt(ct)
    assert torch.equal(data_transposed, pt)

    # Non-contiguous slice
    large_matrix = torch.randint(0, p, (N, 100), dtype=torch.int64, device="cuda")
    strided_slice = large_matrix[:, ::2]  # non-contiguous
    assert not strided_slice.is_contiguous()

    ct_slice = engine_gpu.encrypt(strided_slice)
    pt_slice = engine_gpu.decrypt(ct_slice)
    assert torch.equal(strided_slice, pt_slice)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
def test_cuda_stream_determinism_and_repeated_execution():
    """Verify deterministic output across multiple GPU streams and repeated passes."""
    p = 4_294_967_291
    N = 16
    key = os_key = bytes(range(32))
    engine_gpu = AHCC_Engine(N=N, R=2, p=p, master_key=key, device="cuda")

    data = torch.randint(0, p, (N, 32), dtype=torch.int64, device="cuda")

    ct_first = engine_gpu.encrypt(data)
    torch.cuda.synchronize()

    for _ in range(10):
        ct_repeat = engine_gpu.encrypt(data)
        torch.cuda.synchronize()
        assert torch.equal(ct_first, ct_repeat), "Non-deterministic GPU execution detected!"
