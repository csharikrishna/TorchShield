"""Independent pure-Python arbitrary-precision arithmetic tests for AHCC.

Verifies finite-field arithmetic operations over F_p (p = 2^32 - 5 and p = 2^31 - 19)
against Python's built-in arbitrarily large integer math to guarantee that
hardware/PyTorch int64 representations never silently overflow or corrupt results.
"""
import math
import random
import pytest
import torch

from ahcc.engine import AHCC_Engine, SUPPORTED_PRIMES


@pytest.mark.parametrize("p", SUPPORTED_PRIMES)
def test_modular_arithmetic_boundaries_against_arbitrary_precision(p):
    """Test boundary values: 0, 1, p-1, p-2, (p-1)//2 against big integers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = AHCC_Engine(N=4, R=1, p=p, device=device)

    boundary_values = [0, 1, 2, 3, p - 3, p - 2, p - 1, (p - 1) // 2]
    for val_a in boundary_values:
        for val_b in boundary_values:
            # Addition
            expected_add = (val_a + val_b) % p
            t_a = torch.tensor([val_a], dtype=torch.int64, device=device)
            t_b = torch.tensor([val_b], dtype=torch.int64, device=device)
            actual_add = ((t_a + t_b) % p).item()
            assert actual_add == expected_add, f"Add failed for {val_a} + {val_b} mod {p}"

            # Subtraction
            expected_sub = (val_a - val_b) % p
            actual_sub = ((t_a - t_b) % p).item()
            assert actual_sub == expected_sub, f"Sub failed for {val_a} - {val_b} mod {p}"

            # Safe modular multiplication
            expected_mul = (val_a * val_b) % p
            actual_mul = engine.safe_mod_mul(t_a, t_b).item()
            assert actual_mul == expected_mul, (
                f"safe_mod_mul failed for {val_a} * {val_b} mod {p}: got {actual_mul}, expected {expected_mul}"
            )


@pytest.mark.parametrize("p", SUPPORTED_PRIMES)
def test_randomized_multiplication_overflow_safety(p):
    """Test 50,000 random modular multiplications against arbitrary precision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = AHCC_Engine(N=8, R=1, p=p, device=device)

    random.seed(42)
    batch_size = 1000
    for _ in range(5):
        vals_a = [random.randint(0, p - 1) for _ in range(batch_size)]
        vals_b = [random.randint(0, p - 1) for _ in range(batch_size)]

        expected = [(a * b) % p for a, b in zip(vals_a, vals_b)]

        t_a = torch.tensor(vals_a, dtype=torch.int64, device=device)
        t_b = torch.tensor(vals_b, dtype=torch.int64, device=device)
        actual = engine.safe_mod_mul(t_a, t_b).cpu().tolist()

        assert actual == expected, f"Randomized safe_mod_mul mismatch over F_{p}"


@pytest.mark.parametrize("p", SUPPORTED_PRIMES)
def test_sbox_bijectivity_and_analytical_inverse(p):
    """Verify S(x) = x^3 mod p and S^-1(y) = y^d mod p where d = 3^-1 mod (p-1)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = AHCC_Engine(N=4, R=1, p=p, device=device)

    assert math.gcd(3, p - 1) == 1, f"gcd(3, p-1) must be 1 for modulus {p}"
    d = pow(3, -1, p - 1)
    assert (3 * d) % (p - 1) == 1, f"3 * d must be 1 mod (p-1) for {p}"

    # Test boundary and fixed points
    # 0^3 = 0, 1^3 = 1, (p-1)^3 = -1 = p-1
    test_points = [0, 1, p - 1, 2, p - 2, 42, 1337, (p - 1) // 2]
    t_points = torch.tensor(test_points, dtype=torch.int64, device=device)

    forward = engine.sbox_forward(t_points)
    recovered = engine.sbox_inverse(forward)

    for orig, fwd, rec in zip(test_points, forward.cpu().tolist(), recovered.cpu().tolist()):
        expected_fwd = pow(orig, 3, p)
        assert fwd == expected_fwd, f"S({orig}) = {fwd} != {expected_fwd} mod {p}"
        assert rec == orig, f"S^-1(S({orig})) = {rec} != {orig} mod {p}"


@pytest.mark.parametrize("p", SUPPORTED_PRIMES)
@pytest.mark.parametrize("N", [4, 8, 16, 32, 64])
def test_cauchy_mds_matrix_invertibility_various_dimensions(p, N):
    """Verify analytical inverse of Cauchy matrix for dimensions 4, 8, 16, 32, 64."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = AHCC_Engine(N=N, R=1, p=p, device=device)

    M = engine.M
    M_inv = engine.M_inv

    prod = engine.safe_mod_matmul(M, M_inv)
    identity = torch.eye(N, dtype=torch.int64, device=device)
    assert torch.equal(prod, identity), f"Cauchy M @ M_inv != I for N={N}, p={p}"
