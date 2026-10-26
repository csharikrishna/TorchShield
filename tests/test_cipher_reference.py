"""Independent reference cipher implementation in pure Python.

Validates that AHCC_Engine encryption and decryption match a completely
independent, unvectorized arbitrary-precision reference implementation.
This ensures that round functions, key addition, S-box substitution,
and Cauchy matrix transformations are mathematically consistent and identical.
"""
import pytest
import torch

from ahcc.engine import AHCC_Engine, SUPPORTED_PRIMES


def pure_python_safe_matmul(M_list, V_list, p):
    """Pure Python matrix-vector multiplication over F_p."""
    N = len(M_list)
    out = [0] * N
    for i in range(N):
        s = 0
        for j in range(N):
            s = (s + M_list[i][j] * V_list[j]) % p
        out[i] = s
    return out


def pure_python_ahcc_encrypt(plaintext_vector, round_keys_list, M_list, R, p):
    """Pure Python AHCC encryption round function."""
    state = list(plaintext_vector)
    for r in range(R):
        # 1. Add round key r
        state = [(s + round_keys_list[r][i]) % p for i, s in enumerate(state)]
        # 2. S-box: x^3 mod p
        state = [pow(s, 3, p) for s in state]
        # 3. Cauchy matrix diffusion
        state = pure_python_safe_matmul(M_list, state, p)
        # 4. Add round key r+1
        state = [(s + round_keys_list[r + 1][i]) % p for i, s in enumerate(state)]
    return state


def pure_python_ahcc_decrypt(ciphertext_vector, round_keys_list, M_inv_list, R, p):
    """Pure Python AHCC decryption round function."""
    state = list(ciphertext_vector)
    d = pow(3, -1, p - 1)
    for r in range(R - 1, -1, -1):
        # 1. Subtract round key r+1
        state = [(s - round_keys_list[r + 1][i]) % p for i, s in enumerate(state)]
        # 2. Cauchy inverse matrix diffusion
        state = pure_python_safe_matmul(M_inv_list, state, p)
        # 3. Inverse S-box: y^d mod p
        state = [pow(s, d, p) for s in state]
        # 4. Subtract round key r
        state = [(s - round_keys_list[r][i]) % p for i, s in enumerate(state)]
    return state


@pytest.mark.parametrize("p", SUPPORTED_PRIMES)
@pytest.mark.parametrize("N", [4, 8, 16])
@pytest.mark.parametrize("R", [1, 2, 4])
def test_cipher_against_pure_python_reference(p, N, R):
    """Compare AHCC_Engine with independent pure-Python reference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = AHCC_Engine(N=N, R=R, p=p, device=device)

    # Extract round keys and matrices from engine as Python lists
    round_keys_list = [
        engine.round_keys[r].squeeze().cpu().tolist()
        if N > 1 else [engine.round_keys[r].item()]
        for r in range(R + 1)
    ]
    M_list = engine.M.cpu().tolist()
    M_inv_list = engine.M_inv.cpu().tolist()

    # Test cases: zeros, ones, near-modulus, mixed values
    test_cases = [
        [0] * N,
        [1] * N,
        [p - 1] * N,
        [(i * 1234567 + 89) % p for i in range(N)],
        [(pow(2, 31, p) + i) % p for i in range(N)],
    ]

    for pt in test_cases:
        # Reference encrypt and decrypt
        ref_ct = pure_python_ahcc_encrypt(pt, round_keys_list, M_list, R, p)
        ref_pt = pure_python_ahcc_decrypt(ref_ct, round_keys_list, M_inv_list, R, p)
        assert ref_pt == pt, "Pure-Python reference failed round trip"

        # Engine encrypt and decrypt
        t_pt = torch.tensor(pt, dtype=torch.int64, device=device)
        engine_ct = engine.encrypt(t_pt)
        engine_pt = engine.decrypt(engine_ct)

        assert engine_ct.cpu().tolist() == ref_ct, (
            f"AHCC_Engine output diverged from reference for N={N}, R={R}, p={p}"
        )
        assert engine_pt.cpu().tolist() == pt, "AHCC_Engine failed round trip"
