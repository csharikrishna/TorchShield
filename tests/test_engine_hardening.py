"""Regression tests for AHCC API validation and edge cases.

These tests use only Python assertions so they can also be called by run_all.py
without requiring pytest. pytest can discover them as ordinary test functions.
"""
import torch

from ahcc.engine import AHCC_Engine, TensorCipherWrapper
from demos.ai_security_use_cases import dequantize_fp_to_fp32, quantize_fp32_to_fp


def _expect_raises(exc_type, call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:  # pragma: no cover - useful, detailed failure
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"Expected {exc_type.__name__} but no exception was raised.")


def test_constructor_and_shape_validation():
    _expect_raises(ValueError, AHCC_Engine, N=0)
    _expect_raises(ValueError, AHCC_Engine, N=-1)
    _expect_raises(ValueError, AHCC_Engine, N=1025)
    _expect_raises(ValueError, AHCC_Engine, R=0)
    _expect_raises(ValueError, AHCC_Engine, p=15)
    _expect_raises(ValueError, AHCC_Engine, device="mps")

    engine = AHCC_Engine(N=8, R=2, p=2_147_483_579, master_seed=77, device="cpu")
    vector = torch.tensor([0, 1, 2, engine.p - 2, engine.p - 1, 8, 9, 10], dtype=torch.int64)
    assert torch.equal(engine.decrypt(engine.encrypt(vector)), vector)

    row_major = torch.randint(0, engine.p, (5, engine.N), dtype=torch.int64)
    assert torch.equal(engine.decrypt(engine.encrypt(row_major)), row_major)
    _expect_raises(TypeError, engine.encrypt, torch.ones(engine.N, dtype=torch.float32))
    _expect_raises(ValueError, engine.encrypt, torch.ones((7, 9), dtype=torch.int64))
    _expect_raises(ValueError, engine.encrypt, torch.ones((1, 1, engine.N), dtype=torch.int64))


def test_modular_arithmetic_high_boundary_values():
    p = 4_294_967_291
    engine = AHCC_Engine(N=4, R=2, p=p, master_seed=19, device="cpu")
    vals_a = [0, 1, p - 1, p - 2, 3_000_000_000]
    vals_b = [p - 1, p - 2, p - 1, 2_500_000_000, 4_000_000_000]
    a = torch.tensor(vals_a, dtype=torch.int64)
    b = torch.tensor(vals_b, dtype=torch.int64)
    actual = engine.safe_mod_mul(a, b).tolist()
    expected = [(x * y) % p for x, y in zip(vals_a, vals_b)]
    assert actual == expected

    for n in (1, 2, 4, 8):
        small = AHCC_Engine(N=n, R=1, p=p, master_seed=20 + n, device="cpu")
        product = small.safe_mod_matmul(small.M, small.M_inv)
        eye = torch.eye(n, dtype=torch.int64)
        assert torch.equal(product, eye)


def test_tensor_wrapper_contract_and_empty_tensor():
    engine = AHCC_Engine(N=8, R=2, p=2_147_483_579, master_seed=88, device="cpu")
    wrapper = TensorCipherWrapper(engine)

    data = torch.tensor([[0, 1, 2], [engine.p - 3, engine.p - 2, engine.p - 1]], dtype=torch.int64)
    cipher, meta = wrapper.encrypt_tensor(data)
    recovered = wrapper.decrypt_tensor(cipher, meta)
    assert torch.equal(recovered, data)

    empty = torch.empty((0, 3), dtype=torch.int64)
    empty_cipher, empty_meta = wrapper.encrypt_tensor(empty)
    empty_recovered = wrapper.decrypt_tensor(empty_cipher, empty_meta)
    assert tuple(empty_recovered.shape) == (0, 3)
    assert empty_recovered.numel() == 0

    scalar = torch.tensor(12345, dtype=torch.int64)
    scalar_cipher, scalar_meta = wrapper.encrypt_tensor(scalar)
    scalar_recovered = wrapper.decrypt_tensor(scalar_cipher, scalar_meta)
    assert scalar_recovered.shape == torch.Size([])
    assert torch.equal(scalar_recovered, scalar)

    _expect_raises(TypeError, wrapper.encrypt_tensor, torch.ones((3, 3), dtype=torch.float32))
    _expect_raises(ValueError, wrapper.encrypt_tensor, torch.tensor([-1, 0], dtype=torch.int64))
    _expect_raises(ValueError, wrapper.encrypt_tensor, torch.tensor([engine.p], dtype=torch.int64))

    wrong_meta = dict(meta)
    wrong_meta["engine_R"] = engine.R + 1
    _expect_raises(ValueError, wrapper.decrypt_tensor, cipher, wrong_meta)
    wrong_meta = dict(meta)
    wrong_meta["version"] = True
    _expect_raises(ValueError, wrapper.decrypt_tensor, cipher, wrong_meta)
    wrong_meta = dict(meta)
    wrong_meta["orig_shape"] = (True, 3)
    _expect_raises(ValueError, wrapper.decrypt_tensor, cipher, wrong_meta)
    wrong_meta = dict(meta)
    wrong_meta["total_len"] = float(meta["total_len"])
    _expect_raises(ValueError, wrapper.decrypt_tensor, cipher, wrong_meta)
    wrong_meta = dict(meta)
    wrong_meta["pad_len"] = 0
    _expect_raises(ValueError, wrapper.decrypt_tensor, cipher, wrong_meta)
    _expect_raises(ValueError, wrapper.decrypt_tensor, cipher[:, :-1], meta)
    bad_cipher = cipher.clone()
    bad_cipher[0, 0] = engine.p
    _expect_raises(ValueError, wrapper.decrypt_tensor, bad_cipher, meta)


def test_float_quantization_roundtrip_and_validation():
    p = 2_147_483_579
    source = torch.tensor([-0.123456, -0.02, 0.0, 0.02, 0.123456], dtype=torch.float32)
    q, meta = quantize_fp32_to_fp(source, p, scale=1e6)
    restored = dequantize_fp_to_fp32(q, meta)
    assert torch.max(torch.abs(source - restored)).item() <= 5.1e-7
    _expect_raises(ValueError, quantize_fp32_to_fp, torch.tensor([float("nan")]), p)
    _expect_raises(ValueError, quantize_fp32_to_fp, torch.tensor([100.0]), p, 1e8)
    _expect_raises(ValueError, quantize_fp32_to_fp, torch.tensor([1.0]), p, 0)
    _expect_raises(ValueError, quantize_fp32_to_fp, torch.tensor([1.0]), p, True)
    _expect_raises(ValueError, quantize_fp32_to_fp, torch.tensor([1.0]), float(p), 1e6)
    _expect_raises(ValueError, dequantize_fp_to_fp32, torch.tensor([p], dtype=torch.int64), meta)
    bad_meta = dict(meta, scale=True)
    _expect_raises(ValueError, dequantize_fp_to_fp32, q, bad_meta)
    bad_meta = dict(meta, p=float(p))
    _expect_raises(ValueError, dequantize_fp_to_fp32, q, bad_meta)


def run_api_hardening_smoke_tests():
    """Aggregate callable for the dependency-free master runner."""
    test_constructor_and_shape_validation()
    test_modular_arithmetic_high_boundary_values()
    test_tensor_wrapper_contract_and_empty_tensor()
    test_float_quantization_roundtrip_and_validation()
    print("[PASS] API validation, arithmetic boundary, quantization, and wrapper edge-case tests passed.")
