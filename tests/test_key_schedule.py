import torch
import pytest

from ahcc import AHCC_Engine


def test_hkdf_key_schedule_is_deterministic_and_keyed():
    key_a = bytes(range(32))
    key_b = bytes(reversed(range(32)))
    a1 = AHCC_Engine(N=8, R=2, master_key=key_a, device="cpu")
    a2 = AHCC_Engine(N=8, R=2, master_key=key_a, device="cpu")
    b = AHCC_Engine(N=8, R=2, master_key=key_b, device="cpu")
    assert a1.key_schedule_mode.startswith("HKDF-SHA256")
    assert torch.equal(a1.round_keys, a2.round_keys)
    assert not torch.equal(a1.round_keys, b.round_keys)
    assert bool(torch.all(a1.round_keys >= 0))
    assert bool(torch.all(a1.round_keys < a1.p))
    # Exercise the default-HKDF engine's inverse on one complete block.
    test_plain = torch.arange(8, dtype=torch.int64)
    assert torch.equal(a1.decrypt(a1.encrypt(test_plain)), test_plain)


def test_default_key_is_random_and_can_reconstruct_engine():
    a = AHCC_Engine(N=8, R=2, device="cpu")
    b = AHCC_Engine(N=8, R=2, device="cpu")
    assert a.master_key is not None and len(a.master_key) == 32
    assert b.master_key is not None and len(b.master_key) == 32
    assert a.master_key != b.master_key
    reloaded = AHCC_Engine(N=8, R=2, master_key=a.master_key, device="cpu")
    assert torch.equal(a.round_keys, reloaded.round_keys)


def test_legacy_seed_mode_is_explicit_and_reproducible():
    a = AHCC_Engine(N=8, R=2, master_seed=2026, device="cpu")
    b = AHCC_Engine(N=8, R=2, master_seed=2026, device="cpu")
    assert a.key_schedule_mode.startswith("legacy-chaotic-seed")
    assert torch.equal(a.round_keys, b.round_keys)


def test_key_input_validation_and_resource_limit():
    with pytest.raises(ValueError):
        AHCC_Engine(N=8, R=2, master_seed=1, master_key=bytes(32), device="cpu")
    with pytest.raises(ValueError):
        AHCC_Engine(N=8, R=2, master_key=b"short", device="cpu")
    with pytest.raises(ValueError):
        AHCC_Engine(N=8, R=65, master_key=bytes(32), device="cpu")
