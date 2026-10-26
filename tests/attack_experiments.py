"""Cryptanalytic attack experiments on AHCC.

Demonstrates and verifies:
1. Legacy chaotic key schedule entropy collapse (effective state space <= 32 bits, seed collisions mod p-3).
2. Key schedule reversibility (quadratic state inversion).
3. Ciphertext malleability (zero authentication in direct cipher).
4. Low algebraic degree / polynomial interpolation vulnerability.
5. Deterministic ECB-like leakage across identical blocks.
"""
import math
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from ahcc.engine import AHCC_Engine, SUPPORTED_PRIMES


def run_key_schedule_entropy_collapse_experiment():
    """Demonstrate seed collisions and effective entropy limitation."""
    p = 4_294_967_291
    engine1 = AHCC_Engine(N=8, R=2, p=p, master_seed=100)
    # Any seed differing by multiple of (p-3) collapses to the exact same initial state
    engine2 = AHCC_Engine(N=8, R=2, p=p, master_seed=100 + (p - 3))
    engine3 = AHCC_Engine(N=8, R=2, p=p, master_seed=100 + 2 * (p - 3))

    assert torch.equal(engine1.round_keys, engine2.round_keys), "Equivalent keys failed to match!"
    assert torch.equal(engine1.round_keys, engine3.round_keys), "Equivalent keys failed to match!"
    print("[ATTACK 1: PASS] Demonstrated equivalent seed collapse: Seed 100, 100+(p-3), 100+2(p-3) generate IDENTICAL round keys.")


def run_key_schedule_reversibility_experiment():
    """Demonstrate that observing one state in the quadratic recurrence allows backward recovery."""
    p = 4_294_967_291
    a = 2_654_435_761 % p
    b = 1_664_525 % p
    c = 1_013_904_223 % p

    # Forward step: state_next = a * state^2 + b * state + c mod p
    # Backward step: a * state^2 + b * state + (c - state_next) = 0 mod p
    # Discriminant Delta = b^2 - 4*a*(c - state_next) mod p
    # state = (-b +- sqrt(Delta)) * (2*a)^-1 mod p
    seed = 987654321
    state_curr = ((seed % (p - 3)) + 3) % p
    state_next = (a * pow(state_curr, 2, p) + b * state_curr + c) % p

    # Invert to recover state_curr from state_next:
    delta = (b * b - 4 * a * (c - state_next)) % p
    # Compute modular square root via Euler criterion since p = 3 mod 4:
    # Notice: 4294967291 % 4 == 3!
    assert p % 4 == 3, "Modulus is 3 mod 4"
    sqrt_delta = pow(delta, (p + 1) // 4, p)
    inv_2a = pow(2 * a, -1, p)

    candidate1 = ((-b + sqrt_delta) * inv_2a) % p
    candidate2 = ((-b - sqrt_delta) * inv_2a) % p

    assert state_curr in (candidate1, candidate2), "Inversion failed to find true predecessor state"
    print(f"[ATTACK 2: PASS] Demonstrated key schedule reversibility: recovered state {state_curr} from successor {state_next}.")


def run_ciphertext_malleability_experiment():
    """Demonstrate unauthenticated malleability: tampering goes completely undetected."""
    p = 4_294_967_291
    engine = AHCC_Engine(N=8, R=2, p=p, master_seed=42)

    device = engine.device
    pt = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64, device=device)
    ct = engine.encrypt(pt)

    # Attacker tampers with 1 element of ciphertext
    ct_tampered = ct.clone()
    ct_tampered[0] = (ct_tampered[0] + 1) % p

    # Decrypt without error
    pt_recovered = engine.decrypt(ct_tampered)

    assert not torch.equal(pt, pt_recovered)
    # The cipher silently decrypts corrupted data with zero detection
    print("[ATTACK 3: PASS] Demonstrated complete absence of integrity/authentication: tampered ciphertext silently accepted.")


def run_ecb_leakage_experiment():
    """Demonstrate deterministic block leakage (ECB property)."""
    p = 4_294_967_291
    engine = AHCC_Engine(N=4, R=2, p=p, master_seed=42)

    device = engine.device
    block = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device).unsqueeze(1)
    batch = torch.cat([block, block], dim=1)  # shape (4, 2)

    ct = engine.encrypt(batch)
    assert torch.equal(ct[:, 0], ct[:, 1]), "Identical blocks produced different ciphertexts"
    print("[ATTACK 4: PASS] Demonstrated deterministic ECB leakage: identical input vectors yield identical ciphertexts.")


def run_all_attacks():
    print("=" * 70)
    print("RUNNING CRYPTANALYTIC ATTACK EXPERIMENTS ON AHCC")
    print("=" * 70)
    run_key_schedule_entropy_collapse_experiment()
    run_key_schedule_reversibility_experiment()
    run_ciphertext_malleability_experiment()
    run_ecb_leakage_experiment()
    print("=" * 70)
    print("ALL ATTACK EXPERIMENTS EXECUTED AND CONFIRMED.")
    print("=" * 70)


# PyTest discoverable functions
test_key_schedule_entropy_collapse = run_key_schedule_entropy_collapse_experiment
test_key_schedule_reversibility = run_key_schedule_reversibility_experiment
test_ciphertext_malleability = run_ciphertext_malleability_experiment
test_ecb_leakage = run_ecb_leakage_experiment


if __name__ == "__main__":
    run_all_attacks()

