"""Pytest test suite wrapper for cryptanalytic attack experiments.

Runs and verifies all 4 documented empirical attacks against the AHCC cipher.
"""
from tests.attack_experiments import (
    run_key_schedule_entropy_collapse_experiment,
    run_key_schedule_reversibility_experiment,
    run_ciphertext_malleability_experiment,
    run_ecb_leakage_experiment,
)


def test_key_schedule_entropy_collapse():
    run_key_schedule_entropy_collapse_experiment()


def test_key_schedule_reversibility():
    run_key_schedule_reversibility_experiment()


def test_ciphertext_malleability():
    run_ciphertext_malleability_experiment()


def test_ecb_leakage():
    run_ecb_leakage_experiment()
