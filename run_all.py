"""
Master smoke-test runner for the experimental AHCC engine.
Runs arithmetic/invertibility checks, an empirical avalanche diagnostic, API
hardening checks, workload timing, and illustrative use-case demonstrations.
These checks do not establish cryptographic security.
"""
from tests.test_all_phases import (
    test_phase1_invertibility_and_algebra,
    test_phase2_avalanche_effect,
    test_key_sensitivity_and_statistical_uniformity,
)
from tests.test_engine_hardening import run_api_hardening_smoke_tests
from benchmarks.benchmark_phase3 import run_phase3_benchmarks
from demos.ai_security_use_cases import (
    demo_federated_learning_gradient_encryption,
    demo_on_the_fly_model_weight_decryption,
)


def main():
    test_phase1_invertibility_and_algebra()
    test_phase2_avalanche_effect(100)
    test_key_sensitivity_and_statistical_uniformity()
    run_api_hardening_smoke_tests()
    run_phase3_benchmarks()
    demo_federated_learning_gradient_encryption()
    demo_on_the_fly_model_weight_decryption()
    print("\n======================================================================")
    print("ALL AHCC IMPLEMENTATION SMOKE TESTS AND DEMOS COMPLETED.")
    print("These results verify selected behaviors only; they do NOT certify cryptographic security.")
    print("======================================================================")


if __name__ == "__main__":
    main()
