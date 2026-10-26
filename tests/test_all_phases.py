from typing import Tuple
import torch
import math
from ahcc.engine import AHCC_Engine, TensorCipherWrapper


def test_phase1_invertibility_and_algebra():
    print("======================================================================")
    print("PHASE 1: CORE MATH ENGINE, CAUCHY MDS & INVERTIBILITY VERIFICATION")
    print("======================================================================")
    primes_to_test = [
        ("MVP 31-bit Safe Prime", 2147483579),
        ("Target 32-bit Prime (2^32 - 5)", 4294967291),
    ]

    for label, p_val in primes_to_test:
        engine = AHCC_Engine(N=64, R=4, p=p_val, master_seed=2026)
        print(f"\n--- Testing Field: {label} (p = {engine.p}) ---")
        print(f"Device               : {engine.device}")
        print(f"gcd(3, p-1)          : {math.gcd(3, engine.p - 1)} (Bijective S-Box Verified)")
        print(f"Inverse Exponent (d) : {engine.d}")
        print(f"Check (3 * d)%(p-1)  : {(3 * engine.d) % (engine.p - 1)}")

        # 1. Verify Cauchy MDS Matrix & Inverse: M @ M_inv == I mod p
        identity_check = engine.safe_mod_matmul(engine.M, engine.M_inv)
        expected_eye = torch.eye(engine.N, dtype=torch.int64, device=engine.device)
        assert torch.equal(identity_check, expected_eye), "Cauchy M @ M_inv != Identity!"
        print("[PASS] Cauchy MDS Matrix Inverse Verified (M @ M_inv == I mod p)")

        # 2. Verify S-Box and Inverse S-Box across boundary values & random inputs
        boundary_vals = torch.tensor(
            [0, 1, 2, engine.p - 3, engine.p - 2, engine.p - 1],
            dtype=torch.int64,
            device=engine.device,
        )
        rand_vals = torch.randint(0, engine.p, (1024,), dtype=torch.int64, device=engine.device)
        sbox_test_vec = torch.cat([boundary_vals, rand_vals])
        s_fwd = engine.sbox_forward(sbox_test_vec)
        s_inv = engine.sbox_inverse(s_fwd)
        assert torch.equal(sbox_test_vec, s_inv), "S-Box inversion failed!"
        print("[PASS] Algebraic S-Box & Inverse S-Box Bijectivity Verified")

        # 3. Full Round Encryption & Decryption Invertibility
        plaintext = torch.randint(
            0, engine.p, (engine.N, 256), dtype=torch.int64, device=engine.device
        )
        ciphertext = engine.encrypt(plaintext)
        decrypted = engine.decrypt(ciphertext)

        assert not torch.equal(plaintext, ciphertext), "Ciphertext matches plaintext!"
        assert torch.equal(plaintext, decrypted), "Decryption failed to match original plaintext!"
        print("[PASS] 100% Exact Block Cipher Invertibility Verified (torch.equal == True)")

        # 4. Arbitrary 4D AI Tensor Wrapper Invertibility ([1, 3, 32, 32])
        wrapper = TensorCipherWrapper(engine)
        ai_tensor = torch.randint(
            0, engine.p, (1, 3, 32, 32), dtype=torch.int64, device=engine.device
        )
        ct_tensor, meta = wrapper.encrypt_tensor(ai_tensor)
        recovered_tensor = wrapper.decrypt_tensor(ct_tensor, meta)
        assert torch.equal(ai_tensor, recovered_tensor), "4D AI Tensor recovery failed!"
        print("[PASS] 4D AI Tensor Shape Padding & Decryption Verified ([1, 3, 32, 32])")


def compute_hamming_distance_bits(t1: torch.Tensor, t2: torch.Tensor, bit_width: int = 31) -> Tuple[int, int]:
    """
    Computes exact bitwise Hamming distance between two integer tensors across `bit_width` bits.
    """
    xor_val = t1 ^ t2
    differing_bits = torch.zeros_like(xor_val, dtype=torch.int64)
    for b in range(bit_width):
        differing_bits += (xor_val >> b) & 1
    total_diff = int(differing_bits.sum().item())
    total_bits = t1.numel() * bit_width
    return total_diff, total_bits


def test_phase2_avalanche_effect(num_trials: int = 100):
    print("\n======================================================================")
    print(f"PHASE 2: RANDOM SINGLE-BIT AVALANCHE DIAGNOSTIC ({num_trials} ITERATIONS)")
    print("======================================================================")
    N = 64
    R = 4
    p = 2147483579
    bit_width = 31
    engine = AHCC_Engine(N=N, R=R, p=p, master_seed=9999)

    torch.manual_seed(42)
    ratios = []

    # Each trial flips one randomly selected input bit, rather than testing only
    # the least-significant bit. Chosen flipped values are kept inside F_p.
    torch.manual_seed(42)
    pt_base = torch.randint(0, p, (N, num_trials), dtype=torch.int64, device=engine.device)
    pt_mod = pt_base.clone()
    for t in range(num_trials):
        row_idx = t % N
        bit_idx = t % p.bit_length()
        mask = 1 << bit_idx
        value = int(torch.randint(0, p, (1,), dtype=torch.int64).item())
        while (value ^ mask) >= p:
            value = int(torch.randint(0, p, (1,), dtype=torch.int64).item())
        pt_base[row_idx, t] = value
        pt_mod[row_idx, t] = value ^ mask

    ct_base = engine.encrypt(pt_base)
    ct_mod = engine.encrypt(pt_mod)

    for t in range(num_trials):
        diff_bits, total_bits = compute_hamming_distance_bits(
            ct_base[:, t], ct_mod[:, t], bit_width=bit_width
        )
        ratios.append(diff_bits / total_bits)

    avg_ratio = sum(ratios) / len(ratios)
    min_ratio = min(ratios)
    max_ratio = max(ratios)

    print(f"Trials Executed          : {num_trials}")
    print(f"Vector Dimension (N)     : {N} ({N * bit_width} bits per block)")
    print("Interpretation           : diagnostic avalanche ratio; not a proof of cipher security")
    print(f"Cipher Rounds (R)        : {R}")
    print(f"Average Avalanche Ratio  : {avg_ratio:.6f} ({avg_ratio * 100:.2f}%)")
    print(f"Min / Max Avalanche Ratio: {min_ratio:.4f} / {max_ratio:.4f}")
    print(f"Target Acceptance Range  : [0.4500, 0.5500] (Ideal SAC = 0.5000)")

    assert 0.45 <= avg_ratio <= 0.55, f"Avalanche ratio {avg_ratio:.4f} outside [0.45, 0.55]!"
    print("[PASS] Random single-bit avalanche diagnostic passed (not a formal SAC/security proof).")


def test_key_sensitivity_and_statistical_uniformity():
    print("\n======================================================================")
    print("ADDITIONAL CRYPTOGRAPHIC ANALYSIS: KEY SENSITIVITY & UNIFORMITY")
    print("======================================================================")
    N = 64
    p = 2147483579
    engine_k1 = AHCC_Engine(N=N, R=4, p=p, master_seed=50000)
    engine_k2 = AHCC_Engine(N=N, R=4, p=p, master_seed=50001)  # adjacent-seed diagnostic

    torch.manual_seed(1701)
    pt = torch.randint(0, p, (N, 64), dtype=torch.int64, device=engine_k1.device)
    ct_k1 = engine_k1.encrypt(pt)
    ct_k2 = engine_k2.encrypt(pt)

    diff_bits, total_bits = compute_hamming_distance_bits(ct_k1, ct_k2, bit_width=31)
    key_avalanche = diff_bits / total_bits
    print(f"Adjacent-Seed Sensitivity Ratio (50000 vs 50001): {key_avalanche:.6f} ({key_avalanche*100:.2f}%)")
    print("Interpretation                            : sensitivity diagnostic only; not a key-security proof")
    assert 0.45 <= key_avalanche <= 0.55, "Adjacent-seed sensitivity diagnostic failed!"
    print("[PASS] Adjacent-seed sensitivity diagnostic passed.")

    # Statistical Normalized Mean & Variance check on Ciphertext over [0, 1)
    ct_norm = ct_k1.to(torch.float64) / float(p)
    mean_val = ct_norm.mean().item()
    var_val = ct_norm.var().item()
    # Uniform(0, 1) has theoretical mean = 0.5000 and variance = 1/12 ~= 0.0833
    print(f"Ciphertext Normalized Mean     : {mean_val:.4f} (Ideal Uniform = 0.5000)")
    print(f"Ciphertext Normalized Variance : {var_val:.4f} (Ideal Uniform = 0.0833)")
    assert 0.47 <= mean_val <= 0.53 and 0.075 <= var_val <= 0.092, "Statistical uniformity out of bounds!"
    print("[PASS] Basic ciphertext mean/variance sanity check passed; this is not a randomness-suite result.")


if __name__ == "__main__":
    test_phase1_invertibility_and_algebra()
    test_phase2_avalanche_effect(100)
    test_key_sensitivity_and_statistical_uniformity()
    print("\nALL PHASE 1 & PHASE 2 CRYPTOGRAPHIC TESTS PASSED 100%!")
