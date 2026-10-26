import time
import torch
from ahcc.engine import AHCC_Engine, TensorCipherWrapper


def naive_python_spn_baseline(
    matrix_view: torch.Tensor, M: torch.Tensor, round_keys: torch.Tensor, R: int, p: int
) -> torch.Tensor:
    """
    Sequential Python loop baseline executing the exact same R-round N-dimensional
    AHCC Substitution-Permutation Network (AddKey -> S-Box -> N x N Cauchy MixColumns -> AddKey)
    element-by-element without PyTorch vectorization.
    """
    N, num_blocks = matrix_view.shape
    M_list = M.cpu().tolist()
    rk_list = round_keys.squeeze(-1).cpu().tolist()
    state = matrix_view.cpu().tolist()

    for b in range(num_blocks):
        vec = [state[i][b] for i in range(N)]
        for r in range(R):
            # 1. Add Round Key r
            vec = [(vec[i] + rk_list[r][i]) % p for i in range(N)]
            # 2. SubBytes: x^3 mod p
            vec = [((vec[i] * vec[i] % p) * vec[i]) % p for i in range(N)]
            # 3. MixColumns: N x N Cauchy Matrix-Vector Multiplication mod p
            new_vec = [0] * N
            for i in range(N):
                row_m = M_list[i]
                acc = 0
                for j in range(N):
                    acc = (acc + row_m[j] * vec[j]) % p
                new_vec[i] = acc
            # 4. Add Round Key r+1
            vec = [(new_vec[i] + rk_list[r + 1][i]) % p for i in range(N)]
        for i in range(N):
            state[i][b] = vec[i]

    return torch.tensor(state, dtype=torch.int64)


def run_phase3_benchmarks():
    print("======================================================================")
    print("PHASE 3: AI TENSOR INTEGRATION & THROUGHPUT BENCHMARKING")
    print("======================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Compute Backend : {device.upper()}")

    N = 64
    R = 4
    p = 2147483579
    engine = AHCC_Engine(N=N, R=R, p=p, master_seed=777, device=device)
    wrapper = TensorCipherWrapper(engine)

    # Measure a small naive Python AHCC reference on 16 blocks and verify exact match.
    # This is an internal implementation reference, not an AES-GCM comparison.
    sample_blocks = torch.randint(0, p, (N, 16), dtype=torch.int64, device=engine.device)
    t_b0 = time.perf_counter()
    serial_ct = naive_python_spn_baseline(sample_blocks, engine.M, engine.round_keys, R, p)
    t_b1 = time.perf_counter()
    vec_ct = engine.encrypt(sample_blocks).cpu()
    assert torch.equal(serial_ct, vec_ct), "Serial baseline did not match vectorized AHCC!"
    print(f"Naive Python AHCC reference (16 blocks): {t_b1 - t_b0:.4f} s; correctness match passed.")
    print("Benchmark scope: AHCC vs. its naive Python reference only; no AES-GCM comparison.")
    print("Throughput counts input bytes only; timings are environment-specific, not a security claim.")

    tensor_configs = [
        ("ResNet Image Tensor [1, 3, 224, 224]", (1, 3, 224, 224)),
        ("Batch Vision Tensor [4, 3, 224, 224]", (4, 3, 224, 224)),
        ("Transformer Layer [1, 512, 512]", (1, 512, 512)),
    ]

    results = []

    for name, shape in tensor_configs:
        dummy_ai_tensor = torch.randint(
            0, p, shape, dtype=torch.int64, device=engine.device
        )
        num_elements = dummy_ai_tensor.numel()
        bytes_int64 = num_elements * 8

        # Warmup & Bit-Exact Verification
        ct, meta = wrapper.encrypt_tensor(dummy_ai_tensor)
        dec = wrapper.decrypt_tensor(ct, meta)
        assert torch.equal(dummy_ai_tensor, dec), f"Verification failed on {name}"
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark Vectorized AHCC Encryption
        iters = 3
        t0 = time.perf_counter()
        for _ in range(iters):
            ct, meta = wrapper.encrypt_tensor(dummy_ai_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_enc_time = (t1 - t0) / iters
        throughput_mb_s = (bytes_int64 / (1024 ** 2)) / avg_enc_time
        throughput_gb_s = (bytes_int64 / (1024 ** 3)) / avg_enc_time

        results.append({
            "name": name,
            "elements": num_elements,
            "size_mb": bytes_int64 / (1024 ** 2),
            "enc_ms": avg_enc_time * 1000.0,
            "mb_s": throughput_mb_s,
            "gb_s": throughput_gb_s,
        })

    print("\n-----------------------------------------------------------------------------------------")
    print(f"{'AI Tensor Workload':<36} | {'Elements':>10} | {'Latency (ms)':>12} | {'Throughput':>13}")
    print("-----------------------------------------------------------------------------------------")
    for r in results:
        tp_str = f"{r['mb_s']:7.2f} MB/s" if r["gb_s"] < 1.0 else f"{r['gb_s']:7.3f} GB/s"
        print(
            f"{r['name']:<36} | {r['elements']:>10,d} | {r['enc_ms']:>12.2f} | {tp_str:>13}"
        )
    print("-----------------------------------------------------------------------------------------")
    print("[PASS] Engine matches its Python AHCC reference and round-trips each synthetic tensor.")
    return results


if __name__ == "__main__":
    run_phase3_benchmarks()
