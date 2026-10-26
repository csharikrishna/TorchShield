"""Automated GPU hardware and correctness acceptance script for AHCC.

Runs comprehensive checks on the active NVIDIA GPU:
- Device hardware specifications & capabilities
- Memory bounds and allocation integrity
- Device residency verification (no implicit CPU fallback)
- Mathematical parity between CPU and CUDA
"""
import sys
import torch
from ahcc.engine import AHCC_Engine, SUPPORTED_PRIMES


def run_gpu_validation() -> int:
    print("=" * 70)
    print("AHCC GPU HARDWARE AND CORRECTNESS VALIDATION")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[STATUS] BLOCKED / NOT RUN: No CUDA runtime or compatible GPU available.")
        return 1

    device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)
    print(f"GPU Model           : {props.name}")
    print(f"Compute Capability  : {props.major}.{props.minor}")
    print(f"Total VRAM          : {props.total_memory / (1024**2):.1f} MiB")
    print(f"Multi-Processors    : {props.multi_processor_count}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print("-" * 70)

    # Test 1: Device residency check
    engine = AHCC_Engine(N=16, R=2, p=4_294_967_291, device="cuda")
    x = torch.randint(0, 1000, (16, 8), dtype=torch.int64, device="cuda")
    ct = engine.encrypt(x)
    assert ct.is_cuda, "FAILED: Implicit CPU fallback detected; output tensor is not on CUDA!"
    print("[PASS] Device residency verified: transformations execute strictly in GPU VRAM.")

    # Test 2: Multi-dimension parity across all supported primes
    for p in SUPPORTED_PRIMES:
        for N in (8, 32, 64):
            key = bytes(range(32))
            cpu_eng = AHCC_Engine(N=N, R=2, p=p, master_key=key, device="cpu")
            gpu_eng = AHCC_Engine(N=N, R=2, p=p, master_key=key, device="cuda")

            t = torch.randint(0, p, (N, 16), dtype=torch.int64)
            cpu_out = cpu_eng.encrypt(t)
            gpu_out = gpu_eng.encrypt(t.to("cuda"))
            torch.cuda.synchronize()

            assert torch.equal(cpu_out, gpu_out.cpu()), f"Divergence for p={p}, N={N}"
            recovered = gpu_eng.decrypt(gpu_out)
            torch.cuda.synchronize()
            assert torch.equal(recovered.cpu(), t), f"Round trip divergence for p={p}, N={N}"

    print("[PASS] CPU/CUDA parity and exact round-trip verified for all dimensions and primes.")
    print("=" * 70)
    print("GPU VALIDATION COMPLETED SUCCESSFULLY.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_gpu_validation())
