"""Runtime smoke-test for actual CUDA support; run on the target NVIDIA host.

Run: python -m benchmarks.check_cuda
A CPU-only environment reports SKIP rather than pretending CUDA was tested.
"""
from __future__ import annotations

import sys
import torch

from ahcc import AHCC_Engine
from ahcc.engine import SUPPORTED_PRIMES


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("SKIP: torch.cuda.is_available() is False; CUDA execution was not tested.")
        print("Install a CUDA-enabled PyTorch build and run this command on the target NVIDIA host.")
        return 0

    print(f"CUDA device count: {torch.cuda.device_count()}")
    for device_id in range(torch.cuda.device_count()):
        print(f"CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")

    for p in SUPPORTED_PRIMES:
        key = bytes((i * 17 + p % 251) % 256 for i in range(32))
        cpu = AHCC_Engine(N=16, R=2, p=p, master_key=key, device="cpu")
        gpu = AHCC_Engine(N=16, R=2, p=p, master_key=key, device="cuda")
        # Boundary-heavy inputs exercise limb arithmetic and signed-int64 safety.
        x_cpu = torch.tensor(
            [(0, 1, p - 1, p - 2, p // 2, p // 3, 17, 65535)[i % 8] for i in range(16 * 7)],
            dtype=torch.int64,
        ).view(16, 7)
        cpu_ct = cpu.encrypt(x_cpu)
        gpu_ct = gpu.encrypt(x_cpu.to("cuda"))
        torch.cuda.synchronize()
        if not torch.equal(cpu_ct, gpu_ct.cpu()):
            raise AssertionError(f"CPU/CUDA ciphertext mismatch for p={p}.")
        recovered = gpu.decrypt(gpu_ct)
        torch.cuda.synchronize()
        if not torch.equal(recovered.cpu(), x_cpu):
            raise AssertionError(f"CUDA round-trip mismatch for p={p}.")
        print(f"PASS: CPU/CUDA parity and round-trip for p={p}.")

    print("PASS: CUDA smoke checks completed. This is not a performance benchmark or a cryptographic security proof.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
