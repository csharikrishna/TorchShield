"""Benchmark the AHCC research transform against authenticated AES-GCM.

CPU baseline:
  python -m benchmarks.benchmark_crypto --device cpu

Target CUDA host, AHCC GPU vs a user-provided AES-GCM backend:
  python -m benchmarks.benchmark_crypto --device cuda --aesgcm-module my_gpu_aesgcm_adapter

An optional adapter module must provide:
  BACKEND_NAME = "descriptive name"
  DEVICE = "cuda"  # declaration only; verify the adapter actually runs on GPU
  encrypt(plaintext: bytes, key: bytes, nonce: bytes, associated_data: bytes) -> bytes

The adapter must measure a complete call including input transfer, GPU completion,
and output transfer to bytes. The harness synchronizes the CUDA device around each
call, but it cannot prove the adapter uses the GPU internally.

The returned value must be AES-256-GCM in the standard wire format (ciphertext || 16-byte
 tag) for the provided 32-byte key/nonce/AAD. The harness verifies output byte-for-byte
 against PyCA's AESGCM before including that backend in timings. Use fresh nonces;
 this harness supplies a new nonce for each invocation.

Important: without --aesgcm-module, the AES-GCM implementation is PyCA/OpenSSL on
CPU. That is not a GPU-vs-GPU comparison and must not be described as one.
"""
from __future__ import annotations

import argparse
import importlib
import secrets
import statistics
import time
from typing import Callable, Optional

import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ahcc import AHCC_Engine, TensorCipherWrapper

AAD = b"AHCC-benchmark-v1"


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _timed(fn: Callable[[], object], device: str, warmup: int, iterations: int) -> tuple[float, object]:
    last = None
    for _ in range(warmup):
        last = fn()
    _sync(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        last = fn()
        _sync(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), last


def _load_backend(module_name: Optional[str]):
    if not module_name:
        return None, "PyCA/OpenSSL AES-GCM (CPU)", "cpu"
    module = importlib.import_module(module_name)
    encrypt = getattr(module, "encrypt", None)
    if not callable(encrypt):
        raise TypeError(f"Adapter module {module_name!r} must export callable encrypt().")
    backend_name = str(getattr(module, "BACKEND_NAME", module_name))
    declared_device = str(getattr(module, "DEVICE", "unspecified"))
    if declared_device.lower() != "cuda":
        raise ValueError(
            f"Adapter {module_name!r} must explicitly set DEVICE='cuda' to be treated as a GPU backend; "
            f"it declared {declared_device!r}."
        )
    return encrypt, backend_name, "cuda-adapter (declared; device use must be confirmed by adapter instrumentation)"


def _make_input(size_bytes: int, N: int, p: int) -> torch.Tensor:
    n_values = size_bytes // 8
    n_values = max(N, (n_values // N) * N)
    # Use values below p so AHCC sees valid finite-field elements, while comparing
    # AES-GCM against the exact little-endian int64 serialization of the same data.
    return torch.randint(0, p, (n_values,), dtype=torch.int64, device="cpu")


def run(args: argparse.Namespace) -> int:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("ERROR: CUDA requested but torch.cuda.is_available() is False.")
        print(f"PyTorch={torch.__version__}; torch.version.cuda={torch.version.cuda!r}")
        return 2

    backend_encrypt, backend_name, backend_device = _load_backend(args.aesgcm_module)
    if backend_encrypt is not None and not args.device.startswith("cuda"):
        print("ERROR: use --device cuda when comparing AHCC with a GPU AES-GCM adapter; mixed CPU/GPU results are rejected.")
        return 2
    if backend_encrypt is not None and not torch.cuda.is_available():
        print("ERROR: a GPU AES-GCM adapter was requested but CUDA is unavailable in this PyTorch runtime.")
        return 2
    aesgcm = AESGCM

    print(f"AHCC device: {args.device}")
    print(f"AES-GCM backend: {backend_name} [{backend_device}]")
    if args.device.startswith("cuda") and backend_encrypt is None:
        print("NOTE: AES-GCM still runs on CPU; this is not a GPU-vs-GPU performance comparison.")

    N, R, p = args.N, args.R, args.p
    key = secrets.token_bytes(32)
    engine = AHCC_Engine(N=N, R=R, p=p, master_key=key, device=args.device)
    wrapper = TensorCipherWrapper(engine)

    print("\nSize (input) | AHCC latency | AHCC throughput | AES-GCM latency | AES-GCM throughput")
    print("-------------|--------------|-----------------|-----------------|-------------------")
    for requested_bytes in args.sizes:
        values = _make_input(requested_bytes, N, p)
        plaintext = values.contiguous().numpy().tobytes()
        actual_bytes = len(plaintext)
        if actual_bytes == 0:
            continue
        aad = AAD

        # Correctness checks outside timed regions.
        ahcc_ct, meta = wrapper.encrypt_tensor(values)
        ahcc_back = wrapper.decrypt_tensor(ahcc_ct, meta).cpu()
        if not torch.equal(values, ahcc_back):
            raise AssertionError("AHCC tensor round trip failed during benchmark correctness validation.")

        aes_key = secrets.token_bytes(32)
        reference_cipher = AESGCM(aes_key)
        nonce = secrets.token_bytes(12)
        expected = reference_cipher.encrypt(nonce, plaintext, aad)
        if backend_encrypt is None:
            measured = reference_cipher.encrypt(nonce, plaintext, aad)
        else:
            measured = backend_encrypt(plaintext, aes_key, nonce, aad)
        if measured != expected:
            raise AssertionError(
                f"AES-GCM backend failed byte-for-byte interoperability validation at {actual_bytes} bytes."
            )
        if reference_cipher.decrypt(nonce, measured, aad) != plaintext:
            raise AssertionError("AES-GCM decryption/reference validation failed.")

        # Reuse one AES-GCM key schedule like a normal long-lived context, while
        # guaranteeing a fresh non-repeating nonce for each timed encryption.
        benchmark_key = secrets.token_bytes(32)
        benchmark_cipher = AESGCM(benchmark_key) if backend_encrypt is None else None
        nonce_prefix = secrets.token_bytes(4)
        nonce_counter = 0

        def next_nonce():
            nonlocal nonce_counter
            if nonce_counter >= (1 << 64):
                raise RuntimeError("Benchmark exhausted its unique nonce counter; use a new key.")
            nonce = nonce_prefix + nonce_counter.to_bytes(8, "big")
            nonce_counter += 1
            return nonce

        def ahcc_call():
            result, _ = wrapper.encrypt_tensor(values)
            # Include device-to-host completion so each timing measures a completed result.
            if result.device.type == "cuda":
                result = result.cpu()
            return result

        def aes_call():
            local_nonce = next_nonce()
            if backend_encrypt is None:
                return benchmark_cipher.encrypt(local_nonce, plaintext, aad)
            return backend_encrypt(plaintext, benchmark_key, local_nonce, aad)

        ahcc_seconds, _ = _timed(ahcc_call, args.device, args.warmup, args.iterations)
        aes_device_for_sync = args.device if backend_encrypt is not None else "cpu"
        aes_seconds, _ = _timed(aes_call, aes_device_for_sync, args.warmup, args.iterations)
        ahcc_mib_s = actual_bytes / (1024 * 1024) / ahcc_seconds
        aes_mib_s = actual_bytes / (1024 * 1024) / aes_seconds
        print(
            f"{actual_bytes:>11,d} B | {ahcc_seconds * 1000:>10.3f} ms | {ahcc_mib_s:>12.2f} MiB/s | "
            f"{aes_seconds * 1000:>12.3f} ms | {aes_mib_s:>14.2f} MiB/s"
        )

    print("\nMeasurements are environment-specific. They do not imply AHCC security.")
    if backend_encrypt is None:
        print("This run used CPU AES-GCM. A GPU-AES-GCM comparison requires a verified CUDA adapter module.")
    else:
        print("The harness checked the adapter's output against PyCA AES-GCM byte-for-byte before timing.")
        print("Confirm actual GPU execution separately (e.g. profiler/device instrumentation); DEVICE='cuda' is a declaration only.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="AHCC tensor device: cpu or cuda[:index].")
    parser.add_argument("--N", type=int, default=32, help="AHCC block dimension (default: 32).")
    parser.add_argument("--R", type=int, default=2, help="AHCC rounds (default: 2 for benchmark duration).")
    parser.add_argument("--p", type=int, default=2_147_483_579, help="Supported AHCC field prime.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 16384, 65536], help="Requested payload sizes in bytes.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--aesgcm-module", default=None, help="Optional GPU AES-GCM adapter module to benchmark.")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")
    if args.N < 1 or args.N > 1024 or args.R < 1 or args.R > 64 or args.p not in (2_147_483_579, 4_294_967_291):
        parser.error("Invalid AHCC parameters (require 1 <= N <= 1024, 1 <= R <= 64, and a supported prime).")
    if any(size <= 0 for size in args.sizes):
        parser.error("All --sizes values must be positive byte counts.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
