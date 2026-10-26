# GPU Validation Guide

The package includes a smoke test and benchmark harness, but the current development environment used for the audit has a CPU-only PyTorch build. CUDA execution was therefore **not tested here**. Use these commands on the actual target NVIDIA GPU.

## 1. Install CUDA-enabled PyTorch

Install a PyTorch build appropriate for the operating system, Python version, and NVIDIA driver using the official selector: <https://pytorch.org/get-started/locally/>. Then confirm that the runtime sees the device:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA DEVICE')"
```

`torch.cuda.is_available()` must return `True` before attempting GPU checks.

## 2. Install the project and run the CUDA smoke test

From the extracted `AHCC_GPU_Engine` directory:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
python -m benchmarks.check_cuda
```

The CUDA smoke test compares CPU and CUDA ciphertext outputs for both supported primes, checks boundary-heavy input values, and verifies CUDA decrypt/encrypt round trips. Save the terminal output and record the GPU model, driver, Python, PyTorch, and CUDA versions.

## 3. Benchmark

CPU comparison (included for diagnostics only):

```bash
python -m benchmarks.benchmark_crypto --device cpu
```

AHCC CUDA vs CPU AES-GCM (not GPU-vs-GPU):

```bash
python -m benchmarks.benchmark_crypto --device cuda
```

For a GPU-vs-GPU benchmark, pass an adapter to a validated GPU AES-GCM implementation:

```bash
python -m benchmarks.benchmark_crypto --device cuda --aesgcm-module your_gpu_aesgcm_adapter
```

The adapter must export `BACKEND_NAME`, `DEVICE = "cuda"`, and `encrypt(plaintext, key, nonce, associated_data) -> ciphertext_and_tag`. It must include data transfer and GPU synchronization in its end-to-end timing. The harness verifies its output byte-for-byte against PyCA AES-GCM before timing, but a declared `DEVICE` string is not proof of actual GPU execution; verify with profiler/device instrumentation.

## 4. Report results without overclaiming

Record multiple payload sizes (small tensors, medium batches, and large model-sized buffers), at least 5–10 measured iterations after warm-up, median and spread, and two timing modes where possible:

- **End-to-end:** includes host/device transfers, allocations, launch overhead and output transfer.
- **Kernel-only:** uses CUDA events around device work, reported separately from end-to-end.

Use the same GPU, driver, host, payload bytes, key size, and authenticated-encryption mode for the comparison. Confirm the AES-GCM baseline passes appropriate test vectors and tamper tests. Report failed sizes, errors, and memory use too. The current AHCC transform uses `int64` tensor operations, not Tensor Cores or a fused CUDA cryptographic kernel, so do not claim Tensor Core acceleration.
