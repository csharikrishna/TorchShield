# Benchmark Reproduction Guide

This guide describes how to reproduce every test, benchmark, and statistical analysis reported in this investigation.

---

## 1. Quick Verification Commands

Run the full regression test suite (97 tests):
```bash
python -m pytest -v
```

Run cryptanalytic attack demonstrations:
```bash
python -m tests.attack_experiments
```

Run CUDA hardware validation:
```bash
python -m benchmarks.run_gpu_validation
```

Run end-to-end model checkpoint demonstration:
```bash
python -m examples.encrypted_checkpoint_demo
```

---

## 2. Performance Benchmarking Protocol

Run the fair cipher throughput benchmark:
```bash
python -u -m benchmarks.benchmark_cipher --sizes 4096 65536 1048576 4194304 --dimensions 16 32 64 --rounds 2 4 --iterations 5
```

Run the model loading benchmark:
```bash
python -m benchmarks.benchmark_tensor_loading
```

Run statistical comparison and plot generation:
```bash
python -m analysis.analyze_results
python -m analysis.plot_results
```
Outputs will be generated in:
- `results/summary.csv`
- `results/summary.json`
- `results/raw/cipher_benchmarks_raw.json`
- `results/throughput_comparison.png`
