# Phase Zero — Initial Repository Baseline Audit

## 1. Executive Summary & Repository Identity
- **Repository URL:** `https://github.com/csharikrishna/TorchShield`
- **Inspected Commit:** `7b07f1f` (`master` branch, clean working tree)
- **Declared Version:** `0.7.0` (in `pyproject.toml`, `torchshield/__init__.py`; note discrepancy in `torchshield/cli.py` which fell back to `0.6.0-dev`)
- **Audit Date:** 2026-10-28

## 2. Environment & Dependency Baseline
- **Operating System:** Microsoft Windows 11 (win32)
- **Python Version:** 3.12.10
- **PyTorch Version:** 2.5.1+cu121
- **CUDA Availability:** True (1 GPU: NVIDIA GeForce GTX 1650, Capability: 7.5, 4 GB VRAM)
- **Installed Cryptography:** `cryptography 46.0.5`
- **Installed Key Management / Ecosystem Libraries:**
  - `boto3 1.43.120`
  - `transformers 5.3.0`
  - `accelerate 1.13.0`
  - `datasets 4.8.3`
  - `safetensors 0.7.0`
  - `numpy 2.4.2`
  - `pytest 9.0.2`

## 3. Initial Test Execution & Collection Baseline
### Command 1: Root `pytest` (unscoped)
- **Command:** `pytest --tb=short`
- **Exit Code:** 1 (Failed during test collection)
- **Summary:** Collected 204 items, 7 collection errors.
- **Root Cause:** `pyproject.toml` lacked a `[tool.pytest.ini_options]` configuration setting `testpaths = ["tests"]` and `norecursedirs = ["archives*"]`. Pytest recursed into `archives/v1_chaotic_crypto_prototype/tests/`, which attempts to import historical research modules (`ahcc.engine`) that are not on `sys.path`.

### Command 2: Explicit Target `pytest tests`
- **Command:** `pytest tests`
- **Exit Code:** 0 (Passed)
- **Summary:** 204 passed, 2 warnings in 31.92s.
- **Warnings:** 2 SwigPy deprecation warnings from C-extensions.
- **Test Categories:**
  - AEAD known answers & tampering (9 tests)
  - Signatures & artifact integrity (10 tests)
  - Freshness ledger & rollback (9 tests)
  - Hugging Face integration (5 tests)
  - Key provider contract & KMS provider (14 tests)
  - Optimizer resume & training state (17 tests)
  - Phase 2/4/6 protocol suites (140 tests)

## 4. Coverage Limitations & Missing Infrastructure
1. **Cloud KMS Services:** No live AWS KMS, Google Cloud KMS, Azure Key Vault, or HashiCorp Vault credentials or endpoints configured. Only mock/LocalStack parameter options exist in AWS provider; GCP, Azure, and Vault providers are not yet implemented.
2. **Distributed Infrastructure:** No multi-node cluster, real FSDP cluster, or live etcd cluster available in the local execution environment. Distributed tests run single-process simulated multi-shard loops.
3. **Hardware-Specific Features:**
   - NVIDIA GPUDirect Storage (`cuFile` / GDS) is not supported on Windows / consumer GTX 1650 (requires Linux, GPUDirect Storage driver, supported enterprise NVMe / storage backend).
   - NVIDIA GPU Confidential Computing (requires Hopper H100/H200/B200 architecture and CC driver).
   - Post-quantum cryptography libraries (e.g. `liboqs` / `pqcrypto` / standard FIPS 203/204 packages) are not yet integrated into the core package.

## 5. Identified Suspected Defects & Architectural Issues

### 5.1 Training State Restoration (`torchshield/training_state.py`)
- **Defect: No Strict Validation Mode:** Missing parameters, missing buffers, unexpected checkpoint keys, shape mismatches, and dtype mismatches are silently ignored or cause undefined partial mutations rather than failing closed.
- **Defect: Non-Transactional Restoration:** Mutations are applied directly in-place to `model.named_parameters()` and `optimizer.state`. If an error occurs midway (e.g. invalid tensor or corrupted state), the model and optimizer are left in a corrupted, half-restored state without rollback.
- **Defect: Optimizer Compatibility Checks Missing:** Param group count, optimizer class, momentum tensor dimensions, and param group mappings are not pre-validated before mutation.
- **Defect: Loose Scheduler & Scaler Metadata Validation:** No recursive type validation for scheduler/scaler state dicts.

### 5.2 Cryptographic Protocol & Header Wire Format (`torchshield/secure.py`, `torchshield/checkpoint.py`)
- **Documentation vs Code Inconsistency in Magic Header:** `secure.py` docstring claims `magic (8 bytes)` and comment says `MAGIC = b"TorchShieldSEAL" # 8 bytes`, but `len(b"TorchShieldSEAL")` is 15 bytes (total header 16 bytes).
- **RFC 8785 Canonical JSON Claim:** The documentation claims RFC 8785 JCS, but serialization uses `json.dumps(..., sort_keys=True, separators=(',', ':'))`. This must either be implemented according to RFC 8785 or formally designated as TorchShield Canonical JSON v1.
- **Signature Trust Policy:** Checkpoint loading can accept self-signed checkpoints without enforcing an expected signer public key or explicit trust anchor policy when caller requires publisher identity.

### 5.3 Distributed Sharded Manifests (`torchshield/distributed.py`)
- **Path Traversal Vulnerability:** Shard filenames in `global_manifest.json` are joined via `os.path.join(checkpoint_dir, filename)` without path normalization or verifying the target stays within `checkpoint_dir`.
- **Symlink Attacks:** No check rejecting symlinks when loading shard files.
- **Duplicate/Missing Rank Integrity:** Manifest does not strictly check that shards contain unique, consecutive ranks `0..world_size-1`.

### 5.4 Hugging Face Trainer Integration (`torchshield/integrations/huggingface.py`)
- **Silent Insecure Fallback:** If `TorchShieldTrainer` has a `vault_key` configured, but no `.shield` file is found in the checkpoint directory, it silently falls back to `super()._load_optimizer_and_scheduler()` which invokes unsafe `torch.load` (pickle).
- **Insecure File Deletion Race:** `TorchShieldTrainerCallback` deletes `optimizer.pt`, `scheduler.pt`, and `rng_state.pth` with a swallowed `OSError` without validating the written vault container's cryptographic integrity.

### 5.5 Key Provider Architecture (`torchshield/key_providers/`)
- Only `AWSKMSProvider` and `LocalDevKeyProvider` exist.
- Missing `GCPKMSProvider`, `AzureKeyVaultProvider`, `HashiCorpVaultProvider`.
- Missing optional dependency extras in `pyproject.toml` (`[gcp]`, `[azure]`, `[vault]`).
