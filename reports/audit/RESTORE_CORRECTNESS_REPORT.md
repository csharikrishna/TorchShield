# TorchShield Training-State Restoration & Equivalence Correctness Report

**Audit Date:** 2026-10-28  
**Scope:** Training State Restoration Engine (`torchshield/training_state.py`)  

---

## 1. The Core Architectural Promise: Exact Training Equivalence

A secure checkpoint engine must guarantee that interrupting training and resuming it from disk produces **100% mathematically identical parameter updates, loss trajectories, and optimizer steps** as an uninterrupted training run.

TorchShield achieves this by capturing and restoring six distinct state planes:

1. **Model Weights & Persistent Buffers:** All `state_dict()` parameters plus non-parameter buffers (e.g. BatchNorm running means and running variances).
2. **Optimizer Internal States:** Momentum buffers, adaptive gradient moments ($v_t, m_t$), step counters, and per-group hyperparameters (learning rate, betas, weight decay).
3. **Learning Rate Schedulers:** Step counts, base learning rates, and internal scheduler counters.
4. **Mixed-Precision Scalers:** PyTorch `GradScaler` scaling factors, growth trackers, and unskipped-step counters.
5. **Deterministic Multi-Generator RNG:** Python random, NumPy random, PyTorch CPU RNG, and per-device CUDA RNG seeds/states.
6. **Auxiliary Component State:** Pluggable `SafeStateProvider` interfaces for dataloader indices, custom metrics, and dataset epoch counters.

---

## 2. Two-Stage Transactional Restoration Protocol

To guarantee that corrupted or incompatible checkpoints never leave model memory in an inconsistent or partially mutated state, restoration operates under a strict two-stage transaction:

```
[ INCOMING CHECKPOINT ]
          |
          v
[ STAGE A: PRE-VALIDATION (No State Mutation) ]
  * Authenticate container & verify Ed25519 digital signature
  * Decrypt & authenticate canonical JSON manifest
  * Verify model architecture: missing params, extra params, shapes, dtypes
  * Verify optimizer layout: group counts, tensor dimensions, param mappings
  * Validate scheduler, scaler, and metadata against strict recursive schema
          |
          +---> [ VALIDATION FAILURE ] ──> Raise TrainingStateError (0 mutations)
          |
          v
[ STAGE B: TRANSACTIONAL APPLICATION (Atomic Rollback Guarded) ]
  * Capture snapshots of current model parameters, optimizer buffers, and RNG states
  * Apply model parameters and persistent buffers in-place
  * Apply optimizer momentum and variance tensors
  * Apply scheduler and scaler dictionaries
  * Apply auxiliary SafeStateProviders
  * Restore Python, NumPy, CPU, and CUDA RNG states (applied LAST)
          |
          +---> [ APPLICATION ERROR ] ────> Catch exception, restore snapshots,
          |                                  raise TrainingStateError
          v
[ RESTORATION COMPLETE (100% Fidelity) ]
```

---

## 3. Strict Compatibility Validation Rules

Under default `strict=True` mode, the loader executes the following validation checks prior to altering any destination state:

1. **Parameter Inventory Check:** Every parameter in `model.named_parameters()` must exist in the checkpoint. Missing parameters or unmapped checkpoint parameters trigger immediate failure.
2. **Persistent Buffer Check:** All registered persistent buffers (BatchNorm `running_mean`, `running_var`) must match exactly.
3. **Tensor Geometry & Type Invariants:** Every parameter must match expected tensor rank, shape, and `dtype` (e.g. `torch.float32`, `torch.bfloat16`).
4. **Optimizer Group Mapping:** Parameter groups must align with model parameters; momentum buffers must match parameter shapes.
5. **Metadata Sanitization:** Schedulers, scalers, and custom configs must pass recursive type inspection rejecting arbitrary Python objects, functions, or non-finite values (`NaN`, `Inf`).

---

## 4. Test Verification & Equivalence Metrics

The restoration engine was verified across multiple deterministic training test suites:

- **Uninterrupted vs Resumed Equivalence:** Evaluated in `tests/test_training_resume_equivalence.py`. Model weights, biases, and optimizer states match with zero deviation ($L_\infty = 0.0$, `torch.equal == True`).
- **Consecutive Restart Cycles:** Evaluated in `tests/test_phase7_training_recovery.py`. Checkpointing across multiple consecutive epochs maintains deterministic step trajectories.
- **Fault Injection & Rollback Protection:** Evaluated in `tests/test_phase1_training_state_strict_transactional.py::test_transactional_restore_rollback_on_failure`. Injected mid-stream exceptions verify that model weights and optimizer states revert 100% to their pre-call snapshots.
- **RNG Determinism:** Evaluated in `tests/test_rng_restoration.py` across Python, NumPy, PyTorch CPU, and CUDA generators.
