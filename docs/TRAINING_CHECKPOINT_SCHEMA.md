# Safe, Pickle-Free Training Checkpoint Schema Specification

**Document:** `docs/TRAINING_CHECKPOINT_SCHEMA.md`  
**Date:** 2026-10-26  
**Format Version:** `TorchShield-SafeTrainingCheckpoint-v1` (Container Version 5)  
**Status:** `VERIFIED`

---

## 1. Architectural Philosophy: The Zero-Pickle Imperative

Standard deep learning frameworks historically rely on Python `pickle` (e.g., `torch.save(..., f)`), which serializes arbitrary Python bytecode. This creates an unacceptable remote code execution (RCE) vulnerability when loading checkpoints from external, untrusted, or distributed storage buckets.

The TorchShield Secure Checkpoint Engine establishes a **strictly non-executable serialization contract**:
- **Tensors:** Serialized exclusively as raw binary numerical buffers with strictly allowlisted PyTorch dtypes (`float32`, `float16`, `bfloat16`, `float64`, `int64`, `int32`, `int16`, `int8`, `uint8`, `bool`).
- **Metadata & State:** Stored as deterministic canonical UTF-8 JSON specifications (`RFC 8785`), parsed with duplicate-key rejection and strict schema enforcement.
- **No Deserialization Hooks:** No `__reduce__`, no `eval()`, and zero execution pathways exist during model loading.

---

## 2. Container Format: Version 5 Layout

A Version 5 TorchShield checkpoint file is an authenticated binary container with the following byte sequence:

```
+--------------------------------------------------------------------------------------------------+
|                                  TorchShield VERSION 5 BINARY CONTAINER                                 |
+--------------------------------------------------------------------------------------------------+
| Offset (Bytes) | Length (Bytes) | Field Name       | Description                                 |
+----------------+----------------+------------------+---------------------------------------------+
| 0x00 - 0x07    | 8              | Magic Bytes      | b"TSHDMOD5"                                 |
| 0x08           | 1              | Container Ver    | 0x05 (uint8)                                |
| 0x09 - 0x28    | 32             | Salt             | Public random HKDF derivation salt (non-secret)|
| 0x29           | 1              | Artifact ID Len  | Length of Artifact ID (N <= 64)             |
| 0x2A - (0x2A+N)| N              | Artifact ID      | Unique per-checkpoint UUID bytes            |
| ...            | 1              | Key ID Len       | Length of Key ID string (K <= 255)          |
| ...            | K              | Key ID           | Non-secret KMS Key ID / ARN                 |
| ...            | 2              | Wrapped DEK Len  | Length of wrapped DEK (W <= 65535, BE)      |
| ...            | W              | Wrapped DEK      | KMS-wrapped Data Encryption Key ciphertext  |
| ...            | 4              | Manifest Len     | Length of encrypted manifest (M, uint32 BE) |
| ...            | M              | Encrypted Mani   | AES-256-GCM envelope containing manifest    |
| ...            | Remaining      | Chunk Envelopes  | Sequence of chunk envelopes (b"CHNK" + seq) |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Training State Unified Tensor Namespace

All multidimensional numerical state is unified into a single keyed tensor dictionary prior to chunked authenticated encryption:

| Prefix | Source | Description |
|---|---|---|
| `model_param::{name}` | `model.named_parameters()` | Model trainable weights and biases |
| `model_buffer::{name}` | `model.named_buffers()` | Non-trainable buffers (BatchNorm running_mean, running_var, num_batches_tracked) |
| `opt::{param_name}::{key}` | `optimizer.state[param]` | Optimizer moment buffers (e.g., `exp_avg`, `exp_avg_sq`, `momentum_buffer`) |

### Stable Parameter Name Mapping
PyTorch native optimizers store state keyed by internal integer indices (`0, 1, 2...`) or transient Python object IDs. These indices are unstable across separate process invocations or model reinstantiations. TorchShield maps optimizer state to canonical parameter names (`dict(model.named_parameters())`), ensuring 100% portable and deterministic state restoration.

---

## 4. Training Manifest Specification Schema

The encrypted manifest payload contains the canonical JSON specification (`M_spec`):

```json
{
  "version": 5,
  "artifact_id": "0123456789abcdef0123456789abcdef",
  "key_id": "arn:aws:kms:us-east-1:123456789012:key/abc-123",
  "total_tensors": 48,
  "total_chunks": 12,
  "total_raw_bytes": 48234496,
  "metadata": {
    "schema_version": 1,
    "format": "TorchShield-SafeTrainingCheckpoint-v1",
    "global_step": 15000,
    "epoch": 3,
    "batch_idx": 450,
    "loss": 0.412891,
    "learning_rate": 0.0005,
    "config": {
      "arch": "Transformer-Encoder",
      "batch_size": 32,
      "mixed_precision": "fp16"
    },
    "optimizer": {
      "optimizer_class": "AdamW",
      "param_groups": [
        {
          "lr": 0.0005,
          "betas": [0.9, 0.999],
          "eps": 1e-8,
          "weight_decay": 0.01,
          "param_names": ["layer1.weight", "layer1.bias"]
        }
      ],
      "state_meta": {
        "layer1.weight": {
          "step": 15000
        }
      }
    },
    "scheduler": {
      "step_size": 5000,
      "gamma": 0.5,
      "last_epoch": 15000,
      "_step_count": 15001
    },
    "scaler": {
      "scale": 65536.0,
      "growth_factor": 2.0,
      "backoff_factor": 0.5,
      "growth_interval": 2000,
      "_growth_tracker": 120
    },
    "rng": {
      "python": {
        "version": 3,
        "keys": [1234, 5678, "...625 integers..."],
        "gaussian": null
      },
      "numpy": {
        "algorithm": "MT19937",
        "keys": [9876, 5432, "...uint32 array..."],
        "pos": 624,
        "has_gauss": 0,
        "cached_gaussian": 0.0
      },
      "torch_cpu": "a1b2c3d4... (hex-encoded ByteTensor)",
      "torch_cuda": [
        "e5f6a7b8... (GPU 0 ByteTensor hex)",
        "c9d0e1f2... (GPU 1 ByteTensor hex)"
      ]
    }
  },
  "tensors": [
    {
      "name": "model_param::layer1.weight",
      "dtype": "float32",
      "shape": [512, 512],
      "raw_len": 1048576,
      "num_chunks": 1,
      "start_chunk_seq": 0
    }
  ]
}
```

---

## 5. Safe RNG Deserialization Schema

RNG states are captured across all active generators without pickle:
1. **Python `random`:** Tuple `(3, keys: tuple[int, 625], gaussian: Optional[float])` mapped to JSON object.
2. **NumPy `np.random`:** `('MT19937', ndarray[uint32], pos: int, has_gauss: int, cached_gaussian: float)`.
3. **PyTorch CPU:** Internal state `ByteTensor` (5056 bytes) serialized as a lowercase hex string.
4. **PyTorch CUDA:** Array of hex-encoded `ByteTensor` objects, one per CUDA device.

Restoration reconstructs native C/C++ engine memory buffers directly, guaranteeing that subsequent `random()`, `np.random.randn()`, and `torch.randn()` invocations follow the identical stochastic path.
