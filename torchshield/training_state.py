"""Safe, Pickle-Free Training Checkpoint Serialization and Restoration Engine.

Preserves and restores complete training state:
1. Model parameters and buffers
2. Optimizer state (AdamW, Adam, SGD, etc., mapped to stable parameter names)
3. Learning rate scheduler state
4. Mixed-precision gradient scaler state (torch.amp / torch.cuda.amp)
5. RNG states (Python, NumPy, PyTorch CPU, PyTorch CUDA)
6. Training progress (step, epoch, custom configuration)

Strictly non-executable: Zero pickle deserialization. All tensors are stored as dense
numerical buffers, and metadata is stored as strictly allowlisted canonical JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import random
import secrets
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from torchshield.checkpoint import SecureModelCheckpoint, CheckpointError, ResourceLimits
from torchshield.key_providers.base import KeyProvider


class TrainingStateError(ValueError):
    """Raised when training state serialization or restoration fails."""


# ---------------------------------------------------------------------------
# Safe RNG Serializers (No Pickle)
# ---------------------------------------------------------------------------

def serialize_python_rng() -> dict:
    """Serialize Python random state: (version: 3, keys: tuple of 625 ints, gaussian: Optional[float])."""
    st = random.getstate()
    return {
        "version": int(st[0]),
        "keys": list(st[1]),
        "gaussian": float(st[2]) if st[2] is not None else None,
    }


def restore_python_rng(state_dict: dict) -> None:
    """Restore Python random state from allowlisted dictionary."""
    if not isinstance(state_dict, dict) or "keys" not in state_dict:
        raise TrainingStateError("Invalid Python RNG state dictionary.")
    keys_tuple = tuple(int(k) for k in state_dict["keys"])
    gaussian = state_dict.get("gaussian")
    st = (int(state_dict.get("version", 3)), keys_tuple, float(gaussian) if gaussian is not None else None)
    random.setstate(st)


def serialize_numpy_rng() -> dict:
    """Serialize NumPy MT19937 RNG state without pickle."""
    st = np.random.get_state()
    return {
        "algorithm": str(st[0]),
        "keys": st[1].tolist(),
        "pos": int(st[2]),
        "has_gauss": int(st[3]),
        "cached_gaussian": float(st[4]),
    }


def restore_numpy_rng(state_dict: dict) -> None:
    """Restore NumPy MT19937 RNG state."""
    required = ("algorithm", "keys", "pos", "has_gauss", "cached_gaussian")
    if not isinstance(state_dict, dict) or not all(k in state_dict for k in required):
        raise TrainingStateError("Invalid NumPy RNG state dictionary.")
    keys_arr = np.array(state_dict["keys"], dtype=np.uint32)
    st = (
        str(state_dict["algorithm"]),
        keys_arr,
        int(state_dict["pos"]),
        int(state_dict["has_gauss"]),
        float(state_dict["cached_gaussian"]),
    )
    np.random.set_state(st)


def serialize_torch_cpu_rng() -> str:
    """Serialize PyTorch CPU RNG state ByteTensor as hex string."""
    st = torch.get_rng_state()
    return st.numpy().tobytes().hex()


def restore_torch_cpu_rng(hex_str: str) -> None:
    """Restore PyTorch CPU RNG state from hex string."""
    b = bytes.fromhex(hex_str)
    st = torch.frombuffer(bytearray(b), dtype=torch.uint8)
    torch.set_rng_state(st)


def serialize_torch_cuda_rng() -> List[str]:
    """Serialize PyTorch CUDA RNG states for all devices."""
    if not torch.cuda.is_available():
        return []
    states = torch.cuda.get_rng_state_all()
    return [s.numpy().tobytes().hex() for s in states]


def restore_torch_cuda_rng(hex_list: List[str]) -> None:
    """Restore PyTorch CUDA RNG states for all devices."""
    if not torch.cuda.is_available() or not hex_list:
        return
    states = [torch.frombuffer(bytearray(bytes.fromhex(h)), dtype=torch.uint8) for h in hex_list]
    torch.cuda.set_rng_state_all(states)


def serialize_all_rng() -> dict:
    """Capture full environment RNG state across Python, NumPy, PyTorch CPU and CUDA."""
    return {
        "python": serialize_python_rng(),
        "numpy": serialize_numpy_rng(),
        "torch_cpu": serialize_torch_cpu_rng(),
        "torch_cuda": serialize_torch_cuda_rng(),
    }


def restore_all_rng(rng_dict: dict) -> None:
    """Restore full environment RNG state across Python, NumPy, PyTorch CPU and CUDA."""
    if not isinstance(rng_dict, dict):
        return
    if "python" in rng_dict:
        restore_python_rng(rng_dict["python"])
    if "numpy" in rng_dict:
        restore_numpy_rng(rng_dict["numpy"])
    if "torch_cpu" in rng_dict:
        restore_torch_cpu_rng(rng_dict["torch_cpu"])
    if "torch_cuda" in rng_dict:
        restore_torch_cuda_rng(rng_dict["torch_cuda"])


# ---------------------------------------------------------------------------
# Safe Optimizer Extraction & In-Place Restoration (Mapped to Parameter Names)
# ---------------------------------------------------------------------------

def extract_optimizer_state(
    optimizer: Optimizer, model: nn.Module
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Safely extract optimizer state mapping parameter objects to model parameter names.

    Returns:
        Tuple of (optimizer_tensors_dict, optimizer_metadata_dict).
    """
    param_to_name: Dict[Any, str] = {p: name for name, p in model.named_parameters()}
    optimizer_tensors: Dict[str, torch.Tensor] = {}
    optimizer_scalars: Dict[str, Dict[str, Any]] = {}

    for param, state in optimizer.state.items():
        if param not in param_to_name:
            continue
        param_name = param_to_name[param]
        optimizer_scalars[param_name] = {}

        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                optimizer_tensors[f"opt::{param_name}::{k}"] = v.detach().to("cpu")
            elif isinstance(v, (int, float, str, bool)):
                optimizer_scalars[param_name][k] = v
            else:
                raise TrainingStateError(
                    f"Unsupported optimizer state item '{k}' of type {type(v)}. "
                    "Only torch.Tensor and standard scalar types are permitted."
                )

    # Serialize param_groups without storing Python object references
    serialized_param_groups = []
    for group in optimizer.param_groups:
        grp = {}
        for k, v in group.items():
            if k == "params":
                grp["param_names"] = [param_to_name[p] for p in v if p in param_to_name]
            elif isinstance(v, tuple):
                grp[k] = list(v)
            elif isinstance(v, (int, float, str, bool, list)) or v is None:
                grp[k] = v
            else:
                raise TrainingStateError(f"Unsupported param_group hyperparameter '{k}' of type {type(v)}.")
        serialized_param_groups.append(grp)

    optimizer_meta = {
        "type": type(optimizer).__name__,
        "param_groups": serialized_param_groups,
        "scalars": optimizer_scalars,
    }
    return optimizer_tensors, optimizer_meta


def restore_optimizer_state(
    optimizer: Optimizer,
    model: nn.Module,
    optimizer_tensors: Dict[str, torch.Tensor],
    optimizer_meta: dict,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """Restore optimizer hyperparameters and momentum tensors in-place."""
    name_to_param: Dict[str, nn.Parameter] = dict(model.named_parameters())

    # 1. Restore param_group hyperparameters (lr, weight_decay, betas, etc.)
    saved_groups = optimizer_meta.get("param_groups", [])
    if len(saved_groups) == len(optimizer.param_groups):
        for saved_grp, live_grp in zip(saved_groups, optimizer.param_groups):
            for k, v in saved_grp.items():
                if k == "param_names":
                    continue
                if isinstance(v, list) and k in ("betas",):
                    live_grp[k] = tuple(v)
                else:
                    live_grp[k] = v

    # 2. Restore state tensors and scalars
    saved_scalars = optimizer_meta.get("scalars", {})
    prefix = "opt::"

    for name, param in name_to_param.items():
        param_state: Dict[str, Any] = {}

        # Restore scalars (e.g. integer step)
        if name in saved_scalars:
            for k, v in saved_scalars[name].items():
                param_state[k] = v

        # Restore tensors (e.g. exp_avg, exp_avg_sq, momentum_buffer)
        param_prefix = f"{prefix}{name}::"
        for full_key, tensor in optimizer_tensors.items():
            if full_key.startswith(param_prefix):
                state_key = full_key[len(param_prefix):]
                t = tensor.to(device=param.device)
                param_state[state_key] = t

        if param_state:
            optimizer.state[param] = param_state


# ---------------------------------------------------------------------------
# Safe Training Checkpoint Container
# ---------------------------------------------------------------------------

@dataclass
class TrainingProgress:
    """Structured training progress metadata."""
    global_step: int = 0
    epoch: int = 0
    batch_idx: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)


def save_training_checkpoint(
    output_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    progress: Optional[TrainingProgress] = None,
    key: Optional[bytes] = None,
    key_id: str = "",
    key_provider: Optional[KeyProvider] = None,
    encryption_context: Optional[Dict[str, str]] = None,
    signer_private_key: Optional[bytes] = None,
    chunk_bytes: int = 4 * 1024 * 1024,
    limits: Optional[ResourceLimits] = None,
) -> None:
    """Save an end-to-end encrypted, authenticated AI training checkpoint with zero pickle.

    Encrypts model parameters/buffers, optimizer state tensors, scheduler state,
    scaler state, and all RNG states in a single atomic container.
    """
    if key is None and key_provider is None:
        raise ValueError("Either 'key' (32 bytes) or 'key_provider' must be specified.")

    if key_provider is not None and not key_id:
        key_id = key_provider.get_key_id()

    # 1. Extract model parameters and buffers
    unified_tensors: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        unified_tensors[f"model_param::{name}"] = param.detach().to("cpu")
    for name, buf in model.named_buffers():
        unified_tensors[f"model_buffer::{name}"] = buf.detach().to("cpu")

    # 2. Extract optimizer state if provided
    optimizer_meta: Optional[dict] = None
    if optimizer is not None:
        opt_tensors, optimizer_meta = extract_optimizer_state(optimizer, model)
        unified_tensors.update(opt_tensors)

    # 3. Extract scheduler state if provided
    scheduler_meta: Optional[dict] = None
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        scheduler_meta = scheduler.state_dict()
        scheduler_meta = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in scheduler_meta.items()
        }

    # 4. Extract gradient scaler state if provided
    scaler_meta: Optional[dict] = None
    if scaler is not None and hasattr(scaler, "state_dict"):
        scaler_meta = scaler.state_dict()

    # 5. Capture full RNG state
    rng_meta = serialize_all_rng()

    # 6. Compile structured training manifest
    prog = progress or TrainingProgress()
    training_manifest = {
        "schema_version": 1,
        "format": "TorchShield-SafeTrainingCheckpoint-v1",
        "global_step": prog.global_step,
        "epoch": prog.epoch,
        "batch_idx": prog.batch_idx,
        "loss": prog.loss,
        "learning_rate": prog.learning_rate,
        "config": prog.config,
        "optimizer": optimizer_meta,
        "scheduler": scheduler_meta,
        "scaler": scaler_meta,
        "rng": rng_meta,
        "encryption_context": encryption_context,
    }

    # 7. Encrypt and write checkpoint using SecureModelCheckpoint
    SecureModelCheckpoint.save_checkpoint(
        state_dict=unified_tensors,
        output_path=output_path,
        key=key,
        chunk_bytes=chunk_bytes,
        key_id=key_id,
        metadata=training_manifest,
        key_provider=key_provider,
        encryption_context=encryption_context,
        signer_private_key=signer_private_key,
        limits=limits,
    )


def load_training_checkpoint(
    input_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    key: Optional[bytes] = None,
    key_provider: Optional[KeyProvider] = None,
    encryption_context: Optional[Dict[str, str]] = None,
    expected_signer_public_key: Optional[bytes] = None,
    device: Union[str, torch.device] = "cpu",
    restore_rng: bool = True,
    limits: Optional[ResourceLimits] = None,
) -> TrainingProgress:
    """Authenticate and restore a complete training checkpoint in-place with zero pickle.

    Returns:
        TrainingProgress metadata (global_step, epoch, loss, config).
    """
    if key is None and key_provider is None:
        raise ValueError("Either 'key' (32 bytes) or 'key_provider' must be specified.")

    # Load unified tensors and authenticated training manifest
    restored_tensors, manifest_meta = SecureModelCheckpoint.load_checkpoint(
        input_path=input_path,
        key=key,
        device="cpu",
        target_model=None,
        limits=limits,
        return_metadata=True,
        key_provider=key_provider,
        encryption_context=encryption_context,
        expected_signer_public_key=expected_signer_public_key,
    )

    training_meta = manifest_meta.get("metadata", manifest_meta)

    # 2. In-place restore model parameters and buffers
    name_to_param = dict(model.named_parameters())
    name_to_buf = dict(model.named_buffers())

    for k, tensor in restored_tensors.items():
        if k.startswith("model_param::"):
            param_name = k[len("model_param::"):]
            if param_name in name_to_param:
                name_to_param[param_name].data.copy_(tensor.to(device=name_to_param[param_name].device))
        elif k.startswith("model_buffer::"):
            buf_name = k[len("model_buffer::"):]
            if buf_name in name_to_buf:
                name_to_buf[buf_name].copy_(tensor.to(device=name_to_buf[buf_name].device))

    # 3. Restore optimizer state if provided
    if optimizer is not None and "optimizer" in training_meta and training_meta["optimizer"] is not None:
        opt_tensors = {k: v for k, v in restored_tensors.items() if k.startswith("opt::")}
        restore_optimizer_state(
            optimizer=optimizer,
            model=model,
            optimizer_tensors=opt_tensors,
            optimizer_meta=training_meta["optimizer"],
            device=device,
        )

    # 4. Restore scheduler state if provided
    if scheduler is not None and "scheduler" in training_meta and training_meta["scheduler"] is not None:
        sched_state = training_meta["scheduler"]
        try:
            scheduler.load_state_dict(sched_state)
        except Exception as exc:
            raise TrainingStateError(f"Failed to restore scheduler state: {exc}") from exc

    # 5. Restore gradient scaler state if provided
    if scaler is not None and "scaler" in training_meta and training_meta["scaler"] is not None:
        scaler_state = training_meta["scaler"]
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as exc:
            raise TrainingStateError(f"Failed to restore gradient scaler state: {exc}") from exc

    # 6. Restore RNG state if requested
    if restore_rng and "rng" in training_meta and training_meta["rng"] is not None:
        restore_all_rng(training_meta["rng"])

    # 7. Return progress metadata
    return TrainingProgress(
        global_step=training_meta.get("global_step", 0),
        epoch=training_meta.get("epoch", 0),
        batch_idx=training_meta.get("batch_idx", 0),
        loss=training_meta.get("loss"),
        learning_rate=training_meta.get("learning_rate"),
        config=training_meta.get("config", {}),
    )


def inspect_training_checkpoint(
    input_path: str,
    key: Optional[bytes] = None,
    key_provider: Optional[KeyProvider] = None,
    encryption_context: Optional[Dict[str, str]] = None,
    expected_signer_public_key: Optional[bytes] = None,
    limits: Optional[ResourceLimits] = None,
) -> Dict[str, Any]:
    """Inspect an TorchShield training checkpoint header and optional authenticated metadata.

    If key/key_provider is provided, authenticates the manifest and returns the full
    training metadata (global step, epoch, loss, config, optimizer state summary, tensor inventory)
    without reading full weight tensors.
    If no key is provided, extracts unauthenticated container headers (format, version,
    artifact ID, key ID, wrapped key size, file size) without requiring cryptographic credentials.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Checkpoint file not found: {input_path}")

    file_size = os.path.getsize(input_path)
    lim = limits or ResourceLimits()
    if file_size > lim.max_file_size:
        raise CheckpointError(f"File size {file_size} exceeds maximum limit {lim.max_file_size}.")

    with open(input_path, "rb") as f:
        magic = f.read(8)
        if magic == SecureModelCheckpoint.MAGIC_V5:
            ver_bytes = f.read(1)
            if len(ver_bytes) < 1:
                raise CheckpointError("Truncated checkpoint header.")
            ver = ver_bytes[0]
            if ver != SecureModelCheckpoint.VERSION_V5:
                raise CheckpointError(f"Unsupported checkpoint version: {ver}")
            salt = f.read(SecureModelCheckpoint.SALT_SIZE_V5)
            if len(salt) != SecureModelCheckpoint.SALT_SIZE_V5:
                raise CheckpointError("Truncated checkpoint salt.")
            art_id_len_bytes = f.read(1)
            if len(art_id_len_bytes) < 1:
                raise CheckpointError("Truncated artifact ID length.")
            art_id_len = art_id_len_bytes[0]
            if art_id_len == 0 or art_id_len > 64:
                raise CheckpointError(f"Invalid artifact ID length: {art_id_len}")
            art_id = f.read(art_id_len)
            if len(art_id) != art_id_len:
                raise CheckpointError("Truncated artifact ID.")
            key_id_len_bytes = f.read(1)
            if len(key_id_len_bytes) < 1:
                raise CheckpointError("Truncated key ID length.")
            key_id_len = key_id_len_bytes[0]
            key_id_bytes = f.read(key_id_len)
            if len(key_id_bytes) != key_id_len:
                raise CheckpointError("Truncated key ID.")
            key_id = key_id_bytes.decode("utf-8", errors="replace")
            wrapped_key_len_bytes = f.read(2)
            if len(wrapped_key_len_bytes) < 2:
                raise CheckpointError("Truncated wrapped key length.")
            wrapped_key_len = struct.unpack(">H", wrapped_key_len_bytes)[0]
            wrapped_key = f.read(wrapped_key_len)
            if len(wrapped_key) != wrapped_key_len:
                raise CheckpointError("Truncated wrapped key.")
            manifest_len_bytes = f.read(4)
            if len(manifest_len_bytes) < 4:
                raise CheckpointError("Truncated manifest length.")
            manifest_len = struct.unpack(">I", manifest_len_bytes)[0]
            if manifest_len <= 0 or manifest_len > lim.max_manifest_bytes:
                raise CheckpointError(f"Invalid manifest length: {manifest_len} (limit: {lim.max_manifest_bytes})")
            if f.tell() + manifest_len > file_size:
                raise CheckpointError(f"Truncated checkpoint: file size {file_size} is smaller than header and manifest ({f.tell() + manifest_len} bytes).")

            info: Dict[str, Any] = {
                "format": "TorchShield-SafeTrainingCheckpoint-v5",
                "version": ver,
                "file_size": file_size,
                "artifact_id": art_id.hex(),
                "key_id": key_id,
                "has_wrapped_key": wrapped_key_len > 0,
                "encrypted_manifest_len": manifest_len,
                "authenticated": False,
            }

            if key is not None or key_provider is not None:
                from torchshield.secure import AESGCMEnvelope, AuthenticationError
                from torchshield.checkpoint import _strict_json_loads
                resolved_key = key
                if resolved_key is None and key_provider is not None:
                    if wrapped_key_len == 0:
                        raise CheckpointError("Checkpoint header contains no wrapped key, but key was not provided.")
                    resolved_key = key_provider.unwrap_key(
                        wrapped_key,
                        encryption_context=encryption_context,
                    )
                if resolved_key is None or len(resolved_key) != 32:
                    raise CheckpointError("Resolved master key must be exactly 32 bytes.")

                manifest_key = SecureModelCheckpoint.derive_manifest_key_v5(resolved_key, salt, art_id)
                manifest_aad = art_id + b"::MANIFEST::v5"
                encrypted_manifest = f.read(manifest_len)
                if len(encrypted_manifest) != manifest_len:
                    raise CheckpointError("Truncated checkpoint manifest.")
                try:
                    manifest_bytes = AESGCMEnvelope.decrypt(encrypted_manifest, manifest_key, associated_data=manifest_aad)
                except AuthenticationError as exc:
                    raise CheckpointError("Checkpoint manifest authentication failed (wrong key or tampered header).") from exc

                manifest = _strict_json_loads(manifest_bytes)

                if expected_signer_public_key is not None:
                    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                    from cryptography.exceptions import InvalidSignature
                    sig_hex = manifest.get("signature")
                    if not sig_hex:
                        raise CheckpointError("Expected digital signature, but checkpoint manifest is unsigned.")
                    try:
                        Ed25519PublicKey.from_public_bytes(expected_signer_public_key).verify(
                            bytes.fromhex(sig_hex),
                            bytes.fromhex(manifest["manifest_digest"])
                        )
                    except InvalidSignature as exc:
                        raise CheckpointError("Checkpoint Ed25519 digital signature verification failed.") from exc

                # Check chunk bounds consistency against remaining file size
                expected_chunk_bytes = sum(c.get("ciphertext_len", 0) for c in manifest.get("chunks", []))
                if f.tell() + expected_chunk_bytes > file_size:
                    raise CheckpointError(
                        f"Checkpoint payload is truncated: manifest declares {expected_chunk_bytes} bytes in chunks, "
                        f"but only {file_size - f.tell()} bytes remain in file."
                    )

                training_meta = manifest.get("metadata", {})
                info.update({
                    "authenticated": True,
                    "global_step": training_meta.get("global_step"),
                    "epoch": training_meta.get("epoch"),
                    "batch_idx": training_meta.get("batch_idx"),
                    "loss": training_meta.get("loss"),
                    "learning_rate": training_meta.get("learning_rate"),
                    "config": training_meta.get("config", {}),
                    "total_tensors": manifest.get("total_tensors", 0),
                    "total_chunks": manifest.get("total_chunks", 0),
                    "total_raw_bytes": manifest.get("total_raw_bytes", 0),
                    "has_signature": bool(manifest.get("signature")),
                    "has_optimizer": training_meta.get("optimizer") is not None,
                    "has_scheduler": training_meta.get("scheduler") is not None,
                    "has_scaler": training_meta.get("scaler") is not None,
                    "has_rng": training_meta.get("rng") is not None,
                })

            return info

        elif magic == SecureModelCheckpoint.MAGIC_V4:
            ver_bytes = f.read(1)
            if len(ver_bytes) < 1:
                raise CheckpointError("Truncated checkpoint header.")
            ver = ver_bytes[0]
            if ver != SecureModelCheckpoint.VERSION_V4:
                raise CheckpointError(f"Unsupported checkpoint version: {ver}")
            salt = f.read(SecureModelCheckpoint.SALT_SIZE_V4)
            if len(salt) != SecureModelCheckpoint.SALT_SIZE_V4:
                raise CheckpointError("Truncated checkpoint salt.")
            art_id_len_bytes = f.read(1)
            if len(art_id_len_bytes) < 1:
                raise CheckpointError("Truncated artifact ID length.")
            art_id_len = art_id_len_bytes[0]
            if art_id_len == 0 or art_id_len > 64:
                raise CheckpointError(f"Invalid artifact ID length: {art_id_len}")
            art_id = f.read(art_id_len)
            if len(art_id) != art_id_len:
                raise CheckpointError("Truncated artifact ID.")
            key_id_len_bytes = f.read(1)
            if len(key_id_len_bytes) < 1:
                raise CheckpointError("Truncated key ID length.")
            key_id_len = key_id_len_bytes[0]
            key_id_bytes = f.read(key_id_len)
            if len(key_id_bytes) != key_id_len:
                raise CheckpointError("Truncated key ID.")
            key_id = key_id_bytes.decode("utf-8", errors="replace")
            manifest_len_bytes = f.read(4)
            if len(manifest_len_bytes) < 4:
                raise CheckpointError("Truncated manifest length.")
            manifest_len = struct.unpack(">I", manifest_len_bytes)[0]
            if manifest_len <= 0 or manifest_len > lim.max_manifest_bytes:
                raise CheckpointError(f"Invalid manifest length: {manifest_len} (limit: {lim.max_manifest_bytes})")
            if f.tell() + manifest_len > file_size:
                raise CheckpointError(f"Truncated checkpoint: file size {file_size} is smaller than header and manifest ({f.tell() + manifest_len} bytes).")

            info = {
                "format": "TorchShield-v4",
                "version": ver,
                "file_size": file_size,
                "artifact_id": art_id.hex(),
                "key_id": key_id,
                "encrypted_manifest_len": manifest_len,
                "authenticated": False,
            }

            if key is not None:
                from torchshield.secure import AESGCMEnvelope, AuthenticationError
                from torchshield.checkpoint import _strict_json_loads
                if len(key) != 32:
                    raise CheckpointError("Resolved master key must be exactly 32 bytes.")
                manifest_key = SecureModelCheckpoint.derive_manifest_key_v4(key, salt, art_id)
                manifest_aad = art_id + b"::MANIFEST::v4"
                encrypted_manifest = f.read(manifest_len)
                if len(encrypted_manifest) != manifest_len:
                    raise CheckpointError("Truncated checkpoint manifest.")
                try:
                    manifest_bytes = AESGCMEnvelope.decrypt(encrypted_manifest, manifest_key, associated_data=manifest_aad)
                except AuthenticationError as exc:
                    raise CheckpointError("Checkpoint manifest authentication failed (wrong key or tampered header).") from exc

                manifest = _strict_json_loads(manifest_bytes)
                info.update({
                    "authenticated": True,
                    "total_tensors": manifest.get("total_tensors", manifest.get("total_layers", 0)),
                    "total_chunks": manifest.get("total_chunks", 0),
                    "total_raw_bytes": manifest.get("total_raw_bytes", 0),
                })

            return info
        else:
            raise CheckpointError(f"Not a valid TorchShield secure model checkpoint (magic={magic!r}).")

