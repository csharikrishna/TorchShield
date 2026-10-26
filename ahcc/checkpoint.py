"""Production-grade authenticated checkpoint encryption for AI model weights.

Phase 2 Hardened Implementation:
- NIST SP 800-38D compliant AES-256-GCM.
- Per-artifact domain-separated key derivation via RFC 5869 HKDF-SHA256.
- Strict deterministic chunk nonces eliminating collision risks.
- Authenticated manifests cryptographically binding file identity, layer names,
  chunk sequence numbers, tensor shapes, dtypes, and manifest digest.
- Atomic writes (.tmp staging -> fsync -> os.replace) preventing corrupt states.
- Defensive tensor deserialization bounding memory allocation and dimensions.
- Native support for 0-dim scalars, empty tensors, BatchNorm buffers, and streaming.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import struct
import sys
import uuid
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ahcc.secure import AESGCMEnvelope, AuthenticationError, TensorAESGCMEnvelope
from ahcc.engine import AHCC_Engine, TensorCipherWrapper


class CheckpointError(AuthenticationError):
    """Raised when checkpoint manifest, chunk authentication, or validation fails."""


# Allow-listed PyTorch dtypes for strict tensor deserialization
ALLOWED_DTYPES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

# Defensive resource bounds preventing allocation denial-of-service
MAX_TENSOR_COUNT: int = 50_000
MAX_TENSOR_RANK: int = 8
MAX_DIM_SIZE: int = 2**31 - 1
MAX_MANIFEST_LEN: int = 10 * 1024 * 1024  # 10 MiB maximum manifest
MAX_TOTAL_RAW_BYTES: int = 16 * 1024 * 1024 * 1024  # 16 GiB default ceiling


class SecureModelCheckpoint:
    """End-to-end authenticated model checkpoint storage and streaming loader."""

    MAGIC_V1 = b"AHCCMOD1"
    MAGIC_V2 = b"AHCCMOD2"
    VERSION_V1 = 1
    VERSION_V2 = 2
    CURRENT_VERSION = VERSION_V2

    SALT_SIZE = 16
    NONCE_SIZE = 12

    @staticmethod
    def derive_artifact_key(master_key: bytes, salt: bytes, artifact_id: bytes) -> bytes:
        """Derive an isolated 256-bit AES-GCM subkey using HKDF-SHA256 (RFC 5869)."""
        if len(master_key) != 32:
            raise ValueError("master_key must be exactly 32 bytes.")
        if len(salt) != 16:
            raise ValueError("salt must be exactly 16 bytes.")
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"AHCC-SecureCheckpoint-v2-Key:" + artifact_id,
        )
        return hkdf.derive(master_key)

    @staticmethod
    def derive_chunk_nonce(chunk_seq: int) -> bytes:
        """Derive a deterministic 96-bit nonce from the sequential chunk index.
        
        NIST SP 800-38D requires nonce uniqueness under the same key.
        Chunk 0 is strictly reserved for the manifest. Payload chunks use seq >= 1.
        """
        if chunk_seq < 0 or chunk_seq >= 2**64:
            raise ValueError(f"Invalid chunk sequence number: {chunk_seq}")
        return struct.pack(">Q", chunk_seq) + struct.pack(">I", 0)

    @classmethod
    def save_checkpoint(
        cls,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        key: bytes,
        chunk_bytes: int = 4 * 1024 * 1024,  # 4 MiB chunks
        key_id: str = "",
    ) -> None:
        """Encrypt and serialize a model state dict with chunked authenticated envelopes.
        
        Performs atomic file writing via temporary staging and fsync.
        """
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dictionary of tensors.")
        if len(key) != 32:
            raise ValueError("AES-256 key must be exactly 32 bytes.")
        if len(state_dict) > MAX_TENSOR_COUNT:
            raise ValueError(f"Tensor count {len(state_dict)} exceeds maximum limit {MAX_TENSOR_COUNT}.")
        if chunk_bytes <= 0 or chunk_bytes > 1024 * 1024 * 1024:
            raise ValueError(f"Invalid chunk_bytes: {chunk_bytes}")

        # Generate fresh per-artifact cryptographic parameters
        salt = secrets.token_bytes(cls.SALT_SIZE)
        artifact_id = str(uuid.uuid4()).encode("utf-8")
        artifact_key = cls.derive_artifact_key(key, salt, artifact_id)

        manifest_entries: List[dict] = []
        chunks_data: List[bytes] = []
        global_chunk_seq = 1  # 0 is reserved for manifest
        total_raw_bytes = 0

        # Phase 1: Pre-process tensors and serialize raw bytes
        serialized_layers = []
        for layer_idx, (name, tensor) in enumerate(state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Item '{name}' must be a torch.Tensor.")
            if len(tensor.shape) > MAX_TENSOR_RANK:
                raise ValueError(f"Tensor '{name}' rank {len(tensor.shape)} exceeds maximum {MAX_TENSOR_RANK}.")

            contiguous = tensor.detach().to(device="cpu").contiguous()
            dtype_str = str(tensor.dtype).replace("torch.", "")
            if dtype_str not in ALLOWED_DTYPES:
                raise ValueError(f"Tensor '{name}' has unsupported dtype '{dtype_str}'.")

            shape_list = list(tensor.shape)
            for dim in shape_list:
                if dim < 0 or dim > MAX_DIM_SIZE:
                    raise ValueError(f"Invalid dimension size {dim} in tensor '{name}'.")

            if contiguous.numel() == 0:
                raw_bytes = b""
            else:
                raw_bytes = contiguous.clone().reshape(-1).view(torch.uint8).numpy().tobytes()

            total_raw = len(raw_bytes)
            total_raw_bytes += total_raw
            if total_raw_bytes > MAX_TOTAL_RAW_BYTES:
                raise ValueError(f"Total raw bytes {total_raw_bytes} exceeds limit {MAX_TOTAL_RAW_BYTES}.")

            serialized_layers.append((layer_idx, name, dtype_str, shape_list, raw_bytes, total_raw))

        # Phase 2: Build preliminary manifest to calculate manifest digest
        # Each chunk's AAD will bind the manifest digest to prevent cross-checkpoint substitution
        pre_manifest_dict = {
            "version": cls.VERSION_V2,
            "artifact_id": artifact_id.decode("utf-8"),
            "key_id": str(key_id),
            "total_layers": len(serialized_layers),
            "total_raw_bytes": total_raw_bytes,
            "layers": [
                {
                    "name": name,
                    "dtype": dtype_str,
                    "shape": shape_list,
                    "layer_idx": layer_idx,
                    "raw_len": total_raw,
                    "num_chunks": max(1, math.ceil(total_raw / chunk_bytes)) if total_raw > 0 else 1,
                }
                for layer_idx, name, dtype_str, shape_list, _, total_raw in serialized_layers
            ],
        }
        pre_manifest_bytes = json.dumps(pre_manifest_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        manifest_digest = hashlib.sha256(pre_manifest_bytes).digest()

        # Phase 3: Encrypt chunks with deterministic nonces and authenticated metadata
        for layer_idx, name, dtype_str, shape_list, raw_bytes, total_raw in serialized_layers:
            num_chunks = max(1, math.ceil(total_raw / chunk_bytes)) if total_raw > 0 else 1
            chunk_records = []

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_bytes
                end = min(start + chunk_bytes, total_raw)
                chunk_slice = raw_bytes[start:end]
                chunk_hash = hashlib.sha256(chunk_slice).hexdigest()

                # Deterministic nonce: chunk_seq >= 1
                chunk_nonce = cls.derive_chunk_nonce(global_chunk_seq)

                # AAD binds: artifact_id, manifest_digest, layer name, chunk_seq, chunk_idx, num_chunks, offset, shape
                aad = (
                    artifact_id
                    + b"::" + manifest_digest
                    + b"::layer=" + name.encode("utf-8")
                    + b"::chunk_seq=" + struct.pack(">Q", global_chunk_seq)
                    + b"::chunk_idx=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::offset=" + struct.pack(">Q", start)
                    + b"::shape=" + json.dumps(shape_list).encode("utf-8")
                )
                encrypted_chunk = AESGCMEnvelope._encrypt_with_explicit_nonce(
                    chunk_slice,
                    artifact_key,
                    nonce=chunk_nonce,
                    associated_data=aad,
                )
                chunk_records.append({
                    "chunk_seq": global_chunk_seq,
                    "chunk_idx": chunk_idx,
                    "offset": start,
                    "raw_len": len(chunk_slice),
                    "ciphertext_len": len(encrypted_chunk),
                    "sha256": chunk_hash,
                })
                chunks_data.append(encrypted_chunk)
                global_chunk_seq += 1

            manifest_entries.append({
                "name": name,
                "dtype": dtype_str,
                "shape": shape_list,
                "layer_idx": layer_idx,
                "raw_len": total_raw,
                "num_chunks": num_chunks,
                "chunks": chunk_records,
            })

        # Phase 4: Finalize authenticated manifest
        final_manifest_payload = json.dumps({
            "version": cls.VERSION_V2,
            "artifact_id": artifact_id.decode("utf-8"),
            "key_id": str(key_id),
            "manifest_digest": manifest_digest.hex(),
            "total_layers": len(manifest_entries),
            "total_chunks": global_chunk_seq - 1,
            "total_raw_bytes": total_raw_bytes,
            "layers": manifest_entries,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")

        manifest_nonce = cls.derive_chunk_nonce(0)  # Nonce 0 reserved for manifest
        manifest_aad = artifact_id + b"::MANIFEST::v2"
        encrypted_manifest = AESGCMEnvelope._encrypt_with_explicit_nonce(
            final_manifest_payload,
            artifact_key,
            nonce=manifest_nonce,
            associated_data=manifest_aad,
        )

        # Phase 5: Atomic file writing with temporary staging & fsync
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        staging_path = output_path + f".tmp.{uuid.uuid4().hex}"
        key_id_bytes = key_id.encode("utf-8")[:255]

        try:
            with open(staging_path, "wb") as f:
                # Layout:
                # MAGIC (8B) || VERSION (1B) || salt (16B) || artifact_id_len (1B) || artifact_id (36B)
                # || key_id_len (1B) || key_id || manifest_len (4B) || encrypted_manifest || chunks...
                f.write(cls.MAGIC_V2)
                f.write(bytes([cls.VERSION_V2]))
                f.write(salt)
                f.write(bytes([len(artifact_id)]))
                f.write(artifact_id)
                f.write(bytes([len(key_id_bytes)]))
                f.write(key_id_bytes)
                f.write(struct.pack(">I", len(encrypted_manifest)))
                f.write(encrypted_manifest)
                for chunk in chunks_data:
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename replacing destination
            os.replace(staging_path, output_path)
        except Exception:
            if os.path.exists(staging_path):
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
            raise

    @classmethod
    def load_checkpoint(
        cls,
        input_path: str,
        key: bytes,
        device: Union[str, torch.device] = "cpu",
        target_model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Authenticate manifest and stream-decrypt model layers.
        
        Supports both Version 2 (Hardened) and Version 1 (Legacy) formats.
        Fail-closed: raises CheckpointError on any authentication or integrity failure.
        """
        if len(key) != 32:
            raise ValueError("AES-256 key must be exactly 32 bytes.")

        with open(input_path, "rb") as f:
            magic = f.read(8)
            if magic == cls.MAGIC_V2:
                return cls._load_checkpoint_v2(f, key, device, target_model)
            elif magic == cls.MAGIC_V1:
                return cls._load_checkpoint_v1(f, key, device, target_model)
            else:
                raise CheckpointError(f"Not a valid AHCC secure model checkpoint (magic={magic!r}).")

    @classmethod
    def _load_checkpoint_v2(
        cls,
        f,
        master_key: bytes,
        device: Union[str, torch.device],
        target_model: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Load and authenticate a Version 2 hardened checkpoint."""
        ver_bytes = f.read(1)
        if len(ver_bytes) < 1:
            raise CheckpointError("Truncated checkpoint header.")
        ver = ver_bytes[0]
        if ver != cls.VERSION_V2:
            raise CheckpointError(f"Unsupported checkpoint version: {ver}")

        salt = f.read(cls.SALT_SIZE)
        if len(salt) != cls.SALT_SIZE:
            raise CheckpointError("Truncated checkpoint salt.")

        artifact_id_len_bytes = f.read(1)
        if len(artifact_id_len_bytes) < 1:
            raise CheckpointError("Truncated artifact ID length.")
        artifact_id_len = artifact_id_len_bytes[0]
        artifact_id = f.read(artifact_id_len)
        if len(artifact_id) != artifact_id_len:
            raise CheckpointError("Truncated artifact ID.")

        key_id_len_bytes = f.read(1)
        if len(key_id_len_bytes) < 1:
            raise CheckpointError("Truncated key ID length.")
        key_id_len = key_id_len_bytes[0]
        key_id = f.read(key_id_len)
        if len(key_id) != key_id_len:
            raise CheckpointError("Truncated key ID.")

        manifest_len_bytes = f.read(4)
        if len(manifest_len_bytes) < 4:
            raise CheckpointError("Truncated checkpoint manifest length.")
        manifest_len = struct.unpack(">I", manifest_len_bytes)[0]
        if manifest_len <= 0 or manifest_len > MAX_MANIFEST_LEN:
            raise CheckpointError(f"Invalid manifest length: {manifest_len} (limit: {MAX_MANIFEST_LEN})")

        encrypted_manifest = f.read(manifest_len)
        if len(encrypted_manifest) != manifest_len:
            raise CheckpointError("Truncated checkpoint manifest.")

        # Derive artifact subkey
        try:
            artifact_key = cls.derive_artifact_key(master_key, salt, artifact_id)
        except Exception as exc:
            raise CheckpointError("Failed to derive artifact key.") from exc

        # Authenticate and decrypt manifest
        manifest_aad = artifact_id + b"::MANIFEST::v2"
        try:
            manifest_bytes = AESGCMEnvelope.decrypt(encrypted_manifest, artifact_key, associated_data=manifest_aad)
        except AuthenticationError as exc:
            raise CheckpointError("Checkpoint manifest authentication failed (wrong key or tampered header).") from exc

        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            raise CheckpointError("Corrupted manifest JSON payload.") from exc

        # Validate manifest invariants
        if manifest.get("version") != cls.VERSION_V2:
            raise CheckpointError(f"Manifest declared version mismatch: {manifest.get('version')}")
        if manifest.get("artifact_id") != artifact_id.decode("utf-8"):
            raise CheckpointError("Manifest artifact_id does not match container header.")

        manifest_digest_hex = manifest.get("manifest_digest")
        if not manifest_digest_hex:
            raise CheckpointError("Missing manifest digest in authenticated manifest.")
        manifest_digest = bytes.fromhex(manifest_digest_hex)

        declared_total_layers = manifest.get("total_layers", 0)
        layers = manifest.get("layers", [])
        if len(layers) != declared_total_layers:
            raise CheckpointError(f"Layer count mismatch: declared {declared_total_layers}, found {len(layers)}.")
        if len(layers) > MAX_TENSOR_COUNT:
            raise CheckpointError(f"Layer count {len(layers)} exceeds maximum limit {MAX_TENSOR_COUNT}.")

        expected_chunk_seq = 1
        restored_state: Dict[str, torch.Tensor] = {}

        # Stream and authenticate layers and chunks
        for layer in layers:
            name = layer["name"]
            dtype_str = layer["dtype"]
            if dtype_str not in ALLOWED_DTYPES:
                raise CheckpointError(f"Unsupported dtype '{dtype_str}' in layer '{name}'.")
            dtype = ALLOWED_DTYPES[dtype_str]

            shape = tuple(layer["shape"])
            if len(shape) > MAX_TENSOR_RANK:
                raise CheckpointError(f"Shape rank {len(shape)} exceeds maximum {MAX_TENSOR_RANK}.")
            expected_numel = 1
            for dim in shape:
                if dim < 0 or dim > MAX_DIM_SIZE:
                    raise CheckpointError(f"Invalid dimension size {dim} in layer '{name}'.")
                expected_numel *= dim
                if expected_numel > MAX_DIM_SIZE:
                    raise CheckpointError(f"Arithmetic overflow in tensor dimension calculation for '{name}'.")

            num_chunks = layer["num_chunks"]
            total_raw = layer["raw_len"]
            chunk_records = layer["chunks"]
            if len(chunk_records) != num_chunks:
                raise CheckpointError(f"Chunk count mismatch for '{name}': declared {num_chunks}, found {len(chunk_records)}.")

            # Optimized preallocated layer buffer: writes chunk slices directly into place,
            # avoiding b''.join() reallocations, temporary chunk lists, and intermediate buffer clones.
            layer_buf = bytearray(total_raw) if total_raw > 0 else None

            for chunk_info in chunk_records:
                chunk_seq = chunk_info["chunk_seq"]
                chunk_idx = chunk_info["chunk_idx"]
                offset = chunk_info["offset"]
                expected_raw_len = chunk_info["raw_len"]
                c_len = chunk_info["ciphertext_len"]
                expected_sha256 = chunk_info.get("sha256")

                if chunk_seq != expected_chunk_seq:
                    raise CheckpointError(
                        f"Chunk sequence anomaly: expected {expected_chunk_seq}, got {chunk_seq}. "
                        "Possible reordered, missing, or injected chunk attack."
                    )

                c_encrypted = f.read(c_len)
                if len(c_encrypted) != c_len:
                    raise CheckpointError(f"Truncated chunk {chunk_seq} in layer '{name}'.")

                # Verify AAD binding
                aad = (
                    artifact_id
                    + b"::" + manifest_digest
                    + b"::layer=" + name.encode("utf-8")
                    + b"::chunk_seq=" + struct.pack(">Q", chunk_seq)
                    + b"::chunk_idx=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::offset=" + struct.pack(">Q", offset)
                    + b"::shape=" + json.dumps(list(shape)).encode("utf-8")
                )
                try:
                    decrypted_slice = AESGCMEnvelope.decrypt(c_encrypted, artifact_key, associated_data=aad)
                except AuthenticationError as exc:
                    raise CheckpointError(f"Authentication failed on chunk {chunk_seq} for layer '{name}'.") from exc

                if len(decrypted_slice) != expected_raw_len:
                    raise CheckpointError(f"Decrypted length mismatch in chunk {chunk_seq}: {len(decrypted_slice)} != {expected_raw_len}")

                if expected_sha256 and hashlib.sha256(decrypted_slice).hexdigest() != expected_sha256:
                    raise CheckpointError(f"Digest mismatch on chunk {chunk_seq} in layer '{name}'.")

                if layer_buf is not None:
                    layer_buf[offset : offset + len(decrypted_slice)] = decrypted_slice
                expected_chunk_seq += 1

            if total_raw > 0:
                assert layer_buf is not None
                tensor = torch.frombuffer(layer_buf, dtype=dtype).reshape(shape)
            else:
                tensor = torch.empty(shape, dtype=dtype)

            # Accelerated device placement with pinned memory when streaming to CUDA
            if device != "cpu" and torch.cuda.is_available() and str(device).startswith("cuda"):
                tensor = tensor.pin_memory().to(device, non_blocking=True)
            else:
                tensor = tensor.to(device)

            tensor = tensor.to(device)
            restored_state[name] = tensor

            if target_model is not None:
                try:
                    curr_param = target_model.get_parameter(name)
                    curr_param.data.copy_(tensor)
                except (AttributeError, KeyError):
                    try:
                        curr_buf = target_model.get_buffer(name)
                        curr_buf.copy_(tensor)
                    except (AttributeError, KeyError):
                        pass

        # Check for unexpected trailing data
        trailing = f.read(1)
        if len(trailing) > 0:
            raise CheckpointError("Extraneous unauthenticated trailing bytes detected in checkpoint file.")

        return restored_state

    @classmethod
    def _load_checkpoint_v1(
        cls,
        f,
        key: bytes,
        device: Union[str, torch.device],
        target_model: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Backward-compatible loader for Version 1 checkpoints."""
        ver_bytes = f.read(1)
        if len(ver_bytes) < 1 or ver_bytes[0] != cls.VERSION_V1:
            raise CheckpointError("Invalid Version 1 checkpoint header.")

        file_id_len_bytes = f.read(1)
        if len(file_id_len_bytes) < 1:
            raise CheckpointError("Truncated v1 file_id length.")
        file_id_len = file_id_len_bytes[0]
        file_id = f.read(file_id_len)
        if len(file_id) != file_id_len:
            raise CheckpointError("Truncated v1 file_id.")

        manifest_len_bytes = f.read(4)
        if len(manifest_len_bytes) < 4:
            raise CheckpointError("Truncated v1 manifest length.")
        manifest_len = struct.unpack(">I", manifest_len_bytes)[0]
        encrypted_manifest = f.read(manifest_len)
        if len(encrypted_manifest) != manifest_len:
            raise CheckpointError("Truncated v1 manifest.")

        manifest_aad = file_id + b"::MANIFEST"
        try:
            manifest_bytes = AESGCMEnvelope.decrypt(encrypted_manifest, key, associated_data=manifest_aad)
        except AuthenticationError as exc:
            raise CheckpointError("V1 manifest authentication failed.") from exc

        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if manifest.get("file_id") != file_id.decode("utf-8"):
            raise CheckpointError("V1 manifest file_id mismatch.")

        restored_state = {}
        for layer in manifest["layers"]:
            name = layer["name"]
            dtype_str = layer["dtype"]
            shape = tuple(layer["shape"])
            num_chunks = layer["num_chunks"]
            total_raw = layer["raw_len"]
            dtype = getattr(torch, dtype_str)
            chunks_raw = []

            for chunk_info in layer["chunks"]:
                chunk_idx = chunk_info["chunk_index"]
                c_len = chunk_info["length"]
                c_encrypted = f.read(c_len)
                if len(c_encrypted) != c_len:
                    raise CheckpointError(f"Truncated v1 chunk in layer '{name}'.")

                aad = (
                    file_id
                    + b"::" + name.encode("utf-8")
                    + b"::chunk=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::shape=" + json.dumps(list(shape)).encode("utf-8")
                )
                decrypted_slice = AESGCMEnvelope.decrypt(c_encrypted, key, associated_data=aad)
                chunks_raw.append(decrypted_slice)

            layer_bytes = b"".join(chunks_raw)
            if len(layer_bytes) != total_raw:
                raise CheckpointError(f"Corrupted v1 layer length for '{name}'.")

            if total_raw > 0:
                tensor = torch.frombuffer(bytearray(layer_bytes), dtype=dtype).clone().reshape(shape)
            else:
                tensor = torch.empty(shape, dtype=dtype)

            tensor = tensor.to(device)
            restored_state[name] = tensor

            if target_model is not None:
                try:
                    curr_param = target_model.get_parameter(name)
                    curr_param.data.copy_(tensor)
                except (AttributeError, KeyError):
                    try:
                        curr_buf = target_model.get_buffer(name)
                        curr_buf.copy_(tensor)
                    except (AttributeError, KeyError):
                        pass

        return restored_state
