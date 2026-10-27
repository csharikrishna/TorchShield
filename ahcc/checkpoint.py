"""Production-grade authenticated checkpoint encryption for AI model weights.

Phase 4 Hardened Implementation:
- NIST SP 800-38D compliant AES-256-GCM.
- RFC 5869 HKDF-SHA256 domain separation: dedicated manifest key and chunk key.
- Strict deterministic monotonic chunk nonces (b"CHNK" + uint64_be) eliminating collision risks.
- Dedicated manifest nonce domain (b"MANI" + uint64_be) under manifest key.
- Authenticated manifests cryptographically binding file identity, layer names,
  chunk sequence numbers, tensor shapes, dtypes, and manifest digest.
- Canonical JSON specification hash verification preventing circular dependencies.
- Configurable ResourceLimits preventing allocation denial-of-service.
- Atomic writes (.tmp staging -> flush -> fsync -> os.replace) preventing corrupt states.
- Defensive tensor deserialization bounding memory allocation and dimensions.
- Native support for 0-dim scalars, empty tensors, BatchNorm buffers, and streaming.
- Pinned-memory CUDA streaming and preallocated layer reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import secrets
import struct
import sys
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ahcc.secure import AESGCMEnvelope, AuthenticationError, TensorAESGCMEnvelope
from ahcc.engine import AHCC_Engine, TensorCipherWrapper
from ahcc.key_providers.base import KeyProvider


class CheckpointError(AuthenticationError):
    """Raised when checkpoint manifest, chunk authentication, or validation fails."""


__all__ = [
    "SecureModelCheckpoint",
    "ChunkCryptoSession",
    "_ChunkCryptoSession",
    "CheckpointError",
    "ResourceLimits",
    "DEFAULT_LIMITS",
    "ALLOWED_DTYPES",
    "DTYPE_ITEMSIZE",
]


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

DTYPE_ITEMSIZE: Dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float64": 8,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


@dataclass(frozen=True)
class ResourceLimits:
    """Configurable resource ceilings preventing allocation denial-of-service."""
    max_file_size: int = 128 * 1024 * 1024 * 1024        # 128 GiB max file size
    max_total_raw_bytes: int = 64 * 1024 * 1024 * 1024   # 64 GiB max aggregate decoded tensor bytes
    max_tensor_bytes: int = 16 * 1024 * 1024 * 1024      # 16 GiB max single tensor bytes
    max_manifest_bytes: int = 16 * 1024 * 1024           # 16 MiB max manifest length
    max_tensor_count: int = 50_000                       # 50,000 tensors max
    max_tensor_rank: int = 8                             # rank <= 8
    max_chunk_count: int = 1_000_000                     # 1,000,000 chunks max
    max_chunk_bytes: int = 64 * 1024 * 1024              # 64 MiB max single chunk
    max_dim_size: int = 100_000_000                      # 100,000,000 elements per dim
    max_total_elements: int = 16 * 1024 * 1024 * 1024    # 16G aggregate elements


DEFAULT_LIMITS = ResourceLimits()

# Legacy constants for compatibility
MAX_TENSOR_COUNT: int = DEFAULT_LIMITS.max_tensor_count
MAX_TENSOR_RANK: int = DEFAULT_LIMITS.max_tensor_rank
MAX_DIM_SIZE: int = DEFAULT_LIMITS.max_dim_size
MAX_MANIFEST_LEN: int = DEFAULT_LIMITS.max_manifest_bytes
MAX_TOTAL_RAW_BYTES: int = DEFAULT_LIMITS.max_total_raw_bytes


def _canonical_json(obj: Any) -> bytes:
    """Deterministic UTF-8 canonical JSON serialization with sorted keys and minimal whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _strict_json_loads(data: bytes) -> dict:
    """Strict JSON parser rejecting duplicate keys, malformed syntax, and non-object roots."""
    def _no_duplicates(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise CheckpointError(f"Duplicate key in manifest JSON: {k!r}")
            d[k] = v
        return d
    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError("Malformed manifest JSON payload.") from exc
    if not isinstance(parsed, dict):
        raise CheckpointError("Manifest root must be a JSON object.")
    return parsed


class ChunkCryptoSession:
    """Controlled single-pass chunk encryptor enforcing monotonic nonces and domain separation.
    
    NIST SP 800-38D requires that no two distinct invocations of AES-GCM under the same key
    use the same nonce. Chunk nonces are 96-bit (12 bytes):
        prefix (4 bytes: b"CHNK") || sequence_number (8 bytes: unsigned 64-bit int BE).
    The sequence number increments monotonically from 0 to total_chunks - 1.
    """
    NONCE_PREFIX = b"CHNK"

    def __init__(self, chunk_key: bytes, artifact_id: bytes, max_chunks: int = 1_000_000):
        if len(chunk_key) != 32:
            raise ValueError("chunk_key must be exactly 32 bytes.")
        self._key = chunk_key
        self._artifact_id = artifact_id
        self._max_chunks = max_chunks
        self._seq = 0
        self._sealed = False
        self._cipher = AESGCM(self._key)

    @property
    def current_seq(self) -> int:
        return self._seq

    def encrypt_next_chunk(self, chunk_slice: bytes, aad: bytes) -> Tuple[int, bytes]:
        if self._sealed:
            raise CheckpointError("Crypto session is sealed and cannot encrypt additional chunks.")
        if self._seq >= self._max_chunks:
            raise CheckpointError(f"Maximum chunk count {self._max_chunks} exceeded.")
        if self._seq >= 2**64 - 1:
            raise CheckpointError("Chunk sequence counter overflow.")

        seq = self._seq
        nonce = self.NONCE_PREFIX + struct.pack(">Q", seq)
        self._seq += 1

        authenticated_header = AESGCMEnvelope.HEADER + nonce + aad
        ciphertext = self._cipher.encrypt(nonce, chunk_slice, authenticated_header)
        envelope = AESGCMEnvelope.HEADER + nonce + ciphertext
        return seq, envelope

    def seal(self) -> int:
        self._sealed = True
        return self._seq


# Backward compatibility alias
_ChunkCryptoSession = ChunkCryptoSession


class SecureModelCheckpoint:
    """End-to-end authenticated model checkpoint storage and streaming loader."""

    MAGIC_V1 = b"AHCCMOD1"
    MAGIC_V2 = b"AHCCMOD2"
    MAGIC_V4 = b"AHCCMOD4"
    MAGIC_V5 = b"AHCCMOD5"
    VERSION_V1 = 1
    VERSION_V2 = 2
    VERSION_V4 = 4
    VERSION_V5 = 5
    CURRENT_VERSION = VERSION_V5

    SALT_SIZE_V5 = 32
    SALT_SIZE_V4 = 32
    SALT_SIZE_V2 = 16
    SALT_SIZE = SALT_SIZE_V5  # Current default
    NONCE_SIZE = 12

    @staticmethod
    def derive_artifact_key(master_key: bytes, salt: bytes, artifact_id: bytes) -> bytes:
        """Derive an isolated 256-bit AES-GCM subkey using HKDF-SHA256 (RFC 5869) for Version 2."""
        if len(master_key) != 32:
            raise ValueError("master_key must be exactly 32 bytes.")
        if len(salt) != 16:
            raise ValueError("salt must be exactly 16 bytes for Version 2.")
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"AHCC-SecureCheckpoint-v2-Key:" + artifact_id,
        )
        return hkdf.derive(master_key)

    @staticmethod
    def derive_manifest_key_v4(master_key: bytes, salt: bytes, artifact_id: bytes) -> bytes:
        """Derive isolated 256-bit manifest encryption key using HKDF-SHA256 (RFC 5869)."""
        master_key_bytes = bytes(master_key)
        salt_bytes = bytes(salt)
        artifact_id_bytes = bytes(artifact_id)
        if len(master_key_bytes) != 32:
            raise ValueError("master_key must be exactly 32 bytes.")
        if len(salt_bytes) != 32:
            raise ValueError("salt must be exactly 32 bytes for Version 4.")
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            info=b"AHCC-v4-Manifest-Key::" + artifact_id_bytes,
        ).derive(master_key_bytes)

    @staticmethod
    def derive_chunk_key_v4(master_key: bytes, salt: bytes, artifact_id: bytes) -> bytes:
        """Derive isolated 256-bit chunk encryption key using HKDF-SHA256 (RFC 5869)."""
        master_key_bytes = bytes(master_key)
        salt_bytes = bytes(salt)
        artifact_id_bytes = bytes(artifact_id)
        if len(master_key_bytes) != 32:
            raise ValueError("master_key must be exactly 32 bytes.")
        if len(salt_bytes) != 32:
            raise ValueError("salt must be exactly 32 bytes for Version 4.")
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            info=b"AHCC-v4-Chunk-Key::" + artifact_id_bytes,
        ).derive(master_key_bytes)

    @staticmethod
    def derive_chunk_nonce(chunk_seq: int) -> bytes:
        """Derive a deterministic 96-bit nonce from sequential chunk index (V2 legacy)."""
        if chunk_seq < 0 or chunk_seq >= 2**64:
            raise ValueError(f"Invalid chunk sequence number: {chunk_seq}")
        return struct.pack(">Q", chunk_seq) + struct.pack(">I", 0)

    @staticmethod
    def derive_chunk_nonce_v4(chunk_seq: int) -> bytes:
        """Derive deterministic 96-bit chunk nonce for Version 4: b'CHNK' + uint64_be."""
        if chunk_seq < 0 or chunk_seq >= 2**64 - 1:
            raise ValueError(f"Invalid chunk sequence number: {chunk_seq}")
        return b"CHNK" + struct.pack(">Q", chunk_seq)

    @staticmethod
    def derive_manifest_key_v5(master_key: bytes, salt: bytes, artifact_id: bytes) -> bytes:
        """Derive isolated 256-bit manifest encryption key using HKDF-SHA256 (RFC 5869) for Version 5."""
        master_key_bytes = bytes(master_key)
        salt_bytes = bytes(salt)
        artifact_id_bytes = bytes(artifact_id)
        if len(master_key_bytes) != 32:
            raise ValueError("master_key must be exactly 32 bytes.")
        if len(salt_bytes) != 32:
            raise ValueError("salt must be exactly 32 bytes for Version 5.")
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            info=b"AHCC-v5-Manifest-Key::" + artifact_id_bytes,
        ).derive(master_key_bytes)

    @staticmethod
    def derive_chunk_key_v5(master_key: bytes, salt: bytes, artifact_id: bytes) -> bytes:
        """Derive isolated 256-bit chunk encryption key using HKDF-SHA256 (RFC 5869) for Version 5."""
        master_key_bytes = bytes(master_key)
        salt_bytes = bytes(salt)
        artifact_id_bytes = bytes(artifact_id)
        if len(master_key_bytes) != 32:
            raise ValueError("master_key must be exactly 32 bytes.")
        if len(salt_bytes) != 32:
            raise ValueError("salt must be exactly 32 bytes for Version 5.")
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            info=b"AHCC-v5-Chunk-Key::" + artifact_id_bytes,
        ).derive(master_key_bytes)

    @staticmethod
    def derive_chunk_nonce_v5(chunk_seq: int) -> bytes:
        """Derive deterministic 96-bit chunk nonce for Version 5: b'CHNK' + uint64_be."""
        if chunk_seq < 0 or chunk_seq >= 2**64 - 1:
            raise ValueError(f"Invalid chunk sequence number: {chunk_seq}")
        return b"CHNK" + struct.pack(">Q", chunk_seq)

    @classmethod
    def save_checkpoint(
        cls,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        key: Optional[bytes] = None,
        chunk_bytes: int = 4 * 1024 * 1024,  # 4 MiB chunks
        key_id: str = "",
        format_version: int = VERSION_V4,
        limits: Optional[ResourceLimits] = None,
        metadata: Optional[Dict[str, Any]] = None,
        key_provider: Optional[KeyProvider] = None,
        encryption_context: Optional[Dict[str, str]] = None,
        signer_private_key: Optional[bytes] = None,
    ) -> None:
        """Encrypt and serialize a model state dict with chunked authenticated envelopes.
        
        Performs atomic file writing via temporary staging and fsync.
        Supports Version 4 and Version 5 (KMS envelope, metadata, and Ed25519 signatures).
        """
        limits = limits or DEFAULT_LIMITS

        if (
            format_version == cls.VERSION_V5
            or metadata is not None
            or key_provider is not None
            or signer_private_key is not None
        ):
            format_version = cls.VERSION_V5

        if format_version == cls.VERSION_V5:
            wrapped_key = b""
            if key_provider is not None:
                if not key_id:
                    key_id = key_provider.get_key_id()
                if key is None:
                    resolved_key, wrapped_key = key_provider.generate_data_key(
                        encryption_context=encryption_context,
                    )
                else:
                    wrapped_key = key_provider.wrap_key(
                        key,
                        encryption_context=encryption_context,
                    )
                    resolved_key = key
            else:
                if key is None:
                    raise ValueError("Encryption key must be provided when key_provider is None.")
                resolved_key = key

            cls._save_checkpoint_v5(
                state_dict=state_dict,
                output_path=output_path,
                key=resolved_key,
                chunk_bytes=chunk_bytes,
                key_id=key_id,
                wrapped_key=wrapped_key,
                metadata=metadata,
                signer_private_key=signer_private_key,
                limits=limits,
            )
        elif format_version == cls.VERSION_V4:
            if key is None:
                raise ValueError("AES-256 key must be provided for Version 4 save.")
            cls._save_checkpoint_v4(state_dict, output_path, key, chunk_bytes, key_id, limits)
        elif format_version == cls.VERSION_V2:
            if key is None:
                raise ValueError("AES-256 key must be provided for Version 2 save.")
            cls._save_checkpoint_v2(state_dict, output_path, key, chunk_bytes, key_id, limits)
        else:
            raise ValueError(f"Unsupported format version for saving: {format_version}")

    @classmethod
    def _save_checkpoint_v5(
        cls,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        key: bytes,
        chunk_bytes: int,
        key_id: str,
        wrapped_key: bytes,
        metadata: Optional[Dict[str, Any]],
        signer_private_key: Optional[bytes],
        limits: ResourceLimits,
    ) -> None:
        """Version 5 save implementation supporting metadata, envelope key wrapping, and Ed25519 signatures."""
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dictionary of tensors.")
        if len(key) != 32:
            raise ValueError("AES-256 key must be exactly 32 bytes.")
        if len(state_dict) > limits.max_tensor_count:
            raise ValueError(f"Tensor count {len(state_dict)} exceeds maximum limit {limits.max_tensor_count}.")
        if chunk_bytes <= 0 or chunk_bytes > limits.max_chunk_bytes:
            raise ValueError(f"Invalid chunk_bytes: {chunk_bytes} (limit: {limits.max_chunk_bytes})")
        if len(wrapped_key) > 65535:
            raise ValueError("wrapped_key exceeds maximum length of 65535 bytes.")

        salt = secrets.token_bytes(cls.SALT_SIZE_V5)
        artifact_id = secrets.token_bytes(16)
        manifest_key = cls.derive_manifest_key_v5(key, salt, artifact_id)
        chunk_key = cls.derive_chunk_key_v5(key, salt, artifact_id)

        # Phase 1: Pre-process tensors and validate all dimension bounds
        serialized_layers = []
        total_raw_bytes = 0
        total_chunks = 0

        for layer_idx, (name, tensor) in enumerate(state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Item '{name}' must be a torch.Tensor.")
            if len(tensor.shape) > limits.max_tensor_rank:
                raise ValueError(f"Tensor '{name}' rank {len(tensor.shape)} exceeds maximum {limits.max_tensor_rank}.")

            contiguous = tensor.detach().to(device="cpu").contiguous()
            dtype_str = str(tensor.dtype).replace("torch.", "")
            if dtype_str not in ALLOWED_DTYPES:
                raise ValueError(f"Tensor '{name}' has unsupported dtype '{dtype_str}'.")

            itemsize = DTYPE_ITEMSIZE[dtype_str]
            shape_list = list(tensor.shape)
            numel = 1
            for dim in shape_list:
                if dim < 0 or dim > limits.max_dim_size:
                    raise ValueError(f"Invalid dimension size {dim} in tensor '{name}' (limit: {limits.max_dim_size}).")
                numel *= dim
                if numel > limits.max_total_elements:
                    raise ValueError(f"Tensor '{name}' element count exceeds maximum {limits.max_total_elements}.")

            expected_raw_len = numel * itemsize
            if expected_raw_len > limits.max_tensor_bytes:
                raise ValueError(f"Tensor '{name}' raw bytes {expected_raw_len} exceeds limit {limits.max_tensor_bytes}.")

            if contiguous.numel() == 0:
                raw_bytes = b""
            else:
                raw_bytes = contiguous.clone().reshape(-1).view(torch.uint8).numpy().tobytes()

            if len(raw_bytes) != expected_raw_len:
                raise ValueError(f"Serialized tensor bytes mismatch for '{name}': {len(raw_bytes)} != {expected_raw_len}")

            total_raw = len(raw_bytes)
            total_raw_bytes += total_raw
            if total_raw_bytes > limits.max_total_raw_bytes:
                raise ValueError(f"Total raw bytes {total_raw_bytes} exceeds limit {limits.max_total_raw_bytes}.")

            num_chunks = max(1, math.ceil(total_raw / chunk_bytes)) if total_raw > 0 else 1
            start_seq = total_chunks
            total_chunks += num_chunks
            if total_chunks > limits.max_chunk_count:
                raise ValueError(f"Total chunks {total_chunks} exceeds limit {limits.max_chunk_count}.")

            serialized_layers.append((layer_idx, name, dtype_str, shape_list, raw_bytes, total_raw, num_chunks, start_seq))

        # Phase 2: Construct Canonical Manifest Specification (M_spec)
        clean_metadata = metadata if metadata is not None else {}
        m_spec = {
            "version": cls.VERSION_V5,
            "artifact_id": artifact_id.hex(),
            "key_id": str(key_id),
            "total_tensors": len(serialized_layers),
            "total_chunks": total_chunks,
            "total_raw_bytes": total_raw_bytes,
            "metadata": clean_metadata,
            "tensors": [
                {
                    "name": name,
                    "dtype": dtype_str,
                    "shape": shape_list,
                    "raw_len": total_raw,
                    "num_chunks": num_chunks,
                    "start_chunk_seq": start_seq,
                }
                for (_, name, dtype_str, shape_list, _, total_raw, num_chunks, start_seq) in serialized_layers
            ],
        }
        canonical_spec_bytes = _canonical_json(m_spec)
        manifest_digest = hashlib.sha256(canonical_spec_bytes).digest()

        # Phase 3: Encrypt chunks using controlled crypto session
        session = _ChunkCryptoSession(chunk_key, artifact_id, max_chunks=limits.max_chunk_count)
        chunks_data: List[bytes] = []
        chunk_records: List[dict] = []

        for layer_idx, name, dtype_str, shape_list, raw_bytes, total_raw, num_chunks, _ in serialized_layers:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_bytes
                end = min(start + chunk_bytes, total_raw)
                chunk_slice = raw_bytes[start:end]
                chunk_hash = hashlib.sha256(chunk_slice).hexdigest()

                seq = session.current_seq
                aad = (
                    artifact_id
                    + b"::" + manifest_digest
                    + b"::tensor=" + name.encode("utf-8")
                    + b"::seq=" + struct.pack(">Q", seq)
                    + b"::idx=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::offset=" + struct.pack(">Q", start)
                    + b"::raw_len=" + struct.pack(">Q", len(chunk_slice))
                )
                chunk_seq, encrypted_chunk = session.encrypt_next_chunk(chunk_slice, aad)
                chunks_data.append(encrypted_chunk)
                chunk_records.append({
                    "chunk_seq": chunk_seq,
                    "tensor": name,
                    "chunk_idx": chunk_idx,
                    "offset": start,
                    "raw_len": len(chunk_slice),
                    "ciphertext_len": len(encrypted_chunk),
                    "sha256": chunk_hash,
                })

        session.seal()

        # Phase 4: Construct and Encrypt Final Manifest
        final_manifest = dict(m_spec)
        final_manifest["manifest_digest"] = manifest_digest.hex()
        if signer_private_key is not None:
            priv_key = Ed25519PrivateKey.from_private_bytes(signer_private_key)
            sig = priv_key.sign(manifest_digest)
            pub_bytes = priv_key.public_key().public_bytes_raw()
            final_manifest["signature"] = sig.hex()
            final_manifest["signer_pubkey"] = pub_bytes.hex()

        final_manifest["chunks"] = chunk_records

        final_manifest_bytes = _canonical_json(final_manifest)
        if len(final_manifest_bytes) > limits.max_manifest_bytes:
            raise ValueError(f"Manifest size {len(final_manifest_bytes)} exceeds limit {limits.max_manifest_bytes}.")

        manifest_nonce = b"MANI\x00\x00\x00\x00\x00\x00\x00\x00"
        manifest_aad = artifact_id + b"::MANIFEST::v5"
        auth_header = AESGCMEnvelope.HEADER + manifest_nonce + manifest_aad
        encrypted_manifest_payload = AESGCM(manifest_key).encrypt(manifest_nonce, final_manifest_bytes, auth_header)
        encrypted_manifest = AESGCMEnvelope.HEADER + manifest_nonce + encrypted_manifest_payload

        # Phase 5: Atomic Staging & Replacement
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        staging_path = output_path + f".tmp.{uuid.uuid4().hex}"
        key_id_bytes = key_id.encode("utf-8")[:255]

        try:
            with open(staging_path, "wb") as f:
                f.write(cls.MAGIC_V5)
                f.write(bytes([cls.VERSION_V5]))
                f.write(salt)
                f.write(bytes([len(artifact_id)]))
                f.write(artifact_id)
                f.write(bytes([len(key_id_bytes)]))
                f.write(key_id_bytes)
                f.write(struct.pack(">H", len(wrapped_key)))
                if len(wrapped_key) > 0:
                    f.write(wrapped_key)
                f.write(struct.pack(">I", len(encrypted_manifest)))
                f.write(encrypted_manifest)
                for chunk in chunks_data:
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())

            os.replace(staging_path, output_path)
        except Exception:
            if os.path.exists(staging_path):
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
            raise

    @classmethod
    def _save_checkpoint_v4(
        cls,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        key: bytes,
        chunk_bytes: int,
        key_id: str,
        limits: ResourceLimits,
    ) -> None:
        """Version 4 save implementation with domain-separated keys and canonical manifest specification."""
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dictionary of tensors.")
        if len(key) != 32:
            raise ValueError("AES-256 key must be exactly 32 bytes.")
        if len(state_dict) > limits.max_tensor_count:
            raise ValueError(f"Tensor count {len(state_dict)} exceeds maximum limit {limits.max_tensor_count}.")
        if chunk_bytes <= 0 or chunk_bytes > limits.max_chunk_bytes:
            raise ValueError(f"Invalid chunk_bytes: {chunk_bytes} (limit: {limits.max_chunk_bytes})")

        salt = secrets.token_bytes(cls.SALT_SIZE_V4)
        artifact_id = secrets.token_bytes(16)  # 16-byte raw UUID / unique ID
        manifest_key = cls.derive_manifest_key_v4(key, salt, artifact_id)
        chunk_key = cls.derive_chunk_key_v4(key, salt, artifact_id)

        # Phase 1: Pre-process tensors and validate all dimension bounds
        serialized_layers = []
        total_raw_bytes = 0
        total_chunks = 0

        for layer_idx, (name, tensor) in enumerate(state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Item '{name}' must be a torch.Tensor.")
            if len(tensor.shape) > limits.max_tensor_rank:
                raise ValueError(f"Tensor '{name}' rank {len(tensor.shape)} exceeds maximum {limits.max_tensor_rank}.")

            contiguous = tensor.detach().to(device="cpu").contiguous()
            dtype_str = str(tensor.dtype).replace("torch.", "")
            if dtype_str not in ALLOWED_DTYPES:
                raise ValueError(f"Tensor '{name}' has unsupported dtype '{dtype_str}'.")

            itemsize = DTYPE_ITEMSIZE[dtype_str]
            shape_list = list(tensor.shape)
            numel = 1
            for dim in shape_list:
                if dim < 0 or dim > limits.max_dim_size:
                    raise ValueError(f"Invalid dimension size {dim} in tensor '{name}' (limit: {limits.max_dim_size}).")
                numel *= dim
                if numel > limits.max_total_elements:
                    raise ValueError(f"Tensor '{name}' element count exceeds maximum {limits.max_total_elements}.")

            expected_raw_len = numel * itemsize
            if expected_raw_len > limits.max_tensor_bytes:
                raise ValueError(f"Tensor '{name}' raw bytes {expected_raw_len} exceeds limit {limits.max_tensor_bytes}.")

            if contiguous.numel() == 0:
                raw_bytes = b""
            else:
                raw_bytes = contiguous.clone().reshape(-1).view(torch.uint8).numpy().tobytes()

            if len(raw_bytes) != expected_raw_len:
                raise ValueError(f"Serialized tensor bytes mismatch for '{name}': {len(raw_bytes)} != {expected_raw_len}")

            total_raw = len(raw_bytes)
            total_raw_bytes += total_raw
            if total_raw_bytes > limits.max_total_raw_bytes:
                raise ValueError(f"Total raw bytes {total_raw_bytes} exceeds limit {limits.max_total_raw_bytes}.")

            num_chunks = max(1, math.ceil(total_raw / chunk_bytes)) if total_raw > 0 else 1
            start_seq = total_chunks
            total_chunks += num_chunks
            if total_chunks > limits.max_chunk_count:
                raise ValueError(f"Total chunks {total_chunks} exceeds limit {limits.max_chunk_count}.")

            serialized_layers.append((layer_idx, name, dtype_str, shape_list, raw_bytes, total_raw, num_chunks, start_seq))

        # Phase 2: Construct Canonical Manifest Specification (M_spec)
        m_spec = {
            "version": cls.VERSION_V4,
            "artifact_id": artifact_id.hex(),
            "key_id": str(key_id),
            "total_tensors": len(serialized_layers),
            "total_chunks": total_chunks,
            "total_raw_bytes": total_raw_bytes,
            "tensors": [
                {
                    "name": name,
                    "dtype": dtype_str,
                    "shape": shape_list,
                    "raw_len": total_raw,
                    "num_chunks": num_chunks,
                    "start_chunk_seq": start_seq,
                }
                for (_, name, dtype_str, shape_list, _, total_raw, num_chunks, start_seq) in serialized_layers
            ],
        }
        canonical_spec_bytes = _canonical_json(m_spec)
        manifest_digest = hashlib.sha256(canonical_spec_bytes).digest()

        # Phase 3: Encrypt chunks using controlled crypto session
        session = _ChunkCryptoSession(chunk_key, artifact_id, max_chunks=limits.max_chunk_count)
        chunks_data: List[bytes] = []
        chunk_records: List[dict] = []

        for layer_idx, name, dtype_str, shape_list, raw_bytes, total_raw, num_chunks, _ in serialized_layers:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_bytes
                end = min(start + chunk_bytes, total_raw)
                chunk_slice = raw_bytes[start:end]
                chunk_hash = hashlib.sha256(chunk_slice).hexdigest()

                seq = session.current_seq
                aad = (
                    artifact_id
                    + b"::" + manifest_digest
                    + b"::tensor=" + name.encode("utf-8")
                    + b"::seq=" + struct.pack(">Q", seq)
                    + b"::idx=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::offset=" + struct.pack(">Q", start)
                    + b"::raw_len=" + struct.pack(">Q", len(chunk_slice))
                )
                chunk_seq, encrypted_chunk = session.encrypt_next_chunk(chunk_slice, aad)
                chunks_data.append(encrypted_chunk)
                chunk_records.append({
                    "chunk_seq": chunk_seq,
                    "tensor": name,
                    "chunk_idx": chunk_idx,
                    "offset": start,
                    "raw_len": len(chunk_slice),
                    "ciphertext_len": len(encrypted_chunk),
                    "sha256": chunk_hash,
                })

        session.seal()

        # Phase 4: Construct and Encrypt Final Manifest
        final_manifest = dict(m_spec)
        final_manifest["manifest_digest"] = manifest_digest.hex()
        final_manifest["chunks"] = chunk_records

        final_manifest_bytes = _canonical_json(final_manifest)
        if len(final_manifest_bytes) > limits.max_manifest_bytes:
            raise ValueError(f"Manifest size {len(final_manifest_bytes)} exceeds limit {limits.max_manifest_bytes}.")

        manifest_nonce = b"MANI\x00\x00\x00\x00\x00\x00\x00\x00"
        manifest_aad = artifact_id + b"::MANIFEST::v4"
        auth_header = AESGCMEnvelope.HEADER + manifest_nonce + manifest_aad
        encrypted_manifest_payload = AESGCM(manifest_key).encrypt(manifest_nonce, final_manifest_bytes, auth_header)
        encrypted_manifest = AESGCMEnvelope.HEADER + manifest_nonce + encrypted_manifest_payload

        # Phase 5: Atomic Staging & Replacement
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        staging_path = output_path + f".tmp.{uuid.uuid4().hex}"
        key_id_bytes = key_id.encode("utf-8")[:255]

        try:
            with open(staging_path, "wb") as f:
                f.write(cls.MAGIC_V4)
                f.write(bytes([cls.VERSION_V4]))
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

            os.replace(staging_path, output_path)
        except Exception:
            if os.path.exists(staging_path):
                try:
                    os.remove(staging_path)
                except OSError:
                    pass
            raise

    @classmethod
    def _save_checkpoint_v2(
        cls,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        key: bytes,
        chunk_bytes: int,
        key_id: str,
        limits: ResourceLimits,
    ) -> None:
        """Version 2 legacy save implementation preserved for format backward-compatibility."""
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dictionary of tensors.")
        if len(key) != 32:
            raise ValueError("AES-256 key must be exactly 32 bytes.")
        if len(state_dict) > limits.max_tensor_count:
            raise ValueError(f"Tensor count {len(state_dict)} exceeds maximum limit {limits.max_tensor_count}.")
        if chunk_bytes <= 0 or chunk_bytes > limits.max_chunk_bytes:
            raise ValueError(f"Invalid chunk_bytes: {chunk_bytes}")

        salt = secrets.token_bytes(cls.SALT_SIZE_V2)
        artifact_id = str(uuid.uuid4()).encode("utf-8")
        artifact_key = cls.derive_artifact_key(key, salt, artifact_id)

        manifest_entries: List[dict] = []
        chunks_data: List[bytes] = []
        global_chunk_seq = 1
        total_raw_bytes = 0

        serialized_layers = []
        for layer_idx, (name, tensor) in enumerate(state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Item '{name}' must be a torch.Tensor.")
            if len(tensor.shape) > limits.max_tensor_rank:
                raise ValueError(f"Tensor '{name}' rank {len(tensor.shape)} exceeds maximum {limits.max_tensor_rank}.")

            contiguous = tensor.detach().to(device="cpu").contiguous()
            dtype_str = str(tensor.dtype).replace("torch.", "")
            if dtype_str not in ALLOWED_DTYPES:
                raise ValueError(f"Tensor '{name}' has unsupported dtype '{dtype_str}'.")

            shape_list = list(tensor.shape)
            for dim in shape_list:
                if dim < 0 or dim > limits.max_dim_size:
                    raise ValueError(f"Invalid dimension size {dim} in tensor '{name}'.")

            if contiguous.numel() == 0:
                raw_bytes = b""
            else:
                raw_bytes = contiguous.clone().reshape(-1).view(torch.uint8).numpy().tobytes()

            total_raw = len(raw_bytes)
            total_raw_bytes += total_raw
            if total_raw_bytes > limits.max_total_raw_bytes:
                raise ValueError(f"Total raw bytes {total_raw_bytes} exceeds limit {limits.max_total_raw_bytes}.")

            serialized_layers.append((layer_idx, name, dtype_str, shape_list, raw_bytes, total_raw))

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

        for layer_idx, name, dtype_str, shape_list, raw_bytes, total_raw in serialized_layers:
            num_chunks = max(1, math.ceil(total_raw / chunk_bytes)) if total_raw > 0 else 1
            chunk_records = []

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_bytes
                end = min(start + chunk_bytes, total_raw)
                chunk_slice = raw_bytes[start:end]
                chunk_hash = hashlib.sha256(chunk_slice).hexdigest()

                chunk_nonce = cls.derive_chunk_nonce(global_chunk_seq)
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

        manifest_nonce = cls.derive_chunk_nonce(0)
        manifest_aad = artifact_id + b"::MANIFEST::v2"
        encrypted_manifest = AESGCMEnvelope._encrypt_with_explicit_nonce(
            final_manifest_payload,
            artifact_key,
            nonce=manifest_nonce,
            associated_data=manifest_aad,
        )

        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        staging_path = output_path + f".tmp.{uuid.uuid4().hex}"
        key_id_bytes = key_id.encode("utf-8")[:255]

        try:
            with open(staging_path, "wb") as f:
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
        key: Optional[bytes] = None,
        device: Union[str, torch.device] = "cpu",
        target_model: Optional[nn.Module] = None,
        limits: Optional[ResourceLimits] = None,
        return_metadata: bool = False,
        key_provider: Optional[KeyProvider] = None,
        encryption_context: Optional[Dict[str, str]] = None,
        expected_signer_public_key: Optional[bytes] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], dict]]:
        """Authenticate manifest and stream-decrypt model layers.
        
        Supports Version 5 (KMS envelope + metadata + Ed25519 provenance),
        Version 4 (Hardened), Version 2 (Legacy), and Version 1 formats.
        Fail-closed: raises CheckpointError on any authentication or integrity failure.
        """
        limits = limits or DEFAULT_LIMITS
        if key is not None and len(key) != 32:
            raise ValueError("AES-256 key must be exactly 32 bytes.")

        with open(input_path, "rb") as f:
            # Check file size bound if seekable
            try:
                cur = f.tell()
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                f.seek(cur, os.SEEK_SET)
                if file_size > limits.max_file_size:
                    raise CheckpointError(f"File size {file_size} exceeds limit {limits.max_file_size}.")
            except (AttributeError, OSError):
                pass

            magic = f.read(8)
            if magic == cls.MAGIC_V5:
                return cls._load_checkpoint_v5(
                    f=f,
                    master_key=key,
                    device=device,
                    target_model=target_model,
                    limits=limits,
                    return_metadata=return_metadata,
                    key_provider=key_provider,
                    encryption_context=encryption_context,
                    expected_signer_public_key=expected_signer_public_key,
                )
            elif magic == cls.MAGIC_V4:
                if key is None:
                    raise ValueError("key must be provided for Version 4 checkpoint loading.")
                tensors = cls._load_checkpoint_v4(f, key, device, target_model, limits)
                return (tensors, {}) if return_metadata else tensors
            elif magic == cls.MAGIC_V2:
                if key is None:
                    raise ValueError("key must be provided for Version 2 checkpoint loading.")
                tensors = cls._load_checkpoint_v2(f, key, device, target_model, limits)
                return (tensors, {}) if return_metadata else tensors
            elif magic == cls.MAGIC_V1:
                if key is None:
                    raise ValueError("key must be provided for Version 1 checkpoint loading.")
                tensors = cls._load_checkpoint_v1(f, key, device, target_model, limits)
                return (tensors, {}) if return_metadata else tensors
            else:
                raise CheckpointError(f"Not a valid AHCC secure model checkpoint (magic={magic!r}).")

    @classmethod
    def _load_checkpoint_v5(
        cls,
        f,
        master_key: Optional[bytes],
        device: Union[str, torch.device],
        target_model: Optional[nn.Module],
        limits: ResourceLimits,
        return_metadata: bool = False,
        key_provider: Optional[KeyProvider] = None,
        encryption_context: Optional[Dict[str, str]] = None,
        expected_signer_public_key: Optional[bytes] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], dict]]:
        """Load and authenticate a Version 5 hardened checkpoint."""
        ver_bytes = f.read(1)
        if len(ver_bytes) < 1:
            raise CheckpointError("Truncated checkpoint header.")
        ver = ver_bytes[0]
        if ver != cls.VERSION_V5:
            raise CheckpointError(f"Unsupported checkpoint version: {ver}")

        salt = f.read(cls.SALT_SIZE_V5)
        if len(salt) != cls.SALT_SIZE_V5:
            raise CheckpointError("Truncated checkpoint salt.")

        art_id_len_bytes = f.read(1)
        if len(art_id_len_bytes) < 1:
            raise CheckpointError("Truncated artifact ID length.")
        art_id_len = art_id_len_bytes[0]
        if art_id_len == 0 or art_id_len > 64:
            raise CheckpointError(f"Invalid artifact ID length: {art_id_len}")
        artifact_id = f.read(art_id_len)
        if len(artifact_id) != art_id_len:
            raise CheckpointError("Truncated artifact ID.")

        key_id_len_bytes = f.read(1)
        if len(key_id_len_bytes) < 1:
            raise CheckpointError("Truncated key ID length.")
        key_id_len = key_id_len_bytes[0]
        key_id = f.read(key_id_len)
        if len(key_id) != key_id_len:
            raise CheckpointError("Truncated key ID.")
        key_id_str = key_id.decode("utf-8", errors="replace")

        wrapped_key_len_bytes = f.read(2)
        if len(wrapped_key_len_bytes) < 2:
            raise CheckpointError("Truncated wrapped key length.")
        wrapped_key_len = struct.unpack(">H", wrapped_key_len_bytes)[0]
        wrapped_key = f.read(wrapped_key_len)
        if len(wrapped_key) != wrapped_key_len:
            raise CheckpointError("Truncated wrapped key.")

        manifest_len_bytes = f.read(4)
        if len(manifest_len_bytes) < 4:
            raise CheckpointError("Truncated checkpoint manifest length.")
        manifest_len = struct.unpack(">I", manifest_len_bytes)[0]
        if manifest_len <= 0 or manifest_len > limits.max_manifest_bytes:
            raise CheckpointError(f"Invalid manifest length: {manifest_len} (limit: {limits.max_manifest_bytes})")

        encrypted_manifest = f.read(manifest_len)
        if len(encrypted_manifest) != manifest_len:
            raise CheckpointError("Truncated checkpoint manifest.")

        # Resolve DEK
        resolved_key = master_key
        if resolved_key is None:
            if key_provider is None:
                raise CheckpointError("Checkpoint uses KMS envelope encryption; key_provider or key must be provided.")
            if wrapped_key_len == 0:
                raise CheckpointError("Checkpoint header contains no wrapped key, but key was not provided.")
            try:
                resolved_key = key_provider.unwrap_key(
                    wrapped_key,
                    encryption_context=encryption_context,
                )
            except Exception as exc:
                raise CheckpointError(f"Failed to unwrap DEK via KeyProvider: {exc}") from exc

        if len(resolved_key) != 32:
            raise CheckpointError("Resolved master key must be exactly 32 bytes.")

        # Derive independent keys
        try:
            manifest_key = cls.derive_manifest_key_v5(resolved_key, salt, artifact_id)
            chunk_key = cls.derive_chunk_key_v5(resolved_key, salt, artifact_id)
        except Exception as exc:
            raise CheckpointError("Failed to derive checkpoint keys.") from exc

        # Authenticate and decrypt manifest
        manifest_aad = artifact_id + b"::MANIFEST::v5"
        try:
            manifest_bytes = AESGCMEnvelope.decrypt(encrypted_manifest, manifest_key, associated_data=manifest_aad)
        except AuthenticationError as exc:
            raise CheckpointError("Checkpoint manifest authentication failed (wrong key or tampered header).") from exc

        manifest = _strict_json_loads(manifest_bytes)

        # Validate manifest invariants
        if manifest.get("version") != cls.VERSION_V5:
            raise CheckpointError(f"Manifest declared version mismatch: {manifest.get('version')}")
        if manifest.get("artifact_id") != artifact_id.hex():
            raise CheckpointError("Manifest artifact_id does not match container header.")

        # Reconstruct Manifest Specification to verify digest
        manifest_digest_hex = manifest.get("manifest_digest")
        if not manifest_digest_hex:
            raise CheckpointError("Missing manifest digest in authenticated manifest.")
        manifest_digest = bytes.fromhex(manifest_digest_hex)

        declared_total_tensors = manifest.get("total_tensors", 0)
        tensors = manifest.get("tensors", [])
        if len(tensors) != declared_total_tensors:
            raise CheckpointError(f"Tensor count mismatch: declared {declared_total_tensors}, found {len(tensors)}.")
        if len(tensors) > limits.max_tensor_count:
            raise CheckpointError(f"Tensor count {len(tensors)} exceeds maximum limit {limits.max_tensor_count}.")

        declared_total_raw = manifest.get("total_raw_bytes", 0)
        if declared_total_raw > limits.max_total_raw_bytes:
            raise CheckpointError(f"Declared raw bytes {declared_total_raw} exceeds limit {limits.max_total_raw_bytes}.")

        declared_total_chunks = manifest.get("total_chunks", 0)
        if declared_total_chunks > limits.max_chunk_count:
            raise CheckpointError(f"Declared chunks {declared_total_chunks} exceeds limit {limits.max_chunk_count}.")

        reconstructed_spec = {
            "version": manifest["version"],
            "artifact_id": manifest["artifact_id"],
            "key_id": manifest["key_id"],
            "total_tensors": declared_total_tensors,
            "total_chunks": declared_total_chunks,
            "total_raw_bytes": declared_total_raw,
            "metadata": manifest.get("metadata", {}),
            "tensors": tensors,
        }
        computed_digest = hashlib.sha256(_canonical_json(reconstructed_spec)).digest()
        if not secrets.compare_digest(computed_digest, manifest_digest):
            raise CheckpointError("Manifest specification digest verification failed.")

        # Verify signature if expected or present
        if expected_signer_public_key is not None:
            if "signature" not in manifest or not manifest["signature"]:
                raise CheckpointError("Checkpoint manifest is missing required digital signature.")
            sig_bytes = bytes.fromhex(manifest["signature"])
            try:
                pub_key = Ed25519PublicKey.from_public_bytes(expected_signer_public_key)
                pub_key.verify(sig_bytes, manifest_digest)
            except Exception as exc:
                raise CheckpointError(f"Checkpoint digital signature verification failed: {exc}") from exc

        # Read and decrypt chunks
        manifest_chunks = manifest.get("chunks", [])
        if len(manifest_chunks) != declared_total_chunks:
            raise CheckpointError(f"Chunk list count {len(manifest_chunks)} does not match total_chunks {declared_total_chunks}.")

        chunk_iter = iter(manifest_chunks)
        expected_chunk_seq = 0
        restored_state: Dict[str, torch.Tensor] = {}

        for tensor_info in tensors:
            name = tensor_info["name"]
            dtype_str = tensor_info["dtype"]
            if dtype_str not in ALLOWED_DTYPES:
                raise CheckpointError(f"Unsupported dtype '{dtype_str}' in tensor '{name}'.")
            dtype = ALLOWED_DTYPES[dtype_str]
            itemsize = DTYPE_ITEMSIZE[dtype_str]

            shape = tuple(tensor_info["shape"])
            if len(shape) > limits.max_tensor_rank:
                raise CheckpointError(f"Shape rank {len(shape)} exceeds maximum {limits.max_tensor_rank}.")

            expected_numel = 1
            for dim in shape:
                if dim < 0 or dim > limits.max_dim_size:
                    raise CheckpointError(f"Invalid dimension size {dim} in tensor '{name}' (limit: {limits.max_dim_size}).")
                expected_numel *= dim
                if expected_numel > limits.max_total_elements:
                    raise CheckpointError(f"Element count overflow in tensor '{name}'.")

            expected_bytes = expected_numel * itemsize
            if expected_bytes > limits.max_tensor_bytes:
                raise CheckpointError(f"Tensor '{name}' raw bytes {expected_bytes} exceeds limit {limits.max_tensor_bytes}.")

            total_raw = tensor_info["raw_len"]
            if total_raw != expected_bytes:
                raise CheckpointError(f"Declared raw_len {total_raw} != expected {expected_bytes} for tensor '{name}'.")

            num_chunks = tensor_info["num_chunks"]
            layer_buf = bytearray(total_raw) if total_raw > 0 else None

            for chunk_idx in range(num_chunks):
                try:
                    chunk_info = next(chunk_iter)
                except StopIteration:
                    raise CheckpointError(f"Premature end of chunk records for tensor '{name}'.")

                chunk_seq = chunk_info["chunk_seq"]
                if chunk_seq != expected_chunk_seq:
                    raise CheckpointError(
                        f"Chunk sequence anomaly: expected {expected_chunk_seq}, got {chunk_seq}. "
                        "Possible reordered, missing, or injected chunk attack."
                    )

                if chunk_info["tensor"] != name or chunk_info["chunk_idx"] != chunk_idx:
                    raise CheckpointError(f"Chunk metadata mismatch for tensor '{name}' at index {chunk_idx}.")

                offset = chunk_info["offset"]
                expected_raw_len = chunk_info["raw_len"]
                c_len = chunk_info["ciphertext_len"]
                expected_sha256 = chunk_info.get("sha256")

                if c_len > limits.max_chunk_bytes + AESGCMEnvelope.MIN_ENVELOPE_SIZE:
                    raise CheckpointError(f"Ciphertext length {c_len} exceeds maximum chunk limit.")

                c_encrypted = f.read(c_len)
                if len(c_encrypted) != c_len:
                    raise CheckpointError(f"Truncated chunk {chunk_seq} in tensor '{name}'.")

                aad = (
                    artifact_id
                    + b"::" + manifest_digest
                    + b"::tensor=" + name.encode("utf-8")
                    + b"::seq=" + struct.pack(">Q", chunk_seq)
                    + b"::idx=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::offset=" + struct.pack(">Q", offset)
                    + b"::raw_len=" + struct.pack(">Q", expected_raw_len)
                )

                try:
                    decrypted_slice = AESGCMEnvelope.decrypt(c_encrypted, chunk_key, associated_data=aad)
                except AuthenticationError as exc:
                    raise CheckpointError(f"Authentication failed on chunk {chunk_seq} for tensor '{name}'.") from exc

                if len(decrypted_slice) != expected_raw_len:
                    raise CheckpointError(f"Decrypted length mismatch in chunk {chunk_seq}: {len(decrypted_slice)} != {expected_raw_len}")

                if expected_sha256 and hashlib.sha256(decrypted_slice).hexdigest() != expected_sha256:
                    raise CheckpointError(f"Digest mismatch on chunk {chunk_seq} in tensor '{name}'.")

                if layer_buf is not None:
                    layer_buf[offset : offset + len(decrypted_slice)] = decrypted_slice
                expected_chunk_seq += 1

            if total_raw > 0:
                assert layer_buf is not None
                tensor = torch.frombuffer(layer_buf, dtype=dtype).reshape(shape)
            else:
                tensor = torch.empty(shape, dtype=dtype)

            if device != "cpu" and torch.cuda.is_available() and str(device).startswith("cuda"):
                tensor = tensor.pin_memory().to(device, non_blocking=True)
            else:
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

        trailing = f.read(1)
        if len(trailing) > 0:
            raise CheckpointError("Extraneous unauthenticated trailing bytes detected in checkpoint file.")

        if return_metadata:
            return restored_state, manifest.get("metadata", {})
        return restored_state

    @classmethod
    def _load_checkpoint_v4(
        cls,
        f,
        master_key: bytes,
        device: Union[str, torch.device],
        target_model: Optional[nn.Module],
        limits: ResourceLimits,
    ) -> Dict[str, torch.Tensor]:
        """Load and authenticate a Version 4 hardened checkpoint."""
        ver_bytes = f.read(1)
        if len(ver_bytes) < 1:
            raise CheckpointError("Truncated checkpoint header.")
        ver = ver_bytes[0]
        if ver != cls.VERSION_V4:
            raise CheckpointError(f"Unsupported checkpoint version: {ver}")

        salt = f.read(cls.SALT_SIZE_V4)
        if len(salt) != cls.SALT_SIZE_V4:
            raise CheckpointError("Truncated checkpoint salt.")

        art_id_len_bytes = f.read(1)
        if len(art_id_len_bytes) < 1:
            raise CheckpointError("Truncated artifact ID length.")
        art_id_len = art_id_len_bytes[0]
        if art_id_len == 0 or art_id_len > 64:
            raise CheckpointError(f"Invalid artifact ID length: {art_id_len}")
        artifact_id = f.read(art_id_len)
        if len(artifact_id) != art_id_len:
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
        if manifest_len <= 0 or manifest_len > limits.max_manifest_bytes:
            raise CheckpointError(f"Invalid manifest length: {manifest_len} (limit: {limits.max_manifest_bytes})")

        encrypted_manifest = f.read(manifest_len)
        if len(encrypted_manifest) != manifest_len:
            raise CheckpointError("Truncated checkpoint manifest.")

        # Derive independent keys
        try:
            manifest_key = cls.derive_manifest_key_v4(master_key, salt, artifact_id)
            chunk_key = cls.derive_chunk_key_v4(master_key, salt, artifact_id)
        except Exception as exc:
            raise CheckpointError("Failed to derive checkpoint keys.") from exc

        # Authenticate and decrypt manifest
        manifest_aad = artifact_id + b"::MANIFEST::v4"
        try:
            manifest_bytes = AESGCMEnvelope.decrypt(encrypted_manifest, manifest_key, associated_data=manifest_aad)
        except AuthenticationError as exc:
            raise CheckpointError("Checkpoint manifest authentication failed (wrong key or tampered header).") from exc

        manifest = _strict_json_loads(manifest_bytes)

        # Validate manifest header invariants
        if manifest.get("version") != cls.VERSION_V4:
            raise CheckpointError(f"Manifest declared version mismatch: {manifest.get('version')}")
        if manifest.get("artifact_id") != artifact_id.hex():
            raise CheckpointError("Manifest artifact_id does not match container header.")

        # Reconstruct Manifest Specification to verify digest
        manifest_digest_hex = manifest.get("manifest_digest")
        if not manifest_digest_hex:
            raise CheckpointError("Missing manifest digest in authenticated manifest.")
        manifest_digest = bytes.fromhex(manifest_digest_hex)

        declared_total_tensors = manifest.get("total_tensors", 0)
        tensors = manifest.get("tensors", [])
        if len(tensors) != declared_total_tensors:
            raise CheckpointError(f"Tensor count mismatch: declared {declared_total_tensors}, found {len(tensors)}.")
        if len(tensors) > limits.max_tensor_count:
            raise CheckpointError(f"Tensor count {len(tensors)} exceeds maximum limit {limits.max_tensor_count}.")

        declared_total_raw = manifest.get("total_raw_bytes", 0)
        if declared_total_raw > limits.max_total_raw_bytes:
            raise CheckpointError(f"Declared raw bytes {declared_total_raw} exceeds limit {limits.max_total_raw_bytes}.")

        declared_total_chunks = manifest.get("total_chunks", 0)
        if declared_total_chunks > limits.max_chunk_count:
            raise CheckpointError(f"Declared chunks {declared_total_chunks} exceeds limit {limits.max_chunk_count}.")

        reconstructed_spec = {
            "version": manifest["version"],
            "artifact_id": manifest["artifact_id"],
            "key_id": manifest["key_id"],
            "total_tensors": declared_total_tensors,
            "total_chunks": declared_total_chunks,
            "total_raw_bytes": declared_total_raw,
            "tensors": tensors,
        }
        computed_digest = hashlib.sha256(_canonical_json(reconstructed_spec)).digest()
        if not secrets.compare_digest(computed_digest, manifest_digest):
            raise CheckpointError("Manifest specification digest verification failed.")

        # Build chunk lookup map from manifest
        manifest_chunks = manifest.get("chunks", [])
        if len(manifest_chunks) != declared_total_chunks:
            raise CheckpointError(f"Chunk list count {len(manifest_chunks)} does not match total_chunks {declared_total_chunks}.")

        chunk_iter = iter(manifest_chunks)
        expected_chunk_seq = 0
        restored_state: Dict[str, torch.Tensor] = {}

        for tensor_info in tensors:
            name = tensor_info["name"]
            dtype_str = tensor_info["dtype"]
            if dtype_str not in ALLOWED_DTYPES:
                raise CheckpointError(f"Unsupported dtype '{dtype_str}' in tensor '{name}'.")
            dtype = ALLOWED_DTYPES[dtype_str]
            itemsize = DTYPE_ITEMSIZE[dtype_str]

            shape = tuple(tensor_info["shape"])
            if len(shape) > limits.max_tensor_rank:
                raise CheckpointError(f"Shape rank {len(shape)} exceeds maximum {limits.max_tensor_rank}.")

            expected_numel = 1
            for dim in shape:
                if dim < 0 or dim > limits.max_dim_size:
                    raise CheckpointError(f"Invalid dimension size {dim} in tensor '{name}' (limit: {limits.max_dim_size}).")
                expected_numel *= dim
                if expected_numel > limits.max_total_elements:
                    raise CheckpointError(f"Element count overflow in tensor '{name}'.")

            expected_bytes = expected_numel * itemsize
            if expected_bytes > limits.max_tensor_bytes:
                raise CheckpointError(f"Tensor '{name}' raw bytes {expected_bytes} exceeds limit {limits.max_tensor_bytes}.")

            total_raw = tensor_info["raw_len"]
            if total_raw != expected_bytes:
                raise CheckpointError(f"Declared raw_len {total_raw} != expected {expected_bytes} for tensor '{name}'.")

            num_chunks = tensor_info["num_chunks"]
            layer_buf = bytearray(total_raw) if total_raw > 0 else None

            for chunk_idx in range(num_chunks):
                try:
                    chunk_info = next(chunk_iter)
                except StopIteration:
                    raise CheckpointError(f"Premature end of chunk records for tensor '{name}'.")

                chunk_seq = chunk_info["chunk_seq"]
                if chunk_seq != expected_chunk_seq:
                    raise CheckpointError(
                        f"Chunk sequence anomaly: expected {expected_chunk_seq}, got {chunk_seq}. "
                        "Possible reordered, missing, or injected chunk attack."
                    )

                if chunk_info["tensor"] != name or chunk_info["chunk_idx"] != chunk_idx:
                    raise CheckpointError(f"Chunk metadata mismatch for tensor '{name}' at index {chunk_idx}.")

                offset = chunk_info["offset"]
                expected_raw_len = chunk_info["raw_len"]
                c_len = chunk_info["ciphertext_len"]
                expected_sha256 = chunk_info.get("sha256")

                if c_len > limits.max_chunk_bytes + AESGCMEnvelope.MIN_ENVELOPE_SIZE:
                    raise CheckpointError(f"Ciphertext length {c_len} exceeds maximum chunk limit.")

                c_encrypted = f.read(c_len)
                if len(c_encrypted) != c_len:
                    raise CheckpointError(f"Truncated chunk {chunk_seq} in tensor '{name}'.")

                aad = (
                    artifact_id
                    + b"::" + manifest_digest
                    + b"::tensor=" + name.encode("utf-8")
                    + b"::seq=" + struct.pack(">Q", chunk_seq)
                    + b"::idx=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::offset=" + struct.pack(">Q", offset)
                    + b"::raw_len=" + struct.pack(">Q", expected_raw_len)
                )

                try:
                    decrypted_slice = AESGCMEnvelope.decrypt(c_encrypted, chunk_key, associated_data=aad)
                except AuthenticationError as exc:
                    raise CheckpointError(f"Authentication failed on chunk {chunk_seq} for tensor '{name}'.") from exc

                if len(decrypted_slice) != expected_raw_len:
                    raise CheckpointError(f"Decrypted length mismatch in chunk {chunk_seq}: {len(decrypted_slice)} != {expected_raw_len}")

                if expected_sha256 and hashlib.sha256(decrypted_slice).hexdigest() != expected_sha256:
                    raise CheckpointError(f"Digest mismatch on chunk {chunk_seq} in tensor '{name}'.")

                if layer_buf is not None:
                    layer_buf[offset : offset + len(decrypted_slice)] = decrypted_slice
                expected_chunk_seq += 1

            if total_raw > 0:
                assert layer_buf is not None
                tensor = torch.frombuffer(layer_buf, dtype=dtype).reshape(shape)
            else:
                tensor = torch.empty(shape, dtype=dtype)

            if device != "cpu" and torch.cuda.is_available() and str(device).startswith("cuda"):
                tensor = tensor.pin_memory().to(device, non_blocking=True)
            else:
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

        trailing = f.read(1)
        if len(trailing) > 0:
            raise CheckpointError("Extraneous unauthenticated trailing bytes detected in checkpoint file.")

        return restored_state

    @classmethod
    def _load_checkpoint_v2(
        cls,
        f,
        master_key: bytes,
        device: Union[str, torch.device],
        target_model: Optional[nn.Module],
        limits: Optional[ResourceLimits] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load and authenticate a Version 2 hardened checkpoint."""
        limits = limits or DEFAULT_LIMITS
        ver_bytes = f.read(1)
        if len(ver_bytes) < 1:
            raise CheckpointError("Truncated checkpoint header.")
        ver = ver_bytes[0]
        if ver != cls.VERSION_V2:
            raise CheckpointError(f"Unsupported checkpoint version: {ver}")

        salt = f.read(cls.SALT_SIZE_V2)
        if len(salt) != cls.SALT_SIZE_V2:
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
        if manifest_len <= 0 or manifest_len > limits.max_manifest_bytes:
            raise CheckpointError(f"Invalid manifest length: {manifest_len} (limit: {limits.max_manifest_bytes})")

        encrypted_manifest = f.read(manifest_len)
        if len(encrypted_manifest) != manifest_len:
            raise CheckpointError("Truncated checkpoint manifest.")

        try:
            artifact_key = cls.derive_artifact_key(master_key, salt, artifact_id)
        except Exception as exc:
            raise CheckpointError("Failed to derive artifact key.") from exc

        manifest_aad = artifact_id + b"::MANIFEST::v2"
        try:
            manifest_bytes = AESGCMEnvelope.decrypt(encrypted_manifest, artifact_key, associated_data=manifest_aad)
        except AuthenticationError as exc:
            raise CheckpointError("Checkpoint manifest authentication failed (wrong key or tampered header).") from exc

        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            raise CheckpointError("Corrupted manifest JSON payload.") from exc

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
        if len(layers) > limits.max_tensor_count:
            raise CheckpointError(f"Layer count {len(layers)} exceeds maximum limit {limits.max_tensor_count}.")

        expected_chunk_seq = 1
        restored_state: Dict[str, torch.Tensor] = {}

        for layer in layers:
            name = layer["name"]
            dtype_str = layer["dtype"]
            if dtype_str not in ALLOWED_DTYPES:
                raise CheckpointError(f"Unsupported dtype '{dtype_str}' in layer '{name}'.")
            dtype = ALLOWED_DTYPES[dtype_str]

            shape = tuple(layer["shape"])
            if len(shape) > limits.max_tensor_rank:
                raise CheckpointError(f"Shape rank {len(shape)} exceeds maximum {limits.max_tensor_rank}.")
            expected_numel = 1
            for dim in shape:
                if dim < 0 or dim > 2**31 - 1:
                    raise CheckpointError(f"Invalid dimension size {dim} in layer '{name}'.")
                expected_numel *= dim
                if expected_numel > 2**31 - 1:
                    raise CheckpointError(f"Arithmetic overflow in tensor dimension calculation for '{name}'.")

            num_chunks = layer["num_chunks"]
            total_raw = layer["raw_len"]
            chunk_records = layer["chunks"]
            if len(chunk_records) != num_chunks:
                raise CheckpointError(f"Chunk count mismatch for '{name}': declared {num_chunks}, found {len(chunk_records)}.")

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

            if device != "cpu" and torch.cuda.is_available() and str(device).startswith("cuda"):
                tensor = tensor.pin_memory().to(device, non_blocking=True)
            else:
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
        limits: Optional[ResourceLimits] = None,
    ) -> Dict[str, torch.Tensor]:
        """Backward-compatible loader for Version 1 checkpoints."""
        limits = limits or DEFAULT_LIMITS
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
