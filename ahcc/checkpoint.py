"""Production-grade authenticated checkpoint encryption for AI model weights.

Implements chunked and layer-wise authenticated encryption using AES-256-GCM
with authenticated manifests, strictly binding file identity, layer names,
chunk ordering, and tensor shapes. Also provides experimental AHCC checkpoint
adapters for scientific comparison.
"""
from __future__ import annotations

import io
import json
import math
import os
import secrets
import struct
import sys
import uuid
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ahcc.secure import AESGCMEnvelope, AuthenticationError, TensorAESGCMEnvelope
from ahcc.engine import AHCC_Engine, TensorCipherWrapper


class CheckpointError(AuthenticationError):
    """Raised when checkpoint manifest or chunk authentication fails."""


class SecureModelCheckpoint:
    """End-to-end authenticated model checkpoint storage and streaming loader."""

    MAGIC = b"AHCCMOD1"  # 8 bytes format marker
    VERSION = 1

    @classmethod
    def save_checkpoint(
        cls,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        key: bytes,
        chunk_bytes: int = 4 * 1024 * 1024,  # 4 MiB chunks
    ) -> None:
        """Encrypt and serialize a model state dict with chunked authenticated envelopes."""
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dictionary of tensors.")
        if len(key) != 32:
            raise ValueError("AES-256 key must be exactly 32 bytes.")

        file_id = str(uuid.uuid4()).encode("utf-8")
        manifest_entries = []
        chunks_data = []

        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Item '{name}' must be a torch.Tensor.")
            contiguous = tensor.detach().to(device="cpu").contiguous()
            dtype_str = str(tensor.dtype).replace("torch.", "")
            shape_list = list(tensor.shape)
            raw_bytes = contiguous.view(torch.uint8).numpy().tobytes()
            total_raw = len(raw_bytes)

            num_chunks = max(1, math.ceil(total_raw / chunk_bytes)) if total_raw > 0 else 1
            chunk_records = []

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_bytes
                end = min(start + chunk_bytes, total_raw)
                chunk_slice = raw_bytes[start:end]

                # AAD binds file_id, layer name, chunk index, total chunks, and shape
                aad = (
                    file_id
                    + b"::" + name.encode("utf-8")
                    + b"::chunk=" + struct.pack(">II", chunk_idx, num_chunks)
                    + b"::shape=" + json.dumps(shape_list).encode("utf-8")
                )
                encrypted_chunk = AESGCMEnvelope.encrypt(chunk_slice, key, associated_data=aad)
                chunk_records.append({
                    "chunk_index": chunk_idx,
                    "length": len(encrypted_chunk),
                })
                chunks_data.append(encrypted_chunk)

            manifest_entries.append({
                "name": name,
                "dtype": dtype_str,
                "shape": shape_list,
                "raw_len": total_raw,
                "num_chunks": num_chunks,
                "chunks": chunk_records,
            })

        # Serialize and authenticate manifest
        manifest_payload = json.dumps({
            "version": cls.VERSION,
            "file_id": file_id.decode("utf-8"),
            "layers": manifest_entries,
        }, sort_keys=True).encode("utf-8")

        manifest_aad = file_id + b"::MANIFEST"
        encrypted_manifest = AESGCMEnvelope.encrypt(manifest_payload, key, associated_data=manifest_aad)

        # Write container:
        # MAGIC (8B) || VERSION (1B) || file_id_len (1B) || file_id || manifest_len (4B) || encrypted_manifest || chunks
        with open(output_path, "wb") as f:
            f.write(cls.MAGIC)
            f.write(bytes([cls.VERSION]))
            f.write(bytes([len(file_id)]))
            f.write(file_id)
            f.write(struct.pack(">I", len(encrypted_manifest)))
            f.write(encrypted_manifest)
            for c in chunks_data:
                f.write(c)

    @classmethod
    def load_checkpoint(
        cls,
        input_path: str,
        key: bytes,
        device: Union[str, torch.device] = "cpu",
        target_model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Authenticate manifest and stream-decrypt model layers."""
        if len(key) != 32:
            raise ValueError("AES-256 key must be exactly 32 bytes.")

        with open(input_path, "rb") as f:
            header = f.read(len(cls.MAGIC) + 2)
            if len(header) < len(cls.MAGIC) + 2 or header[:len(cls.MAGIC)] != cls.MAGIC:
                raise CheckpointError("Not a valid AHCC secure model checkpoint.")
            ver = header[len(cls.MAGIC)]
            if ver != cls.VERSION:
                raise CheckpointError(f"Unsupported checkpoint version: {ver}")
            file_id_len = header[len(cls.MAGIC) + 1]
            file_id = f.read(file_id_len)
            if len(file_id) != file_id_len:
                raise CheckpointError("Truncated checkpoint file ID.")

            manifest_len_bytes = f.read(4)
            if len(manifest_len_bytes) < 4:
                raise CheckpointError("Truncated checkpoint manifest length.")
            manifest_len = struct.unpack(">I", manifest_len_bytes)[0]
            encrypted_manifest = f.read(manifest_len)
            if len(encrypted_manifest) != manifest_len:
                raise CheckpointError("Truncated checkpoint manifest.")

            # Authenticate and decrypt manifest
            manifest_aad = file_id + b"::MANIFEST"
            manifest_bytes = AESGCMEnvelope.decrypt(encrypted_manifest, key, associated_data=manifest_aad)
            manifest = json.loads(manifest_bytes.decode("utf-8"))

            if manifest.get("file_id") != file_id.decode("utf-8"):
                raise CheckpointError("Manifest file_id mismatch.")

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
                        raise CheckpointError(f"Truncated chunk {chunk_idx} in layer '{name}'.")

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
                    raise CheckpointError(f"Corrupted layer bytes length for '{name}'.")

                if total_raw > 0:
                    tensor = torch.frombuffer(bytearray(layer_bytes), dtype=dtype).clone().reshape(shape)
                else:
                    tensor = torch.empty(shape, dtype=dtype)

                tensor = tensor.to(device)
                restored_state[name] = tensor

                if target_model is not None:
                    # Stream directly into model parameter if matching
                    curr_param = target_model.get_parameter(name) if hasattr(target_model, "get_parameter") else None
                    if curr_param is not None:
                        curr_param.data.copy_(tensor)

        return restored_state
