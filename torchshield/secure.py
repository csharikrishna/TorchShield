"""Authenticated byte envelopes using the vetted AES-256-GCM primitive.

This module is deliberately separate from the experimental TorchShield transform.
Use :class:`AESGCMEnvelope` for real confidentiality and integrity requirements;
using this wrapper does not make the custom TorchShield block transform a reviewed
cryptographic primitive, and it does not call the TorchShield engine at all.

Envelope wire format::

    magic (15 bytes: b"TorchShieldSEAL") || version (1 byte) || nonce (12 bytes) || ciphertext || tag (16 bytes)

The fixed header (16 bytes), nonce, and caller-supplied associated data are authenticated.
A fresh 96-bit nonce is generated for every encryption. Callers must still
manage key lifetime and rotate keys before encrypting an excessive number of
messages under one key.
"""
from __future__ import annotations

import json
import math
import secrets
import sys
from typing import Optional, Union

import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BytesLike = Union[bytes, bytearray, memoryview]


class AuthenticationError(ValueError):
    """Raised when an envelope is malformed or authentication fails."""


class AESGCMEnvelope:
    """Versioned, authenticated byte envelope backed by AES-256-GCM.

    Keys must be uniformly random 32-byte values. Use ``generate_key()`` to
    create a key, and store it in a proper secrets manager or other protected
    key store. The returned envelope includes a random nonce but never includes
    the key. Do not reuse a key/nonce pair.
    """

    MAGIC = b"TorchShieldSEAL"  # 15 bytes; envelope format branding only, not TorchShield encryption.
    VERSION = 1
    HEADER = MAGIC + bytes([VERSION])
    NONCE_SIZE = 12
    TAG_SIZE = 16
    MIN_ENVELOPE_SIZE = len(HEADER) + NONCE_SIZE + TAG_SIZE

    @staticmethod
    def generate_key() -> bytes:
        """Generate a new cryptographically random 256-bit AES key."""
        return AESGCM.generate_key(bit_length=256)

    @classmethod
    def encrypt(cls, plaintext: BytesLike, key: BytesLike, associated_data: BytesLike = b"") -> bytes:
        """Encrypt bytes and authenticate the envelope plus optional metadata.

        Always generates a fresh, cryptographically secure 96-bit random nonce.
        Ordinary callers cannot supply an explicit nonce, preventing accidental
        nonce reuse vulnerabilities under the same key.
        """
        plaintext_bytes = cls._require_bytes(plaintext, "plaintext")
        key_bytes = cls._validate_key(key)
        aad = cls._require_bytes(associated_data, "associated_data")
        nonce = secrets.token_bytes(cls.NONCE_SIZE)
        authenticated_header = cls.HEADER + nonce + aad
        encrypted = AESGCM(key_bytes).encrypt(nonce, plaintext_bytes, authenticated_header)
        return cls.HEADER + nonce + encrypted

    @classmethod
    def _encrypt_with_explicit_nonce(
        cls,
        plaintext: BytesLike,
        key: BytesLike,
        nonce: bytes,
        associated_data: BytesLike = b"",
    ) -> bytes:
        """Package-private encryption with an explicit nonce.

        INTERNAL USE ONLY: Strictly for controlled protocols (such as
        SecureModelCheckpoint) where nonce uniqueness is mathematically proven
        under an isolated, single-artifact key. Must never be exposed as an
        unrestricted public encryption interface.
        """
        plaintext_bytes = cls._require_bytes(plaintext, "plaintext")
        key_bytes = cls._validate_key(key)
        aad = cls._require_bytes(associated_data, "associated_data")
        if not isinstance(nonce, (bytes, bytearray, memoryview)):
            raise TypeError("nonce must be bytes-like.")
        if len(nonce) != cls.NONCE_SIZE:
            raise ValueError(f"nonce must be exactly {cls.NONCE_SIZE} bytes.")
        nonce_bytes = bytes(nonce)
        authenticated_header = cls.HEADER + nonce_bytes + aad
        encrypted = AESGCM(key_bytes).encrypt(nonce_bytes, plaintext_bytes, authenticated_header)
        return cls.HEADER + nonce_bytes + encrypted

    @classmethod
    def decrypt(cls, envelope: BytesLike, key: BytesLike, associated_data: BytesLike = b"") -> bytes:
        """Authenticate and decrypt a complete envelope.

        Invalid tags, wrong keys, altered payloads and wrong associated data all
        raise the same public exception to avoid exposing a detailed failure
        oracle to callers.
        """
        envelope_bytes = cls._require_bytes(envelope, "envelope")
        key_bytes = cls._validate_key(key)
        aad = cls._require_bytes(associated_data, "associated_data")
        if len(envelope_bytes) < cls.MIN_ENVELOPE_SIZE:
            raise AuthenticationError("Invalid or unauthenticated AES-GCM envelope.")
        if envelope_bytes[:len(cls.HEADER)] != cls.HEADER:
            raise AuthenticationError("Unsupported or invalid AES-GCM envelope header.")
        start = len(cls.HEADER)
        nonce = envelope_bytes[start:start + cls.NONCE_SIZE]
        encrypted = envelope_bytes[start + cls.NONCE_SIZE:]
        authenticated_header = cls.HEADER + nonce + aad
        try:
            return AESGCM(key_bytes).decrypt(nonce, encrypted, authenticated_header)
        except InvalidTag as exc:
            raise AuthenticationError(
                "Authentication failed: the key, associated data, or envelope may be incorrect or modified."
            ) from exc

    @staticmethod
    def _require_bytes(value: BytesLike, name: str) -> bytes:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"{name} must be bytes-like.")
        return bytes(value)

    @classmethod
    def _validate_key(cls, key: BytesLike) -> bytes:
        key_bytes = cls._require_bytes(key, "key")
        if len(key_bytes) != 32:
            raise ValueError("AESGCMEnvelope requires exactly 32 bytes of key material.")
        return key_bytes


class TensorAESGCMEnvelope:
    """Authenticated encryption for dense PyTorch tensors, independent of TorchShield.

    This stores a small JSON dtype/shape header and the contiguous raw tensor
    bytes inside an AESGCMEnvelope. The header is encrypted and authenticated;
    no pickle or ``torch.load`` serialization is used. The returned tensor is
    restored on CPU by default, or on the requested device after authentication.
    """

    TENSOR_MAGIC = b"TorchShieldTEN1"
    MAX_HEADER_BYTES = 64 * 1024
    _DTYPE_BY_NAME = {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    @classmethod
    def encrypt_tensor(
        cls,
        tensor: torch.Tensor,
        key: BytesLike,
        associated_data: BytesLike = b"",
    ) -> bytes:
        """Serialize and encrypt a dense tensor with authenticated shape/dtype."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor.")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise TypeError("Only dense, non-quantized, strided tensors are supported.")
        dtype_names = {dtype: name for name, dtype in cls._DTYPE_BY_NAME.items()}
        if tensor.dtype not in dtype_names:
            raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}.")

        contiguous = tensor.detach().to(device="cpu").contiguous()
        raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        header = json.dumps(
            {
                "version": 1,
                "dtype": dtype_names[tensor.dtype],
                "shape": list(tensor.shape),
                "raw_len": len(raw),
                "byteorder": sys.byteorder,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(header) > cls.MAX_HEADER_BYTES:
            raise ValueError("Tensor metadata header is unexpectedly large.")
        payload = cls.TENSOR_MAGIC + len(header).to_bytes(4, "big") + header + raw
        return AESGCMEnvelope.encrypt(payload, key, associated_data)

    @classmethod
    def decrypt_tensor(
        cls,
        envelope: BytesLike,
        key: BytesLike,
        associated_data: BytesLike = b"",
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Authenticate, validate tensor metadata and restore the tensor."""
        payload = AESGCMEnvelope.decrypt(envelope, key, associated_data)
        prefix_len = len(cls.TENSOR_MAGIC)
        if len(payload) < prefix_len + 4 or payload[:prefix_len] != cls.TENSOR_MAGIC:
            raise AuthenticationError("Authenticated payload is not a supported tensor envelope.")
        header_len = int.from_bytes(payload[prefix_len:prefix_len + 4], "big")
        if header_len < 2 or header_len > cls.MAX_HEADER_BYTES:
            raise AuthenticationError("Invalid tensor metadata header length.")
        header_start = prefix_len + 4
        header_end = header_start + header_len
        if header_end > len(payload):
            raise AuthenticationError("Truncated tensor metadata header.")
        try:
            metadata = json.loads(payload[header_start:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthenticationError("Invalid tensor metadata encoding.") from exc
        version = metadata.get("version") if isinstance(metadata, dict) else None
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise AuthenticationError("Unsupported tensor metadata version.")
        dtype_name = metadata.get("dtype")
        if not isinstance(dtype_name, str) or dtype_name not in cls._DTYPE_BY_NAME:
            raise AuthenticationError("Unsupported tensor dtype metadata.")
        shape_raw = metadata.get("shape")
        if not isinstance(shape_raw, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape_raw
        ):
            raise AuthenticationError("Invalid tensor shape metadata.")
        raw_len = metadata.get("raw_len")
        if isinstance(raw_len, bool) or not isinstance(raw_len, int) or raw_len < 0:
            raise AuthenticationError("Invalid tensor byte-length metadata.")
        if metadata.get("byteorder") != sys.byteorder:
            raise AuthenticationError("Tensor envelope byte order differs from the current platform.")

        raw = payload[header_end:]
        dtype = cls._DTYPE_BY_NAME[dtype_name]
        element_count = math.prod(shape_raw) if shape_raw else 1
        expected_len = element_count * torch.empty((), dtype=dtype).element_size()
        if raw_len != len(raw) or raw_len != expected_len:
            raise AuthenticationError("Tensor metadata does not match the authenticated payload length.")

        if raw:
            bytes_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        else:
            bytes_tensor = torch.empty((0,), dtype=torch.uint8)
        result = bytes_tensor.view(dtype).reshape(tuple(shape_raw))

        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Tensor restore device must be CPU or CUDA.")
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for tensor restore but is unavailable.")
            if target.index is not None and (target.index < 0 or target.index >= torch.cuda.device_count()):
                raise ValueError(f"CUDA device index {target.index} is out of range.")
        return result.to(target)
