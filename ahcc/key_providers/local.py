"""Local / Development Key Provider for testing and offline workflows.

WARNING: This provider is for local development and unit testing only.
Do NOT use in production environments where hardware-backed or managed KMS
(such as AWS KMS, GCP Cloud KMS, or HashiCorp Vault) is required.
"""
from __future__ import annotations

import json
import os
import secrets
from typing import Dict, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from ahcc.key_providers.base import KeyProvider, KeyProviderError


class LocalDevKeyProvider(KeyProvider):
    """Local software key provider using AES-256-GCM envelope key wrapping."""

    def __init__(self, key_id: str = "local-dev-kek-v1", master_kek: Optional[bytes] = None):
        self._key_id = key_id
        if master_kek is not None:
            if len(master_kek) != 32:
                raise ValueError("master_kek must be exactly 32 bytes.")
            self._kek = master_kek
        else:
            # Fall back to environment variable or generate in-memory key
            env_key = os.environ.get("AHCC_LOCAL_KEK")
            if env_key:
                try:
                    self._kek = bytes.fromhex(env_key)
                except ValueError:
                    self._kek = env_key.encode("utf-8")[:32].ljust(32, b"\x00")
            else:
                self._kek = secrets.token_bytes(32)

    @property
    def is_production_secure(self) -> bool:
        return False

    def get_key_id(self) -> str:
        return self._key_id

    def generate_data_key(
        self, encryption_context: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, bytes]:
        """Generate a fresh 32-byte DEK and wrap it with the local KEK."""
        plaintext_dek = secrets.token_bytes(32)
        wrapped_dek = self._wrap_raw(plaintext_dek, self._kek, encryption_context)
        return plaintext_dek, wrapped_dek

    def unwrap_key(
        self, wrapped_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Unwrap a wrapped DEK using the local KEK."""
        return self._unwrap_raw(wrapped_key, self._kek, encryption_context)

    def rewrap_key(
        self,
        wrapped_key: bytes,
        new_key_id: str,
        encryption_context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Unwrap and re-wrap under the same or updated KEK."""
        plaintext_dek = self.unwrap_key(wrapped_key, encryption_context)
        self._key_id = new_key_id
        return self._wrap_raw(plaintext_dek, self._kek, encryption_context)

    def _wrap_raw(
        self, plaintext_dek: bytes, kek: bytes, encryption_context: Optional[Dict[str, str]]
    ) -> bytes:
        nonce = secrets.token_bytes(12)
        aad = json.dumps(encryption_context or {}, sort_keys=True).encode("utf-8")
        cipher = AESGCM(kek)
        ciphertext = cipher.encrypt(nonce, plaintext_dek, aad)
        return nonce + ciphertext

    def _unwrap_raw(
        self, wrapped_key: bytes, kek: bytes, encryption_context: Optional[Dict[str, str]]
    ) -> bytes:
        if len(wrapped_key) < 28:
            raise KeyProviderError("Wrapped key payload is truncated or invalid.")
        nonce = wrapped_key[:12]
        ciphertext = wrapped_key[12:]
        aad = json.dumps(encryption_context or {}, sort_keys=True).encode("utf-8")
        cipher = AESGCM(kek)
        try:
            return cipher.decrypt(nonce, ciphertext, aad)
        except Exception as exc:
            raise KeyProviderError("Failed to unwrap DEK: tag mismatch or wrong context.") from exc
