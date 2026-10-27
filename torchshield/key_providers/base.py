"""Base interface for Enterprise Key Management Service (KMS) providers.

Supports envelope encryption where the Key Management Service manages Key Encryption Keys (KEKs)
and wraps/unwraps per-artifact Data Encryption Keys (DEKs) without transmitting multi-gigabyte
checkpoint payloads over network boundaries.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class KeyProviderError(RuntimeError):
    """Raised when key generation, wrapping, unwrapping, or rotation fails."""


class KeyProvider(ABC):
    """Abstract base class for pluggable cryptographic key providers."""

    @property
    @abstractmethod
    def is_production_secure(self) -> bool:
        """Whether this provider is suitable for production environments."""
        pass

    @abstractmethod
    def get_key_id(self) -> str:
        """Return non-secret identifier or ARN of the active Key Encryption Key (KEK)."""
        pass

    @abstractmethod
    def generate_data_key(
        self, encryption_context: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, bytes]:
        """Generate a fresh 256-bit Data Encryption Key (DEK).

        Returns:
            Tuple of (plaintext_key_32_bytes, wrapped_key_bytes).
            Plaintext key must be securely wiped from memory after use.
        """
        pass

    @abstractmethod
    def unwrap_key(
        self, wrapped_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Unwrap a wrapped Data Encryption Key using the active KEK.

        Returns:
            Plaintext 256-bit AES key (32 bytes).
        """
        pass

    @abstractmethod
    def wrap_key(
        self, plaintext_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Wrap an existing 256-bit Data Encryption Key using the active KEK.

        Returns:
            Wrapped key ciphertext bytes.
        """
        pass

    @abstractmethod
    def rewrap_key(
        self,
        wrapped_key: bytes,
        new_key_id: str,
        encryption_context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Re-wrap an existing DEK under a new KEK version without re-encrypting model payload."""
        pass
