"""Google Cloud KMS Provider for TorchShield Envelope Encryption.

Uses Google Cloud Key Management Service (Cloud KMS) to wrap and unwrap 256-bit
Data Encryption Keys (DEKs) using symmetric CryptoKeys.
Checkpoint payloads are encrypted locally using AES-256-GCM.
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any, Callable, Dict, Optional, Tuple

from torchshield.key_providers.base import KeyProvider, KeyProviderError

logger = logging.getLogger(__name__)

RETRYABLE_GCP_ERRORS = {
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
}


class GCPKMSProvider(KeyProvider):
    """Production KeyProvider integrated with Google Cloud KMS."""

    def __init__(
        self,
        key_name: str,
        kms_client: Optional[Any] = None,
        max_retries: int = 3,
        base_backoff_sec: float = 0.05,
    ):
        """Initialize GCP KMS provider.

        Args:
            key_name: Full resource path of the CryptoKey, e.g.:
                projects/{project}/locations/{location}/keyRings/{keyRing}/cryptoKeys/{key}
            kms_client: Optional pre-configured KeyManagementServiceClient or mock.
            max_retries: Number of retry attempts on transient network/quota errors.
            base_backoff_sec: Base backoff time in seconds for exponential backoff.
        """
        if not key_name or not isinstance(key_name, str):
            raise ValueError("key_name must be a valid GCP KMS CryptoKey resource path.")
        self._key_name = key_name
        self._max_retries = max(0, max_retries)
        self._base_backoff_sec = max(0.001, base_backoff_sec)

        if kms_client is not None:
            self._client = kms_client
        else:
            try:
                from google.cloud import kms
                self._client = kms.KeyManagementServiceClient()
            except ImportError as exc:
                raise ImportError(
                    "google-cloud-kms is required for GCPKMSProvider. Install via `pip install google-cloud-kms`"
                ) from exc
            except Exception as exc:
                raise KeyProviderError(f"Failed to initialize GCP KMS client: {exc}") from exc

    @property
    def is_production_secure(self) -> bool:
        return True

    def get_key_id(self) -> str:
        return self._key_name

    def _is_transient(self, exc: Exception) -> bool:
        msg = str(exc)
        code = getattr(exc, "code", None)
        code_name = code.name if hasattr(code, "name") else str(code)
        return (
            code_name in RETRYABLE_GCP_ERRORS
            or any(err in msg for err in RETRYABLE_GCP_ERRORS)
            or "exhausted" in msg.lower()
            or "unavailable" in msg.lower()
            or "deadline" in msg.lower()
        )

    def _execute_with_retry(self, func: Callable[[], Any], op_name: str) -> Any:
        attempt = 0
        while True:
            try:
                return func()
            except Exception as exc:
                attempt += 1
                if attempt > self._max_retries or not self._is_transient(exc):
                    logger.error(
                        "GCP KMS operation '%s' failed for key %s after %d attempt(s): %s",
                        op_name,
                        self._key_name,
                        attempt,
                        type(exc).__name__,
                    )
                    raise
                backoff = self._base_backoff_sec * (2 ** (attempt - 1))
                time.sleep(backoff)

    def _context_to_aad(self, encryption_context: Optional[Dict[str, str]]) -> bytes:
        if not encryption_context:
            return b""
        import json
        return json.dumps(encryption_context, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def generate_data_key(
        self, encryption_context: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, bytes]:
        """Generate a random 256-bit DEK and wrap it with GCP KMS."""
        plaintext_dek = bytearray(secrets.token_bytes(32))
        try:
            wrapped = self.wrap_key(bytes(plaintext_dek), encryption_context=encryption_context)
            return bytes(plaintext_dek), wrapped
        finally:
            # Overwrite plaintext buffer in memory
            for i in range(len(plaintext_dek)):
                plaintext_dek[i] = 0

    def wrap_key(
        self, plaintext_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Wrap a 256-bit DEK using GCP KMS Encrypt API."""
        if len(plaintext_key) != 32:
            raise ValueError("plaintext_key must be exactly 32 bytes.")
        aad = self._context_to_aad(encryption_context)

        def _call():
            if hasattr(self._client, "encrypt"):
                try:
                    response = self._client.encrypt(
                        request={"name": self._key_name, "plaintext": plaintext_key, "additional_authenticated_data": aad}
                    )
                except TypeError:
                    response = self._client.encrypt(
                        name=self._key_name, plaintext=plaintext_key, additional_authenticated_data=aad
                    )
                return response.ciphertext
            raise KeyProviderError("Client missing encrypt method.")

        try:
            return self._execute_with_retry(_call, "wrap_key")
        except Exception as exc:
            raise KeyProviderError(f"GCP KMS wrap_key request failed: {type(exc).__name__}: {exc}") from exc

    def unwrap_key(
        self, wrapped_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Unwrap a wrapped DEK using GCP KMS Decrypt API."""
        if not wrapped_key:
            raise ValueError("wrapped_key cannot be empty.")
        aad = self._context_to_aad(encryption_context)

        def _call():
            if hasattr(self._client, "decrypt"):
                try:
                    response = self._client.decrypt(
                        request={"name": self._key_name, "ciphertext": wrapped_key, "additional_authenticated_data": aad}
                    )
                except TypeError:
                    response = self._client.decrypt(
                        name=self._key_name, ciphertext=wrapped_key, additional_authenticated_data=aad
                    )
                return response.plaintext
            raise KeyProviderError("Client missing decrypt method.")

        try:
            plaintext = self._execute_with_retry(_call, "unwrap_key")
            if len(plaintext) != 32:
                raise KeyProviderError(f"Decrypted DEK length {len(plaintext)} != 32 bytes.")
            return bytes(plaintext)
        except KeyProviderError:
            raise
        except Exception as exc:
            raise KeyProviderError(f"GCP KMS unwrap_key request failed: {type(exc).__name__}: {exc}") from exc

    def rewrap_key(
        self,
        wrapped_key: bytes,
        new_key_id: str,
        encryption_context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Unwrap under active key and rewrap under new GCP KMS key without exposing secret."""
        plaintext = self.unwrap_key(wrapped_key, encryption_context=encryption_context)
        pt_buf = bytearray(plaintext)
        try:
            temp_provider = GCPKMSProvider(key_name=new_key_id, kms_client=self._client)
            return temp_provider.wrap_key(bytes(pt_buf), encryption_context=encryption_context)
        finally:
            for i in range(len(pt_buf)):
                pt_buf[i] = 0
