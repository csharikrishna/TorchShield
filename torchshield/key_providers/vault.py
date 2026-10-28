"""HashiCorp Vault Transit KeyProvider for TorchShield Envelope Encryption.

Uses HashiCorp Vault Transit secrets engine to generate and wrap 256-bit
Data Encryption Keys (DEKs) using AES-256-GCM.
Multi-gigabyte checkpoint payloads are encrypted locally via AES-256-GCM.
"""
from __future__ import annotations

import base64
import logging
import secrets
import time
from typing import Any, Callable, Dict, Optional, Tuple

from torchshield.key_providers.base import KeyProvider, KeyProviderError

logger = logging.getLogger(__name__)

RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


class HashiCorpVaultProvider(KeyProvider):
    """Production KeyProvider integrated with HashiCorp Vault Transit Secrets Engine."""

    def __init__(
        self,
        key_name: str,
        vault_url: str = "http://127.0.0.1:8200",
        token: Optional[str] = None,
        mount_point: str = "transit",
        hvac_client: Optional[Any] = None,
        max_retries: int = 3,
        base_backoff_sec: float = 0.05,
    ):
        """Initialize HashiCorp Vault Transit provider.

        Args:
            key_name: Name of the Transit key.
            vault_url: URL to Vault server.
            token: Vault authentication token.
            mount_point: Mount path for Transit secrets engine (default 'transit').
            hvac_client: Pre-configured hvac.Client or mock.
            max_retries: Retry attempts on transient errors.
            base_backoff_sec: Base backoff delay.
        """
        if not key_name:
            raise ValueError("key_name must be non-empty.")
        self._key_name = key_name
        self._vault_url = vault_url
        self._mount_point = mount_point
        self._max_retries = max(0, max_retries)
        self._base_backoff_sec = max(0.001, base_backoff_sec)

        if hvac_client is not None:
            self._client = hvac_client
        else:
            try:
                import hvac
                self._client = hvac.Client(url=vault_url, token=token)
            except ImportError as exc:
                raise ImportError(
                    "hvac is required for HashiCorpVaultProvider. Install via `pip install hvac`."
                ) from exc
            except Exception as exc:
                raise KeyProviderError(f"Failed to initialize HashiCorp Vault client: {exc}") from exc

    @property
    def is_production_secure(self) -> bool:
        return True

    def get_key_id(self) -> str:
        return f"{self._vault_url.rstrip('/')}/v1/{self._mount_point}/keys/{self._key_name}"

    def _context_b64(self, encryption_context: Optional[Dict[str, str]]) -> Optional[str]:
        if not encryption_context:
            return None
        import json
        raw = json.dumps(encryption_context, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(raw).decode("utf-8")

    def _is_transient(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        status = getattr(exc, "status_code", None)
        return (
            status in RETRYABLE_HTTP_STATUSES
            or "connection refused" in msg
            or "timeout" in msg
            or "rate limit" in msg
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
                        "HashiCorp Vault operation '%s' failed for key %s after %d attempt(s): %s",
                        op_name,
                        self._key_name,
                        attempt,
                        type(exc).__name__,
                    )
                    raise
                backoff = self._base_backoff_sec * (2 ** (attempt - 1))
                time.sleep(backoff)

    def generate_data_key(
        self, encryption_context: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, bytes]:
        """Request a fresh 256-bit DEK from Vault Transit."""
        ctx_b64 = self._context_b64(encryption_context)

        def _call():
            kwargs: Dict[str, Any] = {
                "name": self._key_name,
                "key_type": "plaintext",
                "mount_point": self._mount_point,
            }
            if ctx_b64:
                kwargs["context"] = ctx_b64
            res = self._client.secrets.transit.generate_data_key(**kwargs)
            data = res.get("data", res)
            plaintext_b64 = data["plaintext"]
            ciphertext_str = data["ciphertext"]
            pt = base64.b64decode(plaintext_b64)
            wrapped = ciphertext_str.encode("utf-8")
            return pt, wrapped

        try:
            pt, wrapped = self._execute_with_retry(_call, "generate_data_key")
            return pt, wrapped
        except Exception as exc:
            raise KeyProviderError(f"Vault generate_data_key failed: {type(exc).__name__}: {exc}") from exc

    def wrap_key(
        self, plaintext_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Wrap a 256-bit DEK using Vault Transit Encrypt API."""
        if len(plaintext_key) != 32:
            raise ValueError("plaintext_key must be exactly 32 bytes.")
        ctx_b64 = self._context_b64(encryption_context)
        pt_b64 = base64.b64encode(plaintext_key).decode("utf-8")

        def _call():
            kwargs: Dict[str, Any] = {
                "name": self._key_name,
                "plaintext": pt_b64,
                "mount_point": self._mount_point,
            }
            if ctx_b64:
                kwargs["context"] = ctx_b64
            res = self._client.secrets.transit.encrypt_data(**kwargs)
            data = res.get("data", res)
            return data["ciphertext"].encode("utf-8")

        try:
            return self._execute_with_retry(_call, "wrap_key")
        except Exception as exc:
            raise KeyProviderError(f"Vault wrap_key failed: {type(exc).__name__}: {exc}") from exc

    def unwrap_key(
        self, wrapped_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Unwrap a wrapped DEK using Vault Transit Decrypt API."""
        if not wrapped_key:
            raise ValueError("wrapped_key cannot be empty.")
        ctx_b64 = self._context_b64(encryption_context)
        ciphertext_str = wrapped_key.decode("utf-8")

        def _call():
            kwargs: Dict[str, Any] = {
                "name": self._key_name,
                "ciphertext": ciphertext_str,
                "mount_point": self._mount_point,
            }
            if ctx_b64:
                kwargs["context"] = ctx_b64
            res = self._client.secrets.transit.decrypt_data(**kwargs)
            data = res.get("data", res)
            pt_b64 = data["plaintext"]
            return base64.b64decode(pt_b64)

        try:
            plaintext = self._execute_with_retry(_call, "unwrap_key")
            if len(plaintext) != 32:
                raise KeyProviderError(f"Decrypted DEK length {len(plaintext)} != 32 bytes.")
            return bytes(plaintext)
        except KeyProviderError:
            raise
        except Exception as exc:
            raise KeyProviderError(f"Vault unwrap_key failed: {type(exc).__name__}: {exc}") from exc

    def rewrap_key(
        self,
        wrapped_key: bytes,
        new_key_id: str,
        encryption_context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Rewrap ciphertext under active or new Transit key."""
        ctx_b64 = self._context_b64(encryption_context)
        ciphertext_str = wrapped_key.decode("utf-8")
        target_name = new_key_id if new_key_id else self._key_name

        def _call():
            kwargs: Dict[str, Any] = {
                "name": target_name,
                "ciphertext": ciphertext_str,
                "mount_point": self._mount_point,
            }
            if ctx_b64:
                kwargs["context"] = ctx_b64
            res = self._client.secrets.transit.rewrap_data(**kwargs)
            data = res.get("data", res)
            return data["ciphertext"].encode("utf-8")

        try:
            return self._execute_with_retry(_call, "rewrap_key")
        except Exception as exc:
            # Fallback to unwrap + wrap if rewrap endpoint is unsupported
            pt = self.unwrap_key(wrapped_key, encryption_context=encryption_context)
            pt_buf = bytearray(pt)
            try:
                temp_p = HashiCorpVaultProvider(
                    key_name=target_name,
                    vault_url=self._vault_url,
                    mount_point=self._mount_point,
                    hvac_client=self._client,
                )
                return temp_p.wrap_key(bytes(pt_buf), encryption_context=encryption_context)
            finally:
                for i in range(len(pt_buf)):
                    pt_buf[i] = 0
