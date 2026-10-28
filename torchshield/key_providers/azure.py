"""Azure Key Vault Provider for TorchShield Envelope Encryption.

Uses Azure Key Vault or Managed HSM to wrap and unwrap 256-bit
Data Encryption Keys (DEKs) using RSA-OAEP-256 or AES Key Wrap.
Checkpoint payloads are encrypted locally via AES-256-GCM.
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any, Callable, Dict, Optional, Tuple

from torchshield.key_providers.base import KeyProvider, KeyProviderError

logger = logging.getLogger(__name__)

RETRYABLE_AZURE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class AzureKeyVaultProvider(KeyProvider):
    """Production KeyProvider integrated with Azure Key Vault."""

    def __init__(
        self,
        vault_url: str,
        key_name: str,
        key_version: Optional[str] = None,
        key_client: Optional[Any] = None,
        crypto_client: Optional[Any] = None,
        algorithm: str = "RSA-OAEP-256",
        max_retries: int = 3,
        base_backoff_sec: float = 0.05,
    ):
        """Initialize Azure Key Vault provider.

        Args:
            vault_url: Azure Key Vault URL, e.g. https://<vault-name>.vault.azure.net/
            key_name: Name of the key within Key Vault.
            key_version: Optional specific key version ID.
            key_client: Optional pre-configured KeyClient.
            crypto_client: Optional pre-configured CryptographyClient or mock.
            algorithm: Key wrap algorithm (default 'RSA-OAEP-256' or 'A256KW').
            max_retries: Retry attempts on 429/503.
            base_backoff_sec: Base backoff delay.
        """
        if not vault_url or not key_name:
            raise ValueError("vault_url and key_name must be non-empty.")
        self._vault_url = vault_url.rstrip("/")
        self._key_name = key_name
        self._key_version = key_version
        self._algorithm = algorithm
        self._max_retries = max(0, max_retries)
        self._base_backoff_sec = max(0.001, base_backoff_sec)

        if crypto_client is not None:
            self._crypto_client = crypto_client
        else:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.keys import KeyClient
                from azure.keyvault.keys.crypto import CryptographyClient

                credential = DefaultAzureCredential()
                if key_client is None:
                    key_client = KeyClient(vault_url=self._vault_url, credential=credential)
                key = key_client.get_key(self._key_name, version=self._key_version)
                self._crypto_client = CryptographyClient(key, credential=credential)
            except ImportError as exc:
                raise ImportError(
                    "azure-keyvault-keys and azure-identity are required for AzureKeyVaultProvider. "
                    "Install via `pip install azure-keyvault-keys azure-identity`."
                ) from exc
            except Exception as exc:
                raise KeyProviderError(f"Failed to initialize Azure Key Vault client: {exc}") from exc

    @property
    def is_production_secure(self) -> bool:
        return True

    def get_key_id(self) -> str:
        version_part = f"/{self._key_version}" if self._key_version else ""
        return f"{self._vault_url}/keys/{self._key_name}{version_part}"

    def _is_transient(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        status_code = getattr(exc, "status_code", None)
        return (
            status_code in RETRYABLE_AZURE_STATUS_CODES
            or "throttl" in msg
            or "timeout" in msg
            or "temporarily unavailable" in msg
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
                        "Azure Key Vault operation '%s' failed for key %s after %d attempt(s): %s",
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
        """Generate a random 256-bit DEK and wrap it with Azure Key Vault."""
        plaintext_dek = bytearray(secrets.token_bytes(32))
        try:
            wrapped = self.wrap_key(bytes(plaintext_dek), encryption_context=encryption_context)
            return bytes(plaintext_dek), wrapped
        finally:
            for i in range(len(plaintext_dek)):
                plaintext_dek[i] = 0

    def wrap_key(
        self, plaintext_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Wrap a 256-bit DEK using Azure Key Vault CryptographyClient."""
        if len(plaintext_key) != 32:
            raise ValueError("plaintext_key must be exactly 32 bytes.")

        def _call():
            result = self._crypto_client.wrap_key(self._algorithm, plaintext_key)
            return result.encrypted_key if hasattr(result, "encrypted_key") else result

        try:
            return self._execute_with_retry(_call, "wrap_key")
        except Exception as exc:
            raise KeyProviderError(f"Azure Key Vault wrap_key failed: {type(exc).__name__}: {exc}") from exc

    def unwrap_key(
        self, wrapped_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Unwrap a wrapped DEK using Azure Key Vault CryptographyClient."""
        if not wrapped_key:
            raise ValueError("wrapped_key cannot be empty.")

        def _call():
            result = self._crypto_client.unwrap_key(self._algorithm, wrapped_key)
            return result.key if hasattr(result, "key") else result

        try:
            plaintext = self._execute_with_retry(_call, "unwrap_key")
            if len(plaintext) != 32:
                raise KeyProviderError(f"Decrypted DEK length {len(plaintext)} != 32 bytes.")
            return bytes(plaintext)
        except KeyProviderError:
            raise
        except Exception as exc:
            raise KeyProviderError(f"Azure Key Vault unwrap_key failed: {type(exc).__name__}: {exc}") from exc

    def rewrap_key(
        self,
        wrapped_key: bytes,
        new_key_id: str,
        encryption_context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Unwrap under active key and rewrap under new Azure Key Vault key."""
        plaintext = self.unwrap_key(wrapped_key, encryption_context=encryption_context)
        pt_buf = bytearray(plaintext)
        try:
            # Parse new key URL or name
            new_provider = AzureKeyVaultProvider(
                vault_url=self._vault_url,
                key_name=new_key_id.split("/")[-1],
                crypto_client=self._crypto_client,
                algorithm=self._algorithm,
            )
            return new_provider.wrap_key(bytes(pt_buf), encryption_context=encryption_context)
        finally:
            for i in range(len(pt_buf)):
                pt_buf[i] = 0
