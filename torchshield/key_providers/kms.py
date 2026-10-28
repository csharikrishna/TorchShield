"""AWS Key Management Service (KMS) Provider for Enterprise Key Lifecycle Management.

Follows official AWS client-side envelope encryption patterns:
https://docs.aws.amazon.com/kms/latest/developerguide/concepts.html

Uses KMS exclusively to generate and wrap 256-bit Data Encryption Keys (DEKs).
Multi-gigabyte checkpoint payloads are encrypted locally via AES-256-GCM.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from torchshield.key_providers.base import KeyProvider, KeyProviderError

logger = logging.getLogger(__name__)

# Transient error codes eligible for bounded exponential backoff retry
RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "RequestTimeout",
    "ServiceUnavailable",
    "InternalErrorException",
    "KMSInternalException",
    "EndpointConnectionError",
    "ConnectTimeoutError",
    "ReadTimeoutError",
}


class AWSKMSProvider(KeyProvider):
    """Production key provider integrated with AWS Key Management Service (KMS)."""

    def __init__(
        self,
        key_arn: str,
        boto3_client: Optional[Any] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        max_retries: int = 3,
        base_backoff_sec: float = 0.05,
    ):
        if not key_arn:
            raise ValueError("key_arn must be a non-empty KMS Key ID, Key ARN, or Alias.")
        self._key_arn = key_arn
        self._max_retries = max(0, max_retries)
        self._base_backoff_sec = max(0.001, base_backoff_sec)

        if boto3_client is not None:
            self._client = boto3_client
        else:
            try:
                import boto3
                client_kwargs: Dict[str, Any] = {}
                if region_name:
                    client_kwargs["region_name"] = region_name
                if endpoint_url:
                    client_kwargs["endpoint_url"] = endpoint_url
                self._client = boto3.client("kms", **client_kwargs)
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required to use AWSKMSProvider. Install with: pip install boto3"
                ) from exc
            except Exception as exc:
                raise KeyProviderError(f"Failed to initialize AWS KMS client: {exc}") from exc

    @property
    def is_production_secure(self) -> bool:
        return True

    def get_key_id(self) -> str:
        return self._key_arn

    def _is_transient_error(self, exc: Exception) -> bool:
        """Inspect exception type or response code to classify transient vs permanent errors."""
        error_code = ""
        # Check botocore ClientError response
        if hasattr(exc, "response") and isinstance(exc.response, dict):
            error_code = exc.response.get("Error", {}).get("Code", "")
        if not error_code:
            error_code = type(exc).__name__
        msg = str(exc)
        return (
            error_code in RETRYABLE_ERROR_CODES
            or any(code in msg for code in RETRYABLE_ERROR_CODES)
            or "throttl" in msg.lower()
            or "timeout" in msg.lower()
        )

    def _execute_with_retry(self, func: Callable[[], Any], op_name: str) -> Any:
        """Execute a KMS API call with bounded retries and exponential backoff."""
        attempt = 0
        while True:
            try:
                return func()
            except Exception as exc:
                attempt += 1
                if attempt > self._max_retries or not self._is_transient_error(exc):
                    # Do not log or format plaintext key material
                    logger.error(
                        "AWS KMS operation '%s' failed for KeyId=%s after %d attempt(s): %s",
                        op_name,
                        self._key_arn,
                        attempt,
                        type(exc).__name__,
                    )
                    raise
                backoff = self._base_backoff_sec * (2 ** (attempt - 1))
                time.sleep(backoff)

    def generate_data_key(
        self, encryption_context: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, bytes]:
        """Request a fresh 256-bit DEK from AWS KMS."""
        params: Dict[str, Any] = {
            "KeyId": self._key_arn,
            "KeySpec": "AES_256",
        }
        if encryption_context:
            params["EncryptionContext"] = encryption_context

        try:
            response = self._execute_with_retry(
                lambda: self._client.generate_data_key(**params),
                op_name="generate_data_key",
            )
            plaintext_dek = response["Plaintext"]
            wrapped_dek = response["CiphertextBlob"]
            return bytes(plaintext_dek), bytes(wrapped_dek)
        except Exception as exc:
            raise KeyProviderError(f"AWS KMS generate_data_key request failed: {exc}") from exc

    def wrap_key(
        self, plaintext_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Wrap an existing 256-bit DEK using AWS KMS Encrypt API."""
        if len(plaintext_key) != 32:
            raise ValueError("plaintext_key must be exactly 32 bytes.")
        params: Dict[str, Any] = {
            "KeyId": self._key_arn,
            "Plaintext": plaintext_key,
        }
        if encryption_context:
            params["EncryptionContext"] = encryption_context

        try:
            response = self._execute_with_retry(
                lambda: self._client.encrypt(**params),
                op_name="encrypt",
            )
            return bytes(response["CiphertextBlob"])
        except Exception as exc:
            raise KeyProviderError(f"AWS KMS wrap_key request failed: {exc}") from exc

    def unwrap_key(
        self, wrapped_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Unwrap a wrapped DEK using AWS KMS Decrypt API."""
        params: Dict[str, Any] = {
            "CiphertextBlob": wrapped_key,
            "KeyId": self._key_arn,
        }
        if encryption_context:
            params["EncryptionContext"] = encryption_context

        try:
            response = self._execute_with_retry(
                lambda: self._client.decrypt(**params),
                op_name="decrypt",
            )
            plaintext_dek = response["Plaintext"]
            return bytes(plaintext_dek)
        except Exception as exc:
            raise KeyProviderError(f"AWS KMS unwrap_key request failed: {exc}") from exc

    def rewrap_key(
        self,
        wrapped_key: bytes,
        new_key_id: str,
        encryption_context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Re-encrypt a wrapped DEK under a new KEK using AWS KMS ReEncrypt API."""
        params: Dict[str, Any] = {
            "CiphertextBlob": wrapped_key,
            "DestinationKeyId": new_key_id,
        }
        if encryption_context:
            params["SourceEncryptionContext"] = encryption_context
            params["DestinationEncryptionContext"] = encryption_context

        try:
            response = self._execute_with_retry(
                lambda: self._client.re_encrypt(**params),
                op_name="re_encrypt",
            )
            self._key_arn = new_key_id
            return bytes(response["CiphertextBlob"])
        except Exception as exc:
            raise KeyProviderError(f"AWS KMS rewrap_key request failed: {exc}") from exc
