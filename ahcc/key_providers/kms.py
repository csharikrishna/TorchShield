"""AWS Key Management Service (KMS) Provider for Enterprise Key Lifecycle Management.

Follows official AWS client-side envelope encryption patterns:
https://docs.aws.amazon.com/kms/latest/developerguide/concepts.html

Uses KMS exclusively to generate and wrap 256-bit Data Encryption Keys (DEKs).
Multi-gigabyte checkpoint payloads are encrypted locally via AES-256-GCM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from ahcc.key_providers.base import KeyProvider, KeyProviderError

logger = logging.getLogger(__name__)


class AWSKMSProvider(KeyProvider):
    """Production key provider integrated with AWS Key Management Service (KMS)."""

    def __init__(
        self,
        key_arn: str,
        boto3_client: Optional[Any] = None,
        region_name: Optional[str] = None,
    ):
        if not key_arn:
            raise ValueError("key_arn must be a non-empty KMS Key ID, Key ARN, or Alias.")
        self._key_arn = key_arn

        if boto3_client is not None:
            self._client = boto3_client
        else:
            try:
                import boto3
                self._client = boto3.client("kms", region_name=region_name)
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
            response = self._client.generate_data_key(**params)
            plaintext_dek = response["Plaintext"]
            wrapped_dek = response["CiphertextBlob"]
            return bytes(plaintext_dek), bytes(wrapped_dek)
        except Exception as exc:
            logger.error("AWS KMS generate_data_key failed for KeyId=%s", self._key_arn)
            raise KeyProviderError(f"AWS KMS generate_data_key request failed: {exc}") from exc

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
            response = self._client.decrypt(**params)
            plaintext_dek = response["Plaintext"]
            return bytes(plaintext_dek)
        except Exception as exc:
            logger.error("AWS KMS decrypt failed for KeyId=%s", self._key_arn)
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
            response = self._client.re_encrypt(**params)
            self._key_arn = new_key_id
            return bytes(response["CiphertextBlob"])
        except Exception as exc:
            logger.error("AWS KMS re_encrypt failed from %s to %s", self._key_arn, new_key_id)
            raise KeyProviderError(f"AWS KMS rewrap_key request failed: {exc}") from exc
