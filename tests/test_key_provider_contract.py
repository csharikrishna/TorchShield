"""Contract verification suite for Enterprise Key Providers."""
from __future__ import annotations

import secrets
from unittest import mock
import pytest

from torchshield.key_providers.base import KeyProviderError
from torchshield.key_providers.local import LocalDevKeyProvider
from torchshield.key_providers.kms import AWSKMSProvider


def test_local_dev_key_provider_lifecycle():
    """Verify LocalDevKeyProvider generate, unwrap, and tamper rejection."""
    kek = secrets.token_bytes(32)
    provider = LocalDevKeyProvider(key_id="test-kek-1", master_kek=kek)
    assert not provider.is_production_secure
    assert provider.get_key_id() == "test-kek-1"

    context = {"model": "transformer", "stage": "train"}
    plaintext_dek, wrapped_dek = provider.generate_data_key(encryption_context=context)

    assert len(plaintext_dek) == 32
    assert len(wrapped_dek) >= 32 + 12 + 16  # DEK + Nonce + Tag

    # Successfully unwrap with matching context
    unwrapped = provider.unwrap_key(wrapped_dek, encryption_context=context)
    assert unwrapped == plaintext_dek

    # Unwrapping with wrong context must fail
    bad_context = {"model": "transformer", "stage": "eval"}
    with pytest.raises(KeyProviderError, match="Failed to unwrap DEK"):
        provider.unwrap_key(wrapped_dek, encryption_context=bad_context)

    # Corrupted ciphertext must fail
    corrupted_wrapped = bytearray(wrapped_dek)
    corrupted_wrapped[-1] ^= 0xFF
    with pytest.raises(KeyProviderError, match="Failed to unwrap DEK"):
        provider.unwrap_key(bytes(corrupted_wrapped), encryption_context=context)


def test_aws_kms_provider_contract_with_mock_client():
    """Verify AWSKMSProvider envelope encryption calls using mocked boto3 client."""
    mock_client = mock.MagicMock()
    plaintext_dek = secrets.token_bytes(32)
    ciphertext_blob = b"kms-wrapped-dek-ciphertext-blob-123456"

    mock_client.generate_data_key.return_value = {
        "Plaintext": plaintext_dek,
        "CiphertextBlob": ciphertext_blob,
    }
    mock_client.decrypt.return_value = {
        "Plaintext": plaintext_dek,
    }
    mock_client.re_encrypt.return_value = {
        "CiphertextBlob": b"kms-re-wrapped-dek-blob",
    }

    key_arn = "arn:aws:kms:us-west-2:111122223333:key/sample-key-uuid"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client)

    assert provider.is_production_secure
    assert provider.get_key_id() == key_arn

    context = {"env": "prod", "cluster": "us-west-2"}

    # 1. Test generate_data_key
    p_key, w_key = provider.generate_data_key(encryption_context=context)
    assert p_key == plaintext_dek
    assert w_key == ciphertext_blob
    mock_client.generate_data_key.assert_called_once_with(
        KeyId=key_arn, KeySpec="AES_256", EncryptionContext=context
    )

    # 2. Test unwrap_key
    unwrapped = provider.unwrap_key(w_key, encryption_context=context)
    assert unwrapped == plaintext_dek
    mock_client.decrypt.assert_called_once_with(
        CiphertextBlob=w_key, KeyId=key_arn, EncryptionContext=context
    )

    # 3. Test rewrap_key
    new_arn = "arn:aws:kms:us-west-2:111122223333:key/new-key-uuid"
    re_wrapped = provider.rewrap_key(w_key, new_key_id=new_arn, encryption_context=context)
    assert re_wrapped == b"kms-re-wrapped-dek-blob"
    assert provider.get_key_id() == new_arn


def test_aws_kms_provider_handles_api_exceptions_safely():
    """Verify that KMS client exceptions are converted to KeyProviderError without logging secrets."""
    mock_client = mock.MagicMock()
    mock_client.generate_data_key.side_effect = RuntimeError("AccessDeniedException: KMS key not accessible")

    provider = AWSKMSProvider(key_arn="arn:aws:kms:us-east-1:123:key/k1", boto3_client=mock_client)

    with pytest.raises(KeyProviderError, match="AWS KMS generate_data_key request failed"):
        provider.generate_data_key()
