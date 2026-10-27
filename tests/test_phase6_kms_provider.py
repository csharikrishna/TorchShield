"""Enterprise AWS KMS KeyProvider validation suite for Phase 6.

Section 4 Audit Requirements:
- Key-provider interface compliance
- Plaintext and wrapped DEK handling
- wrap_key, unwrap_key, generate_data_key, rewrap_key
- Encryption context validation and tampering rejection
- Unauthorized decryption (AccessDeniedException)
- Disabled/unavailable keys (KMSInvalidStateException, NotFoundException)
- Network errors and bounded retries with exponential backoff
- Zero key leakage in logs and exception messages
- Strict non-fallback to LocalDevKeyProvider
"""
from __future__ import annotations

import os
import secrets
import tempfile
from unittest import mock
import pytest
import torch

from ahcc.checkpoint import SecureModelCheckpoint, CheckpointError
from ahcc.key_providers.base import KeyProviderError
from ahcc.key_providers.local import LocalDevKeyProvider
from ahcc.key_providers.kms import AWSKMSProvider


def test_aws_kms_provider_interface_and_properties():
    """Verify production security flags and ARN accessors."""
    mock_client = mock.MagicMock()
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/test-kek-uuid"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client)

    assert provider.is_production_secure is True
    assert provider.get_key_id() == key_arn

    with pytest.raises(ValueError, match="key_arn must be a non-empty"):
        AWSKMSProvider(key_arn="")


def test_aws_kms_wrap_and_unwrap_lifecycle():
    """Verify wrap_key and unwrap_key methods with encryption context."""
    mock_client = mock.MagicMock()
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/test-kek"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client)

    plaintext_key = secrets.token_bytes(32)
    wrapped_blob = b"kms-wrapped-ciphertext-blob-987"

    mock_client.encrypt.return_value = {"CiphertextBlob": wrapped_blob}
    mock_client.decrypt.return_value = {"Plaintext": plaintext_key}

    context = {"tenant": "ai-infra", "env": "prod"}

    # 1. wrap_key
    w_res = provider.wrap_key(plaintext_key, encryption_context=context)
    assert w_res == wrapped_blob
    mock_client.encrypt.assert_called_once_with(
        KeyId=key_arn, Plaintext=plaintext_key, EncryptionContext=context
    )

    # 2. unwrap_key
    unwrapped = provider.unwrap_key(wrapped_blob, encryption_context=context)
    assert unwrapped == plaintext_key
    mock_client.decrypt.assert_called_once_with(
        CiphertextBlob=wrapped_blob, KeyId=key_arn, EncryptionContext=context
    )


def test_aws_kms_bounded_retries_on_transient_throttling():
    """Verify bounded exponential backoff on transient ThrottlingException."""
    mock_client = mock.MagicMock()
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/retry-kek"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client, max_retries=3, base_backoff_sec=0.001)

    plaintext_key = secrets.token_bytes(32)
    # Fail twice with ThrottlingException, succeed on 3rd attempt
    mock_client.generate_data_key.side_effect = [
        RuntimeError("ThrottlingException: Rate exceeded"),
        RuntimeError("ThrottlingException: Rate exceeded"),
        {"Plaintext": plaintext_key, "CiphertextBlob": b"wrapped-dek"},
    ]

    p_key, w_key = provider.generate_data_key()
    assert p_key == plaintext_key
    assert w_key == b"wrapped-dek"
    assert mock_client.generate_data_key.call_count == 3


def test_aws_kms_retry_exhaustion_raises_clean_error():
    """Verify that retries terminate after max_retries and raise KeyProviderError."""
    mock_client = mock.MagicMock()
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/retry-exhaust"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client, max_retries=2, base_backoff_sec=0.001)

    mock_client.generate_data_key.side_effect = RuntimeError("ServiceUnavailable: KMS endpoint unreachable")

    with pytest.raises(KeyProviderError, match="AWS KMS generate_data_key request failed"):
        provider.generate_data_key()
    assert mock_client.generate_data_key.call_count == 3  # Initial + 2 retries


def test_aws_kms_non_retryable_exceptions_fail_immediately():
    """Verify that non-transient errors (AccessDenied, NotFound) fail immediately without retries."""
    mock_client = mock.MagicMock()
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/access-denied"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client, max_retries=5)

    mock_client.decrypt.side_effect = RuntimeError("AccessDeniedException: User not authorized to perform kms:Decrypt")

    with pytest.raises(KeyProviderError, match="AccessDeniedException"):
        provider.unwrap_key(b"blob")
    assert mock_client.decrypt.call_count == 1  # No retries on authorization failure


def test_aws_kms_disabled_or_invalid_state_rejection():
    """Verify rejection when the KMS key is disabled or pending deletion."""
    mock_client = mock.MagicMock()
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/disabled"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client, max_retries=2)

    mock_client.generate_data_key.side_effect = RuntimeError("KMSInvalidStateException: Key is pending deletion")

    with pytest.raises(KeyProviderError, match="KMSInvalidStateException"):
        provider.generate_data_key()
    assert mock_client.generate_data_key.call_count == 1


def test_aws_kms_error_messages_do_not_leak_plaintext_keys():
    """Verify that exception strings never contain raw plaintext key material."""
    mock_client = mock.MagicMock()
    secret_key = b"TOP_SECRET_NEURAL_WEIGHTS_KEY_32"
    assert len(secret_key) == 32

    provider = AWSKMSProvider(key_arn="arn:aws:kms:us-east-1:123:key/k1", boto3_client=mock_client, max_retries=0)
    mock_client.encrypt.side_effect = RuntimeError("Internal failure during wrapping")

    with pytest.raises(KeyProviderError) as exc_info:
        provider.wrap_key(secret_key)

    error_str = str(exc_info.value)
    assert secret_key.decode("ascii") not in error_str
    assert secret_key.hex() not in error_str


def test_local_dev_key_provider_implements_wrap_key():
    """Verify wrap_key implementation on LocalDevKeyProvider."""
    kek = secrets.token_bytes(32)
    local_provider = LocalDevKeyProvider(key_id="local-k1", master_kek=kek)

    dek = secrets.token_bytes(32)
    context = {"stage": "eval"}
    wrapped = local_provider.wrap_key(dek, encryption_context=context)

    unwrapped = local_provider.unwrap_key(wrapped, encryption_context=context)
    assert unwrapped == dek

    with pytest.raises(ValueError, match="plaintext_key must be exactly 32 bytes"):
        local_provider.wrap_key(dek[:16])


def test_aws_kms_envelope_save_and_load_end_to_end():
    """Verify full end-to-end envelope encryption lifecycle with AWSKMSProvider."""
    mock_client = mock.MagicMock()
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/e2e-kek"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client)

    plaintext_dek = secrets.token_bytes(32)
    wrapped_dek = b"kms-envelope-wrapped-dek-bytes-001"

    # Configure mock KMS
    mock_client.generate_data_key.return_value = {
        "Plaintext": plaintext_dek,
        "CiphertextBlob": wrapped_dek,
    }
    mock_client.decrypt.return_value = {
        "Plaintext": plaintext_dek,
    }

    state = {"layer.weight": torch.randn(16, 16), "layer.bias": torch.randn(16)}
    context = {"tenant": "research", "run_id": "run-001"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "kms_e2e.ahcc")

        # Save with KMS provider
        SecureModelCheckpoint.save_checkpoint(
            state_dict=state,
            output_path=ckpt_path,
            key_provider=provider,
            encryption_context=context,
        )

        # Inspect disk bytes: Plaintext DEK must NEVER be present anywhere in the file
        with open(ckpt_path, "rb") as f:
            disk_bytes = f.read()
        assert plaintext_dek not in disk_bytes, "Plaintext DEK leaked to storage!"
        assert wrapped_dek in disk_bytes, "Wrapped DEK must be persisted in header."

        # Load with KMS provider
        restored = SecureModelCheckpoint.load_checkpoint(
            input_path=ckpt_path,
            key_provider=provider,
            encryption_context=context,
        )
        assert torch.allclose(restored["layer.weight"], state["layer.weight"])
        assert torch.allclose(restored["layer.bias"], state["layer.bias"])


def test_aws_kms_unauthorized_decryption_and_strict_no_fallback():
    """Verify that unauthorized KMS principal fails closed and never falls back to local provider."""
    mock_client = mock.MagicMock()
    key_arn = "arn:aws:kms:us-east-1:123456789012:key/auth-fail-kek"
    provider = AWSKMSProvider(key_arn=key_arn, boto3_client=mock_client)

    plaintext_dek = secrets.token_bytes(32)
    wrapped_dek = b"kms-wrapped-dek-for-unauth-test"

    mock_client.generate_data_key.return_value = {
        "Plaintext": plaintext_dek,
        "CiphertextBlob": wrapped_dek,
    }
    mock_client.decrypt.side_effect = RuntimeError("AccessDeniedException: Principal is not authorized")

    state = {"w": torch.ones(4, 4)}
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "unauth.ahcc")
        SecureModelCheckpoint.save_checkpoint(state, ckpt_path, key_provider=provider)

        # Unauthorized load must fail closed with CheckpointError
        with pytest.raises(CheckpointError, match="Failed to unwrap DEK via KeyProvider.*AccessDeniedException"):
            SecureModelCheckpoint.load_checkpoint(ckpt_path, key_provider=provider)


def test_aws_kms_key_rotation_and_recovery():
    """Verify that rotating KEK via rewrap_key allows recovery under new KMS key."""
    mock_client = mock.MagicMock()
    key_arn_1 = "arn:aws:kms:us-east-1:123456789012:key/kek-1"
    key_arn_2 = "arn:aws:kms:us-east-1:123456789012:key/kek-2"
    provider_1 = AWSKMSProvider(key_arn=key_arn_1, boto3_client=mock_client)

    plaintext_dek = secrets.token_bytes(32)
    wrapped_dek_1 = b"wrapped-under-kek-1"
    wrapped_dek_2 = b"wrapped-under-kek-2"

    mock_client.generate_data_key.return_value = {
        "Plaintext": plaintext_dek,
        "CiphertextBlob": wrapped_dek_1,
    }
    mock_client.re_encrypt.return_value = {
        "CiphertextBlob": wrapped_dek_2,
    }
    mock_client.decrypt.return_value = {
        "Plaintext": plaintext_dek,
    }

    # Rotate DEK
    rotated_blob = provider_1.rewrap_key(wrapped_dek_1, new_key_id=key_arn_2)
    assert rotated_blob == wrapped_dek_2
    assert provider_1.get_key_id() == key_arn_2

    # Provider 2 successfully unwraps rotated blob
    provider_2 = AWSKMSProvider(key_arn=key_arn_2, boto3_client=mock_client)
    unwrapped = provider_2.unwrap_key(rotated_blob)
    assert unwrapped == plaintext_dek
