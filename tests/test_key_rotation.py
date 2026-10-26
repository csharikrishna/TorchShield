"""Key rotation and migration verification suite."""
from __future__ import annotations

import secrets
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ahcc.key_providers.local import LocalDevKeyProvider


def test_dek_rewrap_under_rotated_kek():
    """Verify that a DEK can be rewrapped under a new KEK version without touching payload data."""
    kek_v1 = secrets.token_bytes(32)
    kek_v2 = secrets.token_bytes(32)

    provider_v1 = LocalDevKeyProvider(key_id="kek-v1", master_kek=kek_v1)
    provider_v2 = LocalDevKeyProvider(key_id="kek-v2", master_kek=kek_v2)

    context = {"model": "gpt-neo", "epoch": "5"}

    # 1. Generate DEK under KEK v1
    dek, wrapped_dek_v1 = provider_v1.generate_data_key(encryption_context=context)

    # 2. Encrypt a large payload with the DEK
    large_payload = b"NEURAL_NETWORK_WEIGHT_BYTES_" * 1024  # Simulated tensor bytes
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(dek).encrypt(nonce, large_payload, b"AAD_PAYLOAD")

    # 3. Rotate key: simulate rewrapping DEK from v1 to v2
    # In practice: unwrap with provider_v1, wrap with provider_v2
    unwrapped_dek = provider_v1.unwrap_key(wrapped_dek_v1, encryption_context=context)
    wrapped_dek_v2 = provider_v2._wrap_raw(unwrapped_dek, kek_v2, encryption_context=context)

    # 4. Old provider v1 cannot unwrap the new wrapped key
    with pytest.raises(Exception):
        provider_v1.unwrap_key(wrapped_dek_v2, encryption_context=context)

    # 5. New provider v2 unwraps DEK successfully
    recovered_dek = provider_v2.unwrap_key(wrapped_dek_v2, encryption_context=context)
    assert recovered_dek == dek

    # 6. Payload is decrypted using the recovered DEK without ever re-encrypting the payload
    recovered_payload = AESGCM(recovered_dek).decrypt(nonce, ciphertext, b"AAD_PAYLOAD")
    assert recovered_payload == large_payload
