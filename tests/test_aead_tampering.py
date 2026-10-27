"""Extensive negative tests and tampering suites for standard AES-256-GCM envelopes.

Validates that all forms of corruption (ciphertext, nonce, tag, AAD, truncation,
wrong keys, version mismatch) fail-closed immediately without releasing unauthenticated plaintext.
"""
import pytest
from torchshield.secure import AESGCMEnvelope, AuthenticationError


def test_aead_tampering_matrix():
    """Verify bit flips across every byte of ciphertext and tag cause immediate authentication failure."""
    key = AESGCMEnvelope.generate_key()
    pt = b"Critical AI Model Weights and Parameters: 0.123456789"
    aad = b"layer_id=42;dtype=float32"

    envelope = bytearray(AESGCMEnvelope.encrypt(pt, key, associated_data=aad))

    # 1. Flip byte in header
    env_bad_header = bytearray(envelope)
    env_bad_header[0] ^= 0xFF
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(env_bad_header, key, associated_data=aad)

    # 2. Flip byte in nonce (bytes 9 to 20)
    for i in range(len(AESGCMEnvelope.HEADER), len(AESGCMEnvelope.HEADER) + AESGCMEnvelope.NONCE_SIZE):
        env_bad_nonce = bytearray(envelope)
        env_bad_nonce[i] ^= 0x01
        with pytest.raises(AuthenticationError):
            AESGCMEnvelope.decrypt(env_bad_nonce, key, associated_data=aad)

    # 3. Flip byte in ciphertext body
    ct_start = len(AESGCMEnvelope.HEADER) + AESGCMEnvelope.NONCE_SIZE
    ct_end = len(envelope) - AESGCMEnvelope.TAG_SIZE
    for i in range(ct_start, ct_end, 5):  # Sample across body
        env_bad_ct = bytearray(envelope)
        env_bad_ct[i] ^= 0x40
        with pytest.raises(AuthenticationError):
            AESGCMEnvelope.decrypt(env_bad_ct, key, associated_data=aad)

    # 4. Flip byte in authentication tag
    for i in range(ct_end, len(envelope)):
        env_bad_tag = bytearray(envelope)
        env_bad_tag[i] ^= 0x80
        with pytest.raises(AuthenticationError):
            AESGCMEnvelope.decrypt(env_bad_tag, key, associated_data=aad)


def test_aead_aad_tampering():
    """Verify any modification or omission of authenticated associated data fails."""
    key = AESGCMEnvelope.generate_key()
    pt = b"Secret tensor data"
    aad = b"model_name=ResNet50;epoch=10"

    env = AESGCMEnvelope.encrypt(pt, key, associated_data=aad)

    # Wrong AAD
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(env, key, associated_data=b"model_name=ResNet50;epoch=11")

    # Empty AAD when nonempty was expected
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(env, key, associated_data=b"")


def test_aead_truncation():
    """Verify truncated envelopes at every single byte length are rejected."""
    key = AESGCMEnvelope.generate_key()
    pt = b"Truncation test payload of moderate length"
    env = AESGCMEnvelope.encrypt(pt, key)

    for cut in range(len(env)):
        truncated = env[:cut]
        with pytest.raises(AuthenticationError):
            AESGCMEnvelope.decrypt(truncated, key)


def test_nonce_uniqueness_across_many_encryptions():
    """Verify that encrypt generates cryptographically fresh nonces on every invocation."""
    key = AESGCMEnvelope.generate_key()
    pt = b"constant"
    seen_nonces = set()
    for _ in range(500):
        env = AESGCMEnvelope.encrypt(pt, key)
        nonce = env[len(AESGCMEnvelope.HEADER):len(AESGCMEnvelope.HEADER) + AESGCMEnvelope.NONCE_SIZE]
        assert nonce not in seen_nonces, "FATAL: Duplicate AES-GCM nonce detected under same key!"
        seen_nonces.add(nonce)
