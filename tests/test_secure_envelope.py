import pytest
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from torchshield import AESGCMEnvelope, AuthenticationError, TensorAESGCMEnvelope


def test_aesgcm_envelope_round_trip_including_empty_message():
    key = AESGCMEnvelope.generate_key()
    for plaintext in (b"", b"hello TorchShield research", bytes(range(256)) * 4):
        envelope = AESGCMEnvelope.encrypt(plaintext, key, associated_data=b"model=toy-v1")
        assert AESGCMEnvelope.decrypt(envelope, key, associated_data=b"model=toy-v1") == plaintext


def test_aesgcm_envelope_uses_fresh_nonce():
    key = AESGCMEnvelope.generate_key()
    a = AESGCMEnvelope.encrypt(b"same plaintext", key)
    b = AESGCMEnvelope.encrypt(b"same plaintext", key)
    assert a != b
    assert a[:len(AESGCMEnvelope.HEADER)] == b[:len(AESGCMEnvelope.HEADER)]
    start = len(AESGCMEnvelope.HEADER)
    assert a[start:start + AESGCMEnvelope.NONCE_SIZE] != b[start:start + AESGCMEnvelope.NONCE_SIZE]


def test_tampering_wrong_key_and_wrong_aad_fail_authentication():
    key = AESGCMEnvelope.generate_key()
    wrong_key = AESGCMEnvelope.generate_key()
    envelope = AESGCMEnvelope.encrypt(b"sensitive synthetic test bytes", key, b"metadata")

    tampered = bytearray(envelope)
    tampered[-1] ^= 1
    for bad_envelope, key_arg, aad in (
        (bytes(tampered), key, b"metadata"),
        (envelope, wrong_key, b"metadata"),
        (envelope, key, b"different metadata"),
    ):
        with pytest.raises(AuthenticationError):
            AESGCMEnvelope.decrypt(bad_envelope, key_arg, aad)


def test_malformed_envelopes_fail_closed():
    key = AESGCMEnvelope.generate_key()
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(b"too short", key)
    with pytest.raises(AuthenticationError):
        AESGCMEnvelope.decrypt(b"BADMAGIC" + b"\x01" + bytes(40), key)


def test_aead_key_and_argument_validation():
    with pytest.raises(ValueError):
        AESGCMEnvelope.encrypt(b"x", b"short")
    with pytest.raises(TypeError):
        AESGCMEnvelope.encrypt("not bytes", AESGCMEnvelope.generate_key())
    with pytest.raises(TypeError):
        AESGCMEnvelope.decrypt(b"data", AESGCMEnvelope.generate_key(), associated_data="not bytes")


def test_envelope_is_interoperable_with_standard_aesgcm():
    key = AESGCMEnvelope.generate_key()
    aad = b"header metadata"
    plaintext = b"interoperability test payload"
    envelope = AESGCMEnvelope.encrypt(plaintext, key, aad)
    prefix_len = len(AESGCMEnvelope.HEADER)
    nonce = envelope[prefix_len:prefix_len + AESGCMEnvelope.NONCE_SIZE]
    ciphertext_and_tag = envelope[prefix_len + AESGCMEnvelope.NONCE_SIZE:]
    full_aad = AESGCMEnvelope.HEADER + nonce + aad
    assert AESGCM(key).decrypt(nonce, ciphertext_and_tag, full_aad) == plaintext


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
        torch.tensor([1.25, -2.5], dtype=torch.bfloat16),
        torch.tensor(3.14159, dtype=torch.float64),
        torch.empty((0, 3), dtype=torch.int64),
        torch.tensor([True, False, True], dtype=torch.bool),
        torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64),
    ],
)
def test_authenticated_tensor_envelope_round_trip(tensor):
    key = AESGCMEnvelope.generate_key()
    aad = b"asset=toy-model-weights"
    envelope = TensorAESGCMEnvelope.encrypt_tensor(tensor, key, associated_data=aad)
    restored = TensorAESGCMEnvelope.decrypt_tensor(envelope, key, associated_data=aad)
    assert restored.dtype == tensor.dtype
    assert tuple(restored.shape) == tuple(tensor.shape)
    assert torch.equal(restored, tensor.cpu())


def test_tensor_envelope_detects_tampering_and_wrong_metadata_aad():
    key = AESGCMEnvelope.generate_key()
    tensor = torch.arange(8, dtype=torch.int32)
    envelope = TensorAESGCMEnvelope.encrypt_tensor(tensor, key, associated_data=b"weights-v1")
    damaged = bytearray(envelope)
    damaged[-3] ^= 0x20
    with pytest.raises(AuthenticationError):
        TensorAESGCMEnvelope.decrypt_tensor(damaged, key, associated_data=b"weights-v1")
    with pytest.raises(AuthenticationError):
        TensorAESGCMEnvelope.decrypt_tensor(envelope, key, associated_data=b"weights-v2")
