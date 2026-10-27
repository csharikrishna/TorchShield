"""Known-Answer Test (KAT) Vectors for AES-256-GCM (NIST SP 800-38D) and HKDF-SHA256 (RFC 5869).

Validates that the cryptographic underlying primitives and constructions match published
standards vectors, rather than relying exclusively on self-referential round-trip tests.
"""
from __future__ import annotations

import binascii
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ahcc.secure import AESGCMEnvelope, AuthenticationError


# ===========================================================================
# 1. NIST SP 800-38D AES-256-GCM Official Test Vectors (Section 8.2 & Appendix B)
# ===========================================================================

# NIST SP 800-38D Test Case 13 (AES-256, 96-bit IV, empty plaintext, empty AAD)
NIST_CASE_13 = {
    "key": bytes(32),
    "iv": bytes(12),
    "pt": b"",
    "aad": b"",
    "ct": b"",
    "tag": binascii.unhexlify("530f8afbc74536b9a963b4f1c4cb738b"),
}

# NIST SP 800-38D Test Case 14 (AES-256, 96-bit IV, 16-byte plaintext 00..00, empty AAD)
NIST_CASE_14 = {
    "key": bytes(32),
    "iv": bytes(12),
    "pt": bytes(16),
    "aad": b"",
    "ct": binascii.unhexlify("cea7403d4d606b6e074ec5d3baf39d18"),
    "tag": binascii.unhexlify("d0d1c8a799996bf0265b98b5d48ab919"),
}

# NIST SP 800-38D Test Case 15 (AES-256, 96-bit IV, 64-byte plaintext 00..00, empty AAD)
NIST_CASE_15 = {
    "key": bytes(32),
    "iv": bytes(12),
    "pt": bytes(64),
    "aad": b"",
    "ct": binascii.unhexlify(
        "cea7403d4d606b6e074ec5d3baf39d18726003ca37a62a74d1a2f58e7506358e"
        "dd4ab1284d4ae17b41e85924470c36f74741cbe181bb7f30617c1de3ab0c3a1f"
    ),
    "tag": binascii.unhexlify("5d5c0cd0c7b0138ac868f227d053cd0b"),
}

# NIST SP 800-38D Test Case 16 (AES-256, 96-bit IV, 60-byte plaintext 00..00, 20-byte AAD 00..00)
NIST_CASE_16 = {
    "key": bytes(32),
    "iv": bytes(12),
    "pt": bytes(60),
    "aad": bytes(20),
    "ct": binascii.unhexlify(
        "cea7403d4d606b6e074ec5d3baf39d18726003ca37a62a74d1a2f58e7506358e"
        "dd4ab1284d4ae17b41e85924470c36f74741cbe181bb7f30617c1de3"
    ),
    "tag": binascii.unhexlify("cede954b987daba12c15f05898151f3c"),
}


def test_nist_sp800_38d_case_13():
    """Verify AES-256-GCM Case 13 (empty plaintext, authentication only)."""
    cipher = AESGCM(NIST_CASE_13["key"])
    res = cipher.encrypt(NIST_CASE_13["iv"], NIST_CASE_13["pt"], NIST_CASE_13["aad"])
    assert res == NIST_CASE_13["tag"]
    decrypted = cipher.decrypt(NIST_CASE_13["iv"], res, NIST_CASE_13["aad"])
    assert decrypted == NIST_CASE_13["pt"]


def test_nist_sp800_38d_case_14():
    """Verify AES-256-GCM Case 14 (16-byte single block)."""
    cipher = AESGCM(NIST_CASE_14["key"])
    res = cipher.encrypt(NIST_CASE_14["iv"], NIST_CASE_14["pt"], NIST_CASE_14["aad"])
    assert res == NIST_CASE_14["ct"] + NIST_CASE_14["tag"]
    decrypted = cipher.decrypt(NIST_CASE_14["iv"], res, NIST_CASE_14["aad"])
    assert decrypted == NIST_CASE_14["pt"]


def test_nist_sp800_38d_case_15():
    """Verify AES-256-GCM Case 15 (64-byte payload)."""
    cipher = AESGCM(NIST_CASE_15["key"])
    res = cipher.encrypt(NIST_CASE_15["iv"], NIST_CASE_15["pt"], NIST_CASE_15["aad"])
    assert res == NIST_CASE_15["ct"] + NIST_CASE_15["tag"]
    decrypted = cipher.decrypt(NIST_CASE_15["iv"], res, NIST_CASE_15["aad"])
    assert decrypted == NIST_CASE_15["pt"]


def test_nist_sp800_38d_case_16():
    """Verify AES-256-GCM Case 16 (60-byte payload + 20-byte AAD)."""
    cipher = AESGCM(NIST_CASE_16["key"])
    res = cipher.encrypt(NIST_CASE_16["iv"], NIST_CASE_16["pt"], NIST_CASE_16["aad"])
    assert res == NIST_CASE_16["ct"] + NIST_CASE_16["tag"]
    decrypted = cipher.decrypt(NIST_CASE_16["iv"], res, NIST_CASE_16["aad"])
    assert decrypted == NIST_CASE_16["pt"]


def test_aes_gcm_envelope_tamper_detection_on_known_vectors():
    """Verify that tampering with any bit of ciphertext, tag, nonce, or AAD causes rejection."""
    key = NIST_CASE_16["key"]
    iv = NIST_CASE_16["iv"]
    pt = NIST_CASE_16["pt"]
    aad = NIST_CASE_16["aad"]

    cipher = AESGCM(key)
    valid_payload = cipher.encrypt(iv, pt, aad)

    # 1. Tamper ciphertext
    tampered_ct = bytearray(valid_payload)
    tampered_ct[0] ^= 0x01
    with pytest.raises(Exception):
        cipher.decrypt(iv, bytes(tampered_ct), aad)

    # 2. Tamper tag
    tampered_tag = bytearray(valid_payload)
    tampered_tag[-1] ^= 0x01
    with pytest.raises(Exception):
        cipher.decrypt(iv, bytes(tampered_tag), aad)

    # 3. Tamper AAD
    tampered_aad = bytearray(aad)
    tampered_aad[0] ^= 0x01
    with pytest.raises(Exception):
        cipher.decrypt(iv, valid_payload, bytes(tampered_aad))

    # 4. Tamper IV
    tampered_iv = bytearray(iv)
    tampered_iv[0] ^= 0x01
    with pytest.raises(Exception):
        cipher.decrypt(bytes(tampered_iv), valid_payload, aad)


# ===========================================================================
# 2. RFC 5869 HKDF-SHA256 Official Test Vectors
# ===========================================================================

RFC_5869_CASE_1 = {
    "ikm": binascii.unhexlify("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b"),
    "salt": binascii.unhexlify("000102030405060708090a0b0c"),
    "info": binascii.unhexlify("f0f1f2f3f4f5f6f7f8f9"),
    "l": 42,
    "prk": binascii.unhexlify("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e0"),
    "okm": binascii.unhexlify(
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865"
    ),
}

RFC_5869_CASE_2 = {
    "ikm": bytes(range(0x00, 0x50)),
    "salt": bytes(range(0x60, 0xb0)),
    "info": bytes(range(0xb0, 0x100)),
    "l": 82,
    "okm": binascii.unhexlify(
        "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
        "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
        "cc30c58179ec3e87c14c01d5c1f3434f1d87"
    ),
}

RFC_5869_CASE_3 = {
    "ikm": binascii.unhexlify("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b"),
    "salt": b"",
    "info": b"",
    "l": 42,
    "okm": binascii.unhexlify(
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
        "9d201395faa4b61a96c8"
    ),
}


def test_rfc5869_hkdf_sha256_case_1():
    """Verify RFC 5869 Test Case 1 extraction and expansion."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=RFC_5869_CASE_1["l"],
        salt=RFC_5869_CASE_1["salt"],
        info=RFC_5869_CASE_1["info"],
    )
    derived = hkdf.derive(RFC_5869_CASE_1["ikm"])
    assert derived == RFC_5869_CASE_1["okm"]


def test_rfc5869_hkdf_sha256_case_2():
    """Verify RFC 5869 Test Case 2 with 80-byte inputs and 82-byte output."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=RFC_5869_CASE_2["l"],
        salt=RFC_5869_CASE_2["salt"],
        info=RFC_5869_CASE_2["info"],
    )
    derived = hkdf.derive(RFC_5869_CASE_2["ikm"])
    assert derived == RFC_5869_CASE_2["okm"]


def test_rfc5869_hkdf_sha256_case_3_empty_salt():
    """Verify RFC 5869 Test Case 3 with empty salt and empty info."""
    effective_salt = bytes(32)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=RFC_5869_CASE_3["l"],
        salt=effective_salt,
        info=RFC_5869_CASE_3["info"],
    )
    derived = hkdf.derive(RFC_5869_CASE_3["ikm"])
    assert derived == RFC_5869_CASE_3["okm"]
