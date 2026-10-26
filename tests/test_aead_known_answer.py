"""Known-Answer Tests (KAT) for RFC 5869 HKDF-SHA256 and NIST SP 800-38D AES-256-GCM.

Verifies our cryptographic foundation against official published test vectors.
"""
import binascii
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def test_rfc5869_hkdf_sha256_test_case_1():
    """RFC 5869 Appendix A.1 Test Case 1."""
    ikm = bytes.fromhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b")
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    length = 42

    expected_prk = "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e0"
    expected_okm = "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"

    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
    okm = hkdf.derive(ikm)
    assert okm.hex() == expected_okm


def test_rfc5869_hkdf_sha256_test_case_2():
    """RFC 5869 Appendix A.2 Test Case 2 (longer inputs)."""
    ikm = bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f")
    salt = bytes.fromhex("606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeaf")
    info = bytes.fromhex("b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
    length = 82

    expected_okm = (
        "b11e398dc80327a1c8e7f78c596a4934"
        "4f012eda2d4efad8a050cc4c19afa97c"
        "59045a99cac7827271cb41c65e590e09"
        "da3275600c2f09b8367793a9aca3db71"
        "cc30c58179ec3e87c14c01d5c1f3434f"
        "1d87"
    )

    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
    okm = hkdf.derive(ikm)
    assert okm.hex() == expected_okm


def test_nist_sp800_38d_aes256_gcm_case13():
    """NIST SP 800-38D AES-GCM 256-bit Key Case 13 (0-byte PT, 0-byte AAD)."""
    key = bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000")
    iv = bytes.fromhex("000000000000000000000000")
    pt = b""
    aad = b""
    expected_ct_with_tag = "530f8afbc74536b9a963b4f1c4cb738b"

    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(iv, pt, aad)
    assert ct.hex() == expected_ct_with_tag
    assert aesgcm.decrypt(iv, ct, aad) == pt


def test_nist_sp800_38d_aes256_gcm_case14():
    """NIST SP 800-38D AES-GCM 256-bit Key Case 14 (16-byte PT, 0-byte AAD)."""
    key = bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000")
    iv = bytes.fromhex("000000000000000000000000")
    pt = bytes.fromhex("00000000000000000000000000000000")
    aad = b""
    expected_ct = "cea7403d4d606b6e074ec5d3baf39d18"
    expected_tag = "d0d1c8a799996bf0265b98b5d48ab919"

    aesgcm = AESGCM(key)
    res = aesgcm.encrypt(iv, pt, aad)
    assert res[:16].hex() == expected_ct
    assert res[16:].hex() == expected_tag
    assert aesgcm.decrypt(iv, res, aad) == pt


def test_nist_sp800_38d_aes256_gcm_case16():
    """NIST SP 800-38D AES-GCM 256-bit Key Case 16 (60-byte PT, 20-byte AAD)."""
    key = bytes.fromhex("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308")
    iv = bytes.fromhex("cafebabefacedbaddecaf888")
    pt = bytes.fromhex("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39")
    aad = bytes.fromhex("feedfacedeadbeeffeedfacedeadbeefabaddadb")
    expected_ct = "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662"
    expected_tag = "51d082086f22a4aae2a5d550f75aa97c"

    aesgcm = AESGCM(key)
    res = aesgcm.encrypt(iv, pt, aad)
    assert res[:-16].hex() == expected_ct
    assert res[-16:].hex() == expected_tag
    assert aesgcm.decrypt(iv, res, aad) == pt
