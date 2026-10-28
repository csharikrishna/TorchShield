"""Tests for Post-Quantum Hybrid Cryptography Architecture (NIST FIPS 203 & 204)."""
import os
import secrets
import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from torchshield.pqc import (
    HybridKEMSession,
    HybridSignatureEngine,
    ReferencePQCBackend,
    PQCError,
    PQCUnavailableError,
    HYBRID_KEM_ALGORITHM_ID,
    HYBRID_SIG_ALGORITHM_ID,
)


def test_hybrid_kem_encapsulation_and_decapsulation_reference():
    session = HybridKEMSession(use_test_backend=True)

    # Generate classical recipient keypair
    recip_classical_priv = X25519PrivateKey.generate()
    recip_classical_pub = recip_classical_priv.public_key().public_bytes_raw()
    recip_pqc_priv, recip_pqc_pub = ReferencePQCBackend.kem_generate_keypair()

    dek = secrets.token_bytes(32)
    context = {"env": "prod", "workload": "llama-70b"}

    # Encapsulate
    envelope = session.encapsulate(
        classical_recipient_pub=recip_classical_pub,
        pqc_recipient_pub=recip_pqc_pub,
        dek=dek,
        recipient_id="cluster-node-1",
        encryption_context=context,
    )

    assert envelope["algorithm"] == HYBRID_KEM_ALGORITHM_ID
    assert "wrapped_dek" in envelope
    assert "pqc_ciphertext" in envelope

    # Decapsulate
    recovered_dek = session.decapsulate(
        envelope=envelope,
        classical_priv_bytes=recip_classical_priv.private_bytes_raw(),
        pqc_priv_bytes=recip_pqc_priv,
        encryption_context=context,
    )
    assert recovered_dek == dek


def test_hybrid_kem_tamper_and_fail_closed():
    session = HybridKEMSession(use_test_backend=True)
    recip_classical_priv = X25519PrivateKey.generate()
    recip_classical_pub = recip_classical_priv.public_key().public_bytes_raw()
    recip_pqc_priv, recip_pqc_pub = ReferencePQCBackend.kem_generate_keypair()

    dek = secrets.token_bytes(32)
    envelope = session.encapsulate(
        classical_recipient_pub=recip_classical_pub,
        pqc_recipient_pub=recip_pqc_pub,
        dek=dek,
    )

    # Tampered wrapped DEK fails closed
    tampered_env = dict(envelope)
    ct_bytes = bytearray(bytes.fromhex(tampered_env["wrapped_dek"]))
    ct_bytes[-1] ^= 0xFF
    tampered_env["wrapped_dek"] = ct_bytes.hex()

    with pytest.raises(PQCError):
        session.decapsulate(
            envelope=tampered_env,
            classical_priv_bytes=recip_classical_priv.private_bytes_raw(),
            pqc_priv_bytes=recip_pqc_priv,
        )


def test_hybrid_signatures_sign_and_verify_reference():
    sig_engine = HybridSignatureEngine(use_test_backend=True)

    ed_priv = Ed25519PrivateKey.generate()
    ed_priv_bytes = ed_priv.private_bytes_raw()
    ed_pub_bytes = ed_priv.public_key().public_bytes_raw()
    mldsa_priv = secrets.token_bytes(32)

    manifest_digest = secrets.token_bytes(32)
    sig_dict = sig_engine.sign(
        manifest_digest=manifest_digest,
        ed25519_priv_bytes=ed_priv_bytes,
        mldsa_priv_bytes=mldsa_priv,
    )

    assert sig_dict["algorithm"] == HYBRID_SIG_ALGORITHM_ID

    # Verification passes
    assert sig_engine.verify(
        manifest_digest=manifest_digest,
        signature_dict=sig_dict,
        expected_ed25519_pub=ed_pub_bytes,
    ) is True

    # Tampered digest fails
    with pytest.raises(PQCError):
        sig_engine.verify(
            manifest_digest=secrets.token_bytes(32),
            signature_dict=sig_dict,
            expected_ed25519_pub=ed_pub_bytes,
        )


def test_hybrid_signatures_tampered_pqc_component_fails():
    sig_engine = HybridSignatureEngine(use_test_backend=True)

    ed_priv = Ed25519PrivateKey.generate()
    ed_priv_bytes = ed_priv.private_bytes_raw()
    ed_pub_bytes = ed_priv.public_key().public_bytes_raw()
    mldsa_priv = secrets.token_bytes(32)

    manifest_digest = secrets.token_bytes(32)
    sig_dict = sig_engine.sign(
        manifest_digest=manifest_digest,
        ed25519_priv_bytes=ed_priv_bytes,
        mldsa_priv_bytes=mldsa_priv,
    )

    # Tamper only the PQC signature component
    tampered_sig_dict = dict(sig_dict)
    pqc_sig_bytes = bytearray(bytes.fromhex(tampered_sig_dict["mldsa_sig"]))
    pqc_sig_bytes[-1] ^= 0xFF
    tampered_sig_dict["mldsa_sig"] = pqc_sig_bytes.hex()

    with pytest.raises(PQCError, match="Post-quantum ML-DSA signature verification failed"):
        sig_engine.verify(
            manifest_digest=manifest_digest,
            signature_dict=tampered_sig_dict,
            expected_ed25519_pub=ed_pub_bytes,
        )


def test_production_mode_fails_closed_when_pqc_lib_missing():
    # In production mode without test harness, must fail closed if liboqs not installed
    session = HybridKEMSession(use_test_backend=False)
    recip_classical_priv = X25519PrivateKey.generate()
    recip_classical_pub = recip_classical_priv.public_key().public_bytes_raw()

    with pytest.raises((PQCUnavailableError, NotImplementedError)):
        session.encapsulate(
            classical_recipient_pub=recip_classical_pub,
            pqc_recipient_pub=secrets.token_bytes(32),
            dek=secrets.token_bytes(32),
        )
