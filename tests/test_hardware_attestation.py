"""Unit tests for Confidential Computing & Hardware Attestation Architecture."""
import hashlib
import secrets
import time
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from torchshield.attestation import (
    AttestationError,
    AttestationEvidence,
    AttestationPolicy,
    HardwareAttestationVerifier,
    PlatformType,
    AttestationGatedKeyProvider,
)
from torchshield.key_providers.local import LocalKeyProvider


@pytest.fixture
def signing_keys():
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes_raw()
    return priv, pub


def test_attestation_verification_success():
    verifier = HardwareAttestationVerifier()
    golden = hashlib.sha256(b"trusted_vmlinuz_image").hexdigest()
    nonce = secrets.token_bytes(32)

    evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=hashlib.sha256(nonce).hexdigest(),
        timestamp=time.time(),
    )
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        max_age_seconds=60.0,
    )

    assert verifier.verify(evidence, policy, expected_nonce=nonce) is True


def test_attestation_wrong_platform_fails_closed():
    verifier = HardwareAttestationVerifier()
    golden = hashlib.sha256(b"trusted_vmlinuz_image").hexdigest()
    nonce = secrets.token_bytes(32)

    evidence = AttestationEvidence(
        platform=PlatformType.GENERIC_TEE.value,
        measurement=golden,
        report_data=hashlib.sha256(nonce).hexdigest(),
        timestamp=time.time(),
    )
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value, PlatformType.AMD_SEV_SNP.value},
        golden_measurements={golden},
    )

    with pytest.raises(AttestationError, match="not permitted by policy"):
        verifier.verify(evidence, policy, expected_nonce=nonce)


def test_attestation_expired_quote_fails_closed():
    verifier = HardwareAttestationVerifier()
    golden = hashlib.sha256(b"trusted_vmlinuz_image").hexdigest()
    nonce = secrets.token_bytes(32)

    # 400 seconds ago
    evidence = AttestationEvidence(
        platform=PlatformType.AMD_SEV_SNP.value,
        measurement=golden,
        report_data=hashlib.sha256(nonce).hexdigest(),
        timestamp=time.time() - 400.0,
    )
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.AMD_SEV_SNP.value},
        golden_measurements={golden},
        max_age_seconds=300.0,
    )

    with pytest.raises(AttestationError, match="expired"):
        verifier.verify(evidence, policy, expected_nonce=nonce)


def test_attestation_nonce_replay_fails_closed():
    verifier = HardwareAttestationVerifier()
    golden = hashlib.sha256(b"trusted_vmlinuz_image").hexdigest()
    nonce_original = secrets.token_bytes(32)
    nonce_replayed = secrets.token_bytes(32)

    evidence = AttestationEvidence(
        platform=PlatformType.INTEL_TDX.value,
        measurement=golden,
        report_data=hashlib.sha256(nonce_original).hexdigest(),
        timestamp=time.time(),
    )
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.INTEL_TDX.value},
        golden_measurements={golden},
    )

    with pytest.raises(AttestationError, match="nonce mismatch"):
        verifier.verify(evidence, policy, expected_nonce=nonce_replayed)


def test_attestation_measurement_mismatch_fails_closed():
    verifier = HardwareAttestationVerifier()
    golden = hashlib.sha256(b"trusted_vmlinuz_image").hexdigest()
    tampered = hashlib.sha256(b"compromised_rootkit").hexdigest()
    nonce = secrets.token_bytes(32)

    evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=tampered,
        report_data=hashlib.sha256(nonce).hexdigest(),
        timestamp=time.time(),
    )
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
    )

    with pytest.raises(AttestationError, match="does not match any golden measurements"):
        verifier.verify(evidence, policy, expected_nonce=nonce)


def test_attestation_signature_verification(signing_keys):
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-ca-001"
    verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})

    golden = hashlib.sha256(b"trusted_kernel").hexdigest()
    nonce = secrets.token_bytes(32)
    timestamp = time.time()
    report_data = hashlib.sha256(nonce).hexdigest()

    # Create signature
    body = f"{PlatformType.NVIDIA_CC.value}:{golden}:{report_data}:{timestamp:.3f}".encode("utf-8")
    sig = priv_key.sign(body).hex()

    evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=report_data,
        timestamp=timestamp,
        signature=sig,
        signer_id=signer_id,
    )
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={signer_id},
    )

    assert verifier.verify(evidence, policy, expected_nonce=nonce) is True

    # Tampered signature
    tampered_sig = (bytes.fromhex(sig)[:-1] + b"\x00").hex()
    bad_evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=report_data,
        timestamp=timestamp,
        signature=tampered_sig,
        signer_id=signer_id,
    )
    with pytest.raises(AttestationError, match="signature verification failed"):
        verifier.verify(bad_evidence, policy, expected_nonce=nonce)


def test_attestation_gated_key_provider_e2e():
    inner_provider = LocalKeyProvider()
    verifier = HardwareAttestationVerifier()
    golden = hashlib.sha256(b"confidential_app").hexdigest()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
    )

    # Evidence generator callback responding to challenge nonces
    def valid_evidence_provider(challenge_nonce: bytes) -> AttestationEvidence:
        return AttestationEvidence(
            platform=PlatformType.NVIDIA_CC.value,
            measurement=golden,
            report_data=hashlib.sha256(challenge_nonce).hexdigest(),
            timestamp=time.time(),
        )

    gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        evidence_provider=valid_evidence_provider,
    )

    # 1. Key generation and wrapping
    dek, wrapped = gated_provider.generate_data_key()
    assert len(dek) == 32
    assert len(wrapped) > 32

    # 2. Key unwrapping succeeds under valid attestation
    unwrapped = gated_provider.unwrap_key(wrapped)
    assert unwrapped == dek

    # 3. Failing attestation gates unwrap
    def malicious_evidence_provider(challenge_nonce: bytes) -> AttestationEvidence:
        return AttestationEvidence(
            platform=PlatformType.NVIDIA_CC.value,
            measurement="deadbeef" * 8,
            report_data=hashlib.sha256(challenge_nonce).hexdigest(),
            timestamp=time.time(),
        )

    tampered_gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        evidence_provider=malicious_evidence_provider,
    )

    with pytest.raises(AttestationError, match="does not match any golden measurements"):
        tampered_gated_provider.unwrap_key(wrapped)
