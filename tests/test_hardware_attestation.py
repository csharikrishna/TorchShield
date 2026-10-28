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


def test_attestation_verification_success(signing_keys):
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})
    golden = hashlib.sha256(b"trusted_vmlinuz_image").hexdigest()
    nonce = secrets.token_bytes(32)
    timestamp = time.time()
    report_data = hashlib.sha256(nonce).hexdigest()

    quote_body = f"{PlatformType.NVIDIA_CC.value}:{golden}:{report_data}:{timestamp:.3f}".encode("utf-8")
    sig = priv_key.sign(quote_body).hex()

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
        max_age_seconds=60.0,
        allowed_signer_ids={signer_id},
        require_signature=True,
    )

    assert verifier.verify(evidence, policy, expected_nonce=nonce) is True


def test_attestation_missing_signature_fails_closed():
    verifier = HardwareAttestationVerifier()
    golden = hashlib.sha256(b"trusted_vmlinuz_image").hexdigest()
    nonce = secrets.token_bytes(32)

    evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=hashlib.sha256(nonce).hexdigest(),
        timestamp=time.time(),
        signature=None,
        signer_id=None,
    )
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        max_age_seconds=60.0,
        require_signature=True,
    )

    with pytest.raises(AttestationError, match="lacks required platform signature or signer_id"):
        verifier.verify(evidence, policy, expected_nonce=nonce)


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


def test_attestation_gated_key_provider_e2e(signing_keys):
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-ca-001"
    inner_provider = LocalKeyProvider()
    verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})
    golden = hashlib.sha256(b"confidential_app").hexdigest()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={signer_id},
        require_signature=True,
    )

    # Evidence generator callback responding to challenge nonces
    def valid_evidence_provider(challenge_nonce: bytes) -> AttestationEvidence:
        ts = time.time()
        rd = hashlib.sha256(challenge_nonce).hexdigest()
        body = f"{PlatformType.NVIDIA_CC.value}:{golden}:{rd}:{ts:.3f}".encode("utf-8")
        sig = priv_key.sign(body).hex()
        return AttestationEvidence(
            platform=PlatformType.NVIDIA_CC.value,
            measurement=golden,
            report_data=rd,
            timestamp=ts,
            signature=sig,
            signer_id=signer_id,
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
        ts = time.time()
        rd = hashlib.sha256(challenge_nonce).hexdigest()
        bad_measurement = "deadbeef" * 8
        body = f"{PlatformType.NVIDIA_CC.value}:{bad_measurement}:{rd}:{ts:.3f}".encode("utf-8")
        sig = priv_key.sign(body).hex()
        return AttestationEvidence(
            platform=PlatformType.NVIDIA_CC.value,
            measurement=bad_measurement,
            report_data=rd,
            timestamp=ts,
            signature=sig,
            signer_id=signer_id,
        )

    tampered_gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        evidence_provider=malicious_evidence_provider,
    )

    with pytest.raises(AttestationError, match="does not match any golden measurements"):
        tampered_gated_provider.unwrap_key(wrapped)


def test_attestation_gated_key_provider_production_security_classification(signing_keys):
    priv_key, pub_bytes = signing_keys
    signer_id = "prod-root-001"
    golden = hashlib.sha256(b"prod_image").hexdigest()

    class MockProdSecureKeyProvider(LocalKeyProvider):
        @property
        def is_production_secure(self) -> bool:
            return True

    prod_inner = MockProdSecureKeyProvider()
    unsecure_inner = LocalKeyProvider()  # is_production_secure is False

    valid_verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})
    valid_policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={signer_id},
        require_signature=True,
        require_fresh_nonce=True,
        allow_mock=False,
    )

    # 1. Base valid provider is production secure
    gated = AttestationGatedKeyProvider(prod_inner, valid_verifier, valid_policy)
    assert gated.is_production_secure is True

    # 2. Insecure inner provider -> False
    gated_insecure_inner = AttestationGatedKeyProvider(unsecure_inner, valid_verifier, valid_policy)
    assert gated_insecure_inner.is_production_secure is False

    # 3. Verifier allow_mock -> False
    mock_verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes}, allow_mock=True)
    assert AttestationGatedKeyProvider(prod_inner, mock_verifier, valid_policy).is_production_secure is False

    # 4. Policy allow_mock -> False
    mock_policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allow_mock=True,
    )
    assert AttestationGatedKeyProvider(prod_inner, valid_verifier, mock_policy).is_production_secure is False

    # 5. Missing trusted root keys in verifier -> False
    empty_verifier = HardwareAttestationVerifier(trusted_root_public_keys={})
    assert AttestationGatedKeyProvider(prod_inner, empty_verifier, valid_policy).is_production_secure is False

    # 6. Unknown signer ID in policy not present in trusted keys -> False
    unknown_signer_policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={"unknown-untrusted-signer"},
    )
    assert AttestationGatedKeyProvider(prod_inner, valid_verifier, unknown_signer_policy).is_production_secure is False

    # 7. Signature requirement disabled -> False
    no_sig_policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        require_signature=False,
    )
    assert AttestationGatedKeyProvider(prod_inner, valid_verifier, no_sig_policy).is_production_secure is False


def test_attestation_stage_evidence_with_bound_challenge(signing_keys):
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = LocalKeyProvider()
    verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})
    golden = hashlib.sha256(b"confidential_app").hexdigest()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={signer_id},
        require_signature=True,
        require_fresh_nonce=True,
    )

    gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        evidence_provider=None,
    )

    dek, wrapped = gated_provider.generate_data_key()

    # Create fresh challenge from provider
    challenge = gated_provider.create_challenge()
    ts = time.time()
    rd = hashlib.sha256(challenge).hexdigest()
    body = f"{PlatformType.NVIDIA_CC.value}:{golden}:{rd}:{ts:.3f}".encode("utf-8")
    sig = priv_key.sign(body).hex()

    evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=rd,
        timestamp=ts,
        signature=sig,
        signer_id=signer_id,
    )

    # Stage evidence bound to challenge
    gated_provider.stage_evidence(evidence, challenge=challenge)

    # Unwrap succeeds and consumes staged evidence
    unwrapped = gated_provider.unwrap_key(wrapped)
    assert unwrapped == dek

    # Subsequent unwrap fails because staged evidence was consumed (anti-replay)
    with pytest.raises(AttestationError, match="No attestation evidence available"):
        gated_provider.unwrap_key(wrapped)


def test_attestation_stage_evidence_without_challenge_fails_closed(signing_keys):
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = LocalKeyProvider()
    verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})
    golden = hashlib.sha256(b"confidential_app").hexdigest()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        require_fresh_nonce=True,
    )

    gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        evidence_provider=None,
    )

    dek, wrapped = gated_provider.generate_data_key()

    # Evidence created with some arbitrary report data, but NOT bound to challenge
    evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data="unbound_nonce_data",
        timestamp=time.time(),
        signature="deadbeef" * 8,
        signer_id=signer_id,
    )

    gated_provider.stage_evidence(evidence, challenge=None)

    with pytest.raises(AttestationError, match="staged evidence was not bound to a challenge nonce"):
        gated_provider.unwrap_key(wrapped)


def test_attestation_verification_failure_does_not_consume_evidence(signing_keys):
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = LocalKeyProvider()
    verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})
    golden = hashlib.sha256(b"confidential_app").hexdigest()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={signer_id},
        require_signature=True,
    )

    gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        evidence_provider=None,
    )

    dek, wrapped = gated_provider.generate_data_key()
    challenge = gated_provider.create_challenge()

    # Sign quote with bad measurement
    ts = time.time()
    rd = hashlib.sha256(challenge).hexdigest()
    bad_measurement = "badbeef" * 8
    body = f"{PlatformType.NVIDIA_CC.value}:{bad_measurement}:{rd}:{ts:.3f}".encode("utf-8")
    sig = priv_key.sign(body).hex()

    bad_evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=bad_measurement,
        report_data=rd,
        timestamp=ts,
        signature=sig,
        signer_id=signer_id,
    )

    gated_provider.stage_evidence(bad_evidence, challenge=challenge)

    # Verification fails on bad measurement
    with pytest.raises(AttestationError, match="does not match any golden measurements"):
        gated_provider.unwrap_key(wrapped)

    # Assert that evidence was NOT consumed/wiped out by the failure!
    assert gated_provider._staged_evidence is not None
    assert gated_provider._staged_evidence[0] is bad_evidence
