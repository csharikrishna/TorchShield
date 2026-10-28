"""Unit tests for Confidential Computing & Hardware Attestation Architecture."""
import hashlib
import secrets
import threading
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

    # Tampered signature (guarantee modification by bit inversion)
    sig_raw = bytes.fromhex(sig)
    tampered_sig = (sig_raw[:-1] + bytes([sig_raw[-1] ^ 0xFF])).hex()
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

    # 1. Base provider without evidence provider or staged evidence has no usable path -> False
    gated_no_path = AttestationGatedKeyProvider(prod_inner, valid_verifier, valid_policy, evidence_provider=None)
    assert gated_no_path.is_production_secure is False

    # 1b. Base valid provider with automated evidence provider callback -> True
    dummy_evidence_provider = lambda nonce: AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=hashlib.sha256(nonce).hexdigest(),
        timestamp=time.time(),
        signature="dummy",
        signer_id=signer_id,
    )
    gated_with_path = AttestationGatedKeyProvider(
        prod_inner, valid_verifier, valid_policy, evidence_provider=dummy_evidence_provider
    )
    assert gated_with_path.is_production_secure is True

    # 1c. Base provider with valid active staged evidence -> True
    gated_staged = AttestationGatedKeyProvider(prod_inner, valid_verifier, valid_policy, evidence_provider=None)
    chal = gated_staged.create_challenge()
    staged_ev = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=hashlib.sha256(chal).hexdigest(),
        timestamp=time.time(),
        signature="dummy",
        signer_id=signer_id,
    )
    gated_staged.stage_evidence(staged_ev, challenge=chal)
    assert gated_staged.is_production_secure is True

    # 2. Insecure inner provider -> False
    gated_insecure_inner = AttestationGatedKeyProvider(
        unsecure_inner, valid_verifier, valid_policy, evidence_provider=dummy_evidence_provider
    )
    assert gated_insecure_inner.is_production_secure is False

    # 3. Verifier allow_mock -> False
    mock_verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes}, allow_mock=True)
    assert AttestationGatedKeyProvider(
        prod_inner, mock_verifier, valid_policy, evidence_provider=dummy_evidence_provider
    ).is_production_secure is False

    # 4. Policy allow_mock -> False
    mock_policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allow_mock=True,
    )
    assert AttestationGatedKeyProvider(
        prod_inner, valid_verifier, mock_policy, evidence_provider=dummy_evidence_provider
    ).is_production_secure is False

    # 5. Missing trusted root keys in verifier -> False
    empty_verifier = HardwareAttestationVerifier(trusted_root_public_keys={})
    assert AttestationGatedKeyProvider(
        prod_inner, empty_verifier, valid_policy, evidence_provider=dummy_evidence_provider
    ).is_production_secure is False

    # 6. Unknown signer ID in policy not present in trusted keys -> False
    unknown_signer_policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={"unknown-untrusted-signer"},
    )
    assert AttestationGatedKeyProvider(
        prod_inner, valid_verifier, unknown_signer_policy, evidence_provider=dummy_evidence_provider
    ).is_production_secure is False

    # 7. Signature requirement disabled -> False
    no_sig_policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        require_signature=False,
    )
    assert AttestationGatedKeyProvider(
        prod_inner, valid_verifier, no_sig_policy, evidence_provider=dummy_evidence_provider
    ).is_production_secure is False


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

    # Evidence created with some arbitrary report data, but NOT bound to challenge
    evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data="unbound_nonce_data",
        timestamp=time.time(),
        signature="deadbeef" * 8,
        signer_id=signer_id,
    )

    # Must fail immediately at staging time because no active challenge was created
    with pytest.raises(AttestationError, match="no challenge provided and no active challenge was created"):
        gated_provider.stage_evidence(evidence, challenge=None)


def test_attestation_arbitrary_externally_supplied_challenge_rejected(signing_keys):
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

    # Caller tries to pass an arbitrary externally chosen challenge not issued by create_challenge()
    arbitrary_challenge = b"attacker_chosen_challenge_123456"
    ts = time.time()
    rd = hashlib.sha256(arbitrary_challenge).hexdigest()
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

    with pytest.raises(AttestationError, match="Unauthorized challenge"):
        gated_provider.stage_evidence(evidence, challenge=arbitrary_challenge)


def test_attestation_stale_challenge_rejected(signing_keys):
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
        max_age_seconds=60.0,
    )

    gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        evidence_provider=None,
    )

    challenge = gated_provider.create_challenge()
    # Simulate challenge aging beyond policy max_age_seconds
    gated_provider._authorized_challenges[challenge] -= 120.0

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

    with pytest.raises(AttestationError, match="Stale challenge"):
        gated_provider.stage_evidence(evidence, challenge=challenge)


def test_attestation_replay_attempt_with_consumed_challenge_rejected(signing_keys):
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

    gated_provider.stage_evidence(evidence, challenge=challenge)

    # First unwrap succeeds and consumes challenge and evidence
    assert gated_provider.unwrap_key(wrapped) == dek
    assert gated_provider._staged_evidence is None
    assert challenge not in gated_provider._authorized_challenges

    # Replay attempt: trying to re-stage with the already consumed challenge fails
    with pytest.raises(AttestationError, match="Unauthorized challenge"):
        gated_provider.stage_evidence(evidence, challenge=challenge)


def test_attestation_missing_evidence_fails_closed():
    inner_provider = LocalKeyProvider()
    verifier = HardwareAttestationVerifier()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={"dummy"},
    )
    gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        evidence_provider=None,
    )
    dek, wrapped = gated_provider.generate_data_key()
    with pytest.raises(AttestationError, match="No attestation evidence available"):
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


class CountingInnerKeyProvider(LocalKeyProvider):
    def __init__(self):
        super().__init__()
        self.unwrap_call_count = 0
        self.lock = threading.Lock()

    def unwrap_key(self, wrapped_key: bytes, encryption_context=None) -> bytes:
        with self.lock:
            self.unwrap_call_count += 1
        return super().unwrap_key(wrapped_key, encryption_context=encryption_context)


def test_attestation_concurrent_unwrap_interleaved_barrier(signing_keys):
    """Verify that under controlled concurrent interleaving using synchronization barriers,
    two concurrent unwrap_key calls cannot reuse the same staged evidence and the challenge
    reaches the inner key provider at most once.
    """
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = CountingInnerKeyProvider()

    thread1_in_verify = threading.Event()
    thread2_attempted = threading.Event()

    class InterleavedVerifier(HardwareAttestationVerifier):
        def verify(self, evidence, policy, expected_nonce=None) -> bool:
            # Signal that Thread 1 is inside verification
            thread1_in_verify.set()
            # Wait until Thread 2 attempts concurrent unwrap before finishing verification
            if not thread2_attempted.wait(timeout=5.0):
                raise RuntimeError("Thread 2 failed to attempt unwrap in time")
            return super().verify(evidence, policy, expected_nonce=expected_nonce)

    verifier = InterleavedVerifier(trusted_root_public_keys={signer_id: pub_bytes})
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

    gated_provider.stage_evidence(evidence, challenge=challenge)

    results = {}

    def worker_1():
        try:
            res = gated_provider.unwrap_key(wrapped)
            results["thread1"] = ("SUCCESS", res)
        except Exception as exc:
            results["thread1"] = ("ERROR", exc)

    def worker_2():
        try:
            # Wait for Thread 1 to be inside verify()
            if not thread1_in_verify.wait(timeout=5.0):
                results["thread2"] = ("ERROR", RuntimeError("Thread 1 did not enter verify"))
                return
            try:
                res = gated_provider.unwrap_key(wrapped)
                results["thread2"] = ("SUCCESS", res)
            except Exception as exc:
                results["thread2"] = ("ERROR", exc)
        finally:
            thread2_attempted.set()

    t1 = threading.Thread(target=worker_1)
    t2 = threading.Thread(target=worker_2)

    t1.start()
    t2.start()

    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    # Thread 1 must succeed with correct DEK
    assert results["thread1"][0] == "SUCCESS"
    assert results["thread1"][1] == dek

    # Thread 2 must fail closed with AttestationError
    assert results["thread2"][0] == "ERROR"
    assert isinstance(results["thread2"][1], AttestationError)
    assert "currently in use" in str(results["thread2"][1])

    # Inner key provider must be reached EXACTLY ONCE
    assert inner_provider.unwrap_call_count == 1

    # Staged evidence and challenge must be consumed
    assert gated_provider._staged_evidence is None
    assert challenge not in gated_provider._authorized_challenges


def test_attestation_concurrent_race_many_threads(signing_keys):
    """Verify that in a high-concurrency race across 10 threads waiting on a barrier,
    the same staged evidence and challenge reach the inner provider at most once.
    """
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = CountingInnerKeyProvider()
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

    gated_provider.stage_evidence(evidence, challenge=challenge)

    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait(timeout=5.0)
        try:
            res = gated_provider.unwrap_key(wrapped)
            with results_lock:
                results.append(("SUCCESS", res))
        except Exception as exc:
            with results_lock:
                results.append(("ERROR", exc))

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    successes = [r for r in results if r[0] == "SUCCESS"]
    errors = [r for r in results if r[0] == "ERROR"]

    # Exactly 1 thread succeeds
    assert len(successes) == 1
    assert successes[0][1] == dek

    # Exactly num_threads - 1 threads fail with AttestationError
    assert len(errors) == num_threads - 1
    for status, exc in errors:
        assert isinstance(exc, AttestationError)

    # Inner provider called at most once
    assert inner_provider.unwrap_call_count == 1


def test_attestation_concurrent_evidence_provider_callbacks(signing_keys):
    """Verify that when automated evidence provider callbacks are used,
    each concurrent thread generates and verifies fresh nonces independently.
    """
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = CountingInnerKeyProvider()
    verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})
    golden = hashlib.sha256(b"confidential_app").hexdigest()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={signer_id},
        require_signature=True,
        require_fresh_nonce=True,
    )

    def automated_evidence_provider(nonce: bytes) -> AttestationEvidence:
        ts = time.time()
        rd = hashlib.sha256(nonce).hexdigest()
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
        evidence_provider=automated_evidence_provider,
    )

    num_threads = 5
    barrier = threading.Barrier(num_threads)
    results = []
    results_lock = threading.Lock()

    dek, wrapped = gated_provider.generate_data_key()

    def worker():
        barrier.wait(timeout=5.0)
        try:
            res = gated_provider.unwrap_key(wrapped)
            with results_lock:
                results.append(("SUCCESS", res))
        except Exception as exc:
            with results_lock:
                results.append(("ERROR", exc))

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # All 5 threads succeed because each gets a fresh independent challenge
    assert len([r for r in results if r[0] == "SUCCESS"]) == num_threads
    assert inner_provider.unwrap_call_count == num_threads


def test_attestation_cross_challenge_staging_race(signing_keys):
    """A. Cross-challenge staging race:
    Pause verification of A after A has been marked in flight. Create B and stage evidence B.
    Resume A and let it succeed. Assert that A's successful cleanup does not erase B's staged evidence.
    Then verify that B can be processed according to the selected concurrency policy.
    """
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = CountingInnerKeyProvider()

    thread_a_in_verify = threading.Event()
    resume_thread_a = threading.Event()

    class PausableVerifier(HardwareAttestationVerifier):
        def verify(self, evidence, policy, expected_nonce=None) -> bool:
            if expected_nonce == challenge_a:
                thread_a_in_verify.set()
                if not resume_thread_a.wait(timeout=5.0):
                    raise RuntimeError("Thread A timed out waiting to resume")
            return super().verify(evidence, policy, expected_nonce=expected_nonce)

    verifier = PausableVerifier(trusted_root_public_keys={signer_id: pub_bytes})
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

    # Create and stage challenge A
    challenge_a = gated_provider.create_challenge()
    ts_a = time.time()
    rd_a = hashlib.sha256(challenge_a).hexdigest()
    body_a = f"{PlatformType.NVIDIA_CC.value}:{golden}:{rd_a}:{ts_a:.3f}".encode("utf-8")
    sig_a = priv_key.sign(body_a).hex()
    evidence_a = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=rd_a,
        timestamp=ts_a,
        signature=sig_a,
        signer_id=signer_id,
    )
    gated_provider.stage_evidence(evidence_a, challenge=challenge_a)

    results_a = {}

    def worker_a():
        try:
            res = gated_provider.unwrap_key(wrapped)
            results_a["result"] = ("SUCCESS", res)
        except Exception as exc:
            results_a["result"] = ("ERROR", exc)

    thread_a = threading.Thread(target=worker_a)
    thread_a.start()

    # Wait until Thread A enters verification and marks challenge A in-flight
    assert thread_a_in_verify.wait(timeout=5.0), "Thread A did not enter verify"
    assert challenge_a in gated_provider._in_flight_challenges

    # While Thread A is in verify, create and stage challenge B
    challenge_b = gated_provider.create_challenge()
    assert challenge_b != challenge_a
    assert gated_provider._active_challenge == challenge_b

    ts_b = time.time()
    rd_b = hashlib.sha256(challenge_b).hexdigest()
    body_b = f"{PlatformType.NVIDIA_CC.value}:{golden}:{rd_b}:{ts_b:.3f}".encode("utf-8")
    sig_b = priv_key.sign(body_b).hex()
    evidence_b = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=rd_b,
        timestamp=ts_b,
        signature=sig_b,
        signer_id=signer_id,
    )
    # Stage evidence B while A is in flight
    gated_provider.stage_evidence(evidence_b, challenge=challenge_b)
    assert gated_provider._staged_evidence is not None
    assert gated_provider._staged_evidence[1] == challenge_b

    # Resume Thread A and allow it to complete successfully
    resume_thread_a.set()
    thread_a.join(timeout=10.0)

    # Thread A must succeed
    assert results_a["result"][0] == "SUCCESS"
    assert results_a["result"][1] == dek
    assert challenge_a not in gated_provider._in_flight_challenges
    assert challenge_a not in gated_provider._authorized_challenges

    # CRITICAL INVARIANT: A's cleanup must NOT have erased B's staged evidence!
    assert gated_provider._staged_evidence is not None
    assert gated_provider._staged_evidence[1] == challenge_b
    assert gated_provider._staged_evidence[0] is evidence_b

    # Now verify that B can be processed successfully
    unwrapped_b = gated_provider.unwrap_key(wrapped)
    assert unwrapped_b == dek
    assert inner_provider.unwrap_call_count == 2

    # After B succeeds, its staged evidence and challenge are cleanly consumed
    assert gated_provider._staged_evidence is None
    assert challenge_b not in gated_provider._authorized_challenges


def test_attestation_independent_challenge_ordering(signing_keys):
    """E. Independent challenge ordering:
    Exercise both possible completion orders when two separately authorized challenges
    are processed concurrently. Ensure one operation's cleanup does not corrupt the other operation's state.
    """
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    golden = hashlib.sha256(b"confidential_app").hexdigest()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={signer_id},
        require_signature=True,
        require_fresh_nonce=True,
    )

    def create_evidence_for_challenge(ch: bytes) -> AttestationEvidence:
        ts = time.time()
        rd = hashlib.sha256(ch).hexdigest()
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

    # Order 1: Thread A finishes before Thread B
    for finish_first in ("A", "B"):
        inner_provider = CountingInnerKeyProvider()
        event_a_entered = threading.Event()
        event_b_entered = threading.Event()
        release_first = threading.Event()
        release_second = threading.Event()

        first_target = None
        second_target = None

        class OrderVerifier(HardwareAttestationVerifier):
            def verify(self, evidence, pol, expected_nonce=None) -> bool:
                if expected_nonce == ch_a:
                    event_a_entered.set()
                    if finish_first == "A":
                        release_first.wait(timeout=5.0)
                    else:
                        release_second.wait(timeout=5.0)
                elif expected_nonce == ch_b:
                    event_b_entered.set()
                    if finish_first == "B":
                        release_first.wait(timeout=5.0)
                    else:
                        release_second.wait(timeout=5.0)
                return super().verify(evidence, pol, expected_nonce=expected_nonce)

        verifier = OrderVerifier(trusted_root_public_keys={signer_id: pub_bytes})
        provider = AttestationGatedKeyProvider(
            inner_provider=inner_provider,
            verifier=verifier,
            policy=policy,
        )
        dek, wrapped = provider.generate_data_key()

        ch_a = provider.create_challenge()
        ev_a = create_evidence_for_challenge(ch_a)
        provider.stage_evidence(ev_a, challenge=ch_a)

        res_a = {}
        res_b = {}

        def runner_a():
            try:
                res_a["val"] = provider.unwrap_key(wrapped)
            except Exception as e:
                res_a["err"] = e

        def runner_b():
            try:
                res_b["val"] = provider.unwrap_key(wrapped)
            except Exception as e:
                res_b["err"] = e

        t_a = threading.Thread(target=runner_a)
        t_a.start()
        assert event_a_entered.wait(timeout=5.0)

        # While A is in verify, stage B and start runner B
        ch_b = provider.create_challenge()
        ev_b = create_evidence_for_challenge(ch_b)
        provider.stage_evidence(ev_b, challenge=ch_b)

        t_b = threading.Thread(target=runner_b)
        t_b.start()
        assert event_b_entered.wait(timeout=5.0)

        # Both A and B are now simultaneously in verify()
        assert ch_a in provider._in_flight_challenges
        assert ch_b in provider._in_flight_challenges

        # Release the first thread to complete
        release_first.set()
        time.sleep(0.05)

        # Release the second thread to complete
        release_second.set()

        t_a.join(timeout=10.0)
        t_b.join(timeout=10.0)

        assert res_a.get("val") == dek, f"Order {finish_first}: A failed with {res_a.get('err')}"
        assert res_b.get("val") == dek, f"Order {finish_first}: B failed with {res_b.get('err')}"
        assert inner_provider.unwrap_call_count == 2
        assert len(provider._in_flight_challenges) == 0
        assert ch_a not in provider._authorized_challenges
        assert ch_b not in provider._authorized_challenges
        assert provider._staged_evidence is None


def test_attestation_verification_failure_preservation_and_recovery(signing_keys):
    """D. Verification failure preservation:
    Make evidence invalid. Verify that attestation fails closed, the underlying key provider
    is never called, the failed operation releases its in-flight reservation, and diagnostic evidence
    remains available. Then verify that re-staging valid evidence recovers cleanly.
    """
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = CountingInnerKeyProvider()
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
    )
    dek, wrapped = gated_provider.generate_data_key()

    ch = gated_provider.create_challenge()
    ts = time.time()
    rd = hashlib.sha256(ch).hexdigest()
    bad_measurement = "0" * 64
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

    gated_provider.stage_evidence(bad_evidence, challenge=ch)

    with pytest.raises(AttestationError, match="does not match any golden measurements"):
        gated_provider.unwrap_key(wrapped)

    # 1. Inner key provider must NEVER have been called
    assert inner_provider.unwrap_call_count == 0

    # 2. In-flight reservation must be released
    assert ch not in gated_provider._in_flight_challenges

    # 3. Diagnostic evidence remains preserved
    assert gated_provider._staged_evidence is not None
    assert gated_provider._staged_evidence[0] is bad_evidence
    assert gated_provider._staged_evidence[1] == ch

    # 4. Correcting the evidence for the same authorized challenge succeeds
    body_good = f"{PlatformType.NVIDIA_CC.value}:{golden}:{rd}:{ts:.3f}".encode("utf-8")
    sig_good = priv_key.sign(body_good).hex()
    good_evidence = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=rd,
        timestamp=ts,
        signature=sig_good,
        signer_id=signer_id,
    )
    gated_provider.stage_evidence(good_evidence, challenge=ch)
    unwrapped = gated_provider.unwrap_key(wrapped)
    assert unwrapped == dek
    assert inner_provider.unwrap_call_count == 1
    assert gated_provider._staged_evidence is None


def test_attestation_challenge_expiry_and_bounded_lifecycle(signing_keys):
    """F. Challenge expiry and lifecycle cleanup:
    - Opportunistically remove expired, non-in-flight challenges.
    - Keep active in-flight challenges safe from premature deletion.
    - Define and document a maximum number of outstanding authorized challenges.
    - Fail closed when limit reached; reclaim eligible expired entries.
    - Keep _active_challenge and staged evidence consistent when expired entries are removed.
    """
    priv_key, pub_bytes = signing_keys
    signer_id = "nvidia-cc-root-001"
    inner_provider = CountingInnerKeyProvider()
    verifier = HardwareAttestationVerifier(trusted_root_public_keys={signer_id: pub_bytes})
    golden = hashlib.sha256(b"confidential_app").hexdigest()
    policy = AttestationPolicy(
        allowed_platforms={PlatformType.NVIDIA_CC.value},
        golden_measurements={golden},
        allowed_signer_ids={signer_id},
        max_age_seconds=10.0,
    )

    # Capacity bounded to 3 challenges
    gated_provider = AttestationGatedKeyProvider(
        inner_provider=inner_provider,
        verifier=verifier,
        policy=policy,
        max_authorized_challenges=3,
    )

    ch1 = gated_provider.create_challenge()
    ch2 = gated_provider.create_challenge()
    ch3 = gated_provider.create_challenge()
    assert len(gated_provider._authorized_challenges) == 3

    # Capacity reached: creating a 4th challenge fails closed
    with pytest.raises(AttestationError, match="Authorized challenge capacity exceeded"):
        gated_provider.create_challenge()

    # Simulate expiration of ch1 and ch2 by adjusting their timestamps back 100 seconds
    gated_provider._authorized_challenges[ch1] = time.time() - 100.0
    gated_provider._authorized_challenges[ch2] = time.time() - 100.0

    # Next create_challenge cleans up ch1 and ch2, freeing capacity for ch4
    ch4 = gated_provider.create_challenge()
    assert ch1 not in gated_provider._authorized_challenges
    assert ch2 not in gated_provider._authorized_challenges
    assert ch3 in gated_provider._authorized_challenges
    assert ch4 in gated_provider._authorized_challenges
    assert len(gated_provider._authorized_challenges) == 2

    # In-flight protection: mark ch3 in-flight and expire its timestamp
    gated_provider._in_flight_challenges.add(ch3)
    gated_provider._authorized_challenges[ch3] = time.time() - 100.0

    # Run cleanup directly: ch3 MUST NOT be purged while in-flight!
    with gated_provider._lock:
        purged = gated_provider._cleanup_expired_challenges_locked()
    assert purged == 0
    assert ch3 in gated_provider._authorized_challenges

    # Remove from in-flight and run cleanup again: ch3 is now purged
    gated_provider._in_flight_challenges.discard(ch3)
    with gated_provider._lock:
        purged = gated_provider._cleanup_expired_challenges_locked()
    assert purged == 1
    assert ch3 not in gated_provider._authorized_challenges

    # Consistency of _active_challenge and _staged_evidence on expiration
    ch5 = gated_provider.create_challenge()
    assert gated_provider._active_challenge == ch5
    ev5 = AttestationEvidence(
        platform=PlatformType.NVIDIA_CC.value,
        measurement=golden,
        report_data=hashlib.sha256(ch5).hexdigest(),
        timestamp=time.time(),
    )
    gated_provider.stage_evidence(ev5, challenge=ch5)
    assert gated_provider._staged_evidence is not None

    # Expire ch5
    gated_provider._authorized_challenges[ch5] = time.time() - 100.0
    with gated_provider._lock:
        gated_provider._cleanup_expired_challenges_locked()

    # Both _active_challenge and _staged_evidence must be cleared to maintain consistency
    assert gated_provider._active_challenge is None
    assert gated_provider._staged_evidence is None
