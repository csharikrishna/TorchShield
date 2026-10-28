"""Confidential Computing & Hardware Attestation Architecture for TorchShield.

Provides cryptographic verification of Trusted Execution Environments (TEEs)
including NVIDIA Hopper/Blackwell Confidential Computing (CC), AMD SEV-SNP,
and Intel TDX. Hardware attestation quotes and measurements are verified before
releasing cryptographic Data Encryption Keys (DEKs).

ARCHITECTURE & HARDWARE LIMITATION NOTICE:
The current verifier evaluates a structured, cryptographically signed evidence format
(AttestationEvidence with Ed25519 platform signatures over measurement digests, report
nonces, and timestamps). It establishes the verifiable protocol contract for TEE
key-gating. It is NOT yet a complete vendor-specific hardware quote parser (e.g. Intel
TDX PCK / SGX quote verification via Intel QVL, AMD SEV-SNP VCEK verification via AMD KDS,
or NVIDIA SPDM NV-Attestation SDK). Full vendor quote verification requires linking
vendor-specific C/Rust parsing libraries and execution on genuine TEE hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import secrets
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from torchshield.checkpoint import CheckpointError
from torchshield.key_providers.base import KeyProvider

logger = logging.getLogger(__name__)


class AttestationError(CheckpointError):
    """Raised when hardware attestation validation fails or quote is untrusted."""


class PlatformType(str, Enum):
    """Supported Trusted Execution Environment (TEE) hardware platforms."""
    NVIDIA_CC = "nvidia_cc"
    AMD_SEV_SNP = "amd_sev_snp"
    INTEL_TDX = "intel_tdx"
    GENERIC_TEE = "generic_tee"


@dataclass(frozen=True)
class AttestationEvidence:
    """Hardware attestation evidence containing measurement quote and platform signature."""
    platform: str
    measurement: str  # Hex digest of firmware / kernel / launch measurement
    report_data: str   # Hex digest of replay protection nonce or user data
    timestamp: float   # Unix timestamp of quote generation
    signature: Optional[str] = None  # Hex signature over quote by platform root of trust
    signer_id: Optional[str] = None  # Identifier of signing hardware key / cert
    raw_quote: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to serializable dictionary."""
        return {
            "platform": self.platform,
            "measurement": self.measurement,
            "report_data": self.report_data,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "signer_id": self.signer_id,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttestationEvidence":
        """Reconstruct evidence from dictionary."""
        return cls(
            platform=data["platform"],
            measurement=data["measurement"],
            report_data=data["report_data"],
            timestamp=float(data["timestamp"]),
            signature=data.get("signature"),
            signer_id=data.get("signer_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class AttestationPolicy:
    """Policy rules governing hardware attestation verification."""
    allowed_platforms: Set[str]
    golden_measurements: Set[str]  # Known-good software/firmware measurement hashes
    max_age_seconds: float = 300.0  # Freshness window to prevent stale quote replay
    allowed_signer_ids: Optional[Set[str]] = None
    require_fresh_nonce: bool = True
    require_signature: bool = True
    allow_mock: bool = False


class HardwareAttestationVerifier:
    """Cryptographic verifier for hardware attestation quotes and measurements."""

    def __init__(
        self,
        trusted_root_public_keys: Optional[Dict[str, bytes]] = None,
        allow_mock: bool = False,
    ):
        """
        Args:
            trusted_root_public_keys: Mapping of signer_id -> 32-byte Ed25519 public key.
            allow_mock: Enable mock verification for development/testing.
        """
        self._trusted_keys = dict(trusted_root_public_keys or {})
        self._allow_mock = allow_mock

    def verify(
        self,
        evidence: AttestationEvidence,
        policy: AttestationPolicy,
        expected_nonce: Optional[bytes] = None,
    ) -> bool:
        """Verify hardware attestation evidence against security policy.
        
        Fail-closed at every step:
        1. Validates platform is explicitly allowed.
        2. Validates quote timestamp is fresh (prevents replay of expired evidence).
        3. Validates report_data matches expected nonce (prevents relay attacks).
        4. Validates hardware measurement matches golden policy hashes.
        5. Validates hardware signer ID and cryptographic signature.
        """
        # 1. Platform validation
        if evidence.platform not in policy.allowed_platforms:
            raise AttestationError(
                f"Platform '{evidence.platform}' is not permitted by policy. "
                f"Allowed platforms: {sorted(policy.allowed_platforms)}"
            )

        # 2. Freshness check
        current_time = time.time()
        age = current_time - evidence.timestamp
        if age < -60.0:
            raise AttestationError(f"Attestation evidence timestamp is in the future (skew: {age:.2f}s).")
        if age > policy.max_age_seconds:
            raise AttestationError(
                f"Attestation evidence expired. Age {age:.1f}s exceeds max_age_seconds {policy.max_age_seconds}s."
            )

        # 3. Nonce / replay check
        if policy.require_fresh_nonce:
            if not expected_nonce:
                raise AttestationError("Policy requires fresh nonce verification, but no expected nonce was provided.")
            
            # Nonce may match directly as hex or as sha256 hash
            nonce_hex = expected_nonce.hex()
            nonce_hash = hashlib.sha256(expected_nonce).hexdigest()
            matches_raw = secrets.compare_digest(evidence.report_data.lower(), nonce_hex.lower())
            matches_hash = secrets.compare_digest(evidence.report_data.lower(), nonce_hash.lower())
            if not (matches_raw or matches_hash):
                raise AttestationError("Attestation report_data nonce mismatch; possible replay or relay attack.")

        # 4. Golden measurement validation
        measurement_lower = evidence.measurement.lower()
        matched_measurement = any(
            secrets.compare_digest(measurement_lower, golden.lower())
            for golden in policy.golden_measurements
        )
        if not matched_measurement:
            raise AttestationError(
                f"Hardware measurement '{evidence.measurement}' does not match any golden measurements."
            )

        # 5. Signer ID validation
        if policy.allowed_signer_ids is not None:
            if not evidence.signer_id or evidence.signer_id not in policy.allowed_signer_ids:
                raise AttestationError(f"Attestation signer '{evidence.signer_id}' is not an authorized platform signer.")

        # 6. Cryptographic signature verification
        require_signature = getattr(policy, "require_signature", True)
        if require_signature:
            if not evidence.signature or not evidence.signer_id:
                raise AttestationError("Attestation evidence lacks required platform signature or signer_id.")

        if evidence.signature is not None:
            if not evidence.signer_id:
                raise AttestationError("Attestation evidence has signature but missing signer_id.")
            
            pub_bytes = self._trusted_keys.get(evidence.signer_id)
            if not pub_bytes:
                if not (policy.allow_mock or self._allow_mock):
                    raise AttestationError(f"No trusted root key found for platform signer '{evidence.signer_id}'.")
            else:
                # Signature covers: platform:measurement:report_data:timestamp
                quote_body = (
                    f"{evidence.platform}:{evidence.measurement}:{evidence.report_data}:{evidence.timestamp:.3f}".encode("utf-8")
                )
                try:
                    pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
                    sig_bytes = bytes.fromhex(evidence.signature)
                    pub_key.verify(sig_bytes, quote_body)
                except Exception as exc:
                    raise AttestationError(f"Hardware attestation quote signature verification failed: {exc}") from exc

        return True


class AttestationGatedKeyProvider(KeyProvider):
    """KeyProvider wrapper that gates key unwrap operations behind verified hardware attestation."""

    def __init__(
        self,
        inner_provider: KeyProvider,
        verifier: HardwareAttestationVerifier,
        policy: AttestationPolicy,
        evidence_provider: Optional[Callable[[bytes], AttestationEvidence]] = None,
    ):
        """
        Args:
            inner_provider: The underlying KMS or local key provider managing the KEK.
            verifier: HardwareAttestationVerifier instance.
            policy: AttestationPolicy requiring specific platform and measurements.
            evidence_provider: Callable taking a fresh challenge nonce bytes and returning AttestationEvidence.
        """
        self._inner = inner_provider
        self._verifier = verifier
        self._policy = policy
        self._evidence_provider = evidence_provider
        self._staged_evidence: Optional[Tuple[AttestationEvidence, bytes]] = None
        # Track authorized challenges issued by this provider: challenge bytes -> issue timestamp float
        self._authorized_challenges: Dict[bytes, float] = {}
        self._active_challenge: Optional[bytes] = None
        # Concurrency control: protect challenge lifecycle and staged evidence single-use invariants
        self._lock = threading.RLock()
        self._in_flight_challenges: Set[bytes] = set()

    @property
    def is_production_secure(self) -> bool:
        """Return True only if underlying provider, trust configuration, and evidence-acquisition path provide genuine production trust."""
        if not getattr(self._inner, "is_production_secure", False):
            return False
        if getattr(self._verifier, "_allow_mock", False) or getattr(self._policy, "allow_mock", False):
            return False
        if not getattr(self._policy, "require_signature", True) or not getattr(self._policy, "require_fresh_nonce", True):
            return False
        trusted_keys = getattr(self._verifier, "_trusted_keys", {})
        if not trusted_keys:
            return False
        allowed_signers = getattr(self._policy, "allowed_signer_ids", None)
        if allowed_signers is not None:
            if not allowed_signers:
                return False
            for sid in allowed_signers:
                if sid not in trusted_keys:
                    return False

        with self._lock:
            # Account for evidence-acquisition path:
            # A usable path exists if there is an automated evidence provider callback,
            # or if currently valid staged evidence bound to an unconsumed authorized challenge is present.
            has_provider_path = self._evidence_provider is not None and callable(self._evidence_provider)
            has_staged_path = False
            if self._staged_evidence is not None:
                _, staged_challenge = self._staged_evidence
                if (
                    staged_challenge is not None
                    and staged_challenge in self._authorized_challenges
                    and staged_challenge not in self._in_flight_challenges
                ):
                    issued_time = self._authorized_challenges[staged_challenge]
                    age = time.time() - issued_time
                    if 0.0 <= age <= getattr(self._policy, "max_age_seconds", 300.0):
                        has_staged_path = True

            if not (has_provider_path or has_staged_path):
                return False

        return True

    def get_key_id(self) -> str:
        return f"attestation-gated::{self._inner.get_key_id()}"

    def create_challenge(self) -> bytes:
        """Create and register an authorized fresh challenge nonce for staging evidence.

        Returns:
            32-byte cryptographically secure challenge nonce.
        """
        nonce = secrets.token_bytes(32)
        with self._lock:
            self._authorized_challenges[nonce] = time.time()
            self._active_challenge = nonce
        return nonce

    def stage_evidence(
        self,
        evidence: AttestationEvidence,
        challenge: Optional[bytes] = None,
    ) -> None:
        """Stage pre-acquired attestation evidence for the next unwrap operation.

        Enforces explicit trust rules for challenge nonces:
        - If an external challenge is supplied, it MUST match an authorized active
          challenge issued by create_challenge() on this provider. Arbitrary caller-supplied
          byte strings that were not issued by this provider are rejected.
        - If challenge is omitted, binds to the most recent active challenge if valid.
        - Checks that the authorized challenge has not expired according to policy.max_age_seconds.

        Args:
            evidence: AttestationEvidence to stage.
            challenge: The authorized challenge nonce bound into evidence.report_data.
        """
        with self._lock:
            target_challenge = challenge if challenge is not None else self._active_challenge
            if target_challenge is None:
                raise AttestationError(
                    "Cannot stage evidence: no challenge provided and no active challenge was created via create_challenge()."
                )

            if target_challenge not in self._authorized_challenges:
                raise AttestationError(
                    "Unauthorized challenge: externally supplied challenge was not issued by this provider's challenge lifecycle."
                )

            if target_challenge in self._in_flight_challenges:
                raise AttestationError(
                    "Cannot stage evidence: specified challenge is currently undergoing verification in another operation."
                )

            issued_time = self._authorized_challenges[target_challenge]
            age = time.time() - issued_time
            if age < -60.0 or age > getattr(self._policy, "max_age_seconds", 300.0):
                raise AttestationError(
                    f"Stale challenge: challenge nonce expired (age {age:.1f}s > max {getattr(self._policy, 'max_age_seconds', 300.0)}s)."
                )

            self._staged_evidence = (evidence, target_challenge)

    def generate_data_key(
        self, encryption_context: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, bytes]:
        """Delegate data key generation to inner provider."""
        return self._inner.generate_data_key(encryption_context=encryption_context)

    def wrap_key(
        self, plaintext_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Delegate key wrapping to inner provider."""
        return self._inner.wrap_key(plaintext_key, encryption_context=encryption_context)

    def unwrap_key(
        self, wrapped_key: bytes, encryption_context: Optional[Dict[str, str]] = None
    ) -> bytes:
        """Unwrap data key strictly after validating fresh hardware attestation."""
        evidence: Optional[AttestationEvidence] = None
        expected_nonce: Optional[bytes] = None
        bound_challenge: Optional[bytes] = None

        with self._lock:
            if self._evidence_provider is not None:
                nonce = secrets.token_bytes(32)
                evidence = self._evidence_provider(nonce)
                expected_nonce = nonce
            elif self._staged_evidence is not None:
                evidence, bound_challenge = self._staged_evidence
                if bound_challenge in self._in_flight_challenges:
                    raise AttestationError(
                        "Staged attestation challenge is currently in use by another concurrent unwrap operation."
                    )
                if getattr(self._policy, "require_fresh_nonce", True):
                    if bound_challenge not in self._authorized_challenges:
                        raise AttestationError(
                            "Staged attestation challenge is no longer authorized (already consumed, expired, or untrusted)."
                        )
                    issued_time = self._authorized_challenges[bound_challenge]
                    age = time.time() - issued_time
                    if age < -60.0 or age > getattr(self._policy, "max_age_seconds", 300.0):
                        raise AttestationError(
                            f"Stale challenge: staged challenge nonce expired (age {age:.1f}s > max {getattr(self._policy, 'max_age_seconds', 300.0)}s)."
                        )
                    expected_nonce = bound_challenge
                # Mark challenge in-flight while holding the lock so no concurrent unwrap can claim it
                self._in_flight_challenges.add(bound_challenge)
            else:
                raise AttestationError(
                    "Hardware attestation failed: No attestation evidence available to unwrap key."
                )

        # Verification executes outside self._lock to allow concurrency across independent requests
        # and support controlled interleaving/barriers in tests without lock-order inversion or deadlocks.
        # If verification fails, bound_challenge is removed from _in_flight_challenges in the finally block
        # and evidence is preserved in _staged_evidence for diagnostic inspection.
        verification_succeeded = False
        try:
            self._verifier.verify(evidence, self._policy, expected_nonce=expected_nonce)
            verification_succeeded = True
        finally:
            if bound_challenge is not None:
                with self._lock:
                    self._in_flight_challenges.discard(bound_challenge)
                    if verification_succeeded:
                        # Consume challenge and clear staged evidence atomically to enforce single-use / anti-replay
                        self._authorized_challenges.pop(bound_challenge, None)
                        if self._active_challenge == bound_challenge:
                            self._active_challenge = None
                        self._staged_evidence = None

        # Safely delegate unwrap to inner provider ONLY after verification strictly succeeded
        return self._inner.unwrap_key(wrapped_key, encryption_context=encryption_context)

    def rewrap_key(
        self,
        wrapped_key: bytes,
        new_key_id: str,
        encryption_context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Re-wrap existing DEK under a new KEK version after verifying hardware attestation."""
        dek = self.unwrap_key(wrapped_key, encryption_context=encryption_context)
        try:
            return self._inner.rewrap_key(wrapped_key, new_key_id, encryption_context=encryption_context)
        except NotImplementedError:
            return self._inner.wrap_key(dek, encryption_context=encryption_context)
        finally:
            del dek


__all__ = [
    "AttestationError",
    "PlatformType",
    "AttestationEvidence",
    "AttestationPolicy",
    "HardwareAttestationVerifier",
    "AttestationGatedKeyProvider",
]
