"""Publisher Provenance and Monotonic Freshness Ledger (Anti-Rollback) Protection.

Provides:
1. Ed25519 digital signature signing and verification over canonical manifest specifications.
2. File-backed atomic monotonic freshness ledger preventing rollback attacks across
   training resumes and model promotions.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
import uuid
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

from ahcc.checkpoint import CheckpointError


class ProvenanceError(CheckpointError):
    """Raised when signature verification or artifact provenance checks fail."""


class RollbackError(ProvenanceError):
    """Raised when a checkpoint fails monotonic freshness / anti-rollback checks."""


class Ed25519Signer:
    """Ed25519 digital signature generator for checkpoint manifest provenance."""

    def __init__(self, private_bytes: Optional[bytes] = None):
        if private_bytes is not None:
            if len(private_bytes) != 32:
                raise ValueError("Ed25519 private key must be exactly 32 bytes.")
            self._key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        else:
            self._key = Ed25519PrivateKey.generate()

    @property
    def private_bytes(self) -> bytes:
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PrivateFormat,
            NoEncryption,
        )
        return self._key.private_bytes(
            encoding=Encoding.Raw,
            format=PrivateFormat.Raw,
            encryption_algorithm=NoEncryption(),
        )

    @property
    def public_bytes(self) -> bytes:
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )
        return self._key.public_key().public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        )

    def sign(self, message: bytes) -> bytes:
        """Sign a message digest or byte string using Ed25519."""
        if not isinstance(message, (bytes, bytearray)):
            raise TypeError("Message to sign must be bytes.")
        return self._key.sign(bytes(message))


class Ed25519Verifier:
    """Ed25519 digital signature validator for checkpoint manifest provenance."""

    def __init__(self, public_bytes: bytes):
        if len(public_bytes) != 32:
            raise ValueError("Ed25519 public key must be exactly 32 bytes.")
        self._key = Ed25519PublicKey.from_public_bytes(public_bytes)
        self._public_bytes = public_bytes

    @property
    def public_bytes(self) -> bytes:
        return self._public_bytes

    def verify(self, signature: bytes, message: bytes) -> bool:
        """Verify an Ed25519 signature. Raises ProvenanceError on failure."""
        if len(signature) != 64:
            raise ProvenanceError(f"Invalid Ed25519 signature length: {len(signature)} (expected 64 bytes).")
        try:
            self._key.verify(signature, bytes(message))
            return True
        except InvalidSignature as exc:
            raise ProvenanceError("Ed25519 signature verification failed: signature is invalid or message was tampered.") from exc


class FreshnessLedger:
    """Atomic monotonic freshness ledger preventing checkpoint rollback attacks.

    Guarantees that for a given training run or model lineage:
    - Step numbers must strictly advance monotonically.
    - Checkpoints cannot be replayed from an older iteration.
    - Updates are persisted atomically via fsync + replace.
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = os.path.abspath(ledger_path)
        ledger_dir = os.path.dirname(self.ledger_path)
        if ledger_dir and not os.path.exists(ledger_dir):
            os.makedirs(ledger_dir, exist_ok=True)
        if not os.path.exists(self.ledger_path) or os.path.getsize(self.ledger_path) == 0:
            self._write_state({})

    def _read_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.ledger_path) or os.path.getsize(self.ledger_path) == 0:
            return {}
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
        except Exception as exc:
            raise RollbackError(f"Failed to read freshness ledger at {self.ledger_path}: {exc}") from exc

    def _write_state(self, state: Dict[str, Any]) -> None:
        ledger_dir = os.path.dirname(self.ledger_path)
        tmp_path = os.path.join(ledger_dir, f".ledger_tmp_{uuid.uuid4().hex}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.ledger_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def check_freshness(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> bool:
        """Verify whether an incoming checkpoint satisfies monotonic freshness."""
        state = self._read_state()
        if run_id not in state:
            return True

        record = state[run_id]
        recorded_step = record.get("max_step", 0)
        recorded_epoch = record.get("max_epoch", 0)

        if step < recorded_step:
            raise RollbackError(
                f"Rollback attack detected for run '{run_id}': incoming step {step} is strictly "
                f"less than highest authenticated step {recorded_step}."
            )
        if step == recorded_step and epoch < recorded_epoch:
            raise RollbackError(
                f"Rollback attack detected for run '{run_id}': incoming epoch {epoch} is strictly "
                f"less than highest authenticated epoch {recorded_epoch} at step {step}."
            )
        return True

    def record_and_verify(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> None:
        """Verify freshness and advance the monotonic high-water mark."""
        state = self._read_state()
        if run_id in state:
            record = state[run_id]
            recorded_step = record.get("max_step", 0)
            recorded_epoch = record.get("max_epoch", 0)

            if step < recorded_step:
                raise RollbackError(
                    f"Rollback attack detected for run '{run_id}': incoming step {step} is strictly "
                    f"less than highest recorded step {recorded_step}."
                )
            if step == recorded_step and epoch < recorded_epoch:
                raise RollbackError(
                    f"Rollback attack detected for run '{run_id}': incoming epoch {epoch} is strictly "
                    f"less than highest recorded epoch {recorded_epoch} at step {step}."
                )

        state[run_id] = {
            "max_step": max(step, state.get(run_id, {}).get("max_step", step)),
            "max_epoch": max(epoch, state.get(run_id, {}).get("max_epoch", epoch)),
            "latest_artifact_id": artifact_id,
        }
        self._write_state(state)

    def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return the current recorded record for run_id."""
        return self._read_state().get(run_id)
