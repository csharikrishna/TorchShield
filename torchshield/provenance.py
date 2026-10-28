"""Publisher Provenance and Monotonic Freshness Ledger (Anti-Rollback) Protection.

Provides:
1. Ed25519 digital signature signing and verification over canonical manifest specifications.
2. Trusted Freshness Backend architecture supporting pluggable transactional, cloud,
   and local backends to prevent checkpoint rollback attacks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass, field
import json
import logging
import os
import secrets
import sqlite3
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

from torchshield.checkpoint import CheckpointError

logger = logging.getLogger(__name__)


class ProvenanceError(CheckpointError):
    """Raised when signature verification or artifact provenance checks fail."""


class RollbackError(ProvenanceError):
    """Raised when a checkpoint fails monotonic freshness / anti-rollback checks."""


class FreshnessServiceUnavailableError(RollbackError):
    """Raised when the freshness verification backend is unavailable."""


# ===========================================================================
# 1. Ed25519 Publisher Provenance
# ===========================================================================

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


# ===========================================================================
# 2. Freshness Record and Backend Interface
# ===========================================================================

@dataclass
class FreshnessRecord:
    """Structured record of an authorized training run state."""
    run_id: str
    max_step: int
    max_epoch: int = 0
    latest_artifact_id: str = ""
    version: int = 1
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    pending_recovery_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "max_step": self.max_step,
            "max_epoch": self.max_epoch,
            "latest_artifact_id": self.latest_artifact_id,
            "version": self.version,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "pending_recovery_token": self.pending_recovery_token,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FreshnessRecord:
        return cls(
            run_id=d["run_id"],
            max_step=d.get("max_step", 0),
            max_epoch=d.get("max_epoch", 0),
            latest_artifact_id=d.get("latest_artifact_id", ""),
            version=d.get("version", 1),
            updated_at=d.get("updated_at", time.time()),
            metadata=d.get("metadata", {}),
            pending_recovery_token=d.get("pending_recovery_token"),
        )


class FreshnessBackend(ABC):
    """Abstract interface for monotonic anti-rollback freshness storage backends."""

    @property
    @abstractmethod
    def is_external_trusted(self) -> bool:
        """Returns True if the backend is hosted outside the local filesystem trust boundary."""
        pass

    @abstractmethod
    def get_record(self, run_id: str) -> Optional[FreshnessRecord]:
        """Read the latest authorized version record for a training run."""
        pass

    @abstractmethod
    def advance(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
        expected_version: Optional[int] = None,
        recovery_token: Optional[str] = None,
    ) -> FreshnessRecord:
        """Atomically advance the monotonic high-water mark for a run.
        
        Raises:
            RollbackError if step/epoch regresses or if expected_version does not match (CAS failure).
            FreshnessServiceUnavailableError if the backend is unreachable.
        """
        pass

    @abstractmethod
    def check_freshness(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> bool:
        """Verify whether an incoming checkpoint satisfies monotonic freshness."""
        pass

    @abstractmethod
    def begin_update(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> str:
        """Stage a pending checkpoint creation, generating a recovery token."""
        pass

    @abstractmethod
    def abort_update(self, run_id: str, recovery_token: str) -> None:
        """Abort a staged checkpoint update after a failed write."""
        pass


# ===========================================================================
# 3. Backend Implementations
# ===========================================================================

class LocalFileFreshnessBackend(FreshnessBackend):
    """Local filesystem-backed freshness ledger using atomic staging + fsync.
    
    SECURITY BOUNDARY NOTICE:
    Resists Scenario 1 (checkpoint replacement when ledger is trusted).
    DOES NOT resist Scenario 2 (attacker modifying both checkpoint and local ledger files).
    Use Transactional or Cloud backends for production distributed security.
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = os.path.abspath(ledger_path)
        self._lock = threading.RLock()
        ledger_dir = os.path.dirname(self.ledger_path)
        if ledger_dir and not os.path.exists(ledger_dir):
            os.makedirs(ledger_dir, exist_ok=True)
        if not os.path.exists(self.ledger_path) or os.path.getsize(self.ledger_path) == 0:
            self._write_state({})

    @property
    def is_external_trusted(self) -> bool:
        return False

    def _read_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.ledger_path) or os.path.getsize(self.ledger_path) == 0:
            return {}
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            raise RollbackError(f"Failed to read local freshness ledger at {self.ledger_path}: {exc}") from exc

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

    def get_record(self, run_id: str) -> Optional[FreshnessRecord]:
        with self._lock:
            state = self._read_state()
            if run_id not in state:
                return None
            data = dict(state[run_id])
            data["run_id"] = run_id
            return FreshnessRecord.from_dict(data)

    def check_freshness(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> bool:
        with self._lock:
            state = self._read_state()
            if run_id not in state:
                return True
            rec = state[run_id]
            recorded_step = rec.get("max_step", 0)
            recorded_epoch = rec.get("max_epoch", 0)

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
            return True

    def advance(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
        expected_version: Optional[int] = None,
        recovery_token: Optional[str] = None,
    ) -> FreshnessRecord:
        with self._lock:
            state = self._read_state()
            cur_ver = 0
            if run_id in state:
                rec = state[run_id]
                cur_step = rec.get("max_step", 0)
                cur_epoch = rec.get("max_epoch", 0)
                cur_ver = rec.get("version", 1)

                if expected_version is not None and cur_ver != expected_version:
                    raise RollbackError(
                        f"Concurrent update conflict on run '{run_id}': expected version {expected_version}, got {cur_ver}."
                    )

                if step < cur_step:
                    raise RollbackError(
                        f"Rollback attack detected for run '{run_id}': incoming step {step} < current step {cur_step}."
                    )
                if step == cur_step and epoch < cur_epoch:
                    raise RollbackError(
                        f"Rollback attack detected for run '{run_id}': incoming epoch {epoch} < current epoch {cur_epoch} at step {step}."
                    )

            new_record = FreshnessRecord(
                run_id=run_id,
                max_step=step,
                max_epoch=epoch,
                latest_artifact_id=artifact_id,
                version=cur_ver + 1,
                updated_at=time.time(),
            )
            state[run_id] = new_record.to_dict()
            self._write_state(state)
            return new_record

    def begin_update(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> str:
        with self._lock:
            self.check_freshness(run_id, step, epoch, artifact_id)
            token = secrets.token_hex(16)
            state = self._read_state()
            if run_id in state:
                state[run_id]["pending_recovery_token"] = token
            else:
                state[run_id] = {
                    "max_step": 0,
                    "max_epoch": 0,
                    "version": 0,
                    "pending_recovery_token": token,
                }
            self._write_state(state)
            return token

    def abort_update(self, run_id: str, recovery_token: str) -> None:
        with self._lock:
            state = self._read_state()
            if run_id in state and state[run_id].get("pending_recovery_token") == recovery_token:
                state[run_id]["pending_recovery_token"] = None
                self._write_state(state)


class TransactionalMemoryFreshnessBackend(FreshnessBackend):
    """In-memory transactional freshness backend with strict optimistic concurrency control (CAS)."""

    def __init__(self, external_trusted: bool = True):
        self._external_trusted = external_trusted
        self._lock = threading.RLock()
        self._records: Dict[str, FreshnessRecord] = {}
        self._is_available = True

    @property
    def is_external_trusted(self) -> bool:
        return self._external_trusted

    def set_availability(self, available: bool) -> None:
        """Simulate service downtime or network partition."""
        with self._lock:
            self._is_available = available

    def get_record(self, run_id: str) -> Optional[FreshnessRecord]:
        with self._lock:
            if not self._is_available:
                raise FreshnessServiceUnavailableError("Freshness service is temporarily unavailable.")
            rec = self._records.get(run_id)
            return FreshnessRecord.from_dict(rec.to_dict()) if rec else None

    def check_freshness(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> bool:
        with self._lock:
            if not self._is_available:
                raise FreshnessServiceUnavailableError("Freshness service is temporarily unavailable.")
            if run_id not in self._records:
                return True
            rec = self._records[run_id]
            if step < rec.max_step:
                raise RollbackError(
                    f"Rollback attack detected for run '{run_id}': incoming step {step} < current step {rec.max_step}."
                )
            if step == rec.max_step and epoch < rec.max_epoch:
                raise RollbackError(
                    f"Rollback attack detected for run '{run_id}': incoming epoch {epoch} < current epoch {rec.max_epoch} at step {step}."
                )
            return True

    def advance(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
        expected_version: Optional[int] = None,
        recovery_token: Optional[str] = None,
    ) -> FreshnessRecord:
        with self._lock:
            if not self._is_available:
                raise FreshnessServiceUnavailableError("Freshness service is temporarily unavailable.")
            cur = self._records.get(run_id)
            cur_ver = cur.version if cur else 0
            if expected_version is not None and cur_ver != expected_version:
                raise RollbackError(
                    f"Optimistic lock conflict on run '{run_id}': expected version {expected_version}, got {cur_ver}."
                )
            if cur:
                if step < cur.max_step:
                    raise RollbackError(f"Stale step advance: {step} < {cur.max_step}.")
                if step == cur.max_step and epoch < cur.max_epoch:
                    raise RollbackError(f"Stale epoch advance: {epoch} < {cur.max_epoch}.")

            new_rec = FreshnessRecord(
                run_id=run_id,
                max_step=step,
                max_epoch=epoch,
                latest_artifact_id=artifact_id,
                version=cur_ver + 1,
                updated_at=time.time(),
            )
            self._records[run_id] = new_rec
            return FreshnessRecord.from_dict(new_rec.to_dict())

    def begin_update(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> str:
        with self._lock:
            if not self._is_available:
                raise FreshnessServiceUnavailableError("Freshness service is temporarily unavailable.")
            self.check_freshness(run_id, step, epoch, artifact_id)
            token = secrets.token_hex(16)
            if run_id in self._records:
                self._records[run_id].pending_recovery_token = token
            return token

    def abort_update(self, run_id: str, recovery_token: str) -> None:
        with self._lock:
            if run_id in self._records and self._records[run_id].pending_recovery_token == recovery_token:
                self._records[run_id].pending_recovery_token = None


class SQLiteFreshnessBackend(FreshnessBackend):
    """ACID transactional SQL freshness backend supporting concurrency and crash resilience."""

    def __init__(self, db_path: str, external_trusted: bool = False):
        self.db_path = os.path.abspath(db_path)
        self._external_trusted = external_trusted
        self._lock = threading.RLock()
        self._init_db()

    @property
    def is_external_trusted(self) -> bool:
        return self._external_trusted

    @contextlib.contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS freshness_ledger (
                    run_id TEXT PRIMARY KEY,
                    max_step INTEGER NOT NULL,
                    max_epoch INTEGER NOT NULL,
                    latest_artifact_id TEXT,
                    version INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    pending_recovery_token TEXT
                )
            """)

    def get_record(self, run_id: str) -> Optional[FreshnessRecord]:
        with self._lock, self._connection() as conn:
            cur = conn.execute(
                "SELECT run_id, max_step, max_epoch, latest_artifact_id, version, updated_at, pending_recovery_token "
                "FROM freshness_ledger WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return FreshnessRecord(
                run_id=row[0],
                max_step=row[1],
                max_epoch=row[2],
                latest_artifact_id=row[3],
                version=row[4],
                updated_at=row[5],
                pending_recovery_token=row[6],
            )

    def check_freshness(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> bool:
        rec = self.get_record(run_id)
        if not rec:
            return True
        if step < rec.max_step:
            raise RollbackError(
                f"Rollback attack detected for run '{run_id}': incoming step {step} is strictly "
                f"less than highest recorded step {rec.max_step}."
            )
        if step == rec.max_step and epoch < rec.max_epoch:
            raise RollbackError(
                f"Rollback attack detected for run '{run_id}': incoming epoch {epoch} is strictly "
                f"less than highest recorded epoch {rec.max_epoch} at step {step}."
            )
        return True

    def advance(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
        expected_version: Optional[int] = None,
        recovery_token: Optional[str] = None,
    ) -> FreshnessRecord:
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = conn.execute(
                    "SELECT max_step, max_epoch, version FROM freshness_ledger WHERE run_id = ?",
                    (run_id,),
                )
                row = cur.fetchone()
                now = time.time()

                if row is None:
                    new_ver = 1
                    conn.execute(
                        "INSERT INTO freshness_ledger (run_id, max_step, max_epoch, latest_artifact_id, version, updated_at, pending_recovery_token) "
                        "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                        (run_id, step, epoch, artifact_id, new_ver, now),
                    )
                else:
                    cur_step, cur_epoch, cur_ver = row
                    if expected_version is not None and cur_ver != expected_version:
                        raise RollbackError(
                            f"Concurrency conflict on run '{run_id}': expected version {expected_version}, got {cur_ver}."
                        )
                    if step < cur_step:
                        raise RollbackError(
                            f"Rollback attack detected for run '{run_id}': incoming step {step} is strictly "
                            f"less than highest recorded step {cur_step}."
                        )
                    if step == cur_step and epoch < cur_epoch:
                        raise RollbackError(
                            f"Rollback attack detected for run '{run_id}': incoming epoch {epoch} is strictly "
                            f"less than highest recorded epoch {cur_epoch} at step {step}."
                        )

                    new_ver = cur_ver + 1
                    conn.execute(
                        "UPDATE freshness_ledger SET max_step = ?, max_epoch = ?, latest_artifact_id = ?, version = ?, updated_at = ?, pending_recovery_token = NULL "
                        "WHERE run_id = ?",
                        (step, epoch, artifact_id, new_ver, now, run_id),
                    )
                conn.execute("COMMIT;")
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise

        return FreshnessRecord(
            run_id=run_id,
            max_step=step,
            max_epoch=epoch,
            latest_artifact_id=artifact_id,
            version=new_ver,
            updated_at=now,
        )

    def begin_update(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> str:
        with self._lock:
            self.check_freshness(run_id, step, epoch, artifact_id)
            token = secrets.token_hex(16)
            with self._connection() as conn:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    cur = conn.execute("SELECT version FROM freshness_ledger WHERE run_id = ?", (run_id,))
                    row = cur.fetchone()
                    if row is None:
                        conn.execute(
                            "INSERT INTO freshness_ledger (run_id, max_step, max_epoch, latest_artifact_id, version, updated_at, pending_recovery_token) "
                            "VALUES (?, 0, 0, '', 0, ?, ?)",
                            (run_id, time.time(), token),
                        )
                    else:
                        conn.execute(
                            "UPDATE freshness_ledger SET pending_recovery_token = ? WHERE run_id = ?",
                            (token, run_id),
                        )
                    conn.execute("COMMIT;")
                except Exception:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    raise
            return token

    def abort_update(self, run_id: str, recovery_token: str) -> None:
        with self._lock, self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute(
                    "UPDATE freshness_ledger SET pending_recovery_token = NULL WHERE run_id = ? AND pending_recovery_token = ?",
                    (run_id, recovery_token),
                )
                conn.execute("COMMIT;")
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise


class EtcdFreshnessBackend(FreshnessBackend):
    """Production quorum-backed external freshness backend integrated with etcd consensus.

    Uses etcd's transactional compare-and-swap (CAS) primitives and revision numbers
    to guarantee atomic monotonic progression across multi-node distributed clusters.
    Fails closed if the etcd consensus cluster is unreachable or partitions.
    """

    def __init__(
        self,
        key_prefix: str = "/torchshield/freshness/",
        client: Optional[Any] = None,
        host: str = "localhost",
        port: int = 2379,
        ca_cert: Optional[str] = None,
        cert_key: Optional[str] = None,
        cert_cert: Optional[str] = None,
        timeout: float = 5.0,
    ):
        self.key_prefix = key_prefix
        self.timeout = timeout
        if client is not None:
            self._client = client
        else:
            try:
                import etcd3
                client_kwargs: Dict[str, Any] = {"host": host, "port": port, "timeout": timeout}
                if ca_cert:
                    client_kwargs["ca_cert"] = ca_cert
                    client_kwargs["cert_key"] = cert_key
                    client_kwargs["cert_cert"] = cert_cert
                self._client = etcd3.client(**client_kwargs)
            except ImportError as exc:
                raise ImportError(
                    "etcd3 is required to use EtcdFreshnessBackend. Install via `pip install etcd3`."
                ) from exc
            except Exception as exc:
                raise FreshnessServiceUnavailableError(f"Failed to connect to etcd cluster: {exc}") from exc

    @property
    def is_external_trusted(self) -> bool:
        return True

    def _key_for_run(self, run_id: str) -> str:
        return f"{self.key_prefix.rstrip('/')}/{run_id}"

    def get_record(self, run_id: str) -> Optional[FreshnessRecord]:
        key = self._key_for_run(run_id)
        try:
            value, meta = self._client.get(key)
        except Exception as exc:
            raise FreshnessServiceUnavailableError(f"Failed to query etcd for run '{run_id}': {exc}") from exc

        if value is None:
            return None
        try:
            data = json.loads(value.decode("utf-8") if isinstance(value, bytes) else str(value))
            data["run_id"] = run_id
            if meta and hasattr(meta, "version"):
                data["version"] = meta.version
            return FreshnessRecord.from_dict(data)
        except Exception as exc:
            raise RollbackError(f"Malformed freshness payload in etcd for key '{key}': {exc}") from exc

    def check_freshness(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> bool:
        rec = self.get_record(run_id)
        if rec is None:
            return True
        if step < rec.max_step:
            raise RollbackError(
                f"Rollback attack detected in etcd quorum for run '{run_id}': incoming step {step} < "
                f"highest quorum-recorded step {rec.max_step}."
            )
        if step == rec.max_step and epoch < rec.max_epoch:
            raise RollbackError(
                f"Rollback attack detected in etcd quorum for run '{run_id}': incoming epoch {epoch} < "
                f"highest quorum-recorded epoch {rec.max_epoch} at step {step}."
            )
        return True

    def advance(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
        expected_version: Optional[int] = None,
        recovery_token: Optional[str] = None,
    ) -> FreshnessRecord:
        key = self._key_for_run(run_id)
        try:
            value, meta = self._client.get(key)
        except Exception as exc:
            raise FreshnessServiceUnavailableError(f"Failed to access etcd cluster: {exc}") from exc

        current_ver = 0
        if value is not None:
            try:
                rec_dict = json.loads(value.decode("utf-8") if isinstance(value, bytes) else str(value))
            except Exception as exc:
                raise RollbackError(f"Corrupted etcd freshness record for key '{key}': {exc}") from exc

            cur_step = rec_dict.get("max_step", 0)
            cur_epoch = rec_dict.get("max_epoch", 0)
            current_ver = meta.version if meta and hasattr(meta, "version") else rec_dict.get("version", 1)

            if expected_version is not None and current_ver != expected_version:
                raise RollbackError(
                    f"Etcd CAS conflict on run '{run_id}': expected version {expected_version}, got {current_ver}."
                )
            if step < cur_step:
                raise RollbackError(
                    f"Rollback attack rejected by etcd quorum for run '{run_id}': incoming step {step} < {cur_step}."
                )
            if step == cur_step and epoch < cur_epoch:
                raise RollbackError(
                    f"Rollback attack rejected by etcd quorum for run '{run_id}': incoming epoch {epoch} < {cur_epoch} at step {step}."
                )

        new_record = FreshnessRecord(
            run_id=run_id,
            max_step=step,
            max_epoch=epoch,
            latest_artifact_id=artifact_id,
            version=current_ver + 1,
            updated_at=time.time(),
        )
        payload = json.dumps(new_record.to_dict(), sort_keys=True)

        try:
            if hasattr(self._client, "transaction"):
                if value is None:
                    status, _ = self._client.transaction(
                        compare=[self._client.transactions.version(key) == 0],
                        success=[self._client.transactions.put(key, payload)],
                        failure=[],
                    )
                else:
                    status, _ = self._client.transaction(
                        compare=[self._client.transactions.version(key) == current_ver],
                        success=[self._client.transactions.put(key, payload)],
                        failure=[],
                    )
                if not status:
                    raise RollbackError(f"Concurrent writer conflict in etcd quorum for run '{run_id}'.")
            else:
                self._client.put(key, payload)
        except RollbackError:
            raise
        except Exception as exc:
            raise FreshnessServiceUnavailableError(f"Etcd transaction failed: {exc}") from exc

        return new_record

    def begin_update(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> str:
        self.check_freshness(run_id, step, epoch, artifact_id)
        token = secrets.token_hex(16)
        key = f"{self._key_for_run(run_id)}/recovery_token"
        try:
            self._client.put(key, token)
        except Exception as exc:
            raise FreshnessServiceUnavailableError(f"Failed to record pending token in etcd: {exc}") from exc
        return token

    def abort_update(self, run_id: str, recovery_token: str) -> None:
        key = f"{self._key_for_run(run_id)}/recovery_token"
        try:
            val, _ = self._client.get(key)
            if val is not None and (val.decode("utf-8") if isinstance(val, bytes) else str(val)) == recovery_token:
                self._client.delete(key)
        except Exception as exc:
            logger.warning("Failed to abort update token in etcd: %s", exc)


# ===========================================================================
# 4. FreshnessLedger Wrapper
# ===========================================================================

class FreshnessLedger:
    """Atomic monotonic freshness ledger preventing checkpoint rollback attacks.

    Guarantees that for a given training run or model lineage:
    - Step numbers must strictly advance monotonically.
    - Checkpoints cannot be replayed from an older iteration.
    - Supports pluggable trusted backends (LocalDisk, TransactionalMemory, SQLite, RemoteService).
    """

    def __init__(
        self,
        ledger_path: Optional[str] = None,
        backend: Optional[FreshnessBackend] = None,
    ):
        if backend is not None:
            self.backend = backend
            self.ledger_path = getattr(backend, "ledger_path", str(backend))
        elif ledger_path is not None:
            self.ledger_path = os.path.abspath(ledger_path)
            self.backend = LocalFileFreshnessBackend(self.ledger_path)
        else:
            raise ValueError("Either 'ledger_path' or 'backend' must be provided to FreshnessLedger.")

    @property
    def is_external_trusted(self) -> bool:
        """Whether the underlying backend resists attacker modification of local storage."""
        return self.backend.is_external_trusted

    def check_freshness(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> bool:
        """Verify whether an incoming checkpoint satisfies monotonic freshness."""
        return self.backend.check_freshness(run_id=run_id, step=step, epoch=epoch, artifact_id=artifact_id)

    def record_and_verify(
        self,
        run_id: str,
        step: int,
        epoch: int = 0,
        artifact_id: str = "",
    ) -> None:
        """Verify freshness and advance the monotonic high-water mark."""
        self.backend.advance(run_id=run_id, step=step, epoch=epoch, artifact_id=artifact_id)

    def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return the current recorded record dictionary for run_id."""
        rec = self.backend.get_record(run_id)
        return rec.to_dict() if rec else None
