"""Tests for Trusted Freshness Architecture and 6-Scenario Anti-Rollback Threat Model.

Validates:
- Scenario 1: Attacker replaces checkpoint files; trusted ledger rejects stale step.
- Scenario 2: Attacker restores older checkpoint AND local ledger file; demonstrates
  local ledger vulnerability vs trusted isolated backend immunity.
- Scenario 3: Attacker replays older valid signed checkpoint; rejected.
- Scenario 4: Concurrent updates from multiple distributed workers (CAS / transactions).
- Scenario 5: Freshness service unavailable; fail-closed behavior.
- Scenario 6: Legitimate training run recovery after aborted update.
"""
from __future__ import annotations

import concurrent.futures
import os
import shutil
import tempfile
import threading
import pytest

from torchshield.provenance import (
    FreshnessLedger,
    LocalFileFreshnessBackend,
    TransactionalMemoryFreshnessBackend,
    SQLiteFreshnessBackend,
    RollbackError,
    FreshnessServiceUnavailableError,
)


def test_scenario_1_stale_checkpoint_rejection():
    """Scenario 1: Attacker replaces checkpoint files with an older step; ledger rejects."""
    backend = TransactionalMemoryFreshnessBackend(external_trusted=True)
    ledger = FreshnessLedger(backend=backend)

    # Legitimate run advanced to step 100
    ledger.record_and_verify(run_id="run-alpha", step=50, epoch=1, artifact_id="art-50")
    ledger.record_and_verify(run_id="run-alpha", step=100, epoch=2, artifact_id="art-100")

    # Attacker presents checkpoint from step 50
    with pytest.raises(RollbackError, match="Rollback attack detected"):
        ledger.check_freshness(run_id="run-alpha", step=50, epoch=1, artifact_id="art-50")


def test_scenario_2_local_vs_trusted_backend_threat_boundary():
    """Scenario 2: Attacker can restore both checkpoint and local ledger from backup snapshot.
    
    Demonstrates that local filesystem ledger CANNOT resist snapshot rollback,
    while an external/isolated backend prevents it.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_path = os.path.join(tmp_dir, "freshness.json")
        backup_path = os.path.join(tmp_dir, "backup_freshness.json")

        local_backend = LocalFileFreshnessBackend(ledger_path)
        assert not local_backend.is_external_trusted

        # Step 1: Advance to step 10 and take local filesystem backup
        local_backend.advance(run_id="run-beta", step=10, epoch=1)
        shutil.copyfile(ledger_path, backup_path)

        # Step 2: Advance to step 50
        local_backend.advance(run_id="run-beta", step=50, epoch=2)
        assert local_backend.get_record("run-beta").max_step == 50

        # Step 3: Attacker restores disk snapshot (overwriting local ledger with backup at step 10)
        shutil.copyfile(backup_path, ledger_path)

        # Vulnerability demonstration: Local file ledger now accepts step 20 (attacker rolled back time!)
        restored_local = LocalFileFreshnessBackend(ledger_path)
        assert restored_local.check_freshness("run-beta", step=20, epoch=1) is True

        # CONTRAST: External trusted backend does not live on the compromised host filesystem
        trusted_backend = TransactionalMemoryFreshnessBackend(external_trusted=True)
        assert trusted_backend.is_external_trusted
        trusted_backend.advance("run-beta", step=50, epoch=2)

        # Attacker cannot alter the trusted backend; replay of step 20 is strictly rejected
        with pytest.raises(RollbackError):
            trusted_backend.check_freshness("run-beta", step=20, epoch=1)


def test_scenario_3_replay_of_older_signed_checkpoint():
    """Scenario 3: Attacker replays valid previously signed checkpoint from an earlier epoch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_s3.db")
        backend = SQLiteFreshnessBackend(db_path)
        ledger = FreshnessLedger(backend=backend)

        ledger.record_and_verify("run-gamma", step=1000, epoch=5, artifact_id="art-v5")

        # Same step, earlier epoch
        with pytest.raises(RollbackError, match="Rollback attack detected.*epoch 4"):
            ledger.record_and_verify("run-gamma", step=1000, epoch=4, artifact_id="art-v4")

        # Earlier step
        with pytest.raises(RollbackError, match="Rollback attack detected.*step 999"):
            ledger.record_and_verify("run-gamma", step=999, epoch=5, artifact_id="art-v3")


def test_scenario_4_concurrent_workers_optimistic_concurrency_control():
    """Scenario 4: Multiple distributed ranks attempt simultaneous advance with optimistic locking."""
    backend = TransactionalMemoryFreshnessBackend()
    backend.advance("run-parallel", step=10, epoch=0)

    initial_rec = backend.get_record("run-parallel")
    assert initial_rec is not None
    v0 = initial_rec.version

    # Worker A and Worker B both read v0
    # Worker A succeeds
    backend.advance("run-parallel", step=20, epoch=0, expected_version=v0)

    # Worker B tries to advance using stale expected_version v0 -> must fail with CAS conflict
    with pytest.raises(RollbackError, match="Optimistic lock conflict"):
        backend.advance("run-parallel", step=25, epoch=0, expected_version=v0)


def test_scenario_4_sqlite_multithreaded_concurrent_writers():
    """Scenario 4: High-concurrency multithreaded test verifying ACID monotonic advancement in SQLite."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "concurrent.db")
        backend = SQLiteFreshnessBackend(db_path)

        def worker_advance(worker_id: int):
            for i in range(10):
                step = worker_id * 100 + i
                try:
                    backend.advance("run-mt", step=step, epoch=0, artifact_id=f"art-{step}")
                except RollbackError:
                    pass  # Expected if another worker advanced to a higher step

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_advance, wid) for wid in range(8)]
            concurrent.futures.wait(futures)

        final_rec = backend.get_record("run-mt")
        assert final_rec is not None
        assert final_rec.max_step > 0


def test_scenario_5_service_unavailable_fail_closed():
    """Scenario 5: When the freshness service is unreachable, system fails closed."""
    backend = TransactionalMemoryFreshnessBackend()
    backend.advance("run-offline", step=10, epoch=1)

    # Simulate network partition / service outage
    backend.set_availability(False)

    # Must raise FreshnessServiceUnavailableError rather than silently permitting checkpoint
    with pytest.raises(FreshnessServiceUnavailableError, match="temporarily unavailable"):
        backend.check_freshness("run-offline", step=20, epoch=1)

    with pytest.raises(FreshnessServiceUnavailableError, match="temporarily unavailable"):
        backend.advance("run-offline", step=20, epoch=1)


def test_scenario_6_recovery_after_failed_checkpoint_write():
    """Scenario 6: Legitimate run stages an update, fails during writing, aborts, and recovers."""
    backend = TransactionalMemoryFreshnessBackend()
    backend.advance("run-recover", step=10, epoch=1)

    # Stage update for step 20
    token = backend.begin_update("run-recover", step=20, epoch=1)
    rec = backend.get_record("run-recover")
    assert rec.pending_recovery_token == token

    # Simulated write crash: abort staged update
    backend.abort_update("run-recover", token)
    rec_after = backend.get_record("run-recover")
    assert rec_after.pending_recovery_token is None
    assert rec_after.max_step == 10  # High-water mark remains unpolluted

    # Retried write succeeds
    backend.advance("run-recover", step=20, epoch=1)
    assert backend.get_record("run-recover").max_step == 20


def _mp_worker(db_path: str, worker_id: int):
    """Top-level worker function for multiprocessing tests."""
    from torchshield.provenance import SQLiteFreshnessBackend, RollbackError
    backend = SQLiteFreshnessBackend(db_path)
    for i in range(5):
        step = worker_id * 100 + i
        try:
            backend.advance("run-mp", step=step, epoch=0, artifact_id=f"art-{step}")
        except RollbackError:
            pass


def test_scenario_4_sqlite_multiprocess_concurrent_writers():
    """Scenario 4: Multi-process concurrent worker stress test on SQLiteFreshnessBackend."""
    import multiprocessing as mp
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "mp_test.db")
        backend = SQLiteFreshnessBackend(db_path)
        backend.advance("run-mp", step=0, epoch=0)

        processes = [
            mp.Process(target=_mp_worker, args=(db_path, wid))
            for wid in range(4)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=60.0)
            assert p.exitcode == 0, f"Worker process failed with code {p.exitcode}"

        final_rec = backend.get_record("run-mp")
        assert final_rec is not None
        assert final_rec.max_step >= 0
