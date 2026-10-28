"""Tests for EtcdFreshnessBackend quorum-backed anti-rollback consensus."""
import json
import pytest

from torchshield.provenance import (
    EtcdFreshnessBackend,
    RollbackError,
    FreshnessServiceUnavailableError,
)


class MockEtcdClient:
    def __init__(self):
        self.store = {}
        self.versions = {}
        self.should_fail = False

        class Transactions:
            def __init__(self, outer):
                self.outer = outer

            def version(self, key):
                return self.outer.versions.get(key, 0)

            def put(self, key, value):
                def op():
                    self.outer.store[key] = value.encode("utf-8") if isinstance(value, str) else value
                    self.outer.versions[key] = self.outer.versions.get(key, 0) + 1
                return op

        self.transactions = Transactions(self)

    def get(self, key):
        if self.should_fail:
            raise RuntimeError("Simulated network partition from etcd cluster!")
        val = self.store.get(key)
        ver = self.versions.get(key, 0)

        class Meta:
            def __init__(self, v):
                self.version = v

        return val, Meta(ver)

    def transaction(self, compare, success, failure):
        if self.should_fail:
            raise RuntimeError("Simulated network partition during etcd transaction!")
        # Evaluate condition
        passed = all(compare)
        if passed:
            for op in success:
                op()
            return True, None
        else:
            for op in failure:
                op()
            return False, None

    def delete(self, key):
        self.store.pop(key, None)
        self.versions.pop(key, None)


def test_etcd_freshness_advance_and_monotonicity():
    mock_client = MockEtcdClient()
    backend = EtcdFreshnessBackend(client=mock_client)
    assert backend.is_external_trusted is True

    # Advance run_1 to step 10
    rec = backend.advance(run_id="run_1", step=10, epoch=1, artifact_id="art_1")
    assert rec.max_step == 10
    assert rec.version == 1

    # Check freshness
    assert backend.check_freshness("run_1", step=10, epoch=1) is True
    assert backend.check_freshness("run_1", step=11, epoch=1) is True

    # Stale step rollback attack rejected
    with pytest.raises(RollbackError, match="Rollback attack"):
        backend.check_freshness("run_1", step=9, epoch=1)

    with pytest.raises(RollbackError, match="Rollback attack"):
        backend.advance("run_1", step=8, epoch=1)

    # Valid advance to step 20
    rec2 = backend.advance("run_1", step=20, epoch=2, artifact_id="art_2")
    assert rec2.max_step == 20
    assert rec2.version == 2


def test_etcd_freshness_cas_conflict():
    mock_client = MockEtcdClient()
    backend = EtcdFreshnessBackend(client=mock_client)

    backend.advance("run_cas", step=10, epoch=1)

    # CAS conflict with wrong expected version
    with pytest.raises(RollbackError, match="Etcd CAS conflict"):
        backend.advance("run_cas", step=15, epoch=1, expected_version=99)


def test_etcd_freshness_network_outage_fails_closed():
    mock_client = MockEtcdClient()
    backend = EtcdFreshnessBackend(client=mock_client)

    mock_client.should_fail = True
    with pytest.raises(FreshnessServiceUnavailableError, match="Failed to access etcd cluster"):
        backend.advance("run_outage", step=10, epoch=1)
