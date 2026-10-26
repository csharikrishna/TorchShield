"""Tests for anti-rollback monotonic freshness ledger."""
import os
import tempfile
import pytest

from ahcc.provenance import FreshnessLedger, RollbackError


def test_freshness_ledger_advancement_and_rollback_detection():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        ledger_path = tmp.name

    try:
        ledger = FreshnessLedger(ledger_path)
        run_id = "experiment-llm-pretraining-01"

        # 1. First checkpoint at step 1000
        ledger.record_and_verify(run_id=run_id, step=1000, epoch=1, artifact_id="art-1000")
        assert ledger.get_run_info(run_id)["max_step"] == 1000

        # 2. Advance to step 2000
        ledger.record_and_verify(run_id=run_id, step=2000, epoch=2, artifact_id="art-2000")
        assert ledger.get_run_info(run_id)["max_step"] == 2000

        # 3. Rollback attack: attempt to record step 1500
        with pytest.raises(RollbackError, match="Rollback attack detected"):
            ledger.record_and_verify(run_id=run_id, step=1500, epoch=1, artifact_id="art-1500")

        # 4. Rollback check only
        with pytest.raises(RollbackError, match="Rollback attack detected"):
            ledger.check_freshness(run_id=run_id, step=500, epoch=0)

        # 5. Advancement to step 3000 succeeds
        assert ledger.check_freshness(run_id=run_id, step=3000, epoch=3) is True
        ledger.record_and_verify(run_id=run_id, step=3000, epoch=3, artifact_id="art-3000")
        assert ledger.get_run_info(run_id)["max_step"] == 3000

        # 6. Independent run IDs do not interfere
        run_b = "experiment-finetune-lora"
        ledger.record_and_verify(run_id=run_b, step=50, epoch=1, artifact_id="lora-50")
        assert ledger.get_run_info(run_b)["max_step"] == 50
        assert ledger.get_run_info(run_id)["max_step"] == 3000
    finally:
        if os.path.exists(ledger_path):
            os.remove(ledger_path)
