"""Tests for Phase One: Strict Model Validation, Transactional Restoration, Rollback Protection, and SafeStateProvider."""
import copy
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD

from torchshield.training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    TrainingProgress,
    TrainingStateError,
    SafeStateProvider,
    validate_model_checkpoint,
    validate_optimizer_checkpoint,
    validate_safe_metadata,
)
from torchshield.checkpoint import SecureModelCheckpoint


class NetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


class NetB_ExtraParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 2)
        self.extra = nn.Linear(2, 2)

    def forward(self, x):
        return self.extra(self.fc2(torch.relu(self.bn(self.fc1(x)))))


class NetC_ShapeMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)  # 16 instead of 8
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


def test_strict_validation_missing_and_unexpected_parameters():
    key = os.urandom(32)
    model = NetA()
    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(checkpoint_path=path, model=model, key=key)

        # 1. Loading into NetB_ExtraParam has missing param for NetB in strict mode
        net_b = NetB_ExtraParam()
        with pytest.raises(TrainingStateError, match="missing .* model parameter"):
            load_training_checkpoint(checkpoint_path=path, model=net_b, key=key, strict=True)

        # In flexible mode (strict=False), loading succeeds for existing params
        prog = load_training_checkpoint(checkpoint_path=path, model=net_b, key=key, strict=False)
        assert prog.global_step == 0

        # 2. Saving from NetB and loading into NetA causes unexpected parameter error in strict mode
        with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp2:
            path_b = tmp2.name
        try:
            save_training_checkpoint(output_path=path_b, model=net_b, key=key)
            net_a_dest = NetA()
            with pytest.raises(TrainingStateError, match="unexpected .* parameter"):
                load_training_checkpoint(input_path=path_b, model=net_a_dest, key=key, strict=True)
            # Flexible mode succeeds
            load_training_checkpoint(input_path=path_b, model=net_a_dest, key=key, strict=False)
        finally:
            if os.path.exists(path_b):
                os.remove(path_b)

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_strict_validation_shape_and_dtype_mismatches():
    key = os.urandom(32)
    model = NetA()
    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(output_path=path, model=model, key=key)

        # Destination model has different shape for fc1
        mismatched_model = NetC_ShapeMismatch()
        with pytest.raises(TrainingStateError, match="Shape mismatch for parameter"):
            load_training_checkpoint(input_path=path, model=mismatched_model, key=key, strict=True)

        # Shape mismatch must fail even if strict=False
        with pytest.raises(TrainingStateError, match="Shape mismatch for parameter"):
            load_training_checkpoint(input_path=path, model=mismatched_model, key=key, strict=False)

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_strict_optimizer_validation():
    key = os.urandom(32)
    model = NetA()
    opt = AdamW(model.parameters(), lr=1e-3)
    # Populate moments
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(output_path=path, model=model, optimizer=opt, key=key)

        # 1. Type mismatch: AdamW checkpoint into SGD optimizer
        sgd_opt = SGD(model.parameters(), lr=1e-3)
        with pytest.raises(TrainingStateError, match="Optimizer type mismatch"):
            load_training_checkpoint(input_path=path, model=model, optimizer=sgd_opt, key=key, strict=True)

        # 2. Param groups mismatch
        multi_group_opt = AdamW([
            {"params": [model.fc1.weight, model.fc1.bias], "lr": 1e-3},
            {"params": [model.fc2.weight, model.fc2.bias], "lr": 1e-4},
        ])
        with pytest.raises(TrainingStateError, match="Parameter group count mismatch"):
            load_training_checkpoint(input_path=path, model=model, optimizer=multi_group_opt, key=key, strict=True)

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_transactional_restoration_rollback_on_injected_error(monkeypatch):
    key = os.urandom(32)
    model_orig = NetA()
    opt_orig = AdamW(model_orig.parameters(), lr=1e-2)
    for _ in range(2):
        opt_orig.zero_grad()
        loss = model_orig(torch.randn(2, 4)).sum()
        loss.backward()
        opt_orig.step()

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(output_path=path, model=model_orig, optimizer=opt_orig, key=key)

        # Target model initialized with different random weights
        target_model = NetA()
        with torch.no_grad():
            for p in target_model.parameters():
                p.fill_(999.0)
            for b in target_model.buffers():
                b.fill_(888.0)
        target_opt = AdamW(target_model.parameters(), lr=1e-4)

        # Snapshot initial values
        init_weights = {n: p.clone() for n, p in target_model.named_parameters()}
        init_buffers = {n: b.clone() for n, b in target_model.named_buffers()}

        # Inject failure during Stage B (e.g. simulated failure in restore_optimizer_state)
        def faulty_restore_optimizer(*args, **kwargs):
            raise RuntimeError("Injected simulated crash during optimizer state restore!")

        import torchshield.training_state
        monkeypatch.setattr(torchshield.training_state, "restore_optimizer_state", faulty_restore_optimizer)

        with pytest.raises(TrainingStateError, match="Restoration failed; rolled back"):
            load_training_checkpoint(input_path=path, model=target_model, optimizer=target_opt, key=key)

        # Verify rollback succeeded: destination model weights and buffers are unchanged!
        for n, p in target_model.named_parameters():
            assert torch.equal(p, init_weights[n]), f"Parameter {n} was not rolled back!"
        for n, b in target_model.named_buffers():
            assert torch.equal(b, init_buffers[n]), f"Buffer {n} was not rolled back!"

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_safe_metadata_validation():
    # Valid primitives and nested containers
    valid_meta = {
        "step": 10,
        "lr": 0.001,
        "name": "run_1",
        "nested": {"layers": [1, 2, 3], "flag": True, "empty": None},
    }
    validate_safe_metadata(valid_meta)

    # Reject non-finite floats
    with pytest.raises(TrainingStateError, match="Non-finite float"):
        validate_safe_metadata({"bad_float": float("nan")})

    with pytest.raises(TrainingStateError, match="Non-finite float"):
        validate_safe_metadata({"bad_float": float("inf")})

    # Reject arbitrary Python objects
    class ArbitraryClass:
        pass

    with pytest.raises(TrainingStateError, match="Unsafe or non-serializable object"):
        validate_safe_metadata({"arbitrary": ArbitraryClass()})

    with pytest.raises(TrainingStateError, match="Unsafe or non-serializable object"):
        validate_safe_metadata({"callable": lambda x: x})


class CustomStateTracker(SafeStateProvider):
    def __init__(self, initial_count: int = 0):
        self.count = initial_count

    def get_state_key(self) -> str:
        return "custom_tracker"

    def capture_state(self):
        return {"count": self.count}

    def restore_state(self, state):
        self.count = state["count"]


def test_safe_state_provider_roundtrip():
    key = os.urandom(32)
    model = NetA()
    tracker_save = CustomStateTracker(initial_count=42)

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(
            output_path=path,
            model=model,
            key=key,
            state_providers=[tracker_save],
        )

        dest_model = NetA()
        tracker_load = CustomStateTracker(initial_count=0)

        load_training_checkpoint(
            input_path=path,
            model=dest_model,
            key=key,
            state_providers=[tracker_load],
        )

        assert tracker_load.count == 42
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_comprehensive_transactional_rollback_across_all_components(monkeypatch):
    """Verify that staged failures during restoration trigger full atomic rollback of
    model, optimizer, scheduler, scaler, and all RNG state generators.
    """
    import random
    import numpy as np

    key = os.urandom(32)
    model = NetA()
    optimizer = AdamW(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = torch.amp.GradScaler('cpu')

    # Step to advance optimizer, scheduler, scaler, and RNG
    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)

    x = torch.randn(2, 4)
    loss = model(x).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(
            output_path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            key=key,
        )

        # Create distinct destination state
        dest_model = NetA()
        with torch.no_grad():
            for p in dest_model.parameters():
                p.fill_(123.45)
            for b in dest_model.buffers():
                b.fill_(67.89)
        dest_opt = AdamW(dest_model.parameters(), lr=0.01)
        dest_sched = torch.optim.lr_scheduler.StepLR(dest_opt, step_size=10, gamma=0.1)
        dest_scaler = torch.amp.GradScaler('cpu')
        dest_scaler.scale(dest_model(torch.randn(2, 4)).sum()).backward()
        dest_scaler.step(dest_opt)
        dest_scaler.update()

        # Capture snapshots of destination state before failed load
        init_model_params = {n: p.clone() for n, p in dest_model.named_parameters()}
        init_model_bufs = {n: b.clone() for n, b in dest_model.named_buffers()}
        init_sched_state = copy.deepcopy(dest_sched.state_dict())
        init_scaler_state = copy.deepcopy(dest_scaler.state_dict())

        # Set unique initial RNG state
        random.seed(9999)
        np.random.seed(8888)
        torch.manual_seed(7777)
        init_py_rng = random.getstate()
        init_np_rng = np.random.get_state()
        init_torch_cpu_rng = torch.get_rng_state().clone()

        # Inject failure at Stage 6 (first call to RNG restoration fails, rollback succeeds)
        import torchshield.training_state as ts_mod
        real_restore_all_rng = ts_mod.restore_all_rng
        rng_calls = 0

        def faulty_restore_rng(*args, **kwargs):
            nonlocal rng_calls
            rng_calls += 1
            if rng_calls == 1:
                raise RuntimeError("Injected crash during RNG state restoration stage!")
            return real_restore_all_rng(*args, **kwargs)

        monkeypatch.setattr(ts_mod, "restore_all_rng", faulty_restore_rng)

        with pytest.raises(TrainingStateError, match="Restoration failed; rolled back all state mutations"):
            load_training_checkpoint(
                input_path=path,
                model=dest_model,
                optimizer=dest_opt,
                scheduler=dest_sched,
                scaler=dest_scaler,
                key=key,
                restore_rng=True,
            )

        # 1. Model parameters and buffers rolled back
        for n, p in dest_model.named_parameters():
            assert torch.equal(p, init_model_params[n]), f"Model parameter {n} failed rollback!"
        for n, b in dest_model.named_buffers():
            assert torch.equal(b, init_model_bufs[n]), f"Model buffer {n} failed rollback!"

        # 2. Scheduler state rolled back
        assert dest_sched.state_dict() == init_sched_state, "Scheduler state failed rollback!"

        # 3. Scaler state rolled back
        assert dest_scaler.state_dict() == init_scaler_state, "GradScaler state failed rollback!"

        # 4. RNG generators rolled back to initial destination seeds
        assert random.getstate() == init_py_rng, "Python RNG failed rollback!"
        assert np.random.get_state()[0] == init_np_rng[0], "NumPy RNG algorithm failed rollback!"
        assert np.array_equal(np.random.get_state()[1], init_np_rng[1]), "NumPy RNG keys failed rollback!"
        assert torch.equal(torch.get_rng_state(), init_torch_cpu_rng), "PyTorch CPU RNG failed rollback!"

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_transactional_rollback_on_scaler_failure(monkeypatch):
    """Verify that a failure during gradient scaler restoration rolls back model,
    optimizer, and scheduler state mutations.
    """
    key = os.urandom(32)
    model = NetA()
    optimizer = AdamW(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = torch.amp.GradScaler('cpu')

    x = torch.randn(2, 4)
    loss = model(x).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(
            output_path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            key=key,
        )

        dest_model = NetA()
        with torch.no_grad():
            for p in dest_model.parameters():
                p.fill_(444.0)
            for b in dest_model.buffers():
                b.fill_(333.0)
        dest_opt = AdamW(dest_model.parameters(), lr=0.02)
        dest_sched = torch.optim.lr_scheduler.StepLR(dest_opt, step_size=5, gamma=0.2)
        dest_scaler = torch.amp.GradScaler('cpu')

        init_params = {n: p.clone() for n, p in dest_model.named_parameters()}
        init_bufs = {n: b.clone() for n, b in dest_model.named_buffers()}
        init_sched = copy.deepcopy(dest_sched.state_dict())
        init_scaler = copy.deepcopy(dest_scaler.state_dict())

        # Inject failure in scaler load_state_dict on first invocation (restoration)
        calls = [0]
        real_load = dest_scaler.load_state_dict

        def faulty_scaler_load(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("Injected scaler crash!")
            return real_load(*args, **kwargs)

        monkeypatch.setattr(dest_scaler, "load_state_dict", faulty_scaler_load)

        with pytest.raises(TrainingStateError, match="Restoration failed; rolled back all state mutations"):
            load_training_checkpoint(
                input_path=path,
                model=dest_model,
                optimizer=dest_opt,
                scheduler=dest_sched,
                scaler=dest_scaler,
                key=key,
                restore_rng=False,
            )

        # Assert rollback of all earlier stages
        for n, p in dest_model.named_parameters():
            assert torch.equal(p, init_params[n]), f"Parameter {n} not rolled back!"
        for n, b in dest_model.named_buffers():
            assert torch.equal(b, init_bufs[n]), f"Buffer {n} not rolled back!"
        assert dest_sched.state_dict() == init_sched, "Scheduler not rolled back!"
        assert dest_scaler.state_dict() == init_scaler, "Scaler not rolled back!"

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_optimizer_rollback_preserves_parameter_identity_and_updates_model(monkeypatch):
    """P1 Finding 1 Regression Test:
    Verify that when restoration fails and triggers rollback, optimizer parameter groups
    maintain exact object identity ('is') with the destination model's parameters,
    and subsequent optimizer.step() properly mutates the model's weights.
    """
    key = os.urandom(32)
    model = NetA()
    optimizer = SGD(model.parameters(), lr=0.1)

    # Do a step to populate optimizer state
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(
            output_path=path,
            model=model,
            optimizer=optimizer,
            key=key,
        )

        dest_model = NetA()
        dest_opt = SGD(dest_model.parameters(), lr=0.05)

        # Record exact parameter object ids before attempting restoration
        original_model_params = list(dest_model.parameters())
        initial_opt_params = [p for grp in dest_opt.param_groups for p in grp["params"]]

        for orig_p, opt_p in zip(original_model_params, initial_opt_params):
            assert orig_p is opt_p, "Pre-condition: optimizer params must be identical to model params."

        # Monkeypatch restore_all_rng to raise an error during rollback trigger
        from torchshield import training_state
        real_restore_rng = training_state.restore_all_rng
        call_count = [0]

        def faulty_restore_rng(rng_dict):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure after optimizer restoration!")
            return real_restore_rng(rng_dict)

        monkeypatch.setattr(training_state, "restore_all_rng", faulty_restore_rng)

        with pytest.raises(TrainingStateError, match="Restoration failed; rolled back all state mutations"):
            load_training_checkpoint(
                input_path=path,
                model=dest_model,
                optimizer=dest_opt,
                key=key,
                restore_rng=True,
            )

        # 1. Verify exact object identity is preserved for every parameter
        rolled_back_opt_params = [p for grp in dest_opt.param_groups for p in grp["params"]]
        assert len(rolled_back_opt_params) == len(original_model_params)

        for model_p, opt_p in zip(dest_model.parameters(), rolled_back_opt_params):
            assert model_p is opt_p, "Optimizer parameter disconnected from model parameter during rollback!"

        # 2. Verify that a subsequent optimizer step updates the model parameters
        dest_opt.zero_grad()
        loss = dest_model(x).sum()
        loss.backward()

        before_step_weight = dest_model.fc1.weight.clone()
        dest_opt.step()
        after_step_weight = dest_model.fc1.weight.clone()

        assert not torch.equal(before_step_weight, after_step_weight), (
            "Optimizer step did not update model parameters! Parameter identity was broken."
        )

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_transactional_rollback_on_state_provider_failure(monkeypatch):
    """P1 Finding 3 Regression Test:
    Verify that auxiliary state providers are snapshotted and atomically rolled back
    if an error occurs during checkpoint restoration.
    """
    class TestStateProvider(SafeStateProvider):
        def __init__(self, key="test_provider", value=100):
            self.key = key
            self.value = value

        def get_state_key(self) -> str:
            return self.key

        def capture_state(self):
            return {"value": self.value}

        def restore_state(self, state):
            self.value = state["value"]

    key = os.urandom(32)
    model = NetA()
    src_provider = TestStateProvider(value=999)

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(
            output_path=path,
            model=model,
            key=key,
            state_providers=[src_provider],
        )

        dest_model = NetA()
        dest_provider = TestStateProvider(value=42)

        # Inject failure in restore_all_rng so it fails after state provider restore
        from torchshield import training_state
        real_restore_rng = training_state.restore_all_rng
        call_count = [0]

        def faulty_restore_rng(rng_dict):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure after provider restore!")
            return real_restore_rng(rng_dict)

        monkeypatch.setattr(training_state, "restore_all_rng", faulty_restore_rng)

        with pytest.raises(TrainingStateError, match="Restoration failed; rolled back all state mutations"):
            load_training_checkpoint(
                input_path=path,
                model=dest_model,
                key=key,
                state_providers=[dest_provider],
                restore_rng=True,
            )

        # Verify state provider was rolled back to its initial value (42)
        assert dest_provider.value == 42, f"State provider failed rollback: value is {dest_provider.value}, expected 42"

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_state_provider_failing_get_state_key():
    """Verify that a provider whose get_state_key() raises an exception fails fast
    with clear diagnostics without calling get_state_key() a second time.
    """
    call_count = [0]

    class BrokenKeyProvider(SafeStateProvider):
        def get_state_key(self) -> str:
            call_count[0] += 1
            raise RuntimeError("Key lookup catastrophic failure!")

        def capture_state(self):
            return {}

        def restore_state(self, state):
            pass

    key = os.urandom(32)
    model = NetA()
    broken_prov = BrokenKeyProvider()

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        # 1. During save, duplicate/failing key check fails fast
        with pytest.raises(TrainingStateError, match="Failed to obtain state key from provider index 0"):
            save_training_checkpoint(
                output_path=path,
                model=model,
                key=key,
                state_providers=[broken_prov],
            )
        assert call_count[0] == 1, "get_state_key() must NOT be called a second time in exception handling!"

        # 2. Reset count, save valid checkpoint, then test pre-validation on load
        call_count[0] = 0
        save_training_checkpoint(output_path=path, model=model, key=key)

        dest_model = NetA()
        with pytest.raises(TrainingStateError, match="Pre-validation failed for state provider index 0"):
            load_training_checkpoint(
                input_path=path,
                model=dest_model,
                key=key,
                state_providers=[broken_prov],
            )
        assert call_count[0] == 1, "get_state_key() must NOT be called a second time during load pre-validation!"

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_state_provider_duplicate_keys_rejected():
    """Verify duplicate state-provider keys are rejected before mutation or snapshot."""
    class DummyProvider(SafeStateProvider):
        def __init__(self, key):
            self._key = key

        def get_state_key(self) -> str:
            return self._key

        def capture_state(self):
            return {"k": 1}

        def restore_state(self, state):
            pass

    key = os.urandom(32)
    model = NetA()
    prov1 = DummyProvider("dup_key")
    prov2 = DummyProvider("dup_key")

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        # Save fails with duplicate key
        with pytest.raises(TrainingStateError, match="Duplicate SafeStateProvider key: 'dup_key'"):
            save_training_checkpoint(
                output_path=path,
                model=model,
                key=key,
                state_providers=[prov1, prov2],
            )

        # Save valid checkpoint
        save_training_checkpoint(output_path=path, model=model, key=key)

        # Load fails pre-validation with duplicate keys
        dest_model = NetA()
        with pytest.raises(TrainingStateError, match="Duplicate SafeStateProvider key: 'dup_key'"):
            load_training_checkpoint(
                input_path=path,
                model=dest_model,
                key=key,
                state_providers=[prov1, prov2],
            )

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_state_provider_restore_failure_and_rollback_distinction():
    """Verify that a failure during provider restore triggers rollback,
    and if rollback itself fails, the exception cleanly distinguishes
    the restoration failure from the rollback failure without masking either.
    """
    restore_calls = [0]

    class CustomStateProvider(SafeStateProvider):
        def __init__(self, key="custom_p", value=10, fail_on_restore=False, fail_on_rollback=False):
            self.key = key
            self.value = value
            self.fail_on_restore = fail_on_restore
            self.fail_on_rollback = fail_on_rollback

        def get_state_key(self) -> str:
            return self.key

        def capture_state(self):
            return {"value": self.value}

        def restore_state(self, state):
            restore_calls[0] += 1
            if restore_calls[0] == 1 and self.fail_on_restore:
                raise RuntimeError("Injected restore failure in provider!")
            if restore_calls[0] > 1 and self.fail_on_rollback:
                raise RuntimeError("Injected secondary rollback failure in provider!")
            self.value = state["value"]

    key = os.urandom(32)
    model = NetA()
    src_provider = CustomStateProvider(value=123)

    with tempfile.NamedTemporaryFile(suffix=".shield", delete=False) as tmp:
        path = tmp.name

    try:
        save_training_checkpoint(
            output_path=path,
            model=model,
            key=key,
            state_providers=[src_provider],
        )

        # Case A: Restore fails, but rollback succeeds cleanly
        restore_calls[0] = 0
        dest_model = NetA()
        initial_weight = dest_model.fc1.weight.clone()
        dest_prov = CustomStateProvider(value=50, fail_on_restore=True, fail_on_rollback=False)

        with pytest.raises(TrainingStateError) as exc_info:
            load_training_checkpoint(
                input_path=path,
                model=dest_model,
                key=key,
                state_providers=[dest_prov],
            )
        msg = str(exc_info.value)
        assert "Restoration failed; rolled back all state mutations" in msg
        assert "Injected restore failure in provider!" in msg
        # Model rolled back cleanly
        assert torch.equal(dest_model.fc1.weight, initial_weight)
        # Provider rolled back cleanly to 50
        assert dest_prov.value == 50

        # Case B: Restore fails, AND rollback also fails
        restore_calls[0] = 0
        dest_model = NetA()
        dest_prov_b = CustomStateProvider(value=77, fail_on_restore=True, fail_on_rollback=True)

        with pytest.raises(TrainingStateError) as exc_info:
            load_training_checkpoint(
                input_path=path,
                model=dest_model,
                key=key,
                state_providers=[dest_prov_b],
            )
        msg_b = str(exc_info.value)
        # Message distinguishes restoration failure from rollback failure
        assert "Restoration failed (Failed to restore state provider 'custom_p': Injected restore failure in provider!)" in msg_b
        assert "rollback encountered errors: state_providers:" in msg_b
        assert "Injected secondary rollback failure in provider!" in msg_b

    finally:
        if os.path.exists(path):
            os.remove(path)


