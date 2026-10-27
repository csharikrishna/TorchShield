"""Tests for safe, pickle-free optimizer state extraction and restoration."""
import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD

from torchshield.training_state import (
    extract_optimizer_state,
    restore_optimizer_state,
    TrainingStateError,
)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.bn1(self.fc1(x))))


@pytest.mark.parametrize("opt_cls", [Adam, AdamW, SGD])
def test_optimizer_state_extraction_and_restoration(opt_cls):
    torch.manual_seed(42)
    model = SimpleMLP()
    opt_kwargs = {"lr": 1e-3}
    if opt_cls is SGD:
        opt_kwargs["momentum"] = 0.9
    optimizer = opt_cls(model.parameters(), **opt_kwargs)

    # Perform a few optimization steps to populate state (moments, momentum buffers)
    for _ in range(5):
        optimizer.zero_grad()
        x = torch.randn(4, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    # Extract state
    opt_tensors, opt_meta = extract_optimizer_state(optimizer, model)
    assert len(opt_tensors) > 0
    assert "param_groups" in opt_meta

    # Instantiate fresh model and fresh optimizer
    fresh_model = SimpleMLP()
    fresh_optimizer = opt_cls(fresh_model.parameters(), **opt_kwargs)

    # Restore optimizer state
    restore_optimizer_state(
        optimizer=fresh_optimizer,
        model=fresh_model,
        optimizer_tensors=opt_tensors,
        optimizer_meta=opt_meta,
    )

    # Assert parameter groups equality
    for g_orig, g_fresh in zip(optimizer.param_groups, fresh_optimizer.param_groups):
        assert g_orig["lr"] == g_fresh["lr"]
        if "weight_decay" in g_orig:
            assert g_orig["weight_decay"] == g_fresh["weight_decay"]

    # Assert tensor state equality (exp_avg, exp_avg_sq, or momentum_buffer)
    orig_params = list(model.parameters())
    fresh_params = list(fresh_model.parameters())

    for p_orig, p_fresh in zip(orig_params, fresh_params):
        st_orig = optimizer.state.get(p_orig, {})
        st_fresh = fresh_optimizer.state.get(p_fresh, {})

        assert set(st_orig.keys()) == set(st_fresh.keys())
        for k, v in st_orig.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, st_fresh[k])
            elif isinstance(v, int):
                assert v == st_fresh[k]


def test_restore_optimizer_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    torch.manual_seed(42)
    model = SimpleMLP().cuda()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    for _ in range(3):
        optimizer.zero_grad()
        x = torch.randn(4, 8, device="cuda")
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    opt_tensors, opt_meta = extract_optimizer_state(optimizer, model)

    # Restore on a new CUDA model
    fresh_model = SimpleMLP().cuda()
    fresh_optimizer = AdamW(fresh_model.parameters(), lr=1e-3)
    restore_optimizer_state(
        optimizer=fresh_optimizer,
        model=fresh_model,
        optimizer_tensors=opt_tensors,
        optimizer_meta=opt_meta,
        device="cuda",
    )

    # Confirm restored state tensors are on CUDA
    for p, st in fresh_optimizer.state.items():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                assert v.is_cuda
