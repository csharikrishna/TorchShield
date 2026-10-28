"""Unit tests for Native Model-Serving Integrations (vLLM, Triton, TensorRT-LLM)."""
import os
import secrets
import tempfile
import pytest
import torch
import torch.nn as nn

from torchshield.checkpoint import CheckpointError, SecureModelCheckpoint
from torchshield.integrations.serving import SecureInferenceLoader, ServingEngineType


class InferenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.norm = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))


def test_secure_inference_loader_streaming():
    key = secrets.token_bytes(32)
    model = InferenceNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "serving_model.seal")
        SecureModelCheckpoint.save_model(model, ckpt_path, key)

        # Record directory contents before loading
        pre_files = set(os.listdir(tmpdir))

        loader = SecureInferenceLoader(engine=ServingEngineType.TORCH)
        streamed_layers = dict(loader.stream_weights(ckpt_path, key_bytes=key))

        # Check all layers present
        for name, param in model.named_parameters():
            assert name in streamed_layers
            assert torch.allclose(param, streamed_layers[name])

        for name, buf in model.named_buffers():
            assert name in streamed_layers
            assert torch.allclose(buf, streamed_layers[name])

        # Assert zero plaintext disk materialization
        post_files = set(os.listdir(tmpdir))
        assert pre_files == post_files, "Plaintext files were materialized on disk during streaming!"


def test_secure_inference_loader_load_into_model():
    key = secrets.token_bytes(32)
    model = InferenceNet()
    x = torch.randn(4, 16)
    model.eval()
    expected_output = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "serving_model.seal")
        SecureModelCheckpoint.save_model(model, ckpt_path, key)

        target_model = InferenceNet()
        loader = SecureInferenceLoader(engine=ServingEngineType.TORCH)
        loader.load_into_model(ckpt_path, target_model, key_bytes=key)
        target_model.eval()

        actual_output = target_model(x)
        assert torch.allclose(expected_output, actual_output)


def test_secure_inference_loader_vllm_hook():
    key = secrets.token_bytes(32)
    model = InferenceNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "serving_model.seal")
        SecureModelCheckpoint.save_model(model, ckpt_path, key)

        target_model = InferenceNet()
        loader = SecureInferenceLoader(engine=ServingEngineType.VLLM)
        hook = loader.create_vllm_weight_loader(ckpt_path, key_bytes=key)
        hook(target_model)

        for name, param in model.named_parameters():
            assert torch.allclose(param, target_model.get_parameter(name))


def test_secure_inference_loader_triton_wrapper():
    key = secrets.token_bytes(32)
    model = InferenceNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "serving_model.seal")
        SecureModelCheckpoint.save_model(model, ckpt_path, key)

        loader = SecureInferenceLoader(engine=ServingEngineType.TRITON)
        wrapper = loader.create_triton_model_provider(ckpt_path, key_bytes=key)

        @wrapper
        def build_model():
            return InferenceNet()

        initialized_model = build_model
        assert isinstance(initialized_model, nn.Module)

        for name, param in model.named_parameters():
            assert torch.allclose(param, initialized_model.get_parameter(name))


def test_secure_inference_loader_tamper_fails_closed():
    key = secrets.token_bytes(32)
    model = InferenceNet()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "serving_model.seal")
        SecureModelCheckpoint.save_model(model, ckpt_path, key)

        with open(ckpt_path, "r+b") as f:
            data = bytearray(f.read())
            data[len(data) // 2] ^= 0xFF
            f.seek(0)
            f.write(data)

        loader = SecureInferenceLoader()
        target_model = InferenceNet()
        with pytest.raises(CheckpointError):
            loader.load_into_model(ckpt_path, target_model, key_bytes=key)
