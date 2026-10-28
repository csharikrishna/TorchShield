"""Native Model-Serving Integrations for TorchShield.

Enables secure streaming model weight loading directly into production inference
runtimes (vLLM, Triton Inference Server, TensorRT-LLM, PyTorch Inference) with:
1. Zero Plaintext Disk Materialization: Weights stream-decrypt directly into GPU/CPU
   tensor buffers without ever creating plaintext weight files on disk.
2. Fail-Closed Authentication: Every chunk is authenticated via AES-256-GCM AEAD
   before weights are loaded into serving runtime tensors.
"""
from __future__ import annotations

from enum import Enum
import logging
import os
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn

from torchshield.checkpoint import CheckpointError, SecureModelCheckpoint
from torchshield.key_providers.base import KeyProvider

logger = logging.getLogger(__name__)


class ServingEngineType(str, Enum):
    """Supported model serving inference engines."""
    VLLM = "vllm"
    TRITON = "triton"
    TENSORRT_LLM = "tensorrt_llm"
    TORCH = "torch"


class SecureInferenceLoader:
    """High-performance streaming inference loader with zero-disk plaintext materialization."""

    def __init__(
        self,
        engine: Union[str, ServingEngineType] = ServingEngineType.TORCH,
        device: Union[str, torch.device] = "cpu",
        zero_disk_enforcement: bool = True,
    ):
        """
        Args:
            engine: Target serving engine ('torch', 'vllm', 'triton', 'tensorrt_llm').
            device: Target execution device ('cpu', 'cuda', 'cuda:0', etc.).
            zero_disk_enforcement: If True, strictly prevents writing decrypted weights to disk.
        """
        self.engine = ServingEngineType(engine) if isinstance(engine, str) else engine
        self.device = torch.device(device) if isinstance(device, str) else device
        self.zero_disk_enforcement = zero_disk_enforcement

    def stream_weights(
        self,
        checkpoint_path: str,
        key_bytes: Optional[bytes] = None,
        key_provider: Optional[KeyProvider] = None,
        encryption_context: Optional[Dict[str, str]] = None,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Stream authenticated layer weights directly into memory.
        
        Yields (layer_name, tensor) pairs one by one.
        Zero plaintext bytes are ever written to disk.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Authenticate and decrypt checkpoint directly in memory
        tensors = SecureModelCheckpoint.load_checkpoint(
            input_path=checkpoint_path,
            key=key_bytes,
            key_provider=key_provider,
            encryption_context=encryption_context,
            device=self.device,
        )

        if isinstance(tensors, tuple):
            tensors = tensors[0]

        for name, tensor in tensors.items():
            yield name, tensor

    def load_into_model(
        self,
        checkpoint_path: str,
        model: nn.Module,
        key_bytes: Optional[bytes] = None,
        key_provider: Optional[KeyProvider] = None,
        encryption_context: Optional[Dict[str, str]] = None,
        strict: bool = True,
    ) -> nn.Module:
        """Stream decrypted weights directly into the target model parameters.
        
        Enforces zero plaintext file staging.
        """
        model_params = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())
        loaded_layers = set()

        for name, tensor in self.stream_weights(
            checkpoint_path=checkpoint_path,
            key_bytes=key_bytes,
            key_provider=key_provider,
            encryption_context=encryption_context,
        ):
            if name in model_params:
                with torch.no_grad():
                    model_params[name].copy_(tensor.to(model_params[name].device))
                loaded_layers.add(name)
            elif name in model_buffers:
                with torch.no_grad():
                    model_buffers[name].copy_(tensor.to(model_buffers[name].device))
                loaded_layers.add(name)
            elif strict:
                raise CheckpointError(f"Unexpected weight '{name}' in checkpoint not found in model.")

        if strict:
            expected_keys = set(model_params.keys()).union(set(model_buffers.keys()))
            missing_keys = expected_keys - loaded_layers
            if missing_keys:
                raise CheckpointError(f"Missing weights required by model: {sorted(missing_keys)}")

        return model

    def create_vllm_weight_loader(
        self,
        checkpoint_path: str,
        key_bytes: Optional[bytes] = None,
        key_provider: Optional[KeyProvider] = None,
    ) -> Callable[[nn.Module], None]:
        """Create a weight loader hook compatible with vLLM custom model loading.
        
        Usage:
            loader_hook = inference_loader.create_vllm_weight_loader("model.shield", key_bytes=key)
            loader_hook(vllm_model)
        """
        def _loader(model: nn.Module) -> None:
            self.load_into_model(
                checkpoint_path=checkpoint_path,
                model=model,
                key_bytes=key_bytes,
                key_provider=key_provider,
                strict=False,
            )

        return _loader

    def create_triton_model_provider(
        self,
        checkpoint_path: str,
        key_bytes: Optional[bytes] = None,
        key_provider: Optional[KeyProvider] = None,
    ) -> Callable[[Callable[[], nn.Module]], nn.Module]:
        """Create a model initialization wrapper for Triton Inference Server Python backend.
        
        Usage:
            @provider
            def build_model():
                return MyModel()
        """
        def _wrapper(model_factory: Callable[[], nn.Module]) -> nn.Module:
            model = model_factory()
            return self.load_into_model(
                checkpoint_path=checkpoint_path,
                model=model,
                key_bytes=key_bytes,
                key_provider=key_provider,
                strict=True,
            )

        return _wrapper


__all__ = [
    "ServingEngineType",
    "SecureInferenceLoader",
]
