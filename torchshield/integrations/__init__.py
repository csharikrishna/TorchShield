"""Third-party ecosystem integrations for TorchShield."""
from .huggingface import (
    TorchShieldTrainerCallback,
    save_hf_model,
    load_hf_model,
)

from .serving import SecureInferenceLoader, ServingEngineType

try:
    from .huggingface import TorchShieldTrainer
    __all__ = [
        "TorchShieldTrainerCallback",
        "TorchShieldTrainer",
        "save_hf_model",
        "load_hf_model",
        "SecureInferenceLoader",
        "ServingEngineType",
    ]
except ImportError:
    __all__ = [
        "TorchShieldTrainerCallback",
        "save_hf_model",
        "load_hf_model",
        "SecureInferenceLoader",
        "ServingEngineType",
    ]
