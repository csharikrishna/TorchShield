"""Third-party ecosystem integrations for TorchVault / AHCC."""
from .huggingface import (
    TorchVaultTrainerCallback,
    AHCCTrainerCallback,
    save_hf_model,
    load_hf_model,
)

try:
    from .huggingface import TorchVaultTrainer, AHCCTrainer
    __all__ = [
        "TorchVaultTrainerCallback",
        "AHCCTrainerCallback",
        "TorchVaultTrainer",
        "AHCCTrainer",
        "save_hf_model",
        "load_hf_model",
    ]
except ImportError:
    __all__ = [
        "TorchVaultTrainerCallback",
        "AHCCTrainerCallback",
        "save_hf_model",
        "load_hf_model",
    ]
