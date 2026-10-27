"""AHCC Secure AI Checkpoint & Training Platform."""
from .secure import AESGCMEnvelope, AuthenticationError, TensorAESGCMEnvelope
from .checkpoint import SecureModelCheckpoint, CheckpointError, ResourceLimits
from .training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    inspect_training_checkpoint,
    TrainingProgress,
    TrainingStateError,
    serialize_all_rng,
    restore_all_rng,
)
from .integrations.huggingface import (
    TorchVaultTrainerCallback,
    AHCCTrainerCallback,
    save_hf_model,
    load_hf_model,
)

__version__ = "0.7.0"
__all__ = [
    "AESGCMEnvelope",
    "AuthenticationError",
    "TensorAESGCMEnvelope",
    "SecureModelCheckpoint",
    "CheckpointError",
    "ResourceLimits",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "inspect_training_checkpoint",
    "TrainingProgress",
    "TrainingStateError",
    "serialize_all_rng",
    "restore_all_rng",
    "TorchVaultTrainerCallback",
    "AHCCTrainerCallback",
    "save_hf_model",
    "load_hf_model",
    "__version__",
]
