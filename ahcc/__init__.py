"""Adaptive Hyperdimensional Chaotic Cryptography (AHCC) Secure Checkpoint & Training Platform."""
from .engine import AHCC_Engine, TensorCipherWrapper
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

__version__ = "0.7.0"
__all__ = [
    "AHCC_Engine",
    "TensorCipherWrapper",
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
    "__version__",
]
