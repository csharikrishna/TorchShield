"""TorchShield: Production-grade authenticated, zero-pickle checkpoint platform for PyTorch and Hugging Face."""
from .secure import AESGCMEnvelope, AuthenticationError, TensorAESGCMEnvelope
from .checkpoint import SecureModelCheckpoint, CheckpointError, ResourceLimits
from .training_state import (
    save_training_checkpoint,
    load_training_checkpoint,
    inspect_training_checkpoint,
    TrainingProgress,
    TrainingStateError,
    SafeStateProvider,
    validate_model_checkpoint,
    validate_optimizer_checkpoint,
    validate_safe_metadata,
    serialize_all_rng,
    restore_all_rng,
)
from .integrations.huggingface import (
    TorchShieldTrainerCallback,
    TorchShieldTrainer,
    save_hf_model,
    load_hf_model,
)
from .integrations.serving import (
    SecureInferenceLoader,
    ServingEngineType,
)
from .attestation import (
    AttestationEvidence,
    AttestationPolicy,
    HardwareAttestationVerifier,
    AttestationGatedKeyProvider,
    AttestationError,
    PlatformType,
)
from .gds import (
    GDSLoader,
    GDSStatus,
    probe_gds_capability,
)
from .pqc import (
    HybridKEMSession,
    HybridSignatureEngine,
    PQCError,
)
from .provenance import (
    EtcdFreshnessBackend,
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
    "SafeStateProvider",
    "validate_model_checkpoint",
    "validate_optimizer_checkpoint",
    "validate_safe_metadata",
    "serialize_all_rng",
    "restore_all_rng",
    "TorchShieldTrainerCallback",
    "TorchShieldTrainer",
    "save_hf_model",
    "load_hf_model",
    "SecureInferenceLoader",
    "ServingEngineType",
    "AttestationEvidence",
    "AttestationPolicy",
    "HardwareAttestationVerifier",
    "AttestationGatedKeyProvider",
    "AttestationError",
    "PlatformType",
    "GDSLoader",
    "GDSStatus",
    "probe_gds_capability",
    "HybridKEMSession",
    "HybridSignatureEngine",
    "PQCError",
    "EtcdFreshnessBackend",
    "__version__",
]
