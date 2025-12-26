"""
Checkpoint Manifest - Enhanced model checkpointing with metadata.

Provides comprehensive metadata for model checkpoints to ensure:
- Reproducibility: Captures git SHA, settings, and environment
- Compatibility: Validates schema versions before loading
- Traceability: Links model versions to shadow trade data

Usage:
    # Saving with manifest
    >>> from training.checkpoint_manifest import save_checkpoint, load_checkpoint
    >>> save_checkpoint(model, optimizer, epoch, path, settings)

    # Loading with validation
    >>> checkpoint = load_checkpoint(path, model, settings)
"""

import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from config.settings import Settings
from data.features import FEATURE_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Checkpoint manifest schema version
CHECKPOINT_MANIFEST_VERSION = "1.0"

# Current model version - increment when making changes:
# - Major: Breaking changes to model architecture or inputs
# - Minor: New features, backward compatible  
# - Patch: Bug fixes, no model changes
CURRENT_MODEL_VERSION = "1.0.0"


@dataclass
class ModelVersion:
    """
    Semantic versioning for model checkpoints.
    
    Enables safe migrations between model versions:
    - Major version changes = breaking, requires retraining
    - Minor version changes = backward compatible
    - Patch version changes = safe to load
    
    Example:
        >>> v1 = ModelVersion.from_string("1.2.3")
        >>> v2 = ModelVersion.from_string("1.3.0")
        >>> v1.is_compatible_with(v2)  # True (same major)
    """
    
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"
    
    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """Parse version string like 'v1.2.3' or '1.2.3'."""
        parts = version_str.lstrip('v').split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}. Expected 'X.Y.Z'")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))
    
    @classmethod
    def current(cls) -> "ModelVersion":
        """Get current model version."""
        return cls.from_string(CURRENT_MODEL_VERSION)
    
    def is_compatible_with(self, other: "ModelVersion") -> bool:
        """Check if versions are compatible (same major version)."""
        return self.major == other.major
    
    def is_newer_than(self, other: "ModelVersion") -> bool:
        """Check if this version is newer than other."""
        if self.major != other.major:
            return self.major > other.major
        if self.minor != other.minor:
            return self.minor > other.minor
        return self.patch > other.patch


@dataclass
class CheckpointManifest:
    """
    Metadata manifest for model checkpoints.

    Stores all information needed to:
    1. Verify checkpoint compatibility before loading
    2. Trace model predictions back to training conditions
    3. Reproduce training if needed
    """

    # Version info
    manifest_version: str
    checkpoint_timestamp: str

    # Git info
    git_sha: str
    git_branch: str
    git_dirty: bool

    # Model/Data schema
    feature_schema_version: str
    model_architecture: str

    # Settings snapshot (key params only)
    settings_snapshot: dict[str, Any]

    # Training context
    training_host: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: str | None

    model_version: str = CURRENT_MODEL_VERSION  # Semantic version (e.g., "1.0.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointManifest":
        """Create from dictionary."""
        return cls(**d)

    def is_compatible_with(self, current_feature_version: str) -> bool:
        """
        Check if checkpoint is compatible with current feature schema.

        Args:
            current_feature_version: Current feature schema version

        Returns:
            True if compatible, False otherwise
        """
        # Split version strings (e.g., "1.0" -> [1, 0])
        checkpoint_parts = [int(x) for x in self.feature_schema_version.split(".")]
        current_parts = [int(x) for x in current_feature_version.split(".")]

        # Major version must match (breaking changes)
        return checkpoint_parts[0] == current_parts[0]


def get_git_info() -> dict[str, Any]:
    """
    Get current git repository state.

    Returns dict with:
    - sha: Current commit SHA (or "unknown")
    - branch: Current branch name
    - dirty: True if working directory has uncommitted changes
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        dirty = len(status) > 0

        return {"sha": sha or "unknown", "branch": branch or "unknown", "dirty": dirty}
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
        return {"sha": "unknown", "branch": "unknown", "dirty": True}


def get_settings_snapshot(settings: Settings) -> dict[str, Any]:
    """
    Extract key settings for checkpoint manifest.

    Captures the settings that affect model behavior:
    - Hyperparameters
    - Data shapes
    - Thresholds
    """
    return {
        "trading": {
            "symbol": settings.trading.symbol,
            "stake_amount": settings.trading.stake_amount,
        },
        "thresholds": {
            "confidence_threshold_high": settings.thresholds.confidence_threshold_high,
            "learning_threshold_min": settings.thresholds.learning_threshold_min,
            "learning_threshold_max": settings.thresholds.learning_threshold_max,
        },
        "hyperparams": {
            "lstm_hidden_size": settings.hyperparams.lstm_hidden_size,
            "cnn_filters": settings.hyperparams.cnn_filters,
            "latent_dim": settings.hyperparams.latent_dim,
            "dropout_rate": settings.hyperparams.dropout_rate,
            "learning_rate": settings.hyperparams.learning_rate,
        },
        "data_shapes": {
            "sequence_length_ticks": settings.data_shapes.sequence_length_ticks,
            "sequence_length_candles": settings.data_shapes.sequence_length_candles,
        },
    }


def create_manifest(
    settings: Settings, model_architecture: str = "DerivOmniModel"
) -> CheckpointManifest:
    """
    Create a new checkpoint manifest with current environment info.

    Args:
        settings: Current application settings
        model_architecture: Model architecture identifier

    Returns:
        CheckpointManifest with all metadata
    """
    import socket

    git_info = get_git_info()

    cuda_version = None
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda

    return CheckpointManifest(
        manifest_version=CHECKPOINT_MANIFEST_VERSION,
        checkpoint_timestamp=datetime.now(timezone.utc).isoformat(),
        git_sha=git_info["sha"],
        git_branch=git_info["branch"],
        git_dirty=git_info["dirty"],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_architecture=model_architecture,
        model_version=CURRENT_MODEL_VERSION,
        settings_snapshot=get_settings_snapshot(settings),
        training_host=socket.gethostname(),
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=cuda_version,
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    settings: Settings,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    best_val_loss: float = float("inf"),
    global_step: int = 0,
    extra_data: dict[str, Any] | None = None,
) -> Path:
    """
    Save model checkpoint with manifest metadata.

    This is the recommended way to save checkpoints as it includes
    all metadata needed for reproducibility and compatibility checks.

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current training epoch
        path: Checkpoint file path
        settings: Application settings
        scheduler: Optional LR scheduler
        scaler: Optional AMP GradScaler
        best_val_loss: Best validation loss so far
        global_step: Global training step counter
        extra_data: Optional additional data to include

    Returns:
        Path to saved checkpoint
    """
    # Create manifest
    manifest = create_manifest(settings)

    checkpoint = {
        # Manifest (MUST be first for quick inspection)
        "manifest": manifest.to_dict(),
        # Model state
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }

    # Optional components
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if extra_data:
        checkpoint["extra_data"] = extra_data

    # Save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

    logger.info(
        f"Saved checkpoint to {path} "
        f"(epoch={epoch}, git={manifest.git_sha[:8]}, feature_schema={manifest.feature_schema_version})"
    )

    return path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: torch.device | None = None,
    strict: bool = True,
    validate_schema: bool = True,
) -> dict[str, Any]:
    """
    Load model checkpoint with manifest validation.

    Validates feature schema compatibility to prevent loading
    checkpoints that are incompatible with current data pipeline.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        scaler: Optional AMP scaler to load state into
        device: Device to load tensors onto
        strict: If True, require exact state dict match
        validate_schema: If True, validate feature schema compatibility

    Returns:
        Dict with checkpoint data and manifest

    Raises:
        ValueError: If checkpoint is incompatible with current schema
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint (weights_only=False needed for full training state)
    checkpoint = torch.load(path, map_location=device, weights_only=False)  # nosec B614

    # Parse manifest
    manifest = None
    if "manifest" in checkpoint:
        manifest = CheckpointManifest.from_dict(checkpoint["manifest"])

        # Log manifest info
        logger.info(
            f"Loading checkpoint: {path}\n"
            f"  - Created: {manifest.checkpoint_timestamp}\n"
            f"  - Git: {manifest.git_sha[:8]} ({manifest.git_branch})\n"
            f"  - Feature Schema: {manifest.feature_schema_version}\n"
            f"  - PyTorch: {manifest.pytorch_version}"
        )

        # Validate schema compatibility
        if validate_schema:
            if not manifest.is_compatible_with(FEATURE_SCHEMA_VERSION):
                raise ValueError(
                    f"Checkpoint incompatible with current feature schema. "
                    f"Checkpoint: {manifest.feature_schema_version}, "
                    f"Current: {FEATURE_SCHEMA_VERSION}. "
                    f"Set validate_schema=False to force load."
                )

        # Warn about dirty git state
        if manifest.git_dirty:
            logger.warning(
                "Checkpoint was created from dirty git state. Reproducibility may be compromised."
            )
    else:
        logger.warning(
            f"Checkpoint has no manifest (created before v{CHECKPOINT_MANIFEST_VERSION}). "
            f"Compatibility checks skipped."
        )

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load scaler state if provided
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "manifest": manifest,
        "extra_data": checkpoint.get("extra_data", {}),
    }


def get_model_version(checkpoint_path: Path) -> str:
    """
    Get model version string from checkpoint manifest.

    Useful for linking shadow trades to model versions.
    Returns semantic version (e.g., "1.0.0") if available.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Model version string (e.g., "1.0.0" or "git_sha[:8]_schema" for old checkpoints)
    """
    try:
        # Quick load just manifest
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec B614

        if "manifest" in checkpoint:
            manifest = CheckpointManifest.from_dict(checkpoint["manifest"])
            # Prefer semantic version if available, fallback to git-based
            if hasattr(manifest, 'model_version') and manifest.model_version:
                return manifest.model_version
            return f"{manifest.git_sha[:8]}_{manifest.feature_schema_version}"
        else:
            return "unknown"
    except Exception as e:
        logger.warning(f"Could not read checkpoint manifest: {e}")
        return "unknown"


def check_model_compatibility(checkpoint_path: Path, current_version: str = CURRENT_MODEL_VERSION) -> bool:
    """
    Check if a checkpoint is compatible with the current model version.
    
    Args:
        checkpoint_path: Path to checkpoint file
        current_version: Current model version string
    
    Returns:
        True if compatible (same major version), False otherwise
    """
    try:
        checkpoint_version = get_model_version(checkpoint_path)
        if checkpoint_version == "unknown":
            logger.warning("Cannot verify compatibility - checkpoint has no version info")
            return True  # Allow loading legacy checkpoints
        
        current = ModelVersion.from_string(current_version)
        checkpoint = ModelVersion.from_string(checkpoint_version)
        
        return current.is_compatible_with(checkpoint)
    except ValueError as e:
        logger.warning(f"Version parsing failed: {e}")
        return True  # Allow loading if version format is different
