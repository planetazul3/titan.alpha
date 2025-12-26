"""
Unit tests for checkpoint manifest module.

Tests cover:
- Manifest creation with git info
- Schema version validation
- Checkpoint save/load with manifest
- Compatibility checking
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from training.checkpoint_manifest import (
    CHECKPOINT_MANIFEST_VERSION,
    CheckpointManifest,
    create_manifest,
    get_git_info,
    get_model_version,
    load_checkpoint,
    save_checkpoint,
)


class SimpleModel(nn.Module):
    """Simple model for testing checkpoints."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()

    settings.trading = MagicMock()
    settings.trading.symbol = "R_100"
    settings.trading.stake_amount = 1.0

    settings.thresholds = MagicMock()
    settings.thresholds.confidence_threshold_high = 0.75
    settings.thresholds.learning_threshold_min = 0.40
    settings.thresholds.learning_threshold_max = 0.60

    settings.hyperparams = MagicMock()
    settings.hyperparams.lstm_hidden_size = 64
    settings.hyperparams.cnn_filters = 32
    settings.hyperparams.latent_dim = 16
    settings.hyperparams.dropout_rate = 0.1
    settings.hyperparams.learning_rate = 0.001

    settings.data_shapes = MagicMock()
    settings.data_shapes.sequence_length_ticks = 100
    settings.data_shapes.sequence_length_candles = 50

    return settings


@pytest.fixture
def model():
    """Create simple test model."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create optimizer for test model."""
    return torch.optim.Adam(model.parameters())


class TestCheckpointManifest:
    """Tests for CheckpointManifest dataclass."""

    def test_manifest_creation(self, mock_settings):
        """Should create manifest with all fields."""
        manifest = create_manifest(mock_settings)

        assert manifest.manifest_version == CHECKPOINT_MANIFEST_VERSION
        assert manifest.checkpoint_timestamp is not None
        assert manifest.feature_schema_version is not None
        assert manifest.pytorch_version == torch.__version__

    def test_manifest_to_dict_roundtrip(self, mock_settings):
        """Should serialize and deserialize correctly."""
        manifest = create_manifest(mock_settings)

        d = manifest.to_dict()
        restored = CheckpointManifest.from_dict(d)

        assert restored.manifest_version == manifest.manifest_version
        assert restored.git_sha == manifest.git_sha
        assert restored.feature_schema_version == manifest.feature_schema_version

    def test_settings_snapshot_captured(self, mock_settings):
        """Should capture key settings in snapshot."""
        manifest = create_manifest(mock_settings)

        snapshot = manifest.settings_snapshot
        assert "trading" in snapshot
        assert "hyperparams" in snapshot
        assert snapshot["trading"]["symbol"] == "R_100"
        assert snapshot["hyperparams"]["lstm_hidden_size"] == 64

    def test_schema_compatibility_same_major(self, mock_settings):
        """Same major version should be compatible."""
        manifest = create_manifest(mock_settings)
        manifest.feature_schema_version = "1.0"

        assert manifest.is_compatible_with("1.0") is True
        assert manifest.is_compatible_with("1.5") is True

    def test_schema_compatibility_different_major(self, mock_settings):
        """Different major version should be incompatible."""
        manifest = create_manifest(mock_settings)
        manifest.feature_schema_version = "1.0"

        assert manifest.is_compatible_with("2.0") is False


class TestGetGitInfo:
    """Tests for git info extraction."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with sha, branch, dirty keys."""
        info = get_git_info()

        assert "sha" in info
        assert "branch" in info
        assert "dirty" in info

    @patch("subprocess.run")
    def test_handles_git_failure(self, mock_run):
        """Should return unknown when git fails."""
        mock_run.side_effect = Exception("Git not found")

        info = get_git_info()

        assert info["sha"] == "unknown"
        assert info["branch"] == "unknown"
        assert info["dirty"] is True


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_saves_checkpoint_with_manifest(self, model, optimizer, mock_settings, tmp_path):
        """Should save checkpoint with manifest."""
        path = tmp_path / "checkpoint.pt"

        result = save_checkpoint(
            model=model, optimizer=optimizer, epoch=5, path=path, settings=mock_settings
        )

        assert path.exists()
        assert result == path

        # Verify checkpoint structure
        checkpoint = torch.load(path, weights_only=False)
        assert "manifest" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == 5

    def test_saves_optional_scheduler_scaler(self, model, optimizer, mock_settings, tmp_path):
        """Should save scheduler and scaler when provided."""
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler(device="cpu", enabled=False)

        path = tmp_path / "full_checkpoint.pt"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            path=path,
            settings=mock_settings,
            scheduler=scheduler,
            scaler=scaler,
        )

        checkpoint = torch.load(path, weights_only=False)
        assert "scheduler_state_dict" in checkpoint
        assert "scaler_state_dict" in checkpoint

    def test_saves_extra_data(self, model, optimizer, mock_settings, tmp_path):
        """Should save extra data when provided."""
        path = tmp_path / "extra_checkpoint.pt"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            path=path,
            settings=mock_settings,
            extra_data={"custom_metric": 0.95},
        )

        checkpoint = torch.load(path, weights_only=False)
        assert "extra_data" in checkpoint
        assert checkpoint["extra_data"]["custom_metric"] == 0.95


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_loads_checkpoint_with_manifest(self, model, optimizer, mock_settings, tmp_path):
        """Should load checkpoint and restore model state."""
        path = tmp_path / "checkpoint.pt"

        # Modify model weights
        model.linear.weight.data.fill_(1.0)

        save_checkpoint(
            model=model, optimizer=optimizer, epoch=5, path=path, settings=mock_settings
        )

        # Create new model
        new_model = SimpleModel()
        new_model.linear.weight.data.clone()

        # Load checkpoint
        result = load_checkpoint(path, new_model)

        # Weights should be restored
        assert torch.allclose(new_model.linear.weight, model.linear.weight)
        assert result["epoch"] == 5
        assert result["manifest"] is not None

    def test_validates_schema_compatibility(self, model, optimizer, mock_settings, tmp_path):
        """Should reject incompatible schema versions."""
        path = tmp_path / "old_checkpoint.pt"

        # Create checkpoint with different schema version
        manifest = create_manifest(mock_settings)
        manifest.feature_schema_version = "99.0"  # Future version

        checkpoint = {
            "manifest": manifest.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 1,
        }
        torch.save(checkpoint, path)

        # Should raise on incompatible schema
        new_model = SimpleModel()
        with pytest.raises(ValueError, match="incompatible"):
            load_checkpoint(path, new_model, validate_schema=True)

    def test_skip_validation_when_disabled(self, model, optimizer, mock_settings, tmp_path):
        """Should skip validation when validate_schema=False."""
        path = tmp_path / "any_checkpoint.pt"

        manifest = create_manifest(mock_settings)
        manifest.feature_schema_version = "99.0"

        checkpoint = {
            "manifest": manifest.to_dict(),
            "model_state_dict": model.state_dict(),
            "epoch": 1,
        }
        torch.save(checkpoint, path)

        new_model = SimpleModel()
        result = load_checkpoint(path, new_model, validate_schema=False)

        # Should load successfully
        assert result["epoch"] == 1

    def test_handles_legacy_checkpoint_without_manifest(self, model, tmp_path):
        """Should handle old checkpoints without manifest."""
        path = tmp_path / "legacy.pt"

        # Old-style checkpoint without manifest
        torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, path)

        new_model = SimpleModel()
        result = load_checkpoint(path, new_model)

        assert result["epoch"] == 3
        assert result["manifest"] is None


class TestGetModelVersion:
    """Tests for get_model_version function."""

    def test_returns_version_string(self, model, optimizer, mock_settings, tmp_path):
        """Should return git_sha_schema version string."""
        path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            model=model, optimizer=optimizer, epoch=1, path=path, settings=mock_settings
        )

        version = get_model_version(path)

        # Should have format: 1.0.0
        assert "." in version
        parts = version.split(".")
        assert len(parts) >= 3

    def test_returns_unknown_for_missing_manifest(self, model, tmp_path):
        """Should return 'unknown' for legacy checkpoints."""
        path = tmp_path / "legacy.pt"

        torch.save({"model_state_dict": model.state_dict()}, path)

        version = get_model_version(path)
        assert version == "unknown"

    def test_returns_unknown_for_missing_file(self, tmp_path):
        """Should return 'unknown' for non-existent file."""
        version = get_model_version(tmp_path / "nonexistent.pt")
        assert version == "unknown"
