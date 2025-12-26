"""
Unit tests for the Trainer class.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.trainer import Trainer, TrainerConfig

class MockDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            "ticks": torch.randn(100),
            "candles": torch.randn(50, 10),
            "vol_metrics": torch.randn(4),
            "targets": {
                "rise_fall": torch.tensor(1.0),
                "touch": torch.tensor(0.0),
                "range": torch.tensor(1.0)
            }
        }

@pytest.fixture
def mock_model_obj():
    model = MagicMock(spec=nn.Module)
    model.parameters.return_value = [nn.Parameter(torch.randn(1, 1))]
    model.state_dict.return_value = {}
    
    # Mock forward pass to return expected dict of logits
    def side_effect(*args, **kwargs):
        # Match shape (batch, 1) expected by BCEWithLogitsLoss
        return {
            "rise_fall_logit": torch.tensor([[0.5], [0.5]], requires_grad=True),
            "touch_logit": torch.tensor([[0.5], [0.5]], requires_grad=True),
            "range_logit": torch.tensor([[0.5], [0.5]], requires_grad=True)
        }
    mock_model_obj = model
    mock_model_obj.side_effect = side_effect
    # PyTorch modules are callable
    mock_model_obj.return_value = side_effect()
    # Mock 'to' method to return itself (PyTorch pattern)
    mock_model_obj.to.return_value = mock_model_obj
    return mock_model_obj

@pytest.fixture
def trainer_config(tmp_path):
    return TrainerConfig(
        epochs=1,
        warmup_epochs=0,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        use_amp=False
    )

class TestTrainer:
    def test_trainer_initialization(self, mock_model_obj, trainer_config):
        train_loader = DataLoader(MockDataset(), batch_size=2)
        val_loader = DataLoader(MockDataset(), batch_size=2)
        
        trainer = Trainer(mock_model_obj, train_loader, val_loader, trainer_config, device=torch.device("cpu"))
        
        assert trainer.model == mock_model_obj
        assert trainer.config == trainer_config
        assert trainer.device.type == "cpu"
        assert trainer.current_epoch == 0

    @patch("training.trainer.tqdm")
    def test_train_epoch(self, mock_tqdm, mock_model_obj, trainer_config):
        train_loader = DataLoader(MockDataset(size=4), batch_size=2)
        val_loader = DataLoader(MockDataset(size=2), batch_size=2)
        
        # Mock tqdm to return a MagicMock that acts as an iterator
        mock_pb = MagicMock()
        mock_pb.__iter__.return_value = train_loader
        mock_tqdm.return_value = mock_pb
        
        trainer = Trainer(mock_model_obj, train_loader, val_loader, trainer_config, device=torch.device("cpu"))
        
        loss = trainer._train_epoch()
        
        assert isinstance(loss, float)
        assert mock_model_obj.train.called
        assert mock_pb.set_postfix.called

    @patch("training.trainer.tqdm")
    def test_validate_epoch(self, mock_tqdm, mock_model_obj, trainer_config):
        train_loader = DataLoader(MockDataset(size=4), batch_size=2)
        val_loader = DataLoader(MockDataset(size=2), batch_size=2)
        
        mock_pb = MagicMock()
        mock_pb.__iter__.return_value = val_loader
        mock_tqdm.return_value = mock_pb
        
        trainer = Trainer(mock_model_obj, train_loader, val_loader, trainer_config, device=torch.device("cpu"))
        
        val_loss, metrics = trainer._validate_epoch()
        
        assert isinstance(val_loss, float)
        assert "accuracy" in metrics
        assert mock_model_obj.eval.called

    def test_save_checkpoint(self, mock_model_obj, trainer_config):
        train_loader = DataLoader(MockDataset(), batch_size=2)
        val_loader = DataLoader(MockDataset(), batch_size=2)
        
        trainer = Trainer(mock_model_obj, train_loader, val_loader, trainer_config, device=torch.device("cpu"))
        
        trainer._save_checkpoint("test_ckpt.pt")
        
        ckpt_path = trainer_config.checkpoint_dir / "test_ckpt.pt"
        assert ckpt_path.exists()
        
        # Verify content
        # ID-001 Fix: Use weights_only=False because TrainerConfig is a custom class
        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
