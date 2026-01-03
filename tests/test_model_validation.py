
import unittest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import torch
from utils.bootstrap import create_trading_stack
from config.constants import FEATURE_SCHEMA_VERSION

class TestModelValidation(unittest.TestCase):
    def setUp(self):
        self.mock_settings = MagicMock()
        self.mock_settings.get_device.return_value = "cpu"
        self.mock_settings.feature_schema_version = FEATURE_SCHEMA_VERSION
        self.mock_settings.hyperparams.use_tft = False
        
        # Mock data shapes
        self.mock_settings.data_shapes.sequence_length_ticks = 60
        self.mock_settings.data_shapes.sequence_length_candles = 60
        self.mock_settings.system.system_db_path = ":memory:"
        
        # Mock dependencies
        self.mock_model = MagicMock()
        self.mock_model.to.return_value = self.mock_model

    @patch("utils.bootstrap.DerivOmniModel")
    @patch("utils.bootstrap.torch.load")
    @patch("utils.bootstrap.verify_checkpoint")
    @patch("utils.bootstrap.get_feature_builder")
    @patch("utils.bootstrap.DerivClient")
    @patch("utils.bootstrap.SQLiteShadowStore")
    @patch("utils.bootstrap.RegimeVeto")
    @patch("utils.bootstrap.DecisionEngine")
    def test_valid_manifest(
        self, mock_engine, mock_veto, mock_shadow, mock_client, mock_fb, 
        mock_verify, mock_load, mock_model_cls
    ):
        """Test loading a checkpoint with a valid manifest."""
        mock_verify.return_value = True
        mock_model_cls.return_value = self.mock_model
        
        # Mock Path.exists
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True
        fake_path.__str__.return_value = "fake.pt"

        # Valid checkpoint
        checkpoint = {
            "model_state_dict": {},
            "manifest": {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "data_shapes": {
                    "sequence_length_ticks": 60,
                    "sequence_length_candles": 60
                }
            }
        }
        mock_load.return_value = checkpoint
        
        # Should succeed
        stack = create_trading_stack(self.mock_settings, checkpoint_path=fake_path)
        self.assertIsNotNone(stack)

    @patch("utils.bootstrap.DerivOmniModel")
    @patch("utils.bootstrap.torch.load")
    @patch("utils.bootstrap.verify_checkpoint")
    @patch("utils.bootstrap.get_feature_builder")
    @patch("utils.bootstrap.DerivClient")
    @patch("utils.bootstrap.SQLiteShadowStore")
    @patch("utils.bootstrap.RegimeVeto")
    @patch("utils.bootstrap.DecisionEngine")
    def test_invalid_schema_version(
        self, mock_engine, mock_veto, mock_shadow, mock_client, mock_fb, 
        mock_verify, mock_load, mock_model_cls
    ):
        """Test mismatching feature schema version."""
        mock_verify.return_value = True
        mock_model_cls.return_value = self.mock_model
        
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True

        checkpoint = {
            "model_state_dict": {},
            "manifest": {
                "feature_schema_version": "9.9.9",
                "data_shapes": {
                    "sequence_length_ticks": 60,
                    "sequence_length_candles": 60
                }
            }
        }
        mock_load.return_value = checkpoint
        
        with self.assertRaises(RuntimeError) as cm:
            create_trading_stack(self.mock_settings, checkpoint_path=fake_path)
        
        self.assertIn("Incompatible feature schema", str(cm.exception))

    @patch("utils.bootstrap.DerivOmniModel")
    @patch("utils.bootstrap.torch.load")
    @patch("utils.bootstrap.verify_checkpoint")
    @patch("utils.bootstrap.get_feature_builder")
    @patch("utils.bootstrap.DerivClient")
    @patch("utils.bootstrap.SQLiteShadowStore")
    @patch("utils.bootstrap.RegimeVeto")
    @patch("utils.bootstrap.DecisionEngine")
    def test_invalid_shapes(
        self, mock_engine, mock_veto, mock_shadow, mock_client, mock_fb, 
        mock_verify, mock_load, mock_model_cls
    ):
        """Test mismatching data shapes."""
        mock_verify.return_value = True
        mock_model_cls.return_value = self.mock_model
        
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True

        checkpoint = {
            "model_state_dict": {},
            "manifest": {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "data_shapes": {
                    "sequence_length_ticks": 120, # Mismatch (system is 60)
                    "sequence_length_candles": 60
                }
            }
        }
        mock_load.return_value = checkpoint
        
        with self.assertRaises(RuntimeError) as cm:
            create_trading_stack(self.mock_settings, checkpoint_path=fake_path)
            
        self.assertIn("Tick sequence length mismatch", str(cm.exception))

    @patch("utils.bootstrap.DerivOmniModel")
    @patch("utils.bootstrap.torch.load")
    @patch("utils.bootstrap.verify_checkpoint")
    @patch("utils.bootstrap.get_feature_builder")
    @patch("utils.bootstrap.DerivClient")
    @patch("utils.bootstrap.SQLiteShadowStore")
    @patch("utils.bootstrap.RegimeVeto")
    @patch("utils.bootstrap.DecisionEngine")
    def test_legacy_checkpoint(
        self, mock_engine, mock_veto, mock_shadow, mock_client, mock_fb, 
        mock_verify, mock_load, mock_model_cls
    ):
        """Test legacy checkpoint (no manifest) loads with warning."""
        mock_verify.return_value = True
        mock_model_cls.return_value = self.mock_model
        
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True

        # Legacy checkpoint
        checkpoint = {"model_state_dict": {}}
        mock_load.return_value = checkpoint
        
        # Should NOT raise, just log warning (validated via success)
        stack = create_trading_stack(self.mock_settings, checkpoint_path=fake_path)
        self.assertIsNotNone(stack)

if __name__ == '__main__':
    unittest.main()
