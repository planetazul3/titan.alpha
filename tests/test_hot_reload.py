
import unittest
from unittest.mock import MagicMock, patch
from utils.bootstrap import validate_model_compatibility
from config.settings import Settings
from config.constants import FEATURE_SCHEMA_VERSION

class TestHotReload(unittest.TestCase):
    def setUp(self):
        # Don't use spec=Settings to avoid auto-mocking complexities with Pydantic
        self.settings = MagicMock()
        # Setup settings with data_shapes
        self.settings.data_shapes.sequence_length_ticks = 60
        self.settings.data_shapes.sequence_length_candles = 60
        
    def test_validate_valid_manifest(self):
        """Test validation passes with correct manifest."""
        checkpoint = {
            "manifest": {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "data_shapes": {
                    "sequence_length_ticks": 60,
                    "sequence_length_candles": 60
                }
            }
        }
        # Should not raise
        validate_model_compatibility(checkpoint, self.settings)
        
    def test_validate_invalid_schema(self):
        """Test validation raises on mismatched schema version."""
        checkpoint = {
            "manifest": {
                "feature_schema_version": "0.0.0", # Mismatch
                "data_shapes": {
                    "sequence_length_ticks": 60,
                    "sequence_length_candles": 60
                }
            }
        }
        with self.assertRaises(RuntimeError) as cm:
            validate_model_compatibility(checkpoint, self.settings)
        self.assertIn("Incompatible feature schema", str(cm.exception))
        
    def test_validate_invalid_shapes(self):
        """Test validation raises on mismatched shapes."""
        checkpoint = {
            "manifest": {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "data_shapes": {
                    "sequence_length_ticks": 30, # Mismatch (system expects 60)
                    "sequence_length_candles": 60
                }
            }
        }
        with self.assertRaises(RuntimeError) as cm:
            validate_model_compatibility(checkpoint, self.settings)
        self.assertIn("Tick sequence length mismatch", str(cm.exception))
        
    def test_legacy_checkpoint(self):
        """Test legacy checkpoint (no manifest) passes without error."""
        checkpoint = {}
        # Should not raise (just returns)
        validate_model_compatibility(checkpoint, self.settings)

if __name__ == '__main__':
    unittest.main()
