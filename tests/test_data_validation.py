
import unittest
import shutil
import tempfile
import pandas as pd
from pathlib import Path
from utils.data_validation import validate_split_consistency

class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.train_dir = self.root / "train"
        self.val_dir = self.root / "val"
        self.train_dir.mkdir()
        self.val_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_parquet(self, path: Path, start_epoch: int, end_epoch: int, filename="data.parquet"):
        """Create a dummy parquet file with an 'epoch' column."""
        df = pd.DataFrame({
            "epoch": range(start_epoch, end_epoch + 1),
            "close": [1.0] * (end_epoch - start_epoch + 1)
        })
        df.to_parquet(path / filename)

    def test_valid_forward_split(self):
        """Train [0..100], Val [101..200] -> Should Pass"""
        self._create_parquet(self.train_dir, 0, 100)
        self._create_parquet(self.val_dir, 101, 200)
        
        # strict=True
        self.assertTrue(validate_split_consistency(self.train_dir, self.val_dir, strict_forward=True))

    def test_overlap_fail(self):
        """Train [0..100], Val [90..200] -> Should Fail"""
        self._create_parquet(self.train_dir, 0, 100)
        self._create_parquet(self.val_dir, 90, 200)
        
        with self.assertRaises(ValueError) as cm:
            validate_split_consistency(self.train_dir, self.val_dir, strict_forward=True)
        self.assertIn("Strict Forward Validation Failed", str(cm.exception))

    def test_backward_split_fail_strict(self):
        """Train [100..200], Val [0..99] -> Should Fail Strict"""
        self._create_parquet(self.train_dir, 100, 200)
        self._create_parquet(self.val_dir, 0, 99)
        
        with self.assertRaises(ValueError) as cm:
            validate_split_consistency(self.train_dir, self.val_dir, strict_forward=True)
        self.assertIn("Strict Forward Validation Failed", str(cm.exception))

    def test_backward_split_pass_loose(self):
        """Train [100..200], Val [0..99] -> Should Pass Loose (Disjoint)"""
        self._create_parquet(self.train_dir, 100, 200)
        self._create_parquet(self.val_dir, 0, 99)
        
        # strict=False
        self.assertTrue(validate_split_consistency(self.train_dir, self.val_dir, strict_forward=False))

    def test_empty_train(self):
        """Empty train dir -> Should Pass (warn only)"""
        self._create_parquet(self.val_dir, 0, 100)
        self.assertTrue(validate_split_consistency(self.train_dir, self.val_dir))

if __name__ == '__main__':
    unittest.main()
