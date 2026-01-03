
import unittest
import math
import tempfile
import shutil
import logging
from pathlib import Path
from utils.numerical_validation import ensure_finite, validate_numeric_dict
from execution.adaptive_risk import PerformanceTracker
from execution.safety_store import SQLiteSafetyStateStore

class TestNumericalValidation(unittest.TestCase):
    def test_ensure_finite_valid(self):
        """Test that finite values pass through unchanged."""
        self.assertEqual(ensure_finite(10.5, "test"), 10.5)
        self.assertEqual(ensure_finite(-5, "test"), -5)
        self.assertEqual(ensure_finite(0.0, "test"), 0.0)

    def test_ensure_finite_nan(self):
        """Test that NaN is caught and replaced."""
        with self.assertLogs(level='WARNING') as cm:
            result = ensure_finite(float('nan'), "test_metric", default=0.0)
            self.assertEqual(result, 0.0)
            self.assertTrue(any("Non-finite value detected for test_metric" in o for o in cm.output))

    def test_ensure_finite_inf(self):
        """Test that Inf is caught and replaced."""
        with self.assertLogs(level='WARNING') as cm:
            result = ensure_finite(float('inf'), "test_metric", default=1.0)
            self.assertEqual(result, 1.0)
            self.assertTrue(any("Non-finite value detected for test_metric" in o for o in cm.output))
            
    def test_validate_numeric_dict(self):
        """Test validation of a dictionary of metrics."""
        input_data = {
            "valid": 100.0,
            "bad_nan": float('nan'),
            "bad_inf": float('inf'),
            "ignored_str": "string_value"
        }
        
        defaults = {
            "bad_nan": -1.0
        }
        
        # We expect logs for the bad values
        with self.assertLogs(level='WARNING'):
            result = validate_numeric_dict(input_data, defaults)
            
        self.assertEqual(result["valid"], 100.0)
        self.assertEqual(result["bad_nan"], -1.0) # From defaults dict
        self.assertEqual(result["bad_inf"], 0.0)  # Default fallthrough
        self.assertEqual(result["ignored_str"], "string_value")

    def test_performance_tracker_hardening(self):
        """Test PerformanceTracker with bad inputs."""
        tracker = PerformanceTracker()
        
        # 1. Test NaN P&L - should be converted to 0.0
        with self.assertLogs(level='WARNING') as cm:
            tracker.record(float('nan'), current_equity=1000.0)
            # Should have recorded 0.0
            self.assertEqual(tracker._returns[-1], 0.0)
            self.assertTrue(any("Non-finite value detected for PerformanceTracker.pnl" in o for o in cm.output))

        # 2. Test Inf Equity - should use safe default (previous peak or 0)
        tracker._peak_equity = 1000.0
        with self.assertLogs(level='WARNING') as cm:
            tracker.record(10.0, current_equity=float('inf'))
            # Check logs
            self.assertTrue(any("Non-finite value detected for PerformanceTracker.current_equity" in o for o in cm.output))
            # Current drawdown calculation would likely fail if equity was Inf, but ensuring check logic holds.
            # If default logic works, current equity used was 1000.0 (previous peak)
            # So drawdown should be 0 or unchanged.
            self.assertEqual(tracker._current_drawdown, 0.0)

    def test_safety_store_hardening(self):
        """Test that SafetyStore rejects infinite values during update and load."""
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(temp_dir) / "safety_test.db"
            store = SQLiteSafetyStateStore(db_path)
            
            # 1. Update with bad values - should log warning and save safe defaults
            with self.assertLogs(level='WARNING') as cm:
                store.update_risk_metrics(
                    drawdown=float('inf'), # Bad
                    losses=5,
                    peak_equity=float('nan') # Bad
                )
                self.assertTrue(any("Non-finite value detected for persisting_drawdown" in o for o in cm.output))
                self.assertTrue(any("Non-finite value detected for persisting_peak_equity" in o for o in cm.output))
            
            # 2. Verify what was saved (should be 0.0)
            metrics = store.get_risk_metrics()
            self.assertEqual(metrics['current_drawdown'], 0.0)
            self.assertEqual(metrics['peak_equity'], 0.0)
            
            # 3. Test Loading Corruption (Simulate DB corruption manually)
            # We insert "Inf" string into DB to simulate what happens if bad data got in (e.g. from old version)
            store.set_value("risk_current_drawdown", "inf")
            store.set_value("risk_peak_equity", "nan")
            
            # Now load it back
            with self.assertLogs(level='WARNING') as cm:
                metrics = store.get_risk_metrics()
                
                # Should be sanitized to 0.0
                self.assertEqual(metrics['current_drawdown'], 0.0)
                self.assertEqual(metrics['peak_equity'], 0.0)
                
        finally:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()
