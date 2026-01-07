
import unittest
import numpy as np
from execution.regime.tracker import WindowedPercentileTracker
from execution.regime.detectors import HierarchicalRegimeDetector

class TestRegimeTracker(unittest.TestCase):
    def test_tracker_percentiles(self):
        """Test basic percentile calculations."""
        tracker = WindowedPercentileTracker(window_size=10)
        
        # Fill with 0-9
        for i in range(10):
            p = tracker.update(float(i))
        
        # 11 should be 100th percentile
        self.assertEqual(tracker.update(11.0), 100.0)
        
        # -1 should be 0th percentile
        self.assertEqual(tracker.update(-1.0), 0.0)
        
        # 5 should be roughly 50th
        p_mid = tracker.update(5.0)
        print(f"DEBUG: p_mid={p_mid}, history={list(tracker.history)}, sorted={tracker.sorted_window}")
        self.assertTrue(30.0 <= p_mid <= 70.0)

    def test_infinite_handling(self):
        """Test NaN/Inf handling."""
        tracker = WindowedPercentileTracker(window_size=10)
        
        # First value is 50%
        self.assertEqual(tracker.update(10.0), 50.0)
        
        # Inf should be 100% anomaly
        self.assertEqual(tracker.update(float('inf')), 100.0)
        self.assertEqual(tracker.update(float('nan')), 100.0)

    def test_detector_integration(self):
        """Test integration with HierarchicalRegimeDetector."""
        detector = HierarchicalRegimeDetector()
        
        # Mocking prices is hard for full detector, but we can check the tracker component
        # directly or via assess if we mock components.
        # Alternatively, we just check that assess doesn't crash on high error.
        
        # Use longer price series with gradual trend and minimal noise for LOW volatility
        # H3 Hardening: HIGH volatility now vetoes (trust=0.0), so we must generate LOW vol data
        np.random.seed(42)  # Reproducibility
        prices = np.linspace(100, 102, 500) + np.random.normal(0, 0.01, 500)  # Low noise
        
        # Normal error
        # With linear trend and low volatility, trust should be > 0.3
        res_normal = detector.assess(prices, reconstruction_error=0.1)
        print(f"Normal trust: {res_normal.trust_score}, Details: {res_normal.details}")
        # With H3 hardened: HIGH vol -> 0.0, MEDIUM -> 0.7, LOW -> 1.0
        # This data should yield LOW volatility -> trust near 1.0 before other penalties
        self.assertGreater(res_normal.trust_score, 0.3)
        
        # Extreme error (simulated burst)
        # We need to prime the tracker with low values first so this one looks high
        for _ in range(50):
            detector.recon_tracker.update(0.1)
            
        res_high = detector.assess(prices, reconstruction_error=100.0) # Massive error
        
        # Should be penalized heavily (99th percentile -> trust * 0.0)
        self.assertEqual(res_high.trust_score, 0.0)
        self.assertIn("recon_percentile", res_high.details)

if __name__ == "__main__":
    unittest.main()
