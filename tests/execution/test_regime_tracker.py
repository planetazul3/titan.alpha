
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
        
        prices = np.full(100, 100.0) # Constant prices -> Low Volatility, Sideways
        # update: constant prices might trigger issues in Hurst (div by zero?) 
        # Better: slow trend
        prices = np.linspace(100, 105, 100)
        
        # Normal error
        # Note: detector penalties apply (Micro=Random if not enough data/trend?). 
        # With linear trend, hurst should be high (Trending).
        res_normal = detector.assess(prices, reconstruction_error=0.1)
        print(f"Normal trust: {res_normal.trust_score}, Details: {res_normal.details}")
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
