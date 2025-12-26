"""
Verification test for ID-002: Barrier Parsing in DecisionEngine.
"""
import pytest
from unittest.mock import MagicMock
from execution.decision import DecisionEngine

def test_barrier_parsing():
    # Setup
    settings = MagicMock()
    engine = DecisionEngine(settings)
    
    # Test cases: (input_metadata, expected_value)
    test_cases = [
        ({"barrier": 0.5}, 0.5),
        ({"barrier": "0.5"}, 0.5),
        ({"barrier": "+0.5"}, 0.5),
        ({"barrier": "-0.5"}, -0.5),
        ({"barrier": "  +1.23  "}, 1.23),
        ({"barrier": None}, None),
        ({"barrier": ""}, None),
        ({"other": "0.5"}, None),
        ({"barrier": "invalid"}, None),
    ]
    
    for metadata, expected in test_cases:
        result = engine._extract_barrier_value(metadata, "barrier")
        assert result == expected, f"Failed for {metadata}, expected {expected}, got {result}"

if __name__ == "__main__":
    test_barrier_parsing()
    print("Barrier parsing verification PASSED")
