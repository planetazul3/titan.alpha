import sys
from unittest.mock import MagicMock

# Mock talib if not present to allow tests to run on environments without C-libs
try:
    import talib
except ImportError:
    msg = "TA-Lib not found. Mocking for structural verification."
    print(msg)
    
    mock_talib = MagicMock()
    # Common functions used in indicators
    mock_talib.RSI = MagicMock(return_value=[50.0] * 100)
    mock_talib.BBANDS = MagicMock(return_value=([100.0]*100, [100.0]*100, [100.0]*100))
    mock_talib.EMA = MagicMock(return_value=[100.0] * 100)
    mock_talib.ATR = MagicMock(return_value=[1.0] * 100)
    
    sys.modules["talib"] = mock_talib
    
# Mock deriv_api
try:
    import deriv_api
except ImportError:
    print("deriv_api not found. Mocking.")
    mock_deriv = MagicMock()
    mock_deriv.DerivAPI = MagicMock
    mock_deriv.APIError = Exception
    sys.modules["deriv_api"] = mock_deriv
    
# Mock pandera
try:
    import pandera
except ImportError:
    print("pandera not found. Mocking.")
    mock_pandera = MagicMock()
    mock_pandera.typing = MagicMock()
    sys.modules["pandera"] = mock_pandera
    sys.modules["pandera.typing"] = mock_pandera.typing
    
# Mock pandas
try:
    import pandas
except ImportError:
    print("pandas not found. Mocking.")
    sys.modules["pandas"] = MagicMock()


