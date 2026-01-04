import numpy as np

class HurstExponentEstimator:
    """Estimate the Hurst exponent for regime detection."""

    def __init__(self, min_window: int = 8, max_window: int | None = None):
        self.min_window = min_window
        self.max_window = max_window

    def estimate(self, prices: np.ndarray) -> float:
        """Estimate Hurst exponent from price series."""
        if len(prices) < self.min_window * 2:
            return 0.5  # Default to random walk

        returns = np.diff(np.log(prices + 1e-10))

        if len(returns) < self.min_window:
            return 0.5

        max_window = self.max_window or len(returns) // 2

        window_sizes = []
        rs_values = []

        for window in range(self.min_window, min(max_window + 1, len(returns) + 1)):
            n_windows = len(returns) // window
            if n_windows < 1:
                continue

            truncated_len = n_windows * window
            segments = returns[:truncated_len].reshape(n_windows, window)
            means = np.mean(segments, axis=1, keepdims=True)
            deviations = segments - means
            cumdev = np.cumsum(deviations, axis=1)
            R = np.max(cumdev, axis=1) - np.min(cumdev, axis=1)
            S = np.std(segments, axis=1, ddof=1)

            valid = S > 0
            if np.any(valid):
                avg_rs = np.mean(R[valid] / S[valid])
                window_sizes.append(window)
                rs_values.append(avg_rs)

        if len(window_sizes) < 2:
            return 0.5

        log_n = np.log(window_sizes)
        log_rs = np.log(rs_values)

        n = len(log_n)
        sum_x = np.sum(log_n)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_n * log_rs)
        sum_xx = np.sum(log_n ** 2)

        denominator = n * sum_xx - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.5

        hurst = (n * sum_xy - sum_x * sum_y) / denominator
        return float(np.clip(hurst, 0.0, 1.0))
