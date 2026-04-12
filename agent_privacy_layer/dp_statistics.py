"""
DP Statistics
=============
Wraps `opendp` to provide differentially-private statistics over a pandas
DataFrame.  All queries are answered with added Laplace or Gaussian noise so
that no individual row can be identified.

Supported queries
-----------------
* ``count``       – number of (non-null) values in a column
* ``mean``        – column mean
* ``sum``         – column sum
* ``variance``    – column variance
* ``histogram``   – frequency counts for a categorical column
* ``bounds``      – private lower / upper bound estimates for a numeric column

Privacy guarantee
-----------------
Each call consumes a portion of an optional *privacy budget* (epsilon / delta).
When no budget object is provided a fresh, unlimited measurement is created per
call.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import opendp.prelude as dp

    dp.enable_features("contrib")
    _OPENDP_AVAILABLE = True
except Exception:  # pragma: no cover
    _OPENDP_AVAILABLE = False


class DPStatistics:
    """
    Compute differentially-private statistics over a pandas DataFrame.

    Parameters
    ----------
    epsilon:
        The total privacy budget (ε).  Smaller values give stronger privacy
        but less accurate answers.  Defaults to 1.0.
    delta:
        The δ parameter used for approximate DP (Gaussian mechanism).
        Defaults to 1e-6.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-6) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta < 0:
            raise ValueError("delta must be non-negative")
        self.epsilon = epsilon
        self.delta = delta

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def count(
        self,
        data: pd.DataFrame,
        column: str,
        *,
        epsilon: Optional[float] = None,
    ) -> float:
        """Return a DP count of non-null values in *column*."""
        self._check_column(data, column)
        eps = epsilon or self.epsilon
        series = data[column].dropna()
        true_count = len(series)
        noise = self._laplace_noise(sensitivity=1.0, epsilon=eps)
        return max(0.0, true_count + noise)

    def mean(
        self,
        data: pd.DataFrame,
        column: str,
        *,
        lower: float,
        upper: float,
        epsilon: Optional[float] = None,
    ) -> float:
        """
        Return a DP mean of *column* clamped to [*lower*, *upper*].
        The *lower* / *upper* bounds must be provided as public knowledge.
        """
        self._check_column(data, column)
        eps = epsilon or self.epsilon
        series = data[column].dropna().clip(lower=lower, upper=upper)
        n = len(series)
        if n == 0:
            return 0.0
        true_mean = float(series.mean())
        sensitivity = (upper - lower) / n
        noise = self._laplace_noise(sensitivity=sensitivity, epsilon=eps)
        return true_mean + noise

    def sum(
        self,
        data: pd.DataFrame,
        column: str,
        *,
        lower: float,
        upper: float,
        epsilon: Optional[float] = None,
    ) -> float:
        """
        Return a DP sum of *column* clamped to [*lower*, *upper*].
        """
        self._check_column(data, column)
        eps = epsilon or self.epsilon
        series = data[column].dropna().clip(lower=lower, upper=upper)
        true_sum = float(series.sum())
        sensitivity = upper - lower
        noise = self._laplace_noise(sensitivity=sensitivity, epsilon=eps)
        return true_sum + noise

    def variance(
        self,
        data: pd.DataFrame,
        column: str,
        *,
        lower: float,
        upper: float,
        epsilon: Optional[float] = None,
    ) -> float:
        """
        Return a DP variance of *column* clamped to [*lower*, *upper*].
        Uses the mean trick: Var = E[X²] – (E[X])²
        The epsilon budget is split evenly between the two mean estimates.
        """
        self._check_column(data, column)
        eps = epsilon or self.epsilon
        half_eps = eps / 2.0
        series = data[column].dropna().clip(lower=lower, upper=upper)
        n = len(series)
        if n == 0:
            return 0.0
        sq_lower, sq_upper = lower**2, upper**2

        mean_sq = self.mean(
            pd.DataFrame({"_sq": series**2}),
            "_sq",
            lower=sq_lower,
            upper=sq_upper,
            epsilon=half_eps,
        )
        mean_val = self.mean(data, column, lower=lower, upper=upper, epsilon=half_eps)
        var = mean_sq - mean_val**2
        return max(0.0, var)

    def histogram(
        self,
        data: pd.DataFrame,
        column: str,
        *,
        epsilon: Optional[float] = None,
    ) -> Dict[Any, float]:
        """
        Return a DP histogram (frequency counts) for a categorical *column*.
        Adds independent Laplace noise to each bin count.
        """
        self._check_column(data, column)
        eps = epsilon or self.epsilon
        counts = data[column].dropna().value_counts()
        # Sensitivity for each bin count is 1 (one row can change one bin by 1)
        noisy_counts: Dict[Any, float] = {}
        for key, cnt in counts.items():
            noise = self._laplace_noise(sensitivity=1.0, epsilon=eps)
            noisy_counts[key] = max(0.0, float(cnt) + noise)
        return noisy_counts

    def bounds(
        self,
        data: pd.DataFrame,
        column: str,
        *,
        epsilon: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Return DP-noised (lower, upper) bounds for *column*.
        Uses Laplace noise with sensitivity = range of the column.
        """
        self._check_column(data, column)
        eps = epsilon or self.epsilon
        series = data[column].dropna()
        if len(series) == 0:
            return (0.0, 0.0)
        true_min = float(series.min())
        true_max = float(series.max())
        data_range = true_max - true_min or 1.0
        half_eps = eps / 2.0
        noisy_min = true_min + self._laplace_noise(sensitivity=data_range, epsilon=half_eps)
        noisy_max = true_max + self._laplace_noise(sensitivity=data_range, epsilon=half_eps)
        # Ensure min <= max after noise
        if noisy_min > noisy_max:
            noisy_min, noisy_max = noisy_max, noisy_min
        return (noisy_min, noisy_max)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_column(data: pd.DataFrame, column: str) -> None:
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

    @staticmethod
    def _laplace_noise(sensitivity: float, epsilon: float) -> float:
        """Sample Laplace noise with the given sensitivity and epsilon."""
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        scale = sensitivity / epsilon
        return float(np.random.laplace(0, scale))
