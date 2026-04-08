"""Tests for DPStatistics."""
import pytest
import numpy as np
import pandas as pd
from agent_privacy_layer.dp_statistics import DPStatistics


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=1000),
            "salary": rng.uniform(20_000, 200_000, size=1000),
            "gender": rng.choice(["M", "F", "NB"], size=1000),
        }
    )


@pytest.fixture
def dp():
    return DPStatistics(epsilon=5.0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon must be positive"):
        DPStatistics(epsilon=0)


def test_invalid_delta():
    with pytest.raises(ValueError, match="delta must be non-negative"):
        DPStatistics(epsilon=1.0, delta=-0.1)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


def test_count_returns_nonnegative(dp, df):
    result = dp.count(df, "age")
    assert result >= 0


def test_count_close_to_true(df):
    # With high epsilon, noise is tiny
    dp_hi = DPStatistics(epsilon=1000.0)
    result = dp_hi.count(df, "age")
    assert abs(result - 1000) < 50


def test_count_missing_column(dp, df):
    with pytest.raises(KeyError):
        dp.count(df, "nonexistent")


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------


def test_mean_within_bounds(df):
    dp_hi = DPStatistics(epsilon=1000.0)
    result = dp_hi.mean(df, "age", lower=0, upper=120)
    true_mean = float(df["age"].mean())
    assert abs(result - true_mean) < 5


def test_mean_missing_column(dp, df):
    with pytest.raises(KeyError):
        dp.mean(df, "nope", lower=0, upper=100)


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


def test_sum_close_to_true(df):
    dp_hi = DPStatistics(epsilon=1000.0)
    result = dp_hi.sum(df, "age", lower=0, upper=120)
    true_sum = float(df["age"].sum())
    # Allow 5% error
    assert abs(result - true_sum) / true_sum < 0.05


# ---------------------------------------------------------------------------
# variance
# ---------------------------------------------------------------------------


def test_variance_nonnegative(dp, df):
    result = dp.variance(df, "age", lower=0, upper=120)
    assert result >= 0


def test_variance_close_to_true(df):
    dp_hi = DPStatistics(epsilon=500.0)
    result = dp_hi.variance(df, "age", lower=0, upper=120)
    true_var = float(df["age"].var())
    # Allow wide margin due to the mean-of-squares trick noise
    assert abs(result - true_var) / (true_var + 1) < 0.5


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------


def test_histogram_returns_dict(dp, df):
    result = dp.histogram(df, "gender")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"M", "F", "NB"}


def test_histogram_nonnegative(dp, df):
    result = dp.histogram(df, "gender")
    for v in result.values():
        assert v >= 0


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


def test_bounds_returns_tuple(dp, df):
    lo, hi = dp.bounds(df, "age")
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_bounds_lo_lte_hi(dp, df):
    lo, hi = dp.bounds(df, "age")
    assert lo <= hi


def test_bounds_empty_column(dp):
    empty_df = pd.DataFrame({"x": pd.Series([], dtype=float)})
    lo, hi = dp.bounds(empty_df, "x")
    assert lo == hi == 0.0
