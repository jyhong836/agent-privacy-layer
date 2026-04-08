"""Tests for DataSynthesizer."""
import pytest
import pandas as pd
import numpy as np
from agent_privacy_layer.data_synthesizer import DataSynthesizer


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=100).tolist(),
            "salary": rng.uniform(20_000, 200_000, size=100).tolist(),
            "gender": rng.choice(["M", "F", "NB"], size=100).tolist(),
            "active": rng.choice([True, False], size=100).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_synthesize_same_columns(df):
    synth = DataSynthesizer(seed=42)
    result = synth.synthesize(df)
    assert list(result.columns) == list(df.columns)


def test_synthesize_same_row_count_by_default(df):
    synth = DataSynthesizer(seed=42)
    result = synth.synthesize(df)
    assert len(result) == len(df)


def test_synthesize_custom_row_count(df):
    synth = DataSynthesizer(seed=42)
    result = synth.synthesize(df, n_rows=50)
    assert len(result) == 50


def test_synthesize_values_differ_from_real(df):
    """Synthetic data should not be identical to real data."""
    synth = DataSynthesizer(seed=7)
    result = synth.synthesize(df)
    # It is astronomically unlikely that all 100 age values are identical
    assert not (result["age"].values == df["age"].values).all()


# ---------------------------------------------------------------------------
# Dtypes preserved approximately
# ---------------------------------------------------------------------------


def test_synthesize_integer_column(df):
    synth = DataSynthesizer(seed=42)
    result = synth.synthesize(df)
    # age values should all be integers
    for v in result["age"]:
        assert isinstance(v, (int, np.integer))


def test_synthesize_float_column(df):
    synth = DataSynthesizer(seed=42)
    result = synth.synthesize(df)
    for v in result["salary"]:
        assert isinstance(v, float)


def test_synthesize_bool_column(df):
    synth = DataSynthesizer(seed=42)
    result = synth.synthesize(df)
    for v in result["active"]:
        assert v in (True, False)


def test_synthesize_categorical_values_within_domain(df):
    synth = DataSynthesizer(seed=42)
    result = synth.synthesize(df)
    assert set(result["gender"].unique()).issubset({"M", "F", "NB"})


# ---------------------------------------------------------------------------
# Faker strategy
# ---------------------------------------------------------------------------


def test_faker_strategy_returns_correct_shape(df):
    synth = DataSynthesizer(strategy="faker", seed=1)
    result = synth.synthesize(df)
    assert list(result.columns) == list(df.columns)
    assert len(result) == len(df)


def test_faker_strategy_email_column():
    email_df = pd.DataFrame({"email": ["a@b.com", "c@d.org"] * 5})
    synth = DataSynthesizer(strategy="faker", seed=42)
    result = synth.synthesize(email_df)
    # All generated emails should contain '@'
    for v in result["email"]:
        assert "@" in str(v)


def test_faker_strategy_name_column():
    name_df = pd.DataFrame({"name": ["Alice", "Bob"] * 5})
    synth = DataSynthesizer(strategy="faker", seed=42)
    result = synth.synthesize(name_df)
    for v in result["name"]:
        assert isinstance(v, str) and len(v) > 0


# ---------------------------------------------------------------------------
# Invalid strategy
# ---------------------------------------------------------------------------


def test_invalid_strategy():
    with pytest.raises(ValueError, match="strategy must be"):
        DataSynthesizer(strategy="llm")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility(df):
    s1 = DataSynthesizer(seed=99).synthesize(df)
    s2 = DataSynthesizer(seed=99).synthesize(df)
    pd.testing.assert_frame_equal(s1, s2)
