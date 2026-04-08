"""Tests for PrivacyLayer (main orchestrator)."""
import pytest
import numpy as np
import pandas as pd
from agent_privacy_layer.privacy_layer import PrivacyLayer
from agent_privacy_layer.user_confirmation import UserConfirmation
from agent_privacy_layer.code_reviewer import CodeReviewResult
from agent_privacy_layer.data_inspector import DataSchema
from agent_privacy_layer.pii_scrubber import PIIScrubResult


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "email": [
                "alice@example.com",
                "bob@test.org",
                "charlie@mail.com",
                "diana@web.net",
                "eve@sample.io",
            ],
            "age": rng.integers(18, 80, size=5).tolist(),
            "salary": rng.uniform(30_000, 150_000, size=5).tolist(),
            "gender": rng.choice(["M", "F", "NB"], size=5).tolist(),
        }
    )


@pytest.fixture
def layer(sample_df):
    """PrivacyLayer with dry-run confirmation (no interactive prompts)."""
    return PrivacyLayer(
        sample_df,
        epsilon=10.0,
        confirmation=UserConfirmation(dry_run=True),
    )


# ---------------------------------------------------------------------------
# inspect_data
# ---------------------------------------------------------------------------


def test_inspect_data_returns_schema(layer):
    schema = layer.inspect_data()
    assert isinstance(schema, DataSchema)
    assert schema.num_columns == 5
    assert schema.num_rows == 5


def test_column_names(layer):
    names = layer.column_names()
    assert "age" in names
    assert "name" in names


def test_dtypes(layer):
    dtypes = layer.dtypes()
    assert "age" in dtypes


def test_shape(layer):
    assert layer.shape() == (5, 5)


def test_numeric_summary(layer):
    summary = layer.numeric_summary("age")
    assert "min" in summary
    assert "max" in summary


# ---------------------------------------------------------------------------
# DP Statistics
# ---------------------------------------------------------------------------


def test_dp_count(layer):
    result = layer.dp_count("age")
    assert result >= 0


def test_dp_mean(layer):
    result = layer.dp_mean("age", lower=0, upper=120)
    assert isinstance(result, float)


def test_dp_sum(layer):
    result = layer.dp_sum("salary", lower=0, upper=200_000)
    assert isinstance(result, float)


def test_dp_variance(layer):
    result = layer.dp_variance("age", lower=0, upper=120)
    assert result >= 0


def test_dp_histogram(layer):
    result = layer.dp_histogram("gender")
    assert isinstance(result, dict)


def test_dp_bounds(layer):
    lo, hi = layer.dp_bounds("age")
    assert lo <= hi


def test_query_statistics_batch(layer):
    queries = [
        {"type": "count", "column": "age"},
        {"type": "mean", "column": "age", "lower": 0, "upper": 120},
        {"type": "histogram", "column": "gender"},
    ]
    results = layer.query_statistics(queries)
    assert "count:age" in results
    assert "mean:age" in results
    assert "histogram:gender" in results


def test_query_statistics_unknown_type(layer):
    results = layer.query_statistics([{"type": "median", "column": "age"}])
    assert "error" in results["median:age"]


def test_query_statistics_bad_column(layer):
    results = layer.query_statistics([{"type": "count", "column": "nonexistent"}])
    assert "error" in results["count:nonexistent"]


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def test_synthesize_data_schema(layer):
    synthetic = layer.synthesize_data()
    assert list(synthetic.columns) == list(layer._data.columns)


def test_synthesize_data_custom_rows(layer):
    synthetic = layer.synthesize_data(n_rows=20)
    assert len(synthetic) == 20


def test_synthesize_data_faker_strategy(layer):
    synthetic = layer.synthesize_data(strategy="faker")
    assert list(synthetic.columns) == list(layer._data.columns)


# ---------------------------------------------------------------------------
# Code review
# ---------------------------------------------------------------------------


def test_review_code_approved(layer):
    code = "result = df.groupby('gender')['age'].mean()\n"
    result = layer.review_code(code)
    assert isinstance(result, CodeReviewResult)
    assert result.approved


def test_review_code_rejected(layer):
    code = "for idx, row in df.iterrows():\n    print(row)\n"
    result = layer.review_code(code)
    assert not result.approved


# ---------------------------------------------------------------------------
# Scrubbed data (escalated)
# ---------------------------------------------------------------------------


def test_get_scrubbed_data_approved(layer, sample_df):
    scrub_result = layer.get_scrubbed_data("Unit test access", max_rows=3)
    assert isinstance(scrub_result, PIIScrubResult)
    assert len(scrub_result.data) == 3


def test_get_scrubbed_data_pii_removed(layer, sample_df):
    scrub_result = layer.get_scrubbed_data("Unit test access")
    # email column should be scrubbed
    for v in scrub_result.data["email"]:
        assert "example.com" not in str(v)


def test_get_scrubbed_data_column_subset(layer):
    scrub_result = layer.get_scrubbed_data("Test", columns=["age", "gender"])
    assert set(scrub_result.data.columns) == {"age", "gender"}


def test_get_scrubbed_data_denied(sample_df):
    layer_deny = PrivacyLayer(
        sample_df,
        confirmation=UserConfirmation(auto_deny=True),
    )
    with pytest.raises(PermissionError):
        layer_deny.get_scrubbed_data("Denied request")


def test_get_scrubbed_data_missing_column(layer):
    with pytest.raises(KeyError):
        layer.get_scrubbed_data("Test", columns=["nonexistent"])
