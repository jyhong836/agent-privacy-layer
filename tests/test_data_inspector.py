"""Tests for DataInspector."""
import pytest
import pandas as pd
import numpy as np
from agent_privacy_layer.data_inspector import DataInspector, DataSchema, ColumnInfo


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, None],
            "salary": [50000.0, 60000.0, 75000.0, 80000.0, 55000.0],
            "gender": ["M", "F", "M", "F", "NB"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        }
    )


@pytest.fixture
def inspector():
    return DataInspector()


# ---------------------------------------------------------------------------
# inspect()
# ---------------------------------------------------------------------------


def test_inspect_returns_schema(inspector, df):
    schema = inspector.inspect(df)
    assert isinstance(schema, DataSchema)


def test_inspect_num_columns(inspector, df):
    schema = inspector.inspect(df)
    assert schema.num_columns == 4


def test_inspect_num_rows(inspector, df):
    schema = inspector.inspect(df)
    assert schema.num_rows == 5


def test_inspect_column_names(inspector, df):
    schema = inspector.inspect(df)
    assert schema.column_names() == ["age", "salary", "gender", "name"]


def test_inspect_null_count(inspector, df):
    schema = inspector.inspect(df)
    age_info = next(c for c in schema.columns if c.name == "age")
    assert age_info.null_count == 1


def test_inspect_unique_count(inspector, df):
    schema = inspector.inspect(df)
    gender_info = next(c for c in schema.columns if c.name == "gender")
    assert gender_info.unique_count == 3


def test_inspect_sample_values_categorical(inspector, df):
    schema = inspector.inspect(df)
    gender_info = next(c for c in schema.columns if c.name == "gender")
    assert set(gender_info.sample_values) == {"M", "F", "NB"}


def test_inspect_no_sample_values_high_cardinality():
    """High-cardinality column should have no sample values."""
    many_unique = pd.DataFrame({"id": range(1000)})
    inspector = DataInspector(categorical_threshold=50)
    schema = inspector.inspect(many_unique)
    id_info = schema.columns[0]
    assert id_info.sample_values == []


def test_inspect_to_dict(inspector, df):
    schema = inspector.inspect(df)
    d = schema.to_dict()
    assert "num_rows" in d
    assert "num_columns" in d
    assert "columns" in d
    assert len(d["columns"]) == 4


def test_inspect_str(inspector, df):
    schema = inspector.inspect(df)
    s = str(schema)
    assert "Rows" in s
    assert "age" in s


# ---------------------------------------------------------------------------
# numeric_summary()
# ---------------------------------------------------------------------------


def test_numeric_summary_keys(inspector, df):
    summary = inspector.numeric_summary(df, "salary")
    for key in ("min", "max", "mean", "std", "median", "q25", "q75"):
        assert key in summary


def test_numeric_summary_correct_min_max(inspector, df):
    summary = inspector.numeric_summary(df, "salary")
    assert summary["min"] == pytest.approx(50000.0)
    assert summary["max"] == pytest.approx(80000.0)


def test_numeric_summary_missing_column(inspector, df):
    with pytest.raises(KeyError):
        inspector.numeric_summary(df, "nonexistent")


def test_numeric_summary_non_numeric(inspector, df):
    with pytest.raises(TypeError):
        inspector.numeric_summary(df, "gender")


def test_numeric_summary_empty_column(inspector):
    empty = pd.DataFrame({"x": pd.Series([], dtype=float)})
    summary = inspector.numeric_summary(empty, "x")
    assert all(v is None for v in summary.values())


# ---------------------------------------------------------------------------
# column_names(), dtypes(), shape()
# ---------------------------------------------------------------------------


def test_column_names(inspector, df):
    assert inspector.column_names(df) == ["age", "salary", "gender", "name"]


def test_dtypes(inspector, df):
    dtypes = inspector.dtypes(df)
    assert "age" in dtypes
    assert "float" in dtypes["age"] or "int" in dtypes["age"]


def test_shape(inspector, df):
    assert inspector.shape(df) == (5, 4)
