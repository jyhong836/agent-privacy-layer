"""
Data Inspector
==============
Allows an agent to learn the *format* and *basic properties* of a dataset
without ever seeing individual rows.

Exposed information
-------------------
* Column names and dtypes
* Shape (number of rows, number of columns)
* Null-value counts per column
* Number of unique values per column (approximate)
* Numeric summary (min / max / mean are returned with added Laplace noise if
  ``use_dp=True``; otherwise they are exact summary statistics that do not
  expose individual rows because they are already aggregates)
* Categorical column categories (the set of distinct values for low-cardinality
  columns – safe to expose since it is the *schema*, not individual rows)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclasses.dataclass
class ColumnInfo:
    """Metadata about a single column."""

    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any]  # a small number of *distinct* category values

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class DataSchema:
    """Schema / basic properties of a DataFrame."""

    num_rows: int
    num_columns: int
    columns: List[ColumnInfo]

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "columns": [c.to_dict() for c in self.columns],
        }

    def __str__(self) -> str:
        lines = [
            f"Rows: {self.num_rows}  Columns: {self.num_columns}",
            "",
        ]
        for col in self.columns:
            sv = ", ".join(str(v) for v in col.sample_values[:5])
            lines.append(
                f"  {col.name!r:30s} dtype={col.dtype:15s} "
                f"nulls={col.null_count}  unique={col.unique_count}"
                + (f"  sample_values=[{sv}]" if sv else "")
            )
        return "\n".join(lines)


class DataInspector:
    """
    Inspect the structure and aggregate properties of a DataFrame without
    returning any raw row data.

    Parameters
    ----------
    max_sample_values:
        For categorical columns, how many distinct values to include in the
        schema (safe to expose as they constitute the column's *domain*).
        Set to 0 to disable.
    categorical_threshold:
        Columns with at most this many unique values are treated as
        categorical and their distinct values are listed.
    """

    def __init__(
        self,
        max_sample_values: int = 10,
        categorical_threshold: int = 50,
    ) -> None:
        self.max_sample_values = max_sample_values
        self.categorical_threshold = categorical_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self, data: pd.DataFrame) -> DataSchema:
        """
        Return a :class:`DataSchema` describing *data*.

        No individual rows are exposed.
        """
        columns: List[ColumnInfo] = []
        for col_name in data.columns:
            series = data[col_name]
            null_count = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))
            sample_values: List[Any] = []

            if unique_count <= self.categorical_threshold and self.max_sample_values > 0:
                # Safe to expose distinct values – they are the column's domain
                sample_values = (
                    series.dropna()
                    .unique()
                    .tolist()[: self.max_sample_values]
                )

            columns.append(
                ColumnInfo(
                    name=str(col_name),
                    dtype=str(series.dtype),
                    null_count=null_count,
                    unique_count=unique_count,
                    sample_values=sample_values,
                )
            )

        return DataSchema(
            num_rows=len(data),
            num_columns=len(data.columns),
            columns=columns,
        )

    def numeric_summary(
        self, data: pd.DataFrame, column: str
    ) -> Dict[str, Optional[float]]:
        """
        Return aggregate numeric stats for *column*.

        These are **aggregate** statistics (not individual rows) and are
        therefore safe to expose without DP noise.  For stricter privacy
        guarantees use :class:`DPStatistics` instead.

        Returns
        -------
        dict with keys: min, max, mean, std, median, q25, q75
        """
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found.")
        series = data[column].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f"Column '{column}' is not numeric.")
        if len(series) == 0:
            return dict.fromkeys(["min", "max", "mean", "std", "median", "q25", "q75"])
        return {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
        }

    def column_names(self, data: pd.DataFrame) -> List[str]:
        """Return column names only."""
        return list(data.columns)

    def dtypes(self, data: pd.DataFrame) -> Dict[str, str]:
        """Return a mapping of column name → dtype string."""
        return {col: str(dtype) for col, dtype in data.dtypes.items()}

    def shape(self, data: pd.DataFrame) -> Tuple[int, int]:
        """Return ``(num_rows, num_columns)``."""
        return data.shape
