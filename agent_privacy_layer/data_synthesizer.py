"""
Data Synthesizer
================
Generate synthetic data that preserves the *format* (column names, dtypes,
categorical domains) of a real DataFrame but contains **no real values**.

This lets an agent work with realistic-looking data for development or
testing purposes without ever seeing the real rows.

Two synthesis strategies are available:

``"random"`` (default)
    Fills numeric columns with uniform-random values between observed min/max,
    and categorical columns with uniformly-sampled category values.  Fast, no
    external dependencies beyond pandas / numpy.

``"faker"``
    Uses the `Faker <https://faker.readthedocs.io/>`_ library to generate
    contextually realistic fake values based on the column name heuristic
    (e.g. a column named ``email`` will get fake email addresses).  Falls back
    to ``"random"`` if a column cannot be matched.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from faker import Faker as _Faker

    _FAKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Faker column-name heuristics
# ---------------------------------------------------------------------------

# Map regex patterns (matched against lower-cased column names) to Faker methods
_FAKER_COLUMN_MAP: List[tuple[str, str]] = [
    (r"email", "email"),
    (r"phone|tel(ephone)?|mobile", "phone_number"),
    (r"(first|given)[\s_-]?name", "first_name"),
    (r"(last|family|sur)[\s_-]?name", "last_name"),
    (r"^name$|full[\s_-]?name|display[\s_-]?name", "name"),
    (r"address|street|addr", "street_address"),
    (r"city", "city"),
    (r"state|province", "state"),
    (r"zip|postal[\s_-]?code", "zipcode"),
    (r"country", "country"),
    (r"company|employer|org(anization)?", "company"),
    (r"job[\s_-]?title|position|role|occupation", "job"),
    (r"(birth[\s_-]?)?date$|dob$|born$", "date_of_birth"),
    (r"date", "date"),
    (r"url|website|web[\s_-]?site", "url"),
    (r"ssn|social[\s_-]?security", "ssn"),
    (r"ip[\s_-]?addr(ess)?", "ipv4"),
    (r"user[\s_-]?name|login|handle", "user_name"),
    (r"password|passwd", "password"),
    (r"uuid|guid", "uuid4"),
    (r"description|bio|about|summary|text|comment", "text"),
]


def _faker_method_for_column(col_name: str) -> Optional[str]:
    """Return the Faker method name for *col_name*, or ``None``."""
    lower = col_name.lower()
    for pattern, method in _FAKER_COLUMN_MAP:
        if re.search(pattern, lower):
            return method
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DataSynthesizer:
    """
    Generate a synthetic DataFrame that mirrors the schema of a real one.

    Parameters
    ----------
    strategy:
        ``"random"`` – random values within observed ranges.
        ``"faker"``  – contextually realistic fake values via Faker.
    locale:
        Faker locale (e.g. ``"en_US"``).  Ignored for ``"random"`` strategy.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        strategy: str = "random",
        locale: str = "en_US",
        seed: Optional[int] = None,
    ) -> None:
        if strategy not in ("random", "faker"):
            raise ValueError("strategy must be 'random' or 'faker'")
        self.strategy = strategy
        self.locale = locale
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        if strategy == "faker":
            if not _FAKER_AVAILABLE:
                raise ImportError(  # pragma: no cover
                    "The 'faker' package is required for strategy='faker'. "
                    "Install it with: pip install faker"
                )
            self._faker = _Faker(locale)
            if seed is not None:
                _Faker.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, data: pd.DataFrame, n_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Return a synthetic DataFrame with the same columns as *data*.

        Parameters
        ----------
        data:
            The *real* DataFrame used only to infer column schema.
        n_rows:
            Number of synthetic rows to generate.  Defaults to the same
            number of rows as *data*.
        """
        n = n_rows if n_rows is not None else len(data)
        synthetic: Dict[str, Any] = {}

        for col in data.columns:
            series = data[col].dropna()
            if self.strategy == "faker":
                synthetic[col] = self._faker_column(col, series, n)
            else:
                synthetic[col] = self._random_column(col, series, n)

        return pd.DataFrame(synthetic, columns=list(data.columns))

    # ------------------------------------------------------------------
    # Internal column generators
    # ------------------------------------------------------------------

    def _random_column(self, col: str, series: pd.Series, n: int) -> List[Any]:
        """Generate *n* random values matching the dtype of *series*."""
        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            return self._rng.choice([True, False], size=n).tolist()

        if pd.api.types.is_integer_dtype(dtype):
            lo = int(series.min()) if len(series) > 0 else 0
            hi = int(series.max()) if len(series) > 0 else 100
            if lo == hi:
                hi = lo + 1
            return self._rng.integers(lo, hi + 1, size=n).tolist()

        if pd.api.types.is_float_dtype(dtype):
            lo = float(series.min()) if len(series) > 0 else 0.0
            hi = float(series.max()) if len(series) > 0 else 1.0
            if lo == hi:
                hi = lo + 1.0
            return (self._rng.random(size=n) * (hi - lo) + lo).tolist()

        # Categorical / object / string → sample from observed unique values
        unique_vals = series.unique().tolist() if len(series) > 0 else ["<unknown>"]
        indices = self._rng.integers(0, len(unique_vals), size=n)
        return [unique_vals[i] for i in indices]

    def _faker_column(self, col: str, series: pd.Series, n: int) -> List[Any]:
        """Generate *n* Faker values for *col*, falling back to random."""
        method_name = _faker_method_for_column(col)
        if method_name and hasattr(self._faker, method_name):
            method = getattr(self._faker, method_name)
            return [method() for _ in range(n)]
        # Fall back to random if no heuristic matches
        return self._random_column(col, series, n)
