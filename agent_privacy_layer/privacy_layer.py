"""
Privacy Layer (main orchestrator)
==================================
The ``PrivacyLayer`` is the top-level skill that coordinates all privacy
components.  An AI agent should interact *exclusively* with this class instead
of touching the dataset directly.

Workflow
--------
1. **Inspect** – ``inspect_data()`` – learn column names, dtypes, basic stats.
2. **Query statistics** – ``query_statistics()`` – get DP-noised aggregates.
3. **Synthesize** – ``synthesize_data()`` – get fake data for prototyping.
4. **Review code** – ``review_code()`` – check agent-generated code for
   prohibited data-access patterns before execution.
5. **Scrub and share** (escalated, requires user approval) –
   ``get_scrubbed_data()`` – PII-free subset of real data, only after the
   user confirms the request is necessary.

All escalated paths (steps that expose any form of real data) require
explicit user approval via :class:`UserConfirmation`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .code_reviewer import CodeReviewer, CodeReviewResult
from .data_inspector import DataInspector, DataSchema
from .data_synthesizer import DataSynthesizer
from .dp_statistics import DPStatistics
from .pii_scrubber import PIIScrubber, PIIScrubResult
from .user_confirmation import UserConfirmation


class PrivacyLayer:
    """
    The main privacy-layer skill for AI agents.

    Parameters
    ----------
    data:
        The protected DataFrame.  The agent should never receive a reference
        to this object directly.
    epsilon:
        Privacy budget (ε) for differential-privacy queries.
    confirmation:
        :class:`UserConfirmation` instance.  If ``None`` a default one using
        the CLI prompt is created.  Pass ``UserConfirmation(dry_run=True)``
        to skip all prompts (e.g. in tests).
    synthesis_strategy:
        Strategy for :class:`DataSynthesizer`: ``"random"`` or ``"faker"``.
    pii_action:
        Scrubbing action for :class:`PIIScrubber`: ``"redact"``, ``"hash"``,
        or ``"mask"``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        epsilon: float = 1.0,
        confirmation: Optional[UserConfirmation] = None,
        synthesis_strategy: str = "random",
        pii_action: str = "redact",
    ) -> None:
        self._data = data
        self._reviewer = CodeReviewer()
        self._inspector = DataInspector()
        self._dp = DPStatistics(epsilon=epsilon)
        self._synthesizer = DataSynthesizer(strategy=synthesis_strategy)
        self._scrubber = PIIScrubber(action=pii_action)
        self._confirmation = confirmation or UserConfirmation()

    # ------------------------------------------------------------------
    # 1. Inspect – always allowed, returns no raw rows
    # ------------------------------------------------------------------

    def inspect_data(self) -> DataSchema:
        """
        Return the schema and aggregate properties of the protected dataset.

        Safe to expose to the agent – no individual rows are returned.
        """
        return self._inspector.inspect(self._data)

    def column_names(self) -> List[str]:
        """Return the column names of the dataset."""
        return self._inspector.column_names(self._data)

    def dtypes(self) -> Dict[str, str]:
        """Return a ``{column: dtype}`` mapping."""
        return self._inspector.dtypes(self._data)

    def shape(self) -> tuple[int, int]:
        """Return ``(num_rows, num_columns)``."""
        return self._inspector.shape(self._data)

    def numeric_summary(self, column: str) -> Dict[str, Optional[float]]:
        """Return aggregate numeric statistics for *column* (no raw rows)."""
        return self._inspector.numeric_summary(self._data, column)

    # ------------------------------------------------------------------
    # 2. DP Statistics – always allowed
    # ------------------------------------------------------------------

    def dp_count(self, column: str, *, epsilon: Optional[float] = None) -> float:
        """Differentially-private count of non-null values in *column*."""
        return self._dp.count(self._data, column, epsilon=epsilon)

    def dp_mean(
        self,
        column: str,
        *,
        lower: float,
        upper: float,
        epsilon: Optional[float] = None,
    ) -> float:
        """Differentially-private mean of *column*."""
        return self._dp.mean(self._data, column, lower=lower, upper=upper, epsilon=epsilon)

    def dp_sum(
        self,
        column: str,
        *,
        lower: float,
        upper: float,
        epsilon: Optional[float] = None,
    ) -> float:
        """Differentially-private sum of *column*."""
        return self._dp.sum(self._data, column, lower=lower, upper=upper, epsilon=epsilon)

    def dp_variance(
        self,
        column: str,
        *,
        lower: float,
        upper: float,
        epsilon: Optional[float] = None,
    ) -> float:
        """Differentially-private variance of *column*."""
        return self._dp.variance(self._data, column, lower=lower, upper=upper, epsilon=epsilon)

    def dp_histogram(
        self, column: str, *, epsilon: Optional[float] = None
    ) -> Dict[Any, float]:
        """Differentially-private frequency histogram for *column*."""
        return self._dp.histogram(self._data, column, epsilon=epsilon)

    def dp_bounds(
        self, column: str, *, epsilon: Optional[float] = None
    ) -> tuple[float, float]:
        """Differentially-private (lower, upper) bounds for *column*."""
        return self._dp.bounds(self._data, column, epsilon=epsilon)

    def query_statistics(
        self,
        queries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute a batch of DP statistical queries.

        Each query dict must have a ``"type"`` key (one of ``"count"``,
        ``"mean"``, ``"sum"``, ``"variance"``, ``"histogram"``, ``"bounds"``)
        and a ``"column"`` key.  Numeric queries also require ``"lower"`` and
        ``"upper"`` bounds.

        Returns a dict ``{query_key: result}`` where *query_key* is
        ``"{type}:{column}"``.

        Example::

            results = layer.query_statistics([
                {"type": "count",     "column": "age"},
                {"type": "mean",      "column": "age",    "lower": 0, "upper": 120},
                {"type": "histogram", "column": "gender"},
            ])
        """
        results: Dict[str, Any] = {}
        for q in queries:
            q_type = q.get("type", "").lower()
            col = q.get("column", "")
            key = f"{q_type}:{col}"
            try:
                if q_type == "count":
                    results[key] = self.dp_count(col, epsilon=q.get("epsilon"))
                elif q_type == "mean":
                    results[key] = self.dp_mean(
                        col, lower=q["lower"], upper=q["upper"], epsilon=q.get("epsilon")
                    )
                elif q_type == "sum":
                    results[key] = self.dp_sum(
                        col, lower=q["lower"], upper=q["upper"], epsilon=q.get("epsilon")
                    )
                elif q_type == "variance":
                    results[key] = self.dp_variance(
                        col, lower=q["lower"], upper=q["upper"], epsilon=q.get("epsilon")
                    )
                elif q_type == "histogram":
                    results[key] = self.dp_histogram(col, epsilon=q.get("epsilon"))
                elif q_type == "bounds":
                    results[key] = self.dp_bounds(col, epsilon=q.get("epsilon"))
                else:
                    results[key] = {"error": f"Unknown query type: {q_type!r}"}
            except Exception as exc:  # noqa: BLE001
                results[key] = {"error": str(exc)}
        return results

    # ------------------------------------------------------------------
    # 3. Synthetic data – always allowed
    # ------------------------------------------------------------------

    def synthesize_data(
        self, n_rows: Optional[int] = None, strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Return a synthetic DataFrame with the same schema as the real data.

        No real values are included.  Safe to share with the agent.

        Parameters
        ----------
        n_rows:
            Number of synthetic rows.  Defaults to same as real data.
        strategy:
            Override the default synthesis strategy (``"random"`` or
            ``"faker"``).
        """
        synthesizer = self._synthesizer
        if strategy is not None and strategy != self._synthesizer.strategy:
            synthesizer = DataSynthesizer(strategy=strategy)
        return synthesizer.synthesize(self._data, n_rows=n_rows)

    # ------------------------------------------------------------------
    # 4. Code review – always available
    # ------------------------------------------------------------------

    def review_code(self, source_code: str) -> CodeReviewResult:
        """
        Review *source_code* for prohibited data-access patterns.

        The agent should call this before executing any code that touches
        data.  Returns a :class:`CodeReviewResult`; execution should be
        blocked if ``result.approved`` is ``False``.
        """
        return self._reviewer.review(source_code)

    # ------------------------------------------------------------------
    # 5. Scrubbed data – escalated, requires user approval
    # ------------------------------------------------------------------

    def get_scrubbed_data(
        self,
        reason: str,
        *,
        columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> PIIScrubResult:
        """
        Return PII-scrubbed rows from the real dataset.

        This is an escalated action.  The user must approve the request.

        Parameters
        ----------
        reason:
            Why the agent needs access to real (scrubbed) data.
        columns:
            Subset of columns to return.  ``None`` means all columns.
        max_rows:
            Maximum number of rows to return.

        Returns
        -------
        PIIScrubResult
            Contains the scrubbed DataFrame and a report of what was removed.

        Raises
        ------
        PermissionError
            If the user denies the request.
        """
        confirmation_message = (
            f"An AI agent is requesting access to real (PII-scrubbed) data.\n"
            f"Reason: {reason}\n"
            f"Columns: {columns or 'all'}\n"
            f"Max rows: {max_rows or 'all'}"
        )
        self._confirmation.require(confirmation_message)

        subset = self._data
        if columns is not None:
            missing = [c for c in columns if c not in subset.columns]
            if missing:
                raise KeyError(f"Columns not found: {missing}")
            subset = subset[columns]
        if max_rows is not None:
            subset = subset.iloc[:max_rows]

        scrubber = PIIScrubber(action=self._scrubber.action, columns=columns)
        result = scrubber.scrub(subset)

        # Second confirmation: show what PII was found and ask for final approval
        pii_info = (
            str(result.report)
            if result.report.cells_modified > 0
            else "No PII detected."
        )
        final_message = (
            f"PII scan complete.\n{pii_info}\n"
            "Approve sharing this scrubbed data with the agent?"
        )
        self._confirmation.require(final_message)

        return result
