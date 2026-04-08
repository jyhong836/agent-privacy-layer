"""
Code Reviewer
=============
Reviews agent-generated Python code for prohibited data-access patterns.

Allowed patterns:
  - Statistical aggregations (groupby, describe, value_counts, …)
  - DP queries (opendp / dp_statistics)
  - Reading schema / dtypes / column names only

Prohibited patterns:
  - Opening raw files for reading (open(), read_csv, read_excel, …)
  - Row-level iteration over a protected DataFrame (iterrows, itertuples, iloc,
    loc, at, iat used for retrieval, head/tail that exposes values, …)
  - Printing or returning individual row data
"""

from __future__ import annotations

import ast
import dataclasses
from typing import List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Violation:
    """A single prohibited pattern found in the code."""

    line: int
    col: int
    message: str
    severity: str = "error"  # "error" | "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] Line {self.line}, Col {self.col}: {self.message}"


@dataclasses.dataclass
class CodeReviewResult:
    """The outcome of a code review."""

    approved: bool
    violations: List[Violation]
    warnings: List[Violation]

    def __str__(self) -> str:
        lines = [
            f"Code review: {'APPROVED' if self.approved else 'REJECTED'}",
            f"  Violations : {len(self.violations)}",
            f"  Warnings   : {len(self.warnings)}",
        ]
        for v in self.violations + self.warnings:
            lines.append(f"  {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prohibited call / attribute patterns
# ---------------------------------------------------------------------------

# Function / method calls that directly read raw data
_PROHIBITED_CALLS: dict[str, str] = {
    # built-in file open
    "open": "Direct file open() is prohibited. Use the privacy layer to access data.",
    # pandas read functions
    "read_csv": "Direct pd.read_csv() is prohibited. Use DataInspector or DPStatistics.",
    "read_excel": "Direct pd.read_excel() is prohibited. Use DataInspector or DPStatistics.",
    "read_json": "Direct pd.read_json() is prohibited. Use DataInspector or DPStatistics.",
    "read_parquet": "Direct pd.read_parquet() is prohibited. Use DataInspector or DPStatistics.",
    "read_table": "Direct pd.read_table() is prohibited. Use DataInspector or DPStatistics.",
    "read_hdf": "Direct pd.read_hdf() is prohibited. Use DataInspector or DPStatistics.",
    "read_feather": "Direct pd.read_feather() is prohibited. Use DataInspector or DPStatistics.",
    "read_sql": "Direct pd.read_sql() is prohibited. Use DataInspector or DPStatistics.",
    # row-level iteration
    "iterrows": "iterrows() exposes raw row data. Use aggregations or the DP layer.",
    "itertuples": "itertuples() exposes raw row data. Use aggregations or the DP layer.",
    "iteritems": "iteritems() exposes raw item data. Use aggregations or the DP layer.",
    # head / tail without aggregation context — warn, not error, because they
    # can legitimately be used to inspect schema
    "head": "head() may expose raw data rows. Ensure only schema inspection is intended.",
    "tail": "tail() may expose raw data rows. Ensure only schema inspection is intended.",
    # numpy load helpers
    "load": "np.load() / numpy.load() may directly expose raw array data.",
    "loadtxt": "np.loadtxt() may directly expose raw data.",
    "genfromtxt": "np.genfromtxt() may directly expose raw data.",
}

# Attribute accesses on DataFrame/Series that return individual values
_PROHIBITED_ATTRS: dict[str, str] = {
    "values": ".values exposes the underlying NumPy array of raw data.",
    "to_numpy": ".to_numpy() exposes raw data.",
    "to_list": ".to_list() exposes raw data.",
    "tolist": ".tolist() exposes raw data.",
}

# Subscript / slice patterns that are suspicious when used for retrieval
_PROHIBITED_SLICE_METHODS: set[str] = {"iloc", "loc", "at", "iat"}

# Calls that are only warnings (lower severity)
_WARNING_CALLS: set[str] = {"head", "tail"}


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------


class _PrivacyVisitor(ast.NodeVisitor):
    """Walk the AST and collect violations."""

    def __init__(self) -> None:
        self.violations: List[Violation] = []
        self.warnings: List[Violation] = []

    # -- helpers -------------------------------------------------------------

    def _add(self, node: ast.AST, message: str, is_warning: bool = False) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        v = Violation(line=line, col=col, message=message,
                      severity="warning" if is_warning else "error")
        if is_warning:
            self.warnings.append(v)
        else:
            self.violations.append(v)

    # -- visitors ------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func

        # e.g.  open(...)
        if isinstance(func, ast.Name):
            name = func.id
            if name in _PROHIBITED_CALLS:
                msg = _PROHIBITED_CALLS[name]
                self._add(node, msg, is_warning=(name in _WARNING_CALLS))

        # e.g.  pd.read_csv(...)  or  df.iterrows()
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in _PROHIBITED_CALLS:
                msg = _PROHIBITED_CALLS[attr]
                self._add(node, msg, is_warning=(attr in _WARNING_CALLS))
            elif attr in _PROHIBITED_ATTRS:
                self._add(node, _PROHIBITED_ATTRS[attr])

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        attr = node.attr

        # .values / .to_numpy / .to_list when used outside of a Call context
        # (i.e., the attribute is accessed but not called — still exposes data)
        if attr in _PROHIBITED_ATTRS:
            # Only flag if the parent context is NOT a Call (i.e., direct
            # attribute access like `df.values`).  We already handle the Call
            # case in visit_Call via the method name check there.
            # Here we conservatively warn on plain attribute access too.
            self._add(node, _PROHIBITED_ATTRS[attr])

        # .iloc / .loc / .at / .iat  ← flag the attribute access itself
        if attr in _PROHIBITED_SLICE_METHODS:
            self._add(
                node,
                f".{attr} accesses individual rows/values. "
                "Use aggregations or the DP layer instead.",
            )

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CodeReviewer:
    """
    Reviews Python source code for prohibited data-access patterns.

    Usage::

        reviewer = CodeReviewer()
        result = reviewer.review(source_code)
        if not result.approved:
            print(result)
    """

    def review(self, source_code: str) -> CodeReviewResult:
        """
        Parse *source_code* and return a :class:`CodeReviewResult`.

        Parameters
        ----------
        source_code:
            Python source as a string.

        Returns
        -------
        CodeReviewResult
            ``approved`` is ``True`` only when no *errors* are found.
            Warnings do not block approval.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            violation = Violation(
                line=exc.lineno or 0,
                col=exc.offset or 0,
                message=f"Syntax error: {exc.msg}",
            )
            return CodeReviewResult(approved=False, violations=[violation], warnings=[])

        visitor = _PrivacyVisitor()
        visitor.visit(tree)

        approved = len(visitor.violations) == 0
        return CodeReviewResult(
            approved=approved,
            violations=visitor.violations,
            warnings=visitor.warnings,
        )
