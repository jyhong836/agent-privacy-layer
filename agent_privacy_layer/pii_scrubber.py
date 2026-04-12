"""
PII Scrubber
============
Detects and removes Personally Identifiable Information (PII) from a pandas
DataFrame before it is shared with an agent.

Detection is performed via regex patterns and is intentionally conservative
(it may produce false positives to err on the side of privacy).

PII categories detected
-----------------------
* Email addresses
* US/international phone numbers
* US Social Security Numbers (SSN)
* US ZIP codes (when column name contains "zip"/"postal")
* Credit card numbers
* IP addresses (IPv4 and IPv6)
* URLs containing personal tokens
* Names (heuristic: columns named "name", "first_name", "last_name", …)
* US dates of birth (heuristic: columns named "dob", "birth_date", …)

Scrubbing actions
-----------------
``"redact"``   Replace the detected PII with ``"[REDACTED]"``.
``"hash"``     Replace with a SHA-256 hash (preserves equality relationships
               without exposing the value).
``"mask"``     Replace with ``"***"`` (fast, irreversible).
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Dict, List, Optional, Set

import pandas as pd


# ---------------------------------------------------------------------------
# PII regex patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "phone": re.compile(
        r"(\+?1[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})"
    ),
    "ssn": re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|"
        r"3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
    "ipv4": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    "ipv6": re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    ),
    "url_with_token": re.compile(
        r"https?://[^\s]+[?&](?:token|key|secret|password|auth)=[^\s&]+"
    ),
}

# Column-name heuristics that suggest the *entire* column is PII
_PII_COLUMN_PATTERNS: List[re.Pattern] = [
    re.compile(r"(^|\b)(email|e[-_]?mail)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(phone|tel(ephone)?|mobile|cell)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(ssn|social[\s_-]?security)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(name|first[\s_-]?name|last[\s_-]?name|full[\s_-]?name)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(dob|birth[\s_-]?date|date[\s_-]?of[\s_-]?birth)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(address|street|addr)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(zip[\s_-]?code|postal[\s_-]?code)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(credit[\s_-]?card|card[\s_-]?number)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(password|passwd)(\b|$)", re.IGNORECASE),
    re.compile(r"(^|\b)(ip[\s_-]?addr(ess)?)(\b|$)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ScrubReport:
    """Summary of what was scrubbed."""

    columns_scrubbed: List[str]
    cells_modified: int
    pii_types_found: Set[str]

    def __str__(self) -> str:
        types = ", ".join(sorted(self.pii_types_found)) if self.pii_types_found else "none"
        return (
            f"PII Scrub Report:\n"
            f"  Columns scrubbed : {', '.join(self.columns_scrubbed) or 'none'}\n"
            f"  Cells modified   : {self.cells_modified}\n"
            f"  PII types found  : {types}"
        )


@dataclasses.dataclass
class PIIScrubResult:
    """The scrubbed DataFrame together with a report."""

    data: pd.DataFrame
    report: ScrubReport

    def __str__(self) -> str:
        return str(self.report)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PIIScrubber:
    """
    Detect and remove PII from a DataFrame.

    Parameters
    ----------
    action:
        What to do with detected PII values:
        ``"redact"`` (default) – replace with ``"[REDACTED]"``.
        ``"hash"``             – replace with hex SHA-256 digest.
        ``"mask"``             – replace with ``"***"``.
    columns:
        Explicit list of column names to scrub.  When ``None``, columns are
        auto-detected using name heuristics and value-level regex scanning.
    scan_values:
        When ``True``, run regex patterns over cell values in string columns
        even if the column name does not appear suspicious.
    """

    def __init__(
        self,
        action: str = "redact",
        columns: Optional[List[str]] = None,
        scan_values: bool = True,
    ) -> None:
        if action not in ("redact", "hash", "mask"):
            raise ValueError("action must be 'redact', 'hash', or 'mask'")
        self.action = action
        self.columns = columns
        self.scan_values = scan_values

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrub(self, data: pd.DataFrame) -> PIIScrubResult:
        """
        Scrub PII from *data* and return a :class:`PIIScrubResult`.

        The original DataFrame is **not** modified; a copy is returned.
        """
        df = data.copy()
        columns_scrubbed: List[str] = []
        cells_modified = 0
        pii_types_found: Set[str] = set()

        target_columns = self.columns if self.columns is not None else list(df.columns)

        for col in target_columns:
            if col not in df.columns:
                continue

            col_is_pii = self._column_name_is_pii(col)
            modified_in_col = 0

            if col_is_pii:
                # Scrub the entire column
                pii_types_found.add("column_name_heuristic")
                before = df[col].copy()
                df[col] = df[col].apply(lambda v: self._replace(str(v)) if pd.notna(v) else v)
                modified_in_col = int((df[col] != before).sum())
            elif self.scan_values and pd.api.types.is_string_dtype(df[col]):
                # Scan individual cell values for PII patterns
                for idx in df.index:
                    cell = df.at[idx, col]
                    if not isinstance(cell, str):
                        continue
                    new_cell, found_types = self._scrub_string(cell)
                    if found_types:
                        df.at[idx, col] = new_cell
                        pii_types_found.update(found_types)
                        modified_in_col += 1

            if modified_in_col > 0:
                columns_scrubbed.append(col)
                cells_modified += modified_in_col

        report = ScrubReport(
            columns_scrubbed=columns_scrubbed,
            cells_modified=cells_modified,
            pii_types_found=pii_types_found,
        )
        return PIIScrubResult(data=df, report=report)

    def detect(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Return a mapping of ``{column: [pii_types]}`` for detected PII without
        modifying *data*.
        """
        result: Dict[str, List[str]] = {}

        for col in data.columns:
            found: List[str] = []

            if self._column_name_is_pii(col):
                found.append("column_name_heuristic")

            if self.scan_values and pd.api.types.is_string_dtype(data[col]):
                for cell in data[col].dropna():
                    if isinstance(cell, str):
                        _, types = self._scrub_string(cell)
                        for t in types:
                            if t not in found:
                                found.append(t)

            if found:
                result[col] = found

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _column_name_is_pii(col: str) -> bool:
        return any(p.search(col) for p in _PII_COLUMN_PATTERNS)

    def _scrub_string(self, text: str) -> tuple[str, Set[str]]:
        """Replace PII matches in *text* and return (new_text, {pii_types})."""
        found_types: Set[str] = set()
        for pii_type, pattern in _PII_PATTERNS.items():
            if pattern.search(text):
                found_types.add(pii_type)
                text = pattern.sub(self._replace("\\g<0>"), text)
        return text, found_types

    def _replace(self, value: str) -> str:
        """Return the replacement string for *value*."""
        if self.action == "redact":
            return "[REDACTED]"
        if self.action == "hash":
            return hashlib.sha256(value.encode()).hexdigest()
        # mask
        return "***"
