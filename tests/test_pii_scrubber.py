"""Tests for PIIScrubber."""
import pytest
import pandas as pd
from agent_privacy_layer.pii_scrubber import PIIScrubber, PIIScrubResult, ScrubReport


@pytest.fixture
def pii_df():
    return pd.DataFrame(
        {
            "name": ["Alice Smith", "Bob Jones"],
            "email": ["alice@example.com", "bob@test.org"],
            "phone": ["555-123-4567", "+1 (800) 555-0199"],
            "age": [30, 45],
            "salary": [60000, 75000],
            "notes": [
                "Contact at alice@example.com or 555-123-4567",
                "SSN: 123-45-6789",
            ],
        }
    )


@pytest.fixture
def scrubber():
    return PIIScrubber(action="redact")


# ---------------------------------------------------------------------------
# scrub() – basic functionality
# ---------------------------------------------------------------------------


def test_scrub_returns_result(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    assert isinstance(result, PIIScrubResult)


def test_scrub_does_not_modify_original(scrubber, pii_df):
    original_email = pii_df["email"].copy()
    scrubber.scrub(pii_df)
    pd.testing.assert_series_equal(pii_df["email"], original_email)


def test_scrub_name_column_redacted(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    for v in result.data["name"]:
        assert v == "[REDACTED]"


def test_scrub_email_column_redacted(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    for v in result.data["email"]:
        assert v == "[REDACTED]"


def test_scrub_phone_column_redacted(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    for v in result.data["phone"]:
        assert v == "[REDACTED]"


def test_scrub_non_pii_columns_intact(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    assert list(result.data["age"]) == [30, 45]
    assert list(result.data["salary"]) == [60000, 75000]


def test_scrub_email_in_notes(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    # Email pattern in notes should be scrubbed
    for v in result.data["notes"]:
        assert "alice@example.com" not in str(v)


def test_scrub_report_lists_pii_columns(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    assert "name" in result.report.columns_scrubbed
    assert "email" in result.report.columns_scrubbed


def test_scrub_report_cells_modified(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    assert result.report.cells_modified > 0


def test_scrub_report_pii_types(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    assert len(result.report.pii_types_found) > 0


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def test_action_mask():
    df = pd.DataFrame({"email": ["user@example.com"]})
    scrubber = PIIScrubber(action="mask")
    result = scrubber.scrub(df)
    assert result.data["email"].iloc[0] == "***"


def test_action_hash():
    df = pd.DataFrame({"email": ["user@example.com"]})
    scrubber = PIIScrubber(action="hash")
    result = scrubber.scrub(df)
    hashed = result.data["email"].iloc[0]
    assert len(hashed) == 64  # SHA-256 hex digest


def test_invalid_action():
    with pytest.raises(ValueError, match="action must be"):
        PIIScrubber(action="delete")


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_returns_pii_columns(scrubber, pii_df):
    found = scrubber.detect(pii_df)
    assert "name" in found
    assert "email" in found
    assert "phone" in found


def test_detect_non_pii_not_flagged(scrubber, pii_df):
    found = scrubber.detect(pii_df)
    assert "age" not in found
    assert "salary" not in found


# ---------------------------------------------------------------------------
# Explicit column list
# ---------------------------------------------------------------------------


def test_explicit_columns_only_scrubs_those(pii_df):
    scrubber = PIIScrubber(action="redact", columns=["email"])
    result = scrubber.scrub(pii_df)
    for v in result.data["email"]:
        assert v == "[REDACTED]"
    # name column should be untouched
    assert list(result.data["name"]) == list(pii_df["name"])


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


def test_str_report(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    s = str(result)
    assert "PII Scrub Report" in s


def test_str_result(scrubber, pii_df):
    result = scrubber.scrub(pii_df)
    s = str(result)
    assert "PII" in s
