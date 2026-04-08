"""Tests for CodeReviewer."""
import pytest
from agent_privacy_layer.code_reviewer import CodeReviewer, CodeReviewResult


@pytest.fixture
def reviewer():
    return CodeReviewer()


# ---------------------------------------------------------------------------
# Approved code
# ---------------------------------------------------------------------------


def test_approve_empty_code(reviewer):
    result = reviewer.review("")
    assert result.approved


def test_approve_pure_arithmetic(reviewer):
    code = "x = 1 + 2\ny = x * 3\n"
    result = reviewer.review(code)
    assert result.approved
    assert len(result.violations) == 0


def test_approve_aggregation(reviewer):
    code = "total = df['age'].mean()\ncount = df['age'].count()\n"
    result = reviewer.review(code)
    assert result.approved


def test_approve_groupby(reviewer):
    code = "summary = df.groupby('gender')['age'].mean()\n"
    result = reviewer.review(code)
    assert result.approved


def test_approve_dp_query(reviewer):
    code = "from agent_privacy_layer import PrivacyLayer\nresult = layer.dp_mean('age', lower=0, upper=120)\n"
    result = reviewer.review(code)
    assert result.approved


def test_approve_dtypes_access(reviewer):
    code = "cols = df.columns\ndtypes = df.dtypes\n"
    result = reviewer.review(code)
    assert result.approved


def test_approve_value_counts(reviewer):
    code = "counts = df['category'].value_counts()\n"
    result = reviewer.review(code)
    assert result.approved


# ---------------------------------------------------------------------------
# Rejected code (errors)
# ---------------------------------------------------------------------------


def test_reject_open_file(reviewer):
    code = "f = open('data.csv', 'r')\n"
    result = reviewer.review(code)
    assert not result.approved
    assert len(result.violations) >= 1
    assert any("open" in v.message.lower() for v in result.violations)


def test_reject_read_csv(reviewer):
    code = "import pandas as pd\ndf = pd.read_csv('data.csv')\n"
    result = reviewer.review(code)
    assert not result.approved
    assert any("read_csv" in v.message for v in result.violations)


def test_reject_read_excel(reviewer):
    code = "df = pd.read_excel('data.xlsx')\n"
    result = reviewer.review(code)
    assert not result.approved


def test_reject_iterrows(reviewer):
    code = "for idx, row in df.iterrows():\n    print(row)\n"
    result = reviewer.review(code)
    assert not result.approved
    assert any("iterrows" in v.message for v in result.violations)


def test_reject_itertuples(reviewer):
    code = "for row in df.itertuples():\n    process(row)\n"
    result = reviewer.review(code)
    assert not result.approved


def test_reject_iloc(reviewer):
    code = "row = df.iloc[0]\n"
    result = reviewer.review(code)
    assert not result.approved
    assert any("iloc" in v.message for v in result.violations)


def test_reject_loc(reviewer):
    code = "row = df.loc[5]\n"
    result = reviewer.review(code)
    assert not result.approved


def test_reject_values_attribute(reviewer):
    code = "arr = df['age'].values\n"
    result = reviewer.review(code)
    assert not result.approved


def test_reject_to_numpy(reviewer):
    code = "arr = df.to_numpy()\n"
    result = reviewer.review(code)
    assert not result.approved


def test_reject_to_list(reviewer):
    code = "lst = df['name'].to_list()\n"
    result = reviewer.review(code)
    assert not result.approved


def test_reject_numpy_load(reviewer):
    code = "import numpy as np\ndata = np.load('data.npy')\n"
    result = reviewer.review(code)
    assert not result.approved


# ---------------------------------------------------------------------------
# Warnings (not errors)
# ---------------------------------------------------------------------------


def test_warn_head(reviewer):
    code = "sample = df.head(5)\n"
    result = reviewer.review(code)
    # head raises a warning, not an error
    assert result.approved
    assert len(result.warnings) >= 1
    assert any("head" in w.message.lower() for w in result.warnings)


def test_warn_tail(reviewer):
    code = "sample = df.tail(3)\n"
    result = reviewer.review(code)
    assert result.approved
    assert len(result.warnings) >= 1


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


def test_syntax_error(reviewer):
    code = "def broken(:\n    pass\n"
    result = reviewer.review(code)
    assert not result.approved
    assert len(result.violations) == 1
    assert "Syntax error" in result.violations[0].message


# ---------------------------------------------------------------------------
# Result string representation
# ---------------------------------------------------------------------------


def test_str_approved(reviewer):
    result = reviewer.review("x = 1\n")
    s = str(result)
    assert "APPROVED" in s


def test_str_rejected(reviewer):
    result = reviewer.review("open('data.csv')\n")
    s = str(result)
    assert "REJECTED" in s
