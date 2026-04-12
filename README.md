# agent-privacy-layer

A generalist privacy layer for AI agents to access data without seeing data.

Agents interact exclusively with the `PrivacyLayer` class instead of touching
the dataset directly. Every method is designed so that **no individual rows are
exposed** unless the user explicitly approves a PII-scrubbed release.

## Use Cases

### 1. Data Schema Inspection

An agent learns the *structure* of a dataset -- column names, dtypes, null
counts, unique-value counts, and aggregate numeric summaries -- **without ever
seeing individual rows**.

```python
layer = PrivacyLayer(df, confirmation=UserConfirmation(dry_run=True))

schema = layer.inspect_data()   # full schema overview
cols   = layer.column_names()   # ['name', 'age', 'salary', ...]
dtypes = layer.dtypes()         # {'age': 'int64', 'salary': 'float64', ...}
shape  = layer.shape()          # (1000, 5)
stats  = layer.numeric_summary("age")  # min, max, mean, std, median, q25, q75
```

See [`examples/01_data_inspection.py`](examples/01_data_inspection.py) for the
full runnable example.

### 2. Differential-Privacy Statistical Queries

An agent obtains aggregate statistics (count, mean, sum, variance, histogram,
bounds) with calibrated Laplace noise so that **no individual record can be
identified** from the answers.

```python
count = layer.dp_count("age")
mean  = layer.dp_mean("age", lower=18, upper=65)
total = layer.dp_sum("salary", lower=0, upper=200_000)
var   = layer.dp_variance("age", lower=18, upper=65)
hist  = layer.dp_histogram("department")
lo, hi = layer.dp_bounds("salary")

# batch queries
results = layer.query_statistics([
    {"type": "count", "column": "age"},
    {"type": "mean",  "column": "age", "lower": 0, "upper": 120},
    {"type": "histogram", "column": "department"},
])
```

See [`examples/02_dp_statistics.py`](examples/02_dp_statistics.py) for the
full runnable example.

### 3. Synthetic Data Generation

An agent gets synthetic data that preserves the schema (column names, dtypes,
categorical domains) of the real dataset but contains **no real values** --
useful for prototyping analysis code or building visualisations.

```python
# random values within observed ranges (default)
synthetic = layer.synthesize_data(n_rows=100)

# contextually realistic fakes via Faker (e.g. "email" -> fake emails)
synthetic = layer.synthesize_data(n_rows=100, strategy="faker")
```

See [`examples/03_synthetic_data.py`](examples/03_synthetic_data.py) for the
full runnable example.

### 4. Code Review & Safety Validation

Before executing agent-generated code, the code reviewer scans for
**prohibited data-access patterns** (row iteration, direct file reads,
individual value access) and blocks unsafe code.

```python
review = layer.review_code("result = df.groupby('dept')['sales'].sum()")
assert review.approved  # aggregation is safe

review = layer.review_code("for idx, row in df.iterrows(): print(row)")
assert not review.approved  # row iteration is blocked
```

See [`examples/04_code_review.py`](examples/04_code_review.py) for the
full runnable example.

### 5. PII-Scrubbed Real Data Access

When an agent genuinely needs real data (e.g. to debug a data-quality issue),
the privacy layer provides a PII-scrubbed copy -- but **only after explicit
user approval** (two confirmation rounds).

```python
result = layer.get_scrubbed_data(
    reason="Need to validate data quality for the 'notes' column",
    columns=["age", "salary", "notes"],
    max_rows=10,
)
print(result.data)    # PII-free DataFrame
print(result.report)  # what was removed
```

Scrubbing actions: `"redact"` (default), `"hash"`, or `"mask"`.

See [`examples/05_pii_scrubbing.py`](examples/05_pii_scrubbing.py) for the
full runnable example.

## End-to-End Workflow

[`examples/06_full_workflow.py`](examples/06_full_workflow.py) demonstrates the
complete privacy-preserving workflow: inspect, query, synthesize, review, and
scrub -- as a real agent framework would use it.

## Quick Start

```python
import pandas as pd
from agent_privacy_layer import PrivacyLayer, UserConfirmation

df = pd.read_csv("sensitive_data.csv")  # loaded by the data owner, not the agent

layer = PrivacyLayer(
    df,
    epsilon=1.0,                                # privacy budget
    confirmation=UserConfirmation(dry_run=True), # or omit for interactive prompts
    synthesis_strategy="random",                 # "random" or "faker"
    pii_action="redact",                         # "redact", "hash", or "mask"
)

# The agent interacts only with `layer` -- never with `df` directly.
```

## Examples

| Example | Use Case |
| ------- | -------- |
| [`01_data_inspection.py`](examples/01_data_inspection.py) | Schema inspection without seeing rows |
| [`02_dp_statistics.py`](examples/02_dp_statistics.py) | Differential-privacy statistical queries |
| [`03_synthetic_data.py`](examples/03_synthetic_data.py) | Synthetic data generation for prototyping |
| [`04_code_review.py`](examples/04_code_review.py) | Code review & safety validation |
| [`05_pii_scrubbing.py`](examples/05_pii_scrubbing.py) | PII-scrubbed real data access |
| [`06_full_workflow.py`](examples/06_full_workflow.py) | End-to-end agent workflow |

Run any example from the repository root:

```bash
python examples/01_data_inspection.py
```
