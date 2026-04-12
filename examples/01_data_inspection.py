"""
Example 1 -- Data Schema Inspection
====================================
An AI agent can learn the *structure* of a dataset (column names, dtypes,
null counts, unique value counts, numeric summaries) **without ever seeing
individual rows**.

Use case: an agent needs to understand what data is available before
writing analysis code, but must not be exposed to raw records.
"""

import pandas as pd

from agent_privacy_layer import PrivacyLayer, UserConfirmation

# -----------------------------------------------------------------
# 1. Prepare a sample dataset (in production this would be loaded
#    by a trusted data owner, never by the agent itself).
# -----------------------------------------------------------------
df = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "email": [
            "alice@example.com",
            "bob@test.org",
            "charlie@mail.com",
            "diana@example.com",
            "eve@test.org",
        ],
        "age": [25, 30, 28, 35, 22],
        "salary": [50_000, 60_000, 55_000, 70_000, 45_000],
        "department": ["Engineering", "Sales", "Engineering", "Sales", "Engineering"],
    }
)

# -----------------------------------------------------------------
# 2. Create the privacy layer (dry_run=True auto-approves prompts).
# -----------------------------------------------------------------
layer = PrivacyLayer(
    df,
    epsilon=1.0,
    confirmation=UserConfirmation(dry_run=True),
)

# -----------------------------------------------------------------
# 3. Inspect the schema -- safe, no raw rows are returned.
# -----------------------------------------------------------------
schema = layer.inspect_data()
print("=== Full schema ===")
print(schema)
print()

# Individual accessors
print("Column names:", layer.column_names())
print("Dtypes:      ", layer.dtypes())
print("Shape:       ", layer.shape())
print()

# -----------------------------------------------------------------
# 4. Numeric summaries -- aggregate statistics, not individual values.
# -----------------------------------------------------------------
print("=== Numeric summary for 'age' ===")
age_summary = layer.numeric_summary("age")
for stat, value in age_summary.items():
    print(f"  {stat:>7s}: {value:.2f}")

print()
print("=== Numeric summary for 'salary' ===")
salary_summary = layer.numeric_summary("salary")
for stat, value in salary_summary.items():
    print(f"  {stat:>7s}: {value:.2f}")
