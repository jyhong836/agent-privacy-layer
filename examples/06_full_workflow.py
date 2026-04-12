"""
Example 6 -- End-to-End Agent Workflow
======================================
Demonstrates the complete privacy-preserving workflow that an AI agent would
follow when analysing a sensitive dataset:

  1. Inspect  -- learn the schema without seeing rows.
  2. Query    -- obtain DP-noised statistics.
  3. Synthesize -- get fake data for prototyping code.
  4. Review   -- validate proposed analysis code before execution.
  5. Scrub    -- (escalated) request real data with PII removed.

This mirrors how a real agent framework (e.g. LangChain, AutoGen, CrewAI)
would integrate with the privacy layer.
"""

import pandas as pd

from agent_privacy_layer import PrivacyLayer, UserConfirmation

# -----------------------------------------------------------------
# Setup: a trusted data owner loads the dataset and wraps it.
# -----------------------------------------------------------------
df = pd.DataFrame(
    {
        "patient_id": range(1, 51),
        "name": [f"Patient_{i}" for i in range(1, 51)],
        "email": [f"patient{i}@hospital.org" for i in range(1, 51)],
        "age": [20 + (i * 3 % 60) for i in range(50)],
        "blood_pressure": [110 + (i * 7 % 40) for i in range(50)],
        "cholesterol": [150 + (i * 11 % 100) for i in range(50)],
        "diagnosis": (
            ["Healthy"] * 20 + ["Hypertension"] * 15 + ["Diabetes"] * 15
        ),
    }
)

layer = PrivacyLayer(
    df,
    epsilon=1.0,
    confirmation=UserConfirmation(dry_run=True),
    synthesis_strategy="random",
    pii_action="redact",
)

# =================================================================
# Step 1: Inspect -- the agent discovers what data is available.
# =================================================================
print("=" * 60)
print("STEP 1: Data Inspection")
print("=" * 60)

schema = layer.inspect_data()
print(schema)
print()
print(f"Columns available: {layer.column_names()}")
print(f"Dataset shape:     {layer.shape()}")
print()

age_stats = layer.numeric_summary("age")
print("Age statistics (aggregates, no individual rows):")
for k, v in age_stats.items():
    print(f"  {k:>7s}: {v:.1f}")
print()

# =================================================================
# Step 2: Query -- the agent asks statistical questions with DP.
# =================================================================
print("=" * 60)
print("STEP 2: Differential-Privacy Queries")
print("=" * 60)

results = layer.query_statistics(
    [
        {"type": "count", "column": "age"},
        {"type": "mean", "column": "age", "lower": 0, "upper": 100},
        {"type": "mean", "column": "blood_pressure", "lower": 80, "upper": 200},
        {"type": "mean", "column": "cholesterol", "lower": 100, "upper": 300},
        {"type": "histogram", "column": "diagnosis"},
        {"type": "bounds", "column": "blood_pressure"},
    ]
)
print("Batch query results:")
for key, value in results.items():
    print(f"  {key:35s}: {value}")
print()

# =================================================================
# Step 3: Synthesize -- the agent gets fake data for prototyping.
# =================================================================
print("=" * 60)
print("STEP 3: Synthetic Data for Prototyping")
print("=" * 60)

synthetic = layer.synthesize_data(n_rows=5)
print("Synthetic data (5 rows, no real values):")
print(synthetic.to_string(index=False))
print()

# The agent can now use this synthetic data to develop analysis code
# without any risk of exposing real patient information.

# =================================================================
# Step 4: Review -- validate the agent's proposed code.
# =================================================================
print("=" * 60)
print("STEP 4: Code Review")
print("=" * 60)

# The agent proposes this analysis code:
proposed_code = """\
avg_bp_by_diagnosis = df.groupby('diagnosis')['blood_pressure'].mean()
high_risk_count = df[df['cholesterol'] > 200]['diagnosis'].value_counts()
summary = df[['age', 'blood_pressure', 'cholesterol']].describe()
"""

review = layer.review_code(proposed_code)
print(f"Proposed code review: {review}")
print()

if review.approved:
    print("Code is safe to execute -- no prohibited patterns found.")
else:
    print("Code BLOCKED -- violations found. Agent must revise.")
print()

# Now try unsafe code:
unsafe_code = """\
for idx, row in df.iterrows():
    print(row['name'], row['email'])
raw = df.values
"""

review_unsafe = layer.review_code(unsafe_code)
print(f"Unsafe code review: {review_unsafe}")
print()

# =================================================================
# Step 5: Scrub -- escalated access to real (PII-free) data.
# =================================================================
print("=" * 60)
print("STEP 5: PII-Scrubbed Real Data (Escalated)")
print("=" * 60)

result = layer.get_scrubbed_data(
    reason="Need to verify blood_pressure outliers in the real data",
    columns=["age", "blood_pressure", "cholesterol", "diagnosis"],
    max_rows=5,
)
print("Scrubbed real data (user approved, PII removed):")
print(result.data.to_string(index=False))
print()
print(result.report)
print()

# =================================================================
# Summary
# =================================================================
print("=" * 60)
print("WORKFLOW COMPLETE")
print("=" * 60)
print("""
The agent was able to:
  1. Learn the dataset schema without seeing any rows.
  2. Obtain DP-noised statistics for its analysis.
  3. Prototype code using synthetic data.
  4. Get its proposed code validated before execution.
  5. Access a PII-scrubbed subset of real data (with user approval).

At no point did the agent have direct access to the raw DataFrame.
""")
