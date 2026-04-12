"""
Example 2 -- Differential-Privacy Statistical Queries
=====================================================
An AI agent can obtain aggregate statistics (count, mean, sum, variance,
histogram, bounds) with calibrated Laplace noise so that **no individual
record can be identified** from the answers.

Use case: an agent needs to compute statistics over sensitive data (e.g.
employee salaries, patient ages) while providing formal privacy guarantees.
"""

import pandas as pd

from agent_privacy_layer import PrivacyLayer, UserConfirmation

# -----------------------------------------------------------------
# 1. Prepare a sample HR dataset.
# -----------------------------------------------------------------
df = pd.DataFrame(
    {
        "employee_id": range(1, 101),
        "age": [25 + (i * 7 % 40) for i in range(100)],
        "salary": [40_000 + (i * 311 % 60_000) for i in range(100)],
        "department": (["Engineering"] * 40 + ["Sales"] * 30 + ["HR"] * 30),
        "gender": (["F", "M"] * 50),
    }
)

layer = PrivacyLayer(
    df,
    epsilon=1.0,  # privacy budget -- smaller = more private, noisier
    confirmation=UserConfirmation(dry_run=True),
)

# -----------------------------------------------------------------
# 2. Individual DP queries.
# -----------------------------------------------------------------
print("=== Individual DP queries ===")

count = layer.dp_count("age")
print(f"DP count of 'age':                {count:.1f}")

mean_age = layer.dp_mean("age", lower=18, upper=65)
print(f"DP mean of 'age' [18, 65]:        {mean_age:.2f}")

total_salary = layer.dp_sum("salary", lower=0, upper=100_000)
print(f"DP sum of 'salary' [0, 100k]:     {total_salary:,.0f}")

var_age = layer.dp_variance("age", lower=18, upper=65)
print(f"DP variance of 'age' [18, 65]:    {var_age:.2f}")

bounds = layer.dp_bounds("salary")
print(f"DP bounds of 'salary':            ({bounds[0]:,.0f}, {bounds[1]:,.0f})")
print()

# -----------------------------------------------------------------
# 3. DP histogram for a categorical column.
# -----------------------------------------------------------------
print("=== DP histogram of 'department' ===")
hist = layer.dp_histogram("department")
for category, noisy_count in sorted(hist.items()):
    print(f"  {category:15s}: {noisy_count:.1f}")
print()

# -----------------------------------------------------------------
# 4. Batch queries -- ask several questions at once.
# -----------------------------------------------------------------
print("=== Batch query results ===")
results = layer.query_statistics(
    [
        {"type": "count", "column": "salary"},
        {"type": "mean", "column": "salary", "lower": 0, "upper": 100_000},
        {"type": "sum", "column": "salary", "lower": 0, "upper": 100_000},
        {"type": "variance", "column": "age", "lower": 18, "upper": 65},
        {"type": "histogram", "column": "gender"},
        {"type": "bounds", "column": "age"},
    ]
)
for key, value in results.items():
    print(f"  {key:25s}: {value}")

# -----------------------------------------------------------------
# 5. Demonstrate the effect of epsilon on noise.
# -----------------------------------------------------------------
print()
print("=== Effect of epsilon on noise (mean of 'salary') ===")
for eps in [0.1, 0.5, 1.0, 5.0, 10.0]:
    noisy_mean = layer.dp_mean("salary", lower=0, upper=100_000, epsilon=eps)
    print(f"  epsilon={eps:<5.1f}  ->  DP mean = {noisy_mean:>10,.2f}")
print("  (smaller epsilon = stronger privacy, more noise)")
