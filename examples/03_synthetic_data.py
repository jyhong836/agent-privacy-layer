"""
Example 3 -- Synthetic Data Generation
=======================================
An AI agent can generate synthetic data that preserves the *schema* (column
names, dtypes, categorical domains) of a real dataset but contains **no real
values**.

Use case: an agent needs realistic-looking data to prototype analysis code,
build visualisations, or validate a pipeline -- without accessing actual
sensitive records.
"""

import pandas as pd

from agent_privacy_layer import PrivacyLayer, UserConfirmation

# -----------------------------------------------------------------
# 1. Prepare a sample customer dataset.
# -----------------------------------------------------------------
df = pd.DataFrame(
    {
        "name": ["Alice Smith", "Bob Jones", "Charlie Lee", "Diana Park", "Eve Brown"],
        "email": [
            "alice@example.com",
            "bob@test.org",
            "charlie@mail.com",
            "diana@example.com",
            "eve@test.org",
        ],
        "age": [25, 30, 28, 35, 22],
        "salary": [50_000.0, 60_000.0, 55_000.0, 70_000.0, 45_000.0],
        "department": ["Engineering", "Sales", "Engineering", "Sales", "Engineering"],
        "is_active": [True, True, False, True, False],
    }
)

# -----------------------------------------------------------------
# 2. Random strategy (default) -- fast, no extra dependencies.
# -----------------------------------------------------------------
layer = PrivacyLayer(
    df,
    confirmation=UserConfirmation(dry_run=True),
    synthesis_strategy="random",
)

print("=== Synthetic data (random strategy, same row count) ===")
synthetic_random = layer.synthesize_data()
print(synthetic_random.to_string(index=False))
print()

# Generate a different number of rows
print("=== Synthetic data (random strategy, 10 rows) ===")
synthetic_random_10 = layer.synthesize_data(n_rows=10)
print(synthetic_random_10.to_string(index=False))
print()

# -----------------------------------------------------------------
# 3. Faker strategy -- contextually realistic fake values based on
#    column name heuristics (e.g. "email" -> fake emails).
# -----------------------------------------------------------------
layer_faker = PrivacyLayer(
    df,
    confirmation=UserConfirmation(dry_run=True),
    synthesis_strategy="faker",
)

print("=== Synthetic data (faker strategy, 5 rows) ===")
synthetic_faker = layer_faker.synthesize_data(n_rows=5)
print(synthetic_faker.to_string(index=False))
print()

# -----------------------------------------------------------------
# 4. Override strategy at call time.
# -----------------------------------------------------------------
print("=== Override strategy at call time ===")
# Even though the layer defaults to 'random', request faker here:
synthetic_override = layer.synthesize_data(n_rows=3, strategy="faker")
print(synthetic_override.to_string(index=False))
print()

# -----------------------------------------------------------------
# 5. Show that dtypes and column names match the original.
# -----------------------------------------------------------------
print("=== Schema comparison ===")
print(f"  Original columns:  {list(df.columns)}")
print(f"  Synthetic columns: {list(synthetic_random.columns)}")
print()
print("  Original dtypes:")
for col, dtype in df.dtypes.items():
    print(f"    {col:15s}: {dtype}")
print("  Synthetic dtypes:")
for col, dtype in synthetic_random.dtypes.items():
    print(f"    {col:15s}: {dtype}")
