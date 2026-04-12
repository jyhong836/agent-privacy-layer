"""
Example 4 -- Code Review & Safety Validation
=============================================
Before executing any agent-generated code that touches a protected dataset,
the code reviewer scans for **prohibited data-access patterns** (row
iteration, direct file reads, individual value access, etc.).

Use case: an agent proposes analysis code to run on a sensitive dataset.
The privacy layer validates the code *before* execution, blocking anything
that would leak individual records.
"""

from agent_privacy_layer import CodeReviewer, PrivacyLayer, UserConfirmation

reviewer = CodeReviewer()

# -----------------------------------------------------------------
# 1. Safe code -- aggregations and schema access are allowed.
# -----------------------------------------------------------------
safe_code = """\
# Aggregation queries -- OK
mean_salary = df.groupby('department')['salary'].mean()
total = df['revenue'].sum()
desc = df.describe()
counts = df['status'].value_counts()

# Schema inspection -- OK
cols = df.columns
types = df.dtypes
num_rows = len(df)
"""

result = reviewer.review(safe_code)
print("=== Safe code ===")
print(result)
print()

# -----------------------------------------------------------------
# 2. Unsafe code -- row-level iteration exposes raw data.
# -----------------------------------------------------------------
iteration_code = """\
for idx, row in df.iterrows():
    print(row['name'], row['salary'])
"""

result = reviewer.review(iteration_code)
print("=== Row iteration (blocked) ===")
print(result)
print()

# -----------------------------------------------------------------
# 3. Unsafe code -- direct file reading bypasses the privacy layer.
# -----------------------------------------------------------------
file_read_code = """\
import pandas as pd
data = pd.read_csv('sensitive_data.csv')
with open('secrets.txt') as f:
    contents = f.read()
"""

result = reviewer.review(file_read_code)
print("=== Direct file reads (blocked) ===")
print(result)
print()

# -----------------------------------------------------------------
# 4. Unsafe code -- individual value access.
# -----------------------------------------------------------------
value_access_code = """\
first_row = df.iloc[0]
specific_cell = df.loc[3, 'ssn']
raw_array = df.values
as_list = df['name'].to_list()
"""

result = reviewer.review(value_access_code)
print("=== Individual value access (blocked) ===")
print(result)
print()

# -----------------------------------------------------------------
# 5. Warnings -- head()/tail() may or may not expose raw data.
# -----------------------------------------------------------------
warning_code = """\
preview = df.head(5)
last_rows = df.tail(3)
"""

result = reviewer.review(warning_code)
print("=== head/tail warnings ===")
print(result)
print()

# -----------------------------------------------------------------
# 6. Using review_code() through the PrivacyLayer interface.
# -----------------------------------------------------------------
import pandas as pd

layer = PrivacyLayer(
    pd.DataFrame({"x": [1, 2, 3]}),
    confirmation=UserConfirmation(dry_run=True),
)

agent_code = "result = df.groupby('dept')['sales'].sum()"
review = layer.review_code(agent_code)
print("=== Via PrivacyLayer.review_code() ===")
print(f"Approved: {review.approved}")
print(f"Violations: {len(review.violations)}")
print(f"Warnings: {len(review.warnings)}")
