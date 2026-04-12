"""
Example 5 -- PII-Scrubbed Real Data Access
==========================================
When an agent genuinely needs access to *real* data (e.g. to validate data
quality), the privacy layer can provide a PII-scrubbed copy -- but **only
after explicit user approval**.

This is an *escalated* action: it requires two rounds of user confirmation
(one to justify the request, one after the PII scan results are shown).

Use case: an agent has exhausted synthetic data and DP statistics and needs
to inspect actual records -- e.g. to debug a data-quality issue -- but
personal information must still be stripped.
"""

import pandas as pd

from agent_privacy_layer import PIIScrubber, PrivacyLayer, UserConfirmation

# -----------------------------------------------------------------
# 1. Prepare a dataset with various PII types.
# -----------------------------------------------------------------
df = pd.DataFrame(
    {
        "name": ["Alice Smith", "Bob Jones", "Charlie Lee"],
        "email": ["alice@example.com", "bob@test.org", "charlie@mail.com"],
        "phone": ["555-123-4567", "555-987-6543", "555-456-7890"],
        "ssn": ["123-45-6789", "987-65-4321", "456-78-9012"],
        "age": [25, 30, 28],
        "salary": [50_000, 60_000, 55_000],
        "notes": [
            "Contact at alice@example.com or 555-123-4567",
            "Prefers email: bob@test.org",
            "IP address: 192.168.1.100",
        ],
    }
)

# -----------------------------------------------------------------
# 2. Detect PII without modifying data.
# -----------------------------------------------------------------
scrubber = PIIScrubber()
detected = scrubber.detect(df)
print("=== PII detected (no modification) ===")
for col, pii_types in detected.items():
    print(f"  {col:10s}: {pii_types}")
print()

# -----------------------------------------------------------------
# 3. Scrub with different actions.
# -----------------------------------------------------------------
for action in ("redact", "hash", "mask"):
    s = PIIScrubber(action=action)
    result = s.scrub(df)
    print(f"=== Scrubbed data (action='{action}') ===")
    print(result.data.to_string(index=False))
    print(result.report)
    print()

# -----------------------------------------------------------------
# 4. Access scrubbed data through the PrivacyLayer (escalated path).
#    In production, dry_run=False would prompt the user interactively.
# -----------------------------------------------------------------
layer = PrivacyLayer(
    df,
    confirmation=UserConfirmation(dry_run=True),  # auto-approve for demo
    pii_action="redact",
)

print("=== Scrubbed data via PrivacyLayer (escalated, user-approved) ===")
result = layer.get_scrubbed_data(
    reason="Need to validate data quality for the 'notes' column",
    columns=["age", "salary", "notes"],
    max_rows=3,
)
print(result.data.to_string(index=False))
print()
print(result.report)
print()

# -----------------------------------------------------------------
# 5. Demonstrate denial: auto_deny mode blocks the request.
# -----------------------------------------------------------------
layer_deny = PrivacyLayer(
    df,
    confirmation=UserConfirmation(auto_deny=True),
    pii_action="redact",
)

print("=== Denied request (auto_deny mode) ===")
try:
    layer_deny.get_scrubbed_data(reason="I want to see the data")
except PermissionError as e:
    print(f"PermissionError: {e}")
