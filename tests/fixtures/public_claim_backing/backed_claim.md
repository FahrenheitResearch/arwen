# Positive control for tests/test_public_claim_backing.py

A t=0 claim whose breadth is exactly what the linked receipt measured.
Without this fixture the guard would be indistinguishable from one that
rejects every sentence mentioning t=0.

The t=0 comparison reproduces the reference state at the FP32/operator
floor across every carrier group the digest scored
([receipt](receipts/full_state.json)).
