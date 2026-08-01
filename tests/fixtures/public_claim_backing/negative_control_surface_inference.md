# Negative control for tests/test_public_claim_backing.py

This fixture is not documentation.  It holds the published
`docs/public/VERIFICATION.md` sentence that the external review named as
the surface-to-full-state inference, verbatim and unlinked, so the guard
has something it MUST reject.  If the guard ever accepts this file, the
guard has stopped working and every pass it reports is worthless.

| domain | T2 MAE / corr | PSFC MAE | wind10 corr |
|---|---|---|---|
| d04 | 0.000 K / 1.000 | 0.1 Pa | 1.000 |

ArWen's ingest reproduces the WRF initial surface state at the
FP32/operator floor on every domain, so everything in the tables below
is forecast divergence, not initial-condition error.
