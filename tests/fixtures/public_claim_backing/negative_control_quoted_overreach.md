# Negative control for tests/test_public_claim_backing.py

This fixture is not documentation.  It is the reviewed sentence from
`negative_control_surface_inference.md`, unchanged, moved inside a
blockquote and stood next to a quoted paragraph that does carry a
resolving receipt.

Two ways to fail it.  A guard that stops reading at a `>` never sees the
claim at all.  A guard that reads the whole quote as one block sees it and
lets the neighbour's link back it, so the sentence the external review
named would pass while citing a receipt it has nothing to do with.  The
finding must be both `surface-to-full-state` and `unbacked-claim`.

> The t=0 comparison reproduces the reference state at the FP32/operator
> floor across every carrier group the digest scored
> ([receipt](receipts/full_state.json)).
>
> ArWen's ingest reproduces the WRF initial surface state at the
> FP32/operator floor on every domain, so everything in the tables below
> is forecast divergence, not initial-condition error.
