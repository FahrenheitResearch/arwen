# Certification receipts

Machine receipts for the claims the public documentation makes. They ship
with the package (`gpuwm/data/**` is package data) and they are kept by the
release snapshot, because a claim whose evidence is excluded from the
release is a claim a reader cannot check.

A receipt is a measurement, not an advertisement. It is written by the tool
that took the measurement, committed exactly as measured, and linked from
the sentence that relies on it. When a receipt says FAIL, the sentence says
FAIL.

## `t0_state_parity_digest.json` / `.md`

Schema `gpuwm.t0-state-digest/v1`, written by
`tools/matched_wrfout_t0_state_digest.py` over a staged candidate/reference
wrfout pair. The JSON is the receipt; the Markdown is the same numbers as a
reviewer table.

What it carries:

- the pre-registration (`registration`) and its SHA-256, so the scoring
  contract can be shown not to have been edited to suit the numbers;
- the ceilings, imported by name from `gpuwm/verify/nest_gates.py` and
  pinned there long before this receipt existed, so the gate cannot be
  calibrated to its own data;
- an input inventory naming every file opened, its side, and its SHA-256;
- per-domain, per-carrier-group and per-variable ULP metrics from the one
  certified ULP definition (`gpuwm.verify.state_equiv.fp32_signed_ulp`);
- a boundary entry that is `scored` or `unavailable` -- the lateral-boundary
  files are probed, never assumed;
- `covered_groups`, which is the exact breadth any published sentence about
  this measurement is allowed to claim. `tests/test_public_claim_backing.py`
  enforces that.

The instance committed here scores the four-domain matched run of record at
its initial time against the 48-rank WRF v4.6.1 reference run of the same
case. Neither side retained a `wrfbdy_d01`, so the boundary group is
`unavailable` and names both sides; no boundary claim is made anywhere.

Re-derive it by pointing the tool at your own staged pair. The receipt is a
pure function of its inputs plus the evaluator commit -- no timestamps -- so
two runs over the same directories produce byte-identical files, which is
also how the writing machine is checked for silent corruption.
