# Receipts: gray zone — the Shin-Hong (`bl_pbl_physics = 11`) acceptance ladder

The receipts behind the gray-zone statements about Shin-Hong in
[PHYSICS.md](../../PHYSICS.md), committed beside the page that cites them.
The reference side is Honnert, Masson and Couvreux (2011) eq. (9)-(14),
transcribed in `gpuwm/verify/gray_zone.py` as a REFERENCE and nothing else:
no model path reads that module, it participates in no configuration
identity, and it is kept out of the closure's own authority module so that
scoring a closure against it can never become fitting the closure to it.
The bands below were instantiated and committed **before any run with
`--pbl 11` existed anywhere**.

## Instrument identity

The scored quantity is the mixed-layer (`0.05 <= z/h <= 0.85`) mean subgrid
TKE fraction over twelve 300 s snapshots in the final hour of a 4 h dry
convective boundary layer, 64x64 columns, nz=40 to 2 km, on a
3200/1600/800/400/200/100 m grid-spacing ladder. The instrument is
`gpuwm.verify.cases.cbl_dry.partition_run` driven by
`tools/grayzone_phase0.py`, both in this repository, at commit `82c8155b`;
the scoring path is byte-identical to the one the frozen baseline was
measured with (`gpuwm/verify/gray_zone.py` diffs zero against it). The tree
under test is `61488333`. The card is one RTX 5090 — no ECC, so the
same-seed determinism pairs are the corruption screen, not a formality.

`h` is each run's own S3-6f bulk-Richardson depth and the abscissa is
`x = dx/h` at that depth, so every rung is scored at its own `x` rather
than at the baseline's.

## What each file is

- `PHASE1-SHINHONG-20260803.md` — the ladder report: the six-rung scored
  table with each rung's `x`, `h`, seed-mean, seed sigma, the published
  eq. (9) target, the envelope, the band and the verdict; the 36 scored
  cells one seed at a time; the five registered criteria answered line by
  line; the determinism digests; the two non-scored control runs that show
  the merge did not move the baseline; and the measured cost.
- `shinhong_runs.jsonl` — the 42 run receipts the report reduces: 36 scored
  (6 rungs x 6 seeds, `repeat 0`) and 6 determinism rerolls (seed 20260801
  a second time at each rung, `repeat 1`). Registered budget spent exactly;
  no run crashed, no cell is missing, nothing was re-run. Each record
  carries its own scored scalar, `h`, `x`, the target and envelope
  evaluated at that `x`, the twelve per-snapshot samples, the wall time,
  the tree it ran on, and the profile-stack digest. `python
  tools/grayzone_phase0.py score --ledger shinhong_runs.jsonl` reduces it,
  and everything the report measures comes back out of this file alone:
  the six seed-means, the six `sigma_own`, the six `h` and `x`, all six
  determinism digests, and the cost section's 3099.1 s total with its
  per-rung 50.4 / 98.2 / 209.4 / 428.7 / 796.5 / 1515.9 s. The one input
  that is not in here is the frozen baseline sigma, which enters only
  through `sigma_used = max(frozen, own)` — and the report prints the
  result of that step in its `sigma_used` column at every rung, so the
  bands are checkable regardless.
- `npz/cbl_dry_partition_<dx>m_seed<seed>_shinhong.npz` — 36 per-run
  profile stacks, one per scored cell: `subgrid_fraction`, `e_resolved`
  and `e_subgrid` as (12 snapshots x 40 levels), the snapshot times `t`
  and depths `h` as (12,), and the height column `z` (40,). These are the
  arrays behind every scored scalar, so a reader can recompute a rung's
  mixed-layer window rather than take the reduction on trust.

## How to read these

1. **The bands were registered before the runs, and the ladder is scored
   against them as written.** `band = envelope +/- 2*sigma_used/sqrt(6)`,
   `sigma_used = max(frozen baseline sigma at that rung, this leg's own)`.
   No band was widened, re-centred or re-cut after the data existed.
2. **This is one idealized case.** A dry convective boundary layer, six
   seeds, one card, one day. It measures how the scheme's partition of
   turbulence between resolved and subgrid motion moves with grid spacing,
   against a published similarity curve. **It is not skill against
   observations, and no free-forecast obs-skill claim is made for this
   scheme anywhere in this repository.**
3. **The 3200 m rung is advisory and was never gated**, by the registered
   rule and not by hindsight: it sits on 2x2-block-degenerate ground and at
   an `x` the frozen baseline never occupied. It is recorded because it was
   run, not because it counts.
4. **Conformance and this are different axes.** The ladder says nothing
   about agreement with WRF; that is the oracle parity in
   `tests/test_shinhong_wrf461_parity.py`, reported separately. Passing
   this ladder moved no maturity rung: the option is
   `implemented-unverified` in the registry before and after, because no
   matched ArWen-versus-WRF forecast trajectory has been run with it.
5. **The determinism digests cannot be recomputed from the `npz` files.**
   The digest is SHA-256 over each snapshot's own height column as well as
   its three partition arrays, and the `npz` retains the height column
   once. Reproducing a digest means re-running the cell, which is what the
   report's later matches of the 3200 m digest are — five in total for
   that one cell, on a card with no ECC.

## Adaptations (release hygiene)

Same pattern as the neighbouring receipt directories. **37 of the 38
archive files ship byte-identical** — `shinhong_runs.jsonl` and all 36
`npz` stacks, verified by sha256 against the development copies. The one
transformed file is `PHASE1-SHINHONG-20260803.md`, and it differs only in
prose:

- one citation of a document that does not ship is reduced from its in-tree
  path to its bare filename, keeping the name and the commit so it stays
  checkable in the development history, and a sentence is added saying that
  it and the one other non-shipping document the report cites — already
  cited by bare filename — are covered in the omissions list below;
- one word is corrected in criterion 1: the flatness yardstick reads
  `sigma_seed`, not `sigma_used`. Both numbers beside it are the report's
  own and are untouched — `0.004941` is 2 x the largest seed sigma
  (`0.0024704`, at 400 m) and `157x` is `0.775164 / 0.004941` — and the
  registered F1 criterion is itself written in `sigma_seed`, so the
  correction makes the label agree with the arithmetic and with the
  criterion. Read as `sigma_used` the threshold would be `0.006016` and
  the ratio `129x`; the verdict is the same either way and neither number
  appears in the report;
- one sentence of the non-regression section, which referred an internal
  process question to the project owner, is dropped, and a dated addendum
  is appended recording how those four items were actually resolved, with
  the commit for each. The measured statement it sat beside — that the
  registered "full CPU suite passes unchanged" bullet was false as
  measured — is left standing exactly as written; the addendum updates it
  rather than editing it.

Every measured number, digest, band and verdict in that file is
byte-untouched. The introducing commit records the exact mapping, and the
original survives in the development history at `b159f2a0`.

## Deliberately not included

- **The registered expectation,
  `2026-08-03-shinhong-grayzone-expectation.md` (commit `31275170`)** — a
  development-campaign document, excluded from the release like the rest of
  that tree. It is cited by name in the ladder report and its load-bearing
  content is restated there: the five registered criteria are answered one
  at a time, in order, with the measurement each one is answered by; the
  bands it froze are the report's `band (I1)` column; and the claim that
  matters — written and committed before any `--pbl 11` run existed — is
  checkable at that commit in the development history. It also republished
  the frozen sigmas, which reach the bands only as described in the next
  bullet.
- **The frozen Phase-0 baseline, `PHASE0-BASELINE-20260803.md` (commit
  `67ac01b1`)** — likewise internal, and on a separate campaign branch. Its
  load-bearing content survives in the ladder report, and it is worth being
  exact about how much. Its per-rung seed-means are quoted verbatim in the
  report's `HEAD mean` column, for contrast only — they are not a band
  input and no verdict depends on them. Its frozen sigmas reach the bands
  only through `max(frozen, own)`, and the report prints that result at
  every rung: at 3200, 1600 and 100 m the frozen value is the larger one,
  so those three appear verbatim in the `sigma_used` column; 800 m's is
  quoted again in the report's controls table; and at 400 and 200 m the
  leg's own sigma is larger, so the frozen value cannot move the band and
  is not needed to check one.
- **The Phase-0 baseline's own ledger and profile stacks** — the 48 runs
  behind that `HEAD mean` column. They score a different closure
  (`bl_pbl_physics = 900`), no public claim about Shin-Hong rests on them,
  and shipping them would double this directory to back a contrast column.
- **The two non-scored `--pbl 900` control runs' ledger** — the merge check
  in the report's "Controls" section. Both numbers, both abscissae, the
  frozen sigmas they are compared against and the agreement in sigmas are
  stated in the report itself.
- **Wall-clock and node state** — no node time was used; the whole campaign
  ran on the local card, and the per-rung run times that mattered are in
  the report's cost section.
