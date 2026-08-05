# Obs-battery integration wave — errata and recorded deviations, 2026-08-04

Deviations from the battery spec and rulings made during the 1.6 obs-battery
lanes, recorded once, here, at integration. Each entry names its evidence in
the tree; nothing below is a plan.

## 1. Stage-IV source: Iowa Environmental Mesonet archive, not RDA ds507.5

The spec (section 2.2) registered NCAR RDA **ds507.5** as the Stage-IV route.
It is unusable from this box: the TLS handshake fails with
`SEC_E_UNTRUSTED_ROOT`, and the dataset sits behind a registered-account wall
either way. NOMADS' NCEP `pcpanl` production directory serves the product
anonymously but retains only ~14 days, which cannot serve a hindcast.

**Adopted:** the Iowa Environmental Mesonet archive's `stage4` directory
(`https://mesonet.agron.iastate.edu/archive/data`), anonymous HTTPS, verified
against **all 7 case days**. The route is the default in
`tools/rustwx/crates/rw-obs/src/bin/stage4.rs` (`DEFAULT_ARCHIVE`), and the
`--archive URL` flag keeps RDA (or any mirror) reachable the day it answers.
Bytes are hash-pinned per object regardless of route, so the archive choice
is provenance, not trust.

Decode findings that only archived bytes could settle (B1, commit
`00461023`): every archived Stage-IV object measured is GRIB **edition 2**
(grid template 3.20, product template 4.8) despite the archive's `.grib`
naming, and the GRIB template 5.3 missing-value-management hazard is real but
not live — every measured message leaves those octets zero and carries
missing cells in a Section-6 bitmap; `decode` re-proves that per object.

## 2. MRMS `-99` ruling: no-echo is an observation (lead-endorsed)

Over the 24.5 M-cell CONUS composite, `-999` covers 38.4 % of cells and
`-99` covers 52.8 %, with real data starting at −31.5 dBZ (B1 decode
receipts, commit `00461023`; mechanics in
`tools/rustwx/crates/rw-obs/src/bin/mrms.rs`).

* `-999` is **no radar coverage** → masked, excluded by the validity mask.
* `-99` is **no echo** → an observation, valued at **−35 dBZ**
  (`DEFAULT_NO_ECHO_DBZ`), a floor below every registered threshold.

Masking `-99` would delete half the grid along with every correct negative a
fractions skill score rests on: on a case-shaped box the observed fraction
reads 1.0000 with no-echo kept versus 0.238 masked — and the spec fails any
case over 10 % masked, so the wrong ruling would have failed every case for
a reason that is not weather. Ruling endorsed by the lead.

## 3. Storage arithmetic correction: hourly boundaries, 25–37 GB/case

Spec section 2.4 planned 3-hourly LBC donor slots (~10–15 GB per case). The
HRRR route's forcing cadence is pinned **hourly**
(`gpuwm/hrrr_route_inputs.py` `FORCING_INTERVAL_SECONDS = 3600`), so a 24 h
case seals f00..f24 — 25 analyses — and the staged-HRRR figure for the WRF
arm is **~25–37 GB per case**. Better forcing, worse arithmetic; the spec's
section 8.3 table carries the old numbers. First recorded in
`B4-ROUTE-QUALIFICATION.md` section 2, caveat 2; restated here because it is
a spec erratum, not a route caveat.

## 4. B1 sizing table: the whole observation pull is small

Total observation fetch for the battery's case set: **8.6 GB, ~30 minutes**
(MRMS reflectivity from `noaa-mrms-pds`, Stage-IV from the Iowa archive,
ASOS from the IEM window with the frozen hash-pinned station table). The
obs side is not a storage or scheduling constraint; the WRF arm's staged
HRRR (entry 3) is the storage long pole.

## 5. FTZ claim-census red: a release-line finding, not a lane's

`tests/test_ftz_claim_consistency.py::test_census_covers_the_public_tree` is
RED on the release line `56768262` itself: the 1.5.2 release commit inserted
CHANGELOG entries, moving four registered claim lines
(174/176/177/1280 → 210/212/213/1316) without re-running the census. Not
introduced by the battery spec commit (`fc15d9ae` adds only a release-excluded,
claim-free spec file). Fixed in this wave by the census's own registration
mechanism (`python -m tools.ftz_receipt.claim_census`); all touched records
were unattributed with null tokens, so no curator attribution was lost and
the gate is unchanged.
