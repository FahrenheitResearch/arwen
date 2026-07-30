# Driving stock CPU WRF from the same preprocessor — evidence dossier

Audited 2026-07-29 against the rw-wps authorities tree
(`codex/rwwps-gfs-ra0-20cr-authorities` @ `01ac4e78`, fully merged into
this line) and the retained campaign evidence store.  This page states
exactly what is proven, what is packaged-but-unexecuted, and where the
receipts live.

## The claim, framed correctly

RW-WPS does **not** produce `met_em*` for `real.exe` on its native
route.  It replaces the whole `geogrid`/`ungrib`/`metgrid`/`real.exe`
chain and emits `wrfinput_d0N` + `wrfbdy_d01` directly
([migrating-from-wps.md](migrating-from-wps.md);
`gpuwm/native_wrf_distribution.py` declares
`runtime_forbidden: ["WPS", "real.exe"]`).  The provable statement is:
**unchanged stock WRF v4.6.1 `wrf.exe` opened rw-wps-produced
`wrfinput`/`wrfbdy` bytes and integrated the model.**

## What is PROVEN (execution receipts, hash-bound)

Roughly eighteen independent receipts (schemas
`gpuwm-hrrr-stock-wrf-acceptance-v1`,
`gpuwm-native-direct-wrf-stock-oracle-v1`,
`rw-wps-stock-wrf-certification-receipt-v1`; all `status: PASS`,
`exit_status: 0`, the `Input data is acceptable to use:` gates for both
files, `wrf: SUCCESS COMPLETE WRF`, finite `wrfout` readback) spanning:

* **Four source families**: HRRR native GRIB2, GFS `pgrb2.0p25`, ERA5
  GRIB1, and declarative `rw-wps.mapping.v1` compositions.
* **Geometry range**: 192x160 to 1000x1000 mass points at 1/3/4/12 km;
  1- and 12-record boundary files at 3600 s and 10800 s cadence.
* **Execution modes**: serial and 12-/24-rank dmpar MPI; CUDA and
  deterministic Rust-CPU preprocessing; the offline-installed
  distribution as producer.
* **Nesting**: HRRR hierarchies through d06 (linear and branched
  topologies) and the mapped GFS contract through d04, with pre-run and
  post-run input hashes proven identical.
* **Chain of custody**: e.g. the one-command HRRR wrapper receipt pair
  where `consumed_input_sha256.wrfinput_d01 = 92dd0db5...` in the
  stock-WRF proof is byte-identical to the export manifest's
  `files.wrfinput_d01.sha256`; and the 12-rank ERA5 run whose
  `prelaunch.sha256` (`wrfinput_d01 = d7baa0e3...`) matches the
  producer's `proof.json` export hash exactly.

Receipt locations: the receipts are **not inside this repository**.
They are retained on the campaign evidence store beside the authorities
worktree (`...\2026-07-18\files-mentioned-by-the-user-you\outputs\`:
`node1-wrf-direct-f00f12-v2\`, `node1-live-evidence\`,
`co-orchestrator-handoff-20260720\node1-preprocessing\receipts\`,
`stock-wrf-real-gfs-*-gate-20260721\`,
`native-wrf-distribution-6725d3c-v1\`).  `PROVENANCE.md` and
[native-wrf-direct-export.md](native-wrf-direct-export.md) quote the
same hashes.  If this feature ships beyond this machine, the receipt
archives must ship with it.

## The honest boundaries (must accompany any claim)

1. **These are interoperability and stable-advance gates of 5-60 model
   seconds** — not forecast-skill runs and not WPS/METGRID numerical
   parity (masked surface/soil interpolation deliberately differs;
   `PROVENANCE.md`).  The longest proven integration is 60 s.
2. Acceptance requires three receipt-bound namelist deltas
   (`ra_lw_physics 0->1`, `use_theta_m 0->1`, stock-only `ghg_input=0`;
   [native-wrf-direct-support-matrix.md](native-wrf-direct-support-matrix.md)).
3. Certification is confined to the WSM6+YSU+MM5-91+Noah 49-level
   slice, plus one narrow Thompson/Morrison 10-second gate on the
   hash-bound current-GFS d01-d04 export.
4. **Claimed but never executed**: 20CRv3 members (self-declared
   "not_stock_wrf_gated"), named ERA5/GFS adapters nested through d06,
   `max_dom` 7-21 (cardinality gates only), and the entire Windows CPU
   route (real `wrfinput`/`wrfbdy` produced on Windows; no `wrf.exe`
   ever consumed them).  The formal release checkbox in
   `PUBLIC_RELEASE_ACCEPTANCE.md` remains unchecked pending a second
   certified vertical grid (z=35 passed once, self-caveated).
5. **No automated test in this tree executes WRF.**  The in-repo suites
   (`tests/test_native_wrf_contract.py`,
   `tests/test_native_wrf_distribution.py`,
   `tests/test_prepare_hrrr_wrf.py`,
   `tests/test_hrrr_stock_wrf_acceptance.py`) validate schemas,
   packaging integrity, and log parsers — the `SUCCESS COMPLETE WRF`
   strings inside them are fixture text.  Execution proof comes only
   from the operator tools (`tools/run_hrrr_stock_wrf_acceptance.py`,
   `tools/seal_wrf_direct_proof.py`) run on a node with a real WRF
   build.  There is currently no regression guard for WRF-compat
   breaks.
6. **Binary identity caveat**: several receipts label `wrf.exe`
   `f0fb585b...` "unchanged stock", but `PROVENANCE.md` attributes that
   SHA to the instrumented oracle rebuild (audited write-out-only
   patch).  Prefer citing the clean `v4.6.1` tag build `cfac9655...`
   (mapped GFS d01-d04, z-grid gates) and the clean dmpar build
   `328b2b0a...` (MPI ERA5/GFS/HRRR-d06 runs).

## What a release page may say

> The same preprocessor that feeds gpuwm also drives stock CPU WRF: it
> emits standard `wrfinput_d0N`/`wrfbdy_d01` directly from HRRR, GFS,
> ERA5, or a declarative mapped composition — no WPS, no `real.exe` —
> and unchanged WRF v4.6.1 has opened those exact bytes and stepped the
> model, serial and MPI, single domains and nests through d06.

...followed, in the same paragraph, by boundaries 1-4 above.  Anything
stronger is not receipt-backed today.
