# 9. Limits and roadmap

## 9.1 Standing limits, consolidated

Each of these appears in its chapter; they are gathered here so no reader has to
hunt for them.

- **An authored mapping prepares but does not yet run.** The forecast stage
  certifies only the packaged mapping authorities by digest; a caller-supplied
  mapping is refused at the door with the limit named (section 5.4)
  [tests/test_stage_seams.py].
- **One-command orchestration covers GFS only.** Every other source runs stage by
  stage (section 5.8) [gpuwm/go_cli.py:115].
- **Certification is not uniform across sources.** `hrrr`, `gfs`, `era5` are
  certified; the other 13 runnable model rows, plus the generic `mapped` row,
  are runnable-not-certified (section 5.1).
- **Only GRIB2 preparations are sliced per valid time.** The host-memory scale
  bound that stood here -- the 3 km CONUS 7-valid-time prep peaking near 107
  GiB of host RSS, passing on 123 GiB nodes and exhausting smaller boxes -- is
  RETIRED: that prep now peaks at 26.9 GiB, rising 0.04 GiB per forcing hour
  (section 5.7). What remains is narrower. Every declared format other than
  GRIB2 is decoded whole and then carved, so a NetCDF or GRIB1 series still
  holds one whole decode; a composition's donors and terrain supplements stay
  resident for the whole compose; and `inspect` decodes the whole series.
- **A compressed field-per-file source pays wall clock for that slicing.** The
  per-valid-time road inventories every object before decoding any, so a source
  that ships one bz2-compressed message per file decompresses twice and the
  inventory half is serial: measured on ICON-EU, a 6 h window (876 objects)
  takes 99.8 s against 27.6 s whole-decode, of which 37.5 s is the inventory
  pass alone. An uncompressed multi-message source shows no such cost (RAP, 6
  valid times: 13.34 s against 13.33 s).
- **No vertical nesting; explicit eta levels only; nz above 128 admitted but
  unrun** (section 2.2). Sub-km children run their parents' level count, which
  makes the nested 250 m capability coarse LES at the gray-zone edge (effective
  dz 96.7 m measured in-PBL on the 49-level tree, section 2.8).
- **Two-way feedback is experimental and unmeasured**; it feeds back dynamic
  state only and is stamped in provenance (section 2.4).
- **A nest relocation invalidates restart claims and prepared caches** (section
  2.5.3).
- **Effective resolution is measured on one grid** (6.7 dx at 500 m, forecast
  hour 2, gray-zone configuration); the 3 km and 2 km readings are retracted,
  the 6 km case produced no number, and the spectral tail is over-damped
  everywhere measured (section 7.3).
- **Below about 200 km, changing the initialization gives a different forecast,
  not a shifted one**, by the project's own spectral campaign (section 7.2).
- **The t=0 full-state comparison against WRF is a FAIL on all four domains** of
  the reference case; decay tables mix initial-state difference with forecast
  divergence (section 7.1).
- **23 of 40 physics component options are implemented-unverified**; several
  schemes (MYJ, WDM6, P3, Milbrandt-Yau, RRTM 1/1) have no oracle comparison
  against the WRF Fortran at all, and mp=28's 22 end-to-end column fixtures leave
  four missing the flat gate field by field with one clearing only under a named
  allowance (chapter 3).
- **SASE's physics is unvalidated** (2 of 7 acceptance bars on one case), its
  subgrid TKE magnitudes are not to be believed, and it has not completed a
  certified forecast on operational data (section 3.8).
- **The DA/nowcast surface is demo-grade and unscored**, by its own banner; no
  2.5.0-line DA skill measurement exists (section 7.5).
- **The perturbation ensemble engine is experimental** and off any certified
  path; "ensembles ship" means members-as-source-data and ensemble products,
  not the perturbation engine (sections 5.6, 6.3, 9.2).
- **Dual-run byte comparison is not ECC** and its non-detections are enumerated
  (section 4.4).
- **No tile-streaming full-physics capacity multiplier exists**; dry numbers may
  not be projected (section 6.5).
- **The throughput and capacity reference is 1.5.0-era** (section 8.1).
- **The published v2.4.1 carries two defects fixed on this line** (the
  moving-nest accumulator wipe and the sub-km radiation floor); 2.4.1 users are
  exposed to both (sections 2.5.2, 3.9).

## 9.2 Roadmap

Each item below is future work. None of it ships in 2.5.0, and nothing here is a
commitment to a date.

**Regional spectral numerics (Level-2) and a global spectral core (Level-3).**
Research lanes exist for a spectral numerics hook in the regional model and for
a global spectral core, with proof documents and evidence bundles outside the
release line; both were excluded from the 2.5.0 release candidate by
instruction, and their pin-owner items remain open
[receipt:RELEASE-CANDIDATE-2P5-2026-08-18.md]. They are research work: their
numbers have not been audited into this manual and no capability claim is made
for them here.

**An MPAS-class core against the same physics.** The physics orchestration is
already exposed as a persistent column-batch seam
(`gpuwm.core.physics.run_mpas_column_batch`): C-contiguous CuPy float32 arrays
in `[level, column]` layout, exact integer-step cadence bookkeeping, the full
WSM6 species set, native surface-classification override with a receipt naming
which source decided, and restart identity that refuses across a changed sea-ice
threshold [docs/mpas-seam.md]. The direction is a second dynamical core driving
the same validated physics through that seam rather than a fork of the physics.

**A proper coarse-grid effective-resolution measurement.** The 3 km measurement
that failed verification defines its own successor: a larger domain (more modes
per band), 12-24 h of spin-up, and the own-slope knee criterion formalized in
place of the fixed -5/3 reference. Queued as post-cut science
[receipt:ARWEN-EFFECTIVE-RESOLUTION-2026-08-18.md].

**Oracle campaigns for the unmeasured schemes.** MYJ, WDM6, P3, Milbrandt-Yau,
and the RRTM 1/1 pair are declared next-stage oracle subjects in the physics
page; closing them moves real forecast options up the maturity ladder (chapter
3).

**A certificate for caller-authored mappings.** The narrower second certificate
that would let an authored mapping run through `gpuwm sim`, closing the largest
gap in the arbitrary-source story (section 5.4).

**Two-way feedback evidence.** The experimental `feedback = 1` path needs a
gate, a receipt, or a measurement before it can be more than stamped (section
2.4).

**Spectral v2 pin-owner items.** The calibration zero-variance guard and the
coherence/correlation gate dedup found during instrument validation (section
7.2).

**The perturbation ensemble engine.** Currently a self-limiting experimental
tool (no mass/wind balance, shared lateral boundaries, lateral taper only); the
path to shipping it is closing those stated limits, not relabeling them
(section 6.3).
