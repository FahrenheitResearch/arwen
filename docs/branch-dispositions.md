# Branch dispositions

Why this file exists: the 2026-08-31 triage of 198 unmerged local
branches found finished work stranded on them, including confirmed live
shipped defects whose fixes sat complete on branches nobody was going to
merge (the continuous-nowcast daemon crasher fixed at 64b728a60 is the
canonical example, and lane/engine-per-time-decode set the pattern for
the whole class).  A branch that is not merged into
integration/release-2.5.0 is either carrying work the release line
needs, or it is accounted for.  This ledger is where it gets accounted
for, and `tests/test_branch_dispositions.py` fails the suite when an
unmerged branch has no row here.

**A new unmerged branch needs a row in this table before the suite goes
green again.**  One row per branch, three columns, exactly one
disposition from this vocabulary:

- `merged-content-elsewhere` -- the work landed on the release line
  under different commits; the branch's own tip adds nothing.
- `superseded-by-<sha>` -- a strictly better landing replaced it; the
  sha names the replacement (at least 7 hex characters).
- `spent-probe` -- a measurement, experiment, or rescue that served its
  purpose; the numbers were read, nothing is meant to land.
- `parked-<owner>` -- real unlanded work deliberately held; the owner
  suffix names who decides its future.
- `active-lane` -- work in progress; it will merge or be re-dispositioned
  when its lane concludes.

Ground rules:

- `lane/engine-261-salvage` and `lane/engine-261` are active lanes (the
  2.6.1 salvage and integration lines); so are the other
  `lane/engine-261-*` fold branches.
- Bulk rows citing the 2026-08-31 triage are sweep-grade: the triage
  judged roughly 150 of the 198 branches clean as a population, and
  those rows inherit that judgement rather than a fresh per-branch
  verdict.  Anyone re-opening one of those branches should re-verify
  before trusting the row.
- Stale rows (a row whose branch has since been deleted) are allowed
  and are not pruned by the gate: deleting a folded branch is exactly
  the cleanup this ledger exists to encourage, and the row remains as
  the record of where its content went.

## Named deferrals and withdrawn numbers (2.6.1 assembly)

Not branches, but the same accounting: work a fold deliberately did NOT
land, named here so deferring it was a decision with an owner rather
than a second stranding.

- **RTE+RRTMGP P3 preparation profile** -- deferred, unshipped feature.
  The mp=50 x RTE+RRTMGP COMPOSITION is accepted (the cloud-optics
  coupling landed with the salvage fold; the composition walk gained
  exactly the nine mp50 lw4.sw4 rows), but no packaged preparation
  profile for the pairing ships in 2.6.1: the profile-count gate is
  bound to P3_LEGACY_RRTMG_PROFILE_ID and registering the second
  profile is its own feature with its own front door and demo.
- **Concurrent-member driver wiring (--member-workers)** -- the
  machinery landed with the DA fold (measured 1.40x, bit-identical);
  the driver flag that would turn it on in the ensemble runner is NOT
  wired, because wiring it changes results silently where members
  share state.  Port it deliberately, with the bit-identity gate run
  on the wired path.
- **Level-8 200k-sample sizing-integral over-read** -- pre-existing,
  documented in the mesh benchmark's section 5, deliberately unchanged
  by the fine-mesh fold: changing that gate needs its own ruling, not
  a rider.
- **Withdrawn gate-margin numbers** -- the mesh-quality margins quoted
  before the blind-meter fix are withdrawn; the re-measured values are
  11.8849 %/cell against a band of 6.1722 (2.9% margin).  Do not quote
  the old numbers from the superseded records.
- **Open-files guard family** -- with per-valid-time decode holding one
  handle per forcing time, a series past roughly 41 forcing times can
  hit the process file-descriptor ceiling (Errno 24).  Documented as
  scales-with-series by the per-time lane; the guard family is named
  future work, not a 2.6.1 change.
- **The double-decompress kill** -- named 2.6.2 work: with the
  bounded-parallel inventory landed, a compressed field-per-file
  source still decompresses every record twice (once for identity,
  once for decode), the remaining roughly-a-quarter wall premium
  over the pre-fix run.  Killing it means carrying staged bytes
  from identity to decode under the same byte budget, with the
  dual-run identity proof re-run on both source shapes.
- **Whole-decode paths** -- non-GRIB2 formats, inspect, and compose
  donors still decode whole-series; the per-time discipline covers the
  mapped GRIB2 primary path.  Named by the per-time lane;
  extending the discipline is follow-up work, per path, with the same
  dual-run identity proof each time.


## The private sidecar

Thirty-nine rows of this ledger name branches of the internal
subsystem whose identifying vocabulary the containment scans ban from
every tracked line of this branch (tests/test_excluded_subsystem_
absent.py, the senior gate: the public snapshot and the whole branch
diff stay zero-hit).  Those rows are NOT dropped -- they live, in
full, in the untracked private sidecar `docs/branch-dispositions-
private.md` on the development machine, and
tests/test_branch_dispositions.py reads the sidecar beside this file
when it exists.  On any clone without those local branches the
sidecar is unnecessary and its absence changes nothing; on the
development machine, deleting it puts the gate RED by exactly those
branches, which is the accounting working.

| branch | disposition | note |
|---|---|---|
| feature/arwen-global-level5 | active-lane | global-model physics line, in progress |
| lane/cycle-children | active-lane | confirmed daemon-crasher fix (64b728a60) queued into the salvage fold |
| lane/engine-261 | active-lane | the 2.6.1 integration lane itself |
| lane/engine-261-da | active-lane | 2.6.1 fold lane (DA cluster) |
| lane/engine-261-finemesh | active-lane | 2.6.1 fold lane (fine-mesh unlock) |
| lane/engine-261-pertime | active-lane | 2.6.1 fold lane (per-time decode) |
| lane/engine-261-invperf | active-lane | folded at 2a0fc77c8 (bounded-parallel inventory under a staged-byte budget); branch deletable |
| lane/engine-261-salvage | active-lane | the salvage lane folding the triage's confirmed defect carriers; this ledger and its gate live here |
| lane/engine-261-spectral | active-lane | 2.6.1 fold lane (spectral subsystem) |
| lane/engine-per-time-decode | active-lane | mid-fold for 2.6.1; the stranding that set the pattern for this whole ledger |
| lane/harness-falsegreen-11 | active-lane | confirmed cross-checkout import refusal (2 of 4 commits), queued into the salvage fold |
| lane/meshgen-degeneracy | active-lane | one-line trace-precision fix, queued into the salvage fold |
| lane/p3-tables-state-audit | active-lane | confirmed table-version refusal dead code, queued into the salvage fold |
| lane/prove-fine-mesh | active-lane | the only categorical-supersample implementation; folding via lane/engine-261-finemesh |
| lane/spectral-level2 | active-lane | complete Level-2 spectral operator subsystem; folding via lane/engine-261-spectral |
| lane/tiles-small-card-236 | active-lane | small-card sizing-gate fixes, queued into the salvage fold |
| lane/verify-real-card-3080 | active-lane | exFAT output_root defect record, queued into the salvage fold |
| p3/front-door-20260829 | active-lane | four measured engine fixes plus the campaign gate file, queued into the salvage fold |
| worktree-wf_c142aa28-ef6-31 | active-lane | confirmed DA/mp=50 findings, queued into the salvage fold |
| worktree-wf_c142aa28-ef6-32 | active-lane | confirmed DA/mp=50 reflectivity findings, queued into the salvage fold |
| worktree-wf_fad3e143-c03-3 | active-lane | confirmed restart_interval_s contradiction, queued into the salvage fold |
| bench/cpas-hk200m | parked-drew | reproduced mesh-generator G4 misalignment; its gate-margin numbers are superseded, do not quote them |
| integration/da-mpas-arwen-fuse | parked-drew | the coupled DA cycling spine exists nowhere else; its MPAS-pin half is obsolete |
| lane/all-radar-scale | parked-drew | windowed all-radar ingest plus a live control-innovation bugfix |
| lane/da-jacobi-eigensolver | parked-drew | re-measurement reversing a shipped doc's conclusion |
| lane/da-obs-path-10 | parked-drew | per-radial Nyquist dealiasing; the line still dealiases against one scalar |
| lane/da-structure-metrics | parked-drew | nowcast scoring against operational baselines plus structure metrics |
| lane/ens-size-sweep | parked-drew | the resolution half of the skill decomposition |
| lane/p3-cuda-verify | parked-drew | the campaign that proved the 12-fixture suite passes with five processes off, plus the fix and the 13th fixture |
| lane/wah-level2-requeue-verona | parked-drew | the six-arm A/B verdict exists only here |
| lane/wah-overlap-handover | parked-drew | DA verification cluster member |
| tmp/ens-par-lf | parked-drew | concurrent member advance, measured 1.40x, bit-identical |
| lane/agent-enablement | parked-salvage | porting workbench on a 4-week stale base; expensive to revive |
| lane/node1-untouched-sources | parked-salvage | 8 measured configs for four sources the shipped wave never touched |
| lane/nssl-perf | parked-salvage | the only 7-scheme GPU microphysics step-cost census |
| lane/obs-b6-cases | parked-salvage | six authored case TOMLs salvageable from retired paperwork |
| worktree-wf_c142aa28-ef6-11 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-12 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-13 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-14 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-16 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-25 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-40 | parked-salvage | confirmed but half-stale; fold only after rewriting against the landed cal_cldfra1 fix |
| worktree-wf_c142aa28-ef6-41 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-42 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-43 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_c142aa28-ef6-44 | parked-salvage | fail-closed tripwire from the ef6 gate cluster, worth folding |
| worktree-wf_dac365b4-068-5 | parked-salvage | 4 AI-source 10-GiB-class configs; header prose needs correcting before landing |
| lane/hex-fixed-overhead | superseded-by-17cf943ef | YSU KMAX fix beaten by the global-workspace fix on both lines; landing it would break live tests |
| worktree-wf_4c724cf8-7ca-3 | superseded-by-17cf943ef | same YSU KMAX fix, same verdict |
| c2/prefix-ulp | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| campaign/real74-verification-lineage | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| codex/arwen-native | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| docs/dcomp-dqf-ruling | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| experimental/sase-v1 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| feature/phase5 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| feature/user-zero-gate | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| fix/cloud-radiation-seams | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| fix/slab-vram-probe-ambient-base | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| fix/ysu-nest-first-step | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| instr-td-101 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| integration/da-mpas-cycle | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| integration/engine-258-clean | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| integration/flagship-pass-20260720 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| integration/sase-dual-mp-acceptance-20260721 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| integration/unified-20260720 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/16gb-da-frontier | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/260-level5 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/aerosol-ingest | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/arwen-mcp | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/back-half | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/cli-refusals-233-wip | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/continental-cycling | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/cup-gf-phase1 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/cycle-anchor | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/cycle-frontdoor | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/cycle-spine | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/da-ensemble-parallel | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/da-external-baseline | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/da-hrrr-framing | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/docs-parity-233 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/engine-hexp3 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/era5-ysu-user-crash | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/gate-triage | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/gf-seam-parity-231 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/goes-bridge | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/j2k-vendor | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/letkf-default-flip | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/level5-owner | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/lloyd-goldberg-graded | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/mpas-cuda-closed-loop | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/mpas-landing | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/mpas-portnative-diff | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/p3-537-fortran-oracle | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/parallel-battery | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/pdt-selector | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/receipt-aerosol | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/release-258-readiness | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/riders-1.8.1 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/ruc-any-source | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/ruc-column-nzs | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/ruc-geometry | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/ruc-geometry-lf | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/sase-grayzone-spec | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/self-nesting | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/six-level-oracle | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/source-descriptors | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/spectral-level3 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/static-dataset-door | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/surgery-floor-probe | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/tierb-real-parent | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/wah-level2-exploit | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| lane/wif-climatology | merged-content-elsewhere | WIF climatology cluster re-landed sanitized; shipped in 2.5.8/2.6.0 |
| lane/wif-default | merged-content-elsewhere | WIF climatology cluster re-landed sanitized; shipped in 2.5.8/2.6.0 |
| lane/wif-door | merged-content-elsewhere | WIF climatology cluster re-landed sanitized; shipped in 2.5.8/2.6.0 |
| lane/wif-rust | merged-content-elsewhere | WIF climatology cluster re-landed sanitized; shipped in 2.5.8/2.6.0 |
| master | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| p5alias | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| p5perf-host | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| p5perf-nest | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| p5perf-pbl | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| p5vram | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| p6a832 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| p6egui | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| p6launcher | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| paired-cupy14-hrrr-copy | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| probe/205-prefix-crash | spent-probe | probe/scratch namespace; spent per the 2026-08-31 triage |
| probe/ysu-pnw-repro | spent-probe | probe/scratch namespace; spent per the 2026-08-31 triage |
| product/v1.2-integration | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| rescue/stash-from-wf-6d7-2 | spent-probe | probe/scratch namespace; spent per the 2026-08-31 triage |
| rescue/stash-from-wf-6d7-3 | spent-probe | probe/scratch namespace; spent per the 2026-08-31 triage |
| scratch/linux-fixes-reprove-20260817 | spent-probe | probe/scratch namespace; spent per the 2026-08-31 triage |
| scratch/measure-1280 | spent-probe | probe/scratch namespace; spent per the 2026-08-31 triage |
| spec/radar-da-demo | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| verify/dealias-perf | spent-probe | probe/scratch namespace; spent per the 2026-08-31 triage |
| worktree-wf_3840d2d8-518-11 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-10 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-15 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-17 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-18 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-19 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-20 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-21 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-22 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-23 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-24 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-26 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-27 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-28 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-29 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-30 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-33 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-34 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-35 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-36 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-37 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-4 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-5 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-6 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-7 | merged-content-elsewhere | its coverage claim is already double-gated on the release line |
| worktree-wf_c142aa28-ef6-8 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_c142aa28-ef6-9 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_d95e83bb-c5c-1 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
| worktree-wf_dac365b4-068-2 | merged-content-elsewhere | clean per the 2026-08-31 triage (landed under different shas, superseded, or spent); sweep-grade, not an individual verdict |
