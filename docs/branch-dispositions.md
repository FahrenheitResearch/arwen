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
| lane/da | active-lane | level5 programme (separate agent): global-core audit lane |
| lane/engine-261 | active-lane | the 2.6.1 integration lane itself |
| lane/engine-261-da | active-lane | 2.6.1 fold lane (DA cluster) |
| lane/engine-261-finemesh | active-lane | 2.6.1 fold lane (fine-mesh unlock) |
| lane/engine-261-pertime | active-lane | 2.6.1 fold lane (per-time decode) |
| lane/engine-261-invperf | active-lane | folded at 2a0fc77c8 (bounded-parallel inventory under a staged-byte budget); branch deletable |
| lane/engine-261-salvage | active-lane | the salvage lane folding the triage's confirmed defect carriers; this ledger and its gate live here |
| lane/engine-261-spectral | active-lane | 2.6.1 fold lane (spectral subsystem) |
| lane/engine-per-time-decode | active-lane | mid-fold for 2.6.1; the stranding that set the pattern for this whole ledger |
| lane/engine-riders-262 | active-lane | the 2.6.2 rider lane: release-line defects the 2.6.1 merge surfaced |
| lane/harness-falsegreen-11 | active-lane | confirmed cross-checkout import refusal (2 of 4 commits), queued into the salvage fold |
| lane/insitu | active-lane | level5 programme (separate agent): global-core audit lane |
| lane/instruments | active-lane | level5 programme (separate agent): global-core audit lane |
| lane/transform | active-lane | level5 programme (separate agent), created 2026-09-01 mid-session; not this line's to merge |
| lane/meshgen-degeneracy | active-lane | one-line trace-precision fix, queued into the salvage fold |
| lane/p3-tables-state-audit | active-lane | confirmed table-version refusal dead code, queued into the salvage fold |
| lane/prove-fine-mesh | active-lane | the only categorical-supersample implementation; folding via lane/engine-261-finemesh |
| lane/semi-implicit | active-lane | level5 programme (separate agent): global-core lane, created 2026-09-01; not this line's to merge |
| lane/spectral-level2 | active-lane | complete Level-2 spectral operator subsystem; folding via lane/engine-261-spectral |
| lane/statics | active-lane | level5 programme (separate agent): global-core lane, created 2026-09-01; not this line's to merge |
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

## Isolated 2.7 development branches (2026-09-05)

These rows account for the current 2.7 work in the isolated clone, evaluated
against `integration/2.7.0` at `1fc1fbebf69ecd4d51aaf89a2d75f9368842e893`. The gate still compares local
branches with `integration/release-2.5.0`; that comparison is unchanged.
Here, `merged-content-elsewhere` means that the content already landed on
the 2.7 integration line, by ancestry or Git patch equivalence. It does not
claim that this work landed on the old 2.5 line or passed release acceptance.

Each completed row has a per-branch Git basis. Branches with unmatched
patches remain `active-lane` for an explicit remaining-content audit;
this records uncertainty instead of declaring them merged or spent.
The main and global composition lanes remain active. Existing rows retain
their earlier scope and classifications.

| branch | disposition | note |
|---|---|---|
| feature/2.7-launchpad | merged-content-elsewhere | tip e36d8ac6c; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) f6d035393 |
| feature/2.7-launchpad-console | merged-content-elsewhere | tip 83d9bbcd0; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 526035f2b, 5455a9483 |
| feature/2.7-tui | merged-content-elsewhere | tip 5c2906941; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) daf0b7834 |
| feature/starter-template | active-lane | 2.7 remaining-content audit at tip af82154d6; 1 patch-equivalent commit(s), 1 unmatched (af82154d6); do not treat rewritten or WIP content as integrated |
| feature/streamed-move-bounded | merged-content-elsewhere | tip 295d6ec0a; all 12 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) e1bdd7741, 3e9e02cef, 30bd07fd3, d8421b8d9, 3de087a03, d17f1d08b, 0a30d836d, 6857bf2c3, 14367aff2, b18b9f994, 14ee6453a, 993609818 |
| feature/streamed-reconstruction | active-lane | 2.7 remaining-content audit at tip ea058bffa; 2 patch-equivalent commit(s), 1 unmatched (ea058bffa); do not treat rewritten or WIP content as integrated |
| fix/2.7-adaptive-stream | merged-content-elsewhere | tip 0d2a79ed4; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 6d9c9b999 |
| fix/2.7-analyzed-boundaries | merged-content-elsewhere | tip 1b0c191d0; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 3afc4fe9a |
| fix/2.7-auto-admission | merged-content-elsewhere | tip 07f4fd09e; all 5 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 836e93f7f, 73fc2406f, b2c58712e, 12aa8115c, 8e795ff90 |
| fix/2.7-auto-search-host-floor | merged-content-elsewhere | tip dcfd13c27; all 4 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 836e93f7f, 73fc2406f, bd8f708a9, 33bb3ff10 |
| fix/2.7-both-streamed-nesting | merged-content-elsewhere | tip e1a803cbf; all 4 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 6d9c9b999, b51cea2a6, 3458a5764, 90aee6fe7 |
| fix/2.7-ci | merged-content-elsewhere | tip ce7620bf6 is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| fix/2.7-classic-trace-gas | merged-content-elsewhere | tip bcfd1239c is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| fix/2.7-correctness | merged-content-elsewhere | tip 55723d168; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 8156dd58f, 22f09e837 |
| fix/2.7-decoders | merged-content-elsewhere | tip f30b87245 is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| fix/2.7-explicit-polygon-footprint | active-lane | 2.7 remaining-content audit at tip acabd6c5c; 4 patch-equivalent commit(s), 4 unmatched (13bd8b226, afda95f08, 6ef2d1a24, be03a815d); do not treat rewritten or WIP content as integrated |
| fix/2.7-fieldwise-packing | merged-content-elsewhere | tip d237f3811; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) f2c609b0f, ee15eed6c |
| fix/2.7-flat-streaming | merged-content-elsewhere | tip e717863eb; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) d7c16b0d6 |
| fix/2.7-generic-mapped | merged-content-elsewhere | tip 2e50d6a4d; all 5 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 6c64eac8c, 4f4c4ee81, 24f747a64, f99b144b7, 223f1f7a6 |
| fix/2.7-go-case | merged-content-elsewhere | tip de8e4f603; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) a44740ea5 |
| fix/2.7-gpu-marker-fixtures | active-lane | 2.7 remaining-content audit at tip e50eaeddc; 3 patch-equivalent commit(s), 1 unmatched (86b9c93b5); do not treat rewritten or WIP content as integrated |
| fix/2.7-installed-tui | merged-content-elsewhere | tip f00deb342; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 6dbd581bf, 6fd5a2581 |
| fix/2.7-lambert-oracle | merged-content-elsewhere | tip e44f66dbd; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 11f84b9bb |
| fix/2.7-launch | merged-content-elsewhere | tip d5b471440; all 3 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) be11fc00a, 09baae706, 2b3b6b78c |
| fix/2.7-lbc-frame | merged-content-elsewhere | tip 3b829eb20; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 79633b286 |
| fix/2.7-legacy-sw | merged-content-elsewhere | tip d8ef9d086; all 3 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 353723e91, 4d9f0c1f2, f7c2aadea |
| fix/2.7-lifecycle-store-restart | merged-content-elsewhere | tip 00c6501ed; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 3f6cc8eff, 92b1726ef |
| fix/2.7-mapped-provenance | merged-content-elsewhere | tip 9d41e1ec1; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) e39b36e3f |
| fix/2.7-metem | merged-content-elsewhere | tip 7ae6ba554; all 4 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) a2f08f5a2, 1d5befd66, 353723e91, 4d9f0c1f2 |
| fix/2.7-moving-streamed-transport | merged-content-elsewhere | tip 1e5fca7cb; all 3 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) a9fe8de9f, 21611ea8a, 1228656b0 |
| fix/2.7-namelist-receipt | merged-content-elsewhere | tip aecca0f8c; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) b6b00096e |
| fix/2.7-native-companions | merged-content-elsewhere | tip 60fef4b56; all 8 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 4637257ff, ccea613cd, 4687a2289, 267900003, 43ca9a01f, ae266d234, 90919f2d0, 81ff33c36 |
| fix/2.7-nested-stage-reuse | merged-content-elsewhere | tip 3a785a342; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) df43e97c4 |
| fix/2.7-oracle-arithmetic | merged-content-elsewhere | tip 169991939; all 3 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 9a8b0da23, b10e1ec7e, 89d449a53 |
| fix/2.7-physics | merged-content-elsewhere | tip 82b63da84 is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| fix/2.7-physics-capabilities | active-lane | 2.7 remaining-content audit at tip aab9621c8; 15 patch-equivalent commit(s), 8 unmatched (d61a639a9, c64d67af5, d2b6333f3, 741d9b11a, 5f6c0a868, 18d33d6bf, 43e57f1c2, 6d396a109); do not treat rewritten or WIP content as integrated; 3 branch-only merge(s) also need audit |
| fix/2.7-pmsl | merged-content-elsewhere | tip 90aee6fe7 is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| fix/2.7-pmsl-supplement | merged-content-elsewhere | tip c48ccd253; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) da19e4e2d |
| fix/2.7-prep-output-review | merged-content-elsewhere | tip 07d7c3aa4; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 8089d4ea0 |
| fix/2.7-prep-profile | merged-content-elsewhere | tip 14e55d8a6; all 6 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 5bbdb8df5, d8142be2f, a21a0c517, f2b086c34, d2418ef44, 5fa37454c |
| fix/2.7-prepared-followers | merged-content-elsewhere | tip 27e05bdd0; all 8 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) a9fe8de9f, 21611ea8a, 1228656b0, 1ad91a8ea, 8ce86b513, b82100a7e, 4336c7e23, ce07ece15 |
| fix/2.7-prepared-relocation-history | merged-content-elsewhere | tip 46e36b557; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 41fb6008c |
| fix/2.7-public-prepared-resume | merged-content-elsewhere | tip a7937c70c; all 3 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 25492886d, fab090b5f, ec07bf186 |
| fix/2.7-public-sim-restart | merged-content-elsewhere | tip 859c40320; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 1ad91a8ea |
| fix/2.7-qnn-physics-ingest | merged-content-elsewhere | tip 13f3d787a; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 4febd041f, 712176510 |
| fix/2.7-readiness | merged-content-elsewhere | tip 212523619 is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| fix/2.7-rrtm-memory | active-lane | 2.7 remaining-content audit at tip db2e74650; 1 patch-equivalent commit(s), 6 unmatched (d61a639a9, c64d67af5, d2b6333f3, 741d9b11a, 5f6c0a868, 18d33d6bf); do not treat rewritten or WIP content as integrated; 3 branch-only merge(s) also need audit |
| fix/2.7-runtime-store | merged-content-elsewhere | tip 9fb159281; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 8b5f3e584, 810b71117 |
| fix/2.7-sase-cadence | merged-content-elsewhere | tip 91f0c41be; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) d4ec65c0a |
| fix/2.7-single-adaptive | merged-content-elsewhere | tip 2759ec78f; all 4 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 3458a5764, 3f6cc8eff, e51922fb1, e8730465c |
| fix/2.7-single-prepared-clock | merged-content-elsewhere | tip 941621424; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 102031c50, 38943db08 |
| fix/2.7-single-prepared-restart | merged-content-elsewhere | tip d934d767d; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) f0a7021ef, 585745527 |
| fix/2.7-snapshot-lifetime | merged-content-elsewhere | tip b6f9b6334; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 08bb604f7 |
| fix/2.7-static-companions | merged-content-elsewhere | tip 7e24270da; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) b813613e7, 66969807d |
| fix/2.7-static-platform-parity | merged-content-elsewhere | tip 9589784ad; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 2be5ccc43 |
| fix/2.7-stream-arithmetic | merged-content-elsewhere | tip 6b7836113; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 3a37558fa |
| fix/2.7-streamed-child-coupling | merged-content-elsewhere | tip cf2e20565; all 4 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 3afc4fe9a, e1bdd7741, d17f1d08b, 0a30d836d |
| fix/2.7-streamed-child-move | merged-content-elsewhere | tip 5ed2f9425; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 007cb8694 |
| fix/2.7-surface-pressure | merged-content-elsewhere | tip 599318fd1; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) e44be07b0 |
| fix/2.7-water-companions | merged-content-elsewhere | tip e12439cce; all 4 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) cf110056d, e44be07b0, 30339dd59, 2beb431ba |
| fix/2.7-wizard | merged-content-elsewhere | tip dd5ceb0bf is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| fix/2.7-wizard-contract-tests | active-lane | 2.7 remaining-content audit at tip 414be2f65; 0 patch-equivalent commit(s), 2 unmatched (f09b0e85d, 414be2f65); do not treat rewritten or WIP content as integrated |
| fix/2.7-wizard-default-clock | merged-content-elsewhere | tip d2c2489c1; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) b3b8cb0cb |
| fix/2.7-wrf-boundaries | merged-content-elsewhere | tip 3eb49b2fd is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| fix/2.7-wrf-eta | merged-content-elsewhere | tip ca24fb93b; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 130b0bdcc, 3f4ad0ad6 |
| fix/2.7-wrf-native | merged-content-elsewhere | tip da2e1dd8f is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| integrate/global-2.7-snapshot | active-lane | global composition lane at 84f3bda6f; owner integration and qualification remain active |
| integration/2.7.0 | active-lane | main 2.7 integration and acceptance lane; audited snapshot 1fc1fbebf; ongoing work |
| lane/2.7-cli-first-use | merged-content-elsewhere | tip 70726ff7f; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) b59b69727, 7153e1179 |
| lane/2.7-trace-gas | active-lane | 2.7 remaining-content audit at tip 86b9c93b5; 2 patch-equivalent commit(s), 1 unmatched (86b9c93b5); do not treat rewritten or WIP content as integrated |
| perf/2.7-atmospheric-window | merged-content-elsewhere | tip 9cfb43936; all 3 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 36dcc5bed, 8fb59c860, 94dff9610 |
| perf/2.7-frame-field-streaming | merged-content-elsewhere | tip e3a6c2798; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 2689ff3da |
| perf/2.7-preparation-reuse-controls | merged-content-elsewhere | tip 16d557e08; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) a460eca7c |
| perf/2.7-prepared-delayed-activation | merged-content-elsewhere | tip 28f727ca7; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 14367aff2, b18b9f994 |
| perf/mapped-owned-field | merged-content-elsewhere | tip 645a930da; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 14de28143, 9277f409b |
| proof/2.7-metem | merged-content-elsewhere | tip 7a7812dcd; all 2 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 353723e91, 4d9f0c1f2 |
| test/2.7-pmsl-linux-install | merged-content-elsewhere | tip 8e795ff90 is an ancestor of 2.7 snapshot 1fc1fbebf; no branch-only commits remain |
| test/2.7-runplan-contracts | merged-content-elsewhere | tip bfbbc0568; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) e445110e8 |
| test/2.7-streaming-memory-pin | merged-content-elsewhere | tip 2508e576b; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) 016948251 |
| ui/console-3d | active-lane | 2.7 remaining-content audit at tip 0d157a3cc; 0 patch-equivalent commit(s), 2 unmatched (bed5d33b9, 0d157a3cc); do not treat rewritten or WIP content as integrated |
| ui/tui-hybrid | merged-content-elsewhere | tip bb11dac4e; all 1 branch-only non-merge commit(s) are Git patch-equivalent on 2.7; landing(s) ad8ed7065 |
