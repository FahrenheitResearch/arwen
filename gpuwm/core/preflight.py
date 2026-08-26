"""Memory preflight: itemized estimator, scratch registry, N0 ``--alloc``.

Phase-5 Task 11 (panel lane L4; folds robust-5), implementing architecture
section E of docs/superpowers/specs/2026-07-16-phase5-nesting-architecture.md:
the all-resident four-domain decision is gated by an ENFORCED memory
estimate -- ``gpuwm check --alloc`` constructs every persistent device
allocation for the experiment, runs zero steps, and reports the measured
pool/device numbers against the estimate and the measured WDDM budget.
Measured > estimate is a FAILING GATE (milestone N0, ledger records
``alloc_fits_wddm_budget`` / ``alloc_measured_le_estimate`` /
``alloc_estimate_le_wddm_budget`` -- F11 amendment: every leg blocks).

Three-tier model.  Tier 1 is exact arithmetic; tiers 2/3 are PROVISIONAL
POLICY calibrated on two controller measurements (the d01 run fixture
``.superpowers/sdd/codex/n0-preflight-baseline.log`` and the N0
allocation probe ``n0-alloc-probe-r2.json``) -- the tier-2/3 constants
parameterize the reserve proposal for controller ratification; the tests
pin their ALGEBRA (calibration consistency), they do not and cannot
validate the model against independent evidence:

* TIER 1 -- LIVE ARRAYS.  Exact itemized persistent residency from shape
  formulas: (a) ``DomainState`` arrays (transcribed from
  ``gpuwm/core/state.py``); (b) ``PhysicsDriver`` persistents per scheme
  (``gpuwm/core/physics.py``: the surface fields dict, held tendency
  stacks, conditionally retained ``last_ysu`` output, scratch-aliased
  microphysics diagnostics, KF W0AVG + LUT d01-only, RRTMGP lat/lon +
  ozone); (c)
  every named scratch slot from the static registry below; (d) LBC
  residents -- d01 eager interval tables, children's rolling
  one-interval ``nest_*`` tables + F16's arena-aliased full-parent field
  + SINT geometry (the F4 NEST ALLOCATION MANIFEST); (e) the RRTMGP shared
  chunk workspace from
  the phase-maximum chunk formula, plus the per-domain radiation column
  packing and physics-prep transients that coexist with it inside a
  step.  d01 fixture cross-check: itemized residency 1.4544 GiB vs the
  measured full-physics pool-used peak 1.47 GiB (ratio 0.989; the
  residual ~17 MB is per-call KF/coupling transient tails owned by the
  15% headroom).
* TIER 2 -- POOL RETENTION (provisional).  CuPy's pool retains freed
  transient blocks it cannot re-bin.  The N0 probe measured alloc-time
  retention NIL (held - used = 16 MB); the d01 RUN fixture showed
  5.52 GiB held vs 1.47 used, i.e. retention is a run-churn phenomenon.
  ``pool_retention_residual_bytes()`` (fixture held minus the d01
  alloc-estimate basis) is the run-time reserve term; it belongs to the
  N5/N6 run gates, not the N0 allocation gate (split proposed below).
* TIER 3 -- DEVICE-SIDE FOOTPRINT.  ``cudaMemGetInfo`` sees the CUDA
  context, JIT modules, and non-pool allocations on top of the pool.
  The N0 probe measured a fresh allocation-only process at
  ``PROBE_DEVICE_OVERHEAD_BYTES`` = 1.39 GiB; the run fixture's
  apparent 5.72 GiB gap is thereby attributed to 12 h of other-process/
  WDDM drift and RETIRED from the model (kept only as
  ``CAL_FIXTURE_OVERHEAD_BYTES`` for the record).  The probe number is
  a lower bound on run-time overhead (zero steps JIT-compile almost
  nothing), so the run-gate reserve keeps margin above it.

RESERVE POLICY (plan Task 11, amended; controller ratifies at N0,
``--reserve-gib`` overrides): the budget is ALWAYS the MEASURED free
VRAM at startup minus the configured reserve -- never nominal 32 GiB.
Two proposals, split by gate (PENDING RATIFICATION):

* ``ReservePolicy.n0_alloc()`` -- the N0 allocation-gate reserve:
  probe-measured non-pool overhead + external margin (alloc-time
  retention measured nil).  The ``--alloc`` default.
* ``ReservePolicy.run_time()`` -- the N5/N6 run-gate reserve: adds the
  fixture-calibrated run-churn retention residual on top.

Robust-5 fold-ins: a headroom check runs before state construction and
before each high-water phase; persistent scratch is prewarmed at setup;
the RRTMGP ``column_chunk`` is the FIRST over-budget lever
(:func:`recommend_column_chunk`); OOM means record diagnostics and
terminate -- never ``free_all_blocks()``-and-continue.

STAGED-RESIDENCY CONTINGENCY (documented sketch ONLY -- no machinery, per
architecture section E fallback lever (3)): if the enforced chain cannot
fit even after the chunk lever and a transient-scratch shared arena, a
``StagedPlan`` would demote d04 (the dominant block) to staged residency:
its DomainState lives host-side pinned, uploaded per d03-step window,
with the coupler writing boundary tables into a device staging pool.
That is a DESIGN REOPEN materially changing lanes L6/L7 and is flagged as
the top risk; nothing here implements it.

CPU/GPU split: every estimator path is CPU-only (no cupy import); only
:func:`run_alloc_preflight` (the ``--alloc`` mode, controller-run) touches
the device.  The ``gpuwm check`` CLI wiring ships as
:func:`register_cli` per the F2 ownership map -- the one-line ``cli.py``
hookup is a controller handoff commit at merge.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from gpuwm.config import (CUMULUS_ADVECTIVE_FORCING_SCHEMES,
                          DEFAULT_COLUMN_CHUNK, MYJ_PBL_SCHEME,
                          MYJ_SFCLAY_SCHEME, SASE_PBL_SCHEME, RunConfig,
                          radiation_enabled, radiation_scheme_ids,
                          soil_layer_count)
from gpuwm.core import kernel_frame_recordings as _kernel_frame_recordings
from gpuwm.experiment import DomainConfig, ExperimentConfig

# ---------------------------------------------------------------------------
# Calibration fixture (.superpowers/sdd/codex/n0-preflight-baseline.log,
# controller-measured 2026-07-16 -- the plan's [PRE-FLIGHT] values).
# ---------------------------------------------------------------------------

GIB = 1024 ** 3

#: WDDM baseline at fixture time: free / total, and the implied non-gpuwm
#: residency (30.27 of 31.84 GiB free -> 1.57 GiB other processes).  The
#: BUDGET is never taken from these numbers -- it is re-measured at every
#: ``--alloc`` startup; the fixture values calibrate tests and projections.
CAL_WDDM_FREE_BYTES = int(30.27 * GIB)
CAL_WDDM_TOTAL_BYTES = int(31.84 * GIB)

#: d01 full-physics fixture: CuPy pool USED peak (step-boundary sampling ->
#: persistent residency), pool HELD (used + retained churn blocks), and the
#: cudaMemGetInfo device-side footprint (held + non-pool overhead).
CAL_D01_POOL_USED_PEAK_BYTES = int(1.47 * GIB)
CAL_D01_POOL_HELD_BYTES = int(5.52 * GIB)
CAL_D01_DEVICE_FOOTPRINT_BYTES = int(11.24 * GIB)

#: Plan-mandated allocator-headroom factor over the itemized subtotal.
ALLOCATOR_HEADROOM = 1.15

#: The default-chunk RRTMGP workspace AS THE d01 FIXTURE ABOVE RAN IT.
#: Pinned, not recomputed.  :func:`pool_retention_residual_bytes` subtracts
#: an enumerated basis from that fixture's measured held bytes, and both
#: sides have to describe the same run: recomputing the workspace term from
#: today's layout means any change that shrinks the workspace shrinks the
#: basis, inflates the "unexplained" residual by exactly what it saved, and
#: hands the run gate a LARGER reserve as its reward for using less memory.
#: The tightening of the RTE phase layouts (978.18 -> 738.50 MiB at this
#: chunk) is the change that surfaced it.  Re-measure the fixture and this
#: constant moves with it; until then it records what was measured.
CAL_D01_WORKSPACE_BYTES = 1025700000

#: Fixture memGetInfo gap = footprint - pool held = 5.72 GiB.  RETIRED as
#: an overhead model by the N0 probe (a fresh allocation-only process
#: measures 1.39 GiB): the difference is attributed to other-process/WDDM
#: drift across the fixture's 12 h run window.  Kept for the record only.
CAL_FIXTURE_OVERHEAD_BYTES = (CAL_D01_DEVICE_FOOTPRINT_BYTES
                              - CAL_D01_POOL_HELD_BYTES)

#: Tier-2 calibration: pool retention beyond live arrays = held - used =
#: 5.52 - 1.47 = 4.05 GiB on the d01 RUN fixture (run churn; the N0
#: allocation probe measured alloc-time retention nil).  The estimate's
#: workspace + transient terms and the 15% headroom already cover most of
#: that churn; the run-gate reserve carries the calibrated residual (see
#: :func:`pool_retention_residual_bytes`, computed against the d01
#: fixture's own alloc-estimate basis so nothing is double-counted).
CAL_D01_POOL_RETENTION_BYTES = (CAL_D01_POOL_HELD_BYTES
                                - CAL_D01_POOL_USED_PEAK_BYTES)

#: Controller N0 allocation probe (.superpowers/sdd/codex/
#: n0-alloc-probe-r2.json, 2026-07-16, ``--alloc --reserve-gib 2`` on the
#: fresh box): the full four-domain manifest-driven allocation completed;
#: pool retention at allocation time is nil and the non-pool device
#: overhead of a fresh process is 1.39 GiB.  These are the tier-2/3
#: RE-CALIBRATION measurements the fixture could not provide.
PROBE_POOL_USED_PEAK_BYTES = 26_581_917_184
PROBE_POOL_HELD_BYTES = 26_598_071_296
PROBE_DEVICE_FOOTPRINT_BYTES = 28_088_020_992
PROBE_FREE_BYTES = 32_499_564_544
#: Fresh-process non-pool overhead = footprint - held = 1,489,949,696 B.
#: LOWER BOUND on run-time overhead: zero steps JIT-compile almost none
#: of the kernel modules; the run-gate reserve keeps margin above it.
PROBE_DEVICE_OVERHEAD_BYTES = (PROBE_DEVICE_FOOTPRINT_BYTES
                               - PROBE_POOL_HELD_BYTES)

#: Reserve margin for non-gpuwm residency GROWTH during a 12 h run (the
#: baseline 1.57 GiB other-process residency is already outside "free").
EXTERNAL_MARGIN_BYTES = GIB // 2

#: Default ERA5 forcing cadence for the d01 eager LBC table formula; the
#: real value is owned by CaseDataConfig (Task 2/3) and passed through.
DEFAULT_FORCING_INTERVAL_SECONDS = 21600.0

#: N0 ledger record names (gpuwm/verify/nest_gates.py, F11 amendment).
N0_GATE_METRICS = ("alloc_fits_wddm_budget", "alloc_measured_le_estimate",
                   "alloc_estimate_le_wddm_budget")


def gate_display_name(metric: str, *, vram_gib: float | None = None) -> str:
    """A gate's name as THIS platform should print it.

    The strings in :data:`N0_GATE_METRICS` are pre-registered N0 ledger
    record names, read by ``gpuwm/verify/nest_gates.py`` and written into
    certification receipts, so they are identifiers and must not change
    with the host.  What a reader sees may: "WDDM" is a Windows display
    driver model, and printing ``alloc_fits_wddm_budget`` in the first
    ``gpuwm check`` a Linux user ever runs describes their machine with a
    word that does not apply to it (A-6).  The prose beside these lines
    has been platform-correct since ``envelope_platform`` was introduced;
    only the gate names were left behind.

    Display only.  The key stays the key everywhere it is recorded.
    """
    if envelope_platform(vram_gib=vram_gib) == "linux":
        return metric.replace("_wddm_", "_vram_")
    return metric

#: RETIRED multiplicative machine-peak factors, kept as the historical
#: record and for the standalone child fit (`gpuwm downscale`), which
#: still reads them as a deliberately conservative bound.
#:
#: WINDOWS / WDDM -- 1.75.  The Thompson 12-18Z matched rerun (2026-07-28,
#: the only multi-domain Windows run with whole-run machine-wide VRAM
#: sampling; receipts in docs/thompson-rematch-20260728.md and the campaign
#: handoff) measured a true machine peak of 29,004 MiB = 30,412,898,304 B
#: against this estimator's footprint projection of 17,416,429,288 B -- a
#: ratio of 1.7462.  The projection misses CuPy pool retention across the
#: run and output/checkpoint write transients, so under WDDM it
#: under-projects the number the budget actually confronts.  The single
#: observation is rounded UP to 1.75 (an envelope must not under-round).
#:
#: LINUX -- 1.45 over the ITEMIZED ALLOC ESTIMATE, measured-preliminary.
#:
#: Three independent first-run pilots (2026-07-30) instrumented the
#: machine-wide peak across whole forecasts with nvidia-smi sampling:
#:
#:   ======  =====  ==========  ===========  ==========  ==========
#:   node    card   alloc est.  footprint    machine pk  pk / alloc
#:   ======  =====  ==========  ===========  ==========  ==========
#:   node 1  4090   7.20 GiB    11.31 GiB    9.54 GiB    1.32
#:   node 2  4090   7.29 GiB    11.39 GiB    8.99 GiB    1.23
#:   node 3  4070   3.51 GiB     4.90 GiB    4.04 GiB    1.15
#:   ======  =====  ==========  ===========  ==========  ==========
#:
#: Two things follow, and the second is the important one.
#:
#: 1. The peak lands at 0.79-0.82x the FOOTPRINT PROJECTION, not 1.75x.
#: 2. The footprint projection itself is wrong on Linux, because the two
#:    grid-independent constants it adds to the alloc estimate --
#:    ``pool_retention_residual_bytes`` (2.73 GiB) and
#:    ``PROBE_DEVICE_OVERHEAD_BYTES`` (1.39 GiB) -- are Windows-pool
#:    artifacts that did not appear in any of the three measurements.
#:    At node 3's refused 60x48 minimum layout they were 4.12 GiB of a
#:    5.38 GiB projection: 77% of the floor was constants, which is why
#:    shrinking the grid could not help and a 12 GiB card could not be
#:    sized at all while its GPU sat 66% idle.
#:
#: So on Linux the projection IS the itemized alloc estimate (see
#: :func:`platform_projection_constants`), and this factor is the
#: envelope over THAT.  1.45 clears the worst of the three observations
#: (1.32) by 10%.  Three runs on two card models are still not a
#: calibration: re-derive rather than tune.  Receipts:
#: ARWEN-NODE1-4090-PILOT-20260730.md,
#: ARWEN-NODE2-4090-CUDA128-WORLDWIDE-20260730.md,
#: ARWEN-NODE3-4070-12GB-FLOOR-20260730.md.
#:
#: Both are presented as OBSERVED envelopes, not models, and neither
#: changes any gate: the enforced numbers remain the itemized estimate and
#: the measured --alloc legs.
#: 2026-08-01 AMENDMENT -- the multiplier has no intercept, so its ERROR
#: CHANGES SIGN.  A 16 GiB fleet node (RTX 4080, Linux, driver 595.58.03,
#: machine-wide nvidia-smi sampled at 250 ms, GPU otherwise idle) measured
#: whole forecasts across a 6.6x span of grid sizes and found the x1.45
#: envelope OPTIMISTIC below ~3.5 GiB of itemized estimate and pessimistic
#: by 25-30% above it.  A 224x180 first-run domain was declared 3.99 GiB
#: and peaked at 4.38.  The two fleet datapoints that looked contradictory
#: -- a 5090 under-predicted by ~19%, this 4080 over-predicted by 25-30%
#: -- are one model with a missing intercept read at two grid sizes.
#:
#: The intercept is not a fitted nuisance: it is the NON-POOL residency
#: this module already itemizes (:func:`non_pool_device_bytes` -- the CUDA
#: context plus the launch-time local-memory backing store), which the
#: multiplicative form charged in proportion to the grid when it scales
#: with neither.  See :data:`ENVELOPE_UNMODELLED_BYTES` for the affine
#: replacement and the residuals behind it.
#:
#: These factors are RETIRED from every gate: the 2026-08-19 RTX 3080
#: calibration (below) replaced the WDDM multiplier with a measured
#: affine term, and the Linux lane had already moved to the affine form.
#: The dict is kept as the historical record and for `gpuwm downscale`'s
#: deliberately conservative standalone child fit.
PEAK_ENVELOPE_FACTORS = {"windows": 1.75, "linux": 1.45}

#: How much evidence each RETIRED factor rested on.
PEAK_ENVELOPE_BASIS = {
    "windows": "measured, 1 WDDM run (retired multiplier)",
    "linux": "measured-preliminary, 3 runs (retired multiplier)",
}

#: Retained name for the WDDM factor (the original single-platform value).
OBSERVED_PEAK_OVER_FOOTPRINT = PEAK_ENVELOPE_FACTORS["windows"]

#: WDDM pool slack: what a Windows machine-wide peak carries beyond the
#: affine terms, as a FRACTION of the itemized estimate.
#:
#: MEASURED 2026-08-19, RTX 3080 10 GiB / Windows 11 WDDM / driver-level
#: desktop resident on the same card, machine-wide ``nvidia-smi`` at
#: 0.25 s beside the runtime's own GpuPeakMemoryWatcher receipts, across
#: six whole bare-default ``gpuwm go`` forecasts (single domain, 12 km,
#: 60x48 to 240x192, 6 h GFS, rte-rrtmgp and legacy-RRTMG suites).
#: ``machine-wide peak - desktop baseline`` against
#: ``estimate + itemized non-pool (live profile)``:
#:
#:   =========  ========  =========  =========  =========
#:   run        estimate  non-pool   measured   residual
#:   =========  ========  =========  =========  =========
#:   g60x48     1.26 GiB  1.22 GiB   2.28 GiB   -0.20 GiB
#:   g110x88    1.54 GiB  1.22 GiB   2.60 GiB   -0.16 GiB
#:   g170x136   2.10 GiB  1.22 GiB   3.25 GiB   -0.07 GiB
#:   g240x192   3.05 GiB  1.22 GiB   4.11 GiB   -0.16 GiB
#:   t60x48     2.94 GiB  1.42 GiB   4.62 GiB   +0.27 GiB
#:   t110x88    3.16 GiB  1.42 GiB   5.53 GiB   +0.95 GiB
#:   =========  ========  =========  =========  =========
#:
#: The rte-rrtmgp lane's pool tracks the itemization at 0.95-1.00x and
#: its residuals are NEGATIVE: the affine form alone bounds it.  The
#: only positive residuals are the legacy-RRTMG lane's, where the CuPy
#: pool held up to 1.47x the itemized estimate (call-peak retention the
#: itemization does not model), and they grow with the estimate:
#: +0.09x at t60x48, +0.30x at t110x88 -- worst 0.30x, of which the
#: 0.5 GiB unmodelled constant absorbed 0.16x.  0.20x of the estimate on
#: top of that constant covers the worst measured point with 0.33 GiB to
#: spare and is charged on Windows ONLY: the Linux lane keeps its own
#: measured form untouched (re-measure there before exporting this term
#: -- the retention mechanism is a CuPy pool behaviour, not WDDM's, but
#: no Linux legacy-RRTMG run has been instrumented).
#:
#: What this REPLACES on Windows: the 1.75 multiplier over a footprint
#: projection carrying 4.12 GiB of 5090-derived pool constants.  On the
#: 3080 walk that model predicted 9.91 GiB for a run that measured
#: 2.6 GiB of own contribution (3.8x), refused a fitting card, and its
#: printed remedy refused at every grid size because 78% of the floored
#: envelope was grid-independent.  Receipts:
#: docs/public/receipts/wddm/rtx3080-wddm-calibration-20260819.json
#: (every run's measured and priced terms), beside the walk capture in
#: Downloads/ux-walks-replay/gpu-walk-3080.md.
#: 2026-08-20 AMENDMENT (task 206) -- THIS TERM IS NOT WDDM'S, AND IT IS
#: NOT EVERY SUITE'S.  It is the LEGACY-RRTMG call-peak retention, and it
#: splits on the radiation lane on both driver models and all four cards
#: anyone has instrumented.  Pool HELD over the itemized estimate:
#:
#:   ==============  ==========  =================  =================
#:   card            driver      rte-rrtmgp         legacy-RRTMG
#:   ==============  ==========  =================  =================
#:   RTX 3080        WDDM        0.939-0.988        up to 1.47
#:   RTX 5090        Linux       0.879-0.921        1.134, 1.166
#:   RTX 5070 Ti     Linux       0.910-0.956        1.172-1.186
#:   RTX 4080        Linux       0.94-1.00          (not instrumented)
#:   ==============  ==========  =================  =================
#:
#: Three campaigns, two driver models, and the same boundary each time:
#: the legacy engines' LW/SW call-peak workspace is retained by the pool
#: between calls and the itemization does not model it, while the
#: rte-rrtmgp lane's pool tracks the itemization to within a few percent.
#: Charging every suite for a mechanism only one of them has cost an
#: rte-rrtmgp configuration 20% of its estimate for nothing.
#:
#: What was wrong BEFORE this amendment, and it is the safety half: the
#: term was charged by DRIVER MODEL, which meant Linux paid none of it at
#: all.  The Linux envelope then under-predicted every one of fifteen
#: instrumented legacy-RRTMG forecasts:
#: paragraph above said so: "re-measure there before exporting this term
#: -- the retention mechanism is a CuPy pool behaviour, not WDDM's, but
#: no Linux legacy-RRTMG run has been instrumented".  Fifteen have been
#: now, on two Linux cards, and the Linux envelope UNDER-PREDICTED every
#: one of them:
#:
#:   ==============  =======  ==========  ==========  ==========  ========
#:   card            runs     estimate    held/est    measured    residual
#:   ==============  =======  ==========  ==========  ==========  ========
#:   RTX 5070 Ti     10       5.47 GiB    1.172-1.186  7.61-7.69  +0.39..+0.47
#:   RTX 5090         5       5.47 GiB    1.176-1.186  9.08-9.13  +0.69..+0.74
#:   ==============  =======  ==========  ==========  ==========  ========
#:
#: The 2.5.0 release battery's shared 300x300x49 legacy-RRTMG domain,
#: whole 6 h forecasts, each run's own 20 Hz watcher receipt; residual is
#: against the Linux envelope AS IT SHIPPED, i.e. with no slack term at
#: all.  The worst point needs 0.095 of the estimate on top of the
#: 0.5 GiB constant; the WDDM lane's already-shipped 0.20 covers it with
#: better than 2x margin and is the same mechanism measured on the same
#: allocator, so the term is charged on every DRIVER MODEL and priced by
#: RADIATION LANE instead.  Receipts:
#: docs/public/receipts/linux/linux-vram-calibration-20260820.json,
#: docs/public/receipts/wddm/rtx3080-wddm-calibration-20260819.json.
POOL_SLACK_FRACTION = 0.20

#: The name this term shipped under while it was believed to be a WDDM
#: property.  Kept so the 2.5.0 receipts written in its terms still read.
WDDM_POOL_SLACK_FRACTION = POOL_SLACK_FRACTION

#: What the Windows affine envelope rests on, printed beside the number.
ENVELOPE_WDDM_BASIS = (
    "measured, RTX 3080 10 GiB / Windows 11 WDDM, six whole bare-default "
    "forecasts machine-wide at 0.25 s over a 2.5x span of itemized "
    "estimate, rte-rrtmgp + legacy-RRTMG suites")

#: ...and what the Linux one rests on, since 2026-08-20.
ENVELOPE_LINUX_POOL_BASIS = (
    "measured, RTX 5070 Ti 16 GiB and RTX 5090 32 GiB / Linux, fifteen "
    "whole release-battery forecasts sampled at 20 Hz, CuPy pool held "
    "1.17-1.19x the itemized estimate")


#: The platform names this accounting has evidence for.  ``linux``
#: covers WSL and Linux containers, which report ``linux`` too.
_LINUX_PLATFORMS = ("linux",)
_WINDOWS_PLATFORM_PREFIXES = ("win", "msys")
_WINDOWS_PLATFORM_NAMES = ("cygwin",)



#: What `gpuwm check` says about an experimental two-way configuration.
#:
#: `feedback = 1` is a legal schema value, so a node-7 validation run
#: authored one, got a clean PASS and exit 0 here -- output identical to
#: the feedback=0 twin -- and discovered only at prepare, after a 26 s
#: hierarchy build, that the prepared-hierarchy route refused two-way
#: nesting outright.  THAT ROUTE REFUSAL IS LIFTED and this text moved
#: with it: the prepared executor now passes
#: ``skip_feedback_path=(feedback == 0)``
#: (gpuwm/prepared_domain_tree_forecast.py:2203) and the hierarchy
#: stamps the experiment's own setting instead of refusing everything
#: but 0 (gpuwm/source_hierarchy.py:129).  A wizard-built three-domain
#: tree taken through the shipped front doors on real forcing recorded
#: 640 feedback transactions against 0 on the arm that differed only in
#: the two [experiment] keys.  What survives is the experimental stamp
#: and the three shapes the coupler refuses BY NAME when it is built
#: (gpuwm/core/nest.py:232-249) -- those are what a two-way author needs
#: before the ingest rather than after it.  This is an advisory, not a
#: gate: it changes no exit code and blocks nothing.
FEEDBACK_TWO_WAY_ADVISORY = (
    "experimental: feedback = 1 selects two-way nest feedback, which is "
    "stamped experimental and runs on BOTH routes -- the native "
    "experiment-runner route (`gpuwm run`) and the prepared-hierarchy "
    "route, `rw-wps` preparation followed by the domain-tree runner.  "
    "What refuses is not the route but the tree: the nest coupler names "
    "three shapes it cannot feed back when it is built -- unequal "
    "parent/child vertical level counts, mixed parent/child "
    "microphysics, and mismatched active prognostic field inventories.  "
    "A tree clear of those three runs two-way wherever it is launched; "
    "feedback = 0 output is unchanged either way."
)


def feedback_advisory(exp) -> str | None:
    """The two-way advisory when it applies, else None."""

    return (FEEDBACK_TWO_WAY_ADVISORY
            if int(getattr(exp, "feedback", 0) or 0) == 1 else None)


def spawn_reservation_advisories(exp) -> list[str]:
    """One plain sentence per DORMANT (spawn-declared) nest.

    The reservation contract, said where the numbers are: a declared
    spawn-triggered nest is priced by this preflight exactly as if it
    were live -- that is what makes VRAM deterministic and lets this
    report refuse honestly -- so its residency is spent for the whole
    run even if the trigger never fires, and it costs zero compute
    until it spawns.
    """
    import dataclasses as _dc

    lines: list[str] = []
    dormant = [dc for dc in exp.domains
               if getattr(dc, "spawn", None) is not None]
    if not dormant:
        return lines
    full = estimate_experiment(exp).alloc_estimate_bytes
    for dc in dormant:
        without = _dc.replace(exp, domains=tuple(
            d for d in exp.domains if d.grid_id != dc.grid_id))
        delta = full - estimate_experiment(without).alloc_estimate_bytes
        lines.append(
            f"d{dc.grid_id:02d} is a DORMANT spawn-triggered nest "
            f"(trigger {dc.spawn.trigger!r}): declaring it costs "
            f"{delta / GIB:.2f} GiB of this plan's alloc estimate, "
            "reserved from startup and spent even if the trigger never "
            "fires; it costs zero compute until it spawns. Every figure "
            "in this report already includes it.")
    return lines


def anisotropic_w_mixing_advisories(exp) -> list[str]:
    """One line per domain whose per-axis mixing length is over the limit.

    The same sentence :func:`gpuwm.config.warn_anisotropic_w_mixing`
    prints at config load, repeated HERE because that is not where a
    reader is looking.  The 1.6.0 instability that aborted an LES run
    had already been warned about at load, hours earlier, in a stream
    nobody re-read; the run then died 5,467 steps in on a health bound
    that named a vertical velocity and not a cause.  A preflight report
    is the door a user opens before paying for the run, and ``--json``
    puts the same text under ``advisories`` where a script can gate on
    it.

    Advisory, exactly like its neighbours: it changes no exit code and
    blocks nothing.  See ``warn_anisotropic_w_mixing`` for why the
    criterion is not a refusal.
    """

    from gpuwm.config import (anisotropic_w_mixing_advice,
                              auto_mix_isotropic_selection)
    from gpuwm.experiment import (anisotropic_w_mixing_exposure,
                                  auto_selected_isotropic_mixing)

    auto = set(getattr(exp, "auto_mix_isotropic", ()) or ())
    dz_max, exposed, ladder = anisotropic_w_mixing_exposure(exp)
    lines: list[str] = []
    for domain in exposed:
        _, advice = anisotropic_w_mixing_advice(
            where=f"d{domain.grid_id:02d}",
            km_opt=domain.run.km_opt,
            mix_isotropic=domain.run.mix_isotropic,
            mix_upper_bound=domain.run.mix_upper_bound,
            dx=domain.run.dx, dy=domain.run.dy, dz_max=dz_max,
            ladder=ladder, forced=domain.grid_id not in auto)
        if advice:
            lines.append(advice)
    # The domains whose isotropic length was the MODEL'S choice (the
    # 2026-08-16 auto-switch): the report states what the run WILL do --
    # the selection, the ratio and the limit -- rather than an advisory
    # that something is wrong.  A written mix_isotropic = 1 is a
    # legitimate configuration and stays out of this list entirely.
    selected, selected_ladder = auto_selected_isotropic_mixing(exp)
    for grid_id, ratio in selected:
        lines.append(auto_mix_isotropic_selection(
            where=f"d{grid_id:02d}", ratio=ratio, ladder=selected_ladder))
    return lines


#: "The caller did not price this" -- distinct from a caller that priced
#: it and got ``None``, which is the answer "this config does not stream".
_UNPRICED = object()


def streaming_advisory(exp, *, machine=None,
                       envelope=_UNPRICED) -> str | None:
    """Say out loud WHICH allocation this report prices.

    Every ENUMERATION in this module -- ``alloc_estimate_bytes``, the
    itemized peak envelope, the reserve, the whole N0 chain -- prices a
    domain resident in VRAM, and none of them has a concept of a tile
    buffer.  With ``[tiles]`` configured that is a fact the reader has to
    be told, because a report that shows only those figures is otherwise
    indistinguishable from one describing the run they actually asked for.

    What the report's BINDING figures now carry is the streamed envelope
    (:func:`estimate_phases` replaces the forecast term with it, and
    ``check_main`` compares the gate leg and the exit code against it), so
    the sentence below is a statement about the run the config asks for
    rather than a warning that the numbers are about a different one.

    REWRITTEN AT 2.2.0, TWICE OVER.  Both halves of what this used to say
    are now false, and a stale advisory is worse than none: it was the
    release's own headline feature telling users it does not work.

    * "this estimator has no model of a streamed domain" -- it does now.
      :func:`streamed_forecast_envelope` prices the tile working set, the
      buffers and the pinned store off the same measured
      :class:`tilestream.autoplan.Footprint` the run attaches with, and
      :func:`estimate_phases` puts that number in the forecast term.  So a
      refusal here is a statement about the run the config asks for.
    * "'on' is refused by the forecast routes, which wire no
      streamed-domain builder" -- they wire one as of 2.2.0
      (``prepared_single_domain_forecast``, ``prepared_domain_tree_forecast``),
      which is the whole point of the release.

    What remains worth saying is the one thing that is still true and still
    surprising: under ``mode = "auto"`` the DECISION is the planner's, taken
    against free VRAM at the instant the run starts, so a report written
    now can describe the other branch if the card's occupancy changes.

    Advisory, never a gate: it changes no exit code and blocks nothing,
    on the same posture as every other entry in :func:`check_advisories`.

    ONE DECISION PER REPORT.  ``envelope`` is accepted so a caller that
    has already priced the streamed forecast hands that envelope over
    rather than having this function derive a second one.  Under
    ``mode = "auto"`` the two derivations genuinely disagree: this one
    reached ``Machine.detect`` and planned against the card under the
    desk while the report's verdict planned against the card the reader
    declared, and one ``gpuwm check --budget-gib 6`` printed "2 tile
    buffer(s) of 220x174 = 5.80 GiB" at the top and "2 tile buffer(s) of
    311x146 ... 6.17 GiB" in its binding-phase line.  ``machine`` is the
    same escape for a caller that has a card but not yet an envelope.
    """
    options = getattr(exp, "tiles", None)
    mode = getattr(options, "mode", "off")
    if mode == "off":
        return None
    nested_note = ""
    if len(getattr(exp, "domains", ()) or ()) > 1:
        nested_note = (
            "  This tree is NESTED: roads are assigned per domain "
            "(streaming.steppers_for_tree walks parent-first, prices each "
            "domain against the budget its predecessors left -- resident "
            "claims, tile working sets and the coupling corridor's slots "
            "-- and records each domain's road and claim in its decision "
            "receipt), so a refusal or a fit below describes the resident "
            "tree, not the mixed-road one the run will take.")
    if envelope is _UNPRICED:
        envelope = streamed_forecast_envelope(exp, machine=machine)
    if envelope is not None:
        return (f"[tiles] mode = '{mode}' streams this domain, so the memory "
                "numbers in this report price the STREAMED allocation and "
                f"not a resident one: {envelope.summary()}." + nested_note)
    if mode == "auto":
        return (
            "[tiles] mode = 'auto' is configured, so whether this domain "
            "streams is decided at run time by tilestream.autoplan against "
            "the free VRAM of that moment.  The numbers below price the "
            "RESIDENT allocation, which is the branch auto takes when the "
            "domain fits; if it does not fit, the run streams instead and "
            "holds a few tile buffers rather than the whole domain, so a "
            "refusal here is not the last word." + nested_note)
    return (
        f"[tiles] mode = '{mode}' is configured but no streamed envelope "
        "could be priced for this domain, so the numbers below price the "
        "RESIDENT allocation.  That happens when the planner can fit no "
        "tile in this card's budget at all, in which case streaming would "
        "not have saved the run either." + nested_note)


from gpuwm.core.pace import UNPRICED as _PACE_UNPRICED


def pace_machine_from_free_bytes(free_bytes: int | None):
    """An ``autoplan.Machine`` for the card THIS REPORT measured.

    The column bound has to be priced against the allowance the reader
    actually has, and ``gpuwm check`` is the one surface that has already
    measured it.  ``vram_headroom`` is zeroed because ``free_bytes`` is
    already what the card will give this process now, and
    ``autoplan.budget_for`` applies the rung's radiation reservation on
    top -- letting the percentage headroom stand as well would withhold
    the same bytes twice, which is the double count
    :func:`tilestream.autoplan.budget_for` exists to avoid.

    ``None`` when nothing measured the card, so the bound is omitted
    rather than guessed.
    """
    if not free_bytes or int(free_bytes) <= 0:
        return None
    from tilestream import autoplan

    host = _host_total_bytes_or_none()
    return autoplan.Machine(
        vram_bytes=int(free_bytes),
        host_bytes=int(host or free_bytes), name="measured free VRAM",
        vram_headroom=0.0, host_source="gpuwm check")


def _host_total_bytes_or_none() -> int | None:
    from gpuwm.core.streaming import _host_total_bytes

    try:
        return _host_total_bytes()
    except Exception:
        return None


def pace_advisory(exp, *, streamed=_PACE_UNPRICED, machine=None) -> str | None:
    """The pace sentence, re-exported so ``check_main`` reads one name.

    The default is the PACE module's own sentinel, not ``None``.  Passing
    ``None`` here would be the positive claim "this plan runs resident",
    and a re-export that quietly made that claim reported the resident
    road for a config whose [tiles] table streams.
    """
    from gpuwm.core.pace import pace_advisory as _pace_advisory

    return _pace_advisory(exp, streamed=streamed, machine=machine)


def pace_estimate_for_report(exp, *, streamed=None, free_bytes=None):
    """The pace this report should publish, priced against its own card."""
    from gpuwm.core.pace import estimate_pace

    return estimate_pace(exp, streamed=streamed,
                         machine=pace_machine_from_free_bytes(free_bytes))


def check_advisories(exp, config_path=None, *, machine=None,
                     streamed=_UNPRICED) -> list[str]:
    """Every route advisory this config earns, in report order.

    Same posture as ``feedback_advisory``: these change no exit code and
    block nothing.  They exist because a legal config that a later stage
    silently ignores is worse than one it refuses -- the user learns
    after paying for the run instead of before it.

    ``machine`` and ``streamed`` are relayed to :func:`streaming_advisory`
    so a report that has already resolved ``[tiles]`` describes the
    tiling it is about to quote numbers for, and not a second one.
    """

    from gpuwm.checkpoint_routes import (
        checkpoint_route_advisory, config_has_case_data)

    # ONE DECISION, resolved once and given to both sentences below.
    # Deriving it twice is the defect ``streaming_advisory``'s docstring
    # records: under ``mode = "auto"`` two derivations planned against
    # two different cards and one report printed two tilings.
    if streamed is _UNPRICED:
        streamed = streamed_forecast_envelope(exp, machine=machine)
    advisories = [feedback_advisory(exp)]
    advisories.extend(spawn_reservation_advisories(exp))
    advisories.extend(anisotropic_w_mixing_advisories(exp))
    advisories.append(streaming_advisory(exp, machine=machine,
                                         envelope=streamed))
    # THE PACE, immediately after the sentence that says which road this
    # config takes -- the two belong together, because "this domain
    # streams" is only actionable next to what streaming costs and what
    # domain size would not.
    # THE PACE IS NOT AN ADVISORY, and putting it here was wrong.  Every
    # entry in this list is EARNED -- a config that turned something on
    # gets told what that costs it -- and the negative control in
    # tests/test_checkpoint_route_contract.py pins that a plain config
    # earns none, so "a green check stays green for everybody else".  The
    # pace is owed to every run, including the plainest, so it is printed
    # on its own line by ``check_main`` and published as the
    # ``expected_pace`` object in the JSON, not smuggled into a list
    # whose contract is the opposite.
    if config_path is not None:
        advisories.append(checkpoint_route_advisory(
            domain_count=len(exp.domains),
            has_case_data=config_has_case_data(config_path),
            restart_interval_s=getattr(exp, "restart_interval_s", 0.0)))
    return [line for line in advisories if line]


def platform_is_measured(platform: str | None = None) -> bool:
    """Has this platform's memory behaviour actually been observed?

    Windows (including Cygwin and MSYS, which run on WDDM) and Linux
    have measurements behind them.  Nothing else does.
    """

    name = sys.platform if platform is None else str(platform)
    return (name.startswith(_WINDOWS_PLATFORM_PREFIXES)
            or name in _WINDOWS_PLATFORM_NAMES
            or name.startswith(_LINUX_PLATFORMS))


def unknown_platform_note(platform: str | None = None) -> str | None:
    """One line for a platform with no measurements, else ``None``."""

    name = sys.platform if platform is None else str(platform)
    if platform_is_measured(name):
        return None
    return (
        f"note: platform {name!r} has no VRAM measurements behind it "
        f"(only Windows and Linux do), so the conservative Windows/WDDM "
        f"accounting is applied -- it may size a smaller domain than "
        f"this machine can run.")


def envelope_platform(platform: str | None = None,
                      vram_gib: float | None = None) -> str:
    """Which envelope family applies: ``windows`` or ``linux``.

    PLATFORM defaults to :data:`sys.platform`.

    Two platforms have measurements: Windows -- with Cygwin and MSYS,
    which are the same WDDM driver under a different shell -- and Linux,
    which is also what WSL and Linux containers report.  Anything else
    is a platform nobody has measured, and it takes the **conservative**
    (Windows) accounting rather than the optimistic one.

    That is a change from v1.0.0, which returned ``linux`` for every
    non-Windows name and so quietly priced an unknown platform with the
    envelope that omits 4.12 GiB of fixed constants.  Fail-open on an
    unsupported platform is the wrong direction: the Linux numbers are
    not a default, they are three runs on two Linux cards.  Callers
    should print :func:`unknown_platform_note` beside the sizing so the
    substitution is visible rather than silent.

    VRAM_GIB is accepted for callers that know the card and is
    deliberately IGNORED.  It used to select an experimental
    "windows-small" tier at or under 12 GiB, which is how the wizard
    (which knows the card size) and ``gpuwm check`` / ``gpuwm go``
    (which measure free VRAM and passed no size) priced the very same
    bytes with two different formulas -- the wizard's inline check said
    PASS and the standalone check exited 4 seconds later (open task
    #162; the 2026-08-19 3080 walk).  One machine gets ONE envelope
    family, decided by the platform alone; the 3080 calibration that
    made the Windows family measured (:data:`WDDM_POOL_SLACK_FRACTION`)
    is what retired the tier.
    """

    name = sys.platform if platform is None else str(platform)
    if name.startswith(_LINUX_PLATFORMS):
        return "linux"
    return "windows"


def peak_envelope_factor(platform: str | None = None,
                         vram_gib: float | None = None) -> float:
    """The RETIRED multiplicative factor for this platform.

    Nothing gate-side reads it any more; ``gpuwm downscale``'s
    standalone child fit keeps it as a deliberately conservative bound.
    """

    return PEAK_ENVELOPE_FACTORS[envelope_platform(platform, vram_gib)]


def platform_projection_constants(
        platform: str | None = None,
        vram_gib: float | None = None) -> tuple[int, int]:
    """``(retention_residual, device_overhead)`` for the projection.

    Both are grid-independent constants calibrated on one Windows/5090
    fixture, and neither showed up in any of the three instrumented
    Linux runs -- whose peaks tracked the itemized alloc estimate to
    within 1.15-1.32x.  Adding 4.12 GiB of Windows pool accounting to a
    Linux projection is not conservatism, it is a wrong number: it put
    the 12 GiB tier out of reach entirely.  Zero on Linux, unchanged on
    Windows.

    These feed the TIER 2/3 projection DISPLAY lines and nothing else:
    since the 3080 calibration they are not envelope terms on any
    platform (:attr:`ExperimentMemoryEstimate.envelope_intercept_bytes`
    is the itemized non-pool residency alone).
    """

    family = envelope_platform(platform, vram_gib)
    if family == "windows":
        return pool_retention_residual_bytes(), PROBE_DEVICE_OVERHEAD_BYTES
    return 0, 0


#: What the machine-wide peak carries beyond the itemized estimate and
#: the itemized non-pool residency: allocator fragmentation, the driver's
#: own working set, and whatever the shape formulas do not enumerate.
#:
#: MEASURED 2026-08-01, RTX 4080 16 GiB / Linux / driver 595.58.03, GPU
#: otherwise idle, machine-wide ``nvidia-smi`` sampled every 250 ms across
#: whole prepared-cache forecasts.  Fitting ``peak = a x subtotal + b``
#: over the single-domain runs returns a = 0.98 -- i.e. the itemization
#: predicts the pool 1:1 and the residue is a CONSTANT, which is exactly
#: what a CUDA context plus a launch-time local-memory backing store is.
#: :data:`ENVELOPE_UNMODELLED_BYTES` is the part of that constant this
#: module does not already itemize, rounded UP over the worst residual
#: (an envelope must never round down).
#:
#: Residuals of ``measured - (alloc estimate + non-pool)``, single domain,
#: staged prepared-cache route unless marked:
#:
#:   ===========  ==========  ==========  ==========  ==========
#:   run          grid        estimate    measured    residual
#:   ===========  ==========  ==========  ==========  ==========
#:   s07          170x136      2.07 GiB    3.65 GiB   +0.05 GiB
#:   small8       224x180      2.75 GiB    4.14 GiB   -0.14 GiB
#:   small8 (go)  224x180      2.75 GiB    4.38 GiB   +0.10 GiB
#:   s11          340x272      4.82 GiB    5.95 GiB   -0.41 GiB
#:   edge15 (go)  448x360      7.56 GiB    8.75 GiB   -0.34 GiB
#:   L12    (go)  474x378      8.27 GiB    9.25 GiB   -0.55 GiB
#:   over22 (go)  594x476     12.38 GiB   12.59 GiB   -1.33 GiB
#:   big24  (go)  630x504     13.76 GiB   13.88 GiB   -1.42 GiB
#:   ===========  ==========  ==========  ==========  ==========
#:
#: The worst positive residual is +0.10 GiB; the same three-node Linux
#: pilot set of 2026-07-30, re-read through this model, leaves +0.46 GiB
#: of margin at its tightest (node 1, 4090).  A round half-gibibyte
#: covers both with room, and unlike a multiplier it does not grow with
#: the grid -- which is the whole point of having an intercept.
ENVELOPE_UNMODELLED_BYTES = GIB // 2

#: Per NEST beyond the root, as a FRACTION of the itemized estimate.
#:
#: A tree lands consistently above the single-domain line, because the
#: shared scratch arena and the shared dycore-state workspace are priced
#: at their per-slot maximum while the nest coupler's own buffers, the
#: extra per-domain physics drivers and the per-domain output staging are
#: not on that maximum.  Measured, same card and instrument, as
#: ``measured - (estimate + non-pool)`` over the estimate:
#:
#:   =========  =======  ==========  ==========  ==============
#:   run        domains  estimate    residual    per extra dom.
#:   =========  =======  ==========  ==========  ==============
#:   n10        2         4.12 GiB   +0.18 GiB      4.3%
#:   n2_16      2         8.22 GiB   +0.34 GiB      4.2%
#:   c07        4         2.06 GiB   +0.26 GiB      4.2%
#:   =========  =======  ==========  ==========  ==============
#:
#: Three trees, two depths, a 4x span of estimate, and the same number
#: each time -- so it is PROPORTIONAL, not a flat allowance, and 0.05
#: rounds it up.  Zero for a single domain by construction, so no
#: single-domain number in this release moves because of it.
ENVELOPE_PER_NEST_FRACTION = 0.05

#: What the affine envelope rests on, printed beside the number it makes.
ENVELOPE_AFFINE_BASIS = (
    "measured, RTX 4080 16 GiB / Linux, whole forecasts machine-wide at "
    "250 ms across a 6.6x span of itemized estimate, 1/2/4 domains")


def machine_peak_envelope_bytes(
        *, alloc_estimate_bytes: int, non_pool_bytes: int,
        domains: int = 1,
        footprint_projection_bytes: int | None = None,
        family: str = "linux",
        legacy_radiation: bool = True) -> int:
    """The machine-wide peak a run of this configuration should reach.

    AFFINE, not multiplicative.  The old form was ``factor x projection``
    with no intercept, and a model with no intercept cannot describe a
    cost with a large fixed term: it under-predicts small configurations
    and over-predicts large ones, which is precisely what the 16 GiB
    fleet node measured (a 3.99 GiB declaration that peaked at 4.38, and
    a 19.95 GiB declaration that peaked at 13.88).

    The three terms are each something this module already knows:

    * the itemized allocation estimate -- the pool side, which the
      measurements track to within a few percent;
    * :func:`non_pool_device_bytes` -- the CUDA context plus the
      launch-time local-memory backing store, which scale with the DEVICE
      and the kernel set, not with the grid;
    * :data:`ENVELOPE_UNMODELLED_BYTES` (+ a per-nest term) -- the
      measured residue, stated as a constant because that is what it
      measured as.

    The WDDM lane adds ONE more measured term,
    :data:`WDDM_POOL_SLACK_FRACTION` of the estimate -- the pool
    retention the 3080 calibration measured beyond the affine terms
    (worst +0.30x of the estimate, legacy-RRTMG lane).  This REPLACES
    the retired ``footprint x 1.75`` floor, which predicted 3.8x the
    measured peak on the calibration card and refused runs that fit
    with gigabytes to spare.  ``footprint_projection_bytes`` is
    accepted for signature compatibility and no longer read.
    """

    nests = max(0, int(domains) - 1)
    affine = (int(alloc_estimate_bytes) + int(non_pool_bytes)
              + ENVELOPE_UNMODELLED_BYTES
              + math.ceil(ENVELOPE_PER_NEST_FRACTION * nests
                          * int(alloc_estimate_bytes)))
    # Charged on every driver model and priced by RADIATION LANE: the
    # retention is the legacy engines' call-peak workspace sitting in the
    # CuPy pool between calls, which is a property of the suite and of
    # the allocator, not of WDDM (see POOL_SLACK_FRACTION for the three
    # campaigns that split on it).  While it was Windows-only the Linux
    # envelope under-predicted every instrumented run on both Linux
    # cards; while it was unconditional it charged the rte-rrtmgp lane
    # 20% for a mechanism that lane does not have.
    #
    # The default is to CHARGE it.  A caller that has not said which
    # radiation lane it is on gets the conservative answer; an envelope
    # that guesses optimistically is not an envelope.
    if legacy_radiation:
        affine += math.ceil(POOL_SLACK_FRACTION * int(alloc_estimate_bytes))
    return affine


def observed_peak_envelope_bytes(
        footprint_projection_bytes: int,
        *, platform: str | None = None,
        vram_gib: float | None = None,
) -> int:
    """The RETIRED multiplicative machine-peak envelope.

    Kept because the WDDM factor is a real measurement over a real
    projection and the Windows lane still quotes it, and because the
    receipts of three releases are written in its terms.  New callers
    want :func:`machine_peak_envelope_bytes`, which has an intercept.
    """

    return int(footprint_projection_bytes
               * peak_envelope_factor(platform, vram_gib))


# ---------------------------------------------------------------------------
# TIER 3, RE-MEASURED: what the CuPy pool never sees
# ---------------------------------------------------------------------------
#
# ``PROBE_DEVICE_OVERHEAD_BYTES`` above (1.39 GiB) was taken from a process
# that ran ZERO steps, and its own docstring flags it as a lower bound
# "because zero steps JIT-compile almost none of the kernel modules".  It is
# worse than a lower bound: it models the wrong thing.  Measured 2026-07-26
# on the user's RTX 5090 (170 SMs, 1536 resident threads per SM, driver
# 610.74) by running real integrations with ``cudaMemGetInfo`` and the CuPy
# pool sampled together at 1 s, and bracketing the FIRST launch of every
# kernel symbol:
#
#   ============================  ==========  ==========  ==========
#   term                          3 domains   4 domains   scales with
#   ============================  ==========  ==========  ==========
#   CuPy pool, reserved peak      13,697 MiB  21,072 MiB  columns
#   CUDA context                     432 MiB     430 MiB  nothing
#   local-memory backing store     5,738 MiB      64 MiB  ONE kernel
#   other non-pool                  ~400 MiB    ~430 MiB  nothing
#   ============================  ==========  ==========  ==========
#
# NVRTC module images are not on that list because they do not measure: all
# 51 kernel modules compiled and loaded cost 2 MiB of device memory between
# them, and resolving every symbol another 18 MiB.
#
# The dominant term is the LOCAL-MEMORY BACKING STORE.  When a kernel's
# per-thread local frame exceeds the context's default stack limit, the
# driver answers the kernel's first launch by allocating a backing store for
# the device's whole resident-thread capacity -- not for the launched grid --
# and keeps it for the life of the process.  One allocation serves the
# process, sized by the LARGEST per-thread frame launched so far, so the term
# is a MAXIMUM over launched kernels and is completely independent of domain
# count.  That is why the old estimator was short by the same ~5.5 GiB at
# three domains and at four.
#
# The law is exact on this device, verified against a synthetic kernel at
# five local-frame sizes and against two real forecasts:
#
#     reservation = (max_local_size_bytes - default_stack_limit_bytes)
#                   * max_threads_per_multiprocessor * multiprocessor_count
#
# ``kf_column`` measures 24,064 B of local frame, so
# (24064 - 1024) * 1536 * 170 = 6,016,204,800 B = 5,737.5 MiB; the measured
# step across its first launch was 5,738.0 MiB, twice, in two separate runs.
# The baseline term ``default_stack_limit_bytes * capacity`` = 255 MiB is
# already inside :data:`CUDA_CONTEXT_BYTES`, which is why it is subtracted.

#: How far the CuPy pool's RESERVED high-water runs past the enforced
#: estimate.  ``alloc_estimate_bytes`` bounds pool USED, which is what
#: ``alloc_measured_le_estimate`` checks; a rail is spent on what the pool
#: HOLDS.  Measured on the two traced forecasts of 2026-07-26:
#:
#:   ==========  ==============  ================  ==========
#:   domains     estimate (MiB)  pool reserved     over
#:   ==========  ==============  ================  ==========
#:   3           13,373          13,697            2.42%
#:   4           20,480          21,072            2.89%
#:   ==========  ==============  ================  ==========
#:
#: 3% is the rounded-up bound over both, carried as the N0 reserve's
#: retention term whenever the caller supplies the estimate it applies to.
#: It is a fitted constant from two points and is documented as such; what
#: keeps it honest is that it moves the gate in the refusing direction.
POOL_RESERVED_OVER_ESTIMATE_FRACTION = 0.03

#: Device memory a fresh CUDA context holds before gpuwm allocates anything:
#: measured 432 MiB (three fresh processes: 433, 432, 430), of which
#: 1024 B x 1536 x 170 = 255 MiB is the default-stack local-memory store.
CUDA_CONTEXT_BYTES = 432 * 1024 ** 2


#: What a CUDA context grows by once a forecast has loaded its kernel
#: modules, over the BARE context a fresh process stands up.
#:
#: MEASURED 2026-08-20 (task 206), two Linux cards, driver 13030.  The
#: bare context is read by ``tools/vram_reserve_probe.py`` in a process
#: that creates a context and allocates one byte; the run-time figure is
#: ``device footprint peak - pool-held peak`` from fifteen whole
#: forecasts' own 20 Hz :class:`~gpuwm.core.gpu_mem_watch.
#: GpuPeakMemoryWatcher` receipts, minus the reservation law's backing
#: store for the frame those runs launched:
#:
#:   ==============  ==========  ============  ==========
#:   card            bare        at run time   growth
#:   ==============  ==========  ============  ==========
#:   RTX 5070 Ti      230.0 MiB   382.8 MiB     152.8 MiB
#:   RTX 5070 Ti      230.0 MiB   386.5 MiB     156.5 MiB
#:   RTX 5090         506.0 MiB   659.7 MiB     153.7 MiB
#:   RTX 5090         506.0 MiB   664.3 MiB     158.3 MiB
#:   ==============  ==========  ============  ==========
#:
#: The growth is a CONSTANT across a 2.4x span of card -- which is what
#: NVRTC module images and the driver's own working set should be, and
#: is not what a flat total context can be.  192 MiB rounds the worst
#: measurement up; an envelope must not round down.
#:
#: This is what RETIRES :data:`CUDA_CONTEXT_BYTES` as the charged term.
#: That constant is one 2026-07-26 reading of one card, and applied
#: everywhere it was wrong in BOTH directions at once: 48 MiB high on a
#: 5070 Ti and 215 MiB LOW on the very 5090 it was taken from, once that
#: card ran under Linux rather than WDDM.  A term that under-charges is
#: not conservative, it is an OOM waiting for a big enough card.
CONTEXT_RUNTIME_GROWTH_BYTES = 192 * 1024 ** 2

#: Bare-context bytes per resident thread, for a card nobody has measured.
#:
#: Same campaign, the three cards' bare contexts over their
#: resident-thread capacities: 1,748 B (RTX 3080, WDDM, with a live
#: desktop sharing the card), 2,243 B (RTX 5070 Ti) and 2,032 B
#: (RTX 5090).  2,304 B is the round figure above all three, so an
#: unmeasured card is never priced below any card that was.
#:
#: A MODELLED number, and it says so wherever it is printed
#: (:func:`non_pool_basis`).  A present card is always measured instead:
#: this rate exists for ``--card``/``--vram-gib`` sizing of a machine
#: that is somewhere else.
MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD = 2304


@dataclass(frozen=True)
class DeviceLocalMemoryProfile:
    """The device constants the local-memory reservation law needs."""

    name: str
    multiprocessor_count: int
    max_threads_per_multiprocessor: int
    default_stack_limit_bytes: int = 1024
    #: Device bytes a bare CUDA context holds on THIS card, measured.
    #: ``None`` for a card that is not in the machine, which is then
    #: priced from
    #: :data:`MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD`.
    bare_context_bytes: int | None = None

    @property
    def resident_thread_capacity(self) -> int:
        return (self.multiprocessor_count
                * self.max_threads_per_multiprocessor)

    @property
    def context_is_measured(self) -> bool:
        return self.bare_context_bytes is not None

    @property
    def cuda_context_bytes(self) -> int:
        """Device bytes this card's CUDA context holds during a run.

        The bare context -- measured on the card when it is present,
        modelled from its resident-thread capacity when it is not --
        plus :data:`CONTEXT_RUNTIME_GROWTH_BYTES`, the module-load growth
        both instrumented Linux cards showed to within 5 MiB of each
        other.
        """
        bare = self.bare_context_bytes
        if bare is None:
            bare = (MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD
                    * self.resident_thread_capacity)
        return int(bare) + CONTEXT_RUNTIME_GROWTH_BYTES

    def reservation_bytes(self, max_local_size_bytes: int) -> int:
        """Device bytes the driver reserves for a launched kernel whose
        per-thread local frame is ``max_local_size_bytes``.  Zero when the
        frame fits the default stack, whose store the context already
        carries.

        VERIFIED EXACT 2026-08-20 on three cards and both driver models
        (``tools/vram_reserve_probe.py``, validated in both directions:
        frames at or under the default stack step exactly zero device
        bytes, frames above it step this product to the byte on WDDM and
        to within 1.5 MiB on Linux).  The law is not the defect; what was
        wrong was the profile it was evaluated on.
        """
        over = int(max_local_size_bytes) - self.default_stack_limit_bytes
        return 0 if over <= 0 else over * self.resident_thread_capacity


def non_pool_basis(profile: "DeviceLocalMemoryProfile") -> str:
    """One sentence naming the card row a non-pool charge came from.

    Printed beside the number.  A grid-independent term large enough to
    refuse a card on its own has to be traceable to the reading that
    made it, or the reader has no way to tell a measurement from an
    assumption -- which is the whole history of this module.
    """

    if profile.context_is_measured:
        return (
            f"measured on this card ({profile.name}, "
            f"{profile.multiprocessor_count} SMs x "
            f"{profile.max_threads_per_multiprocessor} threads): CUDA "
            f"context {profile.cuda_context_bytes / GIB:.2f} GiB "
            f"(bare {profile.bare_context_bytes / GIB:.2f} + "
            f"{CONTEXT_RUNTIME_GROWTH_BYTES / GIB:.2f} module-load growth) "
            f"plus the local-memory backing store of its kernel set")
    return (
        f"modelled for an absent card ({profile.name}, "
        f"{profile.multiprocessor_count} SMs x "
        f"{profile.max_threads_per_multiprocessor} threads): CUDA context "
        f"{profile.cuda_context_bytes / GIB:.2f} GiB from the measured "
        f"{MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD} B per resident "
        f"thread, plus the local-memory backing store of its kernel set.  "
        f"Sizing for the machine you are on measures this instead")


#: Measured 2026-07-26 from ``cudaGetDeviceProperties`` +
#: ``cudaDeviceGetLimit(cudaLimitStackSize)`` on the run host.
#:
#: ``bare_context_bytes`` is deliberately left unset even though this
#: card's bare context WAS measured (530,579,456 B, weather-node-2,
#: 2026-08-20).  This profile is what an ABSENT card is priced against,
#: and the absent-card path may never be more optimistic than the
#: present-card one -- the 2026-08-03 lesson that retired
#: :data:`CARD_CLASS_MULTIPROCESSORS`.  A card in the machine is
#: measured (:func:`local_memory_profile_from_device`).
MEASURED_LOCAL_MEMORY_PROFILE = DeviceLocalMemoryProfile(
    name="NVIDIA GeForce RTX 5090",
    multiprocessor_count=170,
    max_threads_per_multiprocessor=1536,
    default_stack_limit_bytes=1024,
)


#: RETIRED 2026-08-03 -- the per-class SM discount for cards not in this
#: machine.  Each row claimed the LARGEST SM count sold at that capacity
#: ("(frame - default stack) x SMs x threads per SM" is a property of the
#: DEVICE), so a 12 GiB tier was priced at 70 SMs instead of the 170-SM
#: reference.  Two things killed it, both from the first user-zero
#: cross-architecture stress run (RTX 4090, sm_89):
#:
#: * IT WAS NOT A BOUND.  The rows were a market survey, not a
#:   measurement: a 12 GiB RTX 3080 Ti ships 80 SMs against the row's
#:   70, so the "upper bound" under-priced real cards in the class.
#: * IT MADE THE ABSENT-CARD PATH MORE OPTIMISTIC THAN THE PRESENT-CARD
#:   PATH.  Sizing for a card not in the machine used a 1.45 GiB
#:   non-pool intercept where the same code on the real 4090 measured
#:   2.30 GiB; a config certified "fits with 0.27 GiB to spare" landed
#:   0.015 GiB from the budget -- a margin 18x smaller than advertised,
#:   on exactly the sizing-for-a-card-you-intend-to-buy path, where the
#:   number cannot be checked until the money is spent.
#:
#: Kept because three sizing receipts are written in its terms.  No
#: caller consults it: :func:`card_local_memory_profile` prices every
#: absent card against the measured reference profile.
CARD_CLASS_MULTIPROCESSORS = ((12.0, 70), (16.0, 84), (24.0, 128),
                              (32.0, 170))


def card_local_memory_profile(
        vram_gib: float | None) -> DeviceLocalMemoryProfile:
    """The device profile for a card that is NOT in this machine.

    Always the measured reference profile -- the max of known-device
    intercepts -- whatever capacity is declared.  An absent card's SM
    count is unknown, and the retired per-class discount above priced a
    certified margin 18x too generous on the one machine that could
    check it; hardware that cannot be measured gets the conservative
    intercept, never a discount.  A live device always overrides this
    (see :func:`live_device_local_memory_profile`): the absent-card path
    can over-price relative to the card eventually bought, but it can
    never be MORE optimistic than a present-card measurement.
    """

    return MEASURED_LOCAL_MEMORY_PROFILE


def _nvml_used_bytes_or_none() -> int | None:
    """Device-wide used bytes from NVML, or ``None`` if unreadable."""
    try:
        from gpuwm.supervisor import _run_nvidia_smi

        text = _run_nvidia_smi(["--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"])
        return int(text.strip().splitlines()[0]) * 1024 ** 2
    except Exception:  # noqa: BLE001
        return None


def measured_bare_context_bytes(before: int | None, after: int | None, *,
                                stack_store_bytes: int) -> int | None:
    """A bare CUDA context's cost from an NVML reading either side of it.

    NVML and not ``cudaMemGetInfo``: ``total - free`` counts every other
    process on the card, and on a WDDM desktop that is gigabytes of
    somebody else's compositor (the 3080 read 1.12 GiB that way against a
    174 MiB delta).  What this process ADDED is the delta.

    Two ways the delta can lie, and both are refused rather than
    smoothed: a context that already existed makes it ~0, and a desktop
    that freed memory mid-reading can make it negative or absurd.  A
    reading below the default-stack backing store the driver MUST have
    allocated is not a measurement of this context, and neither is one
    above 4 GiB.  ``None`` means "unmeasured", which prices from
    :data:`MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD` and says so.
    """

    if before is None or after is None:
        return None
    delta = int(after) - int(before)
    if delta < int(stack_store_bytes) or delta > 4 * GIB:
        return None
    return delta


def local_memory_profile_from_device(cp) -> DeviceLocalMemoryProfile:
    """Read the profile off the attached device (``--alloc`` path only).

    Also MEASURES this card's bare CUDA context when this call is what
    creates it, which is the whole point of reading a present card
    instead of pricing it from a table: see
    :func:`measured_bare_context_bytes` for why a context that already
    existed reads as unmeasured rather than as zero.
    """
    # Read the card BEFORE the first CUDA call below, which is what
    # stands the context up.
    before = _nvml_used_bytes_or_none()
    # ``gpuwm multi-run`` masks one physical UUID into each check process;
    # CUDA ordinal 0 is therefore the selected logical device, not a claim
    # that every run belongs on the machine's physical index zero.
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    stack_limit = int(cp.cuda.runtime.deviceGetLimit(0))
    capacity = (int(props["multiProcessorCount"])
                * int(props["maxThreadsPerMultiProcessor"]))
    after = _nvml_used_bytes_or_none()
    return DeviceLocalMemoryProfile(
        name=name.decode() if isinstance(name, bytes) else str(name),
        multiprocessor_count=int(props["multiProcessorCount"]),
        max_threads_per_multiprocessor=int(
            props["maxThreadsPerMultiProcessor"]),
        default_stack_limit_bytes=stack_limit,
        bare_context_bytes=measured_bare_context_bytes(
            before, after, stack_store_bytes=stack_limit * capacity),
    )


#: Per-module MAXIMUM static local frame per thread, in bytes, as the CUDA
#: driver reports it in ``CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`` after NVRTC
#: compiles ``gpuwm/core/kernels/<module>.cu`` with the shipped options and
#: NO integer defines injected -- i.e. at each module's unspecialized bound.
#:
#: THIS IS A CEILING OVER NAMED COMPILE PLATFORMS, not one box's reading.
#: A frame is what NVRTC emitted for one target architecture at one
#: compiler build, and measured three ways it moves with both: ``gf``,
#: ``noah``, ``thompson_aerosol_warm`` and ``ysu`` move with the
#: architecture at a fixed compiler, ``nssl2_fused_gs``, ``rrtmgp_cloud``
#: and ``shinhong`` move with the compiler build at a fixed architecture,
#: and ``noahmp_leaves`` moves with both.  The readings themselves, each
#: naming its box, its ``sm_`` target and its NVRTC build, live in
#: :mod:`gpuwm.core.kernel_frame_recordings`; every row below is the
#: element-wise MAXIMUM over them, checked against them at import.
#:
#: Under-pricing is what a rail gate cannot survive -- one byte of frame
#: is one byte times the whole resident-thread capacity of device memory
#: nobody charged for -- so a platform nobody has measured is priced at
#: the ceiling, never at an average or at the nearest box.  Rows recorded
#: 2026-07-26 through 2026-08-20; regenerated and compared against the
#: compiler in front of it by ``tests/test_preflight.py::
#: test_the_recorded_local_frames_match_the_driver``, which asserts EXACT
#: equality only on a platform this tree has a recording for, and the
#: never-below-a-measurement invariant on every other.
#:
#: Two modules do not launch at their unspecialized bound: ``refl`` and
#: ``wdm6_refl`` compile their column arrays to the
#: configuration's own level count
#: (:data:`LEVEL_SPECIALIZED_KERNEL_FRAMES`), so their rows here are the
#: CEILING, not the price.  Three more are TIERED, and those are the only
#: ones that can price ABOVE their row.  ``acoustic`` compiles the implicit
#: w''-phi'' solve at a coarse ``WPHI_MAX_LEV`` tier chosen by ``nz``
#: (:data:`ACOUSTIC_TIER_FRAME`), so this row is its price at the shipped
#: 129 tier -- every ``nz <= 128`` configuration -- and the deeper tiers add
#: to it.  ``wdm6`` and ``wsm6`` are the same shape, each with a two-rung
#: ladder (:data:`WDM6_TIER_FRAME`, :data:`WSM6_TIER_FRAME`): each row is
#: that module's ``KMAX = 64`` default, every ``nz <= 64`` configuration,
#: and the 80 tier prices above it.
#: Everything else launches as compiled.
#:
#: A module maximum is an UPPER bound over the kernels a scheme can launch --
#: e.g. Morrison's launched sedimentation kernel measures 1,280 B against the
#: module's 5,120 B.  Over-pricing is the safe direction for a rail gate and
#: under-pricing is what put a run 1,630 MiB over; the bound is stated, not
#: silently tightened.
KERNEL_MAX_LOCAL_SIZE_BYTES: dict[str, int] = {
    "acoustic": 544,
    "advection": 0,
    "coriolis_map": 0,
    "diagnostics": 0,
    "diff6": 0,
    "diff6_seam": 0,
    "diffusion": 0,
    "dycore": 0,
    # The FTZ receipt's probe rides the production loader from the same
    # kernels directory, so the local-frame sweep sees it like any model
    # module; it holds no local frame.
    "ftz_probe": 0,
    # Grell-Freitas.  One thread still owns one whole GFDRV column, but
    # the column arrays no longer live in the per-thread local frame:
    # gpuwm/core/kernels/gf.cu keeps them in a global workspace that
    # gpuwm/core/gf.py sizes to the threads IN FLIGHT, because the driver
    # sizes the local-memory backing store to the threads the card can
    # ever hold.  MEASURED on node-1 (RTX 5070 Ti, sm_120): the frame went
    # 22,416 -> 72 B on NVRTC 13.3 and 22,416 -> 88 B on NVRTC 13.0.48,
    # and the launch-time reservation went 2,200.0 MiB -> 4.0 MiB.  Both
    # readings are under the 1,024 B default stack, so the row reserves
    # nothing on any card, and it no longer moves with nz -- the frame is
    # 72 B at the 40, 49, 55 and 64 tiers alike, which retires the
    # level-specialization gap this row used to carry.
    # The workspace itself is priced by :func:`gf_column_workspace_bytes`.
    "gf": 88,
    "health": 0,
    # The tile-streamed health reduction (gpuwm/core/streaming.py:1728).
    # It holds no local frame on either measured architecture.  The row
    # exists because the ``.cu`` does: it shipped with the out-of-core
    # merge after the last 5090 reading, so the regeneration gate had a
    # module it could not even enumerate until 2026-08-20.
    "health_tile": 0,
    # The batched symmetric eigensolver the radar-DA analysis factors with
    # (gpuwm/core/jacobi_eigh.py).  It holds NO local frame at any tier: the
    # whole k x k problem lives in dynamic SHARED memory, which is priced at
    # launch and released with the block rather than reserved per resident
    # thread for the life of the process.  That is the entire reason the
    # kernel is written around shared memory instead of per-thread arrays.
    "jacobi_eigh": 0,
    "kessler": 5120,
    # Kain-Fritsch.  One thread still owns one whole column, but 52 of
    # kf_column's 54 column arrays no longer live in the per-thread local
    # frame: gpuwm/core/kernels/kf.cu keeps them in a global workspace that
    # gpuwm/core/kf.py sizes to the threads IN FLIGHT, because the driver
    # sizes the local-memory backing store to the threads the card can ever
    # hold.  MEASURED on node-1 (RTX 5070 Ti, 70 SMs x 1,536, sm_120, NVRTC
    # 13.0.48): the frame went 9,216 -> 512 B and the launch-time
    # reservation went 840.0 MiB -> 0.0 MiB.
    #
    # 512 B, not 0: ``tv_env`` and ``positive_energy`` stay on the stack
    # because they are the only two whose PLACEMENT moves an output bit
    # (kf.cu says why), and at the unspecialized KF_KMAX = 128 the compiler
    # materialises 512 B of them.  That is half the 1,024 B default stack,
    # so the row reserves nothing on any card -- and it no longer moves
    # with nz, which is what retired kf's LEVEL_SPECIALIZED_KERNEL_FRAMES
    # entry and the per-nz recompile that entry priced.
    # The workspace itself is priced by :func:`kf_column_workspace_bytes`.
    "kf": 512,
    "kf_validation": 0,
    "lbc_flow": 0,
    "lbc_state": 0,
    # Milbrandt-Yau.  MEASURED on an RTX 5090 over all seven kernels of the
    # module: milbrandt2_sediment_256 is the only one with a frame worth
    # naming at 2,048 B -- exactly its two per-thread column arrays
    # ``float VVQ[KMAX]; float VVN[KMAX]`` at MY2_KMAX_GENERIC = 256
    # (2 * 256 * 4) -- with milbrandt2_sediment_64 at 512 B on the same
    # arrays at MY2_KMAX_SHALLOW = 64, and prelim/geometry/warm/cold/
    # diagnostics holding no local frame at all.
    #
    # This row is a CEILING in the same sense Morrison's is: the launcher
    # picks the 64-level kernel for nz <= 64 (gpuwm/core/milbrandt2.py:185),
    # so a shallow configuration is priced 1,536 B per thread over what it
    # reserves.  It is NOT a LEVEL_SPECIALIZED_KERNEL_FRAMES case -- both
    # tiers are compiled into the shipped unit at fixed bounds rather than
    # recompiled at the configuration's nz -- and it needs no tiered entry
    # above the row, because milbrandt2's VERTICAL_LEVEL_BOUNDS = (3, 256)
    # refuses anything deeper than the 256 tier this 2,048 B measures.
    "milbrandt2": 2048,
    "morrison": 5120,
    "microphysics_validation": 0,
    # MYJ.  MEASURED on an RTX 5090 by the driver sweep this table is
    # checked against, at the kernel's compiled MYJ_KMAX = 128 tier: the
    # PBL translation unit holds 9,232 B and the Eta surface layer holds
    # none.  One thread owns one whole column and MYJPBL carries MIXLEN's
    # GM/GH/EL/Q2, DIFCOF's AKM/AKH, VDIFH's tridiagonal coefficients and
    # the species stack all at once, which is where the frame goes; the
    # surface layer is scalar per column, so it has nothing to hold.
    #
    # This closes the port's own open question ("nobody has measured the
    # local-memory cost of this kernel at production width").  By the
    # reservation model at the head of docs/kernel_local_memory_bounds.md
    # that frame reserves (9232 - 1024) * 1536 * 170 = 2,143,764,480 B
    # ~ 2.00 GiB on first launch, for the life of the process -- more than
    # SASE's and a third of GF's, and it is the price of selecting MYJ at
    # all, not of the domain size.  Like SASE's, it should be linear in
    # the compiled level bound and is therefore specializable; that is a
    # separate change, and until then the stated bound is the compiled
    # ceiling, which is the safe direction for a rail gate.
    "myjpbl": 9232,
    "myjsfc": 0,
    # SASE.  MEASURED on an RTX 5090 over all 29 kernels in the module at
    # the closure's compiled tier (SASE_KMAX = 128): the maximum is
    # sase_plume_vent_flux at 6,272 B; sase_moist_n2 5,120 B,
    # sase_vertical_channel 4,096 B, the two Thomas sweeps 3,072 B each,
    # and the remaining 24 kernels hold no local frame at all.  By the
    # reservation model at the head of docs/kernel_local_memory_bounds.md
    # this frame reserves (6272 - 1024) * 1536 * 170 = 1,370,357,760 B
    # ~ 1.28 GiB on first launch, for the life of the process.
    #
    # The frame is EXACTLY LINEAR in the compiled level bound -- measured
    # 1,568 / 3,136 / 6,272 B at SASE_KMAX 32 / 64 / 128, i.e. 49 B per
    # level with a zero intercept -- so this module is specializable the
    # way kf and refl are, and at a 49-level configuration the frame
    # would be 2,401 B and the reservation ~359 MiB.  It is NOT entered
    # in LEVEL_SPECIALIZED_KERNEL_FRAMES here, because that table prices
    # what the launcher actually compiles and this launcher compiles at
    # the fixed tier.  Specializing it is a real ~0.93 GiB saving and a
    # separate change; until then the stated bound is the compiled
    # ceiling, which is the safe direction for a rail gate.
    "sase": 6272,
    "mynn_pbl": 0,
    "mynn_surface": 0,
    "nest": 0,
    "nest_microphysics": 0,
    # sm_86 compiles Noah 48 B wider than sm_120 (224 against 176); the
    # ceiling is the sm_86 reading.  Well under the default stack either
    # way, so it reserves nothing on any card.
    "noah": 224,
    "noahmp_bareflux": 0,
    "noahmp_fluxprep": 0,
    "noahmp_leaves": 272,
    "noahmp_radiation": 0,
    "noahmp_sflx": 0,
    "noahmp_snow": 200,
    "noahmp_soilwater": 0,
    "noahmp_vegeflux": 0,
    "noahmp_vegprecip": 0,
    "noahmp_water": 224,
    "nssl2": 15504,
    "nssl2_diagnostics": 0,
    "nssl2_driver_support": 15504,
    # NVRTC 13.3.33 emits 216 B here where 13.0.48 emitted 112, at the
    # same sm_120 target: a COMPILER-build move, not an architecture one.
    "nssl2_fused_gs": 216,
    "nssl2_nucond": 0,
    "nssl2_qvexcess": 0,
    "openbc": 0,
    "pd_advection": 0,
    "refl": 18432,
    # WDM6's own reflectivity translation unit (see wdm6_refl.cu's
    # header for why it is not part of refl.cu).  63 B/level against
    # refl.cu's widest 72: WDM6 diagnoses one N0 array from nr where
    # Morrison carries three.
    "wdm6_refl": 16128,
    "rrtmg_lw": 0,
    "rrtmg_mcica_wrf": 0,
    # Another compiler-build move at a fixed sm_120: 0 B on NVRTC 13.0.48,
    # 40 B on 13.3.33.
    "rrtmgp_cloud": 40,
    "rrtmgp_gas": 512,
    "rrtmgp_mcica": 0,
    # The RRTMGP optimisation took this to 3,600 where it has been
    # re-measured (sm_86, NVRTC 13.0.48): rrtmgp_sw_2stream dropped
    # denom/dif_dn/dif_up.  The ceiling stays at the two sm_120 readings,
    # which predate that source change and cannot be re-taken -- over-priced
    # is the safe direction, and this module is far from the widest frame.
    "rrtmgp_rte": 5152,
    "rrtmgp_validation": 0,
    "ruc": 144,
    "saxpy": 0,
    "sfclay": 0,
    # Shin-Hong (bl_pbl_physics=11).  MEASURED 2026-08-03 on the RTX 5090
    # the same NVRTC + driver way as every other row: shinhong_column
    # holds 14,040 B (its per-thread column work arrays at the module's
    # fixed SHINHONG_KMAX = 128 tier, one thread per column -- the ysu.cu
    # shape, one tier up); shinhong_partition_probe and the validation
    # kernel hold no local frame.
    # Compiler-build move at a fixed sm_120: NVRTC 13.3.33 emits 17,160 B
    # against 13.0.48's 14,040, and the ceiling is the wider one.  On a
    # 170-SM card that is 0.74 GiB more backing store than the original
    # reading charged.
    "shinhong": 17160,
    "shinhong_validation": 0,
    "smag2d": 0,
    "spec_bdy": 0,
    "thompson": 11264,
    # mp_physics=28 translation units, measured 2026-07-31 the same way on
    # the same RTX 5090.  ``thompson_aerosol_sed``'s 9,216 B is its
    # 256-LEVEL cloud sedimentation variant
    # (``thompson_aa_cloud_sediment_256``); a run with nz <= 64 launches the
    # 64-level entry point and a much smaller frame, so this row is the
    # module CEILING exactly as the header above describes for Morrison.
    # ``thompson_aerosol_probe`` is a table row but never a priced module:
    # no ``physics_kernel_modules`` selector names it (it is the
    # device-helper oracle unit), and it is listed here only so the
    # driver-regeneration test covers every ``.cu`` in the directory.
    #
    # RE-MEASURE THESE SIX before the mp=28 wave closes.  They were taken
    # while a sibling package was still moving shared device helpers into
    # thompson_aerosol_common.cuh; sat/state/probe were measured against the
    # current header and warm/sed reproduce their values with the duplicate
    # definitions stripped, but ``thompson_aerosol_cold`` was measurable
    # only against the pre-move header.  Nothing has to be remembered for
    # that to be caught: ``test_the_recorded_local_frames_match_the_driver``
    # regenerates the whole table from NVRTC + the driver and fails loudly
    # on any row that moved.
    "thompson_aerosol_cold": 0,
    "thompson_aerosol_probe": 0,
    "thompson_aerosol_sat": 0,
    "thompson_aerosol_sed": 9216,
    # RE-MEASURED 2026-08-03 on the reference RTX 5090: 40, not 48.  The
    # module's only framed kernel is ``thompson_aa_init_profile``, and 40 is
    # what NVRTC 13.0 (system CUDA 13.0), NVRTC 12.9 (wheel, forced) and an
    # offline ``nvcc`` 13.0.48 all produce from the byte-identical source, at
    # every option variant this tree could plausibly use.  Nothing reproduces
    # 48 -- see the commit message.  The row cannot bind the reservation
    # either way: it is a MAX over the selected modules, and mp_physics=28
    # always co-selects ``thompson`` at 11,264 B.
    "thompson_aerosol_state": 40,
    # Architecture move at a fixed NVRTC 13.0.48: 0 B on sm_120, 112 B on
    # sm_86.
    "thompson_aerosol_warm": 112,
    "tke_budget": 0,
    "uh_diag": 0,
    "vert_interp": 768,
    # WDM6 (mp_physics=16).  MEASURED 2026-08-10 on the reference RTX 5090
    # the same NVRTC + driver way as every other row: the module's one
    # kernel, ``wdm6_column``, holds 9,776 B at the source's ``#ifndef``
    # default ``WDM6_KMAX = 64`` (88 registers).  One thread owns a whole
    # column, so the frame is that bound's ~30 per-thread work arrays.
    #
    # This row is the 64 TIER, not a ceiling: the launcher compiles the 80
    # tier for 65 <= nz <= 80 and that frame is WIDER (12,208 B measured).
    # :data:`WDM6_TIER_FRAME` carries the ladder, and the measurements
    # behind it -- exactly 152 B per level between the two rungs -- are
    # re-read off the driver by tests/test_kernel_local_bounds.py.
    "wdm6": 9776,
    # WSM6 (mp_physics=6).  This row is the 64 TIER, not a ceiling: the
    # launcher compiles the 80 tier for 65 <= nz <= 80 and that frame is
    # WIDER (9,008 B measured).  :data:`WSM6_TIER_FRAME` carries the ladder
    # and the measurements behind it -- exactly 112 B per level between the
    # two rungs -- and tests/test_kernel_local_bounds.py re-reads both off
    # the driver.
    "wsm6": 7216,
    # YSU (bl_pbl_physics = 1, the SHIPPED DEFAULT).  One thread still owns
    # one whole column, but the column arrays no longer live in the
    # per-thread local frame: gpuwm/core/kernels/ysu.cu keeps them in a
    # global workspace that gpuwm/core/ysu.py sizes to the threads IN
    # FLIGHT, because the driver sizes the local-memory backing store to
    # the threads the card can ever hold.
    #
    # MEASURED on node-1 (weather-node-1, RTX 5070 Ti, 70 SMs x 1,536,
    # sm_120) through the real launcher at nz=49: the frame went 9,232 -> 0
    # B on BOTH NVRTC 13.0 and 13.3, and the launch-time reservation went
    # 842.0 MiB -> nothing.  A zero frame reserves nothing on any card, and
    # it no longer moves with nz -- the workspace extent is a RUNTIME
    # argument, so a 49-level run holds 50 levels of arrays where the frame
    # had to hold 128.
    #
    # This row mattered more than any other: bl_pbl_physics = 1 is the
    # wizard's default (gpuwm/domain_wizard.py:714), so this was the
    # widest frame a BARE DEFAULT run launched, and every default run paid
    # the 842.0 MiB.
    # The workspace itself is priced by :func:`ysu_column_workspace_bytes`.
    "ysu": 0,
    "ysu_validation": 0,
}

#: The readings the ceiling above is made of, each naming its box, its
#: target architecture and its NVRTC build.
KERNEL_LOCAL_FRAME_RECORDINGS = _kernel_frame_recordings.\
    KERNEL_LOCAL_FRAME_RECORDINGS

#: Re-export so a caller that has a recording can price against it
#: without importing a second module.
KernelFrameRecording = _kernel_frame_recordings.KernelFrameRecording

if dict(KERNEL_MAX_LOCAL_SIZE_BYTES) != _kernel_frame_recordings.frame_ceiling():
    _low = sorted(
        module for module, frame
        in _kernel_frame_recordings.frame_ceiling().items()
        if KERNEL_MAX_LOCAL_SIZE_BYTES.get(module, -1) != frame)
    raise RuntimeError(
        "KERNEL_MAX_LOCAL_SIZE_BYTES must be exactly the element-wise "
        "maximum over gpuwm.core.kernel_frame_recordings: "
        f"{', '.join(_low)} disagree.  A row below a measurement "
        "under-charges the local-memory backing store on the platform "
        "that measured it")


def kernel_frame_recording_for(fingerprint) -> "KernelFrameRecording | None":
    """The frame recording taken on THIS compile platform, or ``None``.

    Matched on the two
    :func:`gpuwm.certify.compile_platform.compile_platform_fingerprint`
    keys that decide code generation -- the target architecture and the
    NVRTC build.  ``None`` means "nobody has measured this compiler on
    this architecture", which is the case the ceiling exists for.
    """

    return _kernel_frame_recordings.recording_for(fingerprint)


@dataclass(frozen=True)
class UnderPricedKernelFrame:
    """A module whose real frame is wider than the shipped row.

    ``unpriced_device_bytes`` is the whole point: the reservation is
    ``(frame - default stack) x resident-thread capacity``, so a frame
    delta is a device-memory delta the fit gate never charged.  A run
    admitted on the short number does not fail at the gate -- it fails
    later, out of memory, with nothing pointing back here.
    """

    module: str
    shipped_bytes: int
    observed_bytes: int
    unpriced_device_bytes: int


def under_priced_kernel_frames(
        observed, *, profile: "DeviceLocalMemoryProfile | None" = None
) -> dict[str, UnderPricedKernelFrame]:
    """Modules this compiler emits WIDER than :data:`KERNEL_MAX_LOCAL_SIZE_BYTES`.

    Empty is the only acceptable answer on any machine: over-pricing
    costs headroom, under-pricing breaches the rail.  A module the
    shipped table does not know about is reported too, priced against
    zero -- an unpriced module is the worst case of the same defect.
    """

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    capacity = profile.resident_thread_capacity
    over: dict[str, UnderPricedKernelFrame] = {}
    for module, frame in dict(observed).items():
        shipped = int(KERNEL_MAX_LOCAL_SIZE_BYTES.get(module, 0))
        if int(frame) <= shipped:
            continue
        over[module] = UnderPricedKernelFrame(
            module=module, shipped_bytes=shipped, observed_bytes=int(frame),
            unpriced_device_bytes=(int(frame) - shipped) * capacity)
    return over


@dataclass(frozen=True)
class LevelSpecializedFrame:
    """A kernel module whose per-thread local frame is compiled to ``nz``.

    ``refl.cu`` declares its per-thread column arrays against a
    ``#ifndef``-guarded compile-time bound, and its launcher specializes
    that bound to the field's own level count through
    ``gpuwm.core.kernels.get_kernel_int_defines``.  The column arrays are the
    only thing in the kernel that scales with the bound, so the frame the
    driver reports is ``bytes_per_level * levels`` rounded up to the local
    frame's 8-byte granularity.

    Measured on the RTX 5090 (driver 610.74, NVRTC 13.x, ``-std=c++17``),
    module maximum per bound:

      ==========  ============  ==========  ==========
      module      bound         frame       reserved
      ==========  ============  ==========  ==========
      refl        256 (ceiling)  18,432 B   4,334 MiB
      refl         49            3,528 B      626 MiB
      refl         30            2,160 B      286 MiB
      ==========  ============  ==========  ==========

    ``kf`` used to be the widest row in this table (24,064 B at its 128
    ceiling, 5,738 MiB reserved).  It is not here any more: its column
    arrays live in a global workspace and its frame stopped following
    ``nz``.

    ``tests/test_kernel_local_bounds.py`` re-measures every row against the
    driver, so a compiler or source change that breaks the linear form fails
    loudly instead of silently mispricing a rail gate.
    """

    module: str
    define: str
    unspecialized_levels: int
    bytes_per_level: int
    alignment_bytes: int = 8

    def frame_bytes(self, levels: int) -> int:
        levels = int(levels)
        if levels < 1:
            raise ValueError(
                f"{self.module}: level-specialized frame needs levels >= 1, "
                f"got {levels}")
        if levels > self.unspecialized_levels:
            raise ValueError(
                f"{self.module}: {levels} levels exceeds the {self.define} "
                f"ceiling of {self.unspecialized_levels}")
        raw = self.bytes_per_level * levels
        remainder = raw % self.alignment_bytes
        return raw if remainder == 0 else raw + self.alignment_bytes - remainder


#: The two modules whose local frame follows the configuration's ``nz``
#: exactly.  (``acoustic`` and ``wdm6`` follow a TIER of ``nz`` instead and
#: live in :class:`TieredKernelFrame` below.)  ``bytes_per_level`` is fixed
#: by construction against the unspecialized row of
#: :data:`KERNEL_MAX_LOCAL_SIZE_BYTES` (checked at import below), so the two
#: tables cannot drift apart.
LEVEL_SPECIALIZED_KERNEL_FRAMES: dict[str, LevelSpecializedFrame] = {
    # ``kf`` USED TO SIT HERE at 188 B/level, and it is gone rather than
    # re-fitted.  52 of its 54 column arrays moved into a global workspace
    # on 2026-08-21 (gpuwm/core/kernels/kf.cu), which left 188 B/level
    # wrong by a factor of 47, and -- because the two that stayed are the
    # only thing ``KF_KMAX`` still sizes, at 8 B/level -- made specializing
    # the bound worth 312 B of frame that reserves nothing either way.  So
    # gpuwm/core/kf.py stopped recompiling the module per level count, and
    # the ONE binary that can now launch holds a frame that does not move
    # with ``nz`` at all.  MEASURED on node-1 (RTX 5070 Ti, sm_120, NVRTC
    # 13.0.48): 512 B at the shipped KF_KMAX = 128 -- half the 1,024 B
    # default stack, so it reserves nothing on any card.  The flat row in
    # KERNEL_MAX_LOCAL_SIZE_BYTES is the whole model.
    "refl": LevelSpecializedFrame("refl", "REFL_KMAX", 256, 72),
    # 16-byte granularity, not refl.cu's 8: measured against the driver at
    # ten bounds from 256 down to 10 (63*n rounded up to 16 reproduces every
    # one).  At bound 1 the compiler eliminates the frame entirely and the
    # model over-prices by 64 B, which is the safe direction and the reason
    # tests/test_wdm6.py pins the realistic bounds rather than that one.
    "wdm6_refl": LevelSpecializedFrame(
        "wdm6_refl", "REFL_KMAX", 256, 63, alignment_bytes=16),
}

for _spec in LEVEL_SPECIALIZED_KERNEL_FRAMES.values():
    if (_spec.frame_bytes(_spec.unspecialized_levels)
            != KERNEL_MAX_LOCAL_SIZE_BYTES[_spec.module]):
        raise RuntimeError(
            f"{_spec.module}: level-specialized frame model disagrees with "
            "the driver-measured unspecialized frame")
del _spec


@dataclass(frozen=True)
class TieredKernelFrame:
    """A module whose local frame follows a compile-time TIER, not ``nz``.

    ``refl`` compiles exactly to the configuration's level count;
    ``acoustic`` instead picks the smallest of a short ladder of
    ``WPHI_MAX_LEV`` values (``gpuwm.core.acoustic.WPHI_LEVEL_TIERS``), so
    its frame is a step function of ``nz`` and is constant within a tier.

    The frame at the shipped tier is the driver-MEASURED row of
    :data:`KERNEL_MAX_LOCAL_SIZE_BYTES`.  Deeper tiers are priced from it by
    the one object the bound sizes -- the ``real rhs[WPHI_MAX_LEV]`` column
    that ``advance_w_phi``/``advance_w_phi_msf`` each declare -- at
    ``bytes_per_level`` per added full level.  That extrapolation is
    PROVISIONAL until ``tests/test_kernel_local_bounds.py`` reads the deeper
    tiers back off the driver; a register-allocation change at a deeper tier
    could add more, and under-pricing is the direction that hurts, so the
    device test is a gate rather than a formality.
    """

    module: str
    define: str
    shipped_tier: int
    bytes_per_level: int
    alignment_bytes: int = 8

    def frame_bytes(self, tier: int) -> int:
        tier = int(tier)
        if tier < self.shipped_tier:
            raise ValueError(
                f"{self.module}: {self.define}={tier} is below the shipped "
                f"tier {self.shipped_tier}")
        raw = (KERNEL_MAX_LOCAL_SIZE_BYTES[self.module]
               + self.bytes_per_level * (tier - self.shipped_tier))
        remainder = raw % self.alignment_bytes
        return raw if remainder == 0 else raw + self.alignment_bytes - remainder


#: ``acoustic.cu``'s ``WPHI_MAX_LEV`` sizes one FP32 column per thread.
ACOUSTIC_TIER_FRAME = TieredKernelFrame("acoustic", "WPHI_MAX_LEV", 129, 4)

#: ``wdm6.cu``'s ``WDM6_KMAX`` sizes the whole per-thread column stack, and
#: ``gpuwm/core/wdm6.py`` compiles it at one of two rungs
#: (``wdm6_constants.WDM6_KERNEL_LEVEL_TIERS`` = 64, 80) rather than at
#: ``nz``.  That is why WDM6 is priced HERE and not in
#: :data:`LEVEL_SPECIALIZED_KERNEL_FRAMES`: an nz-linear model would price
#: a 49-level WDM6 run at 7,488 B when the kernel it actually launches
#: holds 9,776 B, and under-pricing a rail gate is the direction that put a
#: run 1,630 MiB over.
#:
#: MEASURED on the reference RTX 5090 over sixteen bounds from 2 to 80.  The
#: two rungs the launcher can compile are exactly linear in the bound --
#: 9,776 B at 64 and 12,208 B at 80, 2,432 B over 16 levels = 152 B/level --
#: which is what this frame reproduces, exactly, at both.  (Between the
#: rungs the driver's frame wanders by up to 16 B against the same line; the
#: launcher never compiles there, and the model is a ceiling on that band,
#: which is the safe direction.)
WDM6_TIER_FRAME = TieredKernelFrame("wdm6", "WDM6_KMAX", 64, 152)

#: ``wsm6.cu``'s ``WSM6_KMAX`` sizes the whole per-thread column stack, and
#: ``gpuwm/core/wsm6.py`` compiles it at one of two rungs
#: (``wsm6_constants.WSM6_KERNEL_LEVEL_TIERS`` = 64, 80) rather than at
#: ``nz``.  Until 1.8.9 the flat row above WAS the price for every WSM6
#: configuration, so an ``nz = 72`` run -- the six shipped tornado-LES
#: configs -- was priced at the 64-tier 7,216 B while the kernel it
#: actually launches holds 9,008 B.  1,792 B/thread of backing store the
#: pool never reports, which is the direction that put a run 1,630 MiB
#: over.
#:
#: RE-MEASURED 2026-08-10 on the reference RTX 5090, NVRTC + driver,
#: eighteen bounds from 2 to 80: ``wsm6_column`` holds 7,216 B at the
#: source's ``#ifndef`` default of 64 and 9,008 B at 80 -- 1,792 B over 16
#: levels = exactly 112 B per level -- which this frame reproduces at both
#: rungs.  (Between the rungs the driver's frame sits up to 16 B below the
#: same line; the launcher never compiles there, and the model is a stated
#: ceiling on that band, which is the safe direction.)
WSM6_TIER_FRAME = TieredKernelFrame("wsm6", "WSM6_KMAX", 64, 112)

#: Kernel modules whose local frame CANNOT be measured at this checkout
#: because they do not compile alone: ``noahmp_driver.cu``,
#: ``noahmp_energy.cu``, ``noahmp_thermal.cu`` and ``noahmp_libm_slab.cu``
#: all fail NVRTC with ``identifier "r_pow" is undefined``, and
#: ``noahmp_glacier.cu`` with ``identifier "MU" is undefined`` -- they are
#: fragments that borrow ``noahmp_leaves.cu``'s single audited libm
#: transcription and compile only through
#: ``noahmp_kernel_sources.translation_unit_source``.  A configuration that
#: selects one as a standalone module is REFUSED rather than priced from a
#: guess -- see :func:`kernel_local_memory_bytes`.
#: The legacy-RRTMG members are the same shape of thing: ``rrtmg_sw.cu``,
#: ``rrtmg_lw_chain.cu`` and the ``rrtmg_lw_taugb*.cu`` band fragments
#: compile only through their own chained translation unit
#: (gpuwm/core/rrtmg_lw.py / rrtmg_sw.py), never as standalone NVRTC
#: modules, and no ``physics_kernel_modules`` selector row names them
#: directly -- a legacy 4/4 radiation request prices its TRANSIENT VRAM
#: through the call-peak envelope (``legacy_radiation_vram_bytes``) and
#: its LOCAL-MEMORY frame through the driver-measured composite rows in
#: :data:`CHAINED_TRANSLATION_UNIT_FRAMES`, which cover these fragments
#: as the translation units they actually launch in.
UNMEASURED_KERNEL_MODULES = frozenset({
    "noahmp_driver", "noahmp_energy", "noahmp_thermal", "noahmp_libm_slab",
    "noahmp_glacier",
    "rrtmg_sw", "rrtmg_lw_chain", "rrtmg_lw_taugb02_10_11_12",
    "rrtmg_lw_taugb03_05", "rrtmg_lw_taugb06_09", "rrtmg_lw_taugb13_16"})


@dataclass(frozen=True)
class ChainedTranslationUnitFrame:
    """A chained translation unit's driver-measured widest local frame.

    The legacy-RRTMG kernels never load per ``.cu`` file: the LW chain
    concatenates ``rrtmg_lw.cu`` + ``rrtmg_lw_chain.cu`` + the four
    ``rrtmg_lw_taugb*.cu`` band fragments into ONE NVRTC translation unit
    (gpuwm/core/rrtmg_lw.py section 10), and the SW composition compiles
    ``rrtmg_sw.cu`` through its own unit (gpuwm/core/rrtmg_sw.py).  The
    fragments therefore stay in :data:`UNMEASURED_KERNEL_MODULES` --
    selecting one standalone still refuses -- while the unit that DOES
    launch carries the frame the driver measured for it.

    ``covers`` names the unmeasured fragments this measurement subsumes;
    the import-time checks below keep the two tables consistent.
    """

    module: str
    max_local_size_bytes: int
    covers: frozenset[str]


#: Measured 2026-07-27 (cupy 14.0.1 NVRTC on sm_120, RTX 5090) over every
#: kernel of each chained unit -- the record and its drift-bound re-audit
#: live in ``tests/test_rrtmg_lw_cuda.py`` (``LOCAL_FRAME_BOUNDS``); the
#: prose record is docs/rrtmg_legacy_integration.md section 6:
#:
#: * LW unit: ``rlw_rtrn_march`` 2,048 B/thread (exactly its four
#:   128-float per-thread work arrays atrans/atot/bbugas/bbutot at the
#:   fixed RLW_MAXLAY = 128 bound -- NOT nz-specialized), ``rlw_cldprmc``
#:   64 B, every other kernel 0 B.  Machine-wide store ~510 MiB, of which
#:   the 1,024 B default-stack half already sits in CUDA_CONTEXT_BYTES.
#: * SW unit: every kernel 0 B after the spcvmc workspace restructure
#:   (the old ~1.65 GiB hidden lmem reservation became pool-priced
#:   transient VRAM, priced by ``legacy_radiation_vram_bytes``).
#:
#: A re-measure past these values must move this table in the same diff.
CHAINED_TRANSLATION_UNIT_FRAMES: dict[str, ChainedTranslationUnitFrame] = {
    "rrtmg_lw_legacy_chain": ChainedTranslationUnitFrame(
        module="rrtmg_lw_legacy_chain",
        max_local_size_bytes=2048,
        covers=frozenset({
            "rrtmg_lw_chain", "rrtmg_lw_taugb02_10_11_12",
            "rrtmg_lw_taugb03_05", "rrtmg_lw_taugb06_09",
            "rrtmg_lw_taugb13_16"})),
    "rrtmg_sw_legacy": ChainedTranslationUnitFrame(
        module="rrtmg_sw_legacy",
        max_local_size_bytes=0,
        covers=frozenset({"rrtmg_sw"})),
}

_seen_covers: set[str] = set()
for _tu_name, _tu in CHAINED_TRANSLATION_UNIT_FRAMES.items():
    if _tu_name != _tu.module or _tu_name in KERNEL_MAX_LOCAL_SIZE_BYTES:
        raise RuntimeError(
            f"{_tu_name}: chained-unit frame must carry its own key and "
            "must not shadow a standalone-measured module")
    if not _tu.covers <= UNMEASURED_KERNEL_MODULES:
        raise RuntimeError(
            f"{_tu_name}: covers must name only fragments that cannot be "
            "measured standalone (UNMEASURED_KERNEL_MODULES); anything "
            "else is priced from KERNEL_MAX_LOCAL_SIZE_BYTES")
    if _seen_covers & _tu.covers:
        raise RuntimeError(
            f"{_tu_name}: a fragment may be covered by exactly one "
            "chained translation unit")
    _seen_covers |= _tu.covers
del _seen_covers, _tu_name, _tu

#: Kernel modules every integration loads regardless of the physics
#: selectors: dynamics, diffusion, nesting, lateral boundaries, health and
#: output diagnostics.  Their maximum local frame is 768 B (``vert_interp``),
#: below the default stack limit, so this set reserves nothing at all.
CORE_KERNEL_MODULES = frozenset({
    "acoustic", "advection", "coriolis_map", "diagnostics", "diff6",
    "diffusion", "dycore", "health", "lbc_flow", "lbc_state", "nest",
    "nest_microphysics", "openbc", "pd_advection", "saxpy", "smag2d",
    "spec_bdy", "tke_budget", "vert_interp"})

#: ``mp_physics`` -> the kernel modules that scheme launches.  Keys are
#: exactly ``gpuwm/config.py``'s accepted set (0, 1, 6, 8, 9, 10, 16, 18,
#: 28, 50), which is priced here ahead of its driver dispatch so a run can
#: never reach ``kernel_local_frame_bytes`` with an unpriced selector.
#: That "exactly" is now MEASURED rather than asserted in prose --
#: tests/test_composition_pricing.py walks every composition the loader
#: accepts and prices each one, which is what caught mp=50 and MYJ
#: shipping accepted-but-unpriceable into the 1.9 assembly.
_MICROPHYSICS_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("kessler", "microphysics_validation"),
    6: ("wsm6", "microphysics_validation"),
    8: ("thompson", "microphysics_validation"),
    # Milbrandt-Yau.  All seven kernels live in the one ``milbrandt2``
    # translation unit (gpuwm/core/milbrandt2.py:179-187), and the scheme
    # takes the NATIVE validation path -- ``accept_microphysics`` routes
    # everything but mp=18 through ``_validate_native_microphysics``
    # (gpuwm/core/physics.py:1753-1765) -- so it launches the shared
    # validator exactly as Kessler/WSM6/Thompson/Morrison do.  ``refl`` is
    # deliberately NOT priced for this selector and 9 is absent from
    # _REFLECTIVITY_MICROPHYSICS below: mp=9 fills the REFL_10CM slot from
    # its own diagnostics kernel and only stashes the array
    # (gpuwm/core/milbrandt2.py:291-296), so no refl kernel is ever loaded.
    9: ("milbrandt2", "microphysics_validation"),
    10: ("morrison", "microphysics_validation"),
    # WDM6 launches ONE column kernel plus the shared moisture validation
    # pass; its cold half is transcribed inside wdm6.cu rather than shared
    # with wsm6.cu, so mp=16 never loads the WSM6 translation unit.
    16: ("wdm6", "microphysics_validation"),
    18: ("nssl2", "nssl2_driver_support", "nssl2_diagnostics",
         "nssl2_fused_gs", "nssl2_nucond", "nssl2_qvexcess"),
    # Thompson aerosol-aware.  ``thompson`` is genuinely launched: mp=28
    # reuses the frozen mp=8 ice/snow/graupel/rain sedimentation and classic
    # graupel-number launchers unchanged.  ``thompson_aerosol_probe`` is
    # excluded on purpose -- it exists only for the device-helper oracle
    # gate and no forecast path loads it, so pricing it here would reserve
    # local memory for a module the run never compiles.
    28: ("thompson", "thompson_aerosol_state", "thompson_aerosol_sat",
         "thompson_aerosol_cold", "thompson_aerosol_warm",
         "thompson_aerosol_sed"),
    # P3 one-category.  The row is ONE module, and the reason is the
    # scheme's execution model rather than a gap in this table: P3 is a
    # HOST float32 transcription.  ``gpuwm/core/p3.py:apply`` pulls the
    # prognostic slabs off the device with ``_host`` (:1683-1691), runs
    # ``mp_p3_wrapper_wrf`` in NumPy on the host, and writes the results
    # back with ``_from_slab`` (:1701-1705).  There is no
    # ``gpuwm/core/kernels/p3.cu``; the module directory has none, and
    # ``p3.py`` contains no ``get_kernel`` call and no CuPy import outside
    # ``_as_backend``'s host->device copy.  So P3 launches NOTHING of its
    # own and reserves no per-thread local frame of its own, and pricing
    # a P3-named module here would reserve backing store for a
    # translation unit that never compiles.
    #
    # ``microphysics_validation`` IS launched, on the same footing as
    # Kessler/WSM6/Thompson/Morrison/Milbrandt-Yau: mp=50 has a
    # five-slot canonical row (gpuwm/core/physics.py:603-608) and P3's
    # diagnostics are exactly those canonical scratch arrays, so
    # ``accept_microphysics`` takes the native path and launches the
    # shared validator (gpuwm/core/microphysics.py:385-386).  Its frame
    # is 0 B, which is the honest price and not a placeholder.
    #
    # 50 is deliberately ABSENT from _REFLECTIVITY_MICROPHYSICS below,
    # for mp=9's reason: P3 computes REFL_10CM inside its own host pass
    # and ``stash_refl_10cm`` only parks the array on the driver
    # (gpuwm/core/refl.py:651-663), so no ``refl`` kernel is ever loaded.
    50: ("microphysics_validation",),
}

#: ``mp_physics`` values with a REFL_10CM path in ``gpuwm/core/refl.py``.
#: The ``refl`` module is priced only when a history frame can come due
#: during the run (:func:`refl_diagnostic_reachable`), because the kernel is
#: launched from the ``refl_10cm_due`` branch of the microphysics drivers and
#: from nowhere else.
#: 28 is included: mp=28 routes REFL_10CM through the SAME Thompson
#: reflectivity kernel as mp=8 (calc_refl10cm takes no droplet number and
#: never re-reads rc), so the ``refl`` module is loaded on exactly the same
#: cadence.  Pricing it here before ``gpuwm/core/refl.py`` admits 28 is the
#: safe direction -- an over-priced rail refuses a run that would have fit,
#: an under-priced one lets a run breach the budget.
_REFLECTIVITY_MICROPHYSICS = frozenset({1, 6, 8, 10, 16, 18, 28})

_CUMULUS_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("kf", "kf_validation"),
    # One translation unit carries all of GFDRV (deep, shallow, driver).
    3: ("gf",)}
_PBL_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("ysu", "ysu_validation"),
    # MYJ (Mellor-Yamada-Janjic).  ONE module: gpuwm/core/myjpbl.py's only
    # kernel load is ``get_kernel("myjpbl", "myjpbl_column")`` (:111), one
    # thread per column, and the scheme does its own implicit vertical
    # diffusion inside that kernel rather than handing tendencies to a
    # separate diffusion launcher (gpuwm/core/physics.py:2574-2582).
    #
    # No validation module, and that is a measured absence rather than an
    # oversight: YSU and Shin-Hong each launch a batched device validator
    # (``ysu_validation``, ``shinhong_validation``), while MYJ's
    # ``validate_myj_pbl_outputs`` is a host reduction over the returned
    # fields (gpuwm/core/physics.py:2625) and compiles nothing.
    #
    # The frame this selects is the widest of any PBL in the tree --
    # KERNEL_MAX_LOCAL_SIZE_BYTES["myjpbl"] = 9,232 B at the compiled
    # MYJ_KMAX = 128 tier, ~2.00 GiB of reservation on the reference card
    # by the model at the head of docs/kernel_local_memory_bounds.md.  It
    # is a flat row and not a tiered one because gpuwm/core/myjpbl.py
    # compiles at the source's fixed bound rather than at nz, so it is the
    # price of selecting MYJ at all; the row's own comment records that
    # specializing it is a real saving and a separate change.
    MYJ_PBL_SCHEME: ("myjpbl",),
    5: ("mynn_pbl",),
    # Shin-Hong launches its column kernel plus its own batched output
    # validator, the YSU pair's shape (gpuwm/core/shinhong.py).
    11: ("shinhong", "shinhong_validation"),
    SASE_PBL_SCHEME: ("sase",)}
_SURFACE_LAYER_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("sfclay",),
    # Eta similarity, MYJ's own surface layer.  gpuwm/core/myjsfc.py loads
    # exactly ``get_kernel("myjsfc", "myjsfc_column")`` (:102) and nothing
    # else; the module measures 0 B because the scheme is scalar per
    # column and holds no per-thread stack.  It is a separate row from the
    # PBL's on purpose: sf_sfclay_physics is its own selector, and
    # gpuwm.config.validate_myj_pairing is what ties the two values
    # together -- this table prices whichever the resolved config names.
    MYJ_SFCLAY_SCHEME: ("myjsfc",),
    5: ("mynn_surface",), 91: ("sfclay",)}
_LAND_SURFACE_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (),
    2: ("noah",),
    3: ("ruc",),
    4: ("noahmp_bareflux", "noahmp_driver", "noahmp_energy",
        "noahmp_fluxprep", "noahmp_leaves", "noahmp_radiation",
        "noahmp_sflx", "noahmp_snow", "noahmp_soilwater", "noahmp_thermal",
        "noahmp_vegeflux", "noahmp_vegprecip", "noahmp_water"),
}
#: ``ra_physics = 4`` is two implementations behind one selector value;
#: the row here is the modern RTE+RRTMGP set and
#: :func:`_radiation_44_kernel_modules` dispatches on
#: ``RunConfig.ra_rrtmg_variant`` -- the legacy variant launches the
#: chained translation units + the device McICA twin instead, and an
#: unknown variant refuses rather than pricing either row (fail-closed).
_RADIATION_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (),
    4: ("rrtmgp_cloud", "rrtmgp_gas", "rrtmgp_mcica", "rrtmgp_rte"),
    90: (),
}

#: What a legacy 4/4 selection launches: the two chained translation
#: units (:data:`CHAINED_TRANSLATION_UNIT_FRAMES`) and the standalone
#: ``rrtmg_mcica_wrf.cu`` device McICA twin (measured 0 B like every
#: other standalone row).
_RRTMG_LEGACY_KERNEL_MODULES: tuple[str, ...] = (
    "rrtmg_mcica_wrf", "rrtmg_lw_legacy_chain", "rrtmg_sw_legacy")


def _radiation_44_kernel_modules(run) -> tuple[str, ...]:
    """Kernel modules a resolved 4/4 radiation request launches."""
    from gpuwm.physics_compat import (
        RRTMG_VARIANT_LEGACY, RRTMG_VARIANT_RTE_RRTMGP, rrtmg_variant)
    variant = rrtmg_variant(run)
    if variant == RRTMG_VARIANT_RTE_RRTMGP:
        return _RADIATION_KERNEL_MODULES[4]
    if variant == RRTMG_VARIANT_LEGACY:
        return _RRTMG_LEGACY_KERNEL_MODULES
    raise ValueError(
        f"no kernel-module row for ra_physics=4 with "
        f"ra_rrtmg_variant={variant!r}; add one to "
        "gpuwm/core/preflight.py before this configuration can be "
        "priced or gated.")

_SELECTOR_TABLES = (
    ("mp_physics", _MICROPHYSICS_KERNEL_MODULES),
    ("cu_physics", _CUMULUS_KERNEL_MODULES),
    ("bl_pbl_physics", _PBL_KERNEL_MODULES),
    ("sf_sfclay_physics", _SURFACE_LAYER_KERNEL_MODULES),
    ("sf_surface_physics", _LAND_SURFACE_KERNEL_MODULES),
    ("ra_physics", _RADIATION_KERNEL_MODULES),
)


def refl_diagnostic_reachable(exp: ExperimentConfig) -> bool:
    """Can a history frame carrying REFL_10CM come due inside the run?

    The reflectivity kernels are launched from the microphysics drivers'
    ``refl_10cm_due`` branch, which follows the history cadence.  A run
    shorter than every domain's history interval never reaches one -- both
    traced forecasts behind this model wrote their t=0 frames and never
    launched a ``refl`` kernel.  Anything else prices it.
    """
    intervals = [float(getattr(dc, "history_interval_s", 0.0) or 0.0)
                 for dc in exp.domains]
    if not intervals or min(intervals) <= 0.0:
        return True
    return float(exp.run_seconds) >= min(intervals)


def domain_kernel_modules(dc: DomainConfig, *,
                          prices_refl: bool) -> frozenset[str]:
    """Kernel modules ONE domain's selectors can launch (no core modules).

    FAIL-CLOSED: a selector value with no table entry raises rather than
    quietly pricing zero.  A new scheme that reaches production without a row
    here stops ``gpuwm check``; it does not slip past it.
    """
    modules: set[str] = set()
    for selector, table in _SELECTOR_TABLES:
        value = int(getattr(dc.run, selector))
        if value not in table:
            raise ValueError(
                f"no kernel-module row for {selector}={value} on "
                f"d{dc.grid_id:02d}; add one to gpuwm/core/preflight.py "
                "before this configuration can be priced or gated.")
        if selector == "ra_physics" and value == 4:
            # One selector value, two implementations: dispatch on the
            # trajectory-bound ra_rrtmg_variant (fail-closed inside).
            modules.update(_radiation_44_kernel_modules(dc.run))
        else:
            modules.update(table[value])
    if prices_refl and int(dc.run.mp_physics) in _REFLECTIVITY_MICROPHYSICS:
        # mp=16 launches its reflectivity from its OWN translation unit, so
        # a WDM6 domain must never reserve refl.cu's wider frame and a
        # non-WDM6 domain must never reserve wdm6_refl.cu's.
        modules.add("wdm6_refl" if int(dc.run.mp_physics) == 16 else "refl")
    return frozenset(modules)


def physics_kernel_modules(exp: ExperimentConfig) -> frozenset[str]:
    """Every kernel module the experiment's selectors can launch."""
    prices_refl = refl_diagnostic_reachable(exp)
    modules = set(CORE_KERNEL_MODULES)
    for dc in exp.domains:
        modules |= domain_kernel_modules(dc, prices_refl=prices_refl)
    return frozenset(modules)


def kernel_local_frame_bytes(exp: ExperimentConfig) -> dict[str, int]:
    """Widest per-thread local frame each launched module compiles to.

    A module launched by several domains is priced at the deepest of them:
    the specialized frame is monotone in the level count, and the driver's
    reservation is a maximum over everything the process ever launches.
    """
    modules = physics_kernel_modules(exp)
    unmeasured = sorted(modules & UNMEASURED_KERNEL_MODULES)
    if unmeasured:
        # THE WAY THROUGH, named (1.8.8 refusal sweep).  This message
        # ended at "Refusing to guess", which is honest and is a dead
        # end: there is no flag that skips the local-memory pricing --
        # `gpuwm check` prices it on every run, and --alloc only ADDS a
        # device measurement on top.  The one exit that exists is a
        # land-surface scheme whose modules ARE measured, so it is
        # named, along with what taking it costs.  The underlying gap is
        # a missing CHAINED_TRANSLATION_UNIT_FRAMES row for the Noah-MP
        # translation unit, not broken code: these fragments compile
        # only through noahmp_kernel_sources.translation_unit_source,
        # exactly as the legacy-RRTMG fragments do, and those have a
        # driver-measured composite row while Noah-MP does not.
        raise ValueError(
            "cannot price the local-memory reservation: "
            f"{', '.join(unmeasured)} do not compile at this checkout "
            "(NVRTC: identifier \"r_pow\" is undefined), so their per-thread "
            "local frame has never been measured.  Refusing to guess.  "
            "No flag skips this pricing -- the memory preflight is what "
            "the estimate is made of.  If Noah-MP is not the point of "
            "this run, select sf_surface_physics = 2 (Noah), whose "
            "kernels are measured and price normally; if it IS the "
            "point, this configuration needs a measured composite frame "
            "for the Noah-MP translation unit "
            "(gpuwm/core/preflight.py CHAINED_TRANSLATION_UNIT_FRAMES, "
            "the same treatment the legacy-RRTMG chain already has).")
    missing = sorted(modules - set(KERNEL_MAX_LOCAL_SIZE_BYTES)
                     - set(CHAINED_TRANSLATION_UNIT_FRAMES))
    if missing:
        raise ValueError(
            "no measured local frame for kernel module(s) "
            f"{', '.join(missing)}; regenerate KERNEL_MAX_LOCAL_SIZE_BYTES.")

    prices_refl = refl_diagnostic_reachable(exp)
    frames = {module: (
                  CHAINED_TRANSLATION_UNIT_FRAMES[module].max_local_size_bytes
                  if module in CHAINED_TRANSLATION_UNIT_FRAMES
                  else KERNEL_MAX_LOCAL_SIZE_BYTES[module])
              for module in modules
              if module not in LEVEL_SPECIALIZED_KERNEL_FRAMES}
    for dc in exp.domains:
        levels = int(dc.run.nz)
        for module in domain_kernel_modules(dc, prices_refl=prices_refl):
            specialized = LEVEL_SPECIALIZED_KERNEL_FRAMES.get(module)
            if specialized is None:
                continue
            frame = specialized.frame_bytes(levels)
            if frame > frames.get(module, -1):
                frames[module] = frame
    # ``acoustic`` is in CORE_KERNEL_MODULES, so it is priced above at its
    # shipped-tier row; a domain deeper than the shipped tier raises that
    # row to its own tier.  The launcher owns the ladder (it is the thing
    # that compiles the tier) and imports no CuPy at module scope, so this
    # still prices a configuration on a host with no device.
    from gpuwm.core.acoustic import wphi_level_tier

    for dc in exp.domains:
        tier = wphi_level_tier(int(dc.run.nz))
        frame = ACOUSTIC_TIER_FRAME.frame_bytes(tier)
        if frame > frames.get(ACOUSTIC_TIER_FRAME.module, -1):
            frames[ACOUSTIC_TIER_FRAME.module] = frame
    # ``wdm6`` and ``wsm6`` are the CONDITIONAL tiered modules, so
    # unlike acoustic it is priced PER DOMAIN THAT SELECTS IT.  Both halves
    # of that matter: a 40-level WDM6 child beside a 100-level WSM6 parent
    # must be priced at WDM6's own 64 tier, not raised to a tier WDM6 has no
    # kernel for -- and the parent's 100 levels must not reach WDM6's
    # level-bound refusal at all, because the parent never launches it.  The
    # ladder lives in the CuPy-free constants leaf, so this still prices on
    # a host with no device.
    from gpuwm.core.wdm6_constants import wdm6_level_tier

    for dc in exp.domains:
        if WDM6_TIER_FRAME.module not in domain_kernel_modules(
                dc, prices_refl=prices_refl):
            continue
        frame = WDM6_TIER_FRAME.frame_bytes(wdm6_level_tier(int(dc.run.nz)))
        if frame > frames.get(WDM6_TIER_FRAME.module, -1):
            frames[WDM6_TIER_FRAME.module] = frame
    # ``wsm6`` prices the same way, and the same two halves matter: a
    # 40-level WSM6 child beside a 100-level Morrison parent takes WSM6's
    # own 64 tier rather than a tier WSM6 has no kernel for, and the
    # parent's 100 levels never reach WSM6's level-bound refusal on behalf
    # of a domain that does not launch it.
    from gpuwm.core.wsm6_constants import wsm6_level_tier

    for dc in exp.domains:
        if WSM6_TIER_FRAME.module not in domain_kernel_modules(
                dc, prices_refl=prices_refl):
            continue
        frame = WSM6_TIER_FRAME.frame_bytes(wsm6_level_tier(int(dc.run.nz)))
        if frame > frames.get(WSM6_TIER_FRAME.module, -1):
            frames[WSM6_TIER_FRAME.module] = frame
    return frames


def kernel_local_memory_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """The launch-time local-memory backing store the experiment reserves.

    One allocation per process, sized by the largest per-thread local frame
    among the kernels the configuration launches -- so this is a maximum, not
    a sum, and it does not grow with domain count.  It DOES grow with the
    level count, because the two widest frames in the tree (``kf`` and
    ``refl``) compile their column arrays to ``nz``.
    """
    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    widest = max(kernel_local_frame_bytes(exp).values(), default=0)
    return profile.reservation_bytes(widest)


def non_pool_device_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device residency of a gpuwm process that the CuPy pool never reports:
    the CUDA context plus the local-memory backing store.

    Both terms are properties of the DEVICE, so both are read off the
    profile.  The context used to be a flat
    :data:`CUDA_CONTEXT_BYTES` -- one card's 2026-07-26 reading charged
    to every card on every platform, which measured 48 MiB high on a
    5070 Ti and 215 MiB LOW on a Linux 5090 (task 206).
    """
    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    return (profile.cuda_context_bytes
            + kernel_local_memory_bytes(exp, profile=profile)
            + column_workspace_bytes(exp, profile=profile))


def kf_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the Kain-Fritsch column workspace holds, or zero.

    This is the term that REPLACED KF's local-memory reservation.
    ``gpuwm/core/kernels/kf.cu`` keeps 52 of ``kf_column``'s 54 column
    arrays in a global workspace instead of the per-thread local frame,
    and ``gpuwm/core/kf.py`` sizes that workspace to the columns it keeps
    in flight -- ``SMs x KF_TILE_BLOCKS_PER_SM x _TPB`` -- rather than to
    the resident-thread capacity the driver would have charged.  That
    ratio is the whole saving: MEASURED on node-1 (RTX 5070 Ti, 70 SMs x
    1,536) at nz = 49, 840.0 MiB of reservation became 174.2 MiB of
    workspace.

    Unlike GF's, this term follows ``nz`` LINEARLY and not a compiled
    tier: the workspace extent is the runtime level count, which is a
    saving the compile-time frame could never give.

    Priced here, beside the context and the backing store, for the same
    reason both of those are: it is a property of the DEVICE and of the
    level count, not of the grid.  Bounded by the column count -- a
    domain smaller than one tile never allocates a whole tile.

    Zero for every configuration that does not select ``cu_physics = 1``.
    """
    from gpuwm.core.kf import (
        KF_TILE_BLOCKS_PER_SM, _TPB, kf_workspace_floats)

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    tile_cap = profile.multiprocessor_count * KF_TILE_BLOCKS_PER_SM * _TPB
    worst = 0
    for dc in exp.domains:
        if int(dc.run.cu_physics) != 1:
            continue
        columns = min(int(dc.run.nx) * int(dc.run.ny), tile_cap)
        worst = max(worst, kf_workspace_floats(int(dc.run.nz), columns) * 4)
    return int(worst)


def column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes held by the column workspaces, together.

    A SUM and not a maximum, deliberately.  The three workspaces are
    ordinary allocations owned by three different launchers, and a
    configuration holds every one whose scheme it selects at the same
    time -- unlike the local-memory backing store above, which is one
    per-context allocation the driver sizes to the widest frame.  Each
    term is already zero for a configuration that does not select its
    scheme, so this costs nothing where only one is reachable.
    """
    return (gf_column_workspace_bytes(exp, profile=profile)
            + kf_column_workspace_bytes(exp, profile=profile)
            + ysu_column_workspace_bytes(exp, profile=profile))


def gf_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the Grell-Freitas column workspace holds, or zero.

    This is the term that REPLACED most of GF's local-memory reservation.
    ``gpuwm/core/kernels/gf.cu`` keeps GFDRV's column arrays in a global
    workspace instead of the per-thread local frame, and
    ``gpuwm/core/gf.py`` sizes that workspace to the columns it keeps in
    flight -- ``SMs x GF_TILE_BLOCKS_PER_SM x GF_BLOCK`` -- rather than to
    the resident-thread capacity the driver would have charged.  That
    ratio is the whole saving: MEASURED on node-1 (RTX 5070 Ti, 70 SMs x
    1,536) at the nz<=40 tier, 2,200.0 MiB of reservation became 422.1
    MiB of workspace.

    It is priced here, beside the context and the backing store, for the
    same reason both of those are: it is a property of the DEVICE and of
    the level count, not of the grid.  It is bounded by the column count
    -- a domain smaller than one tile never allocates a whole tile.

    Zero for every configuration that does not select ``cu_physics = 3``.
    """
    from gpuwm.core.gf import (
        GF_BLOCK, GF_TILE_BLOCKS_PER_SM, gf_workspace_floats)

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    tile_cap = (profile.multiprocessor_count * GF_TILE_BLOCKS_PER_SM
                * GF_BLOCK)
    worst = 0
    for dc in exp.domains:
        if int(dc.run.cu_physics) != 3:
            continue
        columns = min(int(dc.run.nx) * int(dc.run.ny), tile_cap)
        worst = max(worst, gf_workspace_floats(int(dc.run.nz), columns) * 4)
    return int(worst)


def ysu_column_workspace_bytes(
        exp: ExperimentConfig, *,
        profile: DeviceLocalMemoryProfile | None = None) -> int:
    """Device bytes the YSU column workspace holds, or zero.

    The term that REPLACED YSU's local-memory reservation, and the one a
    BARE DEFAULT run pays, because ``bl_pbl_physics = 1`` is the wizard's
    default.  ``gpuwm/core/kernels/ysu.cu`` keeps the scheme's column
    arrays in a global workspace instead of the per-thread local frame,
    and ``gpuwm/core/ysu.py`` sizes it to the columns it keeps in flight
    -- ``SMs x YSU_TILE_BLOCKS_PER_SM x YSU_BLOCK`` -- rather than to the
    resident-thread capacity the driver charged.

    MEASURED on node-1 (weather-node-1, RTX 5070 Ti, 70 SMs x 1,536,
    sm_120) at nz=49 and 102,400 columns: an 842.0 MiB reservation became
    123.0 MiB of workspace, for +1.8% on the kernel's wall clock.

    Unlike the frame it replaced, this term follows ``nz`` rather than the
    kernel's 128-level bound: the extent is a runtime argument, so a
    49-level run holds 50 levels of arrays where the frame held 128.

    Zero for every configuration that does not select
    ``bl_pbl_physics = 1``.
    """
    # physics_inventory, not ysu: the launcher imports cupy at module
    # scope and this pricing must be readable on installs with no GPU
    # runtime (the wizard's estimator runs here).
    from gpuwm.core.physics_inventory import (
        YSU_BLOCK, YSU_TILE_BLOCKS_PER_SM, ysu_workspace_floats)

    profile = MEASURED_LOCAL_MEMORY_PROFILE if profile is None else profile
    tile_cap = (profile.multiprocessor_count * YSU_TILE_BLOCKS_PER_SM
                * YSU_BLOCK)
    worst = 0
    for dc in exp.domains:
        if int(dc.run.bl_pbl_physics) != 1:
            continue
        columns = min(int(dc.run.nx) * int(dc.run.ny), tile_cap)
        worst = max(worst, ysu_workspace_floats(int(dc.run.nz), columns) * 4)
    return int(worst)


# ---------------------------------------------------------------------------
# Itemized shape formulas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryItem:
    """One named device allocation: shape formula result + byte width."""

    name: str
    category: str  # state | physics | scratch | lbc | nest | sase
    #                | transient
    shape: tuple[int, ...]
    itemsize: int = 4
    dtype: str = "float32"

    @property
    def nbytes(self) -> int:
        n = self.itemsize
        for extent in self.shape:
            n *= int(extent)
        return n


def _items(category: str, shapes: dict[str, tuple[int, ...]],
           itemsize: int = 4) -> tuple[MemoryItem, ...]:
    return tuple(MemoryItem(name, category, tuple(shape), itemsize)
                 for name, shape in shapes.items())


def _nest_items(shapes: dict[str, tuple[int, ...]],
                dtypes: dict[str, str]) -> tuple[MemoryItem, ...]:
    """F4 nest items with their semantic dtype recorded explicitly."""
    if shapes.keys() != dtypes.keys():
        raise RuntimeError("nest shape/dtype registries drifted")
    return tuple(MemoryItem(name, "nest", tuple(shape), 4, dtypes[name])
                 for name, shape in shapes.items())


@dataclass(frozen=True)
class PhysicsArrayLifetime:
    """Closed-world lifetime proof for one exact physics-array set."""

    names: tuple[str, ...]
    disposition: str
    evidence: str
    proof: str


_YSU_3D = ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
           "exch_h", "exch_m")
_YSU_2D = ("hpbl", "kpbl", "wstar", "delta", "topdown_radsum",
           "wstar3_2", "cloudflg")
_TENDENCY_COMPONENTS = ("ru", "rv", "rtheta", "rqv", "rqc", "rqr",
                        "rqi", "rqs")
_MICROPHYSICS_COMPONENTS = ("rainnc", "rainncv", "sr", "snownc",
                            "snowncv", "graupelnc", "graupelncv", "hailnc",
                            "hailncv")

#: Persistent-physics counterpart to ``SCRATCH_SLOT_LIFETIME_AUDIT``.
#: Exact names make the three reclamations closed-world: a future diagnostic
#: or tendency component receives no aliasing without a new reviewed row.
PHYSICS_ARRAY_LIFETIME_AUDIT = (
    PhysicsArrayLifetime(
        tuple(f"last_ysu/{name}" for name in (*_YSU_3D, *_YSU_2D)),
        "transient_when_bldt_zero", "gpuwm/core/physics.py:862-893",
        "field copies and coupling are the only readers; bldt=0 releases "
        "the dict after them, while every positive cadence retains it"),
    PhysicsArrayLifetime(
        tuple(f"microphysics/{name}" for name in _MICROPHYSICS_COMPONENTS),
        "aliases_serialized_scratch", "gpuwm/core/dycore.py:1429-1439; "
        "gpuwm/core/physics.py:514-534,569-631,842-856; "
        "gpuwm/runtime.py:571-585",
        "pre-RK Noah reads precede the post-RK scheme write; output reads "
        "after accept, so the driver can alias the canonical mp_* set"),
    PhysicsArrayLifetime(
        tuple(f"tendencies/{name}" for name in _TENDENCY_COMPONENTS),
        "aliases_fresh_pbl_at_bldt_zero", "gpuwm/core/physics.py:779-819,"
        "895-959",
        "bldt=0 YSU replaces pbl_tendencies before every composition; "
        "positive cadence retains the separate target unchanged"),
    PhysicsArrayLifetime(
        tuple(f"{stack}/{name}" for stack in
              ("pbl_tendencies", "radiation_tendencies",
               "cumulus_tendencies") for name in _TENDENCY_COMPONENTS),
        "retained_family_state", "gpuwm/core/physics.py:633-777,895-959",
        "radiation/cumulus carry between due calls; PBL is the proven "
        "bldt=0 composition backing but is otherwise held family state"),
)


def physics_array_lifetime(name: str) -> PhysicsArrayLifetime | None:
    """Return the unique exact-name physics lifetime row, if audited."""
    matches = [row for row in PHYSICS_ARRAY_LIFETIME_AUDIT
               if name in row.names]
    if len(matches) > 1:
        raise RuntimeError(f"physics lifetime audit overlaps for {name!r}")
    return matches[0] if matches else None




#: Strain/stress component order (sase.py launch_strain / authority) and
#: the Germano-lift pair order (authority ``_PAIRS``).
_SASE_S6 = ("xx", "yy", "zz", "xy", "xz", "yz")
_SASE_P6 = ("uu", "vv", "ww", "uv", "uw", "vw")


def sase_workspace_phases(cfg: RunConfig
                          ) -> dict[str, dict[str, tuple[tuple[int, ...],
                                                         int]]]:
    """SASE model-path transient live sets, one dict per phase.

    Exact transcription of the driver-coupled step's per-call device
    allocations (``gpuwm/core/physics.py`` ``_run_sase`` +
    ``gpuwm/core/sase.py`` ``launch_sase_step`` and the launchers it
    composes) on the MODEL path (per-column 3-D ``dz_col``, Task 6).
    Task-6 decision, documented here per the S3-5 pairing contract: the
    per-step temporaries REMAIN CuPy-pool allocations at this stage
    rather than moving into the shared scratch arena -- the pool
    reuses the freed blocks across steps so the steady-state device
    footprint equals this transcribed peak, while arena preallocation
    would require threading an allocator through six nested launchers
    and re-auditing ~58 slot lifetimes (revisit if the estimate ever
    pinches).  Any change to either side must update BOTH the launcher/
    driver and this transcription (the byte pin in tests/test_sase.py
    enforces the pairing).

    * ``solve`` -- the peak inside ``launch_dynamic_solve`` while
      ``launch_germano_lift`` runs on the second (width-4) test level:
      the driver-held work set (u/v A-grid work copies, destaggered w +
      its work copy, n2 plus its S4-2 M1 moist companion n2_eff -- the
      ``launch_moist_n2`` output held beside the dry field for the whole
      step (physics.py ``_run_sase`` step 2; both go down together into
      ``launch_sase_step``), and the ``heat`` output -- 7 full fields;
      the S3-6e governed scalar channel rides the step's exported km_h
      field, so the v0 pre-step e copy is RETIRED), the fused step's
      three PER-COLUMN z-stencil coefficient FIELDS (3-D dz_col mode;
      the (nz,) arrays of the shared-column test path are superseded on
      the model path), the six fine strains, six premultiplied
      eddy-basis integrands, three filtered velocities, six coarse
      strains, six refiltered basis fields, the lift's three filtered
      velocities + six velocity products + six filtered products + six
      lift outputs, and the width-2 iteration's still-referenced FP64
      ``(5, nblocks)`` partial-sum buffer: 58 full fields + partials
      (S4-3 amendment: was 57 pre-M1; the n2_eff field is 1/57 ~ 1.75%
      of the previous peak -- the S4-2 report note-1 obligation).  The
      S3-11b ``(ny, nx)`` float32 rho1 surface moist-density plane
      (``sase_surface_rho1``, computed once before the e source and held
      by the driver through the step-5 scalar-flux deposit, so alive in
      BOTH phases) is transcribed with it -- de-minimis at
      1/(57*nz) ~ 0.04% of the peak (the S3-11b report note-3 flag,
      absorbed explicitly here rather than left to the (nz,)-pack
      de-minimis precedent since this touch amends the pairing anyway).
      S4-5 amendment (the S4-3 in-task pairing law): the SASE-M2 deposit
      seam holds ONE new driver field in BOTH phases -- the pre-step
      ``e_sgs`` copy ``e_pre`` that freezes the venting limb's amplitude
      input (physics.py ``_run_sase``; theta/qv/qc/pressure are
      read-only through the whole slot, so ``e`` is the only state the
      limb needs frozen, and the limb cannot be evaluated beside the
      copy because its (1 - f) two-product blend needs the step's USED
      f).  One full 4*ncell field, 1/58 ~ 1.72% of the previous peak.
      It is NOT the v0 pre-step e copy the S3-6e governed scalar channel
      retired (that one fed ``launch_scalar_mix``'s coefficient mode and
      stays retired); it is a new field with a new consumer, and it is
      transcribed under the DISTINCT key ``driver_vent_e_pre``.  S4-5b
      Item 4b: the S4-5 build gave it the retired field's own key
      ``driver_e_pre`` and deleted the retirement guard that asserted
      that key absent, while stating here that ``driver_e_pre`` "is
      deliberately not its name" -- it was.  The suffixed key restores
      the guard's meaning: ``driver_e_pre`` is absent because the v0
      copy is retired, and the M2 seam's frozen copy is a different
      name.  The byte pin does not move (one ``(nz, ny, nx)`` float32
      field either way; re-derived at
      ``test_sase.test_sase_workspace_accounting_is_exact``).
      The launcher EXPLICITLY drops the previous width's six-field lift
      binding (``lift = None``) before the next width's allocations --
      without that drop the peak would be 6 fields higher; this
      transcription and the launcher are a bound pair (S3-5 review fix),
      so removing the drop requires re-pinning here.  The S3-6f
      partition-bound sub-moment (the ``(ny, nx)`` z_i column field and
      the FP64 ``(5, nblocks)`` w-sensor partials) runs AFTER the solve
      transients release and drops its references before the apply
      allocations -- transcribed in this phase as a covering superset
      (net far below the 57-field peak).
    * ``apply`` -- re-derived for the S3-6e governed split step's true
      allocation profile.  The peak sits at the ``sase_split_tendencies``
      launch: the same driver-held 7 + coefficient 3, the vertical
      channel's kv + leps fields (S3-6h: leps = the BL89 RANS
      dissipation limb, formerly the bare l_B), six strains, six
      stresses, the governed
      diffusivity km + smag-share r fields (S3-6e), TWO horizontal
      e-flux integrands (the vertical leg retired with the explicit
      step), and FIVE tendency fields (du, dv, dw, P_h,e, P_h,heat --
      the S3-6e production split) -- 33 full fields with the S4-3 M1
      n2_eff amendment, + the rho1 plane (S3-6f doc fix, kept for the
      record: the S3-6e text said 33 when the enumeration summed to 32
      pre-M1; the byte pin never drifted).  The launcher then
      DROPS the 13-field strain/stress/r binding before the
      Thomas/production/partials allocations (bound pair with this
      transcription, the S3-5 idiom), so the later sub-moments
      (momentum-Thomas FP64 ``(3, nblocks_col)`` partials -- the third
      row is the S3-6j dKE_sfc drag-work channel -- P_v, the
      S3-6e damping-taper weight field under damp_opt=3, the e-update
      FP64 ``(4, nblocks)`` partials) arrive net NEGATIVE; the partials
      buffers, P_v, and the taper field are transcribed anyway as a
      covering superset.  S4-5: the SASE-M2 seam's THREE
      ``(nz + 1, ny, nx)`` float32 face-flux planes and its
      ``(ny, nx)`` FP64 cap-rescale plane are allocated in this phase,
      after the split step returns and before the scalar loop, and are
      transcribed here for the same covering-superset reason (they land
      after the 13-field strain/stress/r drop, so the apply phase's own
      peak does not move; the solve phase continues to dominate the
      category bound by **20.01 full fields** at 49x250x250 and at
      49x501x501, 19.59 at the test configuration -- S4-5b Item 5b,
      MEASURED this session from :func:`sase_workspace_phases` itself
      as ``(sum(solve) - sum(apply))/(4*nz*ny*nx)``; the S4-5 text said
      "~25", which was never the number.  The M2 seam's own four apply
      entries are what closed the gap: at the BASE commit a5b8d7e the
      same probe measures 23.11 (23.21 at the test configuration), and
      3*(nz + 1)*4 + 8 bytes per column against 4*nz is exactly the
      3.10-field difference).  The Thomas sweeps themselves hold their
      r/c'/d' arrays in per-thread registers/local memory (3x SASE_KMAX
      doubles per thread) -- NO global workspace, which is why no
      Thomas entry appears here.  S3-12 (the in-task pairing law): the
      additive dissipation channel's ONE new device field -- the
      state-independent Blackadar reference length
      (``launch_blackadar_length``), allocated after the 13-field drop
      and only under ``cfg.sase_additive_dissipation`` -- is
      transcribed unconditionally as the covering-superset entry
      ``lb_ref``, exactly the taper_g idiom; the solve phase keeps
      dominating (~20-field margin) so neither the category bound nor
      the byte pin moves.  The post-step scalar phase (kv + km_h
      held + 2 horizontal flux + 4 mixed rates + the coupled set)
      stays far below the solve bound.

    The category bound is the MAXIMUM over phases (the
    :func:`rrtmgp_workspace_phases` idiom); ``solve`` always dominates.
    The solve-internal (nz,) coefficient packs of its uniform-dz strain
    calls remain de-minimis and untranscribed, exactly as before.
    """
    if cfg.bl_pbl_physics != SASE_PBL_SCHEME:
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    ncell = nz * ny * nx
    # Single-sourced with the device define through the closure's own
    # compile-time tier, so this transcription's block count cannot
    # drift from the block size the kernels are actually compiled at.
    from gpuwm.core.sase import _DEFINE_VALUES as _SASE_DEFINES
    tpb = _SASE_DEFINES["SASE_TPB"]
    nblocks = (ncell + tpb - 1) // tpb
    partials = ((5, nblocks), 8)              # FP64 in-kernel reductions

    common: dict[str, tuple[tuple[int, ...], int]] = {"heat": (m, 4)}
    for name in ("zcm", "zc0", "zcp"):        # per-column coefficient pack
        common[name] = (m, 4)
    # S4-3 amendment (S4-2 report note-1 obligation): n2_eff = the M1
    # launch_moist_n2 output, the 6th driver-held work field, alive
    # beside the dry n2 through the whole step (both phases).
    # S4-5 amendment: the SASE-M2 seam's frozen pre-step e_sgs copy is
    # the 7th driver-held work field, alive from before the surface e
    # source through the step-5 scalar loop (both phases).  S4-5b
    # Item 4b: keyed ``vent_e_pre``, NOT ``e_pre`` -- the latter is the
    # retired v0 pre-step e copy's name and stays absent by name.
    for name in ("u_work", "v_work", "w_half", "w_work", "n2", "n2_eff",
                 "vent_e_pre"):
        common[f"driver_{name}"] = (m, 4)     # _run_sase held work set
    # S3-11b note-3 absorbed (same pairing touch): the (ny, nx) float32
    # rho1 surface moist-density plane (sase_surface_rho1), held from
    # before the e source through the step-5 scalar-flux deposit.
    common["driver_rho1"] = ((ny, nx), 4)

    solve = dict(common)
    for comp in _SASE_S6:
        solve[f"s_fine_{comp}"] = (m, 4)
        solve[f"premul_{comp}"] = (m, 4)
        solve[f"s_coarse_{comp}"] = (m, 4)
        solve[f"refilt_{comp}"] = (m, 4)
    for comp in ("u", "v", "w"):
        solve[f"filt_{comp}"] = (m, 4)
        solve[f"lift_filt_{comp}"] = (m, 4)
    for comp in _SASE_P6:
        solve[f"lift_prod_{comp}"] = (m, 4)
        solve[f"lift_fprod_{comp}"] = (m, 4)
        solve[f"lift_{comp}"] = (m, 4)
    solve["partials"] = partials
    # S3-6f partition-bound sub-moment (covering superset -- docstring):
    # the z_i column field and the w-sensor FP64 reduction buffer.
    solve["zi"] = ((ny, nx), 4)
    solve["w_partials"] = ((5, nblocks), 8)

    apply_phase = dict(common)
    for name in ("kv", "leps"):               # vertical-channel fields
        apply_phase[name] = (m, 4)
    for comp in _SASE_S6:
        apply_phase[f"strain_{comp}"] = (m, 4)
        apply_phase[f"tau_{comp}"] = (m, 4)
    for name in ("km_h", "r_smag"):           # S3-6e governed stress
        apply_phase[name] = (m, 4)
    for comp in ("x", "y"):                   # horizontal e-flux only
        apply_phase[f"e_hflux_{comp}"] = (m, 4)
    for comp in ("du", "dv", "dw", "ph_e", "ph_heat"):
        apply_phase[f"tend_{comp}"] = (m, 4)
    # Post-drop sub-moments (net below the 32-field peak; covering
    # superset -- see the docstring): implicit-flux production, the
    # S3-6e damping-taper weight field, and the two FP64 reduction
    # buffers.
    apply_phase["prod_v"] = (m, 4)
    apply_phase["taper_g"] = (m, 4)
    # S3-12: the additive channel's state-independent Blackadar
    # reference field (launch_blackadar_length), allocated after the
    # 13-field strain/stress/r drop and only under
    # cfg.sase_additive_dissipation -- transcribed unconditionally as a
    # covering superset exactly like taper_g (the solve phase dominates
    # by ~20 fields, so the category bound does not move; the byte pin
    # rides the solve phase and is untouched).
    apply_phase["lb_ref"] = (m, 4)
    # S4-5 SASE-M2 deposit seam: the three face-registered flux planes
    # and the per-column FP64 cap-rescale plane.
    for name in ("vent_f_theta", "vent_f_qv", "vent_f_qc"):
        apply_phase[name] = ((nz + 1, ny, nx), 4)
    apply_phase["vent_scale"] = ((ny, nx), 8)
    ncol_blocks = (ny * nx + tpb - 1) // tpb
    # S3-6j: third row = the dKE_sfc drag-work reduction channel.
    apply_phase["partials_mom"] = ((3, ncol_blocks), 8)
    apply_phase["partials_e"] = ((4, nblocks), 8)

    return {"solve": solve, "apply": apply_phase}


def sase_workspace_shapes(cfg: RunConfig
                          ) -> dict[str, tuple[tuple[int, ...], int]]:
    """The SASE step-transient bound: the phase-maximum simultaneous set,
    ``{"sase/<phase>/<name>": (shape, itemsize)}`` (see
    :func:`sase_workspace_phases`)."""

    def total(items):
        return sum(math.prod(shape) * size for shape, size in items.values())

    phases = sase_workspace_phases(cfg)
    if not phases:
        return {}
    phase = max(phases, key=lambda name: total(phases[name]))
    return {f"sase/{phase}/{name}": spec
            for name, spec in phases[phase].items()}


def state_array_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Exact ``DomainState`` allocation list (state.py:52-237 transcribed).

    Every array ``DomainState.__init__`` allocates, keyed by attribute
    name, under the same conditionals (``cfg.moist``, microphysics scheme,
    ``cfg.terrain_opt``).  Cross-checked against the restart
    manifest's attribute classification by test (a state.py field added
    without updating BOTH manifests fails the suite).
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    xs = (nz, ny, nx + 1)
    ys = (nz, ny + 1, nx)
    fl = (nz + 1, ny, nx)
    s2 = (ny, nx)
    shapes: dict[str, tuple[int, ...]] = {
        # Prognostics + EOS diagnostics.
        "u": xs, "v": ys, "w": fl, "thp": m, "php": fl, "mup": s2,
        "p": m, "al": m, "alt": m,
        # RK time-t copies.
        "u0": xs, "v0": ys, "w0": fl, "thp0": m, "php0": fl, "mup0": s2,
        # Slow-tendency slots.
        "ru_t": xs, "rv_t": ys, "rw_t": fl, "rth_t": m, "rph_t": fl,
        "rmu_t": s2,
        # Acoustic-substep perturbations.
        "u_pp": xs, "v_pp": ys, "w_pp": fl, "th_pp": m, "ph_pp": fl,
        "mu_pp": s2, "p_pp": m, "p_pp_old": m, "ww_pp": fl, "al_pp": m,
        # General-form plumbing + map factors / rotation.
        "mub2d": s2, "ht": s2,
        "c1h": (nz,), "c2h": (nz,), "c1f": (nz + 1,), "c2f": (nz + 1,),
        "c3h": (nz,), "c4h": (nz,), "c3f": (nz + 1,), "c4f": (nz + 1,),
        "msft": s2, "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
        "f": s2, "e": s2, "sina": s2, "cosa": s2,
        # Vertical-coordinate arrays.
        "dnw": (nz,), "rdnw": (nz,), "dn": (nz,), "rdn": (nz,),
        "fnp": (nz,), "fnm": (nz,), "znu": (nz,), "znw": (nz + 1,),
    }
    if cfg.terrain_opt == 0:
        shapes.update(thb=(nz,), pb=(nz,), alb=(nz,), phb=(nz + 1,))
    else:
        shapes.update(thb=m, pb=m, alb=m, phb=fl)
    if cfg.moist:
        for name in ("qv", "qc", "qr", "qv0", "qc0", "qr0", "h_diabatic"):
            shapes[name] = m
        if cfg.cu_physics in CUMULUS_ADVECTIVE_FORCING_SCHEMES:
            # WRF RTHFTEN/RQVFTEN, allocated by the same table predicate
            # gpuwm/core/state.py uses.  Two persistent mass-point rates
            # priced in the VRAM projection for the schemes that read them
            # and for nobody else.
            shapes["rthften"] = m
            shapes["rqvften"] = m
        if cfg.mp_physics == 50:
            # P3 one-category (Registry.EM_COMMON:3038, and the mp==50 arm
            # of gpuwm/core/state.py): ONE ice mass with rime mass and rime
            # volume, no qs/qg/effs, plus the two cross-step supersaturation
            # carriers p3_main writes at the end of every call.  Every
            # transported field carries its RK time-t copy.
            for name in ("qi", "ni", "nr", "qir", "qib", "effc", "effi",
                         "th_old", "qv_old",
                         "qi0", "ni0", "nr0", "qir0", "qib0"):
                shapes[name] = m
        if cfg.mp_physics in (6, 8, 9, 10, 16, 18, 28):
            for name in ("qi", "qs", "qg", "qi0", "qs0", "qg0",
                         "effc", "effi", "effs"):
                shapes[name] = m
        if cfg.mp_physics == 9:
            # Milbrandt-Yau two-moment: hail mass beside graupel plus a
            # number moment for EVERY one of the six hydrometeors
            # (gpuwm/core/moist.py::MY2_SPECIES), each transported field
            # with its RK time-t copy (gpuwm/core/state.py, the mp==9
            # arms).  This block was missing at 1.9.0: the state builder
            # requested rebuilt("qi0"...) views from a shared workspace
            # this manifest had never priced, so an ACCEPTED mp=9 config
            # could not build its real-case workspace (1.9.1 D1).
            for name in ("qh", "nc", "nr", "ni", "ns", "ng", "nh",
                         "qh0", "nc0", "nr0", "ni0", "ns0", "ng0", "nh0"):
                shapes[name] = m
        if cfg.mp_physics == 16:
            for name in ("nn", "nc", "nr", "nn0", "nc0", "nr0"):
                shapes[name] = m
        if cfg.mp_physics == 8:
            for name in ("nr", "ni", "nr0", "ni0"):
                shapes[name] = m
        if cfg.mp_physics == 10:
            for name in ("nc", "nr", "ni", "ns", "ng", "nr0", "ni0",
                         "ns0", "ng0", "effr"):
                shapes[name] = m
        if cfg.mp_physics == 18:
            for name in (
                    "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
                    "qnn", "qvolg", "qvolh", "qh0", "qndrop0", "qnr0",
                    "qni0", "qns0", "qng0", "qnh0", "qnn0", "qvolg0",
                    "qvolh0"):
                shapes[name] = m
        if cfg.mp_physics == 28:
            # Thompson aerosol-aware: prognostic droplet number plus the two
            # aerosol number tracers, each with its RK time-t copy
            # (gpuwm/core/state.py, the mp==28 arms).
            for name in ("nc", "nr", "ni", "nwfa", "nifa",
                         "nc0", "nr0", "ni0", "nwfa0", "nifa0"):
                shapes[name] = m
            # QNWFA2D / QNIFA2D surface emission tendencies, # kg-1 s-1.
            # Cross-step constants, allocated once per domain.
            for name in ("nwfa2d", "nifa2d"):
                shapes[name] = s2
    if cfg.km_opt == 2:
        # WRF's two-time-level prognostic TKE (Registry.EM_COMMON:312):
        # the SERIALIZED carrier plus its REBUILT time-t copy.
        shapes["tke"] = m
        shapes["tke0"] = m
    if cfg.bl_pbl_physics in (SASE_PBL_SCHEME, 11):
        # The published subgrid energy (state.py allocates it under the
        # same two-scheme condition): SASE's prognostic closure energy,
        # or Shin-Hong's per-step TKE diagnostic -- the D1 gray-zone
        # instrument reads state.e_sgs whichever closure produced it.
        shapes["e_sgs"] = m
    return shapes


def shared_dycore_state_symbols() -> frozenset[str]:
    """Restart-REBUILT symbols eligible for sequential-domain sharing.

    The restart manifest is the sole inventory authority.  Keeping this as a
    function avoids a second module-level set that could drift independently.
    """
    from gpuwm.io.restart import STATE_REBUILT_ATTRS

    return STATE_REBUILT_ATTRS


def shared_dycore_state_workspace_shapes(
        domains: tuple[object, ...]) -> dict[str, tuple[int, ...]]:
    """Maximum requested shape for each active restart-REBUILT symbol."""
    maxima: dict[str, tuple[int, ...]] = {}
    rebuilt = shared_dycore_state_symbols()
    for dc in domains:
        shapes = state_array_shapes(dc.run)
        for symbol in rebuilt & shapes.keys():
            shape = shapes[symbol]
            previous = maxima.get(symbol)
            if previous is None or math.prod(shape) > math.prod(previous):
                maxima[symbol] = shape
    return {symbol: maxima[symbol] for symbol in sorted(maxima)}


def shared_dycore_state_workspace_bytes(
        domains: tuple[object, ...]) -> int:
    """Exact bytes for one float32 maximum backing per active symbol."""
    return sum(4 * math.prod(shape) for shape in
               shared_dycore_state_workspace_shapes(domains).values())


#: initialize_physics's own 2-D allocations before the SFCLAY/Noah unions
#: (physics.py:935-950).
_PHYSICS_INIT_FIELDS_2D = (
    "landmask", "xland", "tsk", "pblh", "mavail", "lakemask",
    "ivgtyp", "isltyp", "vegfra", "tmn", "xice", "swdown", "glw",
    "snow", "snowh",
)


def physics_field_names_2d(cfg: RunConfig | None = None) -> tuple[str, ...]:
    """The surface ``fields`` dict's 2-D inventory, reconstructed from the
    same name tuples ``initialize_physics`` consumes (physics.py:935-990):
    init fields | SFCLAY_OUTPUTS | NOAH _F2D, plus ``ebal``/``kpbl``.

    ``initialize_physics`` additionally allocates MYNN's extra persistent
    surface diagnostics when ``sf_sfclay_physics == 5``, and deliberately
    does NOT allocate them otherwise.  Passing ``cfg`` reproduces that
    selection; omitting it returns the MM5/Noah union, which is what every
    caller without a configuration in hand means.  Under-counting here is a
    correctness bar on this hardware, not a cosmetic one.
    """
    from gpuwm.core.noah import _F2D as NOAH_FIELDS_2D
    # physics_inventory, not sfclay/mynn_*: those modules import cupy
    # at module scope and this inventory must be readable on installs
    # with no GPU runtime (the wizard's estimator runs here).
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS

    union = dict.fromkeys(_PHYSICS_INIT_FIELDS_2D)
    union.update(dict.fromkeys(SFCLAY_OUTPUTS))
    if cfg is not None and int(cfg.sf_sfclay_physics) == 5:
        from gpuwm.core.physics_inventory import MYNN_SURFACE_OUTPUTS
        union.update(dict.fromkeys(MYNN_SURFACE_OUTPUTS))
    elif (cfg is not None and int(cfg.km_opt) in (2, 3, 4)
          and int(cfg.bl_pbl_physics) == 0):
        # vertical_diffusion_2's isfflx=1 wall stress consumes WRF USTM.
        # MM5 surface schemes otherwise do not retain this MYNN-adjacent
        # diagnostic in ArWen.
        union["ustm"] = None
    union.update(dict.fromkeys(NOAH_FIELDS_2D))
    union.update(dict.fromkeys(("ebal", "kpbl")))
    if cfg is not None and int(cfg.bl_pbl_physics) == 5:
        from gpuwm.core.physics_inventory import (
            MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D,
        )
        union.update(dict.fromkeys(MYNN_PBL_DIAGNOSTICS_2D))
        union.update(dict.fromkeys(MYNN_PBL_DIAGNOSTICS_INT_2D))
    if cfg is not None and int(cfg.sf_surface_physics) == 3:
        from gpuwm.core.ruc_runtime import (
            RUC_DIAGNOSTICS_2D, RUC_FRACTIONAL_SEAICE_FIELDS, RUC_STATE_2D,
        )
        from gpuwm.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS
        union.update(dict.fromkeys(RUC_STATE_2D))
        union.update(dict.fromkeys(RUC_DIAGNOSTICS_2D))
        union.update(dict.fromkeys(RUC_FRACTIONAL_SEAICE_FIELDS))
        if int(cfg.sf_sfclay_physics) == 5:
            # MYNN_SEAICE_WRAPPER performs a full second surface call.  WRF
            # keeps most results as automatic locals and exposes only the
            # wait-for-LSM subset; ArWen retains the full result so the GPU
            # launch allocates nothing outside this preflight inventory.
            from gpuwm.core.physics_inventory import MYNN_SURFACE_OUTPUTS
            union.update(dict.fromkeys(
                f"{name}_sea" for name in MYNN_SURFACE_OUTPUTS))
        union.update(dict.fromkeys(SURFACE_PRECIPITATION_FIELDS))
        union["gsw"] = None
    if cfg is not None and int(cfg.sf_surface_physics) == 4:
        from gpuwm.core.noahmp_runtime import (
            NOAHMP_DIAGNOSTICS_2D, NOAHMP_STATE_2D, NOAHMP_STATE_INT_2D,
        )
        union.update(dict.fromkeys(NOAHMP_STATE_2D))
        union.update(dict.fromkeys(NOAHMP_STATE_INT_2D))
        union.update(dict.fromkeys(NOAHMP_DIAGNOSTICS_2D))
        from gpuwm.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS
        union.update(dict.fromkeys(SURFACE_PRECIPITATION_FIELDS))
        union["coszen"] = None
    return tuple(union)


def physics_array_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """``PhysicsDriver`` persistents per selected scheme (physics.py).

    Includes the surface/Noah ``fields`` dict, held family tendencies, a
    separate composed target except in the proven bldt=0/PBL identity path,
    positive-cadence raw YSU retention, radiative heating rates, mp=0 output
    placeholders, KF W0AVG/LUT, and RRTMGP setup grids.  Active microphysics
    diagnostics and KF ``cu_*`` persistence live in the scratch registry.
    """
    # The runtime-free inventory module, NOT gpuwm.core.physics: that
    # module's body imports cupy, and this estimator is what `gpuwm
    # domain` runs on CPU-only installs (it used to refuse every one of
    # them from exactly this import).
    from gpuwm.core.physics_inventory import (PBL_RQI_MICROPHYSICS,
                                              hmix_k_diag_names,
                                              microphysics_scratch_slots,
                                              physics_driver_required,
                                              physics_retains_ysu_output,
                                              physics_reuses_pbl_composition)

    if not physics_driver_required(cfg):
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    s2 = (ny, nx)
    shapes: dict[str, tuple[int, ...]] = {}
    for name in physics_field_names_2d(cfg):
        shapes[f"fields/{name}"] = s2
    n_soil = soil_layer_count(cfg)
    for name in ("smois", "tslb", "sh2o", "smcrel"):
        shapes[f"fields/{name}"] = (n_soil, ny, nx)
    shapes["fields/exch_h"] = m
    shapes["fields/exch_m"] = m
    if int(cfg.bl_pbl_physics) == 5:
        # initialize_physics allocates MYNN's ten carried 3-D arrays only for
        # this selector.  Missing them here under-counts VRAM by 10*nz*ny*nx
        # FP32 words, which on a four-domain nest is not a rounding error.
        from gpuwm.core.physics_inventory import MYNN_PBL_STATE_3D
        for name in MYNN_PBL_STATE_3D:
            shapes[f"fields/{name}"] = m
    if int(cfg.sf_surface_physics) == 3:
        # RUC's two Registry-package soil-column arrays, SMFR3D and
        # KEEPFR3DFLAG.  At nine levels that is 18*ny*nx FP32 words per
        # domain on top of the four generic soil arrays.
        from gpuwm.core.ruc_runtime import RUC_STATE_3D
        for name in RUC_STATE_3D:
            shapes[f"fields/{name}"] = (n_soil, ny, nx)
    if int(cfg.sf_surface_physics) == 4:
        # Noah-MP's snow stack.  Three (NSNOW, ny, nx) arrays plus one
        # (NSNOW + n_soil, ny, nx); missing them under-counts by
        # (4*NSNOW + n_soil)*ny*nx FP32 words per domain.
        from gpuwm.core.noahmp_runtime import (
            NOAHMP_STATE_SNOWSOIL_3D, NOAHMP_STATE_SNOW_3D, NSNOW,
        )
        for name in NOAHMP_STATE_SNOW_3D:
            shapes[f"fields/{name}"] = (NSNOW, ny, nx)
        for name in NOAHMP_STATE_SNOWSOIL_3D:
            shapes[f"fields/{name}"] = (NSNOW + n_soil, ny, nx)

    stacks = ["pbl_tendencies", "radiation_tendencies", "cumulus_tendencies"]
    reuse_pbl = physics_reuses_pbl_composition(cfg)
    if (radiation_enabled(cfg) or cfg.cu_physics) and not reuse_pbl:
        stacks.append("tendencies")  # composed target, physics.py:426-428
    for stack in stacks:
        shapes[f"{stack}/ru"] = (nz, ny, nx + 1)
        shapes[f"{stack}/rv"] = (nz, ny + 1, nx)
        for comp in ("rtheta", "rqv", "rqc"):
            shapes[f"{stack}/{comp}"] = m
    if cfg.cu_physics:
        # Mixed-phase KF returns QR/QI/QS independently.  At bldt=0 the fresh
        # PBL stack is the composed target; positive cadence keeps the
        # historical separate target.
        target = "pbl_tendencies" if reuse_pbl else "tendencies"
        for comp in ("rqr", "rqi", "rqs"):
            shapes[f"cumulus_tendencies/{comp}"] = m
            shapes[f"{target}/{comp}"] = m
    if cfg.bl_pbl_physics and cfg.mp_physics in PBL_RQI_MICROPHYSICS:
        # Mixed-phase states carry qi; YSU returns dqi and rqi survives
        # composition (physics.py:263-264, :681-692).  mp=28 belongs by
        # Registry/Registry.EM_COMMON:3036 -- the thompsonaero package
        # declares moist:qv,qc,qr,qi,qs,qg, so WRF's F_QI is true and
        # module_first_rk_step_part1.F:1112's CALL pbl_driver hands
        # moist(...,P_QI), F_QI=F_QI (:1199) to the PBL driver.  This budget
        # mirrors physics._pbl_optional_tendency_components; the two sets
        # must stay identical, so they are now ONE constant.
        shapes["pbl_tendencies/rqi"] = m
        if (radiation_enabled(cfg) or cfg.cu_physics) and not reuse_pbl:
            shapes["tendencies/rqi"] = m

    if physics_retains_ysu_output(cfg):
        # Positive cadence preserves the historical retained diagnostic.
        # At bldt=0 the same arrays are step transients itemized below.
        for name in _YSU_3D:
            shapes[f"last_ysu/{name}"] = m
        for name in _YSU_2D:
            shapes[f"last_ysu/{name}"] = s2

    shapes["rthratenlw"] = m
    shapes["rthratensw"] = m
    shapes["_pending_rainbl"] = s2
    if cfg.bl_pbl_physics == SASE_PBL_SCHEME and cfg.sase_flux_diag:
        # SPLIT SUBGRID-FLUX DIAGNOSTIC (physics.py PhysicsDriver
        # __init__): four z-FACE (nz+1, ny, nx) FP32 driver persistents
        # holding the venting and K_v channels of the closure's vertical
        # subgrid moisture/heat flux.  RESIDENT, not step transient --
        # output reads them after the step ends -- which is why they are
        # itemized here and NOT in sase_workspace_phases, whose
        # solve-phase byte pin is therefore unmoved.  Gated on the key,
        # so every estimate that does not set it is byte-identical by
        # construction.
        for name in ("fqv_vent", "fqv_diff", "fth_vent", "fth_diff"):
            shapes[f"sase_flux_diag/{name}"] = (nz + 1, ny, nx)
    if cfg.hmix_k_diag and hmix_k_diag_names(cfg):
        # HORIZONTAL EDDY-VISCOSITY DIAGNOSTIC (physics.py PhysicsDriver
        # __init__): two mass-grid (nz, ny, nx) FP32 driver persistents.
        # RESIDENT for the flux diagnostic's reason -- output reads them
        # after the step ends -- and gated on the key, so every estimate
        # that does not set it is byte-identical by construction.  The
        # producer's own K field is NOT counted again here: the km_opt=4
        # smag_km/smag_kh scratch and the closure's step-transient km_h
        # are already accounted where they are allocated; these two are
        # the published copies.
        for name in hmix_k_diag_names(cfg):
            shapes[f"hmix_k_diag/{name}"] = m
    # Active microphysics diagnostics alias the carrying mp_* scratch set.
    # With mp_physics=0 there is no canonical set, so the historical three
    # zero-filled output-plumbing arrays remain driver-owned.
    if not microphysics_scratch_slots(cfg.mp_physics):
        for comp in ("rainnc", "rainncv", "sr"):
            shapes[f"microphysics/{comp}"] = s2
    if cfg.cu_physics == 1:
        shapes["cumulus/w0avg"] = m  # kf.py:174, WRF Registry r-flagged
        # Once-per-process lru_cache device LUT (kf.py load_kf_table +
        # _device_table): temperature/qsat (250,220) + thetae_base (220,)
        # + log_ratio (200,) FP32 = 441,680 B.  Counted on the cumulus
        # domain -- exactly one in every ratified config (cu is d01-only);
        # the cache itself is process-wide.
        shapes["cumulus/kf_lut_temperature"] = (250, 220)
        shapes["cumulus/kf_lut_qsat"] = (250, 220)
        shapes["cumulus/kf_lut_thetae_base"] = (220,)
        shapes["cumulus/kf_lut_log_ratio"] = (200,)
    ra_lw_physics, ra_sw_physics = radiation_scheme_ids(cfg)
    if ra_lw_physics or ra_sw_physics:
        shapes["radiation/latitude_deg"] = s2
        shapes["radiation/longitude_deg"] = s2
    if (ra_lw_physics, ra_sw_physics) == (4, 4):
        # RRTMGPRadiation.__post_init__ ozone climatology profiles
        # (rrtmgp.py:1076-1077): two 60-level FP32 device arrays, 480 B.
        shapes["radiation/_ozone_logp"] = (60,)
        shapes["radiation/_ozone_vmr"] = (60,)
    if (ra_lw_physics, ra_sw_physics) in ((1, 1), (4, 4)):
        # The OLR publication buffer (physics.py PhysicsDriver __init__):
        # one resident (ny, nx) FP32 driver persistent holding WRF's TOA
        # outgoing longwave for output, allocated when the attached
        # longwave adapter declares ``publishes_olr``.  BOTH built-in 4/4
        # adapters declare it, and so does the 1/1 WRF RRTM + Dudhia
        # composition, which is what this gate reproduces; a
        # caller-injected adapter that does not leaves this counted but
        # unallocated, and over-counting by one 2-D field is the safe
        # direction for a VRAM estimate.
        shapes["olr"] = s2
        # NOT counted for the (1,1) pair, and said here so the omission
        # is deliberate rather than forgotten: RRTM's transfer holds
        # several (column_chunk, nlayers, 140) g-point arrays for the
        # duration of a chunk (gpuwm/core/rrtm_lw.py rtrn_columns).
        # Those are transients, and this whole rail prices PERSISTENT
        # allocations -- adding a peak-transient term for one scheme
        # would make the number mean something different from what it
        # means for every other row.  The size is real, though:
        # 1.67 GiB measured at column_chunk=4096 and 53 layers on an
        # RTX 5090, so about 0.2 GiB at the default 512 and linear in
        # the chunk.  A user who raises column_chunk gets no warning
        # from the budget; the registry warning on wrf-rrtm-dudhia and
        # DEFAULT_COLUMN_CHUNK's own comment carry that fact.
    return shapes


def _perimeter_count(ny: int, nx: int, width: int) -> int:
    """Cells in ``width`` nested frames (lateral_bc.py:220-223 verbatim)."""
    return sum(2 * (nx - 2 * d) + 2 * max(ny - 2 * d - 2, 0)
               for d in range(width))


#: d01 external-LBC field inventory (state boundaries built by
#: build_state_lateral_boundaries: u/v/theta/phi/mu + qv when moist) with
#: each field's (levels, ny-extent, nx-extent) source dims.
def _lbc_field_dims(cfg: RunConfig) -> dict[str, tuple[int, int, int]]:
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    dims = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
            "theta": (nz, ny, nx), "phi": (nz + 1, ny, nx),
            "mu": (1, ny, nx)}
    if cfg.moist:
        dims["qv"] = (nz, ny, nx)
    return dims


def lbc_interval_values(cfg: RunConfig) -> int:
    """FP32 values in ONE interval's side tables (value + tendency), per
    ``_field_boundary`` (lateral_bc.py:149-167): west/east
    ``(lev, ny, W)`` + south/north ``(lev, W, nx)``, each twice."""
    width = cfg.spec_bdy_width
    total = 0
    for lev, ny, nx in _lbc_field_dims(cfg).values():
        total += 2 * (2 * lev * ny * width + 2 * lev * width * nx)
    return total


def lbc_intervals(run_seconds: float, forcing_interval_seconds: float) -> int:
    """Eager interval count for the root's forcing coverage."""
    return max(1, math.ceil(run_seconds / float(forcing_interval_seconds)))


def mynn_pbl_column_chunk(cfg: RunConfig) -> int:
    """Columns per MYNN call for this domain.

    The declared chunk, capped by the domain: a 50x20 verification grid has
    1,000 columns and asks for one chunk of 1,000, while a 600x600 nest asks
    for 22 chunks of 16,384.  The workspace is therefore the same size on
    both, which is the property that lets a launch gate refuse a
    configuration before it allocates.
    """
    from gpuwm.core.mynn_pbl_scratch import MYNN_PBL_COLUMN_CHUNK
    return max(1, min(int(MYNN_PBL_COLUMN_CHUNK), int(cfg.ny) * int(cfg.nx)))


def mynn_pbl_scratch_slots(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Every ``bl_pbl_physics=5`` scratch slot and its exact shape.

    Split out of :func:`scratch_slot_registry` so the MYNN workspace can be
    priced, diffed and gated on its own; ``tests/test_mynn_pbl_scratch.py``
    checks this against the slot names the solver actually requests.
    """
    from gpuwm.core.mynn_pbl_scratch import (
        mynn_pbl_flag_shapes, mynn_pbl_index_shapes,
        mynn_pbl_scratch_shapes, mynn_pbl_tendency_field_shapes,
    )
    chunk = mynn_pbl_column_chunk(cfg)
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    slots: dict[str, tuple[int, ...]] = {}
    slots.update(mynn_pbl_scratch_shapes(chunk, nz))
    slots.update(mynn_pbl_index_shapes(chunk, nz))
    slots.update(mynn_pbl_flag_shapes())
    slots.update(mynn_pbl_tendency_field_shapes(nz, ny, nx))
    return slots


def mynn_pbl_scratch_bytes_for(cfg: RunConfig) -> int:
    """Device bytes the MYNN workspace occupies for this domain."""
    total = 0
    for shape in mynn_pbl_scratch_slots(cfg).values():
        values = 1
        for extent in shape:
            values *= int(extent)
        total += values * 4
    return total


def scratch_slot_registry(cfg: RunConfig, *,
                          n_lbc_intervals: int = 0
                          ) -> dict[str, tuple[int, ...]]:
    """Static registry: every named ``DomainState.scratch`` slot and its
    exact shape formula, keyed by the RunConfig features that create it.

    The completeness test (tests/test_preflight.py) AST-scans every
    ``scratch(...)`` call site in ``gpuwm/`` against this registry -- an
    unclassified slot is an error.  Sources cited per family.  Child
    ``nest_*`` slots live in :func:`nest_allocation_manifest` (the F4
    manifest), not here.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    xs = (nz, ny, nx + 1)
    ys = (nz, ny + 1, nx)
    fl = (nz + 1, ny, nx)
    s2 = (ny, nx)
    slots: dict[str, tuple[int, ...]] = {}

    # dycore.py rk stage fluxes (:102, :154-155) + integration health
    # (:1471-1474; nblocks = min(256, max(1, ceil(largest/256)))).
    slots.update(rk_ww=fl, rk_ru=xs, rk_rv=ys)
    largest = max(nz * ny * (nx + 1), (nz + 1) * ny * nx)
    nblocks = min(256, max(1, (largest + 255) // 256))
    slots["integration_health_partial"] = (nblocks, 9)
    slots["integration_health_result"] = (8,)
    # health.py StateHealthValidator: descriptor metadata and compact result.
    # These are always the fixed MAX_HEALTH_FIELDS footprint, independent of
    # case size; inventory additions fail at that explicit cap.
    from gpuwm.core.health import MAX_HEALTH_FIELDS
    health_words = (MAX_HEALTH_FIELDS * 2,)
    slots.update(
        integration_health_field_ptr=health_words,
        integration_health_aux_ptr=health_words,
        integration_health_field_size=health_words,
        integration_health_bounds=(MAX_HEALTH_FIELDS, 2),
        integration_health_flags=(MAX_HEALTH_FIELDS,),
        integration_health_planes=(MAX_HEALTH_FIELDS,),
        integration_health_status_bits=health_words,
        integration_health_validation=(4,),
    )
    # advection.py:161-163.
    slots.update(adv_ru=xs, adv_rv=ys, adv_rw=fl)
    # acoustic.py:115-116, :223-226.
    slots.update(acoustic_mu_pp_old=s2, acoustic_th_pp_old=m,
                 acoustic_c2a=m, acoustic_a=fl, acoustic_alpha=fl,
                 acoustic_gamma=fl)
    if cfg.moist and cfg.moist_cq:
        # acoustic.py:prepare_moist_cq.  These stage-fixed face arrays alias
        # the disjoint advection-only adv_ru/rv/rw arena backings below.
        slots.update(acoustic_cqu=xs, acoustic_cqv=ys, acoustic_cqw=fl)
    if cfg.open_x:
        slots["openbc_upp_faces"] = (nz, ny, 2)     # acoustic.py:121
    if cfg.open_y:
        slots["openbc_vpp_faces"] = (nz, 2, nx)     # acoustic.py:125
    if cfg.emdiv > 0.0:
        slots.update(acoustic_mudf=s2)

    if cfg.moist or cfg.km_opt == 2:
        # dycore.py acoustic time-averaged mass fluxes (moist scalars
        # and/or the km_opt=2 TKE carrier advect with them).
        slots.update(rk_ru_m=xs, rk_rv_m=ys, rk_ww_m=fl)
        slots["moist_rq_t"] = m                     # moist.py:247
        allocated = state_array_shapes(cfg)
        if "qi" in allocated and "qs" not in allocated:
            # gpuwm/core/moist.py::absent_mass_plane.  The two fused
            # moist-array sums (calc_cq, slow_buoyancy's q_total) select
            # their species by an integer mode with no one-ice-mass arm,
            # so a scheme with qi and no qs/qg hands the absent pair this
            # single shared zero plane.  P3 (mp=50) is the only such
            # scheme today, but the RUNTIME guards are presence-based, so
            # this predicate is too -- a cfg-keyed list here would
            # under-count memory for the next one-ice-category port
            # instead of failing.  Spelled as a literal because this
            # module must stay importable without cupy;
            # tests/test_p3_port.py pins it to moist.ABSENT_MASS_SLOT.
            slots["moist_absent_mass"] = m
        pd = ((cfg.moist and cfg.moist_adv_opt == 1) or cfg.km_opt == 2)             and not (cfg.open_x or cfg.open_y)
        if pd:
            slots.update(pd_fxl=xs, pd_fxc=xs, pd_fyl=ys, pd_fyc=ys,
                         pd_fzl=fl, pd_fzc=fl)      # moist.py:283-288
            slots["moist_pd_q0"] = m                # moist.py:197
        if cfg.specified:
            slots["lbc_qv_held"] = m                # moist.py:274

    if cfg.nwp_diagnostics == 1:
        # gpuwm/core/uh_diag.py: the serialized UP_HELI_MAX running max
        # (eagerly allocated by DomainState.__init__) plus two per-launch
        # work planes (column UH and the use_column flags).
        slots.update(up_heli_max=s2, uh_diag_col=s2, uh_diag_use=s2)
        # The two consumer-owned tracking windows (gpuwm/core/uh_diag.py:
        # TRACKER_WINDOW_SLOTS).  Priced on the same gate that allocates
        # them, because they are allocated whenever the diagnostic runs
        # rather than only when a follow/spawn block is declared: the
        # relocation and spawn tables live on the ExperimentConfig, which
        # DomainState.__init__ does not see, and inventing a RunConfig
        # field to carry them would move the frozen-config surface for
        # two (ny, nx) FP32 planes.
        slots.update(uh_follow_window=s2, uh_spawn_window=s2)

    if cfg.mp_physics == 1:
        # microphysics.py:143-163 (Kessler prep + accumulators).
        slots.update(mp_th=m, mp_rho=m, mp_pii=m, mp_z=m, mp_dz8w=m,
                     mp_z8w=fl, mp_rainnc=s2, mp_rainncv=s2,
                     mp_kessler_sr=s2)
    if cfg.mp_physics == 6:
        # wsm6.py preparation, persistent precipitation and due reflectivity.
        slots.update(wsm6_theta=m, wsm6_rho=m, wsm6_pii=m, wsm6_dz=m,
                     wsm6_z8w=fl, mp_rainnc=s2, mp_rainncv=s2,
                     mp_snownc=s2, mp_snowncv=s2, mp_graupelnc=s2,
                     mp_graupelncv=s2, mp_sr=s2, refl_t=m, refl_10cm=m)
    if cfg.mp_physics == 6 and cfg.ny == 1:
        # mpas_column_batch.py:769-773 (run_phase2 adapter pair):
        # alt = 1/rho_dry and php = z_interface*g, fully rewritten by every
        # phase-2 call before the microphysics dispatch reads them.  The
        # column-batch seam is the only allocator and always builds its
        # RunConfig with ny == 1 (one row of columns), so a plane-shaped
        # WSM6 forecast neither allocates nor is priced for either slot.
        slots.update(physics_column_alt=m, physics_column_php=fl)
    if cfg.mp_physics == 16:
        # wdm6.py preparation, persistent precipitation and due reflectivity.
        # Same slot shape as mp=6: WDM6's three extra moments are STATE, not
        # scratch, so nothing here grows with the double-moment warm rain.
        slots.update(wdm6_theta=m, wdm6_rho=m, wdm6_pii=m, wdm6_dz=m,
                     wdm6_z8w=fl, mp_rainnc=s2, mp_rainncv=s2,
                     mp_snownc=s2, mp_snowncv=s2, mp_graupelnc=s2,
                     mp_graupelncv=s2, mp_sr=s2, refl_t=m, refl_10cm=m)
    if cfg.mp_physics == 8:
        # microphysics.py:_apply_thompson preparation,
        # persistent precipitation, output-due private graupel-number shadow,
        # and Thompson REFL_10CM staging.  The three reference fields are
        # deliberately distinct registry entries even where the runtime can
        # lifetime-alias their arena backings.
        slots.update(
            mp_th=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
            mp_thompson_temperature=m,
            mp_thompson_frozen_reference_density=m,
            mp_thompson_frozen_reference_temperature=m,
            mp_thompson_rain_reference_density=m,
            mp_thompson_snow_melt_marker=m,
            mp_thompson_graupel_melt_marker=m,
            mp_thompson_snow_velocity_boost=m,
            mp_thompson_graupel_number_shadow=m,
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_graupelnc=s2, mp_graupelncv=s2, mp_sr=s2,
            refl_t=m, refl_10cm=m,
        )
    if cfg.mp_physics == 50:
        # P3 one-category (gpuwm/core/p3.py::apply).  The WRF preparation
        # bracket, five precipitation slots and the reflectivity staging,
        # plus P3's three own ice diagnostics.  NO graupel accumulators:
        # P3 has a single ice category and its driver arm passes no
        # GRAUPELNC (module_microphysics_driver.F:1590-1595).
        #
        # vmi3d/di3d/rhopo3d are WRF grid STATE for mp=50
        # (Registry.EM_COMMON:3038) but nothing downstream of the scheme
        # reads them in gpuwm yet, so they are registered as scratch rather
        # than promoted to DomainState fields -- the honest place for an
        # output the model computes and does not consume.  They are
        # registered rather than left unclassified so the allocation gate
        # sees their true size.
        slots.update(
            mp_th=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_sr=s2,
            p3_vmi=m, p3_di=m, p3_rhopo=m,
            refl_10cm=m,
        )
    if cfg.mp_physics == 28:
        # Thompson AEROSOL-AWARE (mp=28).  A deliberate near-clone of the
        # mp=8 block above -- the aerosol adapter reuses every classic
        # preparation, precipitation, melt-marker and reflectivity slot with
        # the same shape and the same lifetime -- plus the aerosol-only
        # working set.  The mp=8 block is NOT shared: mp=8's rows are pinned
        # byte-for-byte by tests/test_mp8_frozen.py (receipt R4) and a shared
        # literal is exactly how a future mp=28 slot would leak into the
        # frozen mp=8 arena layout.
        slots.update(
            mp_th=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
            mp_thompson_temperature=m,
            mp_thompson_frozen_reference_density=m,
            mp_thompson_frozen_reference_temperature=m,
            mp_thompson_rain_reference_density=m,
            mp_thompson_snow_melt_marker=m,
            mp_thompson_graupel_melt_marker=m,
            mp_thompson_snow_velocity_boost=m,
            mp_thompson_graupel_number_shadow=m,
            mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2, mp_snowncv=s2,
            mp_graupelnc=s2, mp_graupelncv=s2, mp_sr=s2,
            refl_t=m, refl_10cm=m,
        )
        # --- The aerosol working set -------------------------------------
        # WRF runs mp=28 as ONE monolithic column loop that freezes
        # nc1d/nwfa1d/nifa1d at entry (module_mp_thompson.F:1795-1812),
        # accumulates ncten/nwfaten/nifaten across widely separated regions,
        # and applies them exactly ONCE with a shared clamp (:3972-4021).
        # ArWen's fused network launchers write state in place, so the port
        # has to materialize that entry-state / accumulator split as device
        # arrays.  Every slot below is one of those, named for the launcher
        # parameter it feeds:
        #
        #   ncten/nwfaten/nifaten  -- the three shared per-kg-per-second
        #       accumulators (:1679-1681 zero them; :3972-4021 apply them).
        #       Written by the cold network, the warm network, the ncten
        #       balance limiter, the saturation adjustment, rain
        #       evaporation, cloud sedimentation and the final phase
        #       cleanup; read by exactly one terminal kernel.
        #   entry_density          -- WRF's entry rho at :1802, the density
        #       rc/nc/the ncten limiter/the terminal clamp are all formed
        #       on.  Distinct from mp_thompson_frozen_reference_density,
        #       which the saturation adjustment OVERWRITES mid-call.
        #   nwfa_entry_m3/nifa_entry_m3 -- the per-m3 entry aerosol of
        #       :1805-1812, consumed by scavenging, iceDeMott and iceKoop.
        #   tau1_density           -- the REFRESHED density of :3193.
        #   nwfa_work_m3           -- the :3211 working CCN snapshot, which
        #       is a genuinely different quantity from nwfa_entry_m3 (no
        #       9999E6 ceiling, tau+1 density) and feeds activ_ncloud only.
        #   qc_entry               -- frozen qc1d, required by the ncten
        #       balance limiter (:2996-3019), which needs BOTH the entry and
        #       the post-source cloud mass.
        #   ni_entry               -- frozen ni1d, credited to ncten by the
        #       cloud-ice melt branch of the final phase cleanup (:3943-3966).
        #   rc_entry/nc_entry_m3/nu_c_entry/l_qc_entry -- the outputs of the
        #       entry droplet-distribution diagnosis (:1826-1848), whose
        #       in-place side effect (zeroing qc1d/nc1d on the qc <= R1
        #       branch, :1844-1845) is what makes state.nc a legitimate
        #       "entry number" for every later kernel.
        #   condensation_rate      -- prw_vcd, held so rain evaporation can
        #       reproduce the :3502 gate that suppresses evaporation in a
        #       cell that just condensed.
        #
        # nu_c_entry / l_qc_entry are int32; every other row is float32.
        # Both are 4 bytes per element, so the byte estimate is unchanged by
        # the dtype and the registry keeps storing shapes only.
        slots.update(
            mp_thompson_aero_ncten=m,
            mp_thompson_aero_nwfaten=m,
            mp_thompson_aero_nifaten=m,
            mp_thompson_aero_entry_density=m,
            mp_thompson_aero_nwfa_entry_m3=m,
            mp_thompson_aero_nifa_entry_m3=m,
            mp_thompson_aero_tau1_density=m,
            mp_thompson_aero_nwfa_work_m3=m,
            mp_thompson_aero_qc_entry=m,
            mp_thompson_aero_ni_entry=m,
            mp_thompson_aero_rc_entry=m,
            mp_thompson_aero_nc_entry_m3=m,
            mp_thompson_aero_nu_c_entry=m,
            mp_thompson_aero_l_qc_entry=m,
            mp_thompson_aero_condensation_rate=m,
        )
    if cfg.mp_physics == 9:
        # milbrandt2.py::apply -- the WRF prep pair, the thirteen scratch
        # volumes the six kernels hand to one another (Part 1 leaves DZ/iDZ
        # frozen while Part 3 refreshes DE/iDE, so both pairs are named
        # slots rather than temporaries), the nine-slot precipitation row
        # its WRF driver arm binds, and the reflectivity the scheme writes
        # itself (module_microphysics_driver.F:1878 binds Zet to
        # refl_10cm, so refl_t is deliberately absent -- no generic radar
        # operator runs under mp=9).
        slots.update(my2_theta=m, my2_pii=m, my2_t=m, my2_z=m, my2_z8w=fl,
                     my2_psfc=s2,
                     my2_pres=m, my2_de=m, my2_ide=m, my2_dz=m, my2_idz=m,
                     my2_gamfact=m, my2_qsw=m, my2_qsi=m,
                     my2_qc_in=m, my2_qr_in=m, my2_nc_in=m, my2_nr_in=m,
                     mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2,
                     mp_snowncv=s2, mp_graupelnc=s2, mp_graupelncv=s2,
                     mp_hailnc=s2, mp_hailncv=s2,
                     mp_sr=s2, refl_10cm=m)
    if cfg.mp_physics == 10:
        # morrison.py:153-172 (prep + accumulators) + refl.py:324-325.
        slots.update(morr_theta=m, morr_rho=m, morr_pii=m, morr_dz=m,
                     morr_ice_to_snow=m, morr_z8w=fl,
                     mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2,
                     mp_snowncv=s2, mp_graupelnc=s2, mp_graupelncv=s2,
                     mp_sr=s2, refl_t=m, refl_10cm=m)
    if cfg.mp_physics:
        # microphysics.py spec-zone ring guard: per-edge snapshot buffers
        # for the WRF tile-clip exclusion (specified/nested only; exact
        # shapes from the single-source helper).
        # physics_inventory, not microphysics: that module imports cupy
        # at module scope and this registry is priced on CPU-only installs.
        from gpuwm.core.physics_inventory import spec_zone_ring_save_slots
        slots.update(spec_zone_ring_save_slots(cfg))
    if cfg.mp_physics == 18:
        # nssl2_runtime.py exact moist-physics prep, post-process radar
        # temperature, output handoff, and persistent precipitation state.
        # The mp_* prep names intentionally reuse the already classified
        # Kessler rebuild slots: their lifetime/shape contract is identical.
        slots.update(mp_th=m, mp_rho=m, mp_pii=m, mp_dz8w=m, mp_z8w=fl,
                     nssl2_driver_state=(16, nz, ny, nx),
                     nssl2_driver_surface_export=(5, ny, nx),
                     nssl2_driver_ignored_accumulator=s2,
                     nssl2_fused_temperature=m,
                     nssl2_primary_ice_target=m,
                     nssl2_nucond_ss=m, refl_t=m, refl_10cm=m,
                     mp_rainnc=s2, mp_rainncv=s2, mp_snownc=s2,
                     mp_snowncv=s2, mp_graupelnc=s2,
                     mp_graupelncv=s2, mp_hailnc=s2, mp_hailncv=s2,
                     mp_sr=s2)
        # gpuwm/da/obsop.py:_nssl_reflectivity H(x) temporaries: dry-air
        # density and (when no temperature is passed) diagnosed T, both
        # mass-shaped, filled and consumed inside one operator call.
        slots.update(da_nssl_rho=m, da_nssl_t=m)

    if cfg.km_opt in (2, 3, 4):
        slots.update(smag_km=m, smag_kh=m)
    if cfg.km_opt in (2, 3):
        # These closures carry the vertical exchange-coefficient pair; BN2
        # borrows the diff6_x face-workspace prefix and needs no slot.
        slots.update(smag_kmv=m, smag_khv=m)
    if cfg.km_opt == 2:
        # Prognostic-TKE forward tendency, its doubling temporary, and the
        # tke_rhs coupled-mass staging.
        slots.update(smag_rtke=m, smag_tke_tmp=m, smag_mut=s2)
        if getattr(cfg, "tke_budget", 0):
            # gpuwm/core/tke_budget.py: the packed per-term field buffer,
            # the pre-bound_tke carrier snapshot, the two coupled-mass
            # planes the reduction reads, the FP64 slab accumulator, and
            # its step counter.  Priced only when the diagnostic is on --
            # it is a report-only toggle, not part of any trajectory.
            from gpuwm.core.tke_budget import TERM_FIELDS, TERMS
            slots.update(
                tke_budget_terms=(len(TERM_FIELDS), nz, ny, nx),
                tke_budget_raw=m,
                tke_budget_mu0=s2, tke_budget_mu=s2,
                tke_budget_acc=(len(TERMS), nz),
                tke_budget_steps=(1,))
    if cfg.km_opt in (2, 3, 4) or cfg.diff_6th_opt:
        # dycore.py prepare_fixed_tendencies: carrying WRF forward
        # tendencies shared by Smagorinsky and sixth-order diffusion.
        slots.update(smag_ru=xs, smag_rv=ys, smag_rw=fl, smag_rth=m)
        if cfg.moist:
            for name in ("qv", "qc", "qr"):
                slots["smag_r" + name] = m
            if cfg.mp_physics in (6, 8, 9, 10, 16, 18, 28):
                for name in ("qi", "qs", "qg"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 9:
                # One held tendency per TRANSPORTED species beyond the
                # ice masses; the set is gpuwm/core/moist.py::MY2_SPECIES,
                # which is what prepare_fixed_tendencies iterates (1.9.1
                # D1's route: mp=9 had no arm here at all).
                for name in ("qh", "nc", "nr", "ni", "ns", "ng", "nh"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 16:
                for name in ("nn", "nc", "nr"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 8:
                for name in ("nr", "ni"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 28:
                # One held tendency per TRANSPORTED species; the set is
                # gpuwm/core/moist.py::THOMPSON_AERO_NUMBER_SPECIES, which
                # is what prepare_fixed_tendencies iterates.  nc is here
                # and is NOT here for mp=10, exactly as in moist.py.
                for name in ("nr", "ni", "nc", "nwfa", "nifa"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 10:
                for name in ("nr", "ni", "ns", "ng"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 50:
                # One held tendency per TRANSPORTED species, and the set is
                # gpuwm/core/moist.py::P3_SPECIES -- which is what
                # prepare_fixed_tendencies iterates.  qs/qg are absent for
                # the same reason they are absent from the state.
                for name in ("qi", "ni", "nr", "qir", "qib"):
                    slots["smag_r" + name] = m
            if cfg.mp_physics == 18:
                for name in ("qh", "qndrop", "qnr", "qni", "qns", "qng",
                             "qnh", "qnn", "qvolg", "qvolh"):
                    slots["smag_r" + name] = m
    if cfg.km_opt in (2, 3, 4) or cfg.diff_6th_opt:
        # Smagorinsky reuses the x/y face workspaces for u/v staging and
        # metric scalar fluxes (km_opt=3 additionally stages BN2 in the
        # diff6_x prefix during the K computation); sixth-order diffusion
        # subsequently overwrites them.  z/m are required only by diff6.
        slots.update(diff6_x=xs, diff6_y=ys)
    if cfg.diff_6th_opt:
        slots.update(diff6_z=fl, diff6_m=m)
    if cfg.khdif > 0.0 or cfg.kvdif > 0.0:
        slots.update(diff_u=xs, diff_v=ys, diff_w=fl, diff_th=m)

    from gpuwm.core.physics_inventory import physics_enabled
    if physics_enabled(cfg):
        slots["physics_qtot"] = m                   # physics.py:369
        if not cfg.moist:
            slots.update(physics_dry_qv=m, physics_dry_qc=m)
        # physics.py:1400-1409 substitutes a zero-filled scratch plane only
        # when the state has no qi/qs of its own, PER FIELD.  mp=28 and
        # mp=16 allocate both, so listing them there would price two full
        # 3-D fields the run never asks for; mp=50 (P3) allocates qi and
        # NOT qs, so it is the one scheme that needs exactly one of the
        # two -- the conditions are therefore split rather than sharing a
        # tuple.
        if cfg.mp_physics not in (6, 8, 10, 16, 18, 28, 50):
            slots["physics_qi"] = m                 # physics.py:1400-1404
        if cfg.mp_physics not in (6, 8, 10, 16, 18, 28):
            slots["physics_qs"] = m                 # physics.py:1405-1409
    if (int(cfg.bl_pbl_physics) in (1, 11)
            or int(cfg.mp_physics) in (1, 6, 8, 10, 16, 28)
            or int(cfg.cu_physics) == 1):
        # 28 is here for the same reason 6/8/10 are, and it was found by the
        # merge rather than by the port: physics.py:1284-1291 takes the
        # native-diagnostics arm for every scheme whose result IS the
        # canonical scratch set, and mp=28's adapter writes exactly the
        # seven canonical slots mp=8 does (physics.py:390-401).  18 is
        # absent because :1286 excludes it explicitly, not because NSSL
        # skips validation.  bl=11 rides the same word as bl=1: Shin-Hong
        # validates its outputs through the identical batched-status
        # policy (physics.py:_run_shinhong).
        slots["physics_validation_status"] = (1,)
    if int(cfg.bl_pbl_physics) == 5:
        # MYNN's whole working set, declared in
        # gpuwm/core/mynn_pbl_scratch.py.  Before it was declared the solver
        # allocated it fresh on every call and this registry knew none of it:
        # measured at nz = 49 on the RTX 5090, 46,160 bytes per column in 439
        # pool allocations per step, which is 15,847.6 MiB at the 360,000
        # columns of a d04 nest against a preflight estimate that did not
        # move at all.  A headroom check that reassuring on a card with no
        # ECC is a hazard, not a safeguard.
        #
        # These shapes are written against the COLUMN CHUNK, not ny*nx, which
        # is what makes the estimate bounded: mynn_pbl_runtime walks the
        # domain in chunks of that width and the split is bitwise identical
        # to the wide call.  Only the six returned A-grid tendency fields are
        # full width, because couple_ysu_tendencies consumes whole fields.
        slots.update(mynn_pbl_scratch_slots(cfg))
    if cfg.cu_physics:
        # physics.py:451-470 KF driver persistence (restart-serialized).
        slots.update(cu_rainc=s2, cu_nca=s2, cu_pratec=s2, cu_raincv=s2,
                     cu_expiring=s2,
                     cu_rthcuten=m, cu_rqvcuten=m, cu_rqccuten=m,
                     cu_rqicuten=m, cu_rqrcuten=m, cu_rqscuten=m)

    if cfg.specified:
        # lateral_bc.py resident attachment: held relax tendencies (:612),
        # Davies weights (:294), the MU boundary frame (:664), and the
        # packed eager forcing tables (:545) when interval count is known.
        slots.update(lbc_relax_u=xs, lbc_relax_v=ys, lbc_relax_theta=m,
                     lbc_relax_phi=fl)
        if cfg.moist:
            slots["lbc_qv_held"] = m
        slots["lbc_weights_0"] = (2, cfg.spec_bdy_width)
        slots[f"lbc_old_mup_frame_{cfg.spec_zone}"] = (
            _perimeter_count(ny, nx, cfg.spec_zone),)
        if cfg.specified and n_lbc_intervals > 0:
            slots["lbc_forcing_tables"] = (
                n_lbc_intervals * lbc_interval_values(cfg),)
    elif cfg.nested:
        # Rolling tables themselves live in the F4/F16 nest manifest.
        # Only the tiny Davies weights and MU finalizer frame use the legacy
        # LBC scratch registry. Nested held increments are recomputed from
        # RK time-t copies, so no extra full-domain carrying slots exist.
        frame_width = min(max(cfg.spec_zone, cfg.relax_zone + 1),
                          cfg.spec_bdy_width)
        slots["lbc_weights_0"] = (2, frame_width)
        slots[f"lbc_old_mup_frame_{cfg.spec_zone}"] = (
            _perimeter_count(ny, nx, cfg.spec_zone),)
        # One temporary is reused serially for u/v/w/theta/phi.  W is the
        # largest field only when horizontal extents exceed nz; retain the
        # true maximum so valid skinny/high-top grids remain capacity-safe.
        slots["lbc_nested_relax"] = _full_field_capacity(cfg)
    return slots


# ---------------------------------------------------------------------------
# F4 NEST ALLOCATION MANIFEST -- authoritative; Tasks 10/13 register slots
# matching these names/shapes EXACTLY (registry-equality gate; any drift is
# a plan amendment).
# ---------------------------------------------------------------------------

def nest_field_kinds(cfg: RunConfig) -> tuple[str, ...]:
    """Child forcing field kinds: u/v/w/t(thm)/ph/mu + ALL active
    moist/scalar species including Thompson/Morrison numbers (architecture
    section D;
    module_bc_em.F:320-345 w coupling; Registry scalar set qnr/qni/qns/
    qng at Registry.EM_COMMON:3026).

    ``nc`` is scheme-dependent, and the reason is the advection copy.  For
    mp_physics=10 it is EXCLUDED: Morrison allocates ``nc`` but has no
    ``nc0``, does not transport it, and diagnoses a fixed 250 cm-3 every
    call, so forcing it across a nest edge would carry a field the child
    immediately overwrites.  For mp_physics=28 it is INCLUDED: aerosol-aware
    Thompson makes droplet number prognostic, allocates ``nc0`` and advects
    it alongside nwfa/nifa (gpuwm/core/moist.py::THOMPSON_AERO_NUMBER_SPECIES),
    so it is a real forced boundary field like ``nr``/``ni``.  The two facts
    are not in tension -- the inventory follows what is transported, not what
    is allocated.

    This inventory contains only REAL prognostic fields handled by WRF
    ``copy_fcn`` (mass-cell or U/V face averaging).  It contains no
    masked/surface ``copy_fcnm`` fields and no integer ``copy_fcni`` fields;
    that is an explicit two-way-feedback scope divergence from stock WRF,
    not a request to apply the mass operator to masked state.
    """
    kinds = ["u", "v", "w", "t", "ph", "mu"]
    if cfg.moist:
        kinds += ["qv", "qc", "qr"]
        if cfg.mp_physics in (6, 8, 9, 10, 16, 18, 28):
            kinds += ["qi", "qs", "qg"]
        if cfg.mp_physics == 9:
            # The inventory follows what is transported
            # (gpuwm/core/moist.py::MY2_SPECIES): hail mass plus all six
            # number moments cross a nest edge with the masses they
            # describe (1.9.1 D1's route: mp=9 had no arm here, so a
            # nested Milbrandt-Yau child would have been forced with
            # WSM6's field set).
            kinds += ["qh", "nc", "nr", "ni", "ns", "ng", "nh"]
        if cfg.mp_physics == 16:
            kinds += ["nn", "nc", "nr"]
        if cfg.mp_physics == 8:
            kinds += ["nr", "ni"]
        if cfg.mp_physics == 28:
            kinds += ["nr", "ni", "nc", "nwfa", "nifa"]
        if cfg.mp_physics == 10:
            kinds += ["nr", "ni", "ns", "ng"]
        if cfg.mp_physics == 18:
            kinds += ["qh", "qndrop", "qnr", "qni", "qns", "qng",
                      "qnh", "qnn", "qvolg", "qvolh"]
        if cfg.mp_physics == 50:
            # P3 one-category: the inventory follows what is transported
            # (gpuwm/core/moist.py::P3_SPECIES), so the rime mass/volume
            # pair is forced across a nest edge exactly like the number
            # moments.  Forcing qi without them would hand the child ice
            # whose rime fraction and rime density came from whatever the
            # child's own last step left behind.
            kinds += ["qi", "ni", "nr", "qir", "qib"]
    return tuple(kinds)


def _kind_dims(kind: str, nz: int, ny: int, nx: int) -> tuple[int, int, int]:
    """(levels, ny-extent, nx-extent) of one field kind on the child grid."""
    if kind == "u":
        return (nz, ny, nx + 1)
    if kind == "v":
        return (nz, ny + 1, nx)
    if kind in ("w", "ph"):
        return (nz + 1, ny, nx)
    if kind == "mu":
        return (1, ny, nx)
    return (nz, ny, nx)


def _full_field_capacity(cfg: RunConfig) -> tuple[int, ...]:
    """Flat capacity of the largest full field for one domain."""
    shapes = (_kind_dims(kind, cfg.nz, cfg.ny, cfg.nx)
              for kind in nest_field_kinds(cfg))
    return (max(math.prod(shape) for shape in shapes),)


def nest_slot_shapes(dc: DomainConfig, spec_bdy_width: int,
                     parent: DomainConfig | None = None
                     ) -> dict[str, tuple[int, ...]]:
    """Every persistent ``nest_*`` device slot for ONE child domain.

    Three families (architecture section E item (d), F4/F16 amendments):

    * Rolling one-interval boundary tables, WRF Registry naming
      (``u_bxs``/``u_btxs`` style): per kind, per side, value
      (``nest_{kind}_b{xs,xe,ys,ye}``) and tendency
      (``nest_{kind}_bt{side}``).  x-sides ``(lev, ny_ext, W)``, y-sides
      ``(lev, W, nx_ext)`` -- the exact ``_field_boundary`` layout the
      Phase-4 resident-table CUDA machinery consumes
      (lateral_bc.py:149-167), refreshed every parent step with dtbc
      reset (mediation_force_domain.F semantics; registered deviation:
      rolling device tables instead of WRF's persistent host bdy
      arrays).
    * TWO simultaneously live force-local coupled fields
      (``nest_parent_field`` and ``nest_child_field``), shared through the
      sequential-domain arena when two distinct dead RK backings have enough
      capacity and otherwise allocated explicitly.  Each is overwritten
      before every ``bdy_interp1`` read.  F16 retires all four-side donor-strip
      rows.
    * SINT geometry: six arrays per stagger class, matching T10's landed
      ``device_tables`` exactly: ``ci/ip`` at ``nx_ext`` and ``cj/jp`` at
      ``ny_ext`` (int32), plus ``xig`` at ``nri`` and ``xjg`` at ``nrj``
      (float32).
    """
    run = dc.run
    nz, ny, nx = run.nz, run.ny, run.nx
    ratio = dc.parent_grid_ratio
    # interp_fcn.F:2517, identical to nest_interp.bdy_width().  The rolling
    # manifest must describe the exact arrays bdy_interp1 writes, even when
    # a case configures a wider maximum boundary allocation.
    width = min(max(int(run.spec_zone), int(run.relax_zone) + 1),
                int(spec_bdy_width))
    shapes: dict[str, tuple[int, ...]] = {}

    for kind in nest_field_kinds(run):
        lev, ny_ext, nx_ext = _kind_dims(kind, nz, ny, nx)
        for prefix in ("b", "bt"):
            shapes[f"nest_{kind}_{prefix}xs"] = (lev, ny_ext, width)
            shapes[f"nest_{kind}_{prefix}xe"] = (lev, ny_ext, width)
            shapes[f"nest_{kind}_{prefix}ys"] = (lev, width, nx_ext)
            shapes[f"nest_{kind}_{prefix}ye"] = (lev, width, nx_ext)

    parent_run = run if parent is None else parent.run
    shapes["nest_parent_field"] = _full_field_capacity(parent_run)
    shapes["nest_child_field"] = _full_field_capacity(run)

    for stag, (nx_ext, ny_ext) in (("m", (nx, ny)), ("x", (nx + 1, ny)),
                                   ("y", (nx, ny + 1))):
        shapes[f"nest_sint_ci_{stag}"] = (nx_ext,)
        shapes[f"nest_sint_ip_{stag}"] = (nx_ext,)
        shapes[f"nest_sint_cj_{stag}"] = (ny_ext,)
        shapes[f"nest_sint_jp_{stag}"] = (ny_ext,)
        shapes[f"nest_sint_xig_{stag}"] = (ratio,)
        shapes[f"nest_sint_xjg_{stag}"] = (ratio,)
    return shapes


def nest_slot_dtypes(dc: DomainConfig, spec_bdy_width: int,
                     parent: DomainConfig | None = None) -> dict[str, str]:
    """Semantic dtype of every F4/F16 nest slot (all are four bytes)."""
    shapes = nest_slot_shapes(dc, spec_bdy_width, parent)
    return {name: ("int32" if name.startswith(("nest_sint_ci_",
                                                "nest_sint_ip_",
                                                "nest_sint_cj_",
                                                "nest_sint_jp_"))
                   else "float32")
            for name in shapes}


def nest_allocation_manifest(exp: ExperimentConfig
                             ) -> dict[int, dict[str, tuple[int, ...]]]:
    """The frozen nest allocation manifest: ``grid_id -> {slot: shape}``
    for every CHILD domain.  N0's ``--alloc`` allocates exactly these
    entries as real device allocations (F4: the residency proof covers
    the actual coupler footprint, not proxies)."""
    manifest: dict[int, dict[str, tuple[int, ...]]] = {}
    by_id = {dc.grid_id: dc for dc in exp.domains}
    for dc in exp.domains:
        if dc.parent_id == 0:
            continue
        manifest[dc.grid_id] = nest_slot_shapes(
            dc, exp.spec_bdy_width, by_id[dc.parent_id])
    return manifest


# ---------------------------------------------------------------------------
# Scratch-slot lifetime audit (architecture section E lever 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScratchSlotLifetime:
    """One reviewed row in the scratch sharing admission table.

    Patterns ending in ``*`` classify a generated slot family.  ``kind`` is
    either ``write_before_read`` (safe on the sequential-domain arena),
    ``carrying`` (observed after the producing step/call), or
    ``excluded_unproven`` (correctness-first exclusion).
    """

    patterns: tuple[str, ...]
    kind: str
    evidence: str
    rationale: str

    @property
    def arena_eligible(self) -> bool:
        return self.kind == "write_before_read"


# Committed audit table. tests/test_preflight.py expands every registry and
# F4-manifest slot through this table, rejects gaps/overlaps, and separately
# proves that every arena-admitted row is write-before-read classified.
SCRATCH_SLOT_LIFETIME_AUDIT = (
    ScratchSlotLifetime(
        ("rk_ww", "rk_ru", "rk_rv", "rk_ru_m", "rk_rv_m", "rk_ww_m"),
        "write_before_read",
        "gpuwm/core/dycore.py:102-113,154-161,1373-1414; "
        "gpuwm/core/nest.py:_coupled_child_field",
        "stage/moist fluxes are filled before stage reads; FORCE borrows "
        "the matching rk_ru/rk_rv/rk_ww staggered backing only after the "
        "prior step and before the next stage rewrite"),
    ScratchSlotLifetime(
        ("adv_ru", "adv_rv", "adv_rw"), "write_before_read",
        "gpuwm/core/advection.py:161-180",
        "the advection-only path fills all three flux arrays before launch"),
    ScratchSlotLifetime(
        ("acoustic_mu_pp_old", "acoustic_th_pp_old", "acoustic_c2a",
         "acoustic_a", "acoustic_alpha", "acoustic_gamma", "acoustic_mudf",
         "acoustic_cqu", "acoustic_cqv", "acoustic_cqw",
         "openbc_upp_faces", "openbc_vpp_faces"),
        "write_before_read",
        "gpuwm/core/acoustic.py:115-190,223-235; gpuwm/core/dycore.py:1360-1395",
        "substep histories/coefficient/filter slots are seeded in their stage"),
    ScratchSlotLifetime(
        ("smag_km", "smag_kh", "smag_ru", "smag_rv", "smag_rw",
         "smag_rth", "smag_rqv", "smag_rqc", "smag_rqr", "smag_rqi",
         "smag_rqs", "smag_rqg", "smag_rnr", "smag_rni", "smag_rns",
         # mp=16 (WDM6) transported number moments.  smag_rnc/smag_rnr are
         # already above -- Morrison named them first -- so only the CCN
         # reservoir is new.
         "smag_rnn",
         # mp=9 (Milbrandt-Yau) hail number.  The rest of its transported
         # set was already audited here by the schemes that named the
         # slots first; nh is the one name no other scheme transports
         # (1.9.1 D1's route, third table: registered by
         # scratch_slot_registry without a row here, invisible until a
         # TREE reached shared_scratch_arena_shapes -- the identical
         # class as the km_opt=2/3 row below).
         "smag_rnh",
         "smag_rng", "smag_rqh", "smag_rqndrop", "smag_rqnr",
         "smag_rqni", "smag_rqns", "smag_rqng", "smag_rqnh",
         "smag_rqnn", "smag_rqvolg", "smag_rqvolh",
         # mp=28 transported number/aerosol moments.  Same construction,
         # same lifetime: prepare_fixed_tendencies writes each held
         # tendency once before the RK loop and every stage only reads it.
         "smag_rnc", "smag_rnwfa", "smag_rnifa",
         # mp=50 (P3) rime mass and rime volume.  Same construction, same
         # lifetime: they are transported scalars like the number moments,
         # so prepare_fixed_tendencies writes each held tendency once
         # before the RK loop and every stage only reads it.
         "smag_rqir", "smag_rqib"),
        "write_before_read",
        "gpuwm/core/dycore.py:prepare_fixed_tendencies",
        "time-t K and its held tendencies are written and consumed before "
        "the RK loop; K is dead before acoustic alpha/gamma overwrite the "
        "borrowed backings, while all three RK stages read only the held "
        "tendencies"),
    # The km_opt=2/3 closure's own slots.  They were registered by
    # scratch_slot_registry (km_opt in (2, 3) above) without a row here,
    # which is invisible to a single-domain run -- only a TREE reaches
    # shared_scratch_arena_shapes, and no LES tree had been built.  The
    # first nested LES domain hit it as
    # `KeyError: scratch slot 'smag_kmv' has no lifetime audit row`.
    ScratchSlotLifetime(
        ("smag_kmv", "smag_khv", "smag_rtke", "smag_tke_tmp", "smag_mut"),
        "write_before_read",
        "gpuwm/core/dycore.py:836-837,928-929 (written by "
        "launch_wrf_tke_km / launch_wrf_smag3d_km), :1207-1208 (read by "
        "launch_wrf_smag2d_vertical), :1288 (smag_rtke zeroed); "
        "gpuwm/core/moist.py:advance_tke_stage",
        "the vertical exchange pair is filled and consumed inside one "
        "prepare_fixed_tendencies call exactly as the horizontal "
        "smag_km/smag_kh pair above, and smag_rtke is a held time-t "
        "tendency on the same footing as the smag_r* row -- zeroed before "
        "the RK loop, read by all three stages, never carried across a "
        "step; smag_tke_tmp and smag_mut are within-call staging"),
    ScratchSlotLifetime(
        ("tke_budget_terms", "tke_budget_raw", "tke_budget_mu0",
         "tke_budget_mu"), "write_before_read",
        "gpuwm/core/tke_budget.py:clear_fields,accumulate; "
        "gpuwm/core/dycore.py:1288,2380",
        "clear_fields zeroes the packed term buffer at the top of the "
        "step and accumulate folds it, the pre-bound_tke snapshot and the "
        "two coupled-mass planes into the accumulator before the same "
        "step() returns"),
    ScratchSlotLifetime(
        ("tke_budget_acc", "tke_budget_steps"), "carrying",
        "gpuwm/core/tke_budget.py:accumulator,reset,accumulate,drain",
        "the window accumulator and its step counter are read many steps "
        "after the step that wrote them (drain ends the window), and both "
        "are float64 while ScratchArena is float32-only"),
    ScratchSlotLifetime(
        ("diff_u", "diff_v", "diff_w", "diff_th"), "write_before_read",
        "gpuwm/core/diffusion.py:121-130",
        "each constant-K temporary is zeroed and filled before accumulation"),
    ScratchSlotLifetime(
        ("diff6_x", "diff6_y", "diff6_z", "diff6_m"),
        "write_before_read", "gpuwm/core/dycore.py:prepare_fixed_tendencies; "
        "gpuwm/core/dycore.py:apply_diff6",
        "the diff6 target loops consume one temporary at a time; however, "
        "km_opt=4 uses x/y simultaneously for momentum staging and for "
        "scalar face fluxes, so every Smagorinsky configuration retains "
        "two distinct face backings"),
    ScratchSlotLifetime(
        ("moist_pd_q0", "moist_rq_t", "moist_absent_mass",
         "pd_fxl", "pd_fxc", "pd_fyl",
         "pd_fyc", "pd_fzl", "pd_fzc"), "write_before_read",
        "gpuwm/core/moist.py:197-203,247-308; "
        "gpuwm/core/moist.py::absent_mass_plane",
        "source copies, tendencies, and six PD fluxes are filled before "
        "use; the absent-mass plane is zeroed inside the same call that "
        "hands it to calc_cq or slow_buoyancy, immediately before the "
        "launch, so no arena neighbour can be observed through it"),
    ScratchSlotLifetime(
        ("mp_th", "mp_rho", "mp_pii", "mp_z", "mp_dz8w", "mp_z8w",
         "nssl2_driver_state", "nssl2_driver_surface_export",
         "nssl2_driver_ignored_accumulator",
         "nssl2_fused_temperature", "nssl2_primary_ice_target",
         "nssl2_nucond_ss"),
        "write_before_read", "gpuwm/core/microphysics.py:143-169; "
        "gpuwm/core/nssl2_runtime.py:_prepare_fields; "
        "gpuwm/core/nssl2_fused_gs.py:launch_fused_gs; "
        "gpuwm/core/kernels/nssl2_fused_gs.cu:nssl2_prepare_fused_gs; "
        "gpuwm/core/nssl2_nucond.py:139-143; "
        "gpuwm/core/kernels/nssl2_nucond.cu:96-107",
        "Kessler and NSSL preparation fields are rebuilt for every scheme "
        "call; gather overwrites the driver state and sediment export planes "
        "while the ignored accumulator is explicitly reset before RMW; the "
        "fused-GS prepass overwrites temperature and primary-ice target "
        "snapshots before the process kernel reads them; NUCOND overwrites "
        "its supersaturation filter before reading it"),
    ScratchSlotLifetime(
        ("p3_vmi", "p3_di", "p3_rhopo"), "write_before_read",
        "gpuwm/core/p3.py:1749-1751 (allocation), :p3_main diagnostic pass; "
        "phys/module_mp_p3.F:1965-1967 (intent(out)), :2282-2284 (zeroed on "
        "entry), :4856-4858 (written from the post-update ice state)",
        "P3's three ice diagnostics are intent(out) of p3_main: it zeroes "
        "them on entry and refills them from the updated ice state before "
        "returning, so nothing in them survives a call boundary and the "
        "driver's read-back always follows that call's own write"),
    ScratchSlotLifetime(
        ("mp_thompson_temperature",
         "mp_thompson_frozen_reference_density",
         "mp_thompson_frozen_reference_temperature",
         "mp_thompson_rain_reference_density",
         "mp_thompson_snow_melt_marker",
         "mp_thompson_graupel_melt_marker",
         "mp_thompson_snow_velocity_boost",
         "mp_thompson_graupel_number_shadow"),
        "write_before_read",
        "gpuwm/core/microphysics.py:_apply_thompson; "
        "gpuwm/core/kernels/thompson.cu:129-217,1999-2020,5674-5682",
        "the adapter fills temperature before its first consumer; cloud/rain "
        "reference kernels write every cell before fallout; the warm source "
        "writes both held melt markers for every cell before their consumers; "
        "the fused cold source resets every velocity boost; output-due "
        "graupel number is initialized across the complete field before "
        "source/fallout reads"),
    # mp=28's aerosol working set.  WRITE-BEFORE-READ, and the evidence is
    # structural rather than incidental: WRF's own column loop freezes the
    # entry state and ZEROES the three accumulators at the top of every call
    # (module_mp_thompson.F:1679-1681), so the adapter must explicitly fill
    # every one of these at entry before any network runs.  That is not a
    # convenience -- gpuwm/core/state.py's scratch pool persists across
    # steps by design, so an accumulator that were merely "usually
    # overwritten" would carry the previous step's aerosol tendency forward
    # as a slow, bounded, entirely plausible-looking drift that no
    # single-step column test could see.  The entry snapshots are written by
    # the entry kernels over the complete field (no branch leaves a cell
    # unassigned) before the first consumer, and the terminal apply/clamp
    # reads each accumulator exactly once.
    ScratchSlotLifetime(
        ("mp_thompson_aero_ncten",
         "mp_thompson_aero_nwfaten",
         "mp_thompson_aero_nifaten",
         "mp_thompson_aero_entry_density",
         "mp_thompson_aero_nwfa_entry_m3",
         "mp_thompson_aero_nifa_entry_m3",
         "mp_thompson_aero_tau1_density",
         "mp_thompson_aero_nwfa_work_m3",
         "mp_thompson_aero_qc_entry",
         "mp_thompson_aero_ni_entry",
         "mp_thompson_aero_rc_entry",
         "mp_thompson_aero_nc_entry_m3",
         "mp_thompson_aero_nu_c_entry",
         "mp_thompson_aero_l_qc_entry",
         "mp_thompson_aero_condensation_rate"),
        "write_before_read",
        "gpuwm/core/thompson_aerosol_state.py:"
        "zero_aerosol_accumulators,launch_aerosol_entry_snapshot,"
        "launch_aerosol_entry_cloud_number,launch_tau1_density,"
        "launch_aerosol_working_number; "
        "gpuwm/core/kernels/thompson_aerosol_state.cu",
        "the adapter zeroes the three accumulators and fills every entry "
        "snapshot at call entry, before any source network reads them; the "
        "terminal state-finalize kernel is the single consumer of the "
        "accumulators and runs after every writer"),
    ScratchSlotLifetime(
        ("wsm6_theta", "wsm6_rho", "wsm6_pii", "wsm6_dz",
         "wsm6_z8w"), "write_before_read",
        "gpuwm/core/wsm6.py:89-98",
        "WSM6 preparation fully assigns each array before the scheme launch "
        "or any dependent read"),
    ScratchSlotLifetime(
        ("wdm6_theta", "wdm6_rho", "wdm6_pii", "wdm6_dz",
         "wdm6_z8w"), "write_before_read",
        "gpuwm/core/wdm6.py:apply",
        "WDM6 preparation fully assigns each array before the scheme launch "
        "or any dependent read"),
    ScratchSlotLifetime(
        ("morr_theta", "morr_rho", "morr_pii", "morr_dz",
         "morr_ice_to_snow", "morr_z8w"), "write_before_read",
        "gpuwm/core/morrison.py:159-202",
        "Morrison preparation fields are rebuilt for every scheme call"),
    ScratchSlotLifetime(
        ("my2_theta", "my2_pii", "my2_t", "my2_z", "my2_z8w", "my2_psfc"),
        "write_before_read", "gpuwm/core/milbrandt2.py::apply",
        "Milbrandt-Yau preparation assigns every element of each array "
        "before the launch, and the surface pressure is derived from the "
        "same call's geopotential"),
    ScratchSlotLifetime(
        ("my2_pres", "my2_de", "my2_ide", "my2_gamfact", "my2_qsw",
         "my2_qsi", "my2_qc_in", "my2_qr_in", "my2_nc_in", "my2_nr_in"),
        "write_before_read", "gpuwm/core/kernels/milbrandt2.cu"
        "::milbrandt2_prelim",
        "the Part 1 kernel writes every one of these for every cell before "
        "any later kernel reads them; entry contents are never consulted"),
    ScratchSlotLifetime(
        ("my2_dz", "my2_idz"), "write_before_read",
        "gpuwm/core/kernels/milbrandt2.cu::milbrandt2_geometry",
        "the geometry kernel writes both for every cell from the Part 1 "
        "density and pressure, before the cold/warm/sedimentation kernels "
        "read them"),
    ScratchSlotLifetime(
        ("refl_t",), "write_before_read", "gpuwm/core/refl.py:349-372; "
        "gpuwm/core/nssl2_runtime.py:apply_nssl2_production",
        "post-scheme temperature preparation is assigned after the final "
        "condensation hook and before the reflectivity launch"),
    ScratchSlotLifetime(
        ("physics_qtot", "physics_qi", "physics_qs"), "write_before_read",
        "gpuwm/core/physics.py:387-414",
        "physics preparation explicitly zeroes these arrays before reads"),
    ScratchSlotLifetime(
        ("physics_validation_status",), "write_before_read",
        "gpuwm/core/physics.py:_run_ysu; "
        "gpuwm/core/physics.py:_run_shinhong; "
        "gpuwm/core/physics.py:_validate_native_microphysics; "
        "gpuwm/core/physics.py:_validate_native_kf_result; "
        "gpuwm/core/ysu.py:validate_ysu_outputs; "
        "gpuwm/core/shinhong.py:invalid_shinhong_outputs",
        "the float32 backing is viewed as uint32 and reset before every "
        "validation launch; its blocking scalar read completes before the "
        "next sequential domain can reuse the shared-arena word"),
    ScratchSlotLifetime(
        ("lbc_qv_held", "lbc_relax_u", "lbc_relax_v",
         "lbc_relax_theta", "lbc_relax_phi"),
        "write_before_read",
        "gpuwm/core/moist.py:263-281; gpuwm/ingest/lateral_bc.py:611-628",
        "RK stage 1 captures held tendencies before later stages consume them"),
    ScratchSlotLifetime(
        ("lbc_old_mup_frame_*",), "write_before_read",
        "gpuwm/ingest/lateral_bc.py:658-728",
        "the MU install kernel writes the frame before fused field finalizers"),
    ScratchSlotLifetime(
        ("lbc_nested_relax",), "write_before_read",
        "gpuwm/ingest/lateral_bc.py:apply_state_lateral_boundaries; "
        "gpuwm/core/acoustic.py:prepare_acoustic_coefficients",
        "each nested field overwrites the temporary before immediate use; "
        "the following acoustic preparation overwrites its aliased backing "
        "before any acoustic read"),
    ScratchSlotLifetime(
        ("cu_expiring",), "excluded_unproven",
        "gpuwm/core/physics.py:_advance_cumulus_clock,finish_step",
        "the mask is cleared immediately after the pre-RK compose, but "
        "internal substeps can read that carried zero without a same-step "
        "overwrite; retain per-domain "
        "identity rather than overclaim write-before-read sharing"),
    # EXCLUDED (unproven): each snapshot is fully written at capture and
    # read back at restore within one microphysics.apply call, but the
    # whole scheme dispatch (which allocates and writes its own scratch)
    # runs BETWEEN that write and read -- sharing a backing with any
    # dispatch-written slot would corrupt the ring restore.  Tiny
    # (~2*(nx+ny)*nz per field); correctness beats the savings.
    ScratchSlotLifetime(
        ("mp_ring_save_*",), "excluded_unproven",
        "gpuwm/core/microphysics.py:_capture_spec_zone_ring,"
        "_restore_spec_zone_ring",
        "the snapshot must survive the full scheme dispatch between its "
        "capture write and restore read; retain per-domain identity"),
    ScratchSlotLifetime(
        ("integration_health_partial", "integration_health_result"),
        "write_before_read",
        "gpuwm/core/dycore.py:1474-1486",
        "the two reduction kernels overwrite their outputs before host read"),
    # EXCLUDED (carrying): T15's validator launch tables are immutable
    # setup-time category maps filled once in HealthValidator.__init__
    # (gpuwm/core/health.py:476-491, "immutable setup-time category maps")
    # and reused every step; arena-sharing across domains would corrupt
    # them.  Tiny (~48 KB/domain); classified at the T15 merge.
    ScratchSlotLifetime(
        ("integration_health_field_ptr", "integration_health_aux_ptr",
         "integration_health_field_size", "integration_health_bounds",
         "integration_health_flags", "integration_health_planes",
         "integration_health_status_bits", "integration_health_validation"),
        "carrying",
        "gpuwm/core/health.py:476-496",
        "setup-time launch tables persist across steps, and the status/"
        "result buffers may be read asynchronously by supervision between "
        "steps; per-domain only (conservative; ~49 KB/domain)"),

    # EXCLUDED (carrying): these microphysics accumulators are read-modify-
    # write and restart-serialized (restart.py:148-158; kessler.cu:91-92;
    # morrison.cu:1087-1103), so per-domain identity must be preserved.
    ScratchSlotLifetime(
        ("mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
         "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
         "mp_sr", "mp_kessler_sr"),
        "carrying", "gpuwm/io/restart.py:148-158",
        "restart-visible live microphysics accumulator/diagnostic state"),
    # EXCLUDED (carrying): KF initialization stores these arrays on the
    # driver and later calls update them (physics.py:474-500); restart.py:
    # 148-158 serializes the same slots.
    ScratchSlotLifetime(
        ("cu_rainc", "cu_nca", "cu_pratec", "cu_raincv", "cu_rthcuten",
         "cu_rqvcuten", "cu_rqccuten", "cu_rqicuten", "cu_rqrcuten",
         "cu_rqscuten"), "carrying",
        "gpuwm/core/physics.py:474-500; gpuwm/io/restart.py:148-158",
        "KF timers, rates, and precipitation persist across scheme calls"),
    # EXCLUDED (carrying): the output driver retains this exact view after the
    # producing step until consumption (physics.py:468-472; refl.py:376-399).
    ScratchSlotLifetime(
        ("refl_10cm",), "carrying", "gpuwm/core/refl.py:376-399",
        "one-frame output handoff can outlive the producing domain step"),
    # DA reflectivity-operator temporaries: both are filled in full at the
    # top of _nssl_reflectivity (rho from 1/alt, T from theta and Exner)
    # before diagnose_radardd02_if_due reads them, inside one H(x) call.
    ScratchSlotLifetime(
        ("da_nssl_rho", "da_nssl_t"), "write_before_read",
        "gpuwm/da/obsop.py:_nssl_reflectivity",
        "the observation operator fills dry-air density and temperature "
        "before the shared NSSL diagnostic reads them; both are dead when "
        "the call returns"),
    # EXCLUDED (carrying): UP_HELI_MAX is a restart-serialized elementwise
    # running max, read-modify-written every step and consumed by history
    # frames; per-domain identity must be preserved.
    ScratchSlotLifetime(
        ("up_heli_max",), "carrying",
        "gpuwm/core/uh_diag.py:update_up_heli_max,reset_up_heli_max; "
        "gpuwm/io/restart.py:SERIALIZED_SCRATCH_SLOTS",
        "restart-visible running-max accumulator with a reset only at "
        "history writes"),
    # UP_HELI_MAX work planes: uh_columns writes every cell of both planes
    # (edge threads included) before uh_smooth_max reads them, all inside
    # one update_up_heli_max call.
    # The consumer-owned tracking windows: the same running-max operator
    # as up_heli_max, folded in the same pass, but reset by the consumer
    # that read them rather than by the history writer, and NOT restart
    # visible -- a restart starts them empty and the first post-restart
    # evaluation may under-read, which is the tolerated-experiment
    # posture the moving-nest and spawn restart rulings already take.
    ScratchSlotLifetime(
        ("uh_follow_window", "uh_spawn_window"), "carrying",
        "gpuwm/core/uh_diag.py:update_up_heli_max,reset_tracker_window; "
        "gpuwm/io/restart.py:CARRIED_SCRATCH_SLOTS",
        "per-consumer running-max windows, reset at every evaluation of "
        "the consumer that owns them and never emitted"),
    ScratchSlotLifetime(
        ("uh_diag_col", "uh_diag_use"), "write_before_read",
        "gpuwm/core/kernels/uh_diag.cu:uh_columns,uh_smooth_max; "
        "gpuwm/core/uh_diag.py:update_up_heli_max",
        "per-launch column UH and use_column planes, fully rewritten by "
        "every diagnostic call before the smoother reads them"),
    # EXCLUDED (unproven): unlike physics_qi, these dry placeholders are only
    # zero-initialized by DomainState.scratch and are not rewritten in
    # _prepare_atmosphere (physics.py:404-414). Correctness beats tiny savings.
    ScratchSlotLifetime(
        ("physics_dry_qv", "physics_dry_qc"), "excluded_unproven",
        "gpuwm/core/physics.py:404-414",
        "constant-zero placeholders lack a per-step write-before-read"),
    # EXCLUDED (unproven): the column-batch adapter pair is fully rewritten
    # by each run_phase2 call (cp.divide/cp.multiply with out=) before the
    # microphysics dispatch reads it, but the whole WSM6 dispatch -- which
    # allocates and writes its own scratch -- runs BETWEEN that write and
    # the scheme's reads, the mp_ring_save_* hazard.  The seam never builds
    # a shared arena anyway; correctness beats the savings.
    ScratchSlotLifetime(
        ("physics_column_alt", "physics_column_php"), "excluded_unproven",
        "gpuwm/core/mpas_column_batch.py:769-774",
        "the adapter pair must survive the WSM6 dispatch between its write "
        "and the scheme's reads; retain per-domain identity"),
    # EXCLUDED (carrying/setup): weights are cached by key and forcing views
    # remain attached across all steps (lateral_bc.py:282-301,535-577).
    ScratchSlotLifetime(
        ("lbc_weights_*", "lbc_forcing_tables"), "carrying",
        "gpuwm/ingest/lateral_bc.py:282-301,535-577",
        "resident forcing tables and cached weights are cross-step setup"),
    # MYNN's declared workspace.  Split three ways on purpose.
    #
    # WRITE-BEFORE-READ: the kernel that owns the slot fills the whole
    # requested prefix before anything reads it, verified against the CUDA
    # source rather than assumed.  Two of them needed a code change to earn
    # the classification: mynn_pbl.cu:945 returns before k == 0, so the nine
    # mym_turbulence products and the seven full-column level-2 fields kept
    # the surface zero of a fresh allocation.  mynn_pbl_gpu now zeroes that
    # one element explicitly.  MynnPblScratch.poison() is the runtime lever
    # and tests/test_mynn_pbl_scratch.py drives it: NaN every slot in this
    # group, run a carried forecast, require the same hash.
    ScratchSlotLifetime(
        ("mynn_pbl_prep", "mynn_pbl_zw", "mynn_pbl_surface",
         "mynn_pbl_delt", "mynn_pbl_diss_heat", "mynn_pbl_exchange",
         "mynn_pbl_pblh", "mynn_pbl_level2_pairs", "mynn_pbl_level2_out",
         "mynn_pbl_level2_full", "mynn_pbl_mixlength",
         "mynn_pbl_mixlength_work", "mynn_pbl_turbulence",
         "mynn_pbl_predict", "mynn_pbl_predict_work",
         "mynn_pbl_condensation", "mynn_pbl_initialize",
         "mynn_pbl_initialize_work", "mynn_pbl_plume_layer",
         "mynn_pbl_plume_face", "mynn_pbl_plume_column",
         "mynn_pbl_plume_work", "mynn_pbl_plume_scratch",
         "mynn_pbl_tendency", "mynn_pbl_tendency_work",
         "mynn_pbl_tendency_face", "mynn_pbl_stage_layer",
         "mynn_pbl_stage_dx", "mynn_pbl_out_du", "mynn_pbl_out_dv",
         "mynn_pbl_out_dtheta", "mynn_pbl_out_dqv", "mynn_pbl_out_dqc",
         "mynn_pbl_out_dqi"),
        "write_before_read",
        "gpuwm/core/mynn_pbl_scratch.py; gpuwm/core/mynn_pbl_gpu.py; "
        "gpuwm/core/kernels/mynn_pbl.cu:945,2482-2489,2799-2821,1400-1406",
        "each solver leaf fills its own outputs and work vectors before any "
        "reader; the six returned A-grid tendency fields are fully written "
        "across the chunk walk and are dead the moment "
        "couple_ysu_tendencies has multiplied them into new arrays",
    ),
    # EXCLUDED (constant): nothing writes these.  WRF passes them to systems
    # this lane's pinned identity switches off, and every reader requires
    # them to be zero -- so they are read-before-write by construction, the
    # same reason physics_dry_qv/physics_dry_qc are excluded above.  Sharing
    # a backing with a slot that IS written would put a nonzero mass flux
    # into a tendency solve that is supposed to have none.
    ScratchSlotLifetime(
        ("mynn_pbl_zero_layer", "mynn_pbl_zero_face",
         "mynn_pbl_zero_column", "mynn_pbl_plume_zero_layer",
         "mynn_pbl_plume_zero_face", "mynn_pbl_tendency_zero"),
        "excluded_unproven",
        "gpuwm/core/mynn_pbl_scratch.py:MYNN_PBL_CONSTANT_ZERO_SLOTS; "
        "gpuwm/core/mynn_pbl_gpu.py:mynn_bl_driver_cuda",
        "constant-zero feeds for the mass-flux, subsidence, detrainment, "
        "snow, ozone, stochastic and ocean-current systems this identity "
        "disables; they have no per-step write to be before",
    ),
    # EXCLUDED (int32): ScratchArena is float32-only by construction, so an
    # index slot cannot draw a view from it.  Excluding them here is what
    # makes DomainState.scratch fall back to a per-state int32 allocation
    # instead of raising on the dtype.
    ScratchSlotLifetime(
        ("mynn_pbl_kpbl", "mynn_pbl_pblh_index", "mynn_pbl_plume_index",
         "mynn_pbl_validity_flags"),
        "excluded_unproven",
        "gpuwm/core/state.py:ScratchArena.view; "
        "gpuwm/core/mynn_pbl_scratch.py:MYNN_PBL_INDEX_SLOTS",
        "int32 one-based level indices and the validity words; the shared "
        "arena admits float32 only",
    ),
    ScratchSlotLifetime(
        ("nest_parent_field", "nest_child_field"), "write_before_read",
        "gpuwm/core/nest.py:_coupled_parent_field,_coupled_child_field,force",
        "each field coupling overwrites the full requested prefix before "
        "bdy_interp1 reads it; the two simultaneous fields use distinct "
        "backings and FORCE ends before any aliased RK-stage read"),
    ScratchSlotLifetime(
        tuple(f"nest_{kind}_b*" for kind in
              ("u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
               "qi", "qs", "qg", "nr", "ni", "ns", "ng", "qh",
               "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
               "qvolg", "qvolh")),
        "carrying", "gpuwm/core/nest.py:force; "
        "gpuwm/ingest/lateral_bc.py:attach_nest_boundaries",
        "rolling value/tendency frames are consumed through the complete "
        "child subcycle and must remain child-owned"),
    ScratchSlotLifetime(
        ("nest_sint_*",), "carrying",
        "gpuwm/core/nest.py:_bind_geometry; "
        "gpuwm/core/nest_interp.py:NestRegistration.device_tables",
        "setup geometry is bound once and reused by every force"),
)


def scratch_slot_lifetime(slot: str) -> ScratchSlotLifetime | None:
    """Return the unique audit row for ``slot``, or ``None`` if unclassified."""
    matches = []
    for row in SCRATCH_SLOT_LIFETIME_AUDIT:
        if any((slot.startswith(pattern[:-1]) if pattern.endswith("*")
                else slot == pattern) for pattern in row.patterns):
            matches.append(row)
    if len(matches) > 1:
        raise RuntimeError(f"scratch lifetime audit overlaps for {slot!r}")
    return matches[0] if matches else None


def scratch_slot_uses_arena(slot: str) -> bool:
    row = scratch_slot_lifetime(slot)
    if row is None:
        raise KeyError(f"scratch slot {slot!r} has no lifetime audit row")
    return row.arena_eligible


def shared_scratch_arena_shapes(
        domains: tuple[DomainConfig, ...]
        ) -> dict[str, tuple[int, ...]]:
    """Max request shape per admitted slot for the sequential-domain arena.

    Max is by element count because :class:`ScratchArena` returns contiguous
    reshaped prefix views. Ties retain the first (parent-first) request. The
    runtime builder and the estimator both call this exact function.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for dc in domains:
        for slot, shape in scratch_slot_registry(
                dc.run, n_lbc_intervals=0).items():
            if not scratch_slot_uses_arena(slot):
                continue
            shape = tuple(shape)
            if slot not in shapes or math.prod(shape) > math.prod(shapes[slot]):
                shapes[slot] = shape
    if all(hasattr(dc, "grid_id") and hasattr(dc, "parent_id")
           for dc in domains):
        by_id = {dc.grid_id: dc for dc in domains}
        for dc in domains:
            if dc.parent_id == 0:
                continue
            force_shapes = {
                "nest_parent_field": _full_field_capacity(
                    by_id[dc.parent_id].run),
                "nest_child_field": _full_field_capacity(dc.run),
            }
            for slot, shape in force_shapes.items():
                if (slot not in shapes
                        or math.prod(shape) > math.prod(shapes[slot])):
                    shapes[slot] = shape
    return shapes


def shared_scratch_arena_aliases(
        domains: tuple[DomainConfig, ...]) -> dict[str, str]:
    """Disjoint-lifetime arena aliases admitted by the reviewed audit.

    FORCE occurs between complete domain steps.  All three RK backings are
    dead after the preceding STEP and overwritten before their next stage
    read.  Parent and child coupled fields are simultaneously live, so the
    alias assignment below admits them only onto two distinct capacity-safe
    RK backings.  Preference order preserves F16's historical parent->rk_ru,
    child->rk_ww addresses on ordinary production grids; an explicit logical
    backing remains when no safe pair exists.

    Sixth-order diffusion itself processes targets serially.  On a diff6-only
    configuration x/y/m can all reuse the largest z backing.  The combined
    Smagorinsky path also borrows x/y, and needs both face buffers
    simultaneously for momentum staging and scalar fluxes.  If any domain
    enables km_opt=4, x and y therefore remain distinct while x/m may still
    reuse z.

    Smagorinsky K_m/K_h are consumed only while the pre-RK preparation builds
    the held mixing tendencies.  Each stage later prepares acoustic alpha and
    gamma from scratch before its first acoustic read, and no stage reads K.
    The two mass-point K arrays can therefore borrow prefixes of those two
    independent full-level coefficient backings.
    """
    aliases = {}
    shapes = shared_scratch_arena_shapes(domains)
    # The standalone advection-only adv_* path and the acoustic RK path are
    # mutually exclusive inside dycore.step.  cq overwrites all three faces
    # once per RK stage before calc_coefs/advance_uv/advance_w reads them.
    for cq, adv in (("acoustic_cqu", "adv_ru"),
                    ("acoustic_cqv", "adv_rv"),
                    ("acoustic_cqw", "adv_rw")):
        if (cq in shapes and adv in shapes
                and math.prod(shapes[cq]) <= math.prod(shapes[adv])):
            aliases[cq] = adv
    parent_slot = "nest_parent_field"
    child_slot = "nest_child_field"
    rk_targets = ("rk_ru", "rk_ww", "rk_rv")
    pair_preferences = (
        ("rk_ru", "rk_ww"), ("rk_ru", "rk_rv"),
        ("rk_rv", "rk_ww"), ("rk_ww", "rk_ru"),
        ("rk_ww", "rk_rv"), ("rk_rv", "rk_ru"),
    )

    def force_slot_fits(slot: str, target: str) -> bool:
        return (slot in shapes and target in shapes
                and math.prod(shapes[slot]) <= math.prod(shapes[target]))

    force_pair = next((
        (parent_target, child_target)
        for parent_target, child_target in pair_preferences
        if force_slot_fits(parent_slot, parent_target)
        and force_slot_fits(child_slot, child_target)
    ), None)
    if force_pair is not None:
        aliases[parent_slot], aliases[child_slot] = force_pair
    else:
        # Correctness-first fallback: alias only the larger logical request
        # when one dead RK backing can hold it; the other remains explicit.
        logical_slots = sorted(
            (slot for slot in (parent_slot, child_slot) if slot in shapes),
            key=lambda slot: math.prod(shapes[slot]), reverse=True)
        for slot in logical_slots:
            target = next((candidate for candidate in rk_targets
                           if force_slot_fits(slot, candidate)), None)
            if target is not None:
                aliases[slot] = target
                break
    if ("lbc_nested_relax" in shapes and "acoustic_a" in shapes
            and math.prod(shapes["lbc_nested_relax"])
            <= math.prod(shapes["acoustic_a"])):
        aliases["lbc_nested_relax"] = "acoustic_a"
    smag_uses_xy = any(dc.run.km_opt in (2, 3, 4) for dc in domains)
    if "diff6_z" in shapes:
        candidates = (("diff6_x", "diff6_m") if smag_uses_xy
                      else ("diff6_x", "diff6_y", "diff6_m"))
        for slot in candidates:
            if (slot in shapes
                    and math.prod(shapes[slot])
                    <= math.prod(shapes["diff6_z"])):
                aliases[slot] = "diff6_z"
    for slot, target in (("smag_km", "acoustic_alpha"),
                         ("smag_kh", "acoustic_gamma")):
        if (slot in shapes and target in shapes
                and math.prod(shapes[slot]) <= math.prod(shapes[target])):
            aliases[slot] = target
    return aliases


def shared_scratch_arena_bytes(
        domains: tuple[DomainConfig, ...]) -> int:
    shapes = shared_scratch_arena_shapes(domains)
    aliases = shared_scratch_arena_aliases(domains)
    return sum(4 * math.prod(shape) for slot, shape in shapes.items()
               if slot not in aliases)


# ---------------------------------------------------------------------------
# RRTMGP workspace + per-step transient formulas
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _gas_table_meta() -> dict[str, int]:
    """ngpt/ngas per band from the shipped k-distributions (lazy import --
    the estimator stays CPU-only; table loading is host NetCDF I/O)."""
    from gpuwm.core.rrtmgp import load_gas_tables

    lw = load_gas_tables("lw")
    sw = load_gas_tables("sw")
    return {"ngpt_lw": lw.ngpt, "ngpt_sw": sw.ngpt,
            "ngas_lw": lw.ngas, "ngas_sw": sw.ngas,
            "nband_lw": lw.nband, "nband_sw": sw.nband}


@lru_cache(maxsize=1)
def k_distribution_bytes() -> int:
    """Device bytes of the lru_cache-shared k-distribution/cloud tables,
    counted ONCE per process (rrtmgp.py:324/:436 -- baseline behavior,
    never claimed as savings).  Uses the to_device dtype rule
    (rrtmgp.py:266-282): float -> f32, int -> i32, bool -> 1 byte."""
    import numpy as np
    from gpuwm.core.rrtmgp import load_cloud_tables, load_gas_tables

    total = 0
    for tables in (load_gas_tables("lw"), load_gas_tables("sw"),
                   load_cloud_tables("lw"), load_cloud_tables("sw")):
        for value in vars(tables).values():
            if isinstance(value, np.ndarray):
                itemsize = 1 if value.dtype == bool else 4
                total += value.size * itemsize
    return total


def rrtmgp_workspace_phases(nz: int, column_chunk: int, p_top: float = 5000.0
                            ) -> dict[str, dict[str, tuple[tuple[int, ...],
                                                           int]]]:
    """Per-chunk SIMULTANEOUS live sets, one dict per solver phase.

    ONE shared working set sized to the fixed column chunk, reused by all
    four domains (architecture section E RRTMGP CHUNK POLICY: legal
    because domains step strictly sequentially and workspace shape is
    f(chunk_columns, nz, p_top) with nz/p_top identical everywhere).  The
    workspace
    bound is the MAXIMUM over phases of each phase's exact live set --
    not a union with substitutions (p5t11 shadow review F2):

    * ``lw_optics`` (rrtmgp.py:1291-1306): gas tau + finalized tau alive
      together with the McICA bool mask, per-chunk VMR, band cloud
      optics, and col_dry (:1615, retained by BOTH the gas and finalized
      optics results -- one shared array, counted once).
    * ``lw_rte`` (:1307-1317): only what it READS BACK -- the finalized tau
      and the VMR Planck consumes -- plus its own Planck lay/lev/sfc sources
      (:1747-1749), emissivity/incident g-point arrays and two flux outputs.
      gas_tau, the three band cloud cubes and col_dry are dead the moment
      the finalized optics exist, so this phase lies its outputs over them
      rather than appending after them.
    * ``sw_optics`` (:1353-1368): gas tau/ssa + finalized tau/ssa/g (five
      g-point cubes) + mask + VMR + band cloud optics + col_dry.
    * ``sw_rte`` (:1369-1381): the three finalized cubes it reads, plus
      albedo/incidence g-point arrays, the materialized (chunk,nz) mu0
      broadcast (:1690-1691) and three (chunk,nz+1) flux arrays over the
      dead gas/cloud/VMR tail.  SW builds no Planck source, so unlike LW it
      does not carry vmr either.
    """
    from gpuwm.core.rrtmgp import rrtmgp_above_model_layer_counts

    meta = _gas_table_meta()
    c = int(column_chunk)
    lw_g, sw_g = meta["ngpt_lw"], meta["ngpt_sw"]
    lw_upper, sw_upper = rrtmgp_above_model_layer_counts(p_top)
    lw_nz, sw_nz = int(nz) + lw_upper, int(nz) + sw_upper
    if max(lw_nz, sw_nz) > 128:
        raise ValueError(
            "RRTMGP workspace cannot exceed the 128-layer CUDA solver limit")

    # CARRIED FIRST.  An RTE phase reads back only a few of its optics
    # phase's slots, and `phase()` assigns offsets by walking the layout in
    # order -- so a carried slot keeps its address only if everything ahead
    # of it does too.  Listing the carried ones first makes the rest ONE
    # contiguous tail that the RTE phase lays its own outputs over and then
    # stops, with no padding anywhere.  Reordering inside an optics phase is
    # free: every slot there is written before it is read in that same phase
    # (RRTMGP_WORKSPACE_LIFETIME_AUDIT).
    #
    # Dropping the dead slots WITHOUT the reorder recovers much less --
    # the holes are scattered, each has to be padded to hold the next slot
    # in place, and lev_source misses gas_tau's hole by a few percent.
    # The finalize is fused into the solvers, so the finalized optics
    # cubes DO NOT EXIST.  The carried set is what the fused solver reads:
    # the gas cube(s), the band cloud cubes it consumes, and the McICA
    # mask.  The mask is 1-byte and sits LAST among the carried slots; its
    # byte count is c*nz*ngpt with ngpt a multiple of 32, so every 4-byte
    # slot after it stays aligned, and `phase()` refuses an unaligned one
    # anyway.
    lw_carried = {
        "gas_tau": ((c, lw_nz, lw_g), 4),
        "vmr": ((c, lw_nz, meta["ngas_lw"] + 1), 4),
        "cld_tau": ((c, lw_nz, meta["nband_lw"]), 4),
        "cld_ssa": ((c, lw_nz, meta["nband_lw"]), 4),
        "mcica_mask": ((c, lw_nz, lw_g), 1),
    }
    lw_dead_in_rte = {
        "cld_asy": ((c, lw_nz, meta["nband_lw"]), 4),
        "col_dry": ((c, lw_nz), 4),
    }
    sw_carried = {
        "gas_tau": ((c, sw_nz, sw_g), 4),
        "gas_ssa": ((c, sw_nz, sw_g), 4),
        "cld_tau": ((c, sw_nz, meta["nband_sw"]), 4),
        "cld_ssa": ((c, sw_nz, meta["nband_sw"]), 4),
        "cld_asy": ((c, sw_nz, meta["nband_sw"]), 4),
        "mcica_mask": ((c, sw_nz, sw_g), 1),
    }
    sw_dead_in_rte = {
        "vmr": ((c, sw_nz, meta["ngas_sw"] + 1), 4),
        "col_dry": ((c, sw_nz), 4),
    }
    return {
        "lw_optics": {**lw_carried, **lw_dead_in_rte},
        # lay_source/lev_source/sfc_source do not exist: the LW solver
        # derives the Planck sources in registers.  That is 455 MiB of this
        # phase at the default chunk, and it is why lw_rte stopped being
        # the maximum.
        "lw_rte": {**lw_carried,
                   "emiss_gpt": ((c, lw_g), 4),
                   "incident": ((c, lw_g), 4),
                   "flux_up": ((c, lw_nz + 1), 4),
                   "flux_dn": ((c, lw_nz + 1), 4)},
        "sw_optics": {**sw_carried, **sw_dead_in_rte},
        "sw_rte": {**sw_carried,
                   "albedo_gpt": ((c, sw_g), 4),
                   "inc_gpt": ((c, sw_g), 4),
                   "mu0": ((c, sw_nz), 4),
                   "flux_up": ((c, sw_nz + 1), 4),
                   "flux_dn": ((c, sw_nz + 1), 4),
                   "flux_dir": ((c, sw_nz + 1), 4)},
    }


def rrtmgp_workspace_shapes(nz: int, column_chunk: int, p_top: float = 5000.0
                            ) -> dict[str, tuple[tuple[int, ...], int]]:
    """The shared chunk workspace: the phase-maximum simultaneous set,
    ``{"<phase>/<name>": (shape, itemsize)}`` (see
    :func:`rrtmgp_workspace_phases`)."""

    def total(items):
        return sum(math.prod(shape) * size for shape, size in items.values())

    phases = rrtmgp_workspace_phases(nz, column_chunk, p_top)
    phase = max(phases, key=lambda name: total(phases[name]))
    return {f"{phase}/{name}": spec
            for name, spec in phases[phase].items()}


def rrtmgp_column_shapes(
        cfg: RunConfig, p_top: float = 5000.0, *,
        column_chunk: int = DEFAULT_COLUMN_CHUNK,
) -> dict[str, tuple[tuple[int, ...], int]]:
    """Per-domain radiation column packing transients (rrtmgp.py
    ``RRTMGPRadiation.__call__``): full-``ncol`` model arrays plus the
    chunk-local above-model profile/path/cloud/interpolation arrays that
    coexist with the shared solver workspace.  Freed at call end; domains
    step sequentially, so the experiment estimate takes the MAX over
    domains."""
    if radiation_scheme_ids(cfg) != (4, 4):
        return {}
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
        # The legacy adapter's transients are priced as ONE shared
        # call-peak envelope in estimate_experiment (LW/SW run
        # sequentially per chunk, one domain at a time), not as
        # per-domain RRTMGP column shapes.
        return {}
    from gpuwm.core.rrtmgp import rrtmgp_above_model_layer_counts

    if (isinstance(column_chunk, bool)
            or not isinstance(column_chunk, int)
            or column_chunk < 1):
        raise ValueError("column_chunk must be a positive integer")
    nz = cfg.nz
    lw_upper, sw_upper = rrtmgp_above_model_layer_counts(p_top)
    peak_upper = max(lw_upper, sw_upper)
    peak_nz = nz + peak_upper
    ncol = cfg.ny * cfg.nx
    cap_ncol = min(ncol, column_chunk)
    shapes: dict[str, tuple[tuple[int, ...], int]] = {}
    lay = ["play", "tlay", "qv", "exner", "qc", "qr", "qi", "qs", "cldfra",
           "clwp", "ciwp", "reliq", "dgice"]
    if cfg.mp_physics == 10:
        lay += ["nc", "nr", "ni", "ns", "effc", "effr", "effi", "effs"]
    elif cfg.mp_physics in (6, 8, 16, 18, 28):
        # WRF's use_mp_re scheme table lists THOMPSONAERO explicitly and
        # separately from THOMPSON in the same disjunction
        # (phys/module_physics_init.F:1005 THOMPSON, :1006 THOMPSONAERO), and
        # the P3/Jensen-Ishmael has_reqs=0 override at :1026-1033 does not
        # exclude it, so has_reqc/has_reqi/has_reqs are all 1 for mp=28 --
        # the same authority gpuwm/core/rrtmg_legacy.py's _MP_DECLARES_RADII
        # already carries.  The mp=28 state allocates effc/effi/effs on the
        # same terms as mp=8 (see state_array_shapes above).
        #
        # STATED RATHER THAN HIDDEN: the RTE+RRTMGP adapter's own scheme map
        # (gpuwm/core/rrtmgp.py:1811) does not yet route 28 and currently
        # falls through to "kessler", which packs no radii columns at all.
        # Pricing them here first is the SAFE direction and the same
        # convention _REFLECTIVITY_MICROPHYSICS above already uses: an
        # over-priced rail refuses a run that would have fit, an under-priced
        # one lets a run breach the budget.  The legacy-RRTMG 4/4 variant is
        # unaffected either way -- it returns {} from this function and is
        # priced as one shared call-peak envelope.
        lay += ["effc", "effi", "effs"]
    for name in lay:
        shapes[f"columns/{name}"] = ((ncol, nz), 4)
    for name in ("metadata_jt", "metadata_jp", "metadata_iatm",
                  "metadata_ftemp", "metadata_fpress"):
        shapes[f"columns/{name}"] = ((cap_ncol, peak_nz), 4)
    for name in ("plev", "tlev", "lw_up", "lw_dn", "sw_up", "sw_dn"):
        shapes[f"columns/{name}"] = ((ncol, nz + 1), 4)
    # LW's 25-layer real74 cap is the peak phase.  The live model columns above
    # remain needed for SW, but only one solver chunk's expanded thermo/cloud
    # copies and metadata coexist.  With no representable upper layer the
    # adapter returns the model slices directly and creates no duplicate cap.
    if peak_upper:
        for name in ("play", "tlay", "qv", "cldfra",
                     "clwp", "ciwp", "reliq", "dgice"):
            shapes[f"columns/upper_peak_{name}"] = ((cap_ncol, peak_nz), 4)
        for name in ("plev", "tlev"):
            shapes[f"columns/upper_peak_{name}"] = (
                (cap_ncol, peak_nz + 1), 4)
    shapes["columns/emiss_bands"] = ((ncol, 16), 4)
    for name in ("tsfc", "mu0", "albedo_surface", "daylight"):
        shapes[f"columns/{name}"] = ((ncol,), 4)
    return shapes


def dudhia_column_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Conservative device-transient envelope for Dudhia shortwave.

    The adapter packs ten top-down layer fields.  SWPARA then carries five
    full-column work arrays plus a heating/output reversal; the driver holds
    its returned/coupled copies while validating.  Twenty-four layer-sized
    arrays and forty column-sized vectors bound those simultaneously live
    values and CuPy expression temporaries without pretending they share the
    RRTMGP chunk workspace.
    """
    if radiation_scheme_ids(cfg)[1] != 1:
        return {}
    ncol = cfg.ny * cfg.nx
    return {
        "dudhia/layer_envelope": (24, ncol, cfg.nz),
        "dudhia/column_envelope": (40, ncol),
        "dudhia/lookup_tables": (2, 4, 5),
    }


def atmosphere_transient_shapes(cfg: RunConfig
                                ) -> dict[str, tuple[int, ...]]:
    """``_prepare_atmosphere`` per-call transients (physics.py:338-400):
    fresh device arrays alive for the whole physics call, including
    through radiation."""
    from gpuwm.core.physics_inventory import physics_enabled

    if not physics_enabled(cfg):
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    m = (nz, ny, nx)
    fl = (nz + 1, ny, nx)
    return {"atmosphere/theta": m, "atmosphere/temperature": m,
            "atmosphere/pressure": m, "atmosphere/exner": m,
            "atmosphere/u": m, "atmosphere/v": m, "atmosphere/dz": m,
            "atmosphere/p_interface": fl, "atmosphere/z_interface": fl}


#: PBL schemes that allocate the raw YSU-shaped output bundle.  MYNN
#: fills the same dict of names through its own launcher; SASE does not
#: -- it hands its rates straight to the coupling helper.
_YSU_OUTPUT_BUNDLE_SCHEMES = (1, 5)


def ysu_output_transient_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Raw per-call YSU outputs before field-copy/coupling consumption.

    These allocations exist on every YSU call.  Positive ``bldt`` retains
    the result after the call (and is also counted as persistent); bldt=0
    releases it at the last consumer, so it appears only in this category.
    """
    # Keyed on the SCHEME, never on the truthiness of the selector: SASE
    # holds the same driver slot but allocates none of YSU's output
    # bundle -- it returns its rates through the coupling helper
    # directly -- so a truthiness test priced a SASE run for fifteen
    # arrays it never asks for.  This mirrors the driver's own rule that
    # a selector VALUE, not its truthiness, decides what runs.
    if int(cfg.bl_pbl_physics) not in _YSU_OUTPUT_BUNDLE_SCHEMES:
        return {}
    m = (cfg.nz, cfg.ny, cfg.nx)
    s2 = (cfg.ny, cfg.nx)
    shapes = {f"ysu_output/{name}": m for name in _YSU_3D}
    shapes.update({f"ysu_output/{name}": s2 for name in _YSU_2D})
    return shapes


#: PBL schemes that allocate the Shin-Hong per-call output bundle
#: (gpuwm/core/shinhong.py launch_shinhong: one empty_like comprehension
#: for the nine 3-D fields, four cp.empty for the 2-D ones -- the five
#: allocation sites the physics allocation inventory prices).  Its own
#: constant rather than a second member of _YSU_OUTPUT_BUNDLE_SCHEMES
#: because the bundles differ: Shin-Hong returns 9 3-D + 4 2-D fields
#: against YSU's set, and keying both off one tuple would price the
#: wrong roster for whichever scheme joined second.
_SHINHONG_OUTPUT_BUNDLE_SCHEMES = (11,)

#: The launcher's exact per-call output roster (single-sourced by test
#: against gpuwm/core/shinhong.py, which is CuPy-importing and therefore
#: not imported here -- the ysu constant-pair idiom above).
_SHINHONG_3D = ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                "exch_h", "tke", "el")
_SHINHONG_2D = ("hpbl", "kpbl", "wstar", "delta")


def shinhong_output_transient_shapes(
        cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Raw per-call Shin-Hong outputs before field-copy/coupling use.

    The :func:`ysu_output_transient_shapes` contract for scheme 11:
    these allocations exist on every _run_shinhong call and are released
    at the last consumer (the scheme has no positive-cadence retention
    -- ``last_ysu`` stays None -- so no persistent counterpart exists).
    ``kpbl`` is int32; every other field is float32, so one 4-byte
    itemsize covers the whole roster.
    """
    if int(cfg.bl_pbl_physics) not in _SHINHONG_OUTPUT_BUNDLE_SCHEMES:
        return {}
    m = (cfg.nz, cfg.ny, cfg.nx)
    s2 = (cfg.ny, cfg.nx)
    shapes = {f"shinhong_output/{name}": m for name in _SHINHONG_3D}
    shapes.update({f"shinhong_output/{name}": s2 for name in _SHINHONG_2D})
    return shapes


#: PBL schemes that allocate the MYJ per-call output bundle
#: (gpuwm/core/myjpbl.py myj_pbl_step: one cp.empty comprehension for the
#: eight 3-D fields plus three cp.empty for the 2-D ones -- the four
#: allocation sites the physics allocation inventory prices).  Its own
#: constant for the reason the Shin-Hong pair above gives: the rosters
#: differ, and keying two schemes off one tuple prices the wrong one.
_MYJ_OUTPUT_BUNDLE_SCHEMES = (2,)

#: The launcher's exact per-call output roster, single-sourced against
#: gpuwm/core/myjpbl.py -- which is CuPy-importing and therefore not
#: imported here -- by tests/test_myj_port.py::
#: test_preflights_myj_output_roster_matches_the_launchers, the
#: ysu/shinhong constant-pair idiom above.  ``tke``
#: is absent on purpose: MYJ's TKE column is CARRIED state that
#: initialize_physics allocates once, not a per-call transient, and it is
#: already priced as a driver field.
_MYJ_3D = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
           "rqiblten", "el_myj", "exch_h")
_MYJ_2D = ("pblh", "kpbl", "mixht")


def myj_output_transient_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Raw per-call MYJ PBL outputs before coupling consumes them.

    The :func:`ysu_output_transient_shapes` contract for scheme 2: these
    allocations exist on every ``_run_myj_pbl`` call and are released at
    the last consumer.  ``kpbl`` is int32; every other field is float32,
    so one 4-byte itemsize covers the whole roster.

    The Eta surface layer (sf_sfclay_physics=2) has no counterpart here
    and needs none: gpuwm/core/myjsfc.py allocates NOTHING per call -- it
    writes into the driver's own persistent surface fields -- which the
    physics allocation inventory records as an empty row.
    """
    if int(cfg.bl_pbl_physics) not in _MYJ_OUTPUT_BUNDLE_SCHEMES:
        return {}
    m = (cfg.nz, cfg.ny, cfg.nx)
    s2 = (cfg.ny, cfg.nx)
    shapes = {f"myj_output/{name}": m for name in _MYJ_3D}
    shapes.update({f"myj_output/{name}": s2 for name in _MYJ_2D})
    return shapes


def noahmp_lsm_transient_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Noah-MP land-surface per-call device transients, both paths.

    Shapes here are ``(columns, bytes-per-column)`` with ``itemsize=1``,
    because the underlying allocations are dozens of named arrays whose
    itemization lives with their owners (the allocation-inventory rows in
    ``tests/test_physics_allocation_inventory.py``); what preflight owes is
    the bound, not the roster.

    * ``slab_chunk_transients`` is the forecast path's per-chunk cost:
      :func:`gpuwm.core.noahmp_column_slab.evaluate_sflx_slab` over
      ``SLAB_COLUMN_CHUNK`` land columns.  The ceiling and the bound are the
      runtime's own constants, so price and bound cannot drift apart.
      Measured on the RTX 5090: 2,723 B of peak allocator demand per column
      for one 65,536-column chunk call, priced at 4,096.  Demand rather than
      CuPy pool growth, because growth reads only what the pool had to
      acquire from the driver and a warm pool serves the transient from
      blocks it already holds; see ``SLAB_TRANSIENT_BYTES_PER_COLUMN``.
    * ``slab_grid_transients`` is the same call's whole-grid residue: the
      device prologue intermediates (Q_ML through FICEOLD), the land index
      arrays and the pool's cross-chunk fragmentation, which scale with
      ``nx*ny`` and not with the chunk.  Measured at 360,000 columns: the
      whole-call pool growth of 311.8 MiB minus the chunk term's measured
      172.0 MiB is 407 B per grid column, priced at 512.
    * ``staged_leaf_batches`` is the per-column seam kept as the paired
      second implementation: four leaf batches at 620 B per staged column
      (52 + 76 + 296 + 196, derived in the allocation inventory), bounded by
      ``COLUMN_BATCH``.

    Only one path runs per call; all are priced because together they are
    still decided by bounds rather than by the nest, and the staged term is
    three orders of magnitude below the slab ones.  At d04 the two slab
    terms price 431.8 MiB against the measured 311.8.
    """
    if getattr(cfg, "sf_surface_physics", 0) != 4:
        return {}
    from gpuwm.core.noahmp_runtime import (
        COLUMN_BATCH, SLAB_COLUMN_CHUNK, SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN,
        SLAB_TRANSIENT_BYTES_PER_COLUMN)

    columns = int(cfg.ny) * int(cfg.nx)
    return {
        "noahmp_lsm/slab_chunk_transients":
            (min(SLAB_COLUMN_CHUNK, columns),
             SLAB_TRANSIENT_BYTES_PER_COLUMN),
        "noahmp_lsm/slab_grid_transients":
            (columns, SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN),
        "noahmp_lsm/staged_leaf_batches":
            (min(COLUMN_BATCH, columns), _NOAHMP_STAGED_BYTES_PER_COLUMN),
    }


#: 620 B of leaf-batch rows per staged column: bare_flux 52 + radiation 76 +
#: water 296 + sflx_pre 196, the derivation the allocation inventory records.
_NOAHMP_STAGED_BYTES_PER_COLUMN = 620


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainMemoryEstimate:
    """Itemized memory for one domain (architecture section E contract)."""

    grid_id: int
    items: tuple[MemoryItem, ...]

    def category_bytes(self, category: str) -> int:
        return sum(item.nbytes for item in self.items
                   if item.category == category)

    @property
    def resident_bytes(self) -> int:
        """Persistent tier-1 residency (state/physics/scratch/lbc/nest).

        The SASE closure's own working set is a per-step transient like
        the radiation columns, so it is excluded here and counted below.
        """
        return sum(item.nbytes for item in self.items
                   if item.category not in ("transient", "sase"))

    @property
    def transient_bytes(self) -> int:
        """Per-step transients (radiation columns + physics prep/YSU outputs)
        coexisting with the chunk workspace; freed between steps."""
        return sum(item.nbytes for item in self.items
                   if item.category in ("transient", "sase"))

    @property
    def arena_scratch_bytes(self) -> int:
        """This domain's requests admitted by the lifetime audit.

        The value remains part of :attr:`resident_bytes` so a bare/single-
        domain estimate keeps the historical per-state accounting. The
        experiment estimate replaces the multi-domain sum with one per-slot
        maximum.
        """
        return sum(item.nbytes for item in self.items
                   if item.category in ("scratch", "lbc", "nest")
                   and scratch_slot_uses_arena(item.name))

    @property
    def rebuilt_state_bytes(self) -> int:
        """This domain's restart-REBUILT state-array requests."""
        rebuilt = shared_dycore_state_symbols()
        return sum(item.nbytes for item in self.items
                   if item.category == "state" and item.name in rebuilt)


def estimate_domain(dc: DomainConfig, *, spec_bdy_width: int | None = None,
                    parent: DomainConfig | None = None,
                    n_lbc_intervals: int = 0,
                    p_top: float = 5000.0,
                    column_chunk: int = DEFAULT_COLUMN_CHUNK,
                    ) -> DomainMemoryEstimate:
    """Itemized :class:`DomainMemoryEstimate` for one domain.

    ``spec_bdy_width`` (the experiment's) sizes child ``nest_*`` tables;
    ``n_lbc_intervals`` sizes the root's eager forcing tables.  Both
    default from the domain's own RunConfig / to zero intervals for a
    bare single-domain estimate.
    """
    run = dc.run
    width = run.spec_bdy_width if spec_bdy_width is None else spec_bdy_width
    items: list[MemoryItem] = []
    items += _items("state", state_array_shapes(run))
    items += _items("physics", physics_array_shapes(run))
    registry = scratch_slot_registry(
        run, n_lbc_intervals=(n_lbc_intervals if run.specified else 0))
    # The (d) LBC residents live in the scratch pool but report under
    # their own itemization category (architecture section E contract).
    items += _items("lbc", {slot: shape for slot, shape in registry.items()
                            if slot.startswith("lbc_")})
    items += _items("scratch",
                    {slot: shape for slot, shape in registry.items()
                     if not slot.startswith("lbc_")})
    if dc.parent_id != 0:
        shapes = nest_slot_shapes(dc, width, parent)
        items += _nest_items(shapes, nest_slot_dtypes(dc, width, parent))
    items += _items("transient", atmosphere_transient_shapes(run))
    items += _items("transient", ysu_output_transient_shapes(run))
    items += _items("transient", shinhong_output_transient_shapes(run))
    items += _items("transient", myj_output_transient_shapes(run))
    items += _items("transient", noahmp_lsm_transient_shapes(run),
                    itemsize=1)
    items += tuple(MemoryItem(name, "transient", shape, size)
                   for name, (shape, size)
                   in rrtmgp_column_shapes(
                       run, p_top, column_chunk=column_chunk).items())
    items += _items("transient", dudhia_column_shapes(run))
    # The SASE closure's step working set gets its OWN category rather
    # than joining "transient": on a wide domain its dynamic-solve peak
    # is the single largest transient in the run, and a user reading a
    # preflight needs to see that it is the closure asking, not the
    # radiation columns.  It is still counted as a step transient, never
    # as residency.
    items += tuple(MemoryItem(name, "sase", shape, size,
                              "float64" if size == 8 else "float32")
                   for name, (shape, size)
                   in sase_workspace_shapes(run).items())
    return DomainMemoryEstimate(dc.grid_id, tuple(items))


@dataclass(frozen=True)
class ExperimentMemoryEstimate:
    """The experiment-level three-tier estimate.

    ``alloc_estimate_bytes`` is THE enforced number of the N0/N5/N6
    chain: ``ALLOCATOR_HEADROOM x (resident + shared chunk workspace +
    max-over-domains step transients)``. Multi-domain resident scratch uses
    the lifetime-audited per-slot maximum; a single domain keeps the original
    per-state sum. ``held/footprint`` projections add the calibrated
    tier-2/tier-3 terms for reserve-policy visibility. ``workspace_bytes`` is
    not a pool-reuse allowance: Task 14 constructs one byte allocation of
    exactly this size and every domain adapter consumes its phase views; the
    builder refuses any runtime/ledger byte drift.
    """

    domains: tuple[DomainMemoryEstimate, ...]
    k_tables_bytes: int
    workspace_bytes: int
    scratch_arena_bytes: int
    uses_shared_scratch_arena: bool
    dycore_state_workspace_bytes: int
    uses_shared_dycore_state_workspace: bool
    column_chunk: int
    headroom: float = ALLOCATOR_HEADROOM
    retention_residual_bytes: int = field(default=0)
    device_overhead_bytes: int = field(
        default_factory=lambda: platform_projection_constants()[1])
    #: CUDA context + launch-time local-memory backing store for the
    #: kernel set THIS configuration launches, on the device profile it
    #: was priced against.  The affine envelope's intercept.
    non_pool_device_bytes: int = 0
    #: Which envelope family priced it (``linux``/``windows``/...).
    envelope_family: str = "linux"
    #: Does this configuration run the LEGACY RRTMG engines?  That is
    #: what decides the pool-slack term (:data:`POOL_SLACK_FRACTION`) --
    #: the retained LW/SW call-peak workspace three campaigns measured on
    #: the legacy lane and on no other.  Defaults to charging it: a
    #: caller that has not said gets the conservative answer.
    uses_legacy_radiation: bool = True

    @property
    def resident_bytes(self) -> int:
        per_domain = sum(d.resident_bytes for d in self.domains)
        if self.uses_shared_scratch_arena:
            per_domain -= self.scratch_arena_request_bytes
            per_domain += self.scratch_arena_bytes
        if self.uses_shared_dycore_state_workspace:
            per_domain -= self.dycore_state_request_bytes
            per_domain += self.dycore_state_workspace_bytes
        return per_domain + self.k_tables_bytes

    @property
    def dycore_state_request_bytes(self) -> int:
        """Unshared sum of restart-REBUILT requests across all domains."""
        return sum(d.rebuilt_state_bytes for d in self.domains)

    @property
    def dycore_state_saved_bytes(self) -> int:
        """Resident bytes removed by per-symbol maximum sharing."""
        if not self.uses_shared_dycore_state_workspace:
            return 0
        return (self.dycore_state_request_bytes
                - self.dycore_state_workspace_bytes)

    @property
    def scratch_arena_request_bytes(self) -> int:
        """Unshared sum of arena-admitted requests across all domains."""
        return sum(d.arena_scratch_bytes for d in self.domains)

    @property
    def scratch_arena_saved_bytes(self) -> int:
        """Resident bytes removed by max-per-slot sharing."""
        if not self.uses_shared_scratch_arena:
            return 0
        return self.scratch_arena_request_bytes - self.scratch_arena_bytes

    @property
    def transient_peak_bytes(self) -> int:
        """Domains step strictly sequentially (architecture section E
        chunk-policy adjudication); one domain's step transients live at
        a time, so the peak takes the max, not the sum."""
        return max((d.transient_bytes for d in self.domains), default=0)

    @property
    def subtotal_bytes(self) -> int:
        return (self.resident_bytes + self.workspace_bytes
                + self.transient_peak_bytes)

    @property
    def alloc_estimate_bytes(self) -> int:
        return math.ceil(self.headroom * self.subtotal_bytes)

    @property
    def held_projection_bytes(self) -> int:
        return self.alloc_estimate_bytes + self.retention_residual_bytes

    @property
    def footprint_projection_bytes(self) -> int:
        return self.held_projection_bytes + self.device_overhead_bytes

    @property
    def envelope_intercept_bytes(self) -> int:
        """Everything in the envelope that does not scale with the grid.

        The itemized non-pool residency, and nothing else.  The 5090
        zero-step probe constant (``device_overhead_bytes``) and the
        pool-retention constant stay in the TIER 2/3 projection display
        they were calibrated for; charging them here on top of the
        itemized non-pool term is how a 10 GiB card came to carry
        4.12 GiB of another machine's accounting (the 3080 walk).
        """
        return self.non_pool_device_bytes

    @property
    def peak_envelope_bytes(self) -> int:
        """The affine machine-peak envelope for this forecast."""
        return machine_peak_envelope_bytes(
            alloc_estimate_bytes=self.alloc_estimate_bytes,
            non_pool_bytes=self.envelope_intercept_bytes,
            domains=len(self.domains),
            family=self.envelope_family,
            legacy_radiation=self.uses_legacy_radiation)

    @property
    def envelope_basis(self) -> str:
        """The evidence behind this configuration's envelope terms."""
        if not self.uses_legacy_radiation:
            return ENVELOPE_AFFINE_BASIS
        # The slack term is the legacy lane's, and its evidence is the
        # union of the campaigns that measured that lane on each driver
        # model -- naming only the local platform's would credit half of
        # what the number rests on.
        if self.envelope_family == "windows":
            return f"{ENVELOPE_WDDM_BASIS}; {ENVELOPE_LINUX_POOL_BASIS}"
        return f"{ENVELOPE_AFFINE_BASIS}; {ENVELOPE_LINUX_POOL_BASIS}"

    def peak_envelope_terms(self) -> str:
        """The envelope's arithmetic, exactly as it was evaluated.

        Printed by both `gpuwm check` and the wizard, from one place, so
        the two can never show a sum whose parts do not add up to it --
        which is what happens the moment a second branch exists and only
        one of them is described.
        """
        nests = max(0, len(self.domains) - 1)
        nest_term = (f" + {ENVELOPE_PER_NEST_FRACTION:.0%} of the estimate "
                     f"x {nests} nest(s)" if nests else "")
        slack_term = (
            f" + {POOL_SLACK_FRACTION:.0%} of the estimate legacy-RRTMG "
            f"pool slack" if self.uses_legacy_radiation else "")
        return (f"estimate {self.alloc_estimate_bytes / GIB:.2f} + "
                f"non-pool {self.envelope_intercept_bytes / GIB:.2f} (CUDA "
                f"context + local-memory backing store + GF, KF and "
                f"YSU column workspaces) + "
                f"{ENVELOPE_UNMODELLED_BYTES / GIB:.2f} unmodelled"
                f"{nest_term}{slack_term} = "
                f"{self.peak_envelope_bytes / GIB:.2f} GiB")


# ---------------------------------------------------------------------------
# Preprocessing (ingest) phase
#
# Everything above prices the FORECAST.  For a long time that was the only
# phase anybody priced, and on a measured 414x330x49 CONUS 12 km case the
# forecast peaked at 7.10 GiB while preprocessing the very same case peaked
# at 14.94 GiB -- so `gpuwm check` reported an envelope for a phase that
# was not the binding one, and a domain sized to a 12 GB card downloaded
# 81 GFS files and then died in preprocessing at 15.82 GB.  Ingest is now
# streamed (gpuwm/ingest/lateral_bc.py:StateBoundaryFrames), and the model
# below is what it actually holds.
# ---------------------------------------------------------------------------

#: Forcing times a streaming ingest holds on the device at once: the ONE
#: currently being built.  Every other time contributes its perimeter
#: frames -- host memory, O(perimeter) -- and is released before the next
#: one is interpolated.
#:
#: THIS WAS 2, AND THE SECOND ONE WAS THE START TIME.  Nothing reads the
#: start time until the boundaries are complete (the prepared cache, the
#: wrfinput export and the surface analysis are all written from it after
#: the loop), but a prepare loop that walked the times in order built it
#: FIRST and therefore held it while every later time was built
#: underneath.  The adapters now build it LAST
#: (gpuwm/ingest/lateral_bc.py:start_last_forcing_order) and retain
#: nothing else, which is a pure reordering: the perimeter frames are
#: accumulated against their positions and the intervals come out
#: byte-identical.  At 800x800x49, mp=10, three GFS times, that is 14.67
#: GiB of device residency dropping to 7.66 and a peak envelope of 23.92
#: GiB dropping to 15.86 -- the whole reason such a domain can be
#: prepared on a 16 GiB card at all.
#:
#: SCOPE, because this number is a gate input and an optimistic gate is
#: the failure mode this section exists to prevent: it describes the
#: prepared-cache adapters `gpuwm/gfs_direct.py`, `gpuwm/era5_direct.py`
#: and `gpuwm/mapped_direct.py`, which are what `gpuwm go` and the domain
#: wizard price.  `gpuwm/runtime.py:prepare_real_case` -- the verify-case
#: preparer, off those routes -- has not been reordered and still holds
#: two, so for a run prepared through THAT path this estimate is
#: optimistic by exactly one `per_time_bytes`.
INGEST_RESIDENT_FORCING_TIMES = 1

#: Pressure levels each forcing product decodes onto the target grid.
#: GFS: the certified 21-level ladder the Rust bridge gates on
#: (gpuwm/gfs_direct.py:_validate_ladder; a case whose p_top sits above
#: 100 hPa is fetched with extra levels, so this is a floor for those).
#: ERA5: the 37 standard pressure levels of the reanalysis product.
#:
#: This map IS the priced-source inventory: a product absent from it is
#: reported NOT PRICED rather than given a plausible number, and every
#: other table in this section is keyed by exactly these names.  The
#: native-hybrid-level ingest lane is deliberately not among them --
#: nothing here has measured it.  A wrong ingest estimate is the defect;
#: an absent one that says so is not.
SOURCE_ANALYSIS_LEVELS = {"era5": 37, "gfs": 21}

#: Cadence each PRICED product is fetched at.  The ingest phase's time
#: COUNT comes from this, not from the forecast's LBC interval: a GFS
#: chain fetched 3-hourly has nine forcing times over 24 h where the
#: 6-hourly default would claim five.  Keys track SOURCE_ANALYSIS_LEVELS
#: exactly; the wizard keeps its own, wider table for the forecast side,
#: which prices products this one does not.
INGEST_FORCING_CADENCE_SECONDS = {"era5": 21600.0, "gfs": 10800.0}

#: Fields each product carries on those levels: T, RH, GHT on mass points
#: plus U and V on their own staggers.
SOURCE_ANALYSIS_MASS_LEVEL_FIELDS = 3
SOURCE_ANALYSIS_WIND_LEVEL_FIELDS = 1  # each of U and V, on its own stagger

#: Single-level fields interpolated alongside them (surface state, skin,
#: snow, ice, land mask and the four soil moisture/temperature layers).
#: Counted from the decoder inventory of a real GFS ingest.
SOURCE_ANALYSIS_SURFACE_FIELDS = {"era5": 19, "gfs": 19}

#: What the itemization below does NOT enumerate: the vertical-interpolation
#: geometry and the elementwise temporaries WRF-real's setup builds and
#: drops inside one call.  One time is built at a time, so this is charged
#: ONCE, as a fraction of one forcing time's residency.
#:
#: MEASURED, both ends of the same CONUS 12 km case (414x330x49, 9 GFS
#: times, RTX 5090, process-attributed peak, 432 MiB CUDA context
#: subtracted): the all-times-resident form itemizes 13.71 GiB and peaked
#: at 14.93 GiB (0.80 GiB unaccounted); the two-resident streaming form
#: itemizes 3.23 GiB and peaked at 4.56 GiB (0.91 GiB unaccounted).
#: Against a 1.50 GiB forcing time that is 0.53x and 0.61x -- additive
#: and stable, which is what a per-call transient should be.  0.65 is the
#: margin over both.
#:
#: The start-last reordering that took residency from two forcing times
#: to one does NOT move this fraction, and that is the point of stating
#: it as a fraction of ONE time: it prices the temporaries a single
#: interpolate/initialize call builds and drops inside itself, which is
#: the same call in either order.  What the reordering removes is a
#: RESIDENT term, not a transient one, so the one-resident itemization is
#: the two-resident measurement minus exactly one `per_time_bytes` --
#: 3.23 GiB down to 1.73 on that case, against a peak that should follow
#: it from 4.56 GiB to about 3.06.  That predicted peak is a prediction:
#: it has not been measured on a device, and the estimate bounds it by
#: 1.15x.
INGEST_TRANSIENT_PER_TIME_FRACTION = 0.65

#: What that measurement was, printed beside the number it produces.
INGEST_PEAK_ENVELOPE_BASIS = (
    "measured, CONUS 12 km 414x330x49 x 9 GFS times, RTX 5090 / Linux: "
    "itemization + 0.65x one forcing time of transients, x1.15 headroom, "
    "+ CUDA context")


def ingest_analysis_shapes(cfg: RunConfig, *, source: str
                           ) -> dict[str, tuple[int, ...]]:
    """One forcing time, horizontally interpolated onto the target grid.

    The source-level fields land on the model's OWN horizontal grid --
    that is what horizontal interpolation is -- so they are sized by the
    target ny/nx and the SOURCE's level count, not the model's nz.
    """
    key = str(source).strip().lower()
    try:
        levels = SOURCE_ANALYSIS_LEVELS[key]
        surface = SOURCE_ANALYSIS_SURFACE_FIELDS[key]
    except KeyError:
        raise ValueError(
            f"no forcing-analysis level inventory for source {source!r}; "
            f"known: {sorted(SOURCE_ANALYSIS_LEVELS)}") from None
    ny, nx = int(cfg.ny), int(cfg.nx)
    shapes: dict[str, tuple[int, ...]] = {}
    for index in range(SOURCE_ANALYSIS_MASS_LEVEL_FIELDS):
        shapes[f"analysis_mass_level_{index}"] = (levels, ny, nx)
    for index in range(SOURCE_ANALYSIS_WIND_LEVEL_FIELDS):
        shapes[f"analysis_u_level_{index}"] = (levels, ny, nx + 1)
        shapes[f"analysis_v_level_{index}"] = (levels, ny + 1, nx)
    for index in range(surface):
        shapes[f"analysis_surface_{index}"] = (ny, nx)
    return shapes


@dataclass(frozen=True)
class IngestMemoryEstimate:
    """Itemized device memory for the preprocessing phase of one TREE.

    Preprocessing builds one complete :class:`DomainState` per forcing
    time from one horizontally interpolated analysis per forcing time.
    Streaming means ``resident_times`` of each coexist rather than all of
    them, which is the whole difference between this phase costing twice
    the forecast and costing less than half of it.  ``resident_times`` is
    now 1 rather than 2: see
    :data:`INGEST_RESIDENT_FORCING_TIMES` for which routes that describes
    and which one it does not.

    2026-08-01 AMENDMENT -- IT IS NOT ONE ROOT.  v1.4.0 priced this phase
    on the root domain alone, so the PREDICTION FELL as nests were added
    (5.46 -> 1.89 -> 1.30 GiB across one, two and four domains) while the
    16 GiB fleet node measured it FLAT at 4.0-4.8 GiB -- under by 2.1x at
    two domains and 3.4x at four, in the unsafe direction, on the very
    number the before-the-fetch gate half-relies on.  The mechanism was
    plain in the tool's own itemization: a deeper ladder has a SMALLER
    root, and only the root was priced, so adding domains made the answer
    shrink.  In the two-domain case the unpriced d02 had 4.9x the cells
    of the priced root.

    The hierarchy is initialized and exported as ONE transaction, so
    every domain's initial state is on the device together at the moment
    it is written: :attr:`nest_state_bytes` prices the nests, and the
    per-call setup transient is charged against the WIDEST domain in the
    tree rather than the root.
    """

    grid_id: int
    items: tuple[MemoryItem, ...]
    resident_times: int
    n_forcing_times: int
    boundary_frame_bytes: int
    #: Initial-state residency of every domain BELOW the root, summed:
    #: the hierarchy export holds the whole tree at once.  Zero for a
    #: single-domain configuration, which is why this amendment cannot
    #: move any single-domain number.
    nest_state_bytes: int = 0
    #: Analysis + state of the widest domain in the tree, whatever its
    #: depth; the setup transient is charged against this.
    widest_domain_time_bytes: int = 0
    #: ``(grid_id, state_bytes)`` per nest, for the itemized report.
    nest_state_items: tuple[tuple[int, int], ...] = ()
    headroom: float = ALLOCATOR_HEADROOM
    context_bytes: int = CUDA_CONTEXT_BYTES
    device_overhead_bytes: int = field(
        default_factory=lambda: platform_projection_constants()[1])

    def category_bytes(self, category: str) -> int:
        return sum(item.nbytes for item in self.items
                   if item.category == category)

    @property
    def per_time_bytes(self) -> int:
        """One ROOT forcing time: its analysis plus the state built from it."""
        return sum(item.nbytes for item in self.items
                   if item.category in ("analysis", "state"))

    @property
    def transient_basis_bytes(self) -> int:
        """The domain the per-call setup transient is charged against."""
        return max(self.widest_domain_time_bytes, self.per_time_bytes)

    @property
    def forcing_table_bytes(self) -> int:
        """The completed boundary tables, uploaded onto the initial state."""
        return self.category_bytes("lbc")

    @property
    def resident_bytes(self) -> int:
        return (self.resident_times * self.per_time_bytes
                + self.forcing_table_bytes + self.nest_state_bytes)

    @property
    def unstreamed_resident_bytes(self) -> int:
        """What this phase cost before it streamed -- every time at once."""
        return (self.n_forcing_times * self.per_time_bytes
                + self.forcing_table_bytes + self.nest_state_bytes)

    @property
    def transient_bytes(self) -> int:
        """Un-enumerated setup temporaries for the ONE domain being built."""
        return math.ceil(
            INGEST_TRANSIENT_PER_TIME_FRACTION * self.transient_basis_bytes)

    @property
    def subtotal_bytes(self) -> int:
        return self.resident_bytes + self.transient_bytes

    @property
    def alloc_estimate_bytes(self) -> int:
        """Subtotal under the same allocator headroom the forecast uses."""
        return math.ceil(self.headroom * self.subtotal_bytes)

    @property
    def peak_envelope_bytes(self) -> int:
        """What a machine watching this phase should expect to see.

        Pool allocations plus the non-pool CUDA context.  No second
        envelope multiplier: unlike the forecast, this phase has no
        per-step churn for a retention factor to model -- it builds one
        forcing time at a time, keeps ``resident_times`` of them, and
        drops the rest.
        """
        return (self.alloc_estimate_bytes + self.context_bytes
                + self.device_overhead_bytes)


def estimate_ingest(exp: ExperimentConfig, *, source: str,
                    forcing_interval_seconds: float
                    = DEFAULT_FORCING_INTERVAL_SECONDS,
                    vram_gib: float | None = None,
                    profile: DeviceLocalMemoryProfile | None = None,
                    ) -> IngestMemoryEstimate:
    """Itemize the preprocessing phase of ``exp``'s WHOLE DOMAIN TREE.

    The root carries the streamed forcing times and the boundary tables.
    Every nest carries one complete initial state, and they are resident
    together: the hierarchy is verified and exported as a single atomic
    transaction, not domain by domain with a release in between.
    """
    dc = exp.root
    run = dc.run
    n_intervals = lbc_intervals(exp.run_seconds, forcing_interval_seconds)
    items: list[MemoryItem] = []
    items += _items("analysis", ingest_analysis_shapes(run, source=source))
    items += _items("state", state_array_shapes(run))
    registry = scratch_slot_registry(
        run, n_lbc_intervals=(n_intervals if run.specified else 0))
    items += _items("lbc", {slot: shape for slot, shape in registry.items()
                            if slot.startswith("lbc_")})
    # Every NEST: one complete initial state each, all resident for the
    # export transaction.  A nest is initialized from its parent, so it
    # carries no second copy of the source analysis -- but the setup
    # transient is charged against whichever domain is widest, which on
    # a real ladder is usually a nest and not the root.
    nest_items: list[tuple[int, int]] = []
    widest = sum(item.nbytes for item in items
                 if item.category in ("analysis", "state"))
    for child in exp.domains:
        if child.grid_id == dc.grid_id:
            continue
        child_run = child.run
        state = sum(4 * math.prod(shape)
                    for shape in state_array_shapes(child_run).values())
        analysis = sum(
            4 * math.prod(shape) for shape in
            ingest_analysis_shapes(child_run, source=source).values())
        nest_items.append((child.grid_id, state))
        widest = max(widest, state + analysis)
    # The host-side perimeter frames StateBoundaryFrames retains: float64,
    # four sides, every forcing time.  Reported so the phase's HOST cost
    # is visible too; it is not part of the device residency above.
    width = int(run.spec_bdy_width)
    frame_elements = sum(
        2 * width * (dims[1] + dims[2]) * dims[0]
        for dims in _lbc_field_dims(run).values())
    return IngestMemoryEstimate(
        grid_id=dc.grid_id, items=tuple(items),
        resident_times=INGEST_RESIDENT_FORCING_TIMES,
        n_forcing_times=n_intervals + 1,
        boundary_frame_bytes=8 * frame_elements * (n_intervals + 1),
        nest_state_bytes=sum(nbytes for _, nbytes in nest_items),
        widest_domain_time_bytes=widest,
        nest_state_items=tuple(nest_items),
        # This card's context, not the retired flat constant: ingest
        # stands up the same CUDA context the forecast does.
        context_bytes=(MEASURED_LOCAL_MEMORY_PROFILE if profile is None
                       else profile).cuda_context_bytes,
        device_overhead_bytes=platform_projection_constants(
            vram_gib=vram_gib)[1],
    )


@dataclass(frozen=True)
class PhaseMemoryEstimate:
    """Both phases of one run, and which of them binds the card.

    A configuration fits when the LARGEST phase fits.  Pricing only the
    forecast is what let a wizard size a domain to a card, watch the user
    download 81 GFS files, and then fail in preprocessing.
    """

    forecast: ExperimentMemoryEstimate
    ingest: IngestMemoryEstimate | None
    forecast_envelope_bytes: int
    ingest_envelope_bytes: int | None
    source: str | None = None
    #: The STREAMED forecast envelope, when ``[tiles]`` resolves to streaming
    #: this domain.  Present means ``forecast_envelope_bytes`` above is the
    #: number that binds -- it is already the streamed one, not the resident
    #: one -- and this carries the tiling that produced it so the verdict can
    #: name it.  ``None`` is every resident run, where nothing changed.
    streamed: "StreamedEnvelope | None" = None
    #: The resident forecast envelope, kept beside the streamed one so a
    #: report can say what streaming BOUGHT.  Equal to
    #: ``forecast_envelope_bytes`` on a resident run.
    resident_forecast_envelope_bytes: int | None = None

    @property
    def ingest_priced(self) -> bool:
        return self.ingest_envelope_bytes is not None

    @property
    def streamed_forecast(self) -> bool:
        return self.streamed is not None

    @property
    def binding_phase(self) -> str:
        if not self.ingest_priced:
            return "forecast"
        return ("ingest"
                if self.ingest_envelope_bytes > self.forecast_envelope_bytes
                else "forecast")

    @property
    def peak_envelope_bytes(self) -> int:
        if not self.ingest_priced:
            return self.forecast_envelope_bytes
        return max(self.forecast_envelope_bytes, self.ingest_envelope_bytes)

    def fits(self, budget_bytes: int) -> bool:
        return self.peak_envelope_bytes <= int(budget_bytes)

    def verdict(self, budget_bytes: int | None) -> str:
        """One sentence naming the binding phase and its number.

        Under ``[tiles]`` the forecast term is the STREAMED envelope, so the
        sentence has to say so: a reader who sees "the forecast needs 6.61
        GiB" for a 550x550 domain that manifestly cannot fit in 6.61 GiB
        resident is owed the reason in the same breath, and the tiling is
        the reason.
        """
        if self.streamed_forecast:
            return self._streamed_verdict(budget_bytes)
        if not self.ingest_priced:
            text = (f"the forecast needs "
                    f"{self.forecast_envelope_bytes / GIB:.2f} GiB peak "
                    f"envelope, and preprocessing for --source "
                    f"{self.source} is NOT PRICED here, so this is the "
                    "forecast phase only")
        else:
            phase = self.binding_phase
            label = ("preprocessing (ingest)" if phase == "ingest"
                     else "the forecast")
            text = (f"{label} is the memory-binding phase at "
                    f"{self.peak_envelope_bytes / GIB:.2f} GiB peak "
                    f"envelope (forecast "
                    f"{self.forecast_envelope_bytes / GIB:.2f} GiB, ingest "
                    f"{self.ingest_envelope_bytes / GIB:.2f} GiB)")
        if budget_bytes is None:
            return text
        if self.fits(budget_bytes):
            return (f"{text}; it fits the "
                    f"{budget_bytes / GIB:.2f} GiB budget with "
                    f"{(budget_bytes - self.peak_envelope_bytes) / GIB:.2f} "
                    "GiB to spare")
        return (f"{text}; that EXCEEDS the {budget_bytes / GIB:.2f} GiB "
                f"budget by "
                f"{(self.peak_envelope_bytes - budget_bytes) / GIB:.2f} GiB")

    def _streamed_verdict(self, budget_bytes: int | None) -> str:
        env = self.streamed
        phase = self.binding_phase
        label = ("preprocessing (ingest)" if phase == "ingest"
                 else "the streamed forecast")
        parts = [f"{label} is the memory-binding phase at "
                 f"{self.peak_envelope_bytes / GIB:.2f} GiB peak envelope "
                 f"(streamed forecast "
                 f"{self.forecast_envelope_bytes / GIB:.2f} GiB"]
        if self.ingest_priced:
            parts.append(f", ingest {self.ingest_envelope_bytes / GIB:.2f} "
                         "GiB")
        parts.append(")")
        text = "".join(parts)
        if self.resident_forecast_envelope_bytes is not None:
            text += (f"; {env.nbuffers} tile buffer(s) of "
                     f"{env.window_nx}x{env.window_ny} instead of the whole "
                     f"domain, against "
                     f"{self.resident_forecast_envelope_bytes / GIB:.2f} GiB "
                     "resident")
        if env.radiation_transient_bytes:
            # NAMED WHERE THE FIGURE IS READ.  A forecast term larger than
            # the tile buffers it describes reads as an arithmetic error
            # unless the reservation is stated beside it.
            text += (f"; the tiles hold {env.vram_bytes / GIB:.2f} GiB and "
                     f"the RRTMGP call adds a measured "
                     f"{env.radiation_transient_bytes / GIB:.2f} GiB "
                     f"transient on top of them, which is the peak the card "
                     "has to hold")
        text += (f", with the forecast itself in "
                 f"{env.host_bytes / GIB:.2f} GiB of pinned host RAM")
        if budget_bytes is None:
            return text
        if self.fits(budget_bytes):
            return (f"{text}; it fits the {budget_bytes / GIB:.2f} GiB "
                    "budget with "
                    f"{(budget_bytes - self.peak_envelope_bytes) / GIB:.2f} "
                    "GiB to spare")
        return (f"{text}; that EXCEEDS the {budget_bytes / GIB:.2f} GiB "
                f"budget by "
                f"{(self.peak_envelope_bytes - budget_bytes) / GIB:.2f} GiB")


def streamed_forecast_envelope(exp: ExperimentConfig, *, machine=None):
    """The ROOT domain's streamed envelope under this config's ``[tiles]``.

    ``None`` whenever this configuration does not stream, which includes the
    cases where the question cannot be answered here: ``mode = "auto"`` with
    no pinned tiling has to consult the planner, and the planner needs a
    card.  Passing ``machine`` (built from an out-of-process probe, never
    from ``Machine.detect`` inside a long-lived CLI) is what lets ``auto``
    be priced.

    NEVER RAISES.  This runs inside an admission gate whose whole job is to
    answer before the user spends a download, and a planner refusal
    ("no tile fits in this budget") is a legitimate answer meaning "streaming
    will not save this either" -- the caller then prices the resident
    envelope and refuses on that, which is the correct and conservative
    outcome.

    The root domain only, deliberately: a nested tree is refused by
    ``prepared_domain_builder`` for a nest anyway, so pricing a nest's
    streamed envelope would describe a run that cannot happen.
    """
    options = getattr(exp, "tiles", None)
    if options is None or getattr(options, "mode", "off") == "off":
        return None
    if not getattr(exp, "domains", None):
        return None
    from gpuwm.core import streaming

    try:
        return streaming.streamed_envelope(
            exp.domains[0].run, options, machine=machine)
    except Exception:                    # a gate never dies on its estimate
        return None


def estimate_phases(exp: ExperimentConfig, *, source: str,
                    column_chunk: int | None = None,
                    forcing_interval_seconds: float
                    = DEFAULT_FORCING_INTERVAL_SECONDS,
                    ingest_forcing_interval_seconds: float | None = None,
                    vram_gib: float | None = None,
                    profile: DeviceLocalMemoryProfile | None = None,
                    machine=None,
                    ) -> PhaseMemoryEstimate:
    """Price every phase of ``exp`` and say which one binds the card.

    ``ingest_forcing_interval_seconds`` defaults to the SOURCE's own
    fetch cadence rather than the forecast's LBC interval, because the
    ingest phase's time count is set by what was downloaded.

    ``[tiles]`` REPLACES THE FORECAST TERM (2.2.0).  Every enumeration in
    this module itemizes a domain resident in VRAM, so with streaming
    configured the forecast term described a run that was not going to
    happen -- and the gate built on it refused, by default, the one
    configuration class streaming exists to enable.  When
    :func:`streamed_forecast_envelope` returns a number the forecast term
    becomes that number; the ingest term is untouched, because preprocessing
    is not streamed and still has to fit the card on its own.

    ``machine`` is only consulted for ``mode = "auto"`` with no pinned
    tiling, where the decision belongs to the planner.
    """
    forecast = estimate_experiment(
        exp, column_chunk=column_chunk,
        forcing_interval_seconds=forcing_interval_seconds,
        vram_gib=vram_gib, profile=profile)
    key = None if source is None else str(source).strip().lower()
    ingest = None
    if key in SOURCE_ANALYSIS_LEVELS:
        cadence = ingest_forcing_interval_seconds
        if cadence is None:
            cadence = INGEST_FORCING_CADENCE_SECONDS.get(
                key, forcing_interval_seconds)
        ingest = estimate_ingest(
            exp, source=key, forcing_interval_seconds=float(cadence),
            vram_gib=vram_gib, profile=profile)
    resident_forecast = forecast.peak_envelope_bytes
    streamed = streamed_forecast_envelope(exp, machine=machine)
    return PhaseMemoryEstimate(
        forecast=forecast, ingest=ingest,
        # THE PEAK, NOT THE HOLD.  ``vram_bytes`` is what a streamed
        # forecast holds between radiation calls; the RRTMGP call's
        # measured per-process transient is on the card too at the instant
        # it matters, and the first call is itimestep == 1.  Pricing the
        # hold here let a card that fits 6.71 GiB admit a run that reaches
        # 9.45 -- every surface below reads this field.
        forecast_envelope_bytes=(resident_forecast if streamed is None
                                 else int(streamed.peak_vram_bytes)),
        ingest_envelope_bytes=(None if ingest is None
                               else ingest.peak_envelope_bytes),
        source=key, streamed=streamed,
        resident_forecast_envelope_bytes=resident_forecast,
    )


def pool_retention_residual_bytes() -> int:
    """Tier-2 RUN-gate reserve term (provisional policy): the d01 RUN
    fixture's pool-held bytes beyond the d01 alloc-estimate basis
    (measured used-peak + default-chunk workspace, with headroom).
    Calibrated, clamped at zero: if the enumerated workspace already
    over-covers the fixture's retention, nothing extra is reserved (no
    double count).  The N0 probe measured allocation-time retention NIL,
    so this term belongs to the N5/N6 run gates, not the N0 alloc gate
    (:meth:`ReservePolicy.run_time` vs :meth:`ReservePolicy.n0_alloc` --
    split pending controller ratification)."""
    meta_free_basis = math.ceil(ALLOCATOR_HEADROOM * (
        CAL_D01_POOL_USED_PEAK_BYTES + CAL_D01_WORKSPACE_BYTES))
    return max(0, CAL_D01_POOL_HELD_BYTES - meta_free_basis)


def _workspace_total_bytes(nz: int, column_chunk: int,
                           p_top: float = 5000.0) -> int:
    return sum(math.prod(shape) * size for shape, size in
               rrtmgp_workspace_shapes(nz, column_chunk, p_top).values())


def estimate_experiment(
        exp: ExperimentConfig, *,
        column_chunk: int | None = None,
        forcing_interval_seconds: float = DEFAULT_FORCING_INTERVAL_SECONDS,
        vram_gib: float | None = None,
        profile: DeviceLocalMemoryProfile | None = None,
) -> ExperimentMemoryEstimate:
    """Sum the per-domain itemizations; count the lru_cache-shared
    k-distribution tables ONCE (rrtmgp.py:324/:436 -- baseline behavior,
    never claimed as savings); replace audited scratch with its per-slot max
    for multi-domain experiments; size the ONE explicitly allocated and
    adapter-consumed shared chunk workspace; apply the 15% allocator-headroom
    factor. The result is compared against the
    MEASURED budget (free VRAM at startup minus the configured reserve --
    never nominal 32 GiB)."""
    column_chunk = (exp.column_chunk if column_chunk is None
                    else column_chunk)
    if (isinstance(column_chunk, bool)
            or not isinstance(column_chunk, int)
            or column_chunk < 1):
        raise ValueError(
            "column_chunk must be a positive integer number of radiation "
            f"columns, got {column_chunk!r}.")
    n_int = lbc_intervals(exp.run_seconds, forcing_interval_seconds)
    by_id = {dc.grid_id: dc for dc in exp.domains}
    domains = tuple(
        estimate_domain(
            dc, spec_bdy_width=exp.spec_bdy_width,
            parent=(None if dc.parent_id == 0 else by_id[dc.parent_id]),
            n_lbc_intervals=n_int, p_top=exp.vertical.p_top,
            column_chunk=column_chunk)
        for dc in exp.domains)
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    variants_44 = {rrtmg_variant(dc.run) for dc in exp.domains
                   if radiation_scheme_ids(dc.run) == (4, 4)}
    if len(variants_44) > 1:
        raise ValueError(
            "mixed ra_rrtmg_variant values across 4/4 domains are not "
            f"supported by the VRAM preflight: {sorted(variants_44)}")
    uses_rrtmgp = bool(variants_44) and variants_44 != {RRTMG_VARIANT_LEGACY}
    uses_legacy = variants_44 == {RRTMG_VARIANT_LEGACY}
    legacy_envelope = 0
    if uses_legacy:
        # One shared call-peak term: the legacy engines run one domain
        # at a time with LW/SW sequenced and freed in between, so the
        # experiment-wide radiation transient is the max single-call
        # envelope (engine-default chunking -- the adapter's
        # column_chunk=None contract).
        from gpuwm.core.rrtmg_legacy import legacy_radiation_vram_bytes
        legacy_envelope = max(
            legacy_radiation_vram_bytes(
                ncol=dc.run.ny * dc.run.nx, nz=dc.run.nz,
                p_top=exp.vertical.p_top, column_chunk=None)
            for dc in exp.domains
            if radiation_scheme_ids(dc.run) == (4, 4))
    uses_arena = len(exp.domains) > 1
    arena_shapes = (shared_scratch_arena_shapes(exp.domains)
                    if uses_arena else {})
    nz = exp.domains[0].run.nz
    return ExperimentMemoryEstimate(
        domains=domains,
        k_tables_bytes=k_distribution_bytes() if uses_rrtmgp else 0,
        workspace_bytes=(_workspace_total_bytes(
            nz, column_chunk, exp.vertical.p_top)
                          if uses_rrtmgp else legacy_envelope),
        scratch_arena_bytes=(shared_scratch_arena_bytes(exp.domains)
                             if uses_arena else 0),
        uses_shared_scratch_arena=uses_arena,
        dycore_state_workspace_bytes=(
            shared_dycore_state_workspace_bytes(exp.domains)
            if uses_arena else 0),
        uses_shared_dycore_state_workspace=uses_arena,
        column_chunk=column_chunk,
        retention_residual_bytes=(
            platform_projection_constants(vram_gib=vram_gib)[0]
            if uses_rrtmgp else 0),
        device_overhead_bytes=platform_projection_constants(
            vram_gib=vram_gib)[1],
        non_pool_device_bytes=non_pool_device_bytes(
            exp, profile=(card_local_memory_profile(vram_gib)
                          if profile is None else profile)),
        envelope_family=envelope_platform(vram_gib=vram_gib),
        uses_legacy_radiation=uses_legacy,
    )


# ---------------------------------------------------------------------------
# Reserve policy + gate evaluation (F11 chain, exact comparisons)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReservePolicy:
    """Configured reserve: what the MEASURED budget subtracts from free
    VRAM.  Two proposals, split by gate, PENDING CONTROLLER RATIFICATION
    at N0 (``--reserve-gib`` overrides with a flat value):

    * :meth:`n0_alloc` -- the allocation-gate reserve: the probe-measured
      fresh-process non-pool overhead (1.39 GiB) + external margin;
      alloc-time pool retention was measured nil, so no retention term.
    * :meth:`run_time` -- the N5/N6 run-gate reserve: adds the
      fixture-calibrated run-churn retention residual.  The probe
      overhead is a lower bound at run time (JIT module loads), which
      the external margin partially covers -- flagged, not resolved.
    """

    retention_residual_bytes: int
    device_overhead_bytes: int
    external_margin_bytes: int = EXTERNAL_MARGIN_BYTES

    @classmethod
    def n0_alloc(cls, exp: ExperimentConfig | None = None, *,
                 profile: DeviceLocalMemoryProfile | None = None,
                 estimate_bytes: int | None = None) -> "ReservePolicy":
        """Allocation-gate reserve.

        With an experiment, ``device_overhead_bytes`` is the MEASURED
        non-pool residency of a process running THAT configuration -- CUDA
        context plus the local-memory backing store its widest launched
        kernel reserves (:func:`non_pool_device_bytes`).  Without one it
        falls back to the 2026-07-16 zero-step probe constant, which is the
        number that was 5.5 GiB short of every real run.

        With ``estimate_bytes``, the retention term carries
        :data:`POOL_RESERVED_OVER_ESTIMATE_FRACTION` -- the measured gap
        between what the pool hands out and what it holds onto.
        """
        overhead = (PROBE_DEVICE_OVERHEAD_BYTES if exp is None
                    else non_pool_device_bytes(exp, profile=profile))
        retention = (0 if estimate_bytes is None else math.ceil(
            POOL_RESERVED_OVER_ESTIMATE_FRACTION * int(estimate_bytes)))
        return cls(retention_residual_bytes=retention,
                   device_overhead_bytes=overhead)

    @classmethod
    def run_time(cls, exp: ExperimentConfig | None = None, *,
                 profile: DeviceLocalMemoryProfile | None = None
                 ) -> "ReservePolicy":
        overhead = (PROBE_DEVICE_OVERHEAD_BYTES if exp is None
                    else non_pool_device_bytes(exp, profile=profile))
        return cls(retention_residual_bytes=pool_retention_residual_bytes(),
                   device_overhead_bytes=overhead)

    @classmethod
    def flat(cls, reserve_bytes: int) -> "ReservePolicy":
        return cls(retention_residual_bytes=0, device_overhead_bytes=0,
                   external_margin_bytes=int(reserve_bytes))

    @property
    def reserve_bytes(self) -> int:
        return (self.retention_residual_bytes + self.device_overhead_bytes
                + self.external_margin_bytes)

    def budget_bytes(self, measured_free_bytes: int) -> int:
        return int(measured_free_bytes) - self.reserve_bytes


def evaluate_alloc_gates(*, measured_used_bytes: int | None,
                         estimate_bytes: int,
                         measured_free_bytes: int | None,
                         reserve: ReservePolicy) -> dict[str, bool | None]:
    """The F11 enforced chain, evaluated EXACTLY (no tolerance), keyed by
    the pre-registered N0 ledger records (verify/nest_gates.py).  ``None``
    marks a leg whose measurement is unavailable (estimator-only mode);
    a missing measurement can never report a pass."""
    budget = (None if measured_free_bytes is None
              else reserve.budget_bytes(measured_free_bytes))
    return {
        "alloc_fits_wddm_budget": (
            None if (measured_used_bytes is None or budget is None)
            else measured_used_bytes <= budget),
        "alloc_measured_le_estimate": (
            None if measured_used_bytes is None
            else measured_used_bytes <= estimate_bytes),
        "alloc_estimate_le_wddm_budget": (
            None if budget is None else estimate_bytes <= budget),
    }


def device_wide_used_bytes() -> int:
    """Device memory held by EVERY process on the card, from NVML.

    ``cudaMemGetInfo`` is not a substitute here.  On this WDDM host it
    reported 1,614 MiB used at a moment NVML reported 3,903 MiB -- it does
    not see the desktop compositor's allocations, so a budget built from its
    ``free`` over-states the card by ~2.3 GiB.  Its DELTAS are exact (a
    traced run's memGetInfo-derived process peak agreed with the NVML
    device-wide series to 27 MiB), which is why the estimator uses it for
    growth and NVML for the whole-machine rail.

    Fails closed through ``supervisor.GPUPreflightError`` when nvidia-smi is
    unavailable: a rail that cannot be measured must never read as headroom.
    """
    from gpuwm.supervisor import _run_nvidia_smi

    text = _run_nvidia_smi(["--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"])
    return int(text.strip().splitlines()[0]) * 1024 ** 2


def device_physical_total_bytes() -> int | None:
    """Physical VRAM total of the card, from NVML, or None if unreadable.

    NVML and not ``cudaMemGetInfo``: capacity is the one device question
    that must be answerable in CPU mode, and ``memGetInfo`` cannot be
    asked without standing up a CUDA context -- which is exactly what an
    estimator-mode preflight promises not to do.  ``device_wide_used_bytes``
    already argues NVML is the authority on this host anyway.

    Unreadable is None, never a number: this figure is used as a CEILING
    only, so a card that cannot be read simply imposes no ceiling.
    """
    try:
        from gpuwm.supervisor import _run_nvidia_smi

        text = _run_nvidia_smi(["--query-gpu=memory.total",
                                "--format=csv,noheader,nounits"])
        total = int(text.strip().splitlines()[0]) * 1024 ** 2
    except Exception:
        return None
    return total if total > 0 else None


def cap_free_to_physical(free_bytes: int, *,
                         card_total_bytes: int | None,
                         measured_total_bytes: int | None
                         ) -> tuple[int, int | None]:
    """Clamp a free-VRAM figure to what the card physically has.

    A DERIVED free figure is arithmetic (declared budget + reserve) and
    nothing in that arithmetic knows how big the card is: the ``--card
    16gb`` wizard tier produced a notional free of 16.68 GiB on a card
    whose physical total is 15.57 GiB, which then bought the estimate a
    gigabyte of budget that does not exist.  Free can never exceed total,
    on any card, so the smaller of the two capacity statements we have --
    the caller's declared card size and a live measurement of it -- is an
    ADDITIONAL ceiling, in the same idiom as the device rail: never a
    replacement and never a widening.

    Returns ``(free, cap)`` where ``cap`` is the ceiling that actually
    bound, or None when the figure was already within capacity.
    """
    caps = [int(c) for c in (card_total_bytes, measured_total_bytes)
            if c is not None and int(c) > 0]
    if not caps:
        return int(free_bytes), None
    cap = min(caps)
    if int(free_bytes) <= cap:
        return int(free_bytes), None
    return cap, cap


def device_rail_free_bytes(rail_bytes: int, *,
                           other_process_bytes: int) -> int:
    """Bytes this process may hold before the WHOLE CARD crosses the rail.

    The rail is a property of the host, not of the model: this box has no
    ECC and has already corrupted fine-domain output near capacity, so the
    bar is on total device residency, and every other process on the card --
    a 3.4 GiB desktop, here -- spends it.
    """
    return int(rail_bytes) - int(other_process_bytes)


def recommend_column_chunk(exp: ExperimentConfig, budget_bytes: int, *,
                           start_chunk: int | None = None,
                           floor: int = 256) -> int | None:
    """FIRST over-budget lever (robust-5 / architecture section E): the
    largest halving of ``start_chunk`` (down to ``floor``) whose estimate
    fits the budget, or None if none fits."""
    chunk = int(exp.column_chunk if start_chunk is None else start_chunk)
    while chunk >= floor:
        estimate = estimate_experiment(exp, column_chunk=chunk)
        if estimate.alloc_estimate_bytes <= budget_bytes:
            return chunk
        chunk //= 2
    return None


# ---------------------------------------------------------------------------
# N0 --alloc runner (GPU; controller-run)
# ---------------------------------------------------------------------------

class PreflightHeadroomError(RuntimeError):
    """Free VRAM is short of the remaining estimate BEFORE construction.

    Carries structured fields so ``check_main`` can still emit the full
    JSON report (estimate-side legs evaluated, abort reason recorded)
    with an exit code distinct from a gate-leg FAIL.
    """

    def __init__(self, message: str, *, phase: str, free_bytes: int,
                 total_bytes: int, reserve_bytes: int,
                 remaining_bytes: int):
        super().__init__(message)
        self.phase = phase
        self.free_bytes = int(free_bytes)
        self.total_bytes = int(total_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.remaining_bytes = int(remaining_bytes)


class PreflightAllocError(RuntimeError):
    """Device allocation failed during --alloc; diagnostics attached."""

    def __init__(self, message: str, *, phase: str,
                 free_bytes: int | None = None):
        super().__init__(message)
        self.phase = phase
        self.free_bytes = None if free_bytes is None else int(free_bytes)


@dataclass
class AllocReport:
    """Measured N0 numbers + gate legs from one --alloc run."""

    estimate: ExperimentMemoryEstimate
    reserve: ReservePolicy
    free_before_bytes: int
    total_bytes: int
    pool_used_peak_bytes: int
    pool_held_peak_bytes: int
    free_at_peak_bytes: int
    free_after_release_bytes: int
    gates: dict[str, bool | None]

    @property
    def device_footprint_bytes(self) -> int:
        return self.free_before_bytes - self.free_at_peak_bytes

    @property
    def measured_overhead_bytes(self) -> int:
        """Re-calibrated tier-3 term: measured footprint minus pool held."""
        return self.device_footprint_bytes - self.pool_held_peak_bytes

    @property
    def passed(self) -> bool:
        return all(leg is True for leg in self.gates.values())


def _require_headroom(cp, remaining_bytes: int, reserve: ReservePolicy,
                      phase: str) -> None:
    """Robust-5: headroom check before state construction and before each
    high-water phase.  Fails loudly BEFORE the allocation that would OOM."""
    free, total = cp.cuda.runtime.memGetInfo()
    if free - reserve.reserve_bytes < remaining_bytes:
        raise PreflightHeadroomError(
            f"headroom check failed before {phase}: free {free / GIB:.2f} "
            f"GiB minus reserve {reserve.reserve_bytes / GIB:.2f} GiB is "
            f"short of the remaining itemized need "
            f"{remaining_bytes / GIB:.2f} GiB (device total "
            f"{total / GIB:.2f} GiB)",
            phase=phase, free_bytes=free, total_bytes=total,
            reserve_bytes=reserve.reserve_bytes,
            remaining_bytes=remaining_bytes)


def _synthetic_root_boundaries(cfg: RunConfig, n_intervals: int):
    """Zero-valued LateralBoundaries with the exact real74 table shapes,
    so attach_lateral_boundaries performs the REAL eager device upload."""
    import numpy as np
    from gpuwm.ingest.lateral_bc import build_lateral_boundaries

    snapshot = {name: np.zeros(dims, dtype=np.float64)
                for name, dims in _lbc_field_dims(cfg).items()}
    # mu snapshots are 2-D in domain_boundary_snapshot; (1, ny, nx) works
    # identically through _field_boundary, but keep the exact contract.
    snapshot["mu"] = np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
    times = [float(i) * DEFAULT_FORCING_INTERVAL_SECONDS
             for i in range(n_intervals + 1)]
    return build_lateral_boundaries(
        [snapshot] * (n_intervals + 1), times,
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)


def _materialize_physics(state, cfg: RunConfig, start_time: datetime,
                         *, glw=None):
    """initialize_physics + the steady-state extras the first steps would
    allocate (composed rqr/rqi/rqs, Morrison accumulator optionals, KF W0AVG),
    so the alloc proof covers the run's real persistent driver set.

    ``glw`` is the experiment's DECLARED constant downward longwave, or
    None: initialize_physics refuses to invent a GLW that something
    would consume or publish, so an alloc proof for a config that
    declared the constant (``constant-downward-longwave-v1``) must type
    the same declaration here that the run preparers type -- otherwise
    ``gpuwm check --alloc`` false-refuses the very configs whose device
    footprint it exists to measure (the MYNN no-radiation d04 pair)."""
    import cupy as cp
    import numpy as np
    from gpuwm.core.physics import (PBL_RQI_MICROPHYSICS, initialize_physics,
                                    physics_retains_ysu_output,
                                    physics_reuses_pbl_composition)
    from gpuwm.core.state import DTYPE

    ny, nx = cfg.ny, cfg.nx
    grid = np.zeros((ny, nx), dtype=np.float64)
    driver = initialize_physics(
        state, cfg, landmask=1.0, tsk=290.0, soil_temperature=285.0,
        soil_moisture=0.30, glw=glw, radiation_start_time=start_time,
        radiation_latitude=grid, radiation_longitude=grid + 1.0)
    m = state.p.shape
    zero_m = lambda: cp.zeros(m, dtype=DTYPE)
    zero_s = lambda: cp.zeros((ny, nx), dtype=DTYPE)
    if cfg.cu_physics:
        target = (driver.pbl_tendencies
                  if physics_reuses_pbl_composition(cfg)
                  else driver.tendencies)
        for comp in ("rqr", "rqi", "rqs"):
            if getattr(driver.cumulus_tendencies, comp) is None:
                setattr(driver.cumulus_tendencies, comp, zero_m())
            if getattr(target, comp) is None:
                setattr(target, comp, zero_m())
        adapter = driver.cumulus_callable
        if (int(cfg.cu_physics) == 1 and adapter is not None
                and getattr(adapter, "w0avg", None) is None):
            adapter.w0avg = zero_m()          # kf.py:174 shape contract
            adapter._history_state = state
    if cfg.bl_pbl_physics and cfg.mp_physics in PBL_RQI_MICROPHYSICS:
        # The materialization side of the pbl_tendencies/rqi budget above.
        # 28 belongs for the same reason: Registry/Registry.EM_COMMON:3036
        # gives the thompsonaero package qi in moist, so WRF's F_QI is true;
        # 16 (wdm6scheme, :3031) declares the same moist inventory.
        # This set and physics._pbl_optional_tendency_components must agree,
        # or the --alloc measurement stops covering true runtime residency --
        # which is why it is now ONE named constant read by all three sites
        # (the shapes budget above, this materializer, and the physics
        # module), pinned equal by
        # tests/test_preflight.py::test_the_rqi_budget_shapes_materialization
        # _and_physics_name_one_set.
        if driver.pbl_tendencies.rqi is None:
            driver.pbl_tendencies.rqi = zero_m()
        if ((radiation_enabled(cfg) or cfg.cu_physics)
                and not physics_reuses_pbl_composition(cfg)):
            if driver.tendencies.rqi is None:
                driver.tendencies.rqi = zero_m()
    if cfg.mp_physics in (6, 10):
        # PhysicsDriver initialization already aliases and materializes all
        # seven carrying mp_* scratch slots; only the behavior-gating counter
        # changes after the first real scheme call.
        driver.microphysics_updates = 1
    if physics_retains_ysu_output(cfg):
        # Materialize the positive-cadence retained YSU output set so the
        # --alloc measurement covers true runtime residency, not just
        # construction (review fix round): same shapes/dtypes as
        # launch_ysu's out dict (ysu.py:79-92), zero-filled.
        last_ysu = {name: zero_m() for name in
                    ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                     "exch_h", "exch_m")}
        for name in ("hpbl", "wstar", "delta", "topdown_radsum",
                     "wstar3_2"):
            last_ysu[name] = zero_s()
        for name in ("kpbl", "cloudflg"):
            last_ysu[name] = cp.zeros((ny, nx), dtype=cp.int32)
        driver.last_ysu = last_ysu
    return driver


def run_alloc_preflight(
        exp: ExperimentConfig, *,
        column_chunk: int | None = None,
        forcing_interval_seconds: float = DEFAULT_FORCING_INTERVAL_SECONDS,
        reserve: ReservePolicy | None = None,
        profile: DeviceLocalMemoryProfile | None = None) -> AllocReport:
    """N0: construct all DomainStates + drivers + d01 LBC + the F4 nest
    manifest allocations + the RRTMGP workspace on the real device, run
    ZERO steps, report pool used/peak vs estimate, free.

    OOM policy (robust-5): any device allocation failure records the
    phase and pool/device diagnostics and TERMINATES the worker via
    :class:`PreflightAllocError` -- never ``free_all_blocks()`` and
    continue.
    """
    import cupy as cp

    from gpuwm.core.state import (DTYPE, DomainState,
                                  build_shared_dycore_state_workspace,
                                  build_shared_scratch_arena)
    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries

    reserve = ReservePolicy.n0_alloc() if reserve is None else reserve
    estimate = estimate_experiment(
        exp, column_chunk=column_chunk,
        forcing_interval_seconds=forcing_interval_seconds, profile=profile)
    column_chunk = estimate.column_chunk
    n_int = lbc_intervals(exp.run_seconds, forcing_interval_seconds)

    pool = cp.get_default_memory_pool()
    free_before, total = cp.cuda.runtime.memGetInfo()
    used_peak = pool.used_bytes()
    held_peak = pool.total_bytes()
    free_at_peak = free_before

    def sample(phase: str) -> None:
        nonlocal used_peak, held_peak, free_at_peak
        cp.cuda.runtime.deviceSynchronize()
        used_peak = max(used_peak, pool.used_bytes())
        held_peak = max(held_peak, pool.total_bytes())
        free_at_peak = min(free_at_peak, cp.cuda.runtime.memGetInfo()[0])

    per_domain = {}
    for d in estimate.domains:
        resident = d.resident_bytes
        if estimate.uses_shared_scratch_arena:
            resident -= d.arena_scratch_bytes
        if estimate.uses_shared_dycore_state_workspace:
            resident -= d.rebuilt_state_bytes
        per_domain[d.grid_id] = resident
    remaining = (sum(per_domain.values()) + estimate.scratch_arena_bytes
                 + estimate.dycore_state_workspace_bytes
                 + estimate.workspace_bytes + estimate.k_tables_bytes)
    holdings: list[object] = []
    phase = "startup"
    arena = None
    dycore_state_workspace = None
    try:
        if estimate.uses_shared_dycore_state_workspace:
            phase = "shared dycore-state workspace"
            _require_headroom(cp, remaining, reserve, phase)
            dycore_state_workspace = build_shared_dycore_state_workspace(
                exp.domains)
            if (dycore_state_workspace.nbytes
                    != estimate.dycore_state_workspace_bytes):
                raise RuntimeError(
                    "runtime shared dycore-state workspace drifted from "
                    f"the estimator: {dycore_state_workspace.nbytes} != "
                    f"{estimate.dycore_state_workspace_bytes} bytes")
            holdings.append(dycore_state_workspace)
            sample(phase)
            remaining -= estimate.dycore_state_workspace_bytes

        if estimate.uses_shared_scratch_arena:
            phase = "shared transient-scratch arena"
            _require_headroom(cp, remaining, reserve, phase)
            arena = build_shared_scratch_arena(exp.domains)
            if arena.nbytes != estimate.scratch_arena_bytes:
                raise RuntimeError(
                    "runtime scratch arena drifted from the estimator: "
                    f"{arena.nbytes} != {estimate.scratch_arena_bytes} bytes")
            holdings.append(arena)
            sample(phase)
            remaining -= estimate.scratch_arena_bytes

        states: dict[int, DomainState] = {}
        by_id = {dc.grid_id: dc for dc in exp.domains}
        for dc in exp.domains:
            phase = f"domain d{dc.grid_id:02d} construction"
            _require_headroom(cp, remaining, reserve, phase)
            # Preserve the historical constructor call on the default/single-
            # domain path; only a multi-domain experiment receives workspaces.
            state_kwargs = {}
            if arena is not None:
                state_kwargs["scratch_arena"] = arena
            if dycore_state_workspace is not None:
                state_kwargs["dycore_state_workspace"] = (
                    dycore_state_workspace)
            state = DomainState(dc.run, **state_kwargs)
            states[dc.grid_id] = state
            # Prewarm every registry scratch slot (robust-5: persistent
            # scratch prewarmed at setup).  The root's packed forcing
            # tables come from the REAL attach below, not a prewarm.
            registry = scratch_slot_registry(dc.run, n_lbc_intervals=0)
            for slot, shape in registry.items():
                state.scratch(shape, slot)
            if dc.parent_id == 0 and dc.run.specified:
                attach_lateral_boundaries(
                    state, _synthetic_root_boundaries(dc.run, n_int))
                # Davies weights are created on first force; prove their
                # bytes now under the exact resident slot name.
                state.scratch((2, dc.run.spec_bdy_width), "lbc_weights_0")
            if dc.parent_id != 0:
                shapes = nest_slot_shapes(
                    dc, exp.spec_bdy_width, by_id[dc.parent_id])
                dtypes = nest_slot_dtypes(
                    dc, exp.spec_bdy_width, by_id[dc.parent_id])
                for slot, shape in shapes.items():
                    state.scratch(shape, slot, dtype=dtypes[slot])
            from gpuwm.core.physics_inventory import physics_driver_required
            if physics_driver_required(dc.run):
                # The experiment's declared constant GLW (or None),
                # exactly as prepare_real_case/prepare_child_case type
                # it: a config that legitimately declared
                # constant-downward-longwave-v1 must reach the device
                # here too, or --alloc refuses the very footprint
                # measurement it exists for.
                from gpuwm.runtime import declared_constant_glw
                _materialize_physics(state, dc.run, exp.start_time,
                                     glw=declared_constant_glw(exp))
            sample(phase)
            remaining -= per_domain[dc.grid_id]

        phase = "rrtmgp k-tables + shared chunk workspace"
        _require_headroom(cp, remaining, reserve, phase)
        from gpuwm.physics_compat import (RRTMG_VARIANT_LEGACY,
                                          rrtmg_variant)
        legacy_44 = [dc for dc in exp.domains
                     if radiation_scheme_ids(dc.run) == (4, 4)
                     and rrtmg_variant(dc.run) == RRTMG_VARIANT_LEGACY]
        if legacy_44:
            # Legacy variant: hold the priced call-peak envelope (the
            # estimate's shared term) so the residency proof covers the
            # adapter's worst single call.
            phase = "legacy-RRTMG call-peak envelope"
            from gpuwm.core.rrtmg_legacy import legacy_radiation_vram_bytes
            envelope = max(
                legacy_radiation_vram_bytes(
                    ncol=dc.run.ny * dc.run.nx, nz=dc.run.nz,
                    p_top=exp.vertical.p_top, column_chunk=None)
                for dc in legacy_44)
            holdings.append(cp.zeros(envelope, dtype=cp.uint8))
        if any(radiation_scheme_ids(dc.run) == (4, 4)
               and rrtmg_variant(dc.run) != RRTMG_VARIANT_LEGACY
               for dc in exp.domains):
            from gpuwm.core.rrtmgp import load_cloud_tables, load_gas_tables
            for kind in ("lw", "sw"):
                holdings.append(load_gas_tables(kind).to_device())
                holdings.append(load_cloud_tables(kind).to_device())
            nz = exp.domains[0].run.nz
            for name, (shape, size) in rrtmgp_workspace_shapes(
                    nz, column_chunk, exp.vertical.p_top).items():
                dtype = cp.bool_ if size == 1 else DTYPE
                holdings.append(cp.zeros(shape, dtype=dtype))
        if any(dc.run.cu_physics == 1 for dc in exp.domains):
            # The once-per-process KF device LUT (kf.py _device_table):
            # materialized so the residency proof covers it.
            from gpuwm.core.kf import _device_table
            holdings.append(_device_table())
        sample(phase)

        gates = evaluate_alloc_gates(
            measured_used_bytes=used_peak,
            estimate_bytes=estimate.alloc_estimate_bytes,
            measured_free_bytes=free_before, reserve=reserve)
    except cp.cuda.memory.OutOfMemoryError as exc:
        free_now, _ = cp.cuda.runtime.memGetInfo()
        raise PreflightAllocError(
            f"--alloc OOM during {phase}: pool used "
            f"{pool.used_bytes() / GIB:.2f} GiB, held "
            f"{pool.total_bytes() / GIB:.2f} GiB, device free "
            f"{free_now / GIB:.2f} GiB of {total / GIB:.2f} GiB; itemized "
            f"estimate {estimate.alloc_estimate_bytes / GIB:.2f} GiB. "
            "Terminating (never free_all_blocks-and-continue); first "
            "lever: shrink the RRTMGP column_chunk.",
            phase=phase, free_bytes=free_now,
        ) from exc

    # Zero steps by contract.  Free everything and report the release.
    # The driver <-> state attachment is a reference cycle; collect it so
    # the arrays actually return to the pool before free_all_blocks.
    import gc
    holdings.clear()
    states.clear()
    # Loop locals otherwise keep the final state (and through it the shared
    # arena) alive past free_all_blocks().
    state = None
    arena = None
    dycore_state_workspace = None
    gc.collect()
    pool.free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    return AllocReport(
        estimate=estimate, reserve=reserve,
        free_before_bytes=int(free_before), total_bytes=int(total),
        pool_used_peak_bytes=int(used_peak),
        pool_held_peak_bytes=int(held_peak),
        free_at_peak_bytes=int(free_at_peak),
        free_after_release_bytes=int(free_after), gates=gates)


# ---------------------------------------------------------------------------
# CLI (`gpuwm check`) -- registrar only; cli.py hookup is a controller
# handoff commit (F2 ownership map).
# ---------------------------------------------------------------------------

def _load_experiment_any(path: Path) -> ExperimentConfig:
    import io
    import tomllib

    from gpuwm.case_data import load_experiment_case
    from gpuwm.config import load_config
    from gpuwm.config_authority import read_config_authority
    from gpuwm.experiment import (build_experiment,
                                  experiment_from_run_config,
                                  is_experiment_toml_bytes)

    authority = read_config_authority(path)
    if is_experiment_toml_bytes(authority.payload):
        raw = tomllib.load(io.BytesIO(authority.payload))
        if "case_data" not in raw:
            # `gpuwm domain --source gfs|hrrr` deliberately emits no
            # [case_data]: those tables feed the native front door.  The
            # memory preflight needs only the experiment geometry, so
            # validate the advisory [fetch] hints and load the tables.
            fetch_table = raw.pop("fetch", None)
            if fetch_table is not None:
                from gpuwm.fetch import validate_fetch_hints
                validate_fetch_hints(fetch_table, source=str(path))
            return build_experiment(raw, source=str(path))
        exp, _case_data = load_experiment_case(path)
        return exp
    return experiment_from_run_config(load_config(path),
                                      datetime(1970, 1, 1))


def config_forcing_source(path: Path, *,
                          priced_only: bool = True) -> str | None:
    """The forcing product a config records, or None if it records none.

    The preprocessing phase is priced against the SOURCE's level count
    and field inventory, so an unpriceable source has to be said out
    loud rather than silently reported as "the forecast is the whole
    story" -- which is exactly how a domain sized to a 12 GB card came
    to die in preprocessing after the download.

    ``priced_only`` (the default, and the contract every existing caller
    was written against) folds "records a source this estimator cannot
    price" into the same ``None`` as "records no source at all".  Those
    are different facts and a REPORT must not conflate them: told only
    ``None``, :func:`unpriced_ingest_note` says "this config records no
    forcing product at all" to a reader looking at a ``[fetch]`` table
    they typed themselves.  Pass ``priced_only=False`` to get the
    recorded name back so the note can say which source it was.
    """
    import io
    import tomllib

    from gpuwm.config_authority import read_config_authority
    from gpuwm.experiment import is_experiment_toml_bytes

    path = Path(path)
    authority = read_config_authority(path)
    if not is_experiment_toml_bytes(authority.payload):
        return None
    raw = tomllib.load(io.BytesIO(authority.payload))
    table = raw.get("fetch")
    if isinstance(table, dict) and isinstance(table.get("source"), str):
        source = table["source"].strip().lower()
        if source in SOURCE_ANALYSIS_LEVELS or not priced_only:
            return source
        return None
    # The config-driven front door decodes ERA5 GRIB1; a [case_data]
    # table is that route by definition.
    return "era5" if "case_data" in raw else None


def unpriced_ingest_note(path: Path, source: str | None = None) -> str:
    """Why the preprocessing phase could not be priced for this config.

    Said in full rather than shortened to a footnote, because the number
    beside it is a FORECAST number and a reader who takes it for the
    whole run is making the exact mistake this phase estimate exists to
    prevent.
    """
    known = ", ".join(sorted(SOURCE_ANALYSIS_LEVELS))
    if source:
        why = (f"--source {source} ingests on a lane this estimator does "
               f"not model (priced sources: {known})")
    else:
        why = ("this config records no forcing product at all, so there "
               f"is no ingest lane to price (priced sources: {known})")
    return (f"preprocessing (ingest) NOT PRICED: {why}, so the envelope "
            "beside it covers the FORECAST only.  Preprocessing has its "
            "own peak and it is not always the smaller one.")


#: ``gpuwm check`` exit code for "every gate passed, but the observed peak
#: envelope exceeds the budget".  Nonzero, because the report says in prose
#: that the run may not fit and a script must be able to see that; distinct
#: from 1, because no gate failed and the levers are different.
_EXIT_ENVELOPE_OVER_BUDGET = 4


def _format_bytes(n: int | None) -> str:
    return "n/a" if n is None else f"{n / GIB:7.2f} GiB"


def _leg_text(value: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "not measured"}[value]


def _warn_unstaged_physics_tables(exp) -> None:
    """One line when a selected scheme's lookup tables are not staged.

    A WARNING, not a gate.  ``gpuwm check`` is the memory preflight and
    the page that calls it "Preflight"; a person who runs it before an
    mp8 case should hear that the tables are absent while the fix is
    still one command and no download has started, and the run doors
    (``--materialize-authorities`` and both prepared runners) are where
    the same condition is actually refused.  Sizing a domain whose
    tables are elsewhere is a legitimate thing to do, so nothing here
    changes an exit code.
    """

    try:
        if not any(int(domain.run.mp_physics) == 8
                   for domain in exp.domains):
            return
        from gpuwm.table_assets import require_thompson_tables

        require_thompson_tables()
    except FileNotFoundError as error:
        from gpuwm.explain import warn

        warn(str(error),
             why="The tables are read at load and validated byte for "
                 "byte, so a run cannot start without them.  This "
                 "preflight sizes memory and does not need them, which "
                 "is why it says so and continues.")
    except Exception:  # pragma: no cover - never let an advisory throw
        return


def live_device_local_memory_profile() -> DeviceLocalMemoryProfile | None:
    """This machine's own local-memory profile, or ``None``.

    The local-memory backing store is ``(frame - default stack) x SM
    count x threads per SM``, so it is a property of the DEVICE, and the
    reference profile in this module is a 170-SM RTX 5090 -- roughly
    2.2x the resident-thread capacity of a 76-SM 4080.  Charging every
    card the 5090's backing store is the same mistake as charging Linux
    the WDDM pool constants: an accounting term measured somewhere else.

    Read only when the free-VRAM figure is being MEASURED off this
    device, i.e. when the answer is about this machine.  A declared
    ``--budget-gib`` says "size for a machine that is not this one", and
    that machine's SM count is unknown, so it keeps the reference
    profile -- which over-prices rather than under-prices.
    """

    from gpuwm.local_gpu import no_local_gpu

    if no_local_gpu():
        # Same switch, same scope as the probe subprocess below: reading
        # the device's SM census is device contact, and a caller that
        # cannot measure prices against the reference profile, which
        # over-prices rather than under-prices.
        return None
    try:
        import cupy as cp

        return local_memory_profile_from_device(cp)
    except Exception:
        return None


#: What :func:`device_memory_probe_subprocess` runs in its short-lived
#: interpreter: both device questions -- the free/total VRAM the budget
#: subtracts from, and the local-memory profile the non-pool terms are
#: priced against -- answered in one process that then exits.
#:
#: TWO exit codes, not one.  Exit 3 is "a card could not be read"; exit
#: :data:`PROBE_EXIT_NO_RUNTIME` is "there is no CuPy here to read it
#: with", and the last stderr line names the module.  They were one code
#: until 2.3.3, and that is how `gpuwm go`'s memory gate came to swallow
#: a missing GPU runtime: the probe exited 3, the gate read "no card
#: here", declined to refuse on a card it could not see, and let the
#: chain fetch gigabytes for a run that could never start.  A gate that
#: hides the reason a run cannot begin is worse than no gate.
_DEVICE_MEMORY_PROBE_SOURCE = """\
import json
import subprocess
import sys

# SELF-CONTAINED ON PURPOSE.  This source runs in a bare interpreter to
# answer "is there a card, and what is it"; importing gpuwm here would
# make the answer depend on the very install the caller may be asking
# about, and it did: importing one helper from gpuwm.core.preflight
# turned the "no CuPy here" exit code into an ImportError traceback,
# which is the exact confusion PROBE_EXIT_NO_RUNTIME exists to end.

def _nvml_used_bytes():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return None


def _bare_context(before, after, stack_store):
    # Mirrors preflight.measured_bare_context_bytes; a reading below the
    # default-stack backing store the driver must have allocated is not a
    # measurement of this context, and neither is one above 4 GiB.
    if before is None or after is None:
        return None
    delta = int(after) - int(before)
    if delta < int(stack_store) or delta > 4 * 1024 ** 3:
        return None
    return delta


# BEFORE cupy: this interpreter has no CUDA context yet, so the NVML
# reading here is the card without us on it.  That is what makes the
# delta below THIS process's context cost rather than the whole card's
# residency, which on a WDDM desktop is mostly somebody else's.
_before = _nvml_used_bytes()
try:
    import cupy as cp
except ImportError as error:
    sys.stderr.write("no-runtime: %s\\n" % (getattr(error, "name", None)
                                            or error))
    sys.exit(4)
try:
    free, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    _stack = int(cp.cuda.runtime.deviceGetLimit(0))
    _capacity = (int(props["multiProcessorCount"])
                 * int(props["maxThreadsPerMultiProcessor"]))
    _used_now = _nvml_used_bytes()
    _bare = _bare_context(_before, _used_now, _stack * _capacity)
    # THE SMALLER OF THE TWO INSTRUMENTS, always.  On WDDM the display
    # driver can evict other processes' allocations, so cudaMemGetInfo
    # answers "free if everything else were paged out" -- measured
    # 2026-08-20 on an RTX 3080 with a loaded desktop, four consecutive
    # samples: memGetInfo said 9,097 MiB free while NVML said 3,375-3,405
    # MiB, a stable 5.7 GiB over-statement of a 10 GiB card.  A budget
    # built on the larger figure spends memory the run would have to
    # evict a desktop to get.  Same idiom as the device rail: an
    # ADDITIONAL ceiling, never a widening.  On Linux the two agree and
    # this is a no-op.
    # From the BEFORE reading -- the card without this probe's own
    # context on it.  The run's context is charged by the reserve, so
    # taking it out of free as well would bill it twice.
    _nvml_free = None if _before is None else max(0, int(total) - _before)
    _free = int(free) if _nvml_free is None else min(int(free), _nvml_free)
    payload = {
        "free_bytes": _free,
        "free_bytes_memgetinfo": int(free),
        "free_bytes_nvml": _nvml_free,
        "total_bytes": int(total),
        "profile": {
            "name": (name.decode() if isinstance(name, bytes)
                     else str(name)),
            "multiprocessor_count": int(props["multiProcessorCount"]),
            "max_threads_per_multiprocessor": int(
                props["maxThreadsPerMultiProcessor"]),
            "default_stack_limit_bytes": _stack,
            "bare_context_bytes": _bare,
        },
    }
except Exception:
    sys.exit(3)
print(json.dumps(payload))
"""

#: Long enough for a cold CuPy import plus context creation on a busy
#: box; a probe that cannot answer inside it reads as "no device", which
#: only ever under-promises (nothing refuses on a card it cannot see).
DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS = 120.0

#: The probe's exit code for "this interpreter has no CuPy at all",
#: distinct from the exit 3 that means "a card could not be read".
PROBE_EXIT_NO_RUNTIME = 4


def device_memory_probe_reason(*, run=None) -> str | None:
    """Why :func:`device_memory_probe_subprocess` has no numbers, or ``None``.

    ``None`` when the probe answered.  Otherwise one short phrase naming
    the CAUSE, so a caller can say something truer than "no card here"
    -- which is what the single-exit-code version forced every caller to
    say, including on a box whose only problem was an uninstalled
    runtime.
    """

    payload, reason = _device_memory_probe(run=run)
    return None if payload is not None else reason


def _device_memory_probe(*, run=None) -> tuple[dict | None, str | None]:
    """``(payload, reason)`` -- the probe result and, when absent, why."""

    import subprocess

    from gpuwm.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu

    # The documented never-open-the-local-device switch, consulted
    # BEFORE anything spawns.  The probe subprocess IS device contact --
    # a CUDA primary context, memGetInfo, deviceGetLimit -- and the
    # 2.5.0 upgrader walk proved this path never asked: the variable was
    # set for every step and `gpuwm go`'s memory gate still reported the
    # local card's free VRAM.  Under the switch there are no measured
    # numbers, on purpose; callers price the DECLARED budget and their
    # verdicts carry this reason so nobody mistakes "not read" for "not
    # there".
    if no_local_gpu():
        return None, (f"{NO_LOCAL_GPU_ENV} is set, so the local card was "
                      "not read")
    runner = subprocess.run if run is None else run
    try:
        completed = runner(
            [sys.executable, "-c", _DEVICE_MEMORY_PROBE_SOURCE],
            capture_output=True, text=True,
            timeout=DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"the probe subprocess did not run ({error})"
    if completed.returncode == PROBE_EXIT_NO_RUNTIME:
        return None, "the GPU runtime (CuPy) is not installed"
    if completed.returncode != 0:
        return None, "no CUDA device answered"
    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        return None, "the probe printed nothing"
    try:
        payload = json.loads(lines[-1])
    except ValueError:
        return None, "the probe printed something that is not its JSON"
    free = payload.get("free_bytes") if isinstance(payload, dict) else None
    if not isinstance(free, int) or isinstance(free, bool):
        return None, "the probe reported no free-memory figure"
    return payload, None


def device_memory_probe_subprocess(*, run=None) -> dict | None:
    """Free/total VRAM and this card's local-memory profile, measured in
    a SHORT-LIVED subprocess; ``None`` when no card answered.

    ``cudaMemGetInfo`` and ``cudaDeviceGetLimit`` cannot be asked
    without standing up a CUDA primary context -- the same fact that
    keeps them out of estimator mode (see
    :func:`device_physical_total_bytes`).  A process that asks them
    in-process therefore keeps that context, and its device memory, for
    the rest of its life.  ``gpuwm check`` can afford that: it exits on
    the next line.  The ``gpuwm go`` orchestrator cannot: after its
    memory gate it lives for the entire chain as the stage runner and
    progress printer, and the context it stood up to ask one question
    sat on the card for the whole run -- measured 0.486 GiB on the RTX
    5090 -- as a consumer no term of the budget it had just computed
    names.  Asked here, the context lives and dies inside the probe
    process and the caller never touches CUDA at all.

    The numbers are the same ones the in-process readers see (the probe
    runs ``sys.executable``, so it resolves the same CuPy), which is
    what keeps this gate and ``gpuwm check`` from disagreeing about one
    card.  ``run`` is the ``subprocess.run`` seam, for tests.

    The numbers only.  A caller that must distinguish "no card" from "no
    runtime" -- the memory gate does, because those two answers licence
    opposite behaviour -- asks :func:`device_memory_probe_reason`.
    """

    return _device_memory_probe(run=run)[0]


def profile_from_device_probe(payload) -> DeviceLocalMemoryProfile | None:
    """The probe payload's device-profile half, typed, or ``None``.

    ``None`` falls back exactly like :func:`live_device_local_memory_profile`
    returning ``None``: the callers price against the reference profile,
    which over-prices rather than under-prices.
    """

    profile = payload.get("profile") if isinstance(payload, dict) else None
    if not isinstance(profile, dict):
        return None
    bare = profile.get("bare_context_bytes")
    if isinstance(bare, bool) or not isinstance(bare, int) or bare <= 0:
        # Absent or unusable reads as UNMEASURED, never as zero: a probe
        # from an older build simply does not carry the field, and a
        # zero-byte CUDA context is not a thing this could mean.
        bare = None
    try:
        return DeviceLocalMemoryProfile(
            name=str(profile["name"]),
            multiprocessor_count=int(profile["multiprocessor_count"]),
            max_threads_per_multiprocessor=int(
                profile["max_threads_per_multiprocessor"]),
            default_stack_limit_bytes=int(
                profile["default_stack_limit_bytes"]),
            bare_context_bytes=bare,
        )
    except (KeyError, TypeError, ValueError):
        return None


#: How close a declared card size has to be to the local card's measured
#: total to BE the local card.  Capacities are reported in whole MiB and
#: converted through GiB floats on the way in, so an exact comparison
#: would fail on rounding alone; 0.5% is far tighter than the gap between
#: any two card tiers.
LOCAL_CARD_MATCH_TOLERANCE = 0.005


def declares_the_local_card(card_total_gib: float | None) -> bool:
    """Is ``--vram-gib`` naming the card that is in this machine?

    A declaration is normally a statement about hardware that is
    somewhere else, and that is priced against the conservative
    reference profile.  But the wizard declares the size of the card it
    just MEASURED when it hands the emitted config to ``gpuwm check``,
    and substituting a 170-SM reference under a 68-SM card there made
    the two doors disagree about one machine.

    False whenever the local card cannot be read: an unreadable card
    cannot be the one being described, and the reference profile is the
    safe answer.
    """

    if card_total_gib is None:
        return False
    total = device_physical_total_bytes()
    if not total:
        return False
    declared = float(card_total_gib) * GIB
    return abs(declared - total) <= LOCAL_CARD_MATCH_TOLERANCE * total


def check_main(args) -> int:
    """``gpuwm check CONFIG [--alloc]``: memory section of the preflight.

    Exit codes: 0 = every requested leg measured and PASSED; 1 = a leg
    FAILED; 2 = fail-closed -- the requested legs could not be evaluated
    (estimator mode with no budget, or an ``--alloc`` that produced no
    measurements); 3 = the ``--alloc`` run ABORTED before measuring
    (headroom precheck / OOM) -- the JSON/text report is still emitted
    with the estimate-side legs and the abort reason; 4 = every gate
    passed but the observed peak envelope EXCEEDS the budget, which the
    report warns about in prose.  4 is distinct from 1 because the gates
    genuinely passed and the levers differ, but it is nonzero because a
    report whose own text says the run may not fit must never read green
    to a script.  A harder verdict wins: 1, 2 and 3 outrank 4.
    """
    exp = _load_experiment_any(args.config)
    _warn_unstaged_physics_tables(exp)
    #: Whether the free-VRAM figure below was measured off the device or
    #: derived from a declared --budget-gib.  Printed, because the two
    #: must never wear the same label.
    free_source = "measured"
    #: Declared physical capacity of the card this preflight is sizing for
    #: (``--vram-gib``, which the wizard passes from its card tier).  A
    #: ceiling on the free figure, never a source of one.
    card_total_gib = getattr(args, "vram_gib", None)
    #: The capacity ceiling that actually bound the free figure, if any.
    capped_to = None
    # Read the card BEFORE anything in this process touches CUDA, so the
    # other-process residency the rail must respect is not inflated by our
    # own context (which the reserve already carries).
    rail_bytes = (None if args.rail_mib is None
                  else int(args.rail_mib) * 1024 ** 2)
    other_process_bytes = None if rail_bytes is None else (
        device_wide_used_bytes())
    chunk = (exp.column_chunk if args.column_chunk is None
             else args.column_chunk)
    #: The device the non-pool terms are priced against.  This machine's
    #: own, whenever the free figure is measured off it; the reference
    #: 5090 profile when a declared budget says the target is elsewhere.
    profile = (None if getattr(args, "budget_gib", None) is not None
               else live_device_local_memory_profile())
    if profile is None and declares_the_local_card(card_total_gib):
        # A DECLARED budget for THIS card.  ``--budget-gib`` alone means
        # "the caller states the budget", not "the caller is describing
        # another machine" -- and the wizard's own follow-up check is
        # exactly that case: it sizes the card it just measured, states
        # the budget it sized against, and used to have the reference
        # 5090 profile substituted underneath it.  The wizard then read
        # 68 SMs and the check it printed read 170, on one card, and the
        # emission failed its own verification (task 206).
        #
        # Recognised by CAPACITY: ``--vram-gib`` naming the same total
        # this machine's card reports IS this machine's card.  A
        # declaration for any other size keeps the conservative
        # reference profile, which is what sizing hardware you do not
        # have is supposed to get.
        profile = live_device_local_memory_profile()
    if profile is None:
        profile = card_local_memory_profile(card_total_gib)
    # The retention term is a fraction OF the estimate, so the estimate is
    # formed first.  It is pure arithmetic and the runners re-derive it from
    # the same inputs, so there is no second source of truth.
    reserve = ReservePolicy.n0_alloc(
        exp, profile=profile, estimate_bytes=estimate_experiment(
            exp, column_chunk=chunk,
            forcing_interval_seconds=args.forcing_interval_s,
            vram_gib=card_total_gib, profile=profile
        ).alloc_estimate_bytes)
    if args.reserve_gib is not None:
        # Flat controller-ratified reserve replacing the proposed stack.
        reserve = ReservePolicy.flat(int(args.reserve_gib * GIB))

    report = None
    abort = None
    if args.alloc:
        try:
            report = run_alloc_preflight(
                exp, column_chunk=chunk,
                forcing_interval_seconds=args.forcing_interval_s,
                reserve=reserve, profile=profile)
        except (PreflightHeadroomError, PreflightAllocError) as exc:
            abort = exc
        if report is not None:
            estimate = report.estimate
            measured_used = report.pool_used_peak_bytes
            free = report.free_before_bytes
            gates = report.gates
        else:
            # Aborted before measurement: still report the estimate side
            # (F1 fix) -- the measured legs stay None and can never pass.
            estimate = estimate_experiment(
                exp, column_chunk=chunk,
                forcing_interval_seconds=args.forcing_interval_s,
                vram_gib=card_total_gib, profile=profile)
            measured_used = None
            free = getattr(abort, "free_bytes", None)
            gates = evaluate_alloc_gates(
                measured_used_bytes=None,
                estimate_bytes=estimate.alloc_estimate_bytes,
                measured_free_bytes=free, reserve=reserve)
    else:
        estimate = estimate_experiment(
            exp, column_chunk=chunk,
            forcing_interval_seconds=args.forcing_interval_s,
            vram_gib=card_total_gib, profile=profile)
        measured_used = None
        free = None
        if args.budget_gib is not None:
            # CPU-mode DECLARED budget: the caller states the budget and
            # the reserve is added back to recover a notional free
            # figure.  It is arithmetic, not a measurement, and it must
            # never print under the same label as one -- the wizard's
            # inline check reported "measured free 19.31 GiB" on a
            # machine with 11.44 GiB free, which is exactly the kind of
            # number a user then trusts.
            free = int(args.budget_gib * GIB) + reserve.reserve_bytes
            free_source = "declared (--budget-gib)"
            # ...and, once a card is named, it is capped by THAT card --
            # the declared one, and only it.  The arithmetic above knows
            # the budget but not the capacity, so on its own it can and
            # did synthesise a free figure larger than the whole card.
            #
            # This box's own physical total is NOT a second ceiling here.
            # --budget-gib is explicitly a "size for a machine that is not
            # this one" flag, and clamping a declared 24 GiB target to a
            # 16 GB card under the desk is how the wizard's own inline
            # check came to refuse a config it had just sized correctly:
            # the check reported a budget of 11.14 GiB for a 24 GiB
            # target, which is this machine's number, not the target's.
            if card_total_gib is not None:
                free, capped_to = cap_free_to_physical(
                    free, card_total_bytes=int(card_total_gib * GIB),
                    measured_total_bytes=None)
            if capped_to is not None:
                free_source = ("declared (--budget-gib), capped at the "
                               "card's physical total")
        else:
            try:
                import cupy as cp
                free = int(cp.cuda.runtime.memGetInfo()[0])
                free_source = "measured"
                # ...and never MORE than the card actually has free.
                # ``memGetInfo`` answers "free if every other process
                # were evicted", which under WDDM it can be: measured
                # 2026-08-20 on a loaded RTX 3080 desktop, four
                # consecutive samples, memGetInfo said 9,097 MiB free
                # while NVML said 3,375-3,405 -- 5.7 GiB of a 10 GiB
                # card.  Spending that is spending a desktop's memory.
                total_now = device_physical_total_bytes()
                used_now = device_wide_used_bytes()
                if total_now and used_now is not None:
                    nvml_free = max(0, int(total_now) - int(used_now))
                    if nvml_free < free:
                        free = nvml_free
                        free_source = ("measured machine-wide (the CUDA "
                                       "runtime reported more, counting "
                                       "memory the driver would have to "
                                       "evict from other processes)")
            except Exception:
                free = None
            if free is not None and card_total_gib is not None:
                # Sizing for a SMALLER card than the one under the desk is
                # a legitimate use of --vram-gib, and it must not inherit
                # this box's headroom.
                free, capped_to = cap_free_to_physical(
                    free, card_total_bytes=int(card_total_gib * GIB),
                    measured_total_bytes=None)
                if capped_to is not None:
                    free_source = ("measured, capped at the declared "
                                   "card's physical total")
        gates = evaluate_alloc_gates(
            measured_used_bytes=None,
            estimate_bytes=estimate.alloc_estimate_bytes,
            measured_free_bytes=free, reserve=reserve)

    # The whole-machine rail, when one is configured, is an ADDITIONAL
    # ceiling on top of measured free VRAM -- never a replacement and never a
    # widening.  ``free`` becomes the smaller of what the driver will hand
    # out and what the rail leaves after every other process on the card.
    rail = None
    if rail_bytes is not None:
        rail_free = device_rail_free_bytes(
            rail_bytes, other_process_bytes=other_process_bytes)
        rail = {"rail_bytes": rail_bytes,
                "other_process_bytes": other_process_bytes,
                "rail_free_bytes": rail_free}
        free = rail_free if free is None else min(int(free), rail_free)
        gates = evaluate_alloc_gates(
            measured_used_bytes=measured_used,
            estimate_bytes=estimate.alloc_estimate_bytes,
            measured_free_bytes=free, reserve=reserve)

    budget = None if free is None else reserve.budget_bytes(free)
    #: A reserve larger than free VRAM leaves NO budget, not a negative
    #: capacity.  ``budget = free - reserve`` is unbounded below, and a
    #: 4000x4000 config drove it to -7.15 GiB, which the report then
    #: printed as a figure to compare an envelope against.  Clamp at
    #: zero and say what happened, in the report, once.
    budget_underwater_bytes = 0
    if budget is not None and budget < 0:
        budget_underwater_bytes = -budget
        budget = 0
    forecast_envelope = estimate.peak_envelope_bytes
    # PREPROCESSING IS A PHASE TOO.  Reporting only the forecast is what
    # let a 12 GB-sized domain download 81 GFS files and then OOM in
    # ingest at 15.82 GB.  A config whose source this estimator cannot
    # price says so; it never silently reports the forecast as the peak.
    #: What the config RECORDS, priceable or not.  A report that is about
    #: to tell the reader their ingest phase is unpriced has to name the
    #: source it could not price; folding it into the same ``None`` as
    #: "no [fetch] table at all" printed "this config records no forcing
    #: product at all" at a reader looking at the one they wrote.
    #: :func:`estimate_phases` keys off the same table either way, so an
    #: unpriceable name still leaves the ingest term absent.
    ingest_source = config_forcing_source(args.config, priced_only=False)
    from gpuwm.core.streaming import planner_machine

    phases = estimate_phases(
        exp, source=ingest_source, column_chunk=chunk,
        forcing_interval_seconds=args.forcing_interval_s,
        vram_gib=card_total_gib, profile=profile,
        # THE CARD THIS REPORT IS ABOUT, and not the one printing it.
        # ``mode = "auto"`` with no pinned tiling is the planner's
        # decision, and asked with no Machine the planner reaches for
        # ``Machine.detect`` -- which reads whatever card is under the
        # desk, or fails outright with no CuPy and silently leaves the
        # RESIDENT estimate standing.  ``free`` here is either measured
        # off this card or derived from a ``--budget-gib`` naming a card
        # that is not in this machine at all; both are the card the
        # reader asked about, and neither costs this process a CUDA
        # context.  ``gpuwm go``'s gate builds the same Machine from its
        # out-of-process probe, through the same function.
        machine=planner_machine(vram_bytes=free,
                                name="gpuwm check budget"))
    #: AN UNPRICED INGEST LANE COSTS THE INGEST SECTION, NOT THE PHASE
    #: ESTIMATE.  This used to be ``phases = None``, which threw away the
    #: streamed forecast term along with the ingest one -- and the streamed
    #: term has nothing to do with the source.  Every ``[tiles]`` config
    #: forced to a source outside :data:`SOURCE_ANALYSIS_LEVELS` was then
    #: reported, and refused, on a resident envelope describing a run that
    #: was not going to happen: measured on a 550x550x49 config with 200x200
    #: tiles, 14.30 GiB reported where the run holds 6.71 GiB, exit 4 from
    #: the envelope and exit 1 from the alloc gate.  The section's ABSENCE
    #: is what gets reported now (:func:`unpriced_ingest_note`), which is
    #: the gap that is real.
    ingest_priced = phases.ingest_priced
    #: The envelope every verdict below compares: the largest phase, not
    #: whichever phase happens to be modelled.
    envelope = phases.peak_envelope_bytes
    binding_phase = phases.binding_phase
    #: What the ENVELOPE is compared against.  NOT ``budget``: that is
    #: the allocation gate's budget and it has already subtracted the
    #: CUDA context and the local-memory backing store, which
    #: ``peak_envelope_bytes`` carries as its intercept.  Comparing the
    #: two charged one process for its own non-pool residency twice --
    #: 2.91 GiB of a 10 GiB card, on the walk that opened task 206 -- and
    #: warned that a configuration would not fit a card it fits.
    #:
    #: The envelope models the whole device residency this process
    #: reaches, so what is left outside it is other processes, which is
    #: :data:`EXTERNAL_MARGIN_BYTES`.  Same seam the wizard sizes with
    #: (:func:`gpuwm.domain_wizard.sizing_budget_bytes`), so the door
    #: that emits a config and the door that verifies it cannot disagree.
    envelope_budget = (None if free is None
                       else max(0, int(free) - EXTERNAL_MARGIN_BYTES))
    #: The report's own prose says this configuration may not fit.  It is
    #: read here, before either renderer, because the exit code has to
    #: carry it whether or not anybody reads the text.
    envelope_over_budget = (envelope_budget is not None
                            and envelope > envelope_budget)
    #: THE ALLOC GATE PRICES THE RUN THE CONFIG ASKS FOR.
    #:
    #: Every leg above was fed ``estimate.alloc_estimate_bytes``, which
    #: itemizes a domain RESIDENT in VRAM.  Under ``[tiles]`` that domain
    #: is never allocated -- the card holds ``nbuffers`` tile buffers and
    #: the domain lives in pinned host RAM -- so the leg was refusing, at
    #: exit 1, the one configuration class streaming exists to enable.
    #: Measured on the 550x550x49 / 200x200 fixture: 11.35 GiB of resident
    #: pool request weighed against a budget the 6.71 GiB streamed run fits
    #: with room to spare.
    #:
    #: COMPARED AS AN ENVELOPE, AGAINST THE ENVELOPE BUDGET, and that pair
    #: is the decision rather than a convenience.
    #: :attr:`StreamedEnvelope.vram_bytes` is envelope-shaped by
    #: construction -- CUDA context, the rung's per-process fixed cost and
    #: the tile buffers, with autoplan's safety factor -- and there is no
    #: alloc-shaped figure to be had from it.  ``Footprint`` lumps the
    #: pool-resident k-distribution tables in with the non-pool module
    #: images inside one measured ``process_fixed_bytes``, so any
    #: subtraction that made it alloc-shaped would be inventing a number.
    #: Both candidate subtractions were priced on this fixture and both
    #: are wrong in a way that matters: taking off the whole
    #: ``process_overhead_bytes`` leaves 2.95 GiB against a run that really
    #: holds 6.71, an UNDER-charge of 1.31 GiB that would have the gate
    #: admit an OOM; taking off only the CUDA context leaves 6.29 GiB and
    #: charges the 2.03 GiB local-memory backing store a second time, on
    #: top of the reserve that already holds it -- the task-206 double
    #: count this file spent a release removing.
    #:
    #: So the streamed figure is compared against ``envelope_budget``
    #: (free VRAM less the other-process margin), which is the same pair
    #: ``gpuwm go``'s memory gate admits on and the same pair the
    #: over-budget WARNING below prints.  The leg's contract --
    #: "the estimate fits the budget" -- is unchanged; what it prices is.
    #:
    #: ``--alloc`` is excluded on purpose: ``run_alloc_preflight``
    #: constructs the RESIDENT domain on the real card, so its measured
    #: legs describe a resident allocation, and swapping the estimate leg
    #: underneath them would have ``alloc_measured_le_estimate`` compare a
    #: resident measurement against a streamed estimate and fail for a
    #: reason that is not about this configuration at all.
    streamed_alloc_gate = (phases.streamed_forecast and not args.alloc
                           and envelope_budget is not None)
    if streamed_alloc_gate:
        gates = dict(gates)
        # THE RADIATION PEAK, not the steady hold: the leg admits a run,
        # and a run admitted on the hold meets the RRTMGP transient at
        # itimestep == 1.  See StreamedEnvelope.peak_vram_bytes.
        gates["alloc_estimate_le_wddm_budget"] = (
            int(phases.streamed.peak_vram_bytes) <= envelope_budget)
    if args.json:
        payload = {
            "config": str(args.config), "experiment": exp.name,
            "column_chunk": estimate.column_chunk,
            "domains": {
                f"d{d.grid_id:02d}": {
                    "resident_bytes": d.resident_bytes,
                    "arena_scratch_request_bytes": d.arena_scratch_bytes,
                    "transient_bytes": d.transient_bytes,
                    "by_category": {c: d.category_bytes(c) for c in
                                    ("state", "physics", "scratch", "lbc",
                                     "nest", "sase", "transient")},
                } for d in estimate.domains},
            "k_tables_bytes": estimate.k_tables_bytes,
            "workspace_bytes": estimate.workspace_bytes,
            "scratch_arena_bytes": estimate.scratch_arena_bytes,
            "scratch_arena_request_bytes":
                estimate.scratch_arena_request_bytes,
            "scratch_arena_saved_bytes": estimate.scratch_arena_saved_bytes,
            "uses_shared_scratch_arena":
                estimate.uses_shared_scratch_arena,
            "dycore_state_workspace_bytes":
                estimate.dycore_state_workspace_bytes,
            "dycore_state_request_bytes":
                estimate.dycore_state_request_bytes,
            "dycore_state_saved_bytes": estimate.dycore_state_saved_bytes,
            "uses_shared_dycore_state_workspace":
                estimate.uses_shared_dycore_state_workspace,
            "resident_bytes": estimate.resident_bytes,
            "transient_peak_bytes": estimate.transient_peak_bytes,
            "subtotal_bytes": estimate.subtotal_bytes,
            "alloc_estimate_bytes": estimate.alloc_estimate_bytes,
            "held_projection_bytes": estimate.held_projection_bytes,
            "footprint_projection_bytes":
                estimate.footprint_projection_bytes,
            "observed_peak_envelope_platform":
                envelope_platform(vram_gib=card_total_gib),
            "observed_peak_envelope_bytes": forecast_envelope,
            "non_pool_device_bytes": estimate.non_pool_device_bytes,
            "envelope_unmodelled_bytes": ENVELOPE_UNMODELLED_BYTES,
            "envelope_per_nest_fraction": ENVELOPE_PER_NEST_FRACTION,
            # Keyed by RADIATION LANE since 2026-08-20, not by driver
            # model.  The old key name is kept because 2.5.0 receipts
            # read it; ``envelope_pool_slack_fraction`` is its name now.
            "envelope_wddm_pool_slack_fraction": (
                POOL_SLACK_FRACTION
                if estimate.uses_legacy_radiation else 0.0),
            "envelope_pool_slack_fraction": (
                POOL_SLACK_FRACTION
                if estimate.uses_legacy_radiation else 0.0),
            "envelope_legacy_radiation": estimate.uses_legacy_radiation,
            "envelope_basis": estimate.envelope_basis,
            "local_memory_profile": estimate.non_pool_device_bytes and
                profile.name,
            "peak_envelope_bytes": envelope,
            "binding_phase": binding_phase,
            "reserve_bytes": reserve.reserve_bytes,
            "reserve_components": {
                "retention_residual_bytes":
                    reserve.retention_residual_bytes,
                "device_overhead_bytes": reserve.device_overhead_bytes,
                "external_margin_bytes": reserve.external_margin_bytes,
            },
            "run_time_reserve_bytes": ReservePolicy.run_time(
                exp).reserve_bytes,
            "pool_reserved_over_estimate_fraction":
                POOL_RESERVED_OVER_ESTIMATE_FRACTION,
            "cuda_context_bytes": CUDA_CONTEXT_BYTES,
            # PRICED AGAINST THE DEVICE THIS REPORT IS ABOUT.  Without
            # the profile this defaulted to the 170-SM reference while
            # the sibling "local_memory_profile" field named the real
            # card, so the two disagreed: a 70-SM RTX 5070 Ti reported
            # 5.20 GiB where its own profile gives 2.14, and the
            # difference propagated into the envelope as a spurious
            # over-budget warning.
            "kernel_local_memory_bytes": kernel_local_memory_bytes(
                exp, profile=profile),
            "kernel_modules": sorted(physics_kernel_modules(exp)),
            "measured_free_bytes": free,
            "free_bytes_source": free_source,
            # A declared budget sizes hardware that is not in this
            # machine; every figure in this report is then an ESTIMATE
            # for hardware not present, priced against the conservative
            # measured reference profile above -- never a measurement.
            "sized_for_hardware_not_present": args.budget_gib is not None,
            "free_bytes_capped_to_physical_bytes": capped_to,
            "budget_bytes": budget,
            "budget_underwater_bytes": budget_underwater_bytes,
            "envelope_budget_bytes": envelope_budget,
            "observed_peak_envelope_exceeds_budget": (
                None if envelope_budget is None else envelope_over_budget),
            "gates": gates,
        }
        # WHICH FORECAST FIGURE THE READER GOT, said in a field rather
        # than inferred from the size of the number.
        payload["streamed_forecast"] = phases.streamed_forecast
        if phases.streamed_forecast:
            env = phases.streamed
            payload["streamed"] = {
                # THE HOLD, kept because it is the figure an NVML
                # steady-state reading can be compared against...
                "vram_bytes": int(env.vram_bytes),
                # ...and the two terms that make the PEAK, which is what
                # peak_envelope_bytes above carries and every gate weighs.
                "radiation_transient_bytes":
                    int(env.radiation_transient_bytes),
                "peak_vram_bytes": int(env.peak_vram_bytes),
                "resident_forecast_envelope_bytes":
                    phases.resident_forecast_envelope_bytes,
                "host_bytes": int(env.host_bytes),
                "host_budget_bytes": env.host_budget_bytes,
                "tile_nx": env.tile_nx, "tile_ny": env.tile_ny,
                "window_nx": env.window_nx, "window_ny": env.window_ny,
                "nbuffers": env.nbuffers, "halo": env.halo,
                "rung": env.rung, "write_mode": env.write_mode,
                # The pair the alloc leg above compared, named, so a
                # script never has to guess which budget it was.
                "alloc_gate_basis": (
                    "the streamed envelope against envelope_budget_bytes"
                    if streamed_alloc_gate else
                    "resident (--alloc measures a resident allocation)"),
            }
        # The verdict compares ``envelope_budget``, not ``budget``: it is
        # an ENVELOPE sentence, and the allocation budget has already
        # subtracted the CUDA context and the local-memory backing store
        # that the envelope carries as its intercept.  This field said
        # otherwise while the printed line beside it said this, which is
        # one report giving two answers about one card (task 206).
        payload["phase_verdict"] = phases.verdict(envelope_budget)
        if not ingest_priced:
            payload["ingest"] = None
            payload["ingest_not_priced_reason"] = unpriced_ingest_note(
                args.config, ingest_source)
        else:
            ingest = phases.ingest
            payload["ingest"] = {
                "source": ingest_source,
                "forcing_times": ingest.n_forcing_times,
                "resident_forcing_times": ingest.resident_times,
                "per_forcing_time_bytes": ingest.per_time_bytes,
                "analysis_bytes": ingest.category_bytes("analysis"),
                "state_bytes": ingest.category_bytes("state"),
                "forcing_table_bytes": ingest.forcing_table_bytes,
                "resident_bytes": ingest.resident_bytes,
                "nest_state_bytes": ingest.nest_state_bytes,
                "nest_state_by_grid": {
                    f"d{grid:02d}": nbytes
                    for grid, nbytes in ingest.nest_state_items},
                "widest_domain_time_bytes":
                    ingest.widest_domain_time_bytes,
                "transient_bytes": ingest.transient_bytes,
                "subtotal_bytes": ingest.subtotal_bytes,
                "unstreamed_resident_bytes":
                    ingest.unstreamed_resident_bytes,
                "boundary_frame_host_bytes": ingest.boundary_frame_bytes,
                "alloc_estimate_bytes": ingest.alloc_estimate_bytes,
                "peak_envelope_bytes": ingest.peak_envelope_bytes,
                "context_bytes": ingest.context_bytes,
                "peak_envelope_basis": INGEST_PEAK_ENVELOPE_BASIS,
            }
        # The same document ``run-plan --estimate`` publishes, on the
        # surface that has actually measured the card: a machine-facing
        # reader of `gpuwm check --json` gets the pace as fields rather
        # than having to parse it back out of the advisory sentence.
        pace = pace_estimate_for_report(exp, streamed=phases.streamed,
                                        free_bytes=free)
        payload["expected_pace"] = None if pace is None else pace.to_json()
        advisories = check_advisories(exp, args.config,
                                      streamed=phases.streamed)
        if advisories:
            payload["advisories"] = advisories
        if rail is not None:
            payload["device_rail"] = rail
        if abort is not None:
            payload["abort"] = {
                "error": type(abort).__name__,
                "phase": abort.phase,
                "reason": str(abort),
                "free_bytes": getattr(abort, "free_bytes", None),
            }
        if report is not None:
            payload["alloc"] = {
                "pool_used_peak_bytes": report.pool_used_peak_bytes,
                "pool_held_peak_bytes": report.pool_held_peak_bytes,
                "device_footprint_bytes": report.device_footprint_bytes,
                "measured_overhead_bytes": report.measured_overhead_bytes,
                "free_after_release_bytes":
                    report.free_after_release_bytes,
            }
        print(json.dumps(payload, indent=2))
    else:
        print(f"gpuwm check: memory preflight for {exp.name!r} "
              f"({len(exp.domains)} domain(s); column_chunk "
              f"{estimate.column_chunk})")
        for advisory in check_advisories(
                exp, args.config, streamed=phases.streamed):
            print(f"  {advisory}")
        # UNCONDITIONAL, unlike the advisories above.  This is the line
        # whose absence let a streamed run at 399,119 columns look like a
        # stall: the memory report was complete and said nothing about
        # what a step costs.  Priced against the card THIS REPORT just
        # measured, so the column bound is about the reader's own
        # allowance and not a declared one.
        pace_line = pace_advisory(
            exp, streamed=phases.streamed,
            machine=pace_machine_from_free_bytes(free))
        if pace_line:
            print(f"  {pace_line}")
        for d in estimate.domains:
            cats = ", ".join(
                f"{c} {d.category_bytes(c) / GIB:.3f}"
                for c in ("state", "physics", "scratch", "lbc", "nest",
                          "sase")
                if d.category_bytes(c))
            print(f"  d{d.grid_id:02d}: resident "
                  f"{d.resident_bytes / GIB:6.2f} GiB ({cats}); step "
                  f"transients {d.transient_bytes / GIB:.2f} GiB")
        print(f"  shared: k-tables {estimate.k_tables_bytes / GIB:.3f} GiB; "
              f"chunk workspace {estimate.workspace_bytes / GIB:.2f} GiB; "
              f"scratch arena {estimate.scratch_arena_bytes / GIB:.2f} GiB "
              f"(saves {estimate.scratch_arena_saved_bytes / GIB:.2f} GiB); "
              f"dycore state {estimate.dycore_state_workspace_bytes / GIB:.2f} "
              f"GiB (saves {estimate.dycore_state_saved_bytes / GIB:.2f} GiB)")
        print(f"  TIER 1  resident: {_format_bytes(estimate.resident_bytes)}"
              f"   subtotal (+workspace+transient peak): "
              f"{_format_bytes(estimate.subtotal_bytes)}")
        print(f"  ESTIMATE (x{estimate.headroom:.2f} headroom): "
              f"{_format_bytes(estimate.alloc_estimate_bytes)}")
        print(f"  TIER 2  pool-held projection: "
              f"{_format_bytes(estimate.held_projection_bytes)}"
              f"   TIER 3 device-footprint projection: "
              f"{_format_bytes(estimate.footprint_projection_bytes)}")
        # Same profile the estimate used (see the JSON field above):
        # this feeds both the printed backing-store figure and the
        # re-measured footprint projection, so pricing it against a
        # card that is not in the machine inflates the envelope.
        widest = kernel_local_memory_bytes(exp, profile=profile)
        gf_ws = gf_column_workspace_bytes(exp, profile=profile)
        kf_ws = kf_column_workspace_bytes(exp, profile=profile)
        ysu_ws = ysu_column_workspace_bytes(exp, profile=profile)
        # Kept out of the f-string: a line break inside an f-string
        # expression is PEP 701 (Python 3.12+) syntax, and the supported
        # floor is 3.11 -- the 1.2.0 release workflow failed on exactly
        # this line before any wheel was published.
        # THIS CARD's context, not the retired flat constant: the two
        # numbers used to disagree on the same page, because this display
        # line kept CUDA_CONTEXT_BYTES while the envelope beside it moved
        # to the per-card term (task 206).
        context_bytes = profile.cuda_context_bytes
        remeasured_bytes = (estimate.alloc_estimate_bytes + widest
                            + gf_ws + kf_ws + ysu_ws
                            + context_bytes
                            + reserve.retention_residual_bytes)
        gf_ws_term = (f" + GF column workspace {_format_bytes(gf_ws)}"
                      if gf_ws else "")
        gf_ws_term += (f" + KF column workspace {_format_bytes(kf_ws)}"
                       if kf_ws else "")
        ysu_ws_term = (f" + YSU column workspace {_format_bytes(ysu_ws)}"
                       if ysu_ws else "")
        print(f"  NON-POOL: CUDA context "
              f"{_format_bytes(context_bytes)} + local-memory backing "
              f"store {_format_bytes(widest)} "
              f"({len(physics_kernel_modules(exp))} kernel modules selected)"
              f"{gf_ws_term}"
              f"{ysu_ws_term}"
              f"; RE-MEASURED device-footprint projection "
              f"{_format_bytes(remeasured_bytes)}"
              " (the TIER 3 line above is the retired zero-step probe model)")
        print(f"  NON-POOL BASIS: {non_pool_basis(profile)}")
        family = envelope_platform(vram_gib=card_total_gib)
        if family == "windows":
            provenance = (
                "affine, calibrated on this card class: six whole "
                "bare-default forecasts on an RTX 3080 10 GiB Windows/WDDM "
                "desktop measured machine-wide peaks of estimate + "
                "itemized non-pool within -0.20..+0.95 GiB, so the "
                "envelope is that sum plus the measured WDDM pool-slack "
                "term -- the retired 1.75 multiplier predicted 3.8x the "
                "measured peak on the same card and is gone from every "
                "gate")
        else:
            provenance = (
                "affine, not a multiplier: a multiplier with no intercept "
                "under-predicts small configurations and over-predicts "
                "large ones, which is what the 16 GiB fleet node measured "
                "(3.99 GiB declared, 4.38 measured; 19.95 declared, 13.88 "
                "measured).  The intercept is the non-pool line above, "
                "which scales with the device and the kernel set, not "
                "with the grid")
        # THE ITEMIZATION IS RESIDENT, AND SAYS SO UNDER [tiles].  Every
        # term above enumerates a domain living in VRAM; with streaming
        # configured the streamed term below replaces this figure, and a
        # reader who takes this line for the answer has the wrong number
        # by the whole point of the feature.  The advisory at the top of
        # this report promises its numbers price the STREAMED allocation:
        # the label is what keeps that promise true of this line too.
        forecast_label = ("RESIDENT FORECAST PEAK ENVELOPE (replaced by the "
                          "streamed term below)"
                          if phases.streamed_forecast
                          else "FORECAST PEAK ENVELOPE")
        print(f"  {forecast_label} ({estimate.peak_envelope_terms()}"
              f"; {estimate.envelope_basis}): "
              f"{_format_bytes(forecast_envelope)} -- {provenance}.")
        if not ingest_priced:
            print("  " + unpriced_ingest_note(args.config, ingest_source))
        else:
            ingest = phases.ingest
            print(f"  INGEST (preprocessing, --source {ingest_source}): "
                  f"root {ingest.n_forcing_times} forcing times x "
                  f"{_format_bytes(ingest.per_time_bytes)} each "
                  f"(analysis {_format_bytes(ingest.category_bytes('analysis'))}"
                  f" + state {_format_bytes(ingest.category_bytes('state'))}); "
                  f"{ingest.resident_times} resident at a time = "
                  f"{_format_bytes(ingest.resident_bytes)} resident")
            if ingest.nest_state_items:
                nested = ", ".join(
                    f"d{grid:02d} {nbytes / GIB:.2f}"
                    for grid, nbytes in ingest.nest_state_items)
                print(f"    + NESTS ({len(ingest.nest_state_items)} of them, "
                      f"one initial state each, all resident for the single "
                      f"export transaction): {nested} = "
                      f"{_format_bytes(ingest.nest_state_bytes)}")
            print(f"    INGEST OBSERVED PEAK ENVELOPE "
                  f"(x{ingest.headroom:.2f} headroom + "
                  f"{_format_bytes(ingest.context_bytes)} CUDA context; "
                  f"{INGEST_PEAK_ENVELOPE_BASIS}): "
                  f"{_format_bytes(ingest.peak_envelope_bytes)}   "
                  f"[streaming; holding all "
                  f"{ingest.n_forcing_times} times would resident "
                  f"{_format_bytes(ingest.unstreamed_resident_bytes)}]")
        # ``envelope_budget``, NOT ``budget``.  This one printed line
        # was the last place the task-206 double count survived: it
        # compared the machine-peak ENVELOPE -- which carries the
        # CUDA context and the local-memory backing store as its
        # intercept -- against the ALLOCATION budget, from which the
        # allocation reserve has already subtracted those same
        # bytes.  Measured 2026-08-20 on the loaded RTX 3080, 5.75
        # GiB free: it printed "5.40 GiB peak envelope; that EXCEEDS
        # the 3.79 GiB budget by 1.61 GiB" while the exit code (read
        # off ``envelope_over_budget``, which uses the line below)
        # said 0.15 GiB, and `gpuwm go` on the same config seconds
        # later admitted it and ran to rc 0 with output byte-
        # identical to the roomy run.  A report whose sentence and
        # whose exit code disagree by 1.46 GiB of budget teaches the
        # reader to ignore one of them.
        #
        # PRINTED WHETHER OR NOT THE INGEST LANE COULD BE PRICED.  It
        # used to live inside the priced branch, so the one line that
        # names the STREAMED forecast term -- and the tiling that
        # produced it -- was withheld from exactly the configs whose
        # envelope the streaming replaced.
        print(f"  BINDING PHASE: {phases.verdict(envelope_budget)}.")
        if streamed_alloc_gate:
            # The gate leg below prices this run as it will actually be
            # allocated, and it compares a different budget from the one
            # on the reserve line: say so where the two are read, so
            # neither number reads as contradicting the other.
            print(f"  ALLOC GATE, STREAMED: the leg below weighs the "
                  f"streamed envelope "
                  f"{_format_bytes(int(phases.streamed.peak_vram_bytes))} "
                  f"against "
                  f"the envelope budget {_format_bytes(envelope_budget)} "
                  f"(free VRAM less the "
                  f"{_format_bytes(EXTERNAL_MARGIN_BYTES)} other-process "
                  f"margin), not the itemized resident estimate against the "
                  f"allocation budget: with [tiles] the resident domain is "
                  f"never allocated, and the streamed figure already "
                  f"carries the CUDA context and the per-process fixed "
                  f"cost that the allocation reserve holds separately.")
        print(f"  reserve {reserve.reserve_bytes / GIB:.2f} GiB "
              f"(retention {reserve.retention_residual_bytes / GIB:.2f} + "
              f"overhead {reserve.device_overhead_bytes / GIB:.2f} + "
              f"external {reserve.external_margin_bytes / GIB:.2f}); "
              f"{free_source} free {_format_bytes(free)}; budget "
              f"{_format_bytes(budget)}")
        if args.budget_gib is not None:
            # The 4090 stress run certified "fits with 0.27 GiB to
            # spare" off this path and the config landed 0.015 GiB from
            # the budget on real hardware.  A declared budget is sizing
            # a card that is not in this machine, and the report has to
            # say so beside the verdict, not leave "fits" to read as a
            # measurement.
            if declares_the_local_card(card_total_gib):
                # Declared, but declared about THIS card -- which is what
                # the wizard's own follow-up check does.  Saying "hardware
                # not present" about the card in the machine, and pricing
                # it on a reference profile to match, is how one box got
                # two answers for one card.
                print(f"  DECLARED BUDGET, MEASURED CARD: the free figure "
                      f"above is declared rather than sampled, so it does "
                      f"not move with what else is on the card right now; "
                      f"the card itself is this machine's and its "
                      f"grid-independent terms are "
                      f"{non_pool_basis(profile)}.")
            else:
                print(f"  ESTIMATE FOR HARDWARE NOT PRESENT: the free "
                      f"figure above is declared, not measured -- this "
                      f"preflight is sizing a card that is not in this "
                      f"machine.  Non-pool terms are priced against the "
                      f"conservative measured reference device profile "
                      f"({profile.name}, "
                      f"{profile.multiprocessor_count} SMs), the largest "
                      f"known-device intercept, so the estimate is never "
                      f"more optimistic than a present-card measurement; "
                      f"verify with `gpuwm check` on the real card before "
                      f"trusting the margin.")
        if budget_underwater_bytes:
            print(f"  NO BUDGET AT ALL: the reserve alone is "
                  f"{_format_bytes(reserve.reserve_bytes)} against "
                  f"{_format_bytes(free)} free, so it exceeds the card by "
                  f"{_format_bytes(budget_underwater_bytes)} before this "
                  f"configuration asks for a single byte.  The budget "
                  f"above is clamped to zero: a negative capacity is not "
                  f"a number anything can be compared against.  The "
                  f"reserve's retention term scales with the "
                  f"configuration, so a smaller one is the lever.")
        if capped_to is not None:
            print(f"  CAPPED: the declared free figure exceeded the card's "
                  f"physical total and was clamped to "
                  f"{_format_bytes(capped_to)}; free VRAM cannot exceed "
                  f"the card")
        if rail is not None:
            print(f"  DEVICE RAIL {_format_bytes(rail['rail_bytes'])} "
                  f"whole-machine; other processes hold "
                  f"{_format_bytes(rail['other_process_bytes'])}, leaving "
                  f"{_format_bytes(rail['rail_free_bytes'])} for this run")
        if report is not None:
            print(f"  --alloc measured: pool used peak "
                  f"{_format_bytes(report.pool_used_peak_bytes)}; held "
                  f"{_format_bytes(report.pool_held_peak_bytes)}; device "
                  f"footprint {_format_bytes(report.device_footprint_bytes)}"
                  f" (non-pool overhead re-calibration "
                  f"{_format_bytes(report.measured_overhead_bytes)}); free "
                  f"after release "
                  f"{_format_bytes(report.free_after_release_bytes)}")
        if abort is not None:
            print(f"  ABORTED before measurement ({type(abort).__name__} "
                  f"during {abort.phase}): {abort}")
        for metric in N0_GATE_METRICS:
            print(f"  {gate_display_name(metric, vram_gib=card_total_gib)}: "
                  f"{_leg_text(gates[metric])}")
        if envelope_over_budget:
            if binding_phase != "forecast":
                print(f"  WARNING: the binding phase here is "
                      f"{binding_phase}, not the forecast; the envelope "
                      "named below is that phase's.")
            # The exit code this block ANNOUNCES has to be the one the
            # process will really return.  It used to assert "(exit code
            # 4: gates passed)" unconditionally -- including when a gate
            # had just failed and the process therefore exited 1, which
            # is a printed contract a script can be written against and
            # then mis-handle.  Read the gates here, once.
            gate_failed = any(leg is False for leg in gates.values())
            unevaluable = not [leg for leg in gates.values()
                               if leg is not None]
            if gate_failed:
                code_note = ("exit code 1: a gate above FAILED as well, "
                             "and the harder verdict wins")
            elif unevaluable:
                code_note = ("exit code 2: no gate above could be "
                             "evaluated, which fails closed")
            else:
                code_note = (f"exit code {_EXIT_ENVELOPE_OVER_BUDGET}: "
                             "gates passed, envelope did not")
            budget_word = ("WDDM budget" if envelope_platform(
                vram_gib=card_total_gib) == "windows" else "budget")
            # The tail sentence exists to explain a gate that PASSES
            # beside an envelope that does not: the resident legs weigh
            # the itemized pool request, which is a smaller thing than
            # the envelope.  Under [tiles] the alloc leg weighs this very
            # envelope against this very budget, so that explanation
            # would be describing a disagreement that cannot happen.
            if streamed_alloc_gate:
                tail = (" The alloc leg above weighs the same streamed "
                        "envelope against the same budget, so it does not "
                        "disagree with this line -- trim the tiling or "
                        "free VRAM.")
            else:
                tail = (" The gates above "
                        "compare the itemized estimate; the envelope is "
                        "what the machine is measured to reach, so this "
                        "configuration may run out of budget even though "
                        "the estimate gate passes -- trim the "
                        "configuration or free VRAM before trusting the "
                        "pass.")
            print(f"  WARNING: observed peak envelope "
                  f"{_format_bytes(envelope)} exceeds the {budget_word} "
                  f"{_format_bytes(envelope_budget)} -- free VRAM less the "
                  f"{_format_bytes(EXTERNAL_MARGIN_BYTES)} other-process "
                  f"margin, which is all the envelope does not already "
                  f"model -- by {_format_bytes(envelope - envelope_budget)}."
                  f"{tail}  ({code_note}.)")
        if streamed_alloc_gate:
            # A STREAMED RUN'S LEVER IS THE TILE, not the RRTMGP column
            # chunk.  The resident alloc estimate is above the budget for
            # essentially every streamed config -- it describes a domain
            # this run never allocates -- so the block below would print
            # "OVER BUDGET; first lever --column-chunk N" beside a
            # verdict that had just said the run fits, and the lever it
            # named would move a number nothing compares.
            if int(phases.streamed.peak_vram_bytes) > envelope_budget:
                env = phases.streamed
                # WHICH TERM IS OVER.  With the radiation transient in the
                # figure, a reader told to trim the tile has to be able to
                # see how much of the overshoot the tile can actually
                # move: the transient is a property of the RUNG and no
                # tile size reduces it by a byte.
                transient = (
                    "" if not env.radiation_transient_bytes else
                    f" plus the measured "
                    f"{_format_bytes(int(env.radiation_transient_bytes))} "
                    f"RRTMGP transient, which no tile size reduces,")
                print(f"  OVER BUDGET, STREAMED: {env.nbuffers} buffer(s) "
                      f"of the {env.window_nx}x{env.window_ny} compute "
                      f"window (tile {env.tile_nx}x{env.tile_ny} + halo "
                      f"{env.halo}) hold "
                      f"{_format_bytes(int(env.vram_bytes))}{transient} "
                      f"and reach "
                      f"{_format_bytes(int(env.peak_vram_bytes))} against "
                      f"the {_format_bytes(envelope_budget)} envelope "
                      f"budget.")
                print("  remedy: a smaller [tiles] tile_nx/tile_ny, or "
                      "nbuffers = 1 to trade overlap for room, or free "
                      "VRAM and re-run")
        elif budget is not None and estimate.alloc_estimate_bytes > budget:
            lever = recommend_column_chunk(exp, budget)
            if lever:
                print("  OVER BUDGET; first lever (RRTMGP column_chunk): "
                      f"--column-chunk {lever}")
            else:
                # It used to end "staged residency (DESIGN REOPEN) per
                # section E".  No pip user has a section E, and the
                # sentence names no action; the actionable one already
                # exists one layer up, in `gpuwm go`'s refusal.  The
                # remedy it then printed -- `gpuwm domain --vram-gib
                # <free>` -- fed a free-VRAM figure to a flag that names
                # a CARD, and on the 3080 walk that recursion refused at
                # every grid size.  The bare wizard measures the card
                # itself, which is the number this remedy actually means.
                print("  OVER BUDGET, and the RRTMGP column_chunk lever "
                      "cannot close it: no chunk halving fits after the "
                      "shared-scratch arena, so the grid itself is what "
                      "has to come down.")
                print("  remedy: re-size against this machine -- gpuwm "
                      "domain ... (bare, it measures this card) -- or "
                      "pick a lighter --physics-profile, or free VRAM "
                      "and re-run")
    if abort is not None:
        return 3
    if args.alloc:
        # A requested measurement run fails CLOSED: every leg must have
        # been measured AND passed (shadow F5 / Fable F6).
        if not all(leg is True for leg in gates.values()):
            return 1
        return _EXIT_ENVELOPE_OVER_BUDGET if envelope_over_budget else 0
    evaluable = [leg for leg in gates.values() if leg is not None]
    if not evaluable:
        # Fail closed, and SAY SO.  This exit used to be silent: the
        # wizard prints `gpuwm check CONFIG` as its own step 2, and on a
        # box with no measurable card that command printed the estimate,
        # three "not measured" gate lines, and exit 2 with no sentence
        # naming why or what to type next (UX finding R1, replay walk C
        # step 1).  A refusal names the breakage -- nothing here to
        # verify an estimate against -- and prints a remedy the reader
        # can type: the declared-budget form of THIS command, and the
        # wizard door that prints that form with the numbers filled in.
        print("gpuwm check: REFUSED (exit 2, fail-closed): no gate could "
              "be evaluated -- no VRAM budget was declared and no card "
              "could be measured in this machine (CuPy or a CUDA device "
              "is absent), so the estimate above has nothing to be "
              "verified against.", file=sys.stderr)
        print(f"  remedy: declare the card this config is sized for and "
              f"re-run:\n"
              f"    gpuwm check {args.config} --budget-gib <N> "
              f"--vram-gib <card GiB>\n"
              f"  # this configuration's peak envelope is "
              f"{envelope / GIB:.2f} GiB; a budget at or above it "
              f"passes.\n"
              f"  # `gpuwm domain --card <tier>` (or --vram-gib N) "
              f"prints this exact check line, numbers filled in, as the "
              f"comment under its step 2.", file=sys.stderr)
        return 2  # nothing verifiable: fail closed at the command boundary
    if not all(evaluable):
        return 1
    # Gates passed.  The report may still have said, in its own words,
    # that the machine peak lands above the budget -- that sentence and
    # exit 0 cannot both be true, and the sentence is the accurate one.
    return _EXIT_ENVELOPE_OVER_BUDGET if envelope_over_budget else 0


def register_cli(subparsers) -> None:
    """Register ``gpuwm check`` (memory section + ``--alloc``).  Per the F2
    ownership map the one-line ``cli.py`` hookup is a controller handoff
    commit at merge; Task 3's input-catalog section joins the same
    subcommand at its own handoff."""
    p = subparsers.add_parser(
        "check",
        help="memory preflight: itemized estimate vs the measured WDDM "
             "budget; --alloc performs the enforced N0 allocation run")
    p.add_argument("config", type=Path, metavar="CONFIG",
                   help="experiment TOML (or legacy RunConfig TOML, "
                        "wrapped as a one-domain experiment)")
    p.add_argument("--alloc", action="store_true",
                   help="construct every persistent allocation on the "
                        "device, zero steps, report measured vs estimate "
                        "(N0; GPU required)")
    p.add_argument("--column-chunk", type=int, default=None,
                   metavar="COLS", help="RRTMGP chunk override (the first "
                   "over-budget lever)")
    p.add_argument("--reserve-gib", type=float, default=None, metavar="GIB",
                   help="override the calibrated reserve policy with a "
                        "flat reserve")
    p.add_argument("--budget-gib", type=float, default=None, metavar="GIB",
                   help="CPU-mode measured budget (free VRAM minus "
                        "reserve) for the estimate<=budget leg")
    p.add_argument("--vram-gib", type=float, default=None, metavar="GIB",
                   help="physical VRAM total of the card being sized for.  "
                        "A CEILING on the free figure, never a source of "
                        "one: a declared --budget-gib plus the reserve can "
                        "otherwise synthesise more free VRAM than the card "
                        "physically has")
    p.add_argument("--rail-mib", type=int, default=None, metavar="MIB",
                   help="whole-machine device residency ceiling: the budget "
                        "is additionally capped at RAIL minus what every "
                        "other process on the card already holds (read from "
                        "NVML before this process touches CUDA).  A property "
                        "of the host, so there is no default")
    p.add_argument("--forcing-interval-s", type=float,
                   default=DEFAULT_FORCING_INTERVAL_SECONDS, metavar="S",
                   help="forcing cadence sizing the root's eager LBC "
                        "tables (default ERA5 6-hourly)")
    p.add_argument("--json", action="store_true",
                   help="machine-readable report")
    p.set_defaults(func=check_main)


__all__ = [
    "CORE_KERNEL_MODULES", "CUDA_CONTEXT_BYTES", "DeviceLocalMemoryProfile",
    "POOL_RESERVED_OVER_ESTIMATE_FRACTION",
    "KERNEL_MAX_LOCAL_SIZE_BYTES", "KERNEL_LOCAL_FRAME_RECORDINGS",
    "KernelFrameRecording", "kernel_frame_recording_for",
    "UnderPricedKernelFrame", "under_priced_kernel_frames",
    "MEASURED_LOCAL_MEMORY_PROFILE",
    "CHAINED_TRANSLATION_UNIT_FRAMES", "ChainedTranslationUnitFrame",
    "UNMEASURED_KERNEL_MODULES", "local_memory_profile_from_device",
    "cap_free_to_physical", "device_physical_total_bytes",
    "device_rail_free_bytes", "device_wide_used_bytes",
    "column_workspace_bytes", "gf_column_workspace_bytes",
    "kf_column_workspace_bytes", "ysu_column_workspace_bytes",
    "kernel_local_memory_bytes", "non_pool_device_bytes",
    "physics_kernel_modules", "refl_diagnostic_reachable",
    "LEVEL_SPECIALIZED_KERNEL_FRAMES", "LevelSpecializedFrame",
    "ACOUSTIC_TIER_FRAME", "TieredKernelFrame", "WDM6_TIER_FRAME",
    "WSM6_TIER_FRAME",
    "domain_kernel_modules", "kernel_local_frame_bytes",
    "ALLOCATOR_HEADROOM", "AllocReport", "CAL_D01_DEVICE_FOOTPRINT_BYTES",
    "CAL_D01_WORKSPACE_BYTES",
    "CAL_D01_POOL_HELD_BYTES", "CAL_D01_POOL_RETENTION_BYTES",
    "CAL_D01_POOL_USED_PEAK_BYTES", "CAL_WDDM_FREE_BYTES",
    "CAL_WDDM_TOTAL_BYTES", "DEFAULT_COLUMN_CHUNK",
    "DEFAULT_FORCING_INTERVAL_SECONDS", "CAL_FIXTURE_OVERHEAD_BYTES",
    "DomainMemoryEstimate", "EXTERNAL_MARGIN_BYTES",
    "ExperimentMemoryEstimate", "GIB", "MemoryItem", "N0_GATE_METRICS",
    "PROBE_DEVICE_FOOTPRINT_BYTES", "PROBE_DEVICE_OVERHEAD_BYTES",
    "PROBE_FREE_BYTES", "PROBE_POOL_HELD_BYTES",
    "PROBE_POOL_USED_PEAK_BYTES",
    "PHYSICS_ARRAY_LIFETIME_AUDIT", "PhysicsArrayLifetime",
    "PreflightAllocError", "PreflightHeadroomError", "ReservePolicy",
    "SCRATCH_SLOT_LIFETIME_AUDIT", "ScratchSlotLifetime",
    "atmosphere_transient_shapes", "check_main", "dudhia_column_shapes",
    "estimate_domain", "estimate_experiment", "evaluate_alloc_gates",
    "gate_display_name",
    "k_distribution_bytes", "lbc_interval_values", "lbc_intervals",
    "myj_output_transient_shapes",
    "nest_allocation_manifest", "nest_field_kinds", "nest_slot_dtypes",
    "nest_slot_shapes",
    "physics_array_lifetime", "physics_array_shapes",
    "physics_field_names_2d",
    "pool_retention_residual_bytes", "recommend_column_chunk",
    "register_cli", "rrtmgp_column_shapes", "rrtmgp_workspace_phases",
    "rrtmgp_workspace_shapes",
    "run_alloc_preflight", "scratch_slot_lifetime",
    "scratch_slot_registry", "scratch_slot_uses_arena",
    "shared_dycore_state_symbols", "shared_dycore_state_workspace_bytes",
    "shared_dycore_state_workspace_shapes",
    "shared_scratch_arena_aliases", "shared_scratch_arena_bytes",
    "shared_scratch_arena_shapes", "shinhong_output_transient_shapes",
    "state_array_shapes",
    "ysu_output_transient_shapes",
    "ENVELOPE_AFFINE_BASIS", "ENVELOPE_PER_NEST_FRACTION",
    "ENVELOPE_UNMODELLED_BYTES", "ENVELOPE_WDDM_BASIS",
    "WDDM_POOL_SLACK_FRACTION", "CARD_CLASS_MULTIPROCESSORS",
    "card_local_memory_profile", "live_device_local_memory_profile",
    "machine_peak_envelope_bytes", "observed_peak_envelope_bytes",
    "peak_envelope_factor", "PEAK_ENVELOPE_FACTORS",
    "PEAK_ENVELOPE_BASIS", "envelope_platform", "estimate_ingest",
    "estimate_phases", "IngestMemoryEstimate", "PhaseMemoryEstimate",
]
