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

from gpuwm.config import (DEFAULT_COLUMN_CHUNK, SASE_PBL_SCHEME, RunConfig,
                          radiation_enabled, radiation_scheme_ids,
                          soil_layer_count)
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

#: Observed machine-peak envelope factor over the footprint projection.
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
#: WINDOWS, SMALL CARDS -- 1.45, EXPERIMENTAL and calibrated on nothing.
#:
#: The 1.75 factor and the 4.12 GiB of projection constants beside it both
#: come from ONE machine: a 32 GiB RTX 5090 running campaign-scale
#: multi-domain forecasts.  On a 12 GiB Windows card those constants are
#: 34% of the whole card before a single grid cell is allocated, and the
#: wizard refused every ladder -- not because the model did not fit, but
#: because an accounting term measured somewhere else did.  Refusing to
#: size a card gpuwm can probably run is a worse failure than sizing it
#: optimistically, because the optimistic failure mode here is bounded:
#: WDDM pages, or the allocation fails cleanly.  Neither corrupts a
#: forecast.
#:
#: So a small Windows card is priced like Linux -- the itemized alloc
#: estimate under the 1.45 envelope -- plus ONE reduced fixed reserve
#: (:data:`WINDOWS_SMALL_CARD_RESERVE_BYTES`) standing in for the WDDM
#: residency the pool never sees.  It is a pioneer tier: see
#: :func:`windows_small_card_advisory` for what users are asked to send
#: back, which is the only thing that will turn this into a measurement.
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
#: These factors are RETAINED for the WDDM lane, whose one instrumented
#: run is a true multiplicative observation over a projection that carries
#: the Windows pool constants, and as the historical record.
PEAK_ENVELOPE_FACTORS = {"windows": 1.75, "windows-small": 1.45,
                         "linux": 1.45}

#: How much evidence each factor rests on, printed beside the number.
PEAK_ENVELOPE_BASIS = {
    "windows": "measured, 1 WDDM run",
    "windows-small": "EXPERIMENTAL, no measurements on this card class",
    "linux": "measured-preliminary, 3 runs",
}

#: Windows cards at or below this size take the experimental accounting.
WINDOWS_SMALL_CARD_MAX_GIB = 12.0

#: The single fixed reserve replacing the 5090-derived pool constants on
#: a small Windows card: WDDM residency the CuPy pool never accounts for.
#: A round 1.5 GiB, chosen to be smaller than the 4.12 GiB it replaces and
#: larger than the 0.43 GiB CUDA context alone -- itself a guess, and the
#: single most valuable number for a pioneer to send back.
WINDOWS_SMALL_CARD_RESERVE_BYTES = 3 * GIB // 2

#: Retained name for the WDDM factor (the original single-platform value).
OBSERVED_PEAK_OVER_FOOTPRINT = PEAK_ENVELOPE_FACTORS["windows"]


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
#: hierarchy build, that the prepared-hierarchy route refuses two-way
#: nesting outright.  The refusal is correct and stays exactly as it is;
#: what misled was the silence upstream of it.  This is an advisory, not
#: a gate: it changes no exit code and blocks nothing.
FEEDBACK_TWO_WAY_ADVISORY = (
    "experimental: feedback = 1 selects two-way nest feedback, which is "
    "stamped experimental and runs on the native experiment-runner route "
    "(`gpuwm run`) only.  The prepared-hierarchy route -- `rw-wps` "
    "preparation followed by the domain-tree runner -- refuses it at "
    "preparation, so a config carrying feedback = 1 cannot be prepared "
    "there no matter what this preflight reports."
)


def feedback_advisory(exp) -> str | None:
    """The two-way advisory when it applies, else None."""

    return (FEEDBACK_TWO_WAY_ADVISORY
            if int(getattr(exp, "feedback", 0) or 0) == 1 else None)


def check_advisories(exp, config_path=None) -> list[str]:
    """Every route advisory this config earns, in report order.

    Same posture as ``feedback_advisory``: these change no exit code and
    block nothing.  They exist because a legal config that a later stage
    silently ignores is worse than one it refuses -- the user learns
    after paying for the run instead of before it.
    """

    from gpuwm.checkpoint_routes import (
        checkpoint_route_advisory, config_has_case_data)

    advisories = [feedback_advisory(exp)]
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
    """Which envelope family applies: ``windows``, ``windows-small``, or
    ``linux``.

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

    VRAM_GIB is the *card* size, and only a caller that knows it -- the
    domain wizard, sizing for a named card -- can select the experimental
    small-Windows tier.  Callers that do not pass it (``gpuwm check``
    among them, which measures free VRAM rather than card size) keep
    today's conservative Windows accounting exactly.  An unknown
    platform does NOT reach ``windows-small``: that tier is an
    experiment about WDDM on a small card, and a platform nobody has
    measured is not the place to run a second experiment.
    """

    name = sys.platform if platform is None else str(platform)
    if name.startswith(_LINUX_PLATFORMS):
        return "linux"
    if not (name.startswith(_WINDOWS_PLATFORM_PREFIXES)
            or name in _WINDOWS_PLATFORM_NAMES):
        return "windows"
    small = (vram_gib is not None and math.isfinite(float(vram_gib))
             and float(vram_gib) <= WINDOWS_SMALL_CARD_MAX_GIB)
    return "windows-small" if small else "windows"


def peak_envelope_factor(platform: str | None = None,
                         vram_gib: float | None = None) -> float:
    """The machine-peak envelope factor this platform's evidence supports."""

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
    Windows; the platform envelope factor carries the margin either way.

    A small Windows card takes neither: one reduced fixed reserve
    (:data:`WINDOWS_SMALL_CARD_RESERVE_BYTES`) stands in for both, because
    the 5090-derived pair is a third of such a card before any grid
    exists.  Experimental -- see :data:`PEAK_ENVELOPE_FACTORS`.
    """

    family = envelope_platform(platform, vram_gib)
    if family == "windows":
        return pool_retention_residual_bytes(), PROBE_DEVICE_OVERHEAD_BYTES
    if family == "windows-small":
        return 0, WINDOWS_SMALL_CARD_RESERVE_BYTES
    return 0, 0


def windows_small_card_advisory(vram_gib: float) -> tuple[str, ...]:
    """The pioneer warning for an experimentally-sized Windows card.

    Says plainly that the accounting is extrapolated from one much larger
    machine, what the worst case is (and that it is not corruption), and
    exactly what measurement would replace the guess.
    """

    return (
        f"EXPERIMENTAL: {vram_gib:g} GiB is at or below the "
        f"{WINDOWS_SMALL_CARD_MAX_GIB:g} GiB Windows small-card threshold, "
        "and this layout was sized with experimental accounting.",
        "  Windows/WDDM memory accounting in gpuwm is calibrated from ONE "
        "much larger machine (a 32 GiB RTX 5090 running campaign-scale "
        "forecasts); its 4.12 GiB of fixed pool constants would consume a "
        "third of this card before a single grid cell, so a reduced "
        f"{WINDOWS_SMALL_CARD_RESERVE_BYTES / GIB:.1f} GiB reserve and the "
        f"Linux {PEAK_ENVELOPE_FACTORS['windows-small']:.2f} envelope were "
        "used instead.",
        "  Worst case is paging (slow) or a clean out-of-memory failure "
        "before or during the run. Neither corrupts a forecast, and "
        "neither damages anything.",
        "  Please report your measured peak so this stops being a guess: "
        "run the forecast, then send the peak line from "
        "`gpuwm check <config>` together with the config file and your "
        "card model.")


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
        family: str = "linux") -> int:
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

    The WDDM lane keeps its one instrumented multiplicative observation
    and takes the LARGER of the two: the affine form is a floor there,
    never a discount.
    """

    nests = max(0, int(domains) - 1)
    affine = (int(alloc_estimate_bytes) + int(non_pool_bytes)
              + ENVELOPE_UNMODELLED_BYTES
              + math.ceil(ENVELOPE_PER_NEST_FRACTION * nests
                          * int(alloc_estimate_bytes)))
    if family == "windows" and footprint_projection_bytes is not None:
        return max(affine, int(footprint_projection_bytes
                               * PEAK_ENVELOPE_FACTORS["windows"]))
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


@dataclass(frozen=True)
class DeviceLocalMemoryProfile:
    """The device constants the local-memory reservation law needs."""

    name: str
    multiprocessor_count: int
    max_threads_per_multiprocessor: int
    default_stack_limit_bytes: int = 1024

    @property
    def resident_thread_capacity(self) -> int:
        return (self.multiprocessor_count
                * self.max_threads_per_multiprocessor)

    def reservation_bytes(self, max_local_size_bytes: int) -> int:
        """Device bytes the driver reserves for a launched kernel whose
        per-thread local frame is ``max_local_size_bytes``.  Zero when the
        frame fits the default stack, whose store the context already
        carries."""
        over = int(max_local_size_bytes) - self.default_stack_limit_bytes
        return 0 if over <= 0 else over * self.resident_thread_capacity


#: Measured 2026-07-26 from ``cudaGetDeviceProperties`` +
#: ``cudaDeviceGetLimit(cudaLimitStackSize)`` on the run host.
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


def local_memory_profile_from_device(cp) -> DeviceLocalMemoryProfile:
    """Read the profile off the attached device (``--alloc`` path only)."""
    # ``gpuwm multi-run`` masks one physical UUID into each check process;
    # CUDA ordinal 0 is therefore the selected logical device, not a claim
    # that every run belongs on the machine's physical index zero.
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    return DeviceLocalMemoryProfile(
        name=name.decode() if isinstance(name, bytes) else str(name),
        multiprocessor_count=int(props["multiProcessorCount"]),
        max_threads_per_multiprocessor=int(
            props["maxThreadsPerMultiProcessor"]),
        default_stack_limit_bytes=int(cp.cuda.runtime.deviceGetLimit(0)),
    )


#: Per-module MAXIMUM static local frame per thread, in bytes, as the CUDA
#: driver reports it in ``CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`` after NVRTC
#: compiles ``gpuwm/core/kernels/<module>.cu`` with the shipped options and
#: NO integer defines injected -- i.e. at each module's unspecialized bound.
#: Measured 2026-07-26 over every ``extern "C" __global__`` symbol in every
#: module; regenerated and compared by
#: ``tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver``.
#:
#: Two modules do not launch at their unspecialized bound: ``kf`` and
#: ``refl`` compile their column arrays to the configuration's own level
#: count (:data:`LEVEL_SPECIALIZED_KERNEL_FRAMES`), so their rows here are
#: the CEILING, not the price.  ``acoustic`` is a third case and the only
#: one that can price ABOVE its row: it compiles the implicit w''-phi''
#: solve at a coarse ``WPHI_MAX_LEV`` tier chosen by ``nz``
#: (:data:`ACOUSTIC_TIER_FRAME`), so this row is its price at the shipped
#: 129 tier -- every ``nz <= 128`` configuration -- and the deeper tiers add
#: to it.  Everything else launches as compiled.
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
    # Grell-Freitas: one thread owns one whole GFDRV column, so the frame
    # is the deep+shallow local-array stack (measured on the driver probe,
    # nz=40 tier).
    "gf": 22416,
    "health": 0,
    # The batched symmetric eigensolver the radar-DA analysis factors with
    # (gpuwm/core/jacobi_eigh.py).  It holds NO local frame at any tier: the
    # whole k x k problem lives in dynamic SHARED memory, which is priced at
    # launch and released with the block rather than reserved per resident
    # thread for the life of the process.  That is the entire reason the
    # kernel is written around shared memory instead of per-thread arrays.
    "jacobi_eigh": 0,
    "kessler": 5120,
    "kf": 24064,
    "kf_validation": 0,
    "lbc_flow": 0,
    "lbc_state": 0,
    "morrison": 5120,
    "microphysics_validation": 0,
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
    "noah": 176,
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
    "nssl2_fused_gs": 112,
    "nssl2_nucond": 0,
    "nssl2_qvexcess": 0,
    "openbc": 0,
    "pd_advection": 0,
    "refl": 18432,
    "rrtmg_lw": 0,
    "rrtmg_mcica_wrf": 0,
    "rrtmgp_cloud": 0,
    "rrtmgp_gas": 512,
    "rrtmgp_mcica": 0,
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
    "shinhong": 14040,
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
    "thompson_aerosol_warm": 0,
    "tke_budget": 0,
    "uh_diag": 0,
    "vert_interp": 768,
    "wsm6": 7216,
    "ysu": 9232,
    "ysu_validation": 0,
}

@dataclass(frozen=True)
class LevelSpecializedFrame:
    """A kernel module whose per-thread local frame is compiled to ``nz``.

    ``kf.cu`` and ``refl.cu`` declare their per-thread column arrays against
    a ``#ifndef``-guarded compile-time bound, and their launchers specialize
    that bound to the field's own level count through
    ``gpuwm.core.kernels.get_kernel_int_defines``.  The column arrays are the
    only thing in either kernel that scales with the bound, so the frame the
    driver reports is ``bytes_per_level * levels`` rounded up to the local
    frame's 8-byte granularity.

    Measured on the RTX 5090 (driver 610.74, NVRTC 13.x, ``-std=c++17``),
    module maximum per bound:

      ==========  ============  ==========  ==========
      module      bound         frame       reserved
      ==========  ============  ==========  ==========
      kf          128 (ceiling)  24,064 B   5,738 MiB
      kf           49            9,216 B    2,040 MiB
      kf           30            5,640 B    1,152 MiB
      refl        256 (ceiling)  18,432 B   4,334 MiB
      refl         49            3,528 B      626 MiB
      refl         30            2,160 B      286 MiB
      ==========  ============  ==========  ==========

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


#: The two modules whose local frame follows the configuration's ``nz``.
#: ``bytes_per_level`` is fixed by construction against the unspecialized
#: row of :data:`KERNEL_MAX_LOCAL_SIZE_BYTES` (checked at import below), so
#: the two tables cannot drift apart.
LEVEL_SPECIALIZED_KERNEL_FRAMES: dict[str, LevelSpecializedFrame] = {
    "kf": LevelSpecializedFrame("kf", "KF_KMAX", 128, 188),
    "refl": LevelSpecializedFrame("refl", "REFL_KMAX", 256, 72),
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

    ``kf`` and ``refl`` compile exactly to the configuration's level count;
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

#: Kernel modules whose local frame CANNOT be measured at this checkout
#: because they do not compile alone: ``noahmp_driver.cu``,
#: ``noahmp_energy.cu``, ``noahmp_thermal.cu`` and ``noahmp_libm_slab.cu``
#: all fail NVRTC with ``identifier "r_pow" is undefined`` -- they are
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
#: exactly ``gpuwm/config.py``'s accepted set (0, 1, 6, 8, 10, 18) plus 28,
#: which is priced here ahead of its driver dispatch so a run can never
#: reach ``kernel_local_frame_bytes`` with an unpriced selector.
_MICROPHYSICS_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("kessler", "microphysics_validation"),
    6: ("wsm6", "microphysics_validation"),
    8: ("thompson", "microphysics_validation"),
    10: ("morrison", "microphysics_validation"),
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
_REFLECTIVITY_MICROPHYSICS = frozenset({1, 6, 8, 10, 18, 28})

_CUMULUS_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("kf", "kf_validation"),
    # One translation unit carries all of GFDRV (deep, shallow, driver).
    3: ("gf",)}
_PBL_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("ysu", "ysu_validation"), 5: ("mynn_pbl",),
    # Shin-Hong launches its column kernel plus its own batched output
    # validator, the YSU pair's shape (gpuwm/core/shinhong.py).
    11: ("shinhong", "shinhong_validation"),
    SASE_PBL_SCHEME: ("sase",)}
_SURFACE_LAYER_KERNEL_MODULES: dict[int, tuple[str, ...]] = {
    0: (), 1: ("sfclay",), 5: ("mynn_surface",), 91: ("sfclay",)}
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
        modules.add("refl")
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
        raise ValueError(
            "cannot price the local-memory reservation: "
            f"{', '.join(unmeasured)} do not compile at this checkout "
            "(NVRTC: identifier \"r_pow\" is undefined), so their per-thread "
            "local frame has never been measured.  Refusing to guess.")
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
    the CUDA context plus the local-memory backing store."""
    return CUDA_CONTEXT_BYTES + kernel_local_memory_bytes(
        exp, profile=profile)


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
        if cfg.mp_physics in (6, 8, 10, 18, 28):
            for name in ("qi", "qs", "qg", "qi0", "qs0", "qg0",
                         "effc", "effi", "effs"):
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
    from gpuwm.core.sfclay import SFCLAY_OUTPUTS

    union = dict.fromkeys(_PHYSICS_INIT_FIELDS_2D)
    union.update(dict.fromkeys(SFCLAY_OUTPUTS))
    if cfg is not None and int(cfg.sf_sfclay_physics) == 5:
        from gpuwm.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS
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
        from gpuwm.core.mynn_pbl_runtime import (
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
            from gpuwm.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS
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
    from gpuwm.core.physics import (hmix_k_diag_names,
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
        from gpuwm.core.mynn_pbl_runtime import MYNN_PBL_STATE_3D
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
    if cfg.bl_pbl_physics and cfg.mp_physics in (6, 8, 10, 18, 28):
        # Mixed-phase states carry qi; YSU returns dqi and rqi survives
        # composition (physics.py:263-264, :681-692).  mp=28 belongs by
        # Registry/Registry.EM_COMMON:3036 -- the thompsonaero package
        # declares moist:qv,qc,qr,qi,qs,qg, so WRF's F_QI is true and
        # module_first_rk_step_part1.F:1112's CALL pbl_driver hands
        # moist(...,P_QI), F_QI=F_QI (:1199) to the PBL driver.  This budget mirrors physics._pbl_optional_tendency_
        # components; the two sets must stay identical.
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
        # The OLR publication buffer (physics.py PhysicsDriver __init__):
        # one resident (ny, nx) FP32 driver persistent holding WRF's TOA
        # outgoing longwave for output, allocated when the attached
        # longwave adapter declares ``publishes_olr``.  BOTH built-in 4/4
        # adapters declare it, which is what this gate reproduces; a
        # caller-injected adapter that does not leaves this counted but
        # unallocated, and over-counting by one 2-D field is the safe
        # direction for a VRAM estimate.
        shapes["olr"] = s2
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
        from gpuwm.core.microphysics import spec_zone_ring_save_slots
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
            if cfg.mp_physics in (6, 8, 10, 18, 28):
                for name in ("qi", "qs", "qg"):
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

    from gpuwm.core.physics import physics_enabled
    if physics_enabled(cfg):
        slots["physics_qtot"] = m                   # physics.py:369
        if not cfg.moist:
            slots.update(physics_dry_qv=m, physics_dry_qc=m)
        if cfg.mp_physics not in (6, 8, 10, 18, 28):
            # physics.py:984-993 substitutes a zero-filled scratch plane only
            # when the state has no qi/qs of its own.  mp=28 allocates both,
            # so listing them here would price two full 3-D fields the run
            # never asks for.
            slots["physics_qi"] = m                 # physics.py:984-988
            slots["physics_qs"] = m                 # physics.py:989-993
    if (int(cfg.bl_pbl_physics) in (1, 11)
            or int(cfg.mp_physics) in (1, 6, 8, 10, 28)
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
        if cfg.mp_physics in (6, 8, 10, 18, 28):
            kinds += ["qi", "qs", "qg"]
        if cfg.mp_physics == 8:
            kinds += ["nr", "ni"]
        if cfg.mp_physics == 28:
            kinds += ["nr", "ni", "nc", "nwfa", "nifa"]
        if cfg.mp_physics == 10:
            kinds += ["nr", "ni", "ns", "ng"]
        if cfg.mp_physics == 18:
            kinds += ["qh", "qndrop", "qnr", "qni", "qns", "qng",
                      "qnh", "qnn", "qvolg", "qvolh"]
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
         "smag_rng", "smag_rqh", "smag_rqndrop", "smag_rqnr",
         "smag_rqni", "smag_rqns", "smag_rqng", "smag_rqnh",
         "smag_rqnn", "smag_rqvolg", "smag_rqvolh",
         # mp=28 transported number/aerosol moments.  Same construction,
         # same lifetime: prepare_fixed_tendencies writes each held
         # tendency once before the RK loop and every stage only reads it.
         "smag_rnc", "smag_rnwfa", "smag_rnifa"),
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
        ("moist_pd_q0", "moist_rq_t", "pd_fxl", "pd_fxc", "pd_fyl",
         "pd_fyc", "pd_fzl", "pd_fzc"), "write_before_read",
        "gpuwm/core/moist.py:197-203,247-308",
        "source copies, tendencies, and six PD fluxes are filled before use"),
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
        ("morr_theta", "morr_rho", "morr_pii", "morr_dz",
         "morr_ice_to_snow", "morr_z8w"), "write_before_read",
        "gpuwm/core/morrison.py:159-202",
        "Morrison preparation fields are rebuilt for every scheme call"),
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
        "gpuwm/core/shinhong.py:validate_shinhong_outputs",
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


def rrtmgp_workspace_phases(nz: int, column_chunk: int, p_top: float = 10000.0
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
    * ``lw_rte`` (:1307-1317): mask deleted; Planck lay/lev/sfc sources
      (:1747-1749) + emissivity/incident g-point arrays join while both
      tau cubes and the two chunk flux outputs stay live through :1313-1316.
    * ``sw_optics`` (:1353-1368): gas tau/ssa + finalized tau/ssa/g (five
      g-point cubes) + mask + VMR + band cloud optics + col_dry.
    * ``sw_rte`` (:1369-1381): mask deleted; albedo/incidence g-point
      arrays, the materialized (chunk,nz) mu0 broadcast (:1690-1691) and
      three (chunk,nz+1) flux arrays join while all five cubes stay live
      until the ``del`` at :1380.
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

    lw_common = {
        "gas_tau": ((c, lw_nz, lw_g), 4),
        "optics_tau": ((c, lw_nz, lw_g), 4),
        "vmr": ((c, lw_nz, meta["ngas_lw"] + 1), 4),
        "cld_tau": ((c, lw_nz, meta["nband_lw"]), 4),
        "cld_ssa": ((c, lw_nz, meta["nband_lw"]), 4),
        "cld_asy": ((c, lw_nz, meta["nband_lw"]), 4),
        "col_dry": ((c, lw_nz), 4),
    }
    sw_common = {
        "gas_tau": ((c, sw_nz, sw_g), 4),
        "gas_ssa": ((c, sw_nz, sw_g), 4),
        "optics_tau": ((c, sw_nz, sw_g), 4),
        "optics_ssa": ((c, sw_nz, sw_g), 4),
        "optics_g": ((c, sw_nz, sw_g), 4),
        "vmr": ((c, sw_nz, meta["ngas_sw"] + 1), 4),
        "cld_tau": ((c, sw_nz, meta["nband_sw"]), 4),
        "cld_ssa": ((c, sw_nz, meta["nband_sw"]), 4),
        "cld_asy": ((c, sw_nz, meta["nband_sw"]), 4),
        "col_dry": ((c, sw_nz), 4),
    }
    return {
        "lw_optics": {**lw_common,
                       "mcica_mask": ((c, lw_nz, lw_g), 1)},
        "lw_rte": {**lw_common,
                   "lay_source": ((c, lw_nz, lw_g), 4),
                   "lev_source": ((c, lw_nz + 1, lw_g), 4),
                   "sfc_source": ((c, lw_g), 4),
                   "emiss_gpt": ((c, lw_g), 4),
                   "incident": ((c, lw_g), 4),
                   "flux_up": ((c, lw_nz + 1), 4),
                   "flux_dn": ((c, lw_nz + 1), 4)},
        "sw_optics": {**sw_common,
                       "mcica_mask": ((c, sw_nz, sw_g), 1)},
        "sw_rte": {**sw_common,
                   "albedo_gpt": ((c, sw_g), 4),
                   "inc_gpt": ((c, sw_g), 4),
                   "mu0": ((c, sw_nz), 4),
                   "flux_up": ((c, sw_nz + 1), 4),
                   "flux_dn": ((c, sw_nz + 1), 4),
                   "flux_dir": ((c, sw_nz + 1), 4)},
    }


def rrtmgp_workspace_shapes(nz: int, column_chunk: int, p_top: float = 10000.0
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
        cfg: RunConfig, p_top: float = 10000.0, *,
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
    elif cfg.mp_physics in (6, 8, 18, 28):
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
    from gpuwm.core.physics import physics_enabled

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
                    p_top: float = 10000.0,
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

        The itemized non-pool residency, PLUS whatever fixed platform
        term the projection carries: zero on Linux, and on the
        experimental Windows small-card tier the 1.5 GiB standing in for
        the WDDM residency the CuPy pool never sees.  That tier has no
        measurements behind it, so replacing its multiplier with an
        intercept must not also drop the term it was carrying -- an
        unmeasured tier is not the place to become more optimistic.
        """
        return self.non_pool_device_bytes + self.device_overhead_bytes

    @property
    def peak_envelope_bytes(self) -> int:
        """The affine machine-peak envelope for this forecast."""
        return machine_peak_envelope_bytes(
            alloc_estimate_bytes=self.alloc_estimate_bytes,
            non_pool_bytes=self.envelope_intercept_bytes,
            domains=len(self.domains),
            footprint_projection_bytes=self.footprint_projection_bytes,
            family=self.envelope_family)

    @property
    def affine_envelope_bytes(self) -> int:
        """The affine form alone, whichever branch actually bound."""
        return machine_peak_envelope_bytes(
            alloc_estimate_bytes=self.alloc_estimate_bytes,
            non_pool_bytes=self.envelope_intercept_bytes,
            domains=len(self.domains), family="linux")

    @property
    def envelope_basis(self) -> str:
        """The evidence behind whichever branch actually bound."""
        if self.envelope_is_wddm_floor:
            return PEAK_ENVELOPE_BASIS["windows"]
        return ENVELOPE_AFFINE_BASIS

    @property
    def envelope_is_wddm_floor(self) -> bool:
        """Did the retained WDDM multiplier bind instead of the sum?"""
        return self.peak_envelope_bytes > self.affine_envelope_bytes

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
        affine = (f"estimate {self.alloc_estimate_bytes / GIB:.2f} + "
                  f"non-pool {self.envelope_intercept_bytes / GIB:.2f} (CUDA "
                  f"context + local-memory backing store) + "
                  f"{ENVELOPE_UNMODELLED_BYTES / GIB:.2f} unmodelled"
                  f"{nest_term} = "
                  f"{self.affine_envelope_bytes / GIB:.2f} GiB")
        if not self.envelope_is_wddm_floor:
            return affine
        factor = PEAK_ENVELOPE_FACTORS["windows"]
        return (f"footprint {self.footprint_projection_bytes / GIB:.2f} x "
                f"{factor:.2f} WDDM floor = "
                f"{self.peak_envelope_bytes / GIB:.2f} GiB, which is above "
                f"the affine form ({affine}) and therefore binds")


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

#: Forcing times a streaming ingest holds on the device at once: the
#: INITIAL time -- which the prepared cache, wrfinput and the surface
#: analysis are all written from -- plus the time currently being built.
#: Every other time contributes its perimeter frames and is released
#: before the next one is interpolated.
INGEST_RESIDENT_FORCING_TIMES = 2

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
#: at 14.93 GiB (0.80 GiB unaccounted); the streaming form itemizes 3.23
#: GiB and peaked at 4.56 GiB (0.91 GiB unaccounted).  Against a 1.50 GiB
#: forcing time that is 0.53x and 0.61x -- additive and stable, which is
#: what a per-call transient should be.  0.65 is the margin over both.
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
    the forecast and costing less than half of it.

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
        forcing time at a time, keeps two, and drops the rest.
        """
        return (self.alloc_estimate_bytes + self.context_bytes
                + self.device_overhead_bytes)


def estimate_ingest(exp: ExperimentConfig, *, source: str,
                    forcing_interval_seconds: float
                    = DEFAULT_FORCING_INTERVAL_SECONDS,
                    vram_gib: float | None = None,
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

    @property
    def ingest_priced(self) -> bool:
        return self.ingest_envelope_bytes is not None

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
        """One sentence naming the binding phase and its number."""
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


def estimate_phases(exp: ExperimentConfig, *, source: str,
                    column_chunk: int | None = None,
                    forcing_interval_seconds: float
                    = DEFAULT_FORCING_INTERVAL_SECONDS,
                    ingest_forcing_interval_seconds: float | None = None,
                    vram_gib: float | None = None,
                    profile: DeviceLocalMemoryProfile | None = None,
                    ) -> PhaseMemoryEstimate:
    """Price every phase of ``exp`` and say which one binds the card.

    ``ingest_forcing_interval_seconds`` defaults to the SOURCE's own
    fetch cadence rather than the forecast's LBC interval, because the
    ingest phase's time count is set by what was downloaded.
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
            vram_gib=vram_gib)
    return PhaseMemoryEstimate(
        forecast=forecast, ingest=ingest,
        forecast_envelope_bytes=forecast.peak_envelope_bytes,
        ingest_envelope_bytes=(None if ingest is None
                               else ingest.peak_envelope_bytes),
        source=key,
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
        CAL_D01_POOL_USED_PEAK_BYTES
        + _workspace_total_bytes(49, DEFAULT_COLUMN_CHUNK)))
    return max(0, CAL_D01_POOL_HELD_BYTES - meta_free_basis)


def _workspace_total_bytes(nz: int, column_chunk: int,
                           p_top: float = 10000.0) -> int:
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


def _materialize_physics(state, cfg: RunConfig, start_time: datetime):
    """initialize_physics + the steady-state extras the first steps would
    allocate (composed rqr/rqi/rqs, Morrison accumulator optionals, KF W0AVG),
    so the alloc proof covers the run's real persistent driver set."""
    import cupy as cp
    import numpy as np
    from gpuwm.core.physics import (initialize_physics,
                                    physics_retains_ysu_output,
                                    physics_reuses_pbl_composition)
    from gpuwm.core.state import DTYPE

    ny, nx = cfg.ny, cfg.nx
    grid = np.zeros((ny, nx), dtype=np.float64)
    driver = initialize_physics(
        state, cfg, landmask=1.0, tsk=290.0, soil_temperature=285.0,
        soil_moisture=0.30, radiation_start_time=start_time,
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
    if cfg.bl_pbl_physics and cfg.mp_physics in (6, 8, 10, 18, 28):
        # The materialization side of the pbl_tendencies/rqi budget above.
        # 28 belongs for the same reason: Registry/Registry.EM_COMMON:3036
        # gives the thompsonaero package qi in moist, so WRF's F_QI is true.
        # This set and physics._pbl_optional_tendency_components must agree,
        # or the --alloc measurement stops covering true runtime residency.
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
            from gpuwm.core.physics import physics_driver_required
            if physics_driver_required(dc.run):
                _materialize_physics(state, dc.run, exp.start_time)
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


def config_forcing_source(path: Path) -> str | None:
    """The forcing product a config records, or None if it records none.

    The preprocessing phase is priced against the SOURCE's level count
    and field inventory, so an unpriceable source has to be said out
    loud rather than silently reported as "the forecast is the whole
    story" -- which is exactly how a domain sized to a 12 GB card came
    to die in preprocessing after the download.
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
        if source in SOURCE_ANALYSIS_LEVELS:
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

    try:
        import cupy as cp

        return local_memory_profile_from_device(cp)
    except Exception:
        return None


#: What :func:`device_memory_probe_subprocess` runs in its short-lived
#: interpreter: both device questions -- the free/total VRAM the budget
#: subtracts from, and the local-memory profile the non-pool terms are
#: priced against -- answered in one process that then exits.  Any
#: failure (no cupy, no device, a wedged driver) is exit 3 with nothing
#: on stdout, which the parent reads as "no card here".
_DEVICE_MEMORY_PROBE_SOURCE = """\
import json
import sys

try:
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    payload = {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "profile": {
            "name": (name.decode() if isinstance(name, bytes)
                     else str(name)),
            "multiprocessor_count": int(props["multiProcessorCount"]),
            "max_threads_per_multiprocessor": int(
                props["maxThreadsPerMultiProcessor"]),
            "default_stack_limit_bytes": int(
                cp.cuda.runtime.deviceGetLimit(0)),
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
    """

    import subprocess

    runner = subprocess.run if run is None else run
    try:
        completed = runner(
            [sys.executable, "-c", _DEVICE_MEMORY_PROBE_SOURCE],
            capture_output=True, text=True,
            timeout=DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except ValueError:
        return None
    free = payload.get("free_bytes") if isinstance(payload, dict) else None
    if not isinstance(free, int) or isinstance(free, bool):
        return None
    return payload


def profile_from_device_probe(payload) -> DeviceLocalMemoryProfile | None:
    """The probe payload's device-profile half, typed, or ``None``.

    ``None`` falls back exactly like :func:`live_device_local_memory_profile`
    returning ``None``: the callers price against the reference profile,
    which over-prices rather than under-prices.
    """

    profile = payload.get("profile") if isinstance(payload, dict) else None
    if not isinstance(profile, dict):
        return None
    try:
        return DeviceLocalMemoryProfile(
            name=str(profile["name"]),
            multiprocessor_count=int(profile["multiprocessor_count"]),
            max_threads_per_multiprocessor=int(
                profile["max_threads_per_multiprocessor"]),
            default_stack_limit_bytes=int(
                profile["default_stack_limit_bytes"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


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
    ingest_source = config_forcing_source(args.config)
    phases = estimate_phases(
        exp, source=ingest_source, column_chunk=chunk,
        forcing_interval_seconds=args.forcing_interval_s,
        vram_gib=card_total_gib, profile=profile)
    if not phases.ingest_priced:
        phases = None
    #: The envelope every verdict below compares: the largest phase, not
    #: whichever phase happens to be modelled.
    envelope = (forecast_envelope if phases is None
                else phases.peak_envelope_bytes)
    binding_phase = "forecast" if phases is None else phases.binding_phase
    #: The report's own prose says this configuration may not fit.  It is
    #: read here, before either renderer, because the exit code has to
    #: carry it whether or not anybody reads the text.
    envelope_over_budget = budget is not None and envelope > budget
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
            "observed_peak_envelope_factor":
                peak_envelope_factor(vram_gib=card_total_gib),
            "observed_peak_envelope_basis": PEAK_ENVELOPE_BASIS[
                envelope_platform(vram_gib=card_total_gib)],
            "observed_peak_envelope_bytes": forecast_envelope,
            "non_pool_device_bytes": estimate.non_pool_device_bytes,
            "envelope_unmodelled_bytes": ENVELOPE_UNMODELLED_BYTES,
            "envelope_per_nest_fraction": ENVELOPE_PER_NEST_FRACTION,
            "envelope_basis": ENVELOPE_AFFINE_BASIS,
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
            "kernel_local_memory_bytes": kernel_local_memory_bytes(exp),
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
            "observed_peak_envelope_exceeds_budget": (
                None if budget is None else envelope_over_budget),
            "gates": gates,
        }
        if phases is None:
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
            payload["phase_verdict"] = phases.verdict(budget)
        advisories = check_advisories(exp, args.config)
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
        for advisory in check_advisories(exp, args.config):
            print(f"  {advisory}")
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
        widest = kernel_local_memory_bytes(exp)
        # Kept out of the f-string: a line break inside an f-string
        # expression is PEP 701 (Python 3.12+) syntax, and the supported
        # floor is 3.11 -- the 1.2.0 release workflow failed on exactly
        # this line before any wheel was published.
        remeasured_bytes = (estimate.alloc_estimate_bytes + widest
                            + CUDA_CONTEXT_BYTES
                            + reserve.retention_residual_bytes)
        print(f"  NON-POOL: CUDA context "
              f"{_format_bytes(CUDA_CONTEXT_BYTES)} + local-memory backing "
              f"store {_format_bytes(widest)} "
              f"({len(physics_kernel_modules(exp))} kernel modules selected)"
              f"; RE-MEASURED device-footprint projection "
              f"{_format_bytes(remeasured_bytes)}"
              " (the TIER 3 line above is the retired zero-step probe model)")
        family = envelope_platform(vram_gib=card_total_gib)
        if family == "windows":
            provenance = (
                "the one machine-peak-instrumented Windows run (Thompson "
                "rematch 2026-07-28) peaked at 1.746x its footprint "
                "projection machine-wide (CuPy pool retention + write "
                "transients the projection does not model), so the WDDM "
                "lane keeps that multiplier as a FLOOR under the affine "
                "form and this envelope is the larger of the two")
        else:
            provenance = (
                "affine, not a multiplier: a multiplier with no intercept "
                "under-predicts small configurations and over-predicts "
                "large ones, which is what the 16 GiB fleet node measured "
                "(3.99 GiB declared, 4.38 measured; 19.95 declared, 13.88 "
                "measured).  The intercept is the non-pool line above, "
                "which scales with the device and the kernel set, not "
                "with the grid")
        print(f"  FORECAST PEAK ENVELOPE ({estimate.peak_envelope_terms()}"
              f"; {estimate.envelope_basis}): "
              f"{_format_bytes(forecast_envelope)} -- {provenance}.")
        if phases is None:
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
            print(f"  BINDING PHASE: {phases.verdict(budget)}.")
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
            print(f"  ESTIMATE FOR HARDWARE NOT PRESENT: the free figure "
                  f"above is declared, not measured -- this preflight is "
                  f"sizing a card that is not in this machine.  Non-pool "
                  f"terms are priced against the conservative measured "
                  f"reference device profile ({profile.name}, "
                  f"{profile.multiprocessor_count} SMs), the largest "
                  f"known-device intercept, so the estimate is never more "
                  f"optimistic than a present-card measurement; verify "
                  f"with `gpuwm check` on the real card before trusting "
                  f"the margin.")
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
            print(f"  WARNING: observed peak envelope "
                  f"{_format_bytes(envelope)} exceeds the {budget_word} "
                  f"{_format_bytes(budget)} by "
                  f"{_format_bytes(envelope - budget)}.  The gates above "
                  f"compare the itemized estimate; the envelope is what "
                  f"the machine is measured to reach, so this "
                  f"configuration may run out of budget even though the "
                  f"estimate gate passes -- trim the configuration or "
                  f"free VRAM before trusting the pass.  ({code_note}.)")
        if budget is not None and estimate.alloc_estimate_bytes > budget:
            lever = recommend_column_chunk(exp, budget)
            if lever:
                print("  OVER BUDGET; first lever (RRTMGP column_chunk): "
                      f"--column-chunk {lever}")
            else:
                # It used to end "staged residency (DESIGN REOPEN) per
                # section E".  No pip user has a section E, and the
                # sentence names no action; the actionable one already
                # exists one layer up, in `gpuwm go`'s refusal.
                # ``--vram-gib`` names the CARD, not the budget, so the
                # number in the remedy is the free VRAM this preflight
                # just used -- the same arithmetic `gpuwm go`'s refusal
                # prints, and the same flag.
                card_gib = max(1, int((free or budget) / GIB))
                print("  OVER BUDGET, and the RRTMGP column_chunk lever "
                      "cannot close it: no chunk halving fits after the "
                      "shared-scratch arena, so the grid itself is what "
                      "has to come down.")
                print(f"  remedy: re-size for this card -- gpuwm domain "
                      f"--vram-gib {card_gib} ... -- or free VRAM and "
                      f"re-run")
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
    "KERNEL_MAX_LOCAL_SIZE_BYTES", "MEASURED_LOCAL_MEMORY_PROFILE",
    "CHAINED_TRANSLATION_UNIT_FRAMES", "ChainedTranslationUnitFrame",
    "UNMEASURED_KERNEL_MODULES", "local_memory_profile_from_device",
    "cap_free_to_physical", "device_physical_total_bytes",
    "device_rail_free_bytes", "device_wide_used_bytes",
    "kernel_local_memory_bytes", "non_pool_device_bytes",
    "physics_kernel_modules", "refl_diagnostic_reachable",
    "LEVEL_SPECIALIZED_KERNEL_FRAMES", "LevelSpecializedFrame",
    "ACOUSTIC_TIER_FRAME", "TieredKernelFrame",
    "domain_kernel_modules", "kernel_local_frame_bytes",
    "ALLOCATOR_HEADROOM", "AllocReport", "CAL_D01_DEVICE_FOOTPRINT_BYTES",
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
    "WINDOWS_SMALL_CARD_MAX_GIB", "WINDOWS_SMALL_CARD_RESERVE_BYTES",
    "windows_small_card_advisory",
    "ysu_output_transient_shapes",
    "ENVELOPE_AFFINE_BASIS", "ENVELOPE_PER_NEST_FRACTION",
    "ENVELOPE_UNMODELLED_BYTES", "CARD_CLASS_MULTIPROCESSORS",
    "card_local_memory_profile", "live_device_local_memory_profile",
    "machine_peak_envelope_bytes", "observed_peak_envelope_bytes",
    "peak_envelope_factor", "PEAK_ENVELOPE_FACTORS",
    "PEAK_ENVELOPE_BASIS", "envelope_platform", "estimate_ingest",
    "estimate_phases", "IngestMemoryEstimate", "PhaseMemoryEstimate",
]
