"""How fast this plan will actually run, said BEFORE it runs.

THE HOLE THIS FILLS.  Every memory figure this package publishes is
careful, itemized and provenanced, and none of them is the number a user
acts on.  A 399,119-column domain on a 10 GiB card is priced correctly,
sent down the streamed road correctly, and started -- and nothing
anywhere says what a streamed step at that size COSTS, so a three-hour
forecast that is going to take hours of wall clock looks, from the
outside, exactly like a stall.  A plan document that answers "will it
fit" and refuses to answer "how long" has answered the easier half of
the question.

NOTHING HERE IS INVENTED.  Every rate is a measurement this repository
already carries, with the card, the grid and the timestep named; where a
figure is a bound rather than a measurement its ``basis`` says
``unmeasured-bound`` in those words, and :attr:`StepRate.measured` is
the field a caller branches on.

THE MEASUREMENTS
----------------
RESIDENT, full physics (Thompson/Morrison + RTE-RRTMGP + YSU + Noah +
Kain-Fritsch), per COLUMN per step at nz=49:

* ``evidence/grounding-3km-conus-run-report.json`` -- 796x636x49 at
  3 km, dt 15 s, 1440 steps in 1366.03 s on an RTX 5090 (Linux, sole
  occupant confirmed) = 0.9486 s/step = **1.874e-6 s per column-step**.
  This is the best-loaded measured run in the fleet and it is the FAST
  end.
* ``docs/public/HARDWARE.md`` -- 438x352x49 at 12 km, dt 60 s, 360 steps
  in 400 s on an RTX 4090 (Linux) = 1.111 s/step = **7.21e-6 s per
  column-step**.  A slower card AND a radiation-heavy cadence (at dt
  60 s a 12-minute ``radt`` fires every 12 steps instead of every 48),
  which is why it is the SLOW end.

STREAMED, full physics, per column per step:

* ``tilestream/HANDOFF-case-imagery.md`` -- a real HRRR case,
  1200x900x49 at 3 km, dt 15 s, tile 400x300 with halo 16 and 2
  buffers, on a single RTX 4090: **11.66 s/step measured**, with 9.6
  s/step on a less contended box and 22.6 s/step with a second forecast
  on the same card.  At 1,080,000 columns that is **8.89e-6 to 2.09e-5 s
  per column-step**.  The same receipt records that the design "moves
  ~27 GB of pinned host RAM per step" and a redundancy of 1.1952x, and
  :func:`streamed_transfer_bytes_per_step` reproduces BOTH from the
  tiling alone -- 27.1 GB and 1.1952x -- which is what licenses using
  that byte model on tilings nobody has timed.

THE THREE EFFECTS THAT MOVE A RATE, all of them measured here:

* **Card.** The identical speedrun grid measured 7.885e-6 s per
  column-step on an RTX 5070 Ti (Linux) and 1.381e-5 on an RTX 3080
  (Windows) -- **1.75x** -- and the 1.5.0 envelope pair measured 7.17e-7
  against 1.231e-6, **1.72x**, independently.  :data:`CARD_SPREAD`
  carries it, and it already contains the Windows/WDDM platform gap
  because the slow member of each pair is the Windows box.
* **Radiation cadence.** Radiation is 69% of a reference step (486 ms of
  it, ``docs/lead-handoff-2026-07-26.md``), and it fires on a wall-clock
  period, so a longer ``dt`` amortizes it over fewer steps and every
  step costs more.  The bracket spans two measured cadences rather than
  modelling this.
* **Launch-bound floor.** Below roughly 100,000 columns the card is not
  loaded and the per-column rate degrades sharply -- measured 1.11e-5 at
  250x200 against 1.874e-6 at 796x636 on the same card class
  (``docs/da-vs-wofs.md``: "small enough to be launch-bound").
  :data:`LAUNCH_BOUND_COLUMNS` is where the sentence says so.

Session-to-session variance on one box is up to 30% (stated in
``docs/public/HARDWARE.md`` and twice more), which is a further reason
every figure here is a bracket and none is presented as exact.

THE COLUMN BOUND.  :func:`resident_column_limit` is the largest column
count whose whole peak -- the domain, the per-process fixed cost, the
CUDA context and the rung's measured RRTMGP transient reservation --
still fits the card's allowance.  It is inverted from
``tilestream.autoplan``'s own cost model and checked against
``autoplan.plan`` at the boundary in the tests, because a report that
advised a size the planner then streamed anyway would be two models of
one card.  It is the actionable number: the memory report says the
current domain does not fit, and this says which one would be fast.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

GIB = 1 << 30


class _Unpriced:
    """'Nobody has resolved the road yet', distinct from 'resident'.

    ``streamed=None`` is a positive statement -- this plan runs
    resident -- so it cannot also mean 'not asked'.  Conflating them
    made a bare ``pace_advisory(exp)`` answer for the RESIDENT road on
    a config whose [tiles] table streams, which is the exact class of
    silent-wrong-road defect this module exists to end.
    """

    def __repr__(self) -> str:              # pragma: no cover
        return "<unpriced>"


UNPRICED = _Unpriced()

#: Every rate below is quoted per column at this level count, and scales
#: LINEARLY in ``nz``.  That is measured, not assumed: the P2 LES sweep
#: (``docs/superpowers/receipts/les/p2-nz-tier-2026-08-04``) ran nz =
#: 64/96/128/160/192 on one card and ``steps/s x nz`` came out 8,743 /
#: 9,146 / 9,102 / 9,042 / 8,915 -- 4.5% peak to peak across a 3x range.
#: "Throughput scales as the level count and no faster", in the
#: receipt's own words.
REFERENCE_NZ = 49

#: Spread between the fastest and slowest cards this fleet has measured
#: on IDENTICAL work, and it is two independent measurements of the same
#: number: the speedrun's 178x144x49 course ran 7.885e-6 s per
#: column-step on an RTX 5070 Ti (Linux) and 1.381e-5 on an RTX 3080
#: (Windows), 1.75x; the 1.5.0 operational envelope measured 7.17e-7 on
#: an RTX 5090 against 1.231e-6 on an RTX 5070 Ti, 1.72x.  The
#: Windows/WDDM platform gap is INSIDE this figure, not on top of it.
CARD_SPREAD = 1.75

#: Below this many columns the card stops being loaded and the
#: per-column rate degrades well outside the bracket -- measured 1.11e-5
#: s per column-step at 250x200 against 1.874e-6 at 796x636 on the same
#: card class.  The pace sentence says so rather than quoting a bracket
#: that does not apply.
LAUNCH_BOUND_COLUMNS = 100_000


@dataclass(frozen=True)
class StepRate:
    """Seconds per COLUMN per model step, at :data:`REFERENCE_NZ`.

    ``measured`` is the field a caller branches on and the fact a reader
    is owed: a rung this project has never timed carries a bound derived
    from the rungs on either side of it, and says so in both places.
    """

    rung: str
    road: str
    low: float
    high: float
    reference_card: str
    basis: str
    measured: bool = True

    def seconds_per_step(self, columns: int, nz: int) -> tuple[float, float]:
        """The bracket, in seconds, for ``columns`` columns at ``nz``."""
        scale = float(columns) * float(nz) / REFERENCE_NZ
        return (self.low * scale, self.high * scale)


_RESIDENT_FULL_BASIS = (
    "MEASURED full-physics resident forecasts, per column-step at nz=49: "
    "the fast end is 1.874e-6 from the 3 km CONUS grounding run "
    "(796x636x49, dt 15 s, 1440 steps in 1366.0 s on an RTX 5090 under "
    "Linux, sole occupant confirmed -- evidence/"
    "grounding-3km-conus-run-report.json), and the slow end is 7.21e-6 "
    "from 438x352x49 at 12 km, dt 60 s, 360 steps in 400 s on an RTX "
    "4090 under Linux (docs/public/HARDWARE.md).  The bracket spans a "
    "card gap AND a radiation-cadence gap: radiation fires on a "
    "wall-clock period and is 69% of a reference step, so dt 60 s "
    "amortizes it over a quarter as many steps as dt 15 s does")

_RESIDENT_MYNN_BASIS = (
    _RESIDENT_FULL_BASIS + ".  Scaled by the MEASURED 1.86x MYNN + "
    "Noah-MP premium: the d04 reference step is 0.699 s amortized and "
    "the MYNN 5/5 step is 1.30 s on the same domain "
    "(docs/lead-handoff-2026-07-26.md), which is a measurement of the "
    "RUNG and not of this grid")

_RESIDENT_DRY_BASIS = (
    "MEASURED monolithic resident dry steps at nz=49 on an RTX 5090 "
    "(tilestream/RESULTS.md s1 and s12.7): 3.325 ns/cell at 1448^2, "
    "3.433 at 1536^2, 3.712 at 1950^2, which is 1.63e-7 to 1.82e-7 s per "
    f"column-step; the high end carries the {CARD_SPREAD:.2f}x measured "
    "card spread.  A dry rate is a TRANSPORT DIAGNOSTIC "
    "(tilestream/NO-DRY-NUMBERS.md) and no forecast a user runs is dry")

_RESIDENT_MOIST_BASIS = (
    "unmeasured-bound: this package has never timed a moisture-only "
    "forecast step, so the bracket is bounded BELOW by the measured dry "
    "rate and ABOVE by the measured full-physics one.  It is a bound and "
    "not a measurement, and it is wide because those two rungs are an "
    "order of magnitude apart.  Timing one mp10 step at a known column "
    "count closes it")

_STREAMED_FULL_BASIS = (
    "MEASURED streamed full-physics forecast: a real HRRR case at "
    "1200x900x49, 3 km, dt 15 s, tile 400x300 with halo 16 and 2 "
    "buffers on a single RTX 4090 ran 11.66 s/step, with 9.6 s/step "
    "measured on a less contended box and 22.6 s/step with a second "
    "forecast sharing the card (tilestream/HANDOFF-case-imagery.md).  At "
    "1,080,000 columns that is 8.89e-6 to 2.09e-5 s per column-step.  "
    "The same receipt records ~27 GB of pinned host RAM moved per step "
    "and a redundancy of 1.1952x, and this module's byte model "
    "reproduces both from the tiling alone (27.1 GB, 1.1952x), which is "
    "what licenses pricing a tiling nobody has timed.  A second MEASURED "
    "point sets the fast end: a 399,119-column 1 km HRRR case at dt 5 s "
    "on an RTX 3080 under Windows/WDDM ran 2.398 s/step median over 251 "
    "settled steps through the real run-plan door -- 6.01e-6 s per "
    "column-step (evidence/streamed-pace-3080-20260824.md)")

_STREAMED_MYNN_BASIS = (
    _STREAMED_FULL_BASIS + ".  Scaled by the MEASURED 1.86x MYNN + "
    "Noah-MP premium (docs/lead-handoff-2026-07-26.md)")

_STREAMED_TAX_BASIS = (
    "unmeasured-bound at this rung: no streamed forecast has been timed "
    "below full physics, so the resident rate is multiplied by the "
    "MEASURED dry tiling tax -- 1.05x at the headline out-of-core size "
    "(tilestream/RESULTS.md s1) up to 1.359x at the worst tile size in "
    "the 1024^2 sweep (docs/manual/06-pipeline.md).  The transfer floor "
    "below is priced from bytes either way, so a tiling that is actually "
    "bus-bound is not quoted a compute-bound answer")

#: The MEASURED premium of the MYNN + Noah-MP rung over the reference
#: physics bundle: 1.30 s amortized against 0.699 s on the same d04
#: domain and card (``docs/lead-handoff-2026-07-26.md``).
MYNN_PREMIUM = 1.86

#: The measured dry tiling tax, used only where no streamed forecast has
#: been timed at the rung: 1.05x at the headline out-of-core size,
#: 1.359x at the worst tile size of the 1024^2 sweep.
STREAM_TAX_LOW, STREAM_TAX_HIGH = 1.05, 1.359

_RESIDENT_FULL = (1.874e-6, 7.21e-6)
_STREAMED_FULL = (6.01e-6, 2.09e-5)
_RESIDENT_DRY = (1.63e-7, 1.82e-7 * CARD_SPREAD)

#: Keyed by ``tilestream.autoplan.rung_of`` and by road, so every road
#: the planner can choose has a row.  A missing row would answer ``None``
#: where a pace is due, which is the silence this module exists to end.
STEP_RATES: dict[tuple[str, str], StepRate] = {
    ("dry", "resident"): StepRate(
        "dry", "resident", *_RESIDENT_DRY, "RTX 5090", _RESIDENT_DRY_BASIS),
    ("moist", "resident"): StepRate(
        "moist", "resident", _RESIDENT_DRY[1], _RESIDENT_FULL[0],
        "RTX 5090 / RTX 4090", _RESIDENT_MOIST_BASIS, measured=False),
    ("full", "resident"): StepRate(
        "full", "resident", *_RESIDENT_FULL, "RTX 5090 (Linux)",
        _RESIDENT_FULL_BASIS),
    ("full+mynn+noahmp", "resident"): StepRate(
        "full+mynn+noahmp", "resident",
        _RESIDENT_FULL[0] * MYNN_PREMIUM, _RESIDENT_FULL[1] * MYNN_PREMIUM,
        "RTX 5090 (Linux)", _RESIDENT_MYNN_BASIS),
    ("dry", "streamed"): StepRate(
        "dry", "streamed", _RESIDENT_DRY[0] * STREAM_TAX_LOW,
        _RESIDENT_DRY[1] * STREAM_TAX_HIGH, "RTX 5090",
        _RESIDENT_DRY_BASIS + ".  " + _STREAMED_TAX_BASIS, measured=False),
    ("moist", "streamed"): StepRate(
        "moist", "streamed", _RESIDENT_DRY[1] * STREAM_TAX_LOW,
        _RESIDENT_FULL[0] * STREAM_TAX_HIGH, "RTX 5090 / RTX 4090",
        _RESIDENT_MOIST_BASIS + ".  " + _STREAMED_TAX_BASIS,
        measured=False),
    ("full", "streamed"): StepRate(
        "full", "streamed", *_STREAMED_FULL, "RTX 3080 / RTX 4090",
        _STREAMED_FULL_BASIS),
    ("full+mynn+noahmp", "streamed"): StepRate(
        "full+mynn+noahmp", "streamed",
        _STREAMED_FULL[0] * MYNN_PREMIUM, _STREAMED_FULL[1] * MYNN_PREMIUM,
        "RTX 4090", _STREAMED_MYNN_BASIS),
}

#: Pinned host<->device bandwidth used for the transfer floor when no
#: probe answers.  Both ends are MEASURED on this fleet: 28.17 GB/s H2D
#: and 28.59 GB/s D2H median on an RTX 5090 at PCIe 4.0 x16
#: (``skeptic-results/duplex_5090.json``), 28.45 GB/s = 90.3% of
#: theoretical on the 4x5070 Ti box
#: (``tilestream/BOX-4X5070TI-BLOCKERS.md``), and 25.8 GB/s near-node /
#: 23.6 GB/s far-node on an RTX 4090 across NUMA
#: (``skeptic-results/numa_4090b.json``).
PCIE_PINNED_BYTES_PER_SECOND_HIGH = int(28.45e9)

#: The slow end: the MEASURED bidirectional figure, which is what a
#: gather and a scatter in the same step actually contend for -- 26.36
#: GB/s median on the RTX 5090 duplex probe, and 10.93 GB/s on a 4090
#: whose duplex collapses against 13.43 H2D
#: (``tilestream/skeptic_duplex.py``).  A degraded link is worse again:
#: ``tilestream/OVERLAP-ATTRIBUTION.md`` measured 2.34 GB/s pinned H2D
#: on a box that had negotiated gen1 x16.
PCIE_PINNED_BYTES_PER_SECOND_LOW = int(10.93e9)

_PCIE_BASIS = (
    f"the transfer floor is priced at "
    f"{PCIE_PINNED_BYTES_PER_SECOND_LOW / 1e9:.1f}-"
    f"{PCIE_PINNED_BYTES_PER_SECOND_HIGH / 1e9:.1f} GB/s of pinned "
    "host<->device bandwidth, both ends MEASURED on this fleet: 28.45 "
    "GB/s H2D at 90.3% of theoretical on PCIe 4.0 x16 "
    "(tilestream/BOX-4X5070TI-BLOCKERS.md) and 28.17/28.59 GB/s H2D/D2H "
    "median on an RTX 5090 (skeptic-results/duplex_5090.json) at the "
    "fast end; the slow end is that box's measured DUPLEX collapse, "
    "10.93 GB/s, which is what a gather and a scatter in one step "
    "actually contend for")


def step_rate(rung: str, road: str = "resident") -> StepRate | None:
    """The rate row for ``rung`` on ``road``, or ``None`` if there is none."""
    return STEP_RATES.get((str(rung), str(road)))


def measured_pinned_bytes_per_second(*, device: int = 0,
                                     nbytes: int = 256 << 20,
                                     reps: int = 3) -> int | None:
    """Time a real pinned round trip on this box, or answer ``None``.

    GRACEFUL ABSENCE, the same discipline every other probe in this
    package keeps: no CuPy, no card, or any refusal from the runtime
    answers ``None`` and the caller falls back to the measured table
    constants with a basis that says which it used.  A probe that raised
    would put a hardware question in front of an estimate whose whole
    purpose is to be answerable before anything is allocated.

    THIS CREATES A CUDA CONTEXT.  It is therefore never called from
    ``run-plan --estimate``, which promises the opposite in
    :func:`gpuwm.runplan.estimate_plan`'s own docstring; a caller that
    already owns a context, or one running out of process, may use it.
    """
    try:
        import time

        import cupy as cp
        import numpy as np

        with cp.cuda.Device(device):
            host = cp.cuda.alloc_pinned_memory(int(nbytes))
            view = np.frombuffer(host, dtype=np.uint8, count=int(nbytes))
            buffer = cp.empty(int(nbytes), dtype=cp.uint8)
            buffer.set(view)                      # warm the path
            cp.cuda.Stream.null.synchronize()
            best = None
            for _ in range(max(1, int(reps))):
                start = time.perf_counter()
                buffer.set(view)
                buffer.get(out=view)
                cp.cuda.Stream.null.synchronize()
                elapsed = time.perf_counter() - start
                if elapsed > 0 and (best is None or elapsed < best):
                    best = elapsed
        if not best:
            return None
        return int(2 * int(nbytes) / best)
    except Exception:
        # Including ImportError, CuPy's own runtime errors, and a driver
        # that refuses to pin.  ``None`` means "not measured here", which
        # is a different and honest answer from a guess.
        return None


def resident_column_limit(cfg, machine, *, footprint=None) -> int | None:
    """The largest column count that still fits the RESIDENT road.

    Inverted from :meth:`tilestream.autoplan.Footprint.resident_bytes`
    against :func:`tilestream.autoplan.budget_for` -- the same two calls
    ``autoplan.plan`` makes to decide resident against tiled -- so the
    advice this produces and the decision the run takes cannot disagree.
    The budget already carries the rung's measured RRTMGP transient as a
    reservation and the price already carries the CUDA context and the
    per-process fixed cost, so the bound is about the whole PEAK and not
    about the domain arrays alone::

        resident_bytes(cells) = (CUDA_CONTEXT + process_fixed
                                 + buffer_fixed + b * cells) * VRAM_SAFETY

    solved for ``cells`` against ``budget_for(machine, fp)`` and divided
    by ``nz``.

    ``None`` when ``machine`` is ``None``: with no allowance there is no
    bound, and a number here would be invented.
    """
    if machine is None:
        return None
    from tilestream import autoplan

    nz = int(getattr(cfg, "nz", 0) or 0)
    if nz <= 0:
        return None
    fp = footprint or autoplan.footprint_for(cfg)
    budget = autoplan.budget_for(machine, fp)
    if budget <= 0 or fp.bytes_per_cell <= 0:
        return None
    cells = ((budget / autoplan.VRAM_SAFETY)
             - autoplan.CUDA_CONTEXT_BYTES
             - fp.process_fixed_bytes
             - fp.buffer_fixed_bytes) / fp.bytes_per_cell
    limit = int(cells // nz)
    if limit <= 0:
        return None
    # Settle the boundary against the pricing itself rather than trusting
    # the float division: what matters is that K fits and K+1 does not,
    # and one step either way is cheaper than a bound that is off by one
    # in the direction that OOMs a forecast.
    while limit > 0 and fp.resident_bytes(limit * nz) > budget:
        limit -= 1
    while fp.resident_bytes((limit + 1) * nz) <= budget:
        limit += 1
    return limit or None


def streamed_transfer_bytes_per_step(envelope, cfg) -> int:
    """Bytes crossing the bus per model step, from the chosen tiling.

    The round trip a streamed sweep actually makes, priced off the tiling
    the stream-init decision SELECTED rather than an idealised one: every
    tile gathers its whole compute window (tile plus a halo on both sides
    of both axes -- the window is what a buffer holds, never the tile),
    and every tile scatters its interior back, which sums to the domain
    exactly once.  ``store_bytes_per_cell`` is the carrier inventory,
    which is what a pinned store holds and is far smaller than the VRAM
    per-cell figure.

    VALIDATED AGAINST A MEASUREMENT, which is the only reason it is
    allowed to price a tiling nobody has timed: for the HRRR case in
    ``tilestream/HANDOFF-case-imagery.md`` -- 1200x900x49, tile 400x300,
    halo 16 -- this returns 27.1 GB and the receipt says the design
    "moves ~27 GB of pinned host RAM per step".
    """
    if envelope is None:
        return 0
    from tilestream import autoplan

    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    fp = autoplan.footprint_for(cfg)
    ntx = -(-nx // max(1, int(envelope.tile_nx)))
    nty = -(-ny // max(1, int(envelope.tile_ny)))
    window_cells = int(envelope.window_nx) * int(envelope.window_ny) * nz
    gather = ntx * nty * window_cells
    scatter = nx * ny * nz
    return int((gather + scatter) * fp.store_bytes_per_cell)


def _domain_steps(domain, run_seconds: float) -> int:
    dt = float(getattr(domain.run, "dt", 0.0) or 0.0)
    if dt <= 0.0:
        return 0
    return max(1, int(math.ceil(float(run_seconds) / dt)))


def _number(value: float) -> str:
    """A figure at the precision it is actually known to.

    Never more than three significant figures, because none of these
    brackets is known to more than two and a long decimal reads as a
    measurement.  A small NONZERO value keeps two significant figures
    rather than rounding to ``0.00``: a sub-centisecond step is a real
    and useful answer, and printing it as zero reads as a broken
    estimator.
    """
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    if value <= 0.0:
        return "0"
    return f"{value:#.2g}"


def _duration(seconds: float) -> str:
    """A wall clock in the unit a reader thinks in.

    Hours for a forecast that takes hours, minutes for one that takes
    minutes.  Quoting "3.5e-06 h" is arithmetically fine and useless: the
    whole point of this sentence is that somebody reads it and decides
    whether to start the run.
    """
    if seconds >= 3600.0:
        return f"{_number(seconds / 3600.0)} h"
    if seconds >= 60.0:
        return f"{_number(seconds / 60.0)} min"
    return f"{_number(seconds)} s"


def _span(low: float, high: float) -> str:
    """``~LOW-HIGH unit``, with the unit named once when both share it."""
    for cut, scale, unit in ((3600.0, 3600.0, "h"), (60.0, 60.0, "min"),
                             (0.0, 1.0, "s")):
        if high >= cut:
            if low >= cut:
                return f"~{_number(low / scale)}-{_number(high / scale)} {unit}"
            # The two ends straddle a unit boundary, so each carries its
            # own rather than rounding the small one to zero of the big.
            return f"~{_duration(low)}-{_duration(high)}"
    return f"~{_number(low)}-{_number(high)} s"


@dataclass(frozen=True)
class PaceEstimate:
    """The pace bracket, and everything a reader needs to judge it."""

    road: str
    seconds_per_step_low: float
    seconds_per_step_high: float
    wall_seconds_low: float
    wall_seconds_high: float
    steps: int
    resident_column_limit: int | None
    resident_seconds_per_step_low: float
    resident_seconds_per_step_high: float
    transfer_bytes_per_step: int
    transfer_seconds_per_step_low: float
    transfer_seconds_per_step_high: float
    columns: int
    measured: bool
    reference_card: str
    basis: str
    run_seconds: float

    @property
    def realtime_ratio_low(self) -> float:
        """Simulated seconds per wall second, at the SLOW end.

        Paired with ``wall_seconds_high`` deliberately: a document that
        put the fast wall against the low ratio would read as a run being
        fastest exactly when it is slowest.
        """
        return (0.0 if self.wall_seconds_high <= 0.0
                else self.run_seconds / self.wall_seconds_high)

    @property
    def realtime_ratio_high(self) -> float:
        return (0.0 if self.wall_seconds_low <= 0.0
                else self.run_seconds / self.wall_seconds_low)

    def sentence(self) -> str:
        """One sentence naming BOTH roads, because both are actionable.

        The road the plan is on tells the reader what they are about to
        wait for; the column bound tells them what to change.  A sentence
        with only the first half is a complaint.
        """
        head = (f"{self.road} road: expect roughly "
                f"{_number(self.seconds_per_step_low)}-"
                f"{_number(self.seconds_per_step_high)} s per model step "
                f"({_span(self.wall_seconds_low, self.wall_seconds_high)} "
                f"wall for this {_duration(self.run_seconds)} forecast) on "
                f"this card")
        if self.resident_column_limit is None:
            tail = ("; the resident road's column bound needs a card to "
                    "price against and none was readable here")
        else:
            tail = (f"; the resident road at <= "
                    f"{self.resident_column_limit:,} columns runs about "
                    f"{_number(self.resident_seconds_per_step_low)}-"
                    f"{_number(self.resident_seconds_per_step_high)} s/step")
        if self.columns and self.columns < LAUNCH_BOUND_COLUMNS:
            tail += (f".  Under {LAUNCH_BOUND_COLUMNS:,} columns the card is "
                     "not loaded and the per-column rate degrades outside "
                     "this bracket, so treat it as an upper bound only")
        return head + tail

    def to_json(self) -> dict:
        """The document ``run-plan --estimate`` and ``gpuwm check`` carry.

        Key names are a published surface: a front end renders them
        verbatim, so they are added to and never renamed.
        """
        return {
            "road": self.road,
            "seconds_per_step_low": round(self.seconds_per_step_low, 6),
            "seconds_per_step_high": round(self.seconds_per_step_high, 6),
            "wall_seconds_low": round(self.wall_seconds_low, 3),
            "wall_seconds_high": round(self.wall_seconds_high, 3),
            "realtime_ratio_low": round(self.realtime_ratio_low, 6),
            "realtime_ratio_high": round(self.realtime_ratio_high, 6),
            "steps": self.steps,
            "columns": self.columns,
            "resident_column_limit": self.resident_column_limit,
            "resident_seconds_per_step_low": round(
                self.resident_seconds_per_step_low, 6),
            "resident_seconds_per_step_high": round(
                self.resident_seconds_per_step_high, 6),
            "transfer_bytes_per_step": self.transfer_bytes_per_step,
            "transfer_seconds_per_step_low": round(
                self.transfer_seconds_per_step_low, 6),
            "transfer_seconds_per_step_high": round(
                self.transfer_seconds_per_step_high, 6),
            "launch_bound_columns": LAUNCH_BOUND_COLUMNS,
            "measured": self.measured,
            "reference_card": self.reference_card,
            "basis": self.basis,
            "sentence": self.sentence(),
        }


def estimate_pace(exp, *, streamed=UNPRICED, machine=None,
                  pinned_bytes_per_second: int | None = None
                  ) -> PaceEstimate | None:
    """The pace bracket for ``exp``, on the road it will actually take.

    ``streamed`` is the :class:`gpuwm.core.streaming.StreamedEnvelope`
    the caller already priced -- ``None`` meaning the resident road,
    and OMITTED meaning nobody has resolved it, in which case this
    resolves it from the config so a bare call never answers for the
    wrong road --
    and is taken rather than re-derived so the pace describes the tiling
    the memory report quoted.  Under ``[tiles] mode = "auto"`` those two
    genuinely disagree when the card's occupancy moves between the calls,
    which is the same reason
    :func:`gpuwm.core.preflight.streaming_advisory` accepts one.

    ``machine`` is the allowance the column bound is computed against.
    When it is ``None`` the configured ``[tiles] vram_budget_bytes`` is
    used if there is one -- that is exactly what
    :func:`gpuwm.core.streaming.decide` does with the key -- and
    otherwise the bound is ``None`` rather than a guess.

    ``None`` when the experiment carries no domain to price.
    """
    domains = list(getattr(exp, "domains", ()) or ())
    if not domains:
        return None
    from tilestream import autoplan

    if streamed is UNPRICED:
        # Resolve it rather than defaulting to the cheap road.  Never
        # raises: streamed_forecast_envelope answers None for a config
        # that cannot be priced here, which IS the resident answer.
        from gpuwm.core.preflight import streamed_forecast_envelope

        streamed = streamed_forecast_envelope(exp, machine=machine)
    root = domains[0]
    cfg = root.run
    road = "resident" if streamed is None else "streamed"
    rung = autoplan.rung_of(cfg)
    rate = step_rate(rung, road)
    if rate is None:
        return None
    run_seconds = float(getattr(exp, "run_seconds", 0.0) or 0.0)
    machine = machine if machine is not None else _configured_machine(exp)
    nz = int(cfg.nz)
    columns = int(cfg.nx) * int(cfg.ny)

    step_low, step_high = rate.seconds_per_step(columns, nz)
    parts = [f"{rung} rung on the {road} road: {rate.basis}"]

    # ------------------------------------------------------ the bus floor
    transfer_bytes = transfer_low = transfer_high = 0.0
    if streamed is not None and not _store_is_device(exp):
        transfer_bytes = streamed_transfer_bytes_per_step(streamed, cfg)
        fast = pinned_bytes_per_second or PCIE_PINNED_BYTES_PER_SECOND_HIGH
        slow = pinned_bytes_per_second or PCIE_PINNED_BYTES_PER_SECOND_LOW
        transfer_low = transfer_bytes / fast
        transfer_high = transfer_bytes / slow
        # A FLOOR, never a replacement.  The measured streamed rate above
        # is a whole-step figure and already contains the transfer it was
        # measured with; this only raises the answer when THIS tiling
        # moves more bytes per column than the measured one did, which is
        # exactly the case a small card produces (small tiles, more
        # tiles, more halo per useful cell).
        step_low = max(step_low, transfer_low)
        step_high = max(step_high, transfer_high)
        parts.append(
            f"this plan's tiling ({streamed.tile_nx}x{streamed.tile_ny} + "
            f"halo {streamed.halo}, {streamed.nbuffers} buffer(s)) moves "
            f"{transfer_bytes / GIB:.2f} GiB per step, so the bus alone "
            f"costs {_number(transfer_low)}-{_number(transfer_high)} s per "
            f"step and the quoted pace is never below it; {_PCIE_BASIS}")
    elif streamed is not None:
        parts.append(
            "the store is on the DEVICE, so nothing crosses the bus and "
            "the only streaming cost is the tiling tax")

    # ---------------------------------------------------- the whole plan
    steps = _domain_steps(root, run_seconds)
    wall_low, wall_high = step_low * steps, step_high * steps
    for domain in domains[1:]:
        # Every NEST runs resident: the streamed envelope is priced for
        # the root alone (preflight.streamed_forecast_envelope), so a
        # tree's wall clock is the root's road plus resident nests.
        nest_rate = step_rate(autoplan.rung_of(domain.run), "resident")
        if nest_rate is None:
            continue
        nest_low, nest_high = nest_rate.seconds_per_step(
            int(domain.run.nx) * int(domain.run.ny), int(domain.run.nz))
        nest_steps = _domain_steps(domain, run_seconds)
        wall_low += nest_low * nest_steps
        wall_high += nest_high * nest_steps

    # --------------------------------------------------- the column bound
    limit = resident_column_limit(cfg, machine)
    resident_rate = step_rate(rung, "resident") or rate
    if limit is None:
        resident_low = resident_high = 0.0
        parts.append(
            "no VRAM allowance was readable at estimate time, so the "
            "resident column bound is omitted rather than guessed; declare "
            "one with [tiles] vram_budget_bytes, or ask `gpuwm check`, "
            "which measures the card")
    else:
        resident_low, resident_high = resident_rate.seconds_per_step(
            limit, nz)
        parts.append(
            "the resident column bound is inverted from the same "
            "tilestream.autoplan pricing the stream/resident decision uses "
            "-- Footprint.resident_bytes against budget_for, so the CUDA "
            "context, the per-process fixed cost and the rung's measured "
            "RRTMGP transient reservation are all inside it -- and is the "
            "largest column count whose whole peak still fits this card")
    if pinned_bytes_per_second:
        parts.append(
            f"the transfer rate is a PROBED "
            f"{pinned_bytes_per_second / 1e9:.1f} GB/s on this box, not the "
            "table constant")
    parts.append(
        f"rates are per column at nz={REFERENCE_NZ} and scale linearly in "
        "nz (MEASURED: steps/s x nz is constant to 4.5% over nz 64-192, "
        "docs/superpowers/receipts/les/p2-nz-tier-2026-08-04).  One box "
        "varies up to 30% between sessions, so this is a BRACKET and never "
        "an exact figure")
    return PaceEstimate(
        road=road,
        seconds_per_step_low=step_low, seconds_per_step_high=step_high,
        wall_seconds_low=wall_low, wall_seconds_high=wall_high,
        steps=steps, resident_column_limit=limit,
        resident_seconds_per_step_low=resident_low,
        resident_seconds_per_step_high=resident_high,
        transfer_bytes_per_step=int(transfer_bytes),
        transfer_seconds_per_step_low=transfer_low,
        transfer_seconds_per_step_high=transfer_high,
        columns=columns, measured=rate.measured,
        reference_card=rate.reference_card,
        basis="; ".join(part for part in parts if part),
        run_seconds=run_seconds)


def _store_is_device(exp) -> bool:
    options = getattr(exp, "tiles", None)
    return str(getattr(options, "store", "host")) == "device"


def _configured_machine(exp):
    """A ``Machine`` from ``[tiles] vram_budget_bytes``, or ``None``.

    The same substitution :func:`gpuwm.core.streaming.decide` makes with
    that key: the configured number IS the budget, so the headroom
    multiplier is dropped rather than applied on top of it -- applying it
    is the defect that turned every declared budget into a smaller one.
    Host RAM is irrelevant to the column bound and is filled from the
    declared pinned budget purely so the dataclass is complete.
    """
    options = getattr(exp, "tiles", None)
    budget = getattr(options, "vram_budget_bytes", None)
    if budget is None:
        return None
    from tilestream import autoplan

    host = getattr(options, "host_budget_bytes", None) or int(budget)
    return autoplan.Machine(
        vram_bytes=int(budget), host_bytes=int(host),
        name="[tiles] vram_budget_bytes", vram_headroom=0.0,
        pinned_fraction=1.0, host_source="explicit")


def pace_advisory(exp, *, streamed=UNPRICED, machine=None) -> str | None:
    """The pace sentence for ``gpuwm check``'s text surface.

    Advisory, never a gate: it changes no exit code and blocks nothing,
    on the same posture as every other entry in
    :func:`gpuwm.core.preflight.check_advisories`.  It exists because the
    memory report is complete and silent about the one thing a user
    watching a run needs to know before starting it -- and a streamed run
    at a size that looks stalled is the concrete breakage.
    """
    estimate = estimate_pace(exp, streamed=streamed, machine=machine)
    if estimate is None:
        return None
    tail = ("" if estimate.measured else
            "  This rung and road have no measured step time: the bracket "
            "is a BOUND, not a measurement, and the basis says which.")
    return f"PACE: {estimate.sentence()}." + tail


__all__ = [
    "CARD_SPREAD", "LAUNCH_BOUND_COLUMNS", "MYNN_PREMIUM",
    "PCIE_PINNED_BYTES_PER_SECOND_HIGH", "PCIE_PINNED_BYTES_PER_SECOND_LOW",
    "PaceEstimate", "REFERENCE_NZ", "STEP_RATES", "StepRate", "estimate_pace",
    "measured_pinned_bytes_per_second", "pace_advisory",
    "resident_column_limit", "step_rate", "streamed_transfer_bytes_per_step",
    "UNPRICED",
]
