"""``gpuwm domain``: turn "my location + my GPU" into an experiment TOML.

The wizard emits a complete ``[experiment]``/``[projection]``/``[shared]``/
``[[domain]]`` TOML centered on ``--point`` or fitted around a local
``--polygon`` GeoJSON footprint, with grid dimensions chosen so
the itemized VRAM estimate (:func:`gpuwm.core.preflight.estimate_experiment`)
plus the machine-peak envelope over it
(:func:`gpuwm.core.preflight.machine_peak_envelope_bytes`) fits the
requested card's budget with headroom to spare.  Nothing here is a new
model of anything: the
physics/dynamics block is the product's default suite (the four-domain
reference configuration's selections with the microphysics slot on
Thompson mp8, the wrf-matched-run scheme), the projection
and nest-registration math is the existing :mod:`gpuwm.static.projection`
(lambert/mercator/polar, auto-selected from the point latitude), and the
memory arithmetic is the existing preflight estimator called in-process.

Honesty contract (terrain/static story): gpuwm builds static fields from a
locally staged NCAR WPS_GEOG tree (nine fixed dataset directories -- see
``GEOG_DATASETS``); ``gpuwm fetch-geog`` downloads and stages it.
Forcing data for the
config-driven ``gpuwm check``/``run`` front door is decoded by the native
GRIB1 route, i.e. ERA5 today.  GFS/HRRR downloads (``gpuwm fetch``) feed
the ``rw-wps``/``gpuwm-wrf-init`` native initialization front door, which
consumes the same ``[experiment]``/``[[domain]]`` tables but not
``[case_data]``.  The wizard therefore emits ``[case_data]`` only for
``--source era5`` and prints the exact honest next step for every source
instead of pretending a pipeline exists.

Sizing conventions (all documented, none silent):

* VRAM budget = the free VRAM a card of that capacity really presents
  (:func:`card_assumed_free_gib` -- never the nameplate; see
  :data:`CARD_UNAVAILABLE_VRAM_GIB`) minus THIS configuration's own
  reserve (:func:`sizing_budget_bytes`, the same
  ``ReservePolicy.n0_alloc`` call ``gpuwm check`` makes).  The reserve is
  not flat: it carries the local-memory backing store of the selected
  kernel set, which is 1.93 GiB for WSM6+MYNN and 3.94 for NSSL2
  double-moment, and a fit loop assuming a flat figure emitted configs
  that failed their own check.
* Fit criterion: ``peak envelope <= budget - fit_headroom_bytes`` --
  the estimator's own machine-peak envelope, which is AFFINE (the
  itemized estimate, plus the non-pool residency that scales with the
  device rather than the grid, plus a measured constant and a per-nest
  fraction).  A Windows card adds the measured WDDM pool-slack term
  (:data:`~gpuwm.core.preflight.WDDM_POOL_SLACK_FRACTION`, the 3080
  calibration) -- the same model `gpuwm check` and `gpuwm go` price, so
  a wizard PASS cannot become a check refusal on the same machine
  state.  The loop stops SHORT of the budget on purpose: a config that
  exactly touches its budget has nothing left for the machine to be
  slightly less generous than the model, which is how every v1.4.0
  ladder came to sit 0.01-0.19 GiB from the wall.
* Root time step: the certified real-data convention, 5 s per km of grid
  spacing (60 s at 12 km), halved inside the tropics; children divide
  down the ratio chain exactly, and a half-second root clock is carried
  exactly through WRF's rational clock keys.
* Ladders: the presets in ``LADDER_RATIOS``, or ``--root-dx`` +
  ``--chain`` for an arbitrary root spacing and integer refinement
  chain.  Both go through the same fit loop, loader, and ``gpuwm
  check``; a chain reaching below 1 km with a 1-D PBL scheme active
  earns a gray-zone advisory (never a refusal).
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np

from gpuwm.core.preflight import (CUDA_CONTEXT_BYTES,
                                  non_pool_basis,
                                  ENVELOPE_UNMODELLED_BYTES,
                                  EXTERNAL_MARGIN_BYTES, GIB,
                                  INGEST_PEAK_ENVELOPE_BASIS,
                                  ReservePolicy,
                                  card_local_memory_profile,
                                  device_memory_probe_reason,
                                  device_memory_probe_subprocess,
                                  envelope_platform, estimate_experiment,
                                  estimate_phases,
                                  profile_from_device_probe,
                                  unknown_platform_note)
from gpuwm.experiment import ExperimentConfig, build_experiment
from gpuwm.explain import explain_enabled, warn
from gpuwm.fetch import parse_cycle
from gpuwm.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                  CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                  RRTMG_VARIANT_LEGACY,
                                  MORRISON_PROFILE_ID, MYNN_PROFILE_ID,
                                  MYNN_RTE_RRTMGP_PROFILE_ID,
                                  MYNN_RUC_PROFILE_ID,
                                  MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
                                  NSSL2_LEGACY_RRTMG_PROFILE_ID,
                                  NSSL2_PROFILE_ID,
                                  RUC_PROFILE_ID,
                                  THOMPSON_LEGACY_RRTMG_PROFILE_ID,
                                  THOMPSON_PROFILE_ID,
                                  THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
                                  WSM6_PROFILE_ID,
                                  first_local_night_time,
                                  single_domain_runtime_switches)
from gpuwm.hrrr_route_inputs import (ROUTE_DEFAULT_PHYSICS_PROFILE,
                                     HrrrRouteInputError, coverage_advisory,
                                     coverage_refusal, route_input_paths,
                                     route_physics_blocker,
                                     write_hrrr_route_inputs)
from gpuwm.physics_menu import (WIZARD_PHYSICS_PROFILES,
                                profile_route_blocker)
from gpuwm.source_adapters import (get_source_adapter, source_adapters,
                                   source_coverage_window,
                                   source_forcing_interval_seconds,
                                   wizard_planable_source_ids)
from gpuwm.source_coverage import points_outside, window_centre
from gpuwm.static.projection import (EARTH_RADIUS_M, WRF_MAP_PROJ_CODES,
                                     _wrap180, projection_class)

#: Card tiers -> total VRAM (GiB).  ``--vram-gib`` accepts anything else.
CARD_VRAM_GIB = {"12gb": 12.0, "16gb": 16.0, "24gb": 24.0, "32gb": 32.0}

#: Nest ladders: dx chain in km (root fixed at 12 km) -> parent grid/time
#: ratios.  Ratios follow the certified 12->3->1 chain (4, 3); the 500 m
#: extension refines the 1 km nest by 2.
LADDER_RATIOS = {
    "12": (),
    "12-3": (4,),
    "12-3-1": (4, 3),
    "12-3-1-0.5": (4, 3, 2),
}
_LADDERS_DEEPEST_FIRST = ("12-3-1-0.5", "12-3-1", "12-3")

#: What a bare invocation emits: one 12 km domain, the shape ``gpuwm
#: go`` runs end to end and the shape FIRST-LIGHT 3a's worked first run
#: uses.  ``auto`` -- the deepest preset that fits the card -- was the
#: default until the first user-zero run of the published wheel piped
#: the default emission (a 4-domain tree) into the default runner and
#: was refused; the interactive door had already ruled the same way
#: (:data:`gpuwm.domain_interactive.DEFAULT_LADDER`, its own constant
#: because that door imports nothing heavy).  Nest trees are explicit
#: opt-in: a deeper preset, ``auto``, or --root-dx/--chain.
#:
#: The argparse default is ``None``, not this value, and that is
#: load-bearing: --root-dx/--chain are refused WITH --ladder, and a
#: reader who typed only the custom form must not be refused for
#: "combining" it with a flag they never passed.  ``domain_main``
#: resolves ``None`` to this constant (bare) or to the custom form's
#: pass-through value (--root-dx/--chain present).
DEFAULT_LADDER = "12"

ROOT_DX_M = 12000.0
#: Certified real-data clock convention: 60 s at 12 km = 5 s per km.
ROOT_TIME_STEP_S = 60

#: Halved root clock (2.5 s per km) for domains inside the tropics.
#:
#: A tropical Mercator domain at Manila (14.6 N) on the wizard's own
#: 60 s clock reached w_max 6.62 m/s and destabilized at +1 h; the same
#: domain at 15 s kept w_max at 1.5-3.0 m/s and completed 6 h.  The old
#: monitor exaggerated that event by pairing the global w maximum with
#: the unrelated thinnest layer; v1.1 now uses co-located |w|/dz.  The
#: trajectory evidence still supports a conservative tropical clock:
#: convection is deeper and more continuous at low latitude, so the
#: 5 s/km rule of thumb (WRF's own 6*dx_km agrees with it) is not
#: conservative enough there.  Halving is the cheapest possible remedy:
#: radiation and cumulus are called on wall-clock intervals, so 4x the
#: steps cost node 2 only +22% wall time (490 s vs 400 s for the same
#: 6 h).
#: Receipt: ARWEN-NODE2-4090-CUDA128-WORLDWIDE-20260730.md, PP-9.
TROPICAL_ROOT_TIME_STEP_S = 30

#: Parent rows a child boundary must clear: spec_bdy_width + blend_width
#: (both 5, the emitted [experiment] values; gpuwm/experiment.py enforces).
_CLEARANCE_ROWS = 10

#: Child linear extent as a fraction of the parent's, by nest depth --
#: the certified four-domain layout's proportions, then quantized.
_CHILD_SPAN_FRACTION = (0.5, 0.36, 0.4)

#: diff_6th_factor by nest depth (certified ladder).
_DIFF6_FACTORS = (0.12, 0.10, 0.08, 0.06)


def _at_depth(table: tuple, depth: int) -> float:
    """``table[depth]``, clamped to its ends -- never an IndexError.

    Both per-depth tables above were tabulated at the CERTIFIED
    four-domain layout's depth and then indexed by nest depth with no
    bound, so ``--chain`` with four or more nests died on a bare
    ``IndexError: tuple index out of range`` -- inside the sizing fit
    loop, where it read as a crash rather than as a refusal.  That is
    precisely the 12 -> 3 -> 1 -> 0.5 -> 0.25 km ladder this program
    exists to run.

    Clamping is the DEFINED behaviour for a depth past the table, not a
    guess: both tables are monotone toward their inner end (spans get
    proportionally tighter, sixth-order damping gets weaker), so the
    last entry is the innermost value anybody certified, and a deeper
    nest inheriting it is the conservative continuation.  It is also the
    convention already in force elsewhere in this codebase --
    ``gpuwm.da.nested_forecast._diff6_factor`` clamps the same table the
    same way for the same reason.

    This is a bound on a LOOKUP, not on a limit.  Nothing here softens a
    refusal: a nest that cannot be hosted inside its parent with the
    boundary clearance still raises :class:`DomainFitError` naming the
    extent, and the fit loop still prices every candidate against the
    card.
    """
    if not table:
        raise ValueError("per-depth table is empty")
    return table[min(max(int(depth), 0), len(table) - 1)]


def _child_span_fraction(depth: int) -> float:
    """Child linear extent as a fraction of its parent's, at ``depth``."""

    return float(_at_depth(_CHILD_SPAN_FRACTION, depth))


def _diff6_factor(depth: int) -> float:
    """``diff_6th_factor`` for the domain at ``depth`` (0 = root)."""

    return float(_at_depth(_DIFF6_FACTORS, depth))


#: What a card of nominal capacity NEVER hands to a process: the driver's
#: own reservation, the display/compositor allocations on a card that has
#: one, and the gap between a marketing "16 GB" and the 16,376 MiB the
#: silicon carries.
#:
#: MEASURED: an idle headless RTX 4080 (16,376 MiB physical = 15.99 GiB)
#: presents 15.33 GiB free to a fresh CUDA context -- a 0.66 GiB gap.  The
#: 16 GiB tier assumed the card would hand over its whole nominal size, so
#: every ladder it emitted was sized against 0.33 GiB of VRAM that does
#: not exist, landed 0.13-0.32 GiB over the real budget, and failed the
#: product's own ``gpuwm check`` minutes after the wizard printed PASS.
#: 0.75 rounds the measured gap UP: a tier must be conservative against
#: real cards of its class, not equal to the best one.
CARD_UNAVAILABLE_VRAM_GIB = 0.75

#: ...and the same gap as a FRACTION, because it is not the same number
#: of gibibytes on every card.  The one 32 GiB free figure this codebase
#: has measured is 30.27 of 31.84 -- a 1.57 GiB gap, because that machine
#: also had a desktop on the card.  A tier has to be conservative against
#: the cards of its class that exist, not against the best one, so the
#: larger of the two forms binds.
CARD_UNAVAILABLE_VRAM_FRACTION = 0.06

#: How much of the budget the fit loop refuses to spend.  The loop grows
#: the grid until the envelope TOUCHES the budget, which is how every
#: emitted ladder landed 0.01-0.19 GiB from the wall -- a rounding error
#: away from a refusal, and with nothing left for the card to be one
#: driver revision less generous than it was when the config was written.
FIT_HEADROOM_FRACTION = 0.05
FIT_HEADROOM_MIN_BYTES = GIB // 4


def card_assumed_free_gib(vram_gib: float) -> float:
    """Free VRAM a card of ``vram_gib`` nominal capacity really presents.

    Never the nominal size: see :data:`CARD_UNAVAILABLE_VRAM_GIB`.
    """

    unavailable = max(CARD_UNAVAILABLE_VRAM_GIB,
                      CARD_UNAVAILABLE_VRAM_FRACTION * float(vram_gib))
    return max(0.0, float(vram_gib) - unavailable)


def fit_headroom_bytes(budget_bytes: int) -> int:
    """Budget the fit loop leaves unspent, so nothing lands on the wall."""

    return max(FIT_HEADROOM_MIN_BYTES,
               int(FIT_HEADROOM_FRACTION * max(0, int(budget_bytes))))


def vram_reserve_gib(vram_gib: float) -> float:
    """Flat VRAM reserve (GiB) by card capacity: WDDM/driver/CUDA context
    plus the near-capacity stability ceiling of consumer cards.

    RETIRED as the sizing path's reserve on 2026-08-01, and kept for the
    callers that have no experiment to price (``gpuwm downscale``'s
    standalone child fit) and for the "your card is too small before we
    even start" refusal.  A FLAT figure was the whole of defect 4: the
    reserve's overhead term tracks the local-memory backing store of the
    SELECTED KERNEL SET, which measured 1.93 GiB for WSM6+MYNN and 3.94
    for NSSL2 double-moment -- so a fit loop assuming a flat 4.0 sized
    both NSSL2 profiles against one budget and then verified them against
    a smaller one, at every card size.  The sizing path now prices the
    reserve from the candidate experiment itself
    (:func:`gpuwm.core.preflight.ReservePolicy.n0_alloc`), which is the
    same call ``gpuwm check`` makes, so the two cannot disagree.

    The small-card figure was 3.0 GiB and it was a promise the preflight
    would not keep: ``ReservePolicy.n0_alloc`` charges the CUDA context
    plus the widest launched kernel's local-memory backing store plus a
    retention residual, which lands at 3.5-3.6 GiB on exactly those 12
    and 16 GiB cards.  Sizing a layout against a 3.0 GiB reserve and then
    handing it to a preflight applying 3.55 is how a node-7 pilot got a
    wizard-certified config whose own check said the envelope did not
    fit.  4.0 -- the figure the 24 GiB tier already uses -- clears the
    measured reserve on both small tiers with about 0.45 GiB to spare.
    """
    if vram_gib <= 24.0:
        return 4.0
    return 6.0


#: Projection auto-selection bands (absolute latitude of --point):
#: below MERCATOR_MAX_LAT the Lambert cone is ill-conditioned and the
#: wizard selects Mercator; above LAMBERT_MAX_LAT it selects polar
#: stereographic; between them, hemisphere-correct Lambert conformal.
#: --projection overrides the choice explicitly.  All three projections
#: are oracle-gated against the pinned WRF v4.6.1 module_llxy.F
#: (tests/test_projection_oracle.py).
MERCATOR_MAX_LAT, LAMBERT_MAX_LAT = 25.0, 60.0
#: Cells of clearance between the domain footprint and the projection
#: pole; a domain containing (or touching) the pole is refused -- the
#: lat-lon source interpolation and static-tile windowing are not
#: pole-capable (genuine limit, not a projection-math one).
_POLE_CLEARANCE_CELLS = 2.0

#: Degrees of latitude the SUGGESTED FORCING BOX keeps clear of a pole.
#:
#: The refusal above guards the domain footprint, but the fetch hint is
#: the footprint plus a margin, and clamping that at exactly +-90 made
#: the wizard print `--area 42.93,-45.56,90.00,83.48` for Tromso: a top
#: edge sitting on the very singularity the README says is refused, with
#: no comment.  `gpuwm fetch` accepted it and downloaded 89 MB.  The
#: same 2-cell clearance the domain refusal enforces, expressed as
#: degrees of meridian at the root dx, keeps the suggestion honest.
def pole_clearance_deg(root_dx_m: float = ROOT_DX_M) -> float:
    """The forcing box's pole clearance in degrees, at this root dx."""

    return _POLE_CLEARANCE_CELLS * float(root_dx_m) / 111_195.0


def max_fetch_abs_lat(root_dx_m: float = ROOT_DX_M) -> float:
    """The most poleward latitude a suggested forcing box will name."""

    return 90.0 - pole_clearance_deg(root_dx_m)


POLE_CLEARANCE_DEG = pole_clearance_deg()
#: The most poleward latitude the suggested forcing box will name.
MAX_FETCH_ABS_LAT = max_fetch_abs_lat()

#: Degrees of forcing margin beyond the root domain for the fetch hint.
#: ERA5 needs only interpolation-halo coverage; HRRR files are CONUS-wide
#: (``--area`` is a coverage check, not a crop, and the hint is clamped
#: into the grid's own envelope -- see ``fetch_area_hint``), so both
#: keep the small margin.  GFS uses
#: :func:`gpuwm.fetch.gfs_suggested_fetch_margin_deg`: the GFS front
#: door's donor-coverage proof must find every model lake's nearest
#: source-water donor INSIDE the crop, so the wizard's suggested area
#: carries that documented margin instead of being rejected downstream.
_FETCH_MARGIN_DEG = 2.0


def _fetch_margin_deg(source: str) -> float:
    if source == "gfs":
        from gpuwm.fetch import gfs_suggested_fetch_margin_deg
        return gfs_suggested_fetch_margin_deg()
    return _FETCH_MARGIN_DEG

#: Forcing cadence per source: estimator LBC-interval sizing + the
#: ``&share/interval_seconds`` the emitted namelist.wps carries.
#:
#: REGISTRY-DERIVED since 2.5.0.  It used to be this three-entry literal:
#:
#:     {"era5": 21600.0, "gfs": 10800.0, "hrrr": 3600.0}
#:
#: and that dict was the whole reason `gpuwm domain --source` offered three
#: sources while the product shipped sixteen runnable ones.  The
#: 2026-08-17 model battery hand-assembled a TOML and a namelist.wps for
#: every other model, typing this one number in from the packaged mapping.
#: It is a source's own published fact, so it lives in the source's own
#: registry row (:func:`gpuwm.source_adapters.source_forcing_interval_seconds`)
#: and a new model reaches this door without touching this file.
SOURCE_FORCING_INTERVAL_S = MappingProxyType({
    source_id: source_forcing_interval_seconds(source_id)
    for source_id in wizard_planable_source_ids()
})

#: Sources whose fetch planner takes no ``--cadence`` at all.  HRRR is
#: hourly by construction and `gpuwm fetch --source hrrr --cadence N`
#: refuses by name ("HRRR is hourly; --cadence does not apply"), so the
#: emitted [fetch] table must not carry the key for it.  Stated as an
#: exclusion rather than by omitting HRRR from a hand-written table,
#: because omission reads as an oversight and this one is a fact about
#: the download planner.
_CADENCE_FREE_FETCH_SOURCES = frozenset({"hrrr"})


def _fetch_ladder_cadence_h() -> dict[str, int]:
    """The ``cadence = N`` an emitted ``[fetch]`` table carries, per source.

    The download ladder's spacing is the source's own native cadence -- a
    fetch that skipped valid times would hand the preparation a series
    with holes in it -- so this comes from the registry rather than from a
    second hand-written table.  It used to be ``{"era5": 6, "gfs": 3}``,
    and `--source gdas` was unreachable from this door precisely because
    nothing had ever added the third entry: opening the door without
    deriving this produced "GFS cadence must be 1 or 3 hours" from a
    ``cadence`` key nobody had filled in.
    """

    from gpuwm.fetch import fetch_front_door_sources

    return {
        source: int(source_forcing_interval_seconds(source) // 3600)
        for source in fetch_front_door_sources()
        if source not in _CADENCE_FREE_FETCH_SOURCES
        and source in SOURCE_FORCING_INTERVAL_S
    }


_SOURCE_CADENCE_H = _fetch_ladder_cadence_h()


def planable_sources() -> tuple[str, ...]:
    """Every source id this door can emit a runnable configuration for."""

    return wizard_planable_source_ids()


def resolve_source(raw: str) -> str:
    """RAW (an id or a registry alias) as the canonical source id.

    Fail-closed in three named ways, because "invalid choice: 'rap'" is a
    refusal that tells a reader nothing about why the model they can see in
    `gpuwm prep --list-sources` cannot be planned for:

    * an unknown name lists the registry, so a misspelling is one glance
      from correct;
    * a registered row with no runnable route says what the row says about
      itself, so the reader learns the state of that model rather than the
      state of this parser;
    * a registered runnable row with no declared forcing cadence names the
      missing fact, because emitting ``interval_seconds`` for it would be a
      guess about boundary times the source may never publish.
    """

    try:
        adapter = get_source_adapter(raw)
    except ValueError:
        raise ValueError(
            f"--source {raw!r} is not a registered source; "
            f"gpuwm domain plans for {', '.join(planable_sources())}.  "
            "`gpuwm prep --list-sources` lists the whole registry, "
            "including the rows "
            "that are registered but have no runnable route yet") from None
    if not adapter.runnable:
        raise ValueError(
            f"--source {adapter.source_id}: this source is registered but "
            f"has no runnable initialization route ({adapter.status.value})"
            + (f" -- {adapter.composition_requirement}"
               if adapter.composition_requirement else "")
            + f".  A configuration emitted for it could not be prepared by "
              f"anything; plan with one of {', '.join(planable_sources())}")
    if adapter.forcing_interval_seconds is None:
        raise ValueError(
            f"--source {adapter.source_id}: this route's boundary cadence "
            "comes from the mapping document a caller supplies, not from "
            "the registry, so this door has no interval_seconds to write "
            "and the namelist.wps it emitted would name boundary times the "
            "inputs may not carry.  Plan with the named source your mapping "
            f"describes ({', '.join(planable_sources())}), or author the "
            "namelist beside the mapping")
    return adapter.source_id


def source_has_fetch_front_door(source: str) -> bool:
    """Can ``gpuwm fetch`` go and get this source's bytes today?

    The wizard emits an advisory ``[fetch]`` table only when the answer is
    yes.  A table naming a source the fetch door does not serve is refused
    at every later config load, and a table that quietly loads would
    advertise a download nothing can make -- so the emission asks here and
    prints the manual acquisition route otherwise.
    """

    from gpuwm.fetch import fetch_front_door_sources

    return source in fetch_front_door_sources()


def source_credential_notes(source: str) -> list[str]:
    """Pointer lines for what SOURCE needs configured and does not have.

    The registry row's CREDENTIAL column, wrapped for a terminal.  This
    used to be an ``if source == <one id>`` arm in two places, so a
    second source needing an account key would have needed two more
    arms; now a declared credential reaches every door that asks here,
    and a row that declares none produces no line at all.

    Never raises.  An unresolvable source id is the caller's business to
    refuse -- printing a traceback in place of a pointer would replace a
    helpful line with a broken command.
    """

    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_credentials import absent_credential_notes

    try:
        adapter = get_source_adapter(source)
    except Exception:  # noqa: BLE001 - an advisory line, not a gate
        return []
    return absent_credential_notes(adapter.credentials)


def _candidate_fetch_hints(source: str) -> dict | None:
    """The ``[fetch]`` stub a sizing candidate carries, or ``None``.

    A candidate is rendered and reloaded through the real experiment
    loader, so it must be a file that LOADS: a ``[fetch]`` table naming a
    source the fetch door does not serve is refused there, which would
    have turned every new source's first fit iteration into a load error
    rather than a size.
    """

    return {"source": source} if source_has_fetch_front_door(source) else None


def source_fetch_takes_a_crop_box(source: str) -> bool:
    """Does this source's fetch accept an ``area``/``point`` crop?

    Asked of the fetch module rather than decided here, for the same
    reason the door question is: the hand-written transports subset at
    the publisher and the table routes take whole objects, and a
    ``[fetch]`` table carrying ``area`` for one of the latter prints a
    step 1 that exits 2 and is refused at every later config load.
    """

    from gpuwm.fetch import fetch_accepts_area

    return fetch_accepts_area(source)


def source_reaches_forecast_leads(source: str) -> bool:
    """Does SOURCE publish forecast leads, or only analyses at valid times?

    The registry's ``max_forecast_hour`` is the whole answer: a reanalysis
    and an every-member analysis archive both declare 0, and the front door
    that used to spell this ``{"gfs", "gdas", "hrrr"}`` refused ``rap`` a
    lead RAP publishes 51 hours of.
    """

    return get_source_adapter(source).max_forecast_hour > 0


def _fetch_cadence_h(source: str, start_hour: int) -> int | None:
    """The fetch cadence this window can actually be taken on.

    The default is the source's usual spacing, and for a window starting
    at f000 that is what every prior release emitted.  A forecast LEAD
    changes the question: a fetch window must CONTAIN the lead it begins
    at, and ``gpuwm fetch`` refuses one that does not.  So a config
    written with ``cadence = 3`` and ``forecast_start_hour = 4`` named a
    download that could never be made -- step 1 of the wizard's own
    printed recipe exited 2 with "f004 is not on the 3 h cadence", and
    for two leads in every three the one-command `gpuwm go` path could
    not be made to work at all.

    The cadence is therefore chosen to divide the lead.  GFS publishes
    hourly through f120, so 1 h is available wherever it is needed; a
    window that would cross f120 hourly is refused by the fetch planner
    below with the structural reason, before the file is written.
    """

    cadence = _SOURCE_CADENCE_H.get(source)
    if cadence is None or not start_hour or start_hour % cadence == 0:
        return cadence
    # Hourly divides every integer lead, and is the only other cadence
    # these sources publish.
    return 1

#: Default output cadences, root and nest, in seconds.
#:
#: They were bare literals inside :func:`_domain_tables` with no knob,
#: which made "how often does this write" a thing a reader could see in
#: the emitted TOML and not change.  Named here because they are now a
#: DEFAULT rather than a fixture: ``--history-interval`` overrides them.
#:
#: The nest writes four times as often as the root on purpose -- a nest
#: exists to resolve what the root cannot, over a window shorter than
#: the root's whole forecast, so its output is the point of running it.
DEFAULT_ROOT_HISTORY_INTERVAL_S = 3600.0
DEFAULT_NEST_HISTORY_INTERVAL_S = 900.0

#: Grid-scale search bounds.  _MIN_SCALE puts the root at 60 x 48 mass
#: points, the smallest layout that still hosts the deepest ladder with
#: full Davies/blend clearance.
#:
#: _MAX_SCALE is deliberately NOT a physical limit -- it exists only so the
#: bisection has a finite upper bracket, and it must stay far enough above
#: what any real budget wants that MEMORY is what decides the answer.  It
#: was 8.0, which is 880 x 704 at the root, and that bound bound: every
#: single-domain budget at or above 64 GiB returned exactly 880 x 704 and
#: reported a comfortable fit, so 64, 96 and 180 GiB cards were all sized
#: like a 64 GiB one.  Raising it to 16.0 changes nothing for budgets that
#: were not already saturated and lets 96 GiB reach 1178 x 944 and 180 GiB
#: reach 1674 x 1340; raising it further to 32.0 changes nothing again,
#: which is the check that the bracket, not the bound, now decides.
#: 64.0 is chosen with that headroom on purpose.  If it ever binds the
#: wizard says so out loud rather than silently under-sizing -- see
#: fit_ladder.
_MIN_SCALE, _MAX_SCALE = 0.55, 64.0

#: The nine WPS_GEOG dataset directories the static builder opens (the
#: ``default`` geog_data_res selector; gpuwm/static/build.py).
GEOG_DATASETS = (
    "topo_gmted2010_30s", "modis_landuse_20class_30s_with_lakes",
    "soiltype_top_30s", "soiltype_bot_30s", "greenfrac_fpar_modis",
    "lai_modis_10m", "albedo_modis", "maxsnowalb_modis", "soiltemp_1deg",
)

#: Certified 49-mass-level vertical coordinate (50 full eta levels) --
#: the reference configuration's ladder.  Eta is normalized, so one
#: ladder serves any emitted model top: certified at 100 hPa, and run
#: A/B at the 50 hPa default on the 2026-08-24 plains receipt (same
#: levels, deeper column, byte-verified P_TOP in both arms' output).
_ETA_LEVELS = (
    1.0, 0.9978, 0.99519, 0.99212, 0.98849,
    0.98422, 0.97918, 0.97325, 0.96627, 0.95808,
    0.94846, 0.93719, 0.92402, 0.90866, 0.89079,
    0.87006, 0.84612, 0.81857, 0.78706, 0.75124,
    0.7108, 0.66556, 0.61547, 0.56067, 0.50519,
    0.45474, 0.40886, 0.36713, 0.32918, 0.29466,
    0.26328, 0.23473, 0.20877, 0.18516, 0.16369,
    0.14417, 0.12641, 0.11026, 0.09557, 0.08222,
    0.07007, 0.05902, 0.04898, 0.03984, 0.03153,
    0.02398, 0.0171, 0.01085, 0.00517, 0.0,
)

_PACKAGED_VTABLE = Path(__file__).parent / "data" / "vtables" / \
    "Vtable.ERA5_CDO"

#: The product's default physics/dynamics selections -- the reference
#: four-domain configuration's [shared] block with the microphysics slot
#: on Thompson (mp_physics 8: the wrf-matched-run scheme,
#: WRF's own tables packaged and hash-pinned): the MM5 surface layer
#: (91), Noah LSM (2), YSU PBL (1), RTE+RRTMGP radiation (4, the
#: ratified WRF-RRTMG 4/4 substitution), and the certified
#: diffusion/damping/acoustic settings.  Morrison (10) stays fully
#: selectable at its registry maturity label; its morr_rimed_ice knob is
#: Morrison-only and is deliberately absent here.
#: The default model top (Pa).  50 hPa (~20.6 km) -- WRF v4.6.1's own
#: p_top_requested Registry default (Registry.EM_COMMON:2275) and the
#: convection-allowing community standard.  The prior 10000 Pa (100 hPa,
#: ~16 km) default put the damp_opt=3 zdamp=5000 sponge base near
#: 10.9 km AGL, inside deep convection's anvil layer: measured on the
#: 2026-08-24 plains A/B, the 100 hPa arm's anvil-layer max updrafts
#: plateaued at ~17 m/s and collapsed at the sponge base while the
#: 50 hPa arm peaked at 22 m/s with natural decay near 13 km, carried
#: more >=40 dBZ core cells and higher cloud tops, scored equal-or-
#: better MRMS FSS, and ran 12.9% faster at the same VRAM (tests/
#: test_ptop_default.py carries the receipt).  Emissions bound this
#: default per source via :func:`emitted_model_top_pa`.
DEFAULT_MODEL_TOP_PA = 5000.0


def emitted_model_top_pa(source: str | None) -> float:
    """The model top (Pa) a config emitted for SOURCE carries.

    The default, unless the source's registry row declares a certified
    inventory top the default sits above (``certified_source_top_pa``
    -- e.g. the GFS 21-level ladder stops at 100 hPa).  Without the
    bound, a bare ``gpuwm domain --source gfs`` emission would ask for
    a model top its own certified inventory cannot cover and refuse at
    preparation, after the user already paid for the acquisition.
    """

    if source is None:
        return DEFAULT_MODEL_TOP_PA
    ceiling = get_source_adapter(source).certified_source_top_pa
    if ceiling is None:
        return DEFAULT_MODEL_TOP_PA
    return max(DEFAULT_MODEL_TOP_PA, float(ceiling))


_SHARED_GRID_AND_DYNAMICS = {
    "nz": 49, "ztop": 20000.0, "p_top": DEFAULT_MODEL_TOP_PA,
    "eta_levels": _ETA_LEVELS,
    "hybrid_opt": 2, "etac": 0.2, "base_temp": 290.0,
    "time_step_sound": 4, "emdiv": 0.01,
    "hypsometric_opt": 2, "h_sca_adv_order": 5, "smdiv": 0.1,
    "moist_adv_opt": 1,
    "w_damping": 1, "damp_opt": 3, "zdamp": 5000.0, "dampcoef": 0.2,
    "khdif": 0.0, "kvdif": 0.0, "spec_zone": 1, "relax_zone": 4,
    "bldt": 0.0,
    # WRF nwp_diagnostics: wizard runs are convective forecasts whose
    # audience reads UH products, so the UP_HELI_MAX running-max
    # diagnostic ships ON (trajectory-inert; tests/test_uh_lifecycle.py).
    "nwp_diagnostics": 1,
}

#: Physics switches the ROOT domain carries rather than ``[shared]``.
_PER_DOMAIN_PHYSICS = ("radt", "cu_physics", "cudt_minutes",
                       "diff_6th_factor")

#: The wizard's default physics for real cases (gfs/era5; hrrr has its
#: own route-constrained default, HRRR_DEFAULT_PROFILE below).
#:
#: Owner directive 2026-08-06, after a shipped 48 h case: the default a
#: real case gets must be a CERTIFIED, NOCTURNALLY VALID suite -- both
#: radiation streams on, registry maturity read off the registry rather
#: than asserted here.  This profile is the registry's only user-ready
#: ``wrf-matched-run`` template with full lw+sw radiation
#: (RTE+RRTMGP 4/4 + Kain-Fritsch), it is declared by every route for
#: every source, it is FIRST-LIGHT section 3a's worked example, and it
#: is already the interactive door's default
#: (:data:`gpuwm.domain_interactive.DEFAULT_PHYSICS_PROFILE_BY_SOURCE`)
#: -- the two doors now agree.  It replaced ``None`` (the unshipped
#: "product default suite", DEFAULT_SUITE_PHYSICS below, supported but
#: not WRF-verified), which remains reachable programmatically and is
#: NOT emitted by any door.  Validation profiles with asymmetric
#: radiation (shortwave on, longwave off) are never a default on any
#: door; they stay selectable explicitly and are emitted with the
#: nocturnal declaration (see render_config).
#:
DEFAULT_PHYSICS_PROFILE = MORRISON_PROFILE_ID

#: The default suite's physics switches, in the registry's OWN radiation
#: representation.
#:
#: v1.0.0 wrote ``ra_physics = 4`` (the legacy combined selector) while
#: every shipped profile -- and the physics registry -- writes the split
#: pair ``ra_physics = 0`` + ``ra_lw_physics = 4`` + ``ra_sw_physics =
#: 4``.  The two are semantically identical, and
#: ``radiation_scheme_ids`` resolves both to (4, 4), but the runner's
#: guard compares the raw switch dicts, so no wizard-emitted config
#: could ever pass it.  Emitting the split form is the fix at the
#: source, and it also removes a real misreading hazard: a pilot report
#: read the profiles' ``ra_physics: 0`` as "radiation off" when three of
#: those profiles run RTE+RRTMGP on both streams.
DEFAULT_SUITE_PHYSICS = {
    "moist": True, "moist_cq": True, "mp_physics": 8, "top_lid": False,
    "epssm": 0.5, "wrf_rrtmg_compatibility":
        "wrf-rrtmg-4-4-to-rte-rrtmgp-v1",
    "ra_physics": 0, "ra_lw_physics": 4, "ra_sw_physics": 4,
    "sf_sfclay_physics": 91, "sf_surface_physics": 2,
    "bl_pbl_physics": 1, "terrain_opt": 1,
    "km_opt": 4, "diff_6th_opt": 2, "diff_6th_slopeopt": 1,
    # Per-domain (root values; nests override in _domain_tables).
    "radt": 12.0, "cu_physics": 1, "cudt_minutes": 5.0,
    "diff_6th_factor": _DIFF6_FACTORS[0],
}

#: The profiles the prepared single-domain forecast runner accepts, in
#: the order the help lists them.
#:
#: DEFINED in :mod:`gpuwm.physics_menu` and imported above, with the
#: ordering rationale (nocturnally valid suites first, the two
#: legacy-RRTMG Thompson entries as the only full-radiation suites the
#: nested HRRR route admits) written out there.  Re-exported under this
#: name because every reader in the tree spells it
#: ``domain_wizard.WIZARD_PHYSICS_PROFILES``.


def _radiation_words(switches: dict) -> str:
    """Plain language for what a profile's radiation switches DO.

    ``ra_physics = 0`` beside ``ra_lw_physics = 4`` means RTE+RRTMGP
    longwave, not "radiation off" -- a reading that has already been got
    wrong once in a pilot report, because the split and legacy
    representations look alike and only one of them is the truth.  The
    wizard therefore never prints the raw switches without the words.
    """

    # Selector 4 is "RRTMG" in WRF's spelling and gpuwm implements it two
    # ways -- the exact legacy-RRTMG transcription and the RTE+RRTMGP
    # substitution -- so the words follow ``ra_rrtmg_variant`` rather than
    # calling every 4 RTE+RRTMGP.  A header that names the wrong solver is
    # the same misreading hazard this function exists to remove.
    legacy = switches.get("ra_rrtmg_variant") == RRTMG_VARIANT_LEGACY
    names = {0: "OFF", 1: "Dudhia",
             4: "legacy RRTMG" if legacy else "RTE+RRTMGP"}
    lw = int(switches.get("ra_lw_physics", -1))
    sw = int(switches.get("ra_sw_physics", -1))
    if (lw, sw) == (-1, -1):
        lw = sw = int(switches.get("ra_physics", 0))
    return (f"longwave {names.get(lw, lw)}, "
            f"shortwave {names.get(sw, sw)}")


def prepared_route_physics_notice(profile: str | None,
                                  source: str) -> list[str]:
    """The emitted suite's standing on the prepared single-domain route.

    Owner ruling 2026-07-31: the prepared single-domain forecast runner
    executes any suite the engine implements, exactly as this file
    writes it -- the profile whitelist and its exact-equality refusal
    are gone, and verification status is reported, never gating.  These
    lines are the --explain detail; the always-printed ``physics:``
    summary line already carries the one-sentence status.  Picking a
    shipped profile still quietly changes the science -- several run no
    cumulus and shortwave Dudhia with longwave OFF -- so each candidate
    is still named with what it actually runs.
    """

    if source not in ("gfs", "hrrr"):
        return []
    if profile is not None:
        return [
            "note: this config is bound to a shipped profile, so every "
            "runner enforces it switch for switch, exactly as emitted."
        ]
    lines = [
        "note: the suite above runs as written on the prepared "
        "single-domain route and the multi-domain (domain-tree) route "
        "alike; no --physics-profile is required, and its "
        "WRF-verification status is reported in the run receipts."
        if source == "gfs" else
        "note: the HRRR route's cold-start evidence contract is keyed "
        "by shipped profile, so it prepares "
        f"{HRRR_DEFAULT_PROFILE} when none is named -- Thompson "
        "microphysics with RRTMG longwave AND shortwave and no cumulus "
        "at 3 km, the operational HRRR composition; pass "
        "--physics-profile <id> to choose another.",
        "  --physics-profile <id> binds the config to a shipped suite "
        "and every runner then enforces it switch for switch.  What "
        "each one ACTUALLY runs:",
    ]
    for candidate in WIZARD_PHYSICS_PROFILES:
        lines.append(f"    {physics_summary(candidate)}")
    return lines


#: Longitude span, in degrees, past which the sized fetch box gets a word
#: said about it.
#:
#: Not a limit and not a refusal: the sizer's job is to use the card it
#: was given, and on a 32 GiB card one 12 km domain legally fills it --
#: an owner's first emission covered 91 degrees of latitude and 152 of
#: longitude with 0.00 GiB of headroom, which is correct arithmetic and
#: an absurd first run.  The number below is a "that is bigger than you
#: probably meant" threshold, chosen so a continental domain (the
#: documented examples run 60-80 degrees of longitude at 24 GiB) passes
#: without comment.
_WIDE_FOOTPRINT_DEGREES = 90.0

#: The same "bigger than you probably meant" bar for latitude.  The
#: longitude test alone let a Linux ``--card 24gb --ladder 12`` print a
#: 144 x 88 degree domain behind a 174-degree fetch box without the
#: latitude -- pole to equator and then some -- being mentioned at all.
#: Documented continental examples run 30-50 degrees of latitude.
_TALL_FOOTPRINT_DEGREES = 60.0


def _area_span_degrees(area: str) -> tuple[float, float] | None:
    """``(lat_span, lon_span)`` for a ``S,W,N,E`` box, or None."""

    parts = str(area).split(",")
    if len(parts) != 4:
        return None
    try:
        south, west, north, east = (float(value) for value in parts)
    except ValueError:
        return None
    lon = east - west
    if lon < 0:
        lon += 360.0
    return north - south, lon


def oversized_footprint_advisory(area: str) -> list[str]:
    """Say when the sized domain fills the card rather than the map.

    An advisory, never a refusal, and it changes no sizing: the layout
    is already chosen by the time this runs.  What it changes is whether
    a first-time reader learns that the hemisphere-wide fetch box they
    are about to download is a single flag away from something smaller.

    It used to open "sized to fill your card, not your map".  That
    asserted a CAUSE this function cannot see, and since 2.5.0 gave the
    fit a servable-crop bound the cause is sometimes false: on a large
    card the wizard now stops on the source and says so on stderr, and
    an advisory blaming the card two lines later contradicts it.  The
    sentence states what the thresholds above actually measure -- the
    box is much bigger than the documented examples -- and keeps the
    remedy, which is true either way.

    Two things the first version got wrong, both found by a wheel user
    on Linux at ``--card 24gb --ladder 12``.  It measured longitude
    only, so a box 88 degrees tall passed unremarked; and its remedy
    named ``--card 12gb``, a tier not every platform runs, while saying
    nothing about the ``--area`` on the very fetch command printed two
    lines below -- the number a reader looks at when they wonder what
    this will cost them.  The sentence now names the box as the fetch
    box, and says why narrowing that flag alone is not the fix: the
    download exists to cover the domain, so the domain is what has to
    get smaller.
    """

    spans = _area_span_degrees(area)
    if spans is None:
        return []
    lat_span, lon_span = spans
    if (lon_span < _WIDE_FOOTPRINT_DEGREES
            and lat_span < _TALL_FOOTPRINT_DEGREES):
        return []
    return [
        f"this domain is much wider than the documented examples: the "
        f"fetch command below downloads a {lon_span:.0f} x "
        f"{lat_span:.0f} degree --area box, so pass --vram-gib N (or a "
        f"finer --root-dx KM) for a smaller first run -- narrowing "
        f"--area on its own would starve the domain it feeds"]


def _guard_exports_block(profile: str | None) -> str:
    """Environment the printed chain needs, printed with the chain.

    The mp8 runners under ``tools/`` keep a launch contract the library
    itself retired: both variables must be set in the SHELL that starts
    the chain.  Preparation launches the forecast runner as a subprocess
    and inherits the environment, so one export pair covers both stages
    -- but only if the reader knows to type it, and nothing printed it.
    A field run of the shipped 1.5.0 wheel discovered the requirement by
    failing twice and then went looking for the table root by hand.

    Empty for every other profile, so the emitted block stays exactly as
    short as it was for the suites that need nothing.
    """

    if profile != THOMPSON_PROFILE_ID:
        return ""
    from gpuwm.physics_compat import thompson_guard_exports

    lines = "".join(f"  {line}\n" for line in thompson_guard_exports())
    return (
        "# this suite's runners are gated on two environment variables "
        "-- export them in\n"
        "#   THIS shell before the chain: preparation launches the "
        "forecast runner as a\n"
        "#   subprocess, so one pair covers both stages.  The root below "
        "is the one this\n"
        "#   install resolves; every asset in it is byte-checked before "
        "GPU setup.\n"
        + lines)


def hrrr_route_commands(out: "Path", exp: ExperimentConfig, *,
                        profile: str | None, data_dir: str,
                        cycle: "datetime | None" = None,
                        forecast_start_hour: int = 0) -> str:
    """The HRRR chain, with every file this emission just wrote bound.

    HRRR reaches the GPU through the native front door: neither ``gpuwm
    run`` (the ERA5 ``[case_data]`` route) nor ``gpuwm go`` (whose five
    commands do not compose this route's stage vocabulary) drives it,
    and it is NOT prepared with ``rw-wps``, which is the GFS door.
    Until 2026-08-01 a multi-domain HRRR emission was told to use
    exactly that, because the multi-domain branch of
    :func:`final_step_command` returned before the HRRR one was ever
    reached.

    **The commands printed here are the SHIPPED ones**, rendered from
    :func:`gpuwm.stage_cli.staged_route_commands` -- the same helper
    ``gpuwm go``'s hrrr refusal and ``gpuwm sim``'s unbindable-tree
    refusal render, so a reader cannot be handed three spellings of one
    route.  What this block printed until 2026-08-18 was the route's
    INTERNALS: ``python -m tools.prepare_hrrr_wrf`` and ``python
    tools/hrrr_single_domain_benchmark.py`` -- two ``tools/`` paths a
    pip wheel does not contain at all -- plus ``python -m
    gpuwm.hrrr_hierarchy_direct`` and ``python -m
    gpuwm.prepared_domain_tree_forecast``.  Every one of those is a
    program ``gpuwm prep``/``gpuwm sim`` spawn (MEASURED: ``gpuwm prep
    --source hrrr ... --dry-run`` prints each line verbatim), so the
    old block was machinery where a door exists.

    Every value this emission knows is bound -- the four input files,
    the cycle, the lead, the run length, the cadence, the profile, and
    ``--statics-corridor`` on the hierarchy stage when the config
    declares a ``[relocation]`` follow source.  What is left as a
    placeholder is what cannot exist yet: the WPS_GEOG root, which is
    the reader's install, and the source manifest's own digest, which
    ``gpuwm fetch`` prints.  The FORECAST stage now asks for no
    placeholder at all -- ``gpuwm sim`` reads the preparation's digests
    off the bundle it is pointed at, so the two ``<printed by the
    hierarchy>`` / ``<sha256 of that file>`` values a reader used to
    have to produce by hand are gone.

    **Every time printed here is the CYCLE, and the lead is printed
    beside it.**  Model time zero (cycle + K) is derived by each stage,
    never typed.  Both stages used to be handed one ``--valid-time``
    string, and the two stages read that flag differently -- the
    preparer as the cycle (it opens ``hrrr.tHHz.wrfnatfNN.grib2``), the
    hierarchy as the model start (it is compared to the namelist's
    start_time).  At lead 0 those are the same instant, which is why one
    string served both for four releases; at lead K one of them is
    wrong by K hours.  Printing the cycle and the lead separately means
    the same two values appear on every line of the chain and neither
    stage has to be told a time the other stage computed.
    """
    paths = route_input_paths(Path(out))
    printed = {key: _printed_path(value) for key, value in paths.items()}
    source_root = _printed_path(data_dir)
    root, tree = "hrrr-root-prep", "hrrr-hierarchy"
    if cycle is None:
        # No lead was resolved by the caller: the model start IS the
        # cycle, which is what every pre-lead emission printed.
        cycle = exp.start_time - timedelta(hours=forecast_start_hour)
    cycle_text = cycle.strftime("%Y-%m-%d_%H:%M:%S")
    lead = ((f"--forecast-start-hour {forecast_start_hour}",)
            if forecast_start_hour else ())
    run_seconds = int(exp.run_seconds)
    cadence = int(exp.domains[0].history_interval_s)
    profile_flag = ((f"--physics-profile {profile}",)
                    if profile is not None else ())
    # The hierarchy stage's corridor flag, on exactly the configs that
    # need it.  Derived from the corridor module's own follow predicate
    # -- the same function `gpuwm go`'s plan and run-plan's decision
    # read -- so a config whose printed chain omits this flag is a
    # config no door would have added it for.  A pasted chain that
    # forgot it would prepare a bundle the last line of the same chain
    # refuses.
    from gpuwm.stage_cli import staged_route_commands
    from gpuwm.static.corridor import config_declares_follow_source

    corridor = (("--statics-corridor",)
                if config_declares_follow_source(exp) else ())
    manifest = f"{source_root}/SHA256SUMS"
    manifest_digest = "<printed by gpuwm fetch>"
    prepare_arguments = (
        f"--source-root {source_root}",
        f"--source-sha256s {manifest}",
        f"--source-sha256s-sha256 {manifest_digest}",
        f"--domain-spec {printed['target_domain']}",
        f"--namelist-input {printed['namelist_input']}",
        # Handed to the PREPARATION, not just to the hierarchy: the
        # forecast stage's HRRR manifest inventory requires a
        # wps_namelist role, and the preparer records the role only if
        # it is given the file.  A chain that omitted it prepared a
        # bundle `gpuwm sim` could not read at all -- which is how HRRR
        # came to be sent to a benchmark script instead of a forecast.
        f"--wps-namelist {printed['wps_namelist']}",
        "--geog-root <your WPS_GEOG>",
        *profile_flag,
        f"--valid-time {cycle_text}",
        *lead,
        f"--run-seconds {run_seconds}",
        f"--history-interval-seconds {cadence}",
    )
    prepare_command, root_forecast = staged_route_commands(
        "hrrr", prep_arguments=prepare_arguments, prepared_root=root,
        outdir="hrrr-forecast", indent="  ", wrap=True)
    prepare = _guard_exports_block(profile) + prepare_command + "\n"
    header = (
        "# HRRR runs the native route -- not `gpuwm run` and not `gpuwm "
        "go`, whose five\n"
        "#   commands do not compose this route's stages.  Two commands "
        "do, and they are\n"
        "#   the shipped ones.  Every input below was written beside this "
        "config; the\n"
        "#   placeholders are your WPS_GEOG and the digest `gpuwm fetch` "
        "printed.\n")
    if len(exp.domains) == 1:
        # The forecast stage is `gpuwm sim`, and that is not a rewording
        # of what stood here.  This block used to print
        # `tools/hrrr_single_domain_benchmark.py` under a comment saying
        # HRRR "does not reach gpuwm.prepared_single_domain_forecast --
        # that runner's --source takes gfs/era5/20crv3".  It does reach
        # it: hrrr is in that runner's SUPPORTED_SOURCES, the
        # preparation publishes proof.json plus the authorities the
        # runner binds on every run, and `gpuwm sim` derives the three
        # digests from them.  A reader was being sent to a benchmark
        # script -- one that lives under tools/ and is therefore absent
        # from every wheel -- for a run the shipped door does.
        return (header + prepare + root_forecast + "\n"
                + "# the second command reads the preparation's own "
                "digests off the bundle;\n"
                "#   nothing is copied by hand.  A ladder with a nest "
                "(--ladder 12-3, or\n"
                "#   --root-dx/--chain) takes one more `gpuwm prep` "
                "between them: the\n"
                "#   hierarchy stage, printed for those configs.")
    hierarchy_arguments = (
        f"--root-preparation {root}",
        f"--domain-spec {printed['target_domain']}",
        f"--wps-namelist {printed['wps_namelist']}",
        f"--namelist-input {printed['namelist_input']}",
        f"--stock-wrf-namelist-input {printed['stock_namelist_input']}",
        "--geog-root <your WPS_GEOG>",
        f"--source-sha256s {manifest}",
        f"--source-sha256s-sha256 {manifest_digest}",
        f"--valid-time {cycle_text}",
        *lead,
        *corridor,
    )
    # The tree runner binds its preparation receipt rather than a
    # namelist digest, so --wps-namelist is left off the forecast line:
    # printing a flag the runner does not read is how a reader learns a
    # printed chain is approximate.
    hierarchy, tree_forecast = staged_route_commands(
        "hrrr", prep_arguments=hierarchy_arguments, prepared_root=tree,
        outdir="hrrr-forecast", experiment_config=_printed_path(out),
        wps_namelist=False, indent="  ", wrap=True)
    return (header + prepare + hierarchy + "\n" + tree_forecast + "\n"
            + "# the last command reads the hierarchy's own preparation "
            "receipt off the\n"
            "#   tree it is pointed at; no digest is copied by hand.")


def final_step_command(out: "Path", *, source: str, profile: str | None,
                       domain_count: int, data_dir: str,
                       case_data: dict | None,
                       exp: ExperimentConfig | None = None,
                       cycle: "datetime | None" = None,
                       forecast_start_hour: int = 0) -> str:
    """Step 3 of the closing block: the command that actually runs THIS file.

    It used to be ``gpuwm run <config>`` for every source, and for GFS
    and HRRR that is a command which refuses by design -- ``gpuwm run``
    executes the ``[case_data]`` config-driven route, which is ERA5's.
    A reader who follows a numbered list to a refusal has been told the
    tool is broken by the tool itself; an owner met exactly that on
    1.3.0.  So the last step names the route the emitted file is on:

    * a ``[case_data]`` config (ERA5): ``gpuwm run`` -- unchanged;
    * a single-domain GFS config bound to a shipped profile:
      ``gpuwm go``, which runs the whole documented chain including the
      fetch above, so it is pointed at that download;
    * anything else on the native route: the chain in FIRST-LIGHT 3a,
      naming the runner that applies, because no single command
      finishes those today.
    """

    printed = _printed_path(out)
    if case_data is not None:
        return f"gpuwm run {printed}"
    if source == "gfs" and domain_count == 1:
        # Bound to a shipped profile or not: the chain runs the suite
        # this file selects, as written (owner ruling 2026-07-31).
        return (f"gpuwm go {printed} --data-dir {_printed_path(data_dir)}"
                "   # authority, front door, forecast and render, in order")
    if source == "hrrr" and exp is not None:
        # BEFORE the multi-domain branch, not after it.  A multi-domain
        # HRRR emission used to fall into the branch below and be told
        # to prepare with rw-wps -- the GFS front door -- because this
        # test ran second.  The HRRR chain is its own, and every file it
        # names was just written beside the config.
        return hrrr_route_commands(
            out, exp, profile=profile, data_dir=data_dir, cycle=cycle,
            forecast_start_hour=forecast_start_hour)
    # Deferred: go_cli owns the pointer (it is the command that refuses
    # most often), and importing it at module scope would make the
    # wizard pay for the whole orchestrator's imports.
    from gpuwm.go_cli import MANUAL_CHAIN

    # The tree route IS installed -- `gpuwm-prepared-tree-forecast` is a
    # declared console script -- and no message named it, so a reader was
    # pointed at a python -m invocation with two bare `...` and at a docs
    # path a pip install does not contain (B-03).  Name the command that
    # exists, say where each value comes from, and give a URL that
    # resolves without a checkout.
    if domain_count > 1:
        # The WHOLE chain, with this emission's real paths filled in.
        # It used to say "prepare it with rw-wps" and print no command
        # for the two stages that come first, so the only road to a
        # ladder run -- the shape every `gpuwm downscale` parent needs,
        # because single-domain emissions disable restart writing -- was
        # reverse-engineering FIRST-LIGHT's single-domain sequence
        # (walked live against the 2.4.1 wheel, 2026-08-17).  Each
        # command below prints the next one filled in, so the reader
        # types the first two and pastes the rest.
        out_path = Path(out)
        authority = out_path.parent / f"{out_path.stem}-authority"
        namelist = out_path.parent / f"{out_path.stem}.namelist.wps"
        profile_flag = ("" if profile is None
                        else f" \\\n#       --physics-profile {profile}")
        return (
            "# this is a " + str(domain_count) + "-domain tree: it runs "
            "stage by stage; each command prints the next one filled "
            "in.\n"
            "#   materialize the physics authority FIRST:\n"
            "#     python -m gpuwm.prepared_single_domain_forecast "
            "--materialize-authorities \\\n"
            f"#       --source {source} --base-experiment-config "
            f"{_printed_path(out)} \\\n"
            f"#       --base-wps-namelist {_printed_path(namelist)}"
            f"{profile_flag} \\\n"
            f"#       --output-directory {_printed_path(authority)}\n"
            "#   author the front-door manifest -- it prints the "
            "complete rw-wps command:\n"
            f"#     gpuwm fetch --source {source} "
            "--author-front-door-manifest \\\n"
            f"#       --out {_printed_path(data_dir)} \\\n"
            f"#       --wps-namelist "
            f"{_printed_path(authority / 'namelist.wps')} \\\n"
            f"#       --experiment-config "
            f"{_printed_path(authority / 'experiment.toml')}\n"
            "#   run the rw-wps line it printed (with your --geog-root); "
            "rw-wps finishes by\n"
            "#   printing the runner command below with both values "
            "filled in:\n"
            "#   gpuwm-prepared-tree-forecast \\\n"
            "#     --prepared-root <the directory rw-wps wrote> \\\n"
            "#     --preparation-receipt-sha256 <the sha256 rw-wps "
            "printed>\n"
            "#   (same runner, if you prefer the module form: "
            "python -m gpuwm.prepared_domain_tree_forecast)\n"
            f"#   the full sequence is {MANUAL_CHAIN}")
    return (
        "# this source runs stage by stage on the native single-domain "
        "route:\n"
        f"#   follow {MANUAL_CHAIN}.  The runner executes the\n"
        "#   suite in this file as written; --physics-profile is an "
        "optional binding.\n"
        "#   (no one-command chain finishes this source yet.)")


#: What ``--source hrrr`` binds when no ``--physics-profile`` is named.
#:
#: Owner directive 2026-08-07 ("i dont want hrrr to be limited"): HRRR
#: gets a nocturnally valid full-radiation default like every other
#: source, and it is chosen to match what an HRRR user already expects.
#: The operational High-Resolution Rapid Refresh runs Thompson
#: aerosol-aware microphysics with RRTMG longwave AND shortwave on a
#: 3 km convection-permitting grid with no cumulus parameterization
#: (NOAA/GSL; the CCPP ``HRRR_suite``).  This profile is that
#: composition as far as the shipped engine carries it -- Thompson mp8,
#: RRTMG 4/4, ``cu_physics = 0`` -- diverging from operations on the two
#: components gpuwm has no route-admissible HRRR implementation for yet
#: (YSU rather than MYNN-EDMF, Noah rather than RUC; both of those
#: shipped profiles are WSM6/no-longwave and are refused by the route's
#: own surface-layer and LSM pins).
#:
#: It replaced :data:`WSM6_PROFILE_ID`, whose rationale here claimed the
#: route "admits ONE physics slice ... ra_lw_physics 0, ra_sw_physics 1".
#: That stopped being true when the route widened to
#: ``ADMITTED_RADIATION_PAIRS = {(0, 1), (4, 4)}`` and
#: ``ADMITTED_PBL_PHYSICS = {1, 11}``; what actually kept a full-radiation
#: default out was that the wizard did not OFFER either legacy-RRTMG
#: Thompson suite and the HRRR root preparer staged no microphysics
#: lookup tables for them.  Both are fixed (WIZARD_PHYSICS_PROFILES
#: above; ``tools/hrrr_single_domain_benchmark._microphysics_table_
#: authority``), so the constraint is gone rather than worked around.
#:
#: The RTE+RRTMGP suites (Morrison, NSSL-2) remain outside the HRRR
#: route on ``cu_physics = 1``: Kain-Fritsch at HRRR's native 3 km would
#: parameterize convection the grid already resolves.  That refusal is
#: physics, not a limitation, and it is why the HRRR default is not the
#: gfs/era5 default.
#:
#: Read from :mod:`gpuwm.hrrr_route_inputs`, never restated: that module
#: owns what the route admits, and the 1.7.1 battery proved a
#: door-local copy of a default is a door-local bug waiting for the next
#: flip.
HRRR_DEFAULT_PROFILE = ROUTE_DEFAULT_PHYSICS_PROFILE


def resolved_physics_profile(source: str, requested: str | None
                             ) -> str | None:
    """The profile this emission actually binds.

    An explicit ``--physics-profile`` always wins -- including one the
    HRRR routes will refuse, which is refused at emission with the
    switch named rather than silently replaced.

    The DEFAULT is derived, not tabled.  This function used to read
    ``if source == "hrrr": return HRRR_DEFAULT_PROFILE`` and hand every
    other source the gfs/era5 default -- which is a branch on a model
    name, and it only answered correctly because exactly one source had
    ever had a route gate written for it.  A second gated source would
    have been handed a default its own route refuses, and a bare run on
    it could not start.

    :func:`gpuwm.physics_menu.default_profile_for` computes it instead:
    the first suite in this module's listed order that the source's
    route admits and that runs both radiation streams.  That reproduces
    both defaults this product shipped -- ``tests/test_physics_menu.py``
    binds the native HRRR route's answer to
    :data:`gpuwm.hrrr_route_inputs.ROUTE_DEFAULT_PHYSICS_PROFILE` and
    the rest to :data:`DEFAULT_PHYSICS_PROFILE` -- and it is what lets a
    source registered tomorrow get a working default as table work.
    """
    if requested is not None:
        return requested
    from gpuwm.physics_menu import default_profile_for

    return default_profile_for(source)


def profile_switches(profile: str | None) -> dict:
    """Every physics switch for PROFILE, or for the default suite."""

    if profile is None:
        return dict(DEFAULT_SUITE_PHYSICS)
    return single_domain_runtime_switches(profile)


def physics_summary(profile: str | None, *,
                    cu_physics: int | None = None) -> str:
    """One line naming what the emitted suite actually runs.

    ``cu_physics`` overrides the suite's cumulus switch with the one the
    EMISSION carries.  The wizard retires the scheme on a
    convection-permitting root (:func:`_domain_tables`), and a summary
    line that reads "Kain-Fritsch cumulus" over a file whose root says
    ``cu_physics = 0`` is the same class of misreading hazard that
    :func:`_radiation_words` exists to remove -- worse here, because the
    line and the table it describes sit in the same file.  Callers
    describing a SUITE rather than an emission leave it None.
    """

    switches = profile_switches(profile)
    selected = (int(switches["cu_physics"]) if cu_physics is None
                else int(cu_physics))
    cumulus = ({1: "Kain-Fritsch cumulus",
                3: "Grell-Freitas cumulus"}.get(selected,
                                                "parameterized cumulus")
               if selected
               else "NO cumulus parameterization")
    label = profile if profile is not None else (
        "product default suite (supported, not yet WRF-verified; every "
        "runner executes it as written)")
    return (f"{label}: mp_physics {switches['mp_physics']}, "
            f"{_radiation_words(switches)} (radt "
            f"{float(switches['radt']):g} min), {cumulus}, "
            f"bl_pbl_physics {switches['bl_pbl_physics']}, "
            f"sf_surface_physics {switches['sf_surface_physics']}")


def shared_physics(profile: str | None) -> dict:
    """The ``[shared]`` block for one physics suite.

    A named profile is taken from :mod:`gpuwm.physics_compat`, never
    restated here: the prepared single-domain forecast runner compares
    an experiment's switches to that same registry for exact equality,
    so an emitted config passes its guard by construction.
    """

    switches = profile_switches(profile)
    for key in _PER_DOMAIN_PHYSICS:
        switches.pop(key, None)
    return {**_SHARED_GRID_AND_DYNAMICS, **switches}


#: The default suite's [shared] block.
_SHARED_CERTIFIED = shared_physics(DEFAULT_PHYSICS_PROFILE)


class DomainFitError(ValueError):
    """The requested ladder cannot fit the requested card."""


#: Appended to every --point parse refusal: the one form that cannot be
#: mis-parsed no matter how the shell or argparse feels about a leading
#: minus sign.
_POINT_FORM_HINT = (
    " -- southern and western points are ordinary here; both "
    "'--point -33.87,151.21' and '--point=-33.87,151.21' are accepted")


def _parse_point(raw: str) -> tuple[float, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError("--point must be lat,lon in decimal degrees"
                         + _POINT_FORM_HINT)
    try:
        lat, lon = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError(
            "--point must be lat,lon in decimal degrees"
            + _POINT_FORM_HINT) from error
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("--point coordinates must be finite")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(
            f"--point latitude {lat:g} must lie within [-90, 90]")
    if abs(lat) == 90.0:
        raise ValueError(
            f"--point latitude {lat:g} is the pole itself; a domain "
            "containing the pole is unsupported (lat-lon source "
            "interpolation and static-tile windowing are not "
            "pole-capable) -- move --point off the pole")
    if not -180.0 <= lon <= 180.0:
        if -360.0 <= lon <= 360.0:
            # float(): _wrap180 is array code (np.where) and returns a 0-d
            # ndarray even for a scalar.  Without the cast the wrapped
            # longitude stays an ndarray all the way into the emitted
            # artifacts, where _toml_value quotes it as a STRING and the
            # namelist's {...!r} renders it `array(-160.)` -- neither of
            # which is the number the user asked for.  The --polygon
            # sibling below already casts; this path never did.
            wrapped = float(_wrap180(lon))
            warn(f"--point longitude {lon:g} wrapped to {wrapped:g} "
                 "(the [-180, 180] convention this project uses)")
            lon = wrapped
        else:
            # Name the range actually enforced.  One wrap is accepted, so
            # claiming [-180, 180] here described a refusal that does not
            # happen: --point 40,270 is taken, with a warning.
            raise ValueError(
                f"--point longitude {lon:g} must lie within [-180, 180] "
                "(or one wrap within [-360, 360])")
    return lat, lon


@dataclass(frozen=True)
class PolygonFootprint:
    """Validated local GeoJSON rings on their minimum longitude branch.

    GeoJSON positions are ``longitude, latitude``.  ``west`` and ``east``
    are deliberately unwrapped around ``center_lon``; their difference is
    therefore the small circular span even when the footprint crosses the
    antimeridian.
    """

    path: Path
    rings: tuple[tuple[tuple[float, float], ...], ...]
    south: float
    west: float
    north: float
    east: float
    center_lat: float
    center_lon: float

    @property
    def longitude_span(self) -> float:
        return self.east - self.west


_POLYGON_TYPES = ("Polygon", "MultiPolygon")
_POLYGON_SAMPLE_STEP_DEG = 0.25
_POLYGON_MAX_SAMPLES = 2_000_000
_POLYGON_FIT_SLACK_CELLS = 1.0


def _geojson_position(value, label: str,
                      wrapped: list[tuple[float, float]]) \
        -> tuple[float, float]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(
            f"--polygon {label} must be a GeoJSON position [lon, lat]")
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        raise ValueError(
            f"--polygon {label} must contain numeric longitude/latitude")
    try:
        lon, lat = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"--polygon {label} must contain numeric longitude/latitude"
        ) from error
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise ValueError(f"--polygon {label} coordinates must be finite")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(
            f"--polygon {label} latitude {lat:g} must lie within [-90, 90]")
    if abs(lat) == 90.0:
        raise ValueError(
            f"--polygon {label} reaches the pole itself; a domain containing "
            "a pole is unsupported (lat-lon source interpolation and "
            "static-tile windowing are not pole-capable)")
    if not -180.0 <= lon <= 180.0:
        if not -360.0 <= lon <= 360.0:
            raise ValueError(
                f"--polygon {label} longitude {lon:g} must lie within "
                "[-180, 180] (or one wrap within [-360, 360])")
        normalized = float(_wrap180(lon))
        wrapped.append((lon, normalized))
        lon = normalized
    return lon, lat


def _geojson_ring(value, label: str,
                  wrapped: list[tuple[float, float]]) \
        -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 4:
        raise ValueError(
            f"--polygon {label} must be a linear ring with at least four "
            "positions")
    ring = tuple(_geojson_position(position, f"{label}[{index}]", wrapped)
                 for index, position in enumerate(value))
    first_lon, first_lat = ring[0]
    last_lon, last_lat = ring[-1]
    if first_lat != last_lat or abs(float(
            _wrap180(last_lon - first_lon))) > 1e-12:
        raise ValueError(
            f"--polygon {label} is not closed (its first and last positions "
            "must match)")
    distinct = {(float(lon % 360.0), lat) for lon, lat in ring[:-1]}
    if len(distinct) < 3:
        raise ValueError(
            f"--polygon {label} needs at least three distinct positions")
    branch = first_lon + np.asarray(_wrap180(
        np.asarray([position[0] for position in ring], dtype=float)
        - first_lon))
    latitudes = np.asarray([position[1] for position in ring], dtype=float)
    twice_area = float(np.sum(
        branch[:-1] * latitudes[1:] - branch[1:] * latitudes[:-1]))
    if abs(twice_area) <= 1e-14:
        raise ValueError(f"--polygon {label} encloses zero area")
    return ring


def _geojson_polygon(value, label: str,
                     wrapped: list[tuple[float, float]]) \
        -> list[tuple[tuple[float, float], ...]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"--polygon {label} has no linear rings")
    return [_geojson_ring(ring, f"{label}[{index}]", wrapped)
            for index, ring in enumerate(value)]


def _geojson_geometry(value, label: str,
                      wrapped: list[tuple[float, float]]) \
        -> list[tuple[tuple[float, float], ...]]:
    if not isinstance(value, dict):
        raise ValueError(f"--polygon {label} must be a GeoJSON geometry")
    kind = value.get("type")
    if kind == "Polygon":
        return _geojson_polygon(value.get("coordinates"),
                                f"{label}.coordinates", wrapped)
    if kind == "MultiPolygon":
        polygons = value.get("coordinates")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"--polygon {label} has no polygons")
        rings = []
        for index, polygon in enumerate(polygons):
            rings.extend(_geojson_polygon(
                polygon, f"{label}.coordinates[{index}]", wrapped))
        return rings
    raise ValueError(
        f"--polygon {label} type {kind!r} is unsupported; accepted geometry "
        f"types are {', '.join(_POLYGON_TYPES)}")


def _geojson_rings(document: object,
                   wrapped: list[tuple[float, float]]) \
        -> tuple[tuple[tuple[float, float], ...], ...]:
    if not isinstance(document, dict):
        raise ValueError("--polygon must contain one GeoJSON object")
    crs = document.get("crs")
    if crs not in (None, {}):
        properties = crs.get("properties", {}) if isinstance(crs, dict) \
            else {}
        name = str(properties.get("name", "")).upper()
        code = properties.get("code")
        longitude_latitude = (
            name in {"EPSG:4326", "OGC:CRS84", "CRS84"}
            or name.endswith(":CRS84")
            or name.endswith(":EPSG::4326")
            or code in {4326, "4326"}
        )
        if not longitude_latitude:
            raise ValueError(
                "--polygon declares a custom coordinate reference system; "
                "GeoJSON longitude/latitude coordinates are required")
    kind = document.get("type")
    if kind in _POLYGON_TYPES:
        rings = _geojson_geometry(document, "geometry", wrapped)
    elif kind == "Feature":
        rings = _geojson_geometry(document.get("geometry"),
                                  "feature.geometry", wrapped)
    elif kind == "FeatureCollection":
        features = document.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("--polygon FeatureCollection has no features")
        rings = []
        for index, feature in enumerate(features):
            if not isinstance(feature, dict) \
                    or feature.get("type") != "Feature":
                raise ValueError(
                    f"--polygon features[{index}] must be a GeoJSON Feature")
            rings.extend(_geojson_geometry(
                feature.get("geometry"), f"features[{index}].geometry",
                wrapped))
    else:
        raise ValueError(
            f"--polygon top-level type {kind!r} is unsupported; use Polygon, "
            "MultiPolygon, Feature, or FeatureCollection")
    if not rings:
        raise ValueError("--polygon contains no polygon coordinates")
    return tuple(rings)


def _minimum_longitude_arc(longitudes) -> tuple[float, float, float]:
    """Return ``(center, west, east)`` on the minimum circular arc."""

    values = np.unique(np.mod(np.asarray(longitudes, dtype=np.float64),
                              360.0))
    if not values.size:
        raise ValueError("--polygon contains no longitude coordinates")
    extended = np.concatenate((values, values[:1] + 360.0))
    gap_index = int(np.argmax(np.diff(extended)))
    start = float(extended[gap_index + 1])
    end = float(extended[gap_index] + 360.0)
    span = end - start
    if span > 180.0 + 1e-10:
        raise ValueError(
            f"--polygon minimum longitude footprint spans {span:.1f} "
            "degrees; footprints wider than 180 degrees cannot be served "
            "as one source crop")
    center = float(_wrap180((0.5 * (start + end)) % 360.0))
    on_branch = center + np.asarray(
        _wrap180(np.asarray(longitudes, dtype=np.float64) - center))
    west, east = float(on_branch.min()), float(on_branch.max())
    # At an exact 180-degree tie either semicircle is legal.  Pin the
    # numerically reconstructed span to the law above rather than allowing
    # a roundoff-scale false refusal.
    if east - west > 180.0 + 1e-10:
        raise ValueError(
            f"--polygon minimum longitude footprint spans {east - west:.1f} "
            "degrees; footprints wider than 180 degrees cannot be served "
            "as one source crop")
    return center, west, east


def load_polygon_footprint(path: str | Path) -> PolygonFootprint:
    """Read and validate a local Polygon-family GeoJSON document."""

    raw_path = str(path)
    if "://" in raw_path or raw_path.lower().startswith("file:"):
        raise ValueError(
            "--polygon accepts a local GeoJSON file path, not a URL")
    polygon_path = Path(path)
    if not polygon_path.is_file():
        # The sentence names the path THIS PROCESS looked at, absolute,
        # plus the directory a relative one was resolved against.
        #
        # It used to echo the argument as typed.  A caller that passes a
        # relative path and runs the wizard in a different working
        # directory than its own -- which is what every stage-runner
        # subprocess does -- then produced "local GeoJSON file does not
        # exist: danow\case\domain-box.geojson" for a file that was on
        # disk, and sent the user hunting for it.  A refusal that states
        # something the user can check and find false is worse than no
        # refusal: absolute here means the claim is always true, and
        # `cd`-shaped bugs identify themselves on the first read.
        resolved = polygon_path.expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:  # pragma: no cover - unresolvable path shapes
            resolved = polygon_path
        where = ""
        if not polygon_path.is_absolute():
            where = (f" (relative to the working directory "
                     f"{Path.cwd()})")
        kind = ("is not a regular file"
                if polygon_path.exists() else "does not exist")
        raise ValueError(
            f"--polygon local GeoJSON file {kind}: {resolved}{where}")
    try:
        document = json.loads(polygon_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"--polygon local GeoJSON file could not be read: "
            f"{polygon_path} ({error})") from error
    except UnicodeDecodeError as error:
        raise ValueError(
            f"--polygon {polygon_path} is not UTF-8 text") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"--polygon {polygon_path} is not valid JSON: "
            f"line {error.lineno}, column {error.colno}") from error

    wrapped: list[tuple[float, float]] = []
    rings = _geojson_rings(document, wrapped)
    if wrapped:
        example, normalized = wrapped[0]
        warn(f"--polygon wrapped {len(wrapped)} longitude coordinate(s) "
             f"into [-180, 180] (for example {example:g} to "
             f"{normalized:g})")
    positions = [position for ring in rings for position in ring]
    lons = np.asarray([position[0] for position in positions], dtype=float)
    lats = np.asarray([position[1] for position in positions], dtype=float)
    center_lon, west, east = _minimum_longitude_arc(lons)
    south, north = float(lats.min()), float(lats.max())
    return PolygonFootprint(
        path=polygon_path, rings=rings, south=south, west=west,
        north=north, east=east, center_lat=0.5 * (south + north),
        center_lon=center_lon)


def _polygon_samples(footprint: PolygonFootprint, *,
                     max_step_deg: float = _POLYGON_SAMPLE_STEP_DEG) \
        -> tuple[np.ndarray, np.ndarray]:
    """Densify GeoJSON segments on the footprint's longitude branch."""

    if not math.isfinite(max_step_deg) or max_step_deg <= 0.0:
        raise ValueError("polygon sampling step must be finite and positive")
    sample_lons: list[float] = []
    sample_lats: list[float] = []
    for ring in footprint.rings:
        lons = footprint.center_lon + np.asarray(_wrap180(
            np.asarray([point[0] for point in ring], dtype=float)
            - footprint.center_lon))
        lats = np.asarray([point[1] for point in ring], dtype=float)
        for index in range(len(ring) - 1):
            lon0, lon1 = float(lons[index]), float(lons[index + 1])
            lat0, lat1 = float(lats[index]), float(lats[index + 1])
            steps = max(1, int(math.ceil(max(abs(lon1 - lon0),
                                               abs(lat1 - lat0))
                                         / max_step_deg)))
            if len(sample_lons) + steps + 1 > _POLYGON_MAX_SAMPLES:
                raise ValueError(
                    "the polygon and requested grid spacing require more "
                    f"than {_POLYGON_MAX_SAMPLES:,} containment samples; "
                    "choose a coarser --root-dx, fewer refinement levels, "
                    "or a simpler polygon")
            for fraction in np.arange(steps, dtype=float) / steps:
                sample_lons.append(lon0 + fraction * (lon1 - lon0))
                sample_lats.append(lat0 + fraction * (lat1 - lat0))
        sample_lons.append(float(lons[-1]))
        sample_lats.append(float(lats[-1]))
    return (np.asarray(sample_lats, dtype=float),
            np.asarray(sample_lons, dtype=float))


def parse_level_buffers(raw: str | None) -> tuple[float, ...] | None:
    """Parse comma-separated outer-to-inner buffer distances in km."""

    if raw is None:
        return None
    fields = str(raw).split(",")
    if not fields or any(not field.strip() for field in fields):
        raise ValueError(
            "--buffer-km must be one distance or a comma-separated distance "
            "per domain level")
    values = []
    for field in fields:
        try:
            value = float(field)
        except ValueError as error:
            raise ValueError(
                f"--buffer-km entry {field!r} is not a number") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--buffer-km entry {field!r} must be finite and "
                "nonnegative")
        values.append(value)
    return tuple(values)


def _buffers_for_levels(values: tuple[float, ...] | None,
                        count: int) -> tuple[float, ...]:
    if values is None:
        return (0.0,) * count
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            f"--buffer-km supplies {len(values)} distances but the selected "
            f"ladder has {count} domain levels; supply one distance to use "
            "at every level or exactly one outer-to-inner distance per level")
    return values


def _resolve_cycle(raw: str, *, source: str, hours: int,
                   start_hour: int = 0) -> datetime:
    """Parse ``--cycle``, resolving ``latest`` the way ``fetch`` does.

    v1.0.0 refused ``--cycle latest`` here with a message that said
    ``latest`` was allowed, and since the documented order is
    wizard-then-fetch there was nothing to tell a user which cycle was
    current -- they had to run a throwaway fetch first.  The resolver
    already existed; the wizard now calls it, and says what it picked.

    ``start_hour`` is the forecast lead the run will START at, so
    ``latest`` resolves to a cycle complete through the END of the
    window (lead + length) rather than through its length alone.
    """

    if raw.strip().lower() != "latest":
        return parse_cycle(raw, source)
    # NO PER-SOURCE BRANCH HERE.  This door used to refuse era5 by name
    # ("a reanalysis with weeks of latency") and everything without a
    # fetch front door by another, so `latest` was a capability three
    # models had.  It is now one question asked of the source's declared
    # initialization grid, and the refusal for a source that declares
    # none is that resolver's own -- which names the missing declaration
    # rather than a list this door would have to keep in step.
    from gpuwm.fetch import resolve_latest_cycle
    try:
        cycle = resolve_latest_cycle(source, start_hour + hours)
    except (RuntimeError, OSError) as error:
        raise ValueError(
            f"--cycle latest could not be resolved for {source}: {error}"
            " -- the resolver probes the public mirrors, so this needs "
            "network access; pass an explicit YYYY-MM-DDTHH (UTC) cycle "
            "instead") from error
    # "complete" is a claim a PROBE earns.  A source with no object to
    # probe resolves from its declared publication delay, and saying
    # "complete" there would attest to a check nothing ran.
    from gpuwm.fetch import cycle_is_probeable

    standing = ("newest complete" if cycle_is_probeable(source)
                else "newest published")
    print(f"gpuwm domain: --cycle latest resolved to "
          f"{cycle:%Y-%m-%dT%H}Z ({standing} {source} cycle "
          f"covering f{start_hour + hours:03d})")
    return cycle


def _even(value: float) -> int:
    return max(2, 2 * round(value / 2.0))


def _dims_for_scale(scale: float, ratios: tuple[int, ...]
                    ) -> list[tuple[int, int]]:
    """Mass dimensions per domain at ``scale`` (root 110 x 88 at 1.0).

    Everything even so centered children register exactly; child mass
    dimensions are span * ratio, satisfying the loader's divisibility and
    clearance rules by construction (still re-validated by the loader).
    """
    dims = [(_even(110.0 * scale), _even(88.0 * scale))]
    for depth, ratio in enumerate(ratios):
        pnx, pny = dims[-1]
        spans = []
        for parent_extent in (pnx, pny):
            span = _even(_child_span_fraction(depth) * parent_extent)
            span = min(span, parent_extent - 2 * _CLEARANCE_ROWS)
            if span < 12:
                raise DomainFitError(
                    f"parent extent {parent_extent} cannot host a nest "
                    f"with {_CLEARANCE_ROWS}-row clearance at scale "
                    f"{scale:g}")
            spans.append(span)
        dims.append((spans[0] * ratio, spans[1] * ratio))
    return dims


def radt_ladder_minutes(root_radt_minutes: float,
                        domains: int) -> list[float]:
    """``radt`` per emitted domain, outer to inner: the root's, inherited.

    A nest INHERITS its parent's radiation cadence.  Radiative transfer
    varies on cloud timescales, not on grid scales, so nothing about
    halving dx makes a shorter radiation interval more correct -- and
    WRF's own namelist guidance says so outright: set ``radt`` once for
    the coarsest domain and use the same value for every nest.

    Until 2.5.0 this was ``max(1.0, dx_km)`` per nest, which was wrong in
    both directions and expensive in one:

    * Under the 12-minute suites (every RRTMGP/RRTMG profile the wizard
      offers) the 12-3-1-0.5 ladder emitted 12/3/1/1 -- radiation once a
      simulated MINUTE on both sub-km rungs, measured at 79% of a real
      1 km run's wall clock.  The floor also flattened the bottom of the
      ladder: 1 km and 500 m were handed the same 1.0, so the refinement
      it was charging for had already stopped.  At the 250 m LES target
      this program exists for -- 12-3-1-0.5-0.25 -- it taxed three rungs
      out of five.
    * Under the ``radt = 1.0`` suites (the mp8 validation profile and the
      four no-radiation profiles) it ran the other way and emitted 3.0 on
      the 3 km nest: a CHILD calling radiation three times less often
      than the parent feeding its boundaries.

    Inheritance ships default-ON and takes no flag: an opt-in remedy for
    a correctness defect is a workaround, not a fix.  Per-domain ``radt``
    stays overridable in the emitted TOML for anyone who wants a nest to
    depart deliberately.
    """

    return [float(root_radt_minutes)] * int(domains)


def radiation_cadence_advisory(profile: str | None,
                               domains: int) -> list[str]:
    """The spoken half of :func:`radt_ladder_minutes`: one line, or none.

    "Fixed means default" ships a remedy default-on WITH an advisory, and
    the inheritance rule earns its one line only where the AUTO
    derivation actually decides something a reader could mistake: a
    NESTED emission.  Layered like the gray-zone advisories: the default
    screen carries a compact clause on the physics line domain_main
    already prints (the one-screen cap is a measured gate), and this
    full line prints under --explain.  It names the single cadence
    every domain runs, says the nests inherit
    it -- the pre-2.5.0 wizard refined it with dx instead, flooring
    sub-km nests at 1-minute radiation, measured at 79% of a real run's
    wall clock -- and points at the per-domain ``radt`` key in the
    emitted config, which is the ONE user override the wizard honours
    (it takes no radt flag, and an explicit value in the file always
    wins at load).

    Silent for a single domain: nothing inherits, and the physics
    summary already speaks the root's radt.  There is no radiation-off
    condition because no shipped suite is radiation-off: every profile
    in the registry runs at least shortwave (the ``*-no-radiation-*``
    names mean longwave OFF, Dudhia shortwave still on, radt = 1), so
    radt paces real work in every nested emission.
    """

    if int(domains) < 2:
        return []
    switches = profile_switches(profile)
    return [
        f"radt {float(switches['radt']):g} min on all {int(domains)} "
        "domains: nests inherit the root's radiation cadence rather than "
        "refining it with dx (pre-2.5.0 the wizard floored sub-km nests "
        "at 1-minute radiation -- 79% of a measured run's wall clock); "
        "set a domain's radt in the emitted config to depart "
        "deliberately"]


#: Step the hosting-scale scan walks upward by.  Small enough that the
#: layout it returns is within 5% of the smallest one that hosts the
#: ladder, coarse enough that the whole scan is under a hundred
#: iterations of integer arithmetic.
_HOSTING_SCALE_STEP = 1.05


def _min_hosting_scale(ratios: tuple[int, ...]) -> float:
    """Smallest scale in the bracket whose layout can host ``ratios``.

    ``_MIN_SCALE``'s comment claims it "still hosts the deepest ladder"
    -- true of the deepest PRESET ladder, which is three nests.  A
    custom ``--chain`` can ask for more, and each level spends both a
    span fraction and a fixed :data:`_CLEARANCE_ROWS` boundary margin,
    so a four-nest chain of ratio-2 refinements runs out of interior at
    the 60x48 root ``_MIN_SCALE`` bottoms out at.  The fit loop probed
    exactly that scale first, so those ladders were refused outright
    even though a larger root hosts them comfortably -- the search never
    looked.

    Hosting is monotone in scale (a larger parent has more interior), so
    the first scale that works is the floor of the whole feasible range,
    and returning it lets the existing bisection do the rest.  For every
    preset -- and for any chain of three nests or fewer -- this returns
    ``_MIN_SCALE`` unchanged, so nothing that fitted before moves.

    Refuses, with the depth and the remedy named, when the ladder cannot
    be hosted anywhere in the bracket.
    """
    scale = _MIN_SCALE
    while scale <= _MAX_SCALE:
        try:
            _dims_for_scale(scale, ratios)
        except DomainFitError:
            scale *= _HOSTING_SCALE_STEP
            continue
        return scale
    raise DomainFitError(
        f"a ladder of {len(ratios)} nests "
        f"({'-'.join(f'{v:g}' for v in _ladder_dx_km(ratios))} km at a "
        "12 km root) cannot be hosted at any layout this wizard will "
        f"consider: every level spends {_CLEARANCE_ROWS} parent rows of "
        "Davies/blend clearance on each side plus its share of the "
        "parent's interior, and by the innermost level there are fewer "
        f"than 12 cells left even at the {_MAX_SCALE:g}x scale ceiling. "
        "Ask for fewer nests, or reach the same spacing in fewer steps "
        "with larger --chain ratios")


def seconds_per_km(ref_lat: float) -> Fraction:
    """The clock convention that applies at REF_LAT, in s per km of dx.

    5 s/km outside the tropics, halved to 2.5 s/km inside them -- see
    :data:`TROPICAL_ROOT_TIME_STEP_S` for the measurement behind it.
    """

    tropical = abs(float(ref_lat)) < MERCATOR_MAX_LAT
    return Fraction(5, 2) if tropical else Fraction(5)


def root_time_step_s(ref_lat: float,
                     root_dx_m: float = ROOT_DX_M) -> Fraction:
    """The exact root clock for a domain centred at REF_LAT.

    Returns a :class:`~fractions.Fraction`: an arbitrary ``--root-dx``
    in the tropics can land on a half second (3 km -> 7.5 s), which the
    WRF rational clock keys represent exactly.
    """

    return seconds_per_km(ref_lat) * Fraction(float(root_dx_m)) / 1000


def _clock_keys(dt: Fraction) -> dict[str, int]:
    """WRF's rational clock keys for an exact root time step."""

    whole = dt.numerator // dt.denominator
    remainder = dt - whole
    keys = {"time_step": int(whole)}
    if remainder:
        keys["time_step_fract_num"] = remainder.numerator
        keys["time_step_fract_den"] = remainder.denominator
    return keys


def snap_cadences_to_clock(time_step: Fraction | int | float,
                           physics: dict
                           ) -> tuple[dict, tuple[str, ...]]:
    """Whole-step minute cadences for a derived root clock: (physics, notes).

    The wizard derives BOTH sides of the cadence check: ``--root-dx``
    fixes dt through the s-per-km convention, and the profile fixes
    ``radt``/``cudt_minutes``.  At ``--root-dx 9`` those meet as dt =
    45 s against cudt = 300 s, 300/45 = 20/3 steps, and the loader
    rightly refuses fractional-step cadences -- so the wizard exited 2
    over arithmetic the user supplied no part of (UX finding N14).  The
    author reconciles its own derivation instead: each profile cadence
    that is not a whole number of root steps moves to the NEAREST
    whole-step cadence, and the move is spoken.

    Hand-written configs are untouched -- the loader's refusal in
    :mod:`gpuwm.experiment` still stands wherever a USER pinned an
    incompatible pair, because there both numbers are the user's.

    The snapped value must survive the round trip the loader takes:
    ``Fraction(float(minutes)) * 60 / dt`` has to land on a whole
    number, and not every whole-step cadence has minutes a float can
    carry exactly (17 steps of a tropical 17.5 s clock is 297.5 s =
    4.9583... min).  So the nearest step count whose minutes round-trip
    exactly is taken -- one exists within a few steps for every clock
    the s-per-km convention can produce, because dt's denominator is a
    power of two times the km value's own binary fraction.
    """

    dt = Fraction(time_step)
    adjusted = dict(physics)
    notes: list[str] = []
    for key in ("radt", "cudt_minutes"):
        minutes = adjusted.get(key)
        if minutes is None or float(minutes) <= 0.0:
            continue  # 0 = every step (WRF convention); nothing to snap
        if key == "cudt_minutes" and int(adjusted.get("cu_physics", 0)) != 1:
            continue  # the loader paces cudt only under Kain-Fritsch
        seconds = Fraction(float(minutes)) * 60
        steps = seconds / dt
        if steps.denominator == 1:
            continue
        target = max(1, round(float(steps)))
        chosen = None
        for offset in range(0, 64):
            for count in ((target,) if offset == 0
                          else (target - offset, target + offset)):
                if count < 1:
                    continue
                snapped_minutes = float(count * dt / 60)
                if Fraction(snapped_minutes) * 60 == count * dt:
                    chosen = (count, snapped_minutes)
                    break
            if chosen is not None:
                break
        if chosen is None:  # pragma: no cover - no wizard clock reaches this
            continue  # leave the pair for the loader's refusal
        count, snapped_minutes = chosen
        adjusted[key] = snapped_minutes
        notes.append(
            f"{key} adjusted {float(minutes):g} -> {snapped_minutes:g} "
            f"min: the profile's {float(seconds):g} s is {steps} steps "
            f"of the derived {float(dt):g} s root clock, not a whole "
            f"number, and the loader refuses fractional-step cadences; "
            f"{float(count * dt):g} s = {count} steps is the nearest "
            f"cadence that is")
    return adjusted, tuple(notes)


def _domain_tables(dims: list[tuple[int, int]],
                   ratios: tuple[int, ...],
                   *, time_step: Fraction | int = ROOT_TIME_STEP_S,
                   root_dx_m: float = ROOT_DX_M,
                   profile: str | None = DEFAULT_PHYSICS_PROFILE,
                   cumulus_requested: bool = False,
                   history_interval_s: float | None = None,
                   nest_history_interval_s: float | None = None
                   ) -> list[dict]:
    """[[domain]] table dicts (centered children, certified cadences).

    The ROOT's radiation/cumulus/diffusion cadences come from the shipped
    physics profile, so the emitted d01 satisfies the prepared-forecast
    runner's exact-equality guard at any --root-dx.  Nests keep the
    certified ladder's depth-varying ``diff_6th_factor`` and their pinned
    ``cu_physics = 0``: those two really are grid-scale decisions, and
    the multi-domain runner has no profile whitelist to stop them.

    ROOT ``cu_physics`` IS ONE OF THOSE GRID-SCALE DECISIONS TOO, below
    the convection-permitting bound.  Until 2.5.0 the root took the
    profile's cumulus switch at every spacing, so `--root-dx 3` emitted
    Kain-Fritsch on a grid that resolves its own deep convection, printed
    the sentence naming the heating and rainfall it therefore counts
    twice, and wrote the file anyway -- a defect the product diagnosed
    correctly and then shipped.  The default now follows the diagnosis:
    below :data:`CUMULUS_CONVECTION_PERMITTING_DX_KM` the root's cumulus
    scheme is OFF, and ``cudt_minutes`` goes to the registry's own
    spelling for a domain with no scheme to pace (0.0, as every
    ``cu_physics = 0`` template in :mod:`gpuwm.physics_compat` carries).
    That bound is not a new number: it is the one
    :func:`cumulus_gray_zone_advisory` already judged the emission
    against, so the switch and its advisory read the same declaration.

    ``cumulus_requested`` is the user having NAMED the suite.  Naming
    ``--physics-profile`` asserts the config IS that shipped suite --
    :func:`gpuwm.gfs_direct.front_door_physics_selection` enforces it
    switch for switch on both routes -- so a named suite is emitted
    verbatim at any spacing and keeps the advisory, which is the whole
    point of the advisory being advisory.  Per-domain ``cu_physics``
    stays overridable in the emitted TOML either way.

    ``radt`` is NOT one of them.  It is the root's, inherited by every
    nest (:func:`radt_ladder_minutes`) -- radiation varies on cloud
    timescales, not grid scales.  Refining it with dx is what the
    ``max(1.0, dx_km)`` rule did until 2.5.0, and it cost 79% of a
    measured 1 km run's wall clock for no science.

    EVERY domain, root and nest alike, gets the profile's ``epssm``.
    Until 2026-08-01 the nests were written ``epssm = 0.1`` while the
    root took 0.5 from the profile, and that one line was the whole of
    a reported nested-forecast blow-up: epssm is the vertical-acoustic
    off-centering coefficient, 0.1 is nearly centred, and the nest is
    exactly where the terrain is steepest -- a wizard 3 km -> 750 m
    ladder over the Cascades grew w from -15 to -289 to -976 m/s to
    non-finite in seven acoustic substeps at the steepest cell in the
    child (34.6 degrees; the 3 km parent smooths the same peak to 15.7
    and survives).  Setting the nest to the profile's 0.5 and changing
    nothing else ran both geometries clean for the full two hours.

    The 0.1 was not invented here -- it is WRF's Registry default, and
    it is what an IMPORTED namelist legitimately produces: ``epssm =
    0.5`` in a namelist.input assigns d01 only, so the shipped
    real74 ladders carry 0.5/0.1/0.1/0.1 and are right to
    (:mod:`gpuwm.namelist_import`, which reads the Registry column).
    The wizard is an author, not an importer, and copying an
    importer's per-domain tail is how the value got here.  Terrain-
    scaled epssm -- more off-centering where slopes are steepest -- is
    a real future refinement; it is deliberately NOT attempted here.
    Per-domain ``epssm`` stays overridable in the emitted TOML.
    """
    root_physics = {key: profile_switches(profile)[key]
                    for key in _PER_DOMAIN_PHYSICS}
    if (int(root_physics.get("cu_physics", 0))
            and not cumulus_requested
            and convection_permitting(float(root_dx_m) / 1000.0)):
        root_physics["cu_physics"] = 0
        root_physics["cudt_minutes"] = 0.0
    # The author reconciles its own two derivations (dt from --root-dx,
    # cadences from the profile) rather than emitting a file the loader
    # refuses: see snap_cadences_to_clock (UX finding N14).  It runs
    # AFTER the cumulus decision because cudt is paced only under an
    # active Kain-Fritsch, and the decision above may have retired it.
    root_physics, _ = snap_cadences_to_clock(
        Fraction(time_step), root_physics)
    epssm = profile_switches(profile)["epssm"]
    radt = radt_ladder_minutes(root_physics["radt"], len(dims))
    tables = []
    for index, (nx, ny) in enumerate(dims):
        if index == 0:
            table = {
                "grid_id": 1, "parent_id": 0, "i_parent_start": 1,
                "j_parent_start": 1, "parent_grid_ratio": 1,
                "parent_time_step_ratio": 1, "nx": nx, "ny": ny,
                **_clock_keys(Fraction(time_step)),
                "dx": float(root_dx_m),
                "specified": True, "nested": False,
                "history_interval_s": (
                    DEFAULT_ROOT_HISTORY_INTERVAL_S
                    if history_interval_s is None
                    else float(history_interval_s)),
                **root_physics,
            }
        else:
            ratio = ratios[index - 1]
            pnx, pny = dims[index - 1]
            table = {
                "grid_id": index + 1, "parent_id": index,
                "i_parent_start": (pnx - nx // ratio) // 2 + 1,
                "j_parent_start": (pny - ny // ratio) // 2 + 1,
                "parent_grid_ratio": ratio,
                "parent_time_step_ratio": ratio, "nx": nx, "ny": ny,
                "specified": False, "nested": True,
                "history_interval_s": (
                    DEFAULT_NEST_HISTORY_INTERVAL_S
                    if nest_history_interval_s is None
                    else float(nest_history_interval_s)),
                "epssm": epssm,
                "radt": radt[index], "cu_physics": 0,
                "diff_6th_factor": _diff6_factor(index),
            }
        tables.append(table)
    return tables


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, (list, tuple)):
        if len(value) > 5:  # long numeric arrays: 5 per line
            lines = ["["]
            for start in range(0, len(value), 5):
                chunk = ", ".join(
                    _toml_value(v) for v in value[start:start + 5])
                lines.append(f"    {chunk},")
            lines.append("]")
            return "\n".join(lines)
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, (str, Path)):
        return '"' + str(value).replace("\\", "/") + '"'
    scalar = _builtin_scalar(value)
    if scalar is not None:
        return _toml_value(scalar)
    # Everything used to land in the quoted branch, so a numeric value of
    # any type this function did not enumerate was emitted as a STRING
    # into a typed config file, silently.  That is how a wrapped --point
    # longitude shipped as ref_lon = "-160.0".  An unrenderable value is
    # now a loud failure at emission rather than a quiet mistyping.
    raise TypeError(
        f"cannot render {value!r} ({type(value).__name__}) into TOML: it "
        "is neither a scalar, a string/path, nor a list of those.  "
        "Quoting it would emit a value of the wrong TYPE under the right "
        "key, which reads as valid TOML and is not what was computed.")


def _builtin_scalar(value):
    """Python bool/int/float for a numpy scalar or 0-d array, else None.

    numpy's scalar types are not all builtins -- np.float64 subclasses
    float but np.float32 does not, and a 0-d ndarray subclasses nothing
    -- so array code returning "a number" can hand this module something
    no isinstance branch above matches.
    """
    if isinstance(value, np.generic) or (
            isinstance(value, np.ndarray) and value.ndim == 0):
        item = value.item()
        if isinstance(item, (bool, int, float)):
            return item
    return None


def _render_table(name: str, entries: dict, array_of_tables: bool = False,
                  comment: str | None = None) -> str:
    header = f"[[{name}]]" if array_of_tables else f"[{name}]"
    lines = ([f"# {comment}"] if comment else []) + [header]
    for key, value in entries.items():
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _ladder_dx_km(ratios: tuple[int, ...],
                  root_dx_m: float = ROOT_DX_M) -> list[float]:
    chain = [float(root_dx_m) / 1000.0]
    for ratio in ratios:
        chain.append(chain[-1] / ratio)
    return chain


def auto_projection(lat: float) -> str:
    """The auto-selected WPS projection for a point latitude."""
    if abs(lat) < MERCATOR_MAX_LAT:
        return "mercator"
    if abs(lat) <= LAMBERT_MAX_LAT:
        return "lambert"
    return "polar"


def _projection_entries(lat: float, lon: float,
                        choice: str = "auto") -> dict:
    """[projection] table for the point.

    ``auto`` selects by |lat| (:func:`auto_projection`).  Lambert:
    hemisphere-correct secant cone bracketing the point, truelats at
    |lat| +/- 10 deg clamped to [15, 70] and signed with the point's
    hemisphere.  Mercator: true at the point's latitude (stand_lon is
    recorded but does not enter Mercator math, module_llxy semantics).
    Polar stereographic: the point's hemisphere pole, scale true at the
    point's latitude, stand_lon at the point.
    """
    map_proj = auto_projection(lat) if choice == "auto" else choice
    if map_proj == "lambert":
        sign = -1.0 if lat < 0.0 else 1.0
        alat = abs(lat)
        return {
            "map_proj": "lambert", "ref_lat": lat, "ref_lon": lon,
            "truelat1": sign * round(max(15.0, alat - 10.0), 2),
            "truelat2": sign * round(min(70.0, alat + 10.0), 2),
            "stand_lon": lon,
        }
    if map_proj in ("mercator", "polar"):
        return {
            "map_proj": map_proj, "ref_lat": lat, "ref_lon": lon,
            "truelat1": round(lat, 2), "truelat2": round(lat, 2),
            "stand_lon": lon,
        }
    raise ValueError(
        f"--projection {map_proj!r} is not implemented (choices: auto, "
        "lambert, mercator, polar). Regular/rotated latitude-longitude "
        "needs angular dx/dy rather than metre spacing and WRF's "
        "global/pole polar filter; rotated grids also need "
        "pole_lat/pole_lon state and the map_proj == 6 curvature branch.")


#: Bounds on a custom root dx (km).  Wide, because the point of
#: --root-dx is that the presets are not the whole product; narrow
#: enough that a typo (metres for kilometres, say) is caught.
MIN_ROOT_DX_KM, MAX_ROOT_DX_KM = 0.05, 200.0
#: Bounds on one custom nest ratio.  WRF's own guidance is odd ratios of
#: 3 or 5; 2 and 4 are routine here, and beyond 8 the interpolation
#: stencil and the boundary blend stop being defensible in one step.
MIN_CHAIN_RATIO, MAX_CHAIN_RATIO = 2, 8
#: Most nests a custom chain may declare (the presets go to 4 domains).
MAX_CHAIN_DEPTH = 8


def parse_chain(raw: str) -> tuple[int, ...]:
    """``"4,3,3"`` -> ``(4, 3, 3)``; each entry an integer nest ratio."""

    text = str(raw).strip()
    if not text:
        return ()
    ratios = []
    for field in text.split(","):
        field = field.strip()
        try:
            ratio = int(field)
        except ValueError:
            raise ValueError(
                f"--chain entry {field!r} is not an integer; --chain is a "
                "comma-separated list of whole nest ratios, e.g. "
                "--chain 4,3,3") from None
        if ratio < MIN_CHAIN_RATIO:
            raise ValueError(
                f"--chain ratio {ratio} is below {MIN_CHAIN_RATIO}; a "
                "ratio-1 child is not a refinement")
        if ratio > MAX_CHAIN_RATIO:
            warn(f"--chain ratio {ratio} exceeds the blessed maximum of "
                 f"{MAX_CHAIN_RATIO}; continuing with it as written",
                 why="WRF's own guidance is odd ratios of 3 or 5; beyond "
                     f"{MAX_CHAIN_RATIO} the interpolation stencil and "
                     "the boundary blend are undemonstrated in one step "
                     "-- refining in more steps is the proven route.")
        ratios.append(ratio)
    if len(ratios) > MAX_CHAIN_DEPTH:
        warn(f"--chain declares {len(ratios)} nests, more than the "
             f"{MAX_CHAIN_DEPTH} any configuration has demonstrated; "
             "continuing")
    return tuple(ratios)


def parse_custom_ladder(*, root_dx_km, chain, ladder: str):
    """``(root_dx_m, ratios)`` for a custom ladder, or None for a preset.

    ``--root-dx``/``--chain`` are the general form of ``--ladder``: an
    arbitrary root spacing and an arbitrary chain of integer refinement
    ratios.  Everything downstream -- the estimator fit loop, the
    clearance and cadence rules in the real experiment loader, the
    projection math, ``gpuwm check`` -- is the same code the presets go
    through, so a custom ladder is validated exactly as strictly.
    """

    if root_dx_km is None and chain is None:
        return None
    if ladder != "auto":
        raise ValueError(
            "--ladder is a preset chain and cannot be combined with "
            "--root-dx / --chain; drop --ladder to use the custom form")
    root_km = (ROOT_DX_M / 1000.0 if root_dx_km is None
               else float(root_dx_km))
    if not math.isfinite(root_km) or root_km <= 0.0:
        raise ValueError(
            f"--root-dx {root_km:g} km must be a positive spacing")
    if not MIN_ROOT_DX_KM <= root_km <= MAX_ROOT_DX_KM:
        warn(f"--root-dx {root_km:g} km is outside the expected "
             f"[{MIN_ROOT_DX_KM:g}, {MAX_ROOT_DX_KM:g}] km window "
             "(check the unit -- this flag takes kilometres); continuing")
    ratios = parse_chain("" if chain is None else chain)
    return root_km * 1000.0, ratios


#: Grid spacing (km) below which a 1-D PBL parameterization and resolved
#: convection overlap -- the "terra incognita" / gray zone.
GRAY_ZONE_DX_KM = 1.0


def gray_zone_advisory(chain_km, shared: dict) -> list[str]:
    """One honest sentence when a domain lands in the PBL gray zone.

    Advisory, never a refusal: sub-kilometre nests are exactly what this
    product is for, and people will run them.  But a 1-D column PBL
    scheme assumes the whole boundary-layer eddy spectrum is
    subgrid-scale, and below about 1 km the largest eddies are partly
    resolved, so the scheme and the dynamics do the same transport
    twice.  Saying so once, in the file and on stdout, is the honest
    thing; refusing would be wrong, and silence would be worse.

    THE RECIPE CARRIES ``mix_isotropic = 1``, and that is a correctness
    repair rather than a wording preference.  This sentence is the only
    place the product tells anybody how to configure turbulence below a
    kilometre, so it is where the recommended configuration is actually
    chosen.  It used to name ``km_opt`` and ``bl_pbl_physics`` and stop
    there -- but ``mix_isotropic`` defaults to 0 (WRF's Registry value,
    :class:`gpuwm.config.RunConfig`), and with ``km_opt`` 2 or 3 that is
    the per-axis path where the vertical exchange coefficient is built
    and capped on the LAYER DEPTH and then handed to the horizontal
    diffusion of ``w``.  A reader who followed the old recipe at 250 m
    landed on ``mix_upper_bound*(dz_max/dx)^2 = 0.702`` against a limit
    of 0.25 -- the tier where the operator amplifies a 2-grid-interval
    mode instead of damping it -- and found out only if they later read
    stderr at config load.  Recommending a configuration and separately
    warning about it is not a fix; the recipe now is one that holds.

    SINCE THE AUTO-SWITCH (Drew, 2026-08-16) the recipe's key is also
    the running default: a config that leaves ``mix_isotropic`` unset
    and violates the criterion runs isotropic anyway
    (``gpuwm.experiment.resolve_auto_mix_isotropic``, announced at load
    and in ``gpuwm check``).  The recipe keeps NAMING the key so the
    file a reader authors says what it runs;
    ``gpuwm.config.warn_anisotropic_w_mixing`` still only advises when a
    config WRITES ``mix_isotropic = 0`` -- an explicit setting is kept,
    and refusing it would make the frozen crash records unloadable.
    """

    from gpuwm.config import EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT

    if not shared.get("bl_pbl_physics"):
        return []
    below = [dx for dx in chain_km if dx < GRAY_ZONE_DX_KM]
    if not below:
        return []
    finest = min(below)
    return [
        f"GRAY ZONE: {len(below)} domain(s) refine below "
        f"{GRAY_ZONE_DX_KM:g} km (finest {finest * 1000:.0f} m) with the "
        f"1-D PBL scheme bl_pbl_physics = {shared['bl_pbl_physics']} "
        "active, so boundary-layer eddies are partly resolved by the "
        "dynamics and simultaneously parameterized as if they were not. "
        "The proper tool at these scales is a 3-D turbulence closure with "
        "the PBL off: set km_opt = 3 (3-D Smagorinsky) or km_opt = 2 "
        "(1.5-order prognostic TKE) with bl_pbl_physics = 0 AND "
        "mix_isotropic = 1 on the "
        "domain(s) below the gray zone -- both are per-domain, so a PBL "
        "parent can carry a PBL-off child (see docs/public/LES.md) -- or "
        "the SASE closure (bl_pbl_physics = 900), which is implemented "
        "and selectable but EXPERIMENTAL and not WRF-verified. "
        "mix_isotropic = 1 is part of that recipe and not an optional "
        "extra: km_opt 2 and 3 with mix_isotropic = 0 (WRF's default; "
        "left unset, ArWen auto-selects 1 where the criterion below "
        "fails, and says so) build "
        "the vertical exchange coefficient on the LAYER DEPTH and then "
        "hand it to the horizontal diffusion of w, and the reachable "
        "mix_upper_bound*(dz_max/dx)^2 rises as the grid narrows while "
        "the layers do not -- it reads 0.702 on this project's own 250 m "
        "LES child against a limit of "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT}, which is the tier where "
        "the horizontal mixing of w amplifies a 2-grid-interval mode "
        "instead of damping it. Choose it at t = 0: mix_isotropic is "
        "inside the RunConfig restart fingerprint, so it cannot be "
        "changed part-way through a campaign you intend to resume. Until "
        "you do, treat sub-kilometre PBL structure as indicative rather "
        "than quantitative.",
    ]


#: Grid spacing (km) below which operational convection-permitting
#: practice turns the cumulus parameterization OFF: the dynamics resolve
#: deep convection at these spacings, and an active scheme convects the
#: same air a second time.
CUMULUS_CONVECTION_PERMITTING_DX_KM = 4.0

#: Upper edge (km) of the convective gray zone.  Between
#: :data:`CUMULUS_CONVECTION_PERMITTING_DX_KM` and this spacing deep
#: convection is neither fully subgrid (the closure assumption every
#: cumulus scheme makes) nor well resolved (the convection-permitting
#: assumption) -- the genuine gray zone, where running the scheme is
#: common operational practice and still worth one honest sentence.
CUMULUS_GRAY_ZONE_TOP_DX_KM = 10.0


def convection_permitting(dx_km: float) -> bool:
    """True where the dynamics resolve deep convection themselves.

    ONE predicate off :data:`CUMULUS_CONVECTION_PERMITTING_DX_KM`, read
    by both the switch the wizard EMITS (:func:`_domain_tables`) and the
    sentence that judges it (:func:`cumulus_gray_zone_advisory`), so the
    emission and its own advisory can never disagree about where the
    bound is -- which is how the wizard came to print "counted twice"
    about a file it had just written.
    """

    return float(dx_km) < CUMULUS_CONVECTION_PERMITTING_DX_KM


def cumulus_retired_note(profile: str | None, root_dx_km: float, *,
                         cumulus_requested: bool) -> list[str]:
    """One sentence when the wizard turned the suite's cumulus OFF.

    The emission changed a switch the named suite carries, so it says
    so, in the file and on the screen, with the bound and the way back.
    Silence here would be the mirror of the defect this replaced: a file
    that does not run what its own PHYSICS line claims.
    """

    scheme = int(profile_switches(profile).get("cu_physics", 0))
    if (not scheme or cumulus_requested
            or not convection_permitting(root_dx_km)):
        return []
    return [
        f"CUMULUS OFF AT {float(root_dx_km):g} KM: the derived suite "
        f"carries cu_physics = {scheme} and this root sits below the "
        f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km "
        "convection-permitting bound, so the emitted root runs "
        "cu_physics = 0 (cudt_minutes = 0.0) instead of convecting the "
        "same air twice -- the dynamics resolve deep convection at this "
        "spacing.  Every other switch is the suite's; name the suite "
        "with --physics-profile to emit it verbatim, cumulus included, "
        "or set cu_physics yourself in the emitted [[domain]] table."]


def cumulus_retired_headline(profile: str | None, root_dx_km: float, *,
                             cumulus_requested: bool) -> list[str]:
    """The retirement's first clause -- same contract as
    :func:`cumulus_gray_zone_headline`: derived from the same call, so a
    headline that could disagree with the sentence cannot exist."""

    return [line.split(", so ", 1)[0] + "."
            for line in cumulus_retired_note(
                profile, root_dx_km, cumulus_requested=cumulus_requested)]


def cumulus_gray_zone_advisory(chain_km, cu_physics_by_domain
                               ) -> list[str]:
    """Honest sentences when an active cumulus scheme meets fine grids.

    Advisory, never a refusal -- the same contract, channel and tone as
    :func:`gray_zone_advisory`: the full sentence lives in the emitted
    file's header comment, stdout carries the finding (headline) by
    default and the whole sentence under ``--explain``.  At most one
    sentence per finding, strong one first:

    * below ~4 km the dynamics resolve deep convection, so an active
      scheme double-counts it; operational convection-permitting
      practice runs ``cu_physics = 0`` there.
    * in the 4-10 km band convection is neither fully subgrid nor well
      resolved -- the genuine gray zone.  Running the scheme there is
      common operational practice, so the note is softer.

    ``cu_physics_by_domain`` is per-domain (outer to inner, same order
    as ``chain_km``) because cumulus is a per-domain switch here, not a
    ``[shared]`` one -- the wizard activates it on the root only.

    Scale awareness is honoured per scheme, as this docstring always
    promised it would be.  Grell-Freitas (``cu_physics = 3``) carries
    its own ``sig = (1-frh)**2`` taper -- the parameterized contribution
    withdraws as dx approaches cloud-resolving spacing, which is the
    scheme's whole design point -- so a GF domain in the 4-10 km gray
    zone earns no advisory, and below 4 km it earns the SOFT sentence
    (the taper is the scheme's own answer there, but no ArWen-vs-WRF
    trajectory receipt backs it yet, so the wording says measured
    restraint rather than silence).  Classic KF (``cu_physics = 1``) is
    not scale-aware and keeps both sentences.
    """

    pairs = [(float(dx), int(cu)) for dx, cu
             in zip(chain_km, cu_physics_by_domain, strict=True)]
    active = [(dx, cu) for dx, cu in pairs if cu]
    lines: list[str] = []
    scale_aware_below = [(dx, cu) for dx, cu in active
                         if cu == 3 and convection_permitting(dx)]
    if scale_aware_below:
        finest = min(dx for dx, _ in scale_aware_below)
        lines.append(
            f"CUMULUS: {len(scale_aware_below)} domain(s) run "
            "Grell-Freitas (cu_physics = 3) at convection-permitting "
            f"spacing below {CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km "
            f"(finest {finest:g} km); the scheme's own sig = (1-frh)^2 "
            "taper withdraws the parameterized contribution as the grid "
            "resolves convection, so this is the scheme answering the "
            "gray zone rather than double-counting it -- but no "
            "ArWen-versus-WRF trajectory receipt backs GF yet "
            "(implemented-unverified), so verify convective placement "
            "against observations before trusting it there.")
    below = [(dx, cu) for dx, cu in active
             if cu != 3 and convection_permitting(dx)]
    if below:
        finest = min(dx for dx, _ in below)
        switch = "/".join(str(cu) for cu
                          in sorted({cu for _, cu in below}))
        lines.append(
            f"CUMULUS: {len(below)} domain(s) run at "
            "convection-permitting spacing below "
            f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km (finest "
            f"{finest:g} km) with the cumulus parameterization "
            f"cu_physics = {switch} active, so deep convection is "
            "resolved by the dynamics and parameterized by the scheme "
            "at the same time -- its heating and rainfall counted "
            "twice; operational convection-permitting practice runs "
            "cu_physics = 0 below about "
            f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g} km and lets the "
            "resolved dynamics convect (per-domain override in the "
            "emitted [[domain]] tables).")
    band = [(dx, cu) for dx, cu in active
            if cu != 3
            and not convection_permitting(dx)
            and dx <= CUMULUS_GRAY_ZONE_TOP_DX_KM]
    if band:
        finest = min(dx for dx, _ in band)
        switch = "/".join(str(cu) for cu
                          in sorted({cu for _, cu in band}))
        lines.append(
            f"CUMULUS GRAY ZONE: {len(band)} domain(s) sit in the "
            f"{CUMULUS_CONVECTION_PERMITTING_DX_KM:g}-"
            f"{CUMULUS_GRAY_ZONE_TOP_DX_KM:g} km convective gray zone "
            f"(finest {finest:g} km) with cu_physics = {switch} "
            "active, so deep convection is neither fully subgrid nor "
            "well resolved and the scheme's closure assumptions only "
            "partly hold; this pairing is common operational practice "
            "-- keep it if it is deliberate, and read convective "
            "placement and intensity on those domains as indicative "
            "rather than quantitative.")
    return lines


def cumulus_by_domain(dims, ratios, *,
                      profile: str | None,
                      root_dx_m: float = ROOT_DX_M,
                      cumulus_requested: bool = False) -> list[int]:
    """``cu_physics`` per emitted domain, outer to inner.

    Read from the same ``[[domain]]`` tables the emitted file carries
    (:func:`_domain_tables`: root from the profile unless the spacing
    resolves its own convection, nests pinned to 0), never re-derived,
    so the advisory cannot disagree with the emission.

    ``root_dx_m`` used to be left at its default here, on the reasoning
    that it "cannot change which cumulus switch a domain carries".  That
    reasoning ended when the root's switch became convection-permitting
    aware: pass the emission's own spacing, or this reports the 12 km
    answer for a 3 km file.  ``time_step`` genuinely cannot change a
    cumulus switch and is still left alone.
    """

    return [int(table["cu_physics"])
            for table in _domain_tables(
                dims, ratios, profile=profile, root_dx_m=root_dx_m,
                cumulus_requested=cumulus_requested)]


def _root_grid(projection: dict, nx: int, ny: int,
               root_dx_m: float = ROOT_DX_M):
    cls = projection_class(projection["map_proj"])
    return cls(
        ref_lat=projection["ref_lat"], ref_lon=projection["ref_lon"],
        truelat1=projection["truelat1"], truelat2=projection["truelat2"],
        stand_lon=projection["stand_lon"],
        dx=float(root_dx_m), dy=float(root_dx_m),
        e_we=nx + 1, e_sn=ny + 1)


def _footprint_contains_pole(projection: dict, nx: int, ny: int,
                             root_dx_m: float = ROOT_DX_M) -> bool:
    """Does this root contain (or nearly touch) the projection pole?

    The predicate behind :func:`_pole_clearance_refusal`, split out
    because a second caller needs to ASK it without refusing: a
    pole-containing footprint spans every longitude, so it also trips
    the 180-degree servable-crop bound, and the crop bound's remedy
    ("size a narrower domain") is the wrong instruction for a domain
    whose problem is the singularity inside it.
    """

    if projection["map_proj"] == "mercator":  # never reaches a pole
        return False
    grid = _root_grid(projection, nx, ny, root_dx_m)
    pole_lat = 90.0 if projection["truelat1"] >= 0.0 else -90.0
    px, py = (float(v) for v in grid.latlon_to_ij(
        pole_lat, projection["stand_lon"]))
    margin = _POLE_CLEARANCE_CELLS
    return bool(0.5 - margin <= px <= grid.e_we - 0.5 + margin
                and 0.5 - margin <= py <= grid.e_sn - 0.5 + margin)


def _pole_clearance_refusal(projection: dict, nx: int, ny: int,
                            root_dx_m: float = ROOT_DX_M,
                            target_option: str = "--point") -> None:
    """Refuse a root footprint that contains (or nearly touches) the
    projection pole -- a genuine pipeline limit (lat-lon source
    interpolation and static windowing are not pole-capable), not a
    projection-math one.  Mercator never reaches a pole."""
    if _footprint_contains_pole(projection, nx, ny, root_dx_m):
        pole_lat = 90.0 if projection["truelat1"] >= 0.0 else -90.0
        raise ValueError(
            f"the fitted root domain ({nx} x {ny} mass points at "
            f"{float(root_dx_m) / 1000:g} km) contains or touches the "
            f"{'north' if pole_lat > 0 else 'south'} pole; lat-lon "
            "source interpolation and static-tile windowing are not "
            f"pole-capable -- move {target_option} away from the pole or "
            "choose "
            "a smaller layout (--vram-gib / a shallower --ladder)")


def _margined_longitude_span(lon_c, center: float,
                             margin_deg: float) -> tuple["np.ndarray", float]:
    """Root longitudes on the branch nearest ``center``, and their span.

    ONE expression of the arithmetic, because two readers depend on it
    agreeing with itself: :func:`_fetch_area`, which refuses to emit a
    box wider than one source crop, and :func:`fetch_crop_refusal`,
    which is what stops the fit loop from choosing that layout in the
    first place.  A sizing bound computed one way and an emission gate
    computed another is how a wizard sizes a domain for twelve seconds
    and then refuses the file it just sized.
    """

    lon_u = center + np.asarray(
        _wrap180(np.asarray(lon_c, dtype=float) - center))
    span = ((float(lon_u.max()) + margin_deg)
            - (float(lon_u.min()) - margin_deg))
    return lon_u, span


def fetch_crop_refusal(projection: dict, nx: int, ny: int, *, source: str,
                       root_dx_m: float = ROOT_DX_M,
                       target_option: str = "--point") -> str | None:
    """Why SOURCE cannot force this root as ONE crop, or ``None``.

    The same bound :func:`_fetch_area` enforces at emission, asked early
    enough that :func:`fit_ladder` can shrink against it -- a sizing
    constraint that is not the card, exactly like
    :func:`source_coverage_refusal`.

    The breakage it names is specific and was measured, not theorized: a
    forcing box whose MARGINED longitude span exceeds 180 degrees is
    read back by :func:`gpuwm.fetch.parse_area` as the complementary
    antimeridian-crossing box -- the wrong crop, silently -- so
    ``_fetch_area`` refuses to write one.  Until this bound joined the
    fit, the fit was free to choose a layout the emission would then
    refuse: on Linux, where the peak envelope carries no WDDM floor and
    the same card therefore buys a much larger grid, ``gpuwm domain
    --point=35.3,-97.5 --source gfs --ladder 12 --vram-gib 32`` sized a
    181.2-degree footprint and died with rc 2 and no layout to fall back
    to.  The identical command on Windows emitted a 102 x 67 degree box.
    Shrinking is the answer the refusal's own remedy asks for ("shrink
    the configuration"), and the wizard is the thing that knows by how
    much.
    """

    margin_deg = _fetch_margin_deg(source)
    _, lon_c = _root_grid(projection, nx, ny, root_dx_m).latlon_c()
    _, span = _margined_longitude_span(
        lon_c, float(projection["ref_lon"]), margin_deg)
    if span <= 180.0:
        return None
    return (
        f"the {nx}x{ny} root's forcing box spans {span:.1f} degrees of "
        f"longitude once {source}'s {margin_deg:g}-degree fetch margin is "
        "added, and boxes wider than 180 degrees cannot be served as a "
        f"single source crop.  Move {target_option} equatorward, or size a "
        "narrower domain (--vram-gib N, or a finer --root-dx)")


def _fetch_area(projection: dict, nx: int, ny: int,
                margin_deg: float = _FETCH_MARGIN_DEG,
                *, notes: list[str] | None = None,
                root_dx_m: float = ROOT_DX_M,
                target_option: str = "--point",
                coverage: tuple[float, float, float, float] | None = None,
                coverage_label: str = "the source grid",
                coverage_notes: list[str] | None = None,
                ) -> tuple[float, float, float, float]:
    """Forcing bbox (S, W, N, E) = root corners + margin, worldwide.

    Corner longitudes are unwrapped onto the branch nearest the
    reference longitude, so a footprint straddling the antimeridian
    yields a continuous span; the emitted box then wraps back into the
    signed convention, producing W > E for a crossing box (the
    ``gpuwm fetch`` contract; NOMADS and CDS both consume it).  The
    latitude edges clamp to [-90, 90].  A footprint wider than 180
    degrees of longitude is refused (genuine limit: no source crop can
    serve it as one box).  The refusal is checked on the MARGINED span
    -- the margin is part of the emitted box, and a box whose margined
    width exceeds 180 degrees would be read back by
    :func:`gpuwm.fetch.parse_area` as the complementary
    antimeridian-crossing box (the wrong crop, silently).

    ``coverage`` is a source's own ``(S, W, N, E)`` lat/lon envelope --
    :func:`gpuwm.fetch.source_coverage_envelope`'s data, never a source
    name -- and the box is clamped INTO it, quantized inward to the
    hint's printed precision so the formatted string cannot round back
    out of coverage.  Clamp rather than shrink the domain, and clamp
    rather than refuse: by emission time the DOMAIN is already bounded
    by source coverage (the fit loop keeps every HRRR candidate's
    interpolation window inside the native grid), so what overruns here
    is only the margined lat/lon bbox -- a projection artifact of a
    Lambert footprint's curved edges plus the fetch margin, not data
    the preparation needs; for a coverage-boxed source ``--area`` is a
    coverage check, not a crop.  The field exhibit: a 1234 x 986 root
    at 3 km (39, -98) fits the HRRR grid with rows to spare, yet its
    margined bbox named 54.39 N -- north of anything HRRR carries --
    and the wizard's own printed fetch refused to run.  Every clamp is
    disclosed through ``coverage_notes``.
    """
    lat_c, lon_c = _root_grid(projection, nx, ny, root_dx_m).latlon_c()
    center = float(projection["ref_lon"])
    lon_u, span = _margined_longitude_span(lon_c, center, margin_deg)
    if span > 180.0:
        raise ValueError(
            f"the root domain's forcing footprint spans {span:.1f} "
            "degrees of longitude; boxes wider than 180 degrees cannot "
            "be served as a single source crop -- shrink the "
            f"configuration or move {target_option} equatorward")
    # Pole-clear, not pole-touching: the box a user is handed must not
    # name the singularity the pipeline refuses (see POLE_CLEARANCE_DEG).
    pole_clear = max_fetch_abs_lat(root_dx_m)
    lat_s = max(-pole_clear, float(lat_c.min()) - margin_deg)
    lat_n = min(pole_clear, float(lat_c.max()) + margin_deg)
    if notes is not None:
        for edge, raw, clamped in (
                ("south", float(lat_c.min()) - margin_deg, lat_s),
                ("north", float(lat_c.max()) + margin_deg, lat_n)):
            if raw != clamped:
                notes.append(
                    f"the suggested forcing box's {edge} edge was clamped "
                    f"from {raw:.2f} to {clamped:.2f} to stay "
                    f"{pole_clearance_deg(root_dx_m):.2f} deg clear of "
                    "the pole")
    lon_w = float(_wrap180(float(lon_u.min()) - margin_deg))
    lon_e = float(_wrap180(float(lon_u.max()) + margin_deg))
    if lon_e == -180.0:
        lon_e = 180.0
    if coverage is not None and lon_w <= lon_e:
        # A crossing box (W > E) cannot lie inside a non-crossing
        # envelope; it is left for the emission-time proof to refuse
        # loudly rather than silently reshaped here.
        from gpuwm.fetch import area_bounds_inward

        cov_s, cov_w, cov_n, cov_e = area_bounds_inward(coverage)
        clamped_box = {
            "south": (lat_s, max(lat_s, cov_s)),
            "west": (lon_w, max(lon_w, cov_w)),
            "north": (lat_n, min(lat_n, cov_n)),
            "east": (lon_e, min(lon_e, cov_e)),
        }
        for edge, (raw, clamped) in clamped_box.items():
            if raw != clamped and coverage_notes is not None:
                coverage_notes.append(
                    f"the suggested forcing box's {edge} edge was "
                    f"clamped from {raw:.2f} to {clamped:.2f}: "
                    f"{coverage_label} coverage ends there (grid "
                    f"envelope lat {cov_s:.2f}..{cov_n:.2f}, "
                    f"lon {cov_w:.2f}..{cov_e:.2f})")
        lat_s, lat_n = clamped_box["south"][1], clamped_box["north"][1]
        lon_w, lon_e = clamped_box["west"][1], clamped_box["east"][1]
        if lat_s >= lat_n or lon_w >= lon_e:
            raise ValueError(
                f"the root domain's forcing footprint lies outside "
                f"{coverage_label} coverage (grid envelope lat "
                f"{cov_s:.2f}..{cov_n:.2f}, lon {cov_w:.2f}..{cov_e:.2f})"
                " entirely; choose a source whose coverage includes "
                f"{target_option}")
    return lat_s, lon_w, lat_n, lon_e


def fetch_area_hint(projection: dict, nx: int, ny: int, *, source: str,
                    root_dx_m: float = ROOT_DX_M,
                    target_option: str = "--point",
                    notes: list[str] | None = None,
                    coverage_notes: list[str] | None = None) -> str:
    """The exact ``--area`` string the wizard prints and writes, for SOURCE.

    One seam for the emission and its tests: the fitted root's forcing
    box (root corners + the source's own margin), bounded by the
    source's coverage envelope
    (:func:`gpuwm.fetch.source_coverage_envelope` -- itself derived
    from the native grid definition, the same data the fetch guard
    enforces), formatted at the fixed precision the fetch parser reads
    back.  Every emission is then round-tripped through
    :func:`gpuwm.fetch.validate_fetch_hints` before the file is
    written, so a hint this function produces and a command ``gpuwm
    fetch`` refuses cannot coexist -- the field defect this closes.
    """

    from gpuwm.fetch import AREA_HINT_DECIMALS, source_coverage_envelope

    area = _fetch_area(
        projection, nx, ny, margin_deg=_fetch_margin_deg(source),
        notes=notes, root_dx_m=root_dx_m, target_option=target_option,
        coverage=source_coverage_envelope(source),
        coverage_label=source.upper(), coverage_notes=coverage_notes)
    return ",".join(f"{value:.{AREA_HINT_DECIMALS}f}" for value in area)


def source_coverage_refusal(projection: dict, nx: int, ny: int, *,
                            source: str,
                            root_dx_m: float = ROOT_DX_M,
                            target_option: str = "--point") -> str | None:
    """Why SOURCE's native grid cannot force this root, or ``None``.

    THE plan-time answer to the question the 2026-08-17 model battery could
    only get out of a preparation traceback.  ICON-EU over a central-US
    domain is a refusal by construction -- a European grid cannot reach
    Kansas -- and the run learned it after decoding 1,752 objects, 73
    seconds into a preparation, as ten lines of internal call stack.  The
    facts needed to say it first are all in the registry row: the source's
    declared window, and this root's own mass points.

    The message is built to be the same three sentences the preparation
    stage gets right: WHICH point is outside, WHERE it lands in the
    source's own index space, and WHAT the source covers.  That is what
    separates "this source does not reach the target" from "the crop is
    too small", and it is the difference between moving the domain and
    filing a bug.

    A global source (no declared window) returns ``None``: it reaches
    everything, and there is no bound to state.
    """

    window = source_coverage_window(source)
    if window is None:
        return None
    latitude, longitude = _root_grid(
        projection, nx, ny, root_dx_m).latlon_c()
    outside = points_outside(window, latitude, longitude)
    if not bool(outside.any()):
        return None
    index = int(np.argmax(outside))
    bad_lat = float(np.asarray(latitude).reshape(-1)[index])
    bad_lon = float(np.asarray(longitude).reshape(-1)[index])
    centre_lat, centre_lon = window_centre(window)
    return (
        f"the {nx}x{ny} root's point at lat/lon ({bad_lat:.4f}, "
        f"{_wrap180(bad_lon):.4f}) {window.locate(bad_lat, bad_lon)}; "
        f"{int(outside.sum())} of {outside.size} root mass points are "
        f"outside it.  {source}'s grid is centred at "
        f"({centre_lat:.2f}, {centre_lon:.2f}) -- move {target_option} "
        f"inside that grid, shrink the ladder, or choose a source whose "
        f"coverage includes this domain")


def _posix(path) -> str:
    return str(path).replace("\\", "/")


def _printed_path(path) -> str:
    """A path inside a printed command, quoted if a shell would split it.

    ``--out C:/my domains/case`` is a valid destination and was printed
    bare, so the "next:" command -- whose entire value is that it can be
    pasted -- became two arguments the moment it was.  Ordinary paths
    come back unquoted.
    """

    return shlex.quote(_posix(path))


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return _posix(os.path.relpath(path, base))
    except ValueError:  # different drive on Windows
        return _posix(Path(path).resolve())


def declared_nocturnal_night(profile: str | None, *, start_time: datetime,
                             hours: int, projection: dict):
    """First local night of this window IF the suite forces a declaration.

    ``None`` when nothing is declared: either the profile runs both
    radiation streams, or the window is all daylight.  A ``datetime``
    means :func:`render_config` will write
    ``acknowledgements = [ASYMMETRIC_RADIATION_NOCTURNAL_ACK]`` into the
    emitted ``[experiment]`` -- which disarms the load-time guard
    (:func:`gpuwm.physics_compat.nocturnal_radiation_refusal`) at every
    other front door for this file.

    One function, two readers: the emission rule in
    :func:`render_config` and the spoken advisory in
    :func:`domain_main`.  They were allowed to be two expressions of the
    same predicate exactly once, and the result was a wizard that wrote
    the declaration and said nothing about it -- so the only statement of
    a nocturnally invalid run was a comment inside a file the reader had
    no reason to open.
    """

    shared = shared_physics(profile)
    lw = int(shared.get("ra_lw_physics", shared.get("ra_physics", 0)))
    sw = int(shared.get("ra_sw_physics", shared.get("ra_physics", 0)))
    if not (sw > 0 and lw == 0):
        return None
    return first_local_night_time(
        start_time, float(hours * 3600),
        ref_lat=projection["ref_lat"], ref_lon=projection["ref_lon"])


def render_config(*, name: str, start_time: datetime, hours: int,
                  projection: dict, dims: list[tuple[int, int]],
                  ratios: tuple[int, ...],
                  fetch_hints: dict, case_data: dict | None,
                  root_dx_m: float = ROOT_DX_M,
                  profile: str | None = DEFAULT_PHYSICS_PROFILE,
                  cumulus_requested: bool = False,
                  interactive: bool = False,
                  level_buffers_km: tuple[float, ...] | None = None,
                  history_interval_s: float | None = None,
                  nest_history_interval_s: float | None = None,
                  acknowledgements: tuple[str, ...] = ()) -> str:
    """The emitted TOML text (the exact bytes the wizard validates).

    ``acknowledgements`` is written into ``[experiment]`` verbatim and is
    the ONLY source of that field.  Until 2026-08-09 this function wrote
    ``[ASYMMETRIC_RADIATION_NOCTURNAL_ACK]`` into every emitted file
    whose selected suite met a night window -- a declaration the user
    never made, in a file that outlives the terminal session, which
    disarmed the load-time guard at ``gpuwm check``, ``gpuwm run``,
    ``gpuwm go``, run-plan and both prepared runners forever after.  An
    acknowledgement is a person stating that they know what they are
    running; a program cannot make it on their behalf and have it mean
    anything.  ``gpuwm domain --ack <id>`` is how it is made now, and
    :func:`domain_main` REFUSES rather than emit a file that needs one it
    was not given.

    ``interactive`` records WHICH front door authored the file -- the
    prompt session or the flags -- in the header the file already
    carries.  Both doors produce the same config for the same answers,
    so this is provenance rather than a difference: it tells whoever
    reads the file later how the numbers in it were arrived at, which
    is the question asked of an emitted file nobody remembers writing.
    """
    experiment = {
        "name": name, "start_time": start_time,
        "run_seconds": float(hours * 3600), "feedback": 0,
        "smooth_option": 0, "blend_width": 5, "spec_bdy_width": 5,
        # Single-domain emissions disable restart writing: the portable
        # prepared-forecast contract requires restart_interval_s = 0.
        "restart_interval_s": 0.0 if not ratios else 3600.0,
    }
    shared = shared_physics(profile)
    # The vertical default is bounded by the source's certified column:
    # the ladder is eta-normalized, so only p_top moves (see
    # DEFAULT_MODEL_TOP_PA / emitted_model_top_pa).
    shared["p_top"] = emitted_model_top_pa(
        (fetch_hints or {}).get("source"))
    shared["map_proj"] = WRF_MAP_PROJ_CODES[projection["map_proj"]]
    # Nocturnal validity of the emitted radiation pairing, stated in the
    # header of EVERY emitted file and, where the pairing is asymmetric
    # (shortwave on, longwave off) across a window that includes local
    # night, DECLARED in [experiment].acknowledgements -- the same
    # declaration the load-time guard in gpuwm.experiment demands, so an
    # explicitly selected validation profile emits a file that still
    # loads.  Asymmetric suites are never a default on the gfs/era5
    # doors; explicit selection is the declaration, and it is made in
    # ink here rather than by silence.
    emitted_lw = int(shared.get("ra_lw_physics", shared.get("ra_physics", 0)))
    emitted_sw = int(shared.get("ra_sw_physics", shared.get("ra_physics", 0)))
    asymmetric_radiation = emitted_sw > 0 and emitted_lw == 0
    # One predicate, read by the emission rule here and by the spoken
    # advisory in domain_main: see declared_nocturnal_night.  It asks
    # exactly what the inline expression here used to ask -- shortwave on,
    # longwave off, and a local night inside the window.
    first_night = declared_nocturnal_night(
        profile, start_time=start_time, hours=hours, projection=projection)
    # The SECOND declaration, and a separate question from the first:
    # with no longwave scheme under a land-surface scheme, downward
    # longwave is a declared constant rather than a computed flux, at
    # noon as much as at midnight.  The nocturnal token used to stand in
    # for this by accident -- it is checked before any physics is
    # inspected, so a config carrying it never had its GLW source looked
    # at -- and an all-daylight asymmetric emission declared nothing at
    # all while still running Noah on a fixed 300 W m-2.  Both are now
    # stated, separately, in ink.  The condition is the load guard's own
    # classification, not a re-derivation, so an emission can never pass
    # here and refuse there.
    from gpuwm.physics_compat import downward_longwave_disposition
    _glw_kind, _ = downward_longwave_disposition(
        ra_lw_physics=emitted_lw, ra_sw_physics=emitted_sw,
        sf_surface_physics=int(shared.get("sf_surface_physics", 0)))
    constant_longwave = _glw_kind in ("consumed", "published")
    # WHO DECLARES WHICH TOKEN, and why the two are not treated alike.
    #
    # The NOCTURNAL token is a claim about the WINDOW the user chose --
    # "I know this run contains night and I want it anyway" -- and the
    # wizard cannot make it for them.  It wrote that line by itself
    # through 1.8.7, into a file that outlives the terminal session, and
    # the line disarmed the load guard at every other front door
    # forever after.  It now comes from ``gpuwm domain --ack`` or not at
    # all, and domain_main refuses rather than emit a file needing one
    # it was not given.
    #
    # The CONSTANT-LONGWAVE token is not a judgment: it is a mechanical
    # consequence of the SUITE the user named on the command line, true
    # of every window and every place that suite is run in.  Naming the
    # profile IS the declaration, so it is written here in ink -- with
    # the JUSTIFY line the shipped-config convention requires -- rather
    # than left to silence.  Emitting it is what keeps the wizard's own
    # daylight output loadable, which is the property that made this
    # worth separating: an all-daylight asymmetric run has no nocturnal
    # claim to make and still integrates a fabricated flux.
    acknowledgements = tuple(acknowledgements)
    emitted_acks = list(acknowledgements)
    if constant_longwave and CONSTANT_DOWNWARD_LONGWAVE_ACK not in emitted_acks:
        emitted_acks.append(CONSTANT_DOWNWARD_LONGWAVE_ACK)
    if emitted_acks:
        experiment["acknowledgements"] = emitted_acks
    declared = ASYMMETRIC_RADIATION_NOCTURNAL_ACK in acknowledgements
    if not asymmetric_radiation:
        nocturnal_note = (
            "# NOCTURNALLY VALID: longwave and shortwave both run, so the "
            "surface longwave\n"
            "# budget stays closed through the night.\n")
    elif first_night is None:
        nocturnal_note = (
            "# NOT NOCTURNALLY VALID (shortwave on, longwave OFF) -- "
            "acceptable for this\n"
            "# all-daylight window; re-emit with a full lw+sw profile "
            "before running any\n"
            "# window that includes local night.\n")
    elif declared:
        nocturnal_note = (
            "# NOT NOCTURNALLY VALID: shortwave heats by day, longwave is "
            "OFF, and this\n"
            f"# window includes local night (first at "
            f"{first_night:%Y-%m-%dT%H:%M}Z), so the surface\n"
            "# radiates with no downward longwave after sunset and skin "
            "temperature and\n"
            "# 2 m moisture collapse.  Emitted because YOU declared it: "
            "the acknowledgement\n"
            "# in [experiment] below came from `gpuwm domain --ack "
            f"{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}`,\n"
            "# and it disarms the load guard at every other front door "
            "for this file --\n"
            "# check, run, go, run-plan and both prepared runners.\n")
    else:
        # `gpuwm domain` refuses this combination before it renders (see
        # domain_main), so this is the LIBRARY caller's copy: a file that
        # will not load must not also look fine.
        nocturnal_note = (
            "# NOT NOCTURNALLY VALID, AND THIS FILE WILL NOT LOAD: "
            "shortwave heats by day,\n"
            "# longwave is OFF, and this window includes local night "
            "(first at\n"
            f"# {first_night:%Y-%m-%dT%H:%M}Z).  Every front door refuses "
            "it at config load.\n"
            "# Re-emit with a full lw+sw profile, or -- if you mean the "
            "daytime validation\n"
            "# suite and accept the night -- re-emit with `gpuwm domain "
            "--ack\n"
            f"# {ASYMMETRIC_RADIATION_NOCTURNAL_ACK}`.\n")
    if constant_longwave:
        # The justification convention shipped configs are held to
        # (tests/test_shipped_acknowledgement_justifications.py), written
        # into every emitted file that needs the token so an emitted
        # config meets the same bar as a committed one.  The middle
        # sentence states the actual exposure -- integrated by the land
        # surface, or merely published to wrfout -- from the same
        # disposition the guard read.
        if _glw_kind == "consumed":
            exposure = (
                "# the surface integrates a declared constant 300 W m-2 "
                "for the whole\n"
                "# forecast.")
        else:
            exposure = (
                "# the declared constant 300 W m-2 is published as the "
                "GLW row of every\n"
                "# wrfout frame.")
        nocturnal_note += (
            f"# JUSTIFY {CONSTANT_DOWNWARD_LONGWAVE_ACK}: this suite "
            "sets\n"
            "# ra_lw_physics = 0, so NOTHING computes downward longwave "
            "and\n"
            + exposure +
            "  Emitted only because this profile was selected "
            "explicitly.\n"
            "# Re-emit with a full lw+sw profile for any run whose "
            "surface fields you\n"
            "# intend to believe.\n")
    time_step = root_time_step_s(projection["ref_lat"], root_dx_m)
    chain_km = _ladder_dx_km(ratios, root_dx_m)
    cu_by_domain = cumulus_by_domain(
        dims, ratios, profile=profile, root_dx_m=root_dx_m,
        cumulus_requested=cumulus_requested)
    root_cu = cu_by_domain[0]
    retired = cumulus_retired_note(
        profile, root_dx_m / 1000.0, cumulus_requested=cumulus_requested)
    # THE VERBATIM CLAIM IS TRUE OR IT IS NOT MADE.  Where the emission
    # retired the suite's cumulus it names the switch it moved, instead
    # of asserting a switch-for-switch identity this file does not have.
    verbatim_claim = (
        "Taken verbatim from gpuwm.physics_compat EXCEPT the root's "
        "cumulus\n# switch (see CUMULUS OFF below), so this file passes "
        "the prepared-"
        if retired else
        "Taken verbatim from gpuwm.physics_compat, so this file passes "
        "the prepared-")
    if level_buffers_km is None:
        # This is the original point header byte-for-byte.  Polygon support
        # must not perturb existing point-authored artifacts.
        header = (
            "# Emitted by `gpuwm domain`"
            + (" (interactive session)" if interactive else "")
            + " -- point "
            f"{projection['ref_lat']:g},{projection['ref_lon']:g}, ladder "
            f"{'-'.join(f'{v:g}' for v in chain_km)} km.\n"
            f"# PHYSICS: {physics_summary(profile, cu_physics=root_cu)}.\n"
            f"# {verbatim_claim}\n"
            "# forecast runner's profile guard as emitted.  Child dx/dt "
            "derive exactly from\n"
            "# the parent chain and are never hand-typed "
            "(gpuwm/experiment.py).\n")
    else:
        buffers = ", ".join(f"{value:g}" for value in level_buffers_km)
        header = (
            "# Emitted by `gpuwm domain` -- polygon center "
            f"{projection['ref_lat']:g},{projection['ref_lon']:g}, ladder "
            f"{'-'.join(f'{v:g}' for v in chain_km)} km.\n"
            f"# Polygon buffers by domain level (outer to inner): "
            f"{buffers} km.\n"
            f"# PHYSICS: {physics_summary(profile, cu_physics=root_cu)}.\n"
            f"# {verbatim_claim}\n"
            "# forecast runner's profile guard as emitted.  Child dx/dt "
            "derive exactly from\n"
            "# the parent chain and are never hand-typed "
            "(gpuwm/experiment.py).\n")
    header += nocturnal_note
    per_km = seconds_per_km(projection["ref_lat"])
    if per_km != 5:
        header += (
            f"# TROPICAL CLOCK: |lat| < {MERCATOR_MAX_LAT:g}, so "
            f"time_step is {float(time_step):g} s "
            f"({float(per_km):g} s per km), half the "
            f"{float(time_step) * 2:g} s\n"
            "# the 5 s/km convention would give at this dx.  The "
            "stability gate uses the\n"
            "# maximum co-located vertical |w|/layer-thickness; "
            "tropical convection\n"
            "# destabilised a measured 12 km Mercator domain at 60 s and "
            "was comfortably\n"
            "# stable at a shorter step, for +22% wall time (radiation "
            "and cumulus are\n"
            "# called on wall-clock intervals, so extra dynamics steps "
            "are cheap).\n")
    for line in retired:
        header += f"# {line}\n"
    for line in gray_zone_advisory(chain_km, shared):
        header += f"# {line}\n"
    for line in cumulus_gray_zone_advisory(chain_km, cu_by_domain):
        header += f"# {line}\n"
    parts = [
        header,
        _render_table("experiment", experiment),
        _render_table("projection", projection),
        _render_table("shared", shared),
    ]
    for table in _domain_tables(
            dims, ratios, time_step=time_step, root_dx_m=root_dx_m,
            profile=profile, cumulus_requested=cumulus_requested,
            history_interval_s=history_interval_s,
            nest_history_interval_s=nest_history_interval_s):
        parts.append(_render_table("domain", table, array_of_tables=True))
    if fetch_hints:
        parts.append(_render_table(
            "fetch", fetch_hints,
            comment="Advisory data-acquisition hints (validated, not "
                    "executed); keys mirror `gpuwm fetch` flags."))
    else:
        # NO [fetch] TABLE, DELIBERATELY.  The table is validated at every
        # config load against the sources `gpuwm fetch` can actually
        # download, so writing one for a source with no fetch route would
        # emit a file that refuses to load -- and writing one that loads
        # would advertise a download this ArWen cannot make.  The
        # acquisition route is stated as a comment instead, which is the
        # honest shape until the fetch door grows the route.
        parts.append(
            "# NO [fetch] TABLE: `gpuwm fetch` has no download route for\n"
            "# this source yet, so its bytes are staged by hand (see\n"
            "# docs/public/SOURCES.md, 'Sources with no fetch door').  The\n"
            "# geometry, levels, physics and boundary cadence in this file\n"
            "# are complete -- only the acquisition step is manual.\n")
    if case_data is not None:
        parts.append(_render_table(
            "case_data", case_data,
            comment="Declared inputs for the config-driven "
                    "check/static/ingest/run front door (ERA5 native-GRIB1 "
                    "route; era5_z_invariant source orography)."))
    return "\n".join(parts)


def experiment_from_text(text: str, *, source: str) -> ExperimentConfig:
    """Round-trip emitted TEXT through the real loaders (advisory [fetch]
    and [case_data] are split off exactly as the CLI loaders do)."""
    raw = tomllib.loads(text)
    fetch_table = raw.pop("fetch", None)
    if fetch_table is not None:
        from gpuwm.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source=source)
    raw.pop("case_data", None)
    return build_experiment(raw, source=source)


def sizing_budget_bytes(exp: ExperimentConfig, *, free_bytes: int,
                        vram_gib: float | None,
                        forcing_interval_seconds: float,
                        profile=None) -> int:
    """The budget the machine-peak ENVELOPE is compared against.

    ``free_bytes`` minus what this process's own envelope does not model,
    and nothing else.  :func:`~gpuwm.core.preflight.
    machine_peak_envelope_bytes` is a model of the WHOLE device residency
    a run of this configuration reaches -- the itemized pool, the CUDA
    context, the local-memory backing store of its kernel set, the
    measured pool slack and the measured residue -- so the only thing
    left outside it is OTHER processes, which is exactly
    :data:`~gpuwm.core.preflight.EXTERNAL_MARGIN_BYTES`.

    IT USED TO SUBTRACT THE ALLOCATION RESERVE, and the allocation
    reserve carries the non-pool residency too.  The envelope was then
    compared against a budget from which its own CUDA context and its own
    backing store had ALREADY been removed: one process charged twice for
    the same bytes.  On a 10 GiB RTX 3080 that was 2.91 GiB of a 10 GiB
    card spent twice, and the hrrr ladder's SMALLEST layout -- 60x48 --
    was refused on a card that fits it (task 206; the acceptance walk is
    in the lane report).  The two doors are still coherent, and more
    strictly than before: this comparison is provably tighter than
    ``gpuwm check``'s allocation gate at every grid size, so a wizard
    PASS remains a check PASS (``test_vram_measured_reserve.py::
    test_the_fit_gate_is_never_looser_than_the_allocation_gate``).

    ``profile`` is the device the non-pool terms were priced against and
    is accepted so callers can pass the card they MEASURED; it does not
    enter this arithmetic, and is kept in the signature because every
    caller of this function has to have decided it.
    """

    del exp, vram_gib, forcing_interval_seconds, profile  # see above
    return int(free_bytes) - EXTERNAL_MARGIN_BYTES


def _lighter_profiles_than(profile: str | None, source: str,
                           price_bytes) -> list[str]:
    """Shipped suites this source can run that PRICE less than ``profile``.

    Ranked by the estimator's own peak envelope at the refused layout --
    ``price_bytes(profile_name) -> int | None`` -- never by a species
    heuristic.  The heuristic ranked by microphysics species and
    radiation presence, and on the 3080 walk it advised three
    legacy-RRTMG suites as "lighter" than the rte-rrtmgp default; the
    calibration runs then MEASURED the advised suite at 5.53 GiB against
    the default's 2.60 at the same 110x88 grid, because the legacy
    call-peak workspace dominates the envelope and the heuristic never
    priced it.  A candidate the pricer cannot price is dropped, not
    guessed at.

    Returned in the order a reader should try them -- cheapest last, so
    the first name is the smallest step down rather than the biggest
    sacrifice.

    Every candidate passes :func:`profile_route_blocker` -- the SAME
    pairing predicate the emission refuses by -- before it is priced.
    Admissibility used to be checked for one hard-coded source only,
    so the 2.5.0 walk's gfs refusal ranked a RUC-LSM suite FIRST while
    the very same wizard refuses that pairing outright: following the
    printed advice was refused by the door that printed it.

    Named in a refusal, never applied: a suite is the operator's choice
    and a wizard that silently downgraded physics to make a number fit
    would be lying about what it emitted.
    """

    if profile is None:
        return []
    cost = price_bytes(profile)
    if cost is None:
        return []
    lighter = []
    for candidate in WIZARD_PHYSICS_PROFILES:
        if candidate == profile:
            continue
        try:
            single_domain_runtime_switches(candidate)
        except ValueError:
            continue
        if profile_route_blocker(candidate, source) is not None:
            continue
        candidate_cost = price_bytes(candidate)
        if candidate_cost is None or candidate_cost >= cost:
            continue
        lighter.append((candidate_cost, candidate))
    return [name for _cost, name in sorted(lighter, reverse=True)][:3]


def fit_ladder(*, ladder: str | None = None, free_bytes: int, hours: int,
               start_time: datetime, projection: dict, source: str,
               name: str, ratios: tuple[int, ...] | None = None,
               root_dx_m: float = ROOT_DX_M,
               profile: str | None = DEFAULT_PHYSICS_PROFILE,
               cumulus_requested: bool = False,
               vram_gib: float | None = None,
               device_profile=None,
               acknowledgements: tuple[str, ...] = (),
               ) -> tuple[list[tuple[int, int]], ExperimentConfig]:
    """Largest centered layout whose peak envelope fits the budget, with
    headroom left over.

    ``device_profile`` is the CARD the non-pool terms are priced against
    -- the one this machine MEASURED when no ``--card``/``--vram-gib``
    was declared, and ``None`` (the conservative reference) when the
    caller is sizing for a machine that is somewhere else.  It used to be
    ``None`` unconditionally, so a wizard that had just measured a 68-SM
    RTX 3080 priced that card's local-memory backing store on the 170-SM
    reference profile: 1.49 GiB of another card's shader count, on the
    one term shrinking the grid cannot move, while ``gpuwm check`` on the
    same box used the live profile and disagreed (task 206, open task
    #162's mechanism).

    Bisects a continuous scale factor; every candidate is validated by the
    real experiment loader (clearance, cadence, ratio rules) and priced by
    the real estimator -- the wizard owns no memory arithmetic of its own.
    A custom ``--root-dx``/``--chain`` goes through this same loop, so a
    hand-specified ladder is validated and sized exactly like a preset.

    Takes FREE VRAM, not a budget: the reserve is a property of the
    candidate experiment (see :func:`sizing_budget_bytes`), so it cannot
    be computed before the candidate exists.  And it stops short of the
    budget by :func:`fit_headroom_bytes` -- a config that exactly touches
    its budget is a config with nothing left for the machine to be
    slightly less generous than the model.
    """
    if (ladder is None) == (ratios is None):
        raise ValueError("fit_ladder takes exactly one of ladder / ratios")
    if ratios is None:
        ratios = LADDER_RATIOS[ladder]
    label = ladder if ladder is not None else "-".join(
        f"{v:g}" for v in _ladder_dx_km(ratios, root_dx_m))
    interval = source_forcing_interval_seconds(source)

    def candidate(scale: float):
        dims = _dims_for_scale(scale, ratios)
        text = render_config(
            name=name, start_time=start_time, hours=hours,
            projection=projection, dims=dims, ratios=ratios,
            fetch_hints=_candidate_fetch_hints(source), case_data=None,
            root_dx_m=root_dx_m, profile=profile,
            # The candidate is the same file the user will get, so it
            # carries the same declaration -- otherwise the sizing loop
            # would refuse a layout the emission is allowed to write.
            # The cumulus decision rides along for the same reason: a
            # retired scheme is a kernel set the envelope no longer
            # prices, and sizing against one the file will not carry
            # would fit a smaller domain than the card can hold.
            cumulus_requested=cumulus_requested,
            acknowledgements=acknowledgements)
        exp = experiment_from_text(text, source=f"<candidate {label}>")
        # Every PHASE, not just the forecast.  Sizing a domain against the
        # forecast alone is what let this wizard hand a user a config that
        # fit their card, take a multi-gigabyte download, and then OOM in
        # preprocessing -- the phase it had never priced.
        phases = estimate_phases(
            exp, source=source, forcing_interval_seconds=interval,
            vram_gib=vram_gib, profile=device_profile)
        budget = sizing_budget_bytes(
            exp, free_bytes=free_bytes, vram_gib=vram_gib,
            forcing_interval_seconds=interval, profile=device_profile)
        return dims, exp, phases.peak_envelope_bytes, budget

    def uncovered(exp, dims) -> str | None:
        """Why the SOURCE cannot force this layout, if it cannot.

        The budget is not the only constraint on how large a domain may
        be.  A regional source's native grid is finite, so a card-filling
        ladder near its edge is a legal, well-sized experiment that no
        fetch of that source can force.  Sizing against VRAM alone
        produced exactly that on a 24 GiB card: a 3 km root whose halo
        ran nine rows off the top of the HRRR grid, discovered by the
        root preparation after the download.

        THREE checks, and the order matters.  HRRR's certified route
        knows more than the grid rectangle -- the interpolation stencil
        plus the surface-fallback halo need real source cells outside
        the target on every side, and the donor-search margin rides
        along -- so its own refusal runs first and is the stricter one.
        Every other regional source is bounded by its declared window
        (:func:`source_coverage_refusal`), which is what turned ICON-EU
        over a central-US domain from a preparation traceback into a
        sizing bound this loop can shrink against.  Last, and for every
        source including the global ones, the fitted root has to be
        servable as ONE crop (:func:`fetch_crop_refusal`) -- the bound
        that stops a card-filling Linux layout from being sized and then
        refused by its own emission.
        """
        if source == "hrrr":
            try:
                refusal = coverage_refusal(exp)
            except (HrrrRouteInputError, ValueError) as error:
                # Not a coverage answer: the spec itself cannot be built.
                # Shrinking the ladder cannot fix that, so it refuses with
                # its OWN cause and remedy instead of masquerading as an
                # off-grid polygon.
                raise DomainFitError(str(error)) from None
            if refusal is not None:
                return (f"{refusal}.  Move --point away from the edge of "
                        f"the {source.upper()} grid, or choose a source "
                        "whose coverage includes it")
        else:
            refusal = source_coverage_refusal(
                projection, dims[0][0], dims[0][1], source=source,
                root_dx_m=root_dx_m)
            if refusal is not None:
                return refusal
        # Last, and only when the source reaches the domain at all: can
        # ONE crop of it be fetched?  Asked last because it is the only
        # one of the three that costs a fresh grid evaluation on the
        # global sources, which have no window to answer from.
        #
        # A pole-containing footprint is handed back to the genuine-limit
        # refusal that runs after the fit (:func:`_pole_clearance_refusal`,
        # whose call site says so): such a footprint spans every longitude,
        # so this bound is true of it and useless -- shrinking a domain
        # does not move the pole out of it, and the crop bound's remedy
        # would send the reader after the wrong flag.
        if _footprint_contains_pole(
                projection, dims[0][0], dims[0][1], root_dx_m):
            return None
        return fetch_crop_refusal(
            projection, dims[0][0], dims[0][1], source=source,
            root_dx_m=root_dx_m)

    # The MINIMUM layout is a property of the ladder, not a constant: a
    # chain deeper than any preset needs a larger root before its
    # innermost nest has any interior at all (:func:`_min_hosting_scale`).
    min_scale = _min_hosting_scale(ratios)
    dims, exp, envelope, budget = candidate(min_scale)
    #: The part of the envelope no grid can move: this suite's CUDA
    #: context, the local-memory backing store of its kernel set, and the
    #: measured residue.  When THAT alone is the whole card there is
    #: nothing to size -- every layout on every ladder is refused for the
    #: same reason, and the fit loop's per-layout refusal would name a
    #: grid the reader cannot usefully shrink.
    #:
    #: This used to be spelled ``budget <= 0``, which worked only while
    #: the budget subtracted the whole allocation reserve.  It no longer
    #: does (that reserve carries the same non-pool bytes the envelope
    #: carries, and charging both is the double count task 206 removed),
    #: so the floor is asked directly instead of inferred from a
    #: subtraction that has stopped containing it.
    floor_estimate = estimate_experiment(
        exp, forcing_interval_seconds=interval, vram_gib=vram_gib,
        profile=device_profile)
    grid_independent = (floor_estimate.envelope_intercept_bytes
                        + ENVELOPE_UNMODELLED_BYTES)
    if budget <= 0 or grid_independent >= budget:
        raise DomainFitError(
            f"this card has no budget for ladder {label} at all: the "
            f"suite's grid-independent envelope (CUDA context + the "
            f"local-memory backing store of its kernel set + the "
            f"measured unmodelled residue) is "
            f"{grid_independent / GIB:.2f} GiB, and with the "
            f"{EXTERNAL_MARGIN_BYTES / GIB:.2f} GiB external margin that "
            f"is already the whole of about "
            f"{free_bytes / GIB:.2f} GiB free -- before the grid asks for "
            f"a single byte, so no smaller layout on any ladder can "
            f"help.  Choose a physics profile with a smaller kernel set, "
            f"or a larger card")
    smallest_uncovered = uncovered(exp, dims)
    if smallest_uncovered is not None:
        raise DomainFitError(
            f"ladder {label} cannot be forced by {source} even at the "
            f"minimum layout ({dims[0][0]}x{dims[0][1]} root): "
            f"{smallest_uncovered}")
    if envelope > budget:
        # Say WHY it does not fit.  "your card is too small" is what the
        # bare number reads as, and at the minimum layout the honest
        # answer is usually that the grid-independent terms dominate --
        # but ONLY when they actually do.  The old wording asserted
        # "so a smaller grid cannot help" beside a printed 0%, which is
        # a sentence contradicting the number in front of it.
        floor = estimate_experiment(
            exp, forcing_interval_seconds=interval, vram_gib=vram_gib,
            profile=device_profile)
        constants = (floor.envelope_intercept_bytes
                     + ENVELOPE_UNMODELLED_BYTES)
        share = (100.0 * constants / envelope if envelope else 0.0)
        if share >= 25.0:
            why = ("is grid-independent (CUDA context, the local-memory "
                   "backing store of the selected kernel set, and the "
                   "measured unmodelled residue), so shrinking the grid "
                   "moves only the rest")
        else:
            why = ("is grid-independent; the grid itself is most of this "
                   "layout, and this IS the smallest layout the ladder "
                   "has")
        detail = (
            f"the model itself wants {floor.alloc_estimate_bytes / GIB:.2f} "
            f"GiB at this layout; the other "
            f"{constants / GIB:.2f} GiB ({share:.0f}% of the envelope) "
            f"{why}.  This is already the minimum layout, so there is no "
            "smaller grid on this ladder to fall back to")
        phases = estimate_phases(
            exp, source=source, forcing_interval_seconds=interval,
            vram_gib=vram_gib, profile=device_profile)
        # Name the THIRD lever too.  A large share of the envelope is the
        # selected kernel set's own local-memory backing store, so the
        # suite is often the cheapest thing to change -- and after 1.8
        # gave every source a full-radiation default, it is the lever a
        # small card most often needs.  Omitting it read as "your card is
        # too small" when a lighter shipped profile fits the same grid.
        # Candidates are PRICED at this exact layout by the same
        # estimator that just refused, so a suite whose envelope is
        # larger -- legacy-RRTMG's call-peak workspace, measured 2.1x
        # the rte-rrtmgp default on the 3080 -- can never be advised.

        def _price(candidate_profile: str) -> int | None:
            try:
                candidate_text = render_config(
                    name=name, start_time=start_time, hours=hours,
                    projection=projection, dims=dims, ratios=ratios,
                    fetch_hints=_candidate_fetch_hints(source),
                    case_data=None, root_dx_m=root_dx_m,
                    profile=candidate_profile,
                    # Pricing a suite the user would have to NAME to get,
                    # so it is priced as a named suite: verbatim.
                    cumulus_requested=True,
                    acknowledgements=acknowledgements)
                candidate_exp = experiment_from_text(
                    candidate_text, source=f"<candidate {label} "
                                           f"{candidate_profile}>")
                return estimate_phases(
                    candidate_exp, source=source,
                    forcing_interval_seconds=interval,
                    vram_gib=vram_gib,
                    profile=device_profile).peak_envelope_bytes
            except Exception:
                return None

        lighter = _lighter_profiles_than(profile, source, _price)
        remedy = ("choose a shallower ladder, a lighter --physics-profile "
                  f"({', '.join(lighter)}), or a larger card"
                  if lighter else
                  "choose a shallower ladder or a larger card")
        raise DomainFitError(
            f"ladder {label} does not fit a {budget / GIB:.1f} GiB "
            f"budget even at the minimum layout ({dims[0][0]}x{dims[0][1]} "
            f"root): {phases.verdict(budget)}.  {detail}; {remedy}")
    lo, hi = min_scale, _MAX_SCALE
    best = (dims, exp)
    # WHY the search stopped where it did, kept as it happens.  A memory
    # bound needs no explanation -- the sizing line prints the envelope
    # against the budget -- but a NON-memory bound is invisible from the
    # outside, and an invisible saturation is the exact defect
    # tests/test_domain_wizard_budget_monotonic.py was written for: a 180
    # GiB card sized like a 64 GiB one and reported a comfortable fit.
    # ``None`` means nothing SOURCE-shaped bound: the card did, or the
    # experiment loader refused the layout on its own terms.
    binding_reason: str | None = None
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        try:
            dims, exp, envelope, budget = candidate(mid)
        except DomainFitError:
            # A layout the experiment loader itself refuses -- neither
            # the card nor the source, so the sentence below would name
            # the wrong thing.  Claim nothing.
            hi = mid
            binding_reason = None
            continue
        target = budget - fit_headroom_bytes(budget)
        if envelope > target:
            hi = mid
            binding_reason = None
            continue
        reason = uncovered(exp, dims)
        if reason is None:
            best = (dims, exp)
            lo = mid
        else:
            hi = mid
            binding_reason = reason
    # A search that converges on its own upper bracket did not find the grid
    # the budget affords -- it found the largest grid it was willing to look
    # at.  Those are different answers and they used to be indistinguishable
    # from the outside, which is how an 8.0 ceiling sized 64, 96 and 180 GiB
    # cards identically while every one of them reported a comfortable fit.
    # Saying it costs nothing when the bound does not bind.
    if lo >= _MAX_SCALE * (1.0 - 1e-6):
        root = best[0][0]
        warn(f"domain search reached its scale ceiling at {root[0]}x{root[1]}; "
             "this is the largest layout considered, not necessarily the "
             "largest your budget affords",
             why="The wizard brackets its grid-scale bisection between "
                 f"_MIN_SCALE and _MAX_SCALE (currently {_MAX_SCALE}).  A "
                 "result sitting on the upper bracket means memory never "
                 "became the binding constraint, so a larger card will not "
                 "buy a larger domain until the ceiling is raised.")
    elif binding_reason is not None:
        # The same defect one bound over.  A source-shaped ceiling
        # (coverage window, or a forcing box too wide to be fetched as
        # one crop) stops the search below what the card affords, and
        # from the outside that is indistinguishable from a comfortable
        # fit -- the sizing line prints an envelope well under budget and
        # says nothing about why the grid is not larger.  So the wizard
        # says it, and says that a bigger card is not the lever.
        root = best[0][0]
        warn(f"domain search stopped at {root[0]}x{root[1]} on the SOURCE, "
             "not the card: a larger card buys no more grid here",
             why="The fit is bounded by every constraint, not only memory."
                 f"  The next larger layout was rejected because "
                 f"{binding_reason}")
    return best


def _maximum_map_factor(grid, south: float, north: float) -> float:
    """Maximum conformal scale over a latitude interval."""

    latitudes = np.linspace(south, north, 257, dtype=np.float64)
    map_factors = np.abs(np.asarray(grid.map_factor(latitudes), dtype=float))
    if not np.all(np.isfinite(map_factors)) or not map_factors.size \
            or float(map_factors.max()) <= 0.0:
        raise ValueError(
            "the polygon cannot be represented finitely in the "
            f"selected {grid.map_proj} projection")
    return float(map_factors.max())


def _polygon_sample_step_deg(projection: dict,
                             footprint: PolygonFootprint,
                             finest_dx_m: float) -> float:
    """Angular segment step whose projection error fits inside cell slack."""

    grid = _root_grid(projection, 2, 2, finest_dx_m)
    scale = _maximum_map_factor(grid, footprint.south, footprint.north)
    # A lon/lat step has spherical path length no greater than
    # sqrt(2)*R*step radians.  Bound its projected length to one quarter of
    # the finest cell.  The fitter adds a whole cell outside every sample,
    # so every point between samples remains inside that proven envelope.
    step = math.degrees(
        finest_dx_m / (4.0 * math.sqrt(2.0) * EARTH_RADIUS_M * scale))
    return min(_POLYGON_SAMPLE_STEP_DEG, step)


def _buffer_cells(grid, footprint: PolygonFootprint,
                  buffer_km: float) -> float:
    """Conservative projected-cell distance for a ground buffer."""

    if buffer_km == 0.0:
        return 0.0
    buffer_m = float(buffer_km) * 1000.0
    latitude_reach = math.degrees(buffer_m / EARTH_RADIUS_M)
    south = footprint.south - latitude_reach
    north = footprint.north + latitude_reach
    if south <= -90.0 or north >= 90.0:
        edge = "south" if south <= -90.0 else "north"
        raise ValueError(
            f"--buffer-km {buffer_km:g} on this footprint reaches the "
            f"{edge} pole; lat-lon source interpolation and static-tile "
            "windowing are not pole-capable")
    # These projections are conformal and their scale depends on latitude.
    # Taking the maximum over the whole buffered latitude range makes a
    # ground-distance buffer no smaller in projected grid cells.
    scale = _maximum_map_factor(grid, south, north)
    return buffer_m * scale / float(grid.dx)


def _round_up_multiple(value: float, multiple: int) -> int:
    return int(multiple * max(1, math.ceil(value / multiple - 1e-12)))


def polygon_ladder_dims(*, footprint: PolygonFootprint,
                        projection: dict, ratios: tuple[int, ...],
                        buffers_km: tuple[float, ...],
                        root_dx_m: float = ROOT_DX_M
                        ) -> list[tuple[int, int]]:
    """Smallest centered legal ladder containing the buffered footprint."""

    count = len(ratios) + 1
    if len(buffers_km) != count:
        raise ValueError(
            f"polygon_ladder_dims needs {count} buffers, got "
            f"{len(buffers_km)}")
    baseline = _dims_for_scale(_min_hosting_scale(ratios), ratios)
    finest_dx_m = float(root_dx_m) / math.prod(ratios)
    sample_step = _polygon_sample_step_deg(
        projection, footprint, finest_dx_m)
    sample_lats, sample_lons = _polygon_samples(
        footprint, max_step_deg=sample_step)
    dimensions: list[tuple[int, int]] = []
    dx = float(root_dx_m)
    for level in range(count):
        if level:
            dx /= ratios[level - 1]
        grid = _root_grid(projection, 2, 2, dx)
        i, j = grid.latlon_to_ij(sample_lats, sample_lons)
        i = np.asarray(i, dtype=float)
        j = np.asarray(j, dtype=float)
        if not np.all(np.isfinite(i)) or not np.all(np.isfinite(j)):
            raise ValueError(
                "the polygon cannot be represented finitely in the "
                f"selected {projection['map_proj']} projection")
        margin = _buffer_cells(grid, footprint, buffers_km[level])
        half_x = (float(np.max(np.abs(i - grid.known_x))) + margin
                  + _POLYGON_FIT_SLACK_CELLS)
        half_y = (float(np.max(np.abs(j - grid.known_y))) + margin
                  + _POLYGON_FIT_SLACK_CELLS)
        quantum = 2 if level == 0 else 2 * ratios[level - 1]
        nx = max(baseline[level][0],
                 _round_up_multiple(2.0 * half_x, quantum))
        ny = max(baseline[level][1],
                 _round_up_multiple(2.0 * half_y, quantum))
        dimensions.append((nx, ny))

    # Every child must also clear its parent's external-boundary and blend
    # rows.  Propagate that requirement from the innermost level outward;
    # this can enlarge an outer level beyond its own geometric buffer, but
    # never makes any requested buffer smaller.
    for level in range(count - 1, 0, -1):
        ratio = ratios[level - 1]
        child_nx, child_ny = dimensions[level]
        parent_nx, parent_ny = dimensions[level - 1]
        parent_quantum = 2 if level == 1 else 2 * ratios[level - 2]
        parent_nx = max(parent_nx, _round_up_multiple(
            child_nx // ratio + 2 * _CLEARANCE_ROWS, parent_quantum))
        parent_ny = max(parent_ny, _round_up_multiple(
            child_ny // ratio + 2 * _CLEARANCE_ROWS, parent_quantum))
        dimensions[level - 1] = parent_nx, parent_ny
    return dimensions


def verify_polygon_containment(exp: ExperimentConfig,
                               footprint: PolygonFootprint,
                               buffers_km: tuple[float, ...]) -> None:
    """Prove each emitted projected grid contains its requested envelope."""

    from gpuwm.static.projection import grids_from_projection_config

    grids = grids_from_projection_config(exp)
    if len(grids) != len(buffers_km):
        raise DomainFitError(
            "internal polygon fit regression: emitted domain count does not "
            "match the per-level buffer count")
    sample_step = _polygon_sample_step_deg(
        {
            "map_proj": grids[0].map_proj,
            "ref_lat": grids[0].ref_lat,
            "ref_lon": grids[0].ref_lon,
            "truelat1": grids[0].truelat1,
            "truelat2": grids[0].truelat2,
            "stand_lon": grids[0].stand_lon,
        }, footprint, min(float(grid.dx) for grid in grids))
    sample_lats, sample_lons = _polygon_samples(
        footprint, max_step_deg=sample_step)
    tolerance = 1e-8
    for level, (grid, buffer_km) in enumerate(zip(grids, buffers_km), 1):
        i, j = grid.latlon_to_ij(sample_lats, sample_lons)
        i = np.asarray(i, dtype=float)
        j = np.asarray(j, dtype=float)
        margin = _buffer_cells(grid, footprint, buffer_km)
        clearances = (
            float(i.min()) - 0.5,
            (float(grid.e_we) - 0.5) - float(i.max()),
            float(j.min()) - 0.5,
            (float(grid.e_sn) - 0.5) - float(j.max()),
        )
        if not np.all(np.isfinite(clearances)) \
                or min(clearances) + tolerance < margin:
            raise DomainFitError(
                f"internal polygon fit regression: domain d{level:02d} "
                f"does not contain the footprint plus its {buffer_km:g} km "
                "buffer; refusing to emit a partial target domain")


def fit_polygon_ladder(*, footprint: PolygonFootprint,
                       buffers_km: tuple[float, ...],
                       free_bytes: int, hours: int,
                       device_profile=None,
                       start_time: datetime, projection: dict, source: str,
                       name: str, ratios: tuple[int, ...],
                       root_dx_m: float = ROOT_DX_M,
                       profile: str | None = DEFAULT_PHYSICS_PROFILE,
                       cumulus_requested: bool = False,
                       vram_gib: float | None = None,
                       acknowledgements: tuple[str, ...] = (),
                       ) -> tuple[list[tuple[int, int]], ExperimentConfig]:
    """Fit one polygon-bound ladder, refusing rather than clipping it.

    Takes FREE VRAM for the same reason :func:`fit_ladder` does: the
    reserve carries the local-memory backing store of the SELECTED kernel
    set, so it is a property of the candidate experiment and cannot be
    computed before that candidate exists.  Sizing this path against a
    flat reserve is what let the point path emit configs that failed the
    product's own ``gpuwm check``, and the polygon path priced its layout
    with the identical arithmetic.

    It does NOT take :func:`fit_headroom_bytes` off the budget, and that
    asymmetry with :func:`fit_ladder` is deliberate.  That headroom is a
    property of a BISECTION: the point fitter grows the grid until the
    envelope touches the budget, so without it every emitted ladder
    landed a rounding error from the wall.  Here the polygon and its
    buffers determine the layout outright -- there is no loop to stop
    short in, and spending the headroom would refuse a user's explicit
    footprint that the enforced ``estimate <= budget`` gate accepts,
    which is a narrowing neither this fix nor the polygon route asked
    for.  The refusal below is still the suite-priced budget, so a
    layout this accepts is a layout ``gpuwm check`` accepts, and it names
    the LAYOUT rather than the card: on this route the footprint is the
    fixed thing the user controls.
    """

    dims = polygon_ladder_dims(
        footprint=footprint, projection=projection, ratios=ratios,
        buffers_km=buffers_km, root_dx_m=root_dx_m)
    text = render_config(
        name=name, start_time=start_time, hours=hours,
        projection=projection, dims=dims, ratios=ratios,
        fetch_hints=_candidate_fetch_hints(source), case_data=None,
        root_dx_m=root_dx_m, profile=profile,
        cumulus_requested=cumulus_requested,
        acknowledgements=acknowledgements)
    exp = experiment_from_text(text, source="<polygon candidate>")
    if source == "hrrr":
        try:
            refusal = coverage_refusal(exp)
        except (HrrrRouteInputError, ValueError) as error:
            # A spec-construction failure is its own refusal with its
            # own remedy.  It used to be concatenated into the coverage
            # sentence below, sending the user to move a polygon that
            # was never the problem.
            raise DomainFitError(str(error)) from None
        if refusal is not None:
            raise DomainFitError(
                "polygon plus the requested per-level buffers falls outside "
                f"HRRR coverage: {refusal}.  Move --polygon inside the HRRR "
                "grid or choose a source whose coverage includes it")
    else:
        # Every other regional source is bounded by its declared window.
        # A polygon cannot be shrunk toward coverage the way a fitted
        # ladder can -- the footprint is the user's explicit request --
        # so this is a refusal rather than a sizing bound.
        refusal = source_coverage_refusal(
            projection, dims[0][0], dims[0][1], source=source,
            root_dx_m=root_dx_m, target_option="--polygon")
        if refusal is not None:
            raise DomainFitError(
                "polygon plus the requested per-level buffers falls "
                f"outside {source} coverage: {refusal}")
    interval = source_forcing_interval_seconds(source)
    phases = estimate_phases(
        exp, source=source,
        forcing_interval_seconds=interval,
        vram_gib=vram_gib, profile=device_profile)
    budget_bytes = sizing_budget_bytes(
        exp, free_bytes=free_bytes, vram_gib=vram_gib,
        forcing_interval_seconds=interval, profile=device_profile)
    if budget_bytes <= 0 or phases.peak_envelope_bytes > budget_bytes:
        layout = ", ".join(
            f"d{index:02d} {nx}x{ny}"
            for index, (nx, ny) in enumerate(dims, 1))
        # A non-positive budget is NOT the card-is-too-small case here --
        # that one is layout-independent and already refused in `main` by
        # the CUDA-context-plus-margin floor.  This one is the layout's
        # own doing: the reserve carries a retention fraction of the
        # estimate, so a big enough footprint drives its own reserve past
        # the whole card.  `verdict` would print a negative budget, so
        # say the arithmetic instead -- but keep naming the LAYOUT first
        # either way, because on this route the layout is the fixed thing
        # the user controls and the only thing they can act on.
        if budget_bytes > 0:
            detail = phases.verdict(budget_bytes)
        else:
            detail = (
                "the external margin this card must keep for other "
                f"processes is already "
                f"{(free_bytes - budget_bytes) / GIB:.2f} GiB against "
                f"about {free_bytes / GIB:.2f} GiB free")
        raise DomainFitError(
            "polygon plus the requested per-level buffers requires "
            f"{layout}, but {detail}; reduce the "
            "buffer, choose fewer levels, increase grid spacing, or use a "
            "larger card")
    verify_polygon_containment(exp, footprint, buffers_km)
    return dims, exp


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="\n", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(temporary, path)


def render_wps_namelist(projection: dict, dims: list[tuple[int, int]],
                        ratios: tuple[int, ...],
                        root_dx_m: float = ROOT_DX_M,
                        source: str = "era5") -> str:
    """Minimal namelist.wps matching the TOML bit-for-bit.

    The config-driven pipeline reads only geog_data_res/max_dom from it,
    but the native-WRF contract checker cross-checks every projection and
    layout key against the [projection]/[[domain]] tables, so the emitted
    pair must agree exactly.

    ``&share/interval_seconds`` is the source's own forcing cadence, read
    from the source's REGISTRY ROW, as an INTEGER.  It was omitted
    entirely until 2026-08-01, and the HRRR
    domain-tree route's raw-WPS contract gate requires exactly 3600
    there -- so every wizard-emitted HRRR tree failed that route's first
    gate on a key the wizard had never written.  The importer drops the
    key for the other routes, so stating it costs them nothing and
    stating it wrong (3600.0 is refused, type-strictly) costs
    everything.
    """
    tables = _domain_tables(dims, ratios, root_dx_m=root_dx_m)
    interval_seconds = int(source_forcing_interval_seconds(source))

    def csv(values):
        return ", ".join(str(v) for v in values) + ","

    return (
        "&share\n"
        " wrf_core = 'ARW',\n"
        f" max_dom = {len(tables)},\n"
        f" interval_seconds = {interval_seconds},\n"
        " io_form_geogrid = 2,\n"
        "/\n"
        "&geogrid\n"
        f" parent_id         = {csv([1] + [t['parent_id'] for t in tables[1:]])}\n"
        f" parent_grid_ratio = {csv(t['parent_grid_ratio'] for t in tables)}\n"
        f" i_parent_start    = {csv(t['i_parent_start'] for t in tables)}\n"
        f" j_parent_start    = {csv(t['j_parent_start'] for t in tables)}\n"
        f" e_we              = {csv(t['nx'] + 1 for t in tables)}\n"
        f" e_sn              = {csv(t['ny'] + 1 for t in tables)}\n"
        f" geog_data_res     = {csv(chr(39) + 'default' + chr(39) for _ in tables)}\n"
        f" dx = {float(root_dx_m):g},\n"
        f" dy = {float(root_dx_m):g},\n"
        f" map_proj = '{projection['map_proj']}',\n"
        f" ref_lat   = {_namelist_number(projection['ref_lat'])},\n"
        f" ref_lon   = {_namelist_number(projection['ref_lon'])},\n"
        f" truelat1  = {_namelist_number(projection['truelat1'])},\n"
        f" truelat2  = {_namelist_number(projection['truelat2'])},\n"
        f" stand_lon = {_namelist_number(projection['stand_lon'])},\n"
        "/\n")


def _namelist_number(value) -> str:
    """One projection number, as Fortran will read it.

    `{value!r}` was used here, which is right for a builtin and silently
    wrong for anything else: a 0-d ndarray reprs as ``array(-160.)``,
    which the emitted namelist.wps carried verbatim into a file the
    docstring above promises matches the TOML bit-for-bit.  WPS cannot
    parse it.
    """
    scalar = _builtin_scalar(value)
    if scalar is not None:
        value = scalar
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"namelist.wps projection value {value!r} "
            f"({type(value).__name__}) is not a number; the emitted "
            "namelist has to be readable by WPS, and no repr of a "
            "non-number is.")
    return repr(float(value))


def _default_name(lat: float, lon: float) -> str:
    ns = "n" if lat >= 0 else "s"
    ew = "e" if lon >= 0 else "w"
    return (f"area_{abs(lat):.2f}{ns}_{abs(lon):.2f}{ew}"
            .replace(".", "p"))


def sizing_summary(exp: ExperimentConfig, estimate, budget_bytes: int,
                   vram_gib: float, phases=None) -> str:
    """The whole sizing verdict on one line: what fits, in what.

    The itemized table -- per-domain dx, mass grid, dt, resident bytes,
    the envelope factor and its measurement basis -- is nine lines of
    real accounting that belongs in front of anyone tuning a ladder.
    It is not what the reader of a first run needs, and it was the top
    of the wall the field exhibit opened with.  So the numbers that
    decide whether this config runs at all stay, and the derivation
    moves to ``--explain``.
    """

    envelope = estimate.peak_envelope_bytes
    phase = ""
    if phases is not None:
        envelope = phases.peak_envelope_bytes
        phase = f", binding phase {phases.binding_phase}"
        if not phases.ingest_priced:
            phase += " -- ingest NOT PRICED for this source"
    # The wizard always sizes a DECLARED card (--card/--vram-gib), never
    # a measured one, and the 4090 stress run showed what an unlabelled
    # "fits" costs on that path: a certified 0.27 GiB margin that landed
    # 0.015 GiB from the budget.  The verdict carries the label.
    return (f"sizing: {len(exp.domains)} domain(s); alloc "
            f"{estimate.alloc_estimate_bytes / GIB:.2f} GiB, peak "
            f"envelope {envelope / GIB:.2f} GiB of a "
            f"{budget_bytes / GIB:.2f} GiB budget "
            f"({(budget_bytes - envelope) / GIB:.2f} GiB headroom{phase}) "
            f"-- an estimate for a declared {vram_gib:g} GiB card, not a "
            f"measurement of hardware in this machine; `gpuwm check` on "
            f"the real card is what measures it")


def _print_sizing_table(exp: ExperimentConfig, estimate,
                        budget_bytes: int, vram_gib: float,
                        phases=None) -> None:
    envelope = estimate.peak_envelope_bytes
    print("sizing (itemized preflight estimator, in-process):")
    print("  domain    dx        mass grid      dt         resident")
    for dc, dom in zip(exp.domains, estimate.domains):
        dx_km = exp.dx_exact(dc.grid_id) / 1000
        dt = exp.dt_exact(dc.grid_id)
        dt_text = (f"{int(dt)} s" if dt.denominator == 1
                   else f"{dt.numerator}/{dt.denominator} s")
        print(f"  d{dc.grid_id:02d}     {float(dx_km):6.3f} km  "
              f"{dc.run.nx:4d} x {dc.run.ny:<4d}   {dt_text:>8}   "
              f"{dom.resident_bytes / GIB:6.2f} GiB")
    family = envelope_platform(vram_gib=vram_gib)
    print(f"  peak envelope: {estimate.peak_envelope_terms()}")
    print(f"    envelope basis: {family}; {estimate.envelope_basis}")
    # An unmeasured platform gets the conservative accounting, which is
    # a substitution the user has to be able to see.
    platform_note = unknown_platform_note()
    if platform_note is not None:
        print(f"    {platform_note}")
    if phases is not None and not phases.ingest_priced:
        print(f"  ingest (preprocessing): NOT PRICED for --source "
              f"{phases.source} -- that lane is the native-hybrid-level "
              "ingest, which this estimator does not model; the envelope "
              "above is the forecast phase only")
    elif phases is not None:
        ingest = phases.ingest
        nest_ingest = (
            f" + {len(ingest.nest_state_items)} nest initial state(s) "
            f"{ingest.nest_state_bytes / GIB:.2f} GiB, all resident for "
            f"the single export transaction"
            if ingest.nest_state_items else "")
        print(f"  ingest (preprocessing): root {ingest.n_forcing_times} "
              f"forcing times x {ingest.per_time_bytes / GIB:.2f} GiB each, "
              f"{ingest.resident_times} resident at a time"
              f"{nest_ingest} = "
              f"{ingest.resident_bytes / GIB:.2f} GiB resident; peak "
              f"envelope {ingest.peak_envelope_bytes / GIB:.2f} GiB")
        print(f"    ingest envelope basis: {INGEST_PEAK_ENVELOPE_BASIS}")
    if phases is not None:
        print(f"  BINDING PHASE: {phases.verdict(budget_bytes)}")
        envelope = phases.peak_envelope_bytes
    reserve_gib = (card_assumed_free_gib(vram_gib)
                   - budget_bytes / GIB)
    print(f"  budget {budget_bytes / GIB:.2f} GiB "
          f"({vram_gib:g} GiB card presents about "
          f"{card_assumed_free_gib(vram_gib):g} GiB free, minus this "
          f"suite's {reserve_gib:.2f} GiB reserve); headroom "
          f"{(budget_bytes - envelope) / GIB:.2f} GiB")
    print("  ESTIMATE FOR HARDWARE NOT PRESENT: every figure above is an "
          "estimate for the declared card, not a measurement of hardware "
          "in this machine.  Non-pool terms are priced against the "
          "conservative measured reference device profile (the largest "
          "known-device intercept), so the estimate is never more "
          "optimistic than a present-card measurement; `gpuwm check` on "
          "the real card is what measures it.")


def _missing_case_inputs(out: Path, case_data: dict) -> list[str]:
    from gpuwm.case_data import expand_path_variables
    base = out.parent
    missing = []

    def resolve(raw: str) -> Path:
        expanded = Path(expand_path_variables(raw, "wizard", str(out)))
        return expanded if expanded.is_absolute() else base / expanded

    for raw in case_data["forcing"]:
        if not resolve(raw).is_file():
            missing.append(f"forcing {raw}")
    for key in ("vtable", "wps_namelist"):
        if not resolve(case_data[key]).is_file():
            missing.append(f"{key} {case_data[key]}")
    root = resolve(case_data["geog_root"])
    if not root.is_dir():
        missing.append(f"geog_root {case_data['geog_root']}")
    else:
        for dataset in GEOG_DATASETS:
            if not (root / dataset).is_dir():
                missing.append(f"geog_root dataset {dataset}")
    return missing


def gray_zone_headline(chain_km, shared: dict) -> list[str]:
    """The gray-zone advisory's first clause: the finding, no mechanism.

    The full :func:`gray_zone_advisory` sentence is four printed lines
    of correct and load-bearing science, and it is what the emitted
    config carries in its header comment -- permanently, where it is
    read next to the settings it is about.  On stdout it was competing
    with the one line the reader needed, so stdout gets the finding and
    ``--explain`` (and the file itself) keep the reasoning.

    Derived from the same call, never a second transcription of the
    numbers: a headline that could disagree with the advisory would be
    worse than no headline.
    """

    full = gray_zone_advisory(chain_km, shared)
    if not full:
        return []
    return [full[0].split(", so ", 1)[0] + "."]


def cumulus_gray_zone_headline(chain_km, cu_physics_by_domain
                               ) -> list[str]:
    """Each cumulus finding's first clause -- same contract as
    :func:`gray_zone_headline`: derived from the same call, never a
    second transcription of the numbers, so a headline that could
    disagree with the advisory cannot exist.  The emitted config's
    header comment carries the full sentences either way."""

    return [line.split(", so ", 1)[0] + "."
            for line in cumulus_gray_zone_advisory(
                chain_km, cu_physics_by_domain)]


def _print_geog_help() -> None:
    print("  static geography: gpuwm reads a locally staged NCAR WPS_GEOG "
          "tree; `gpuwm fetch-geog` downloads and stages it (~1.3 GB "
          "compressed, ~16 GB unpacked, resumable).  The geog_root "
          "directory must contain these dataset directories:")
    print("    " + ", ".join(GEOG_DATASETS))


#: The pairing predicate -- why a source cannot prepare a profile, or
#: ``None`` -- and the emission-route gate table behind it.  Both are
#: DEFINED in :mod:`gpuwm.physics_menu` and imported above; the
#: predicate is re-exported under this name because every reader in the
#: tree spells it ``domain_wizard.profile_route_blocker``.


def profiles_blocked_on_source(source: str) -> tuple[str, ...]:
    """The shipped profiles ``source`` refuses, in listed order.

    Exists so ``--help`` can say which of the eight it advertises are
    not selectable on a given route.  DERIVED from the registry, never
    listed: a hard-coded pair would be a second declaration of the same
    fact and would go stale the moment a route gained the component
    back, leaving the help lying in the other direction.
    """

    return tuple(profile for profile in WIZARD_PHYSICS_PROFILES
                 if profile_route_blocker(profile, source) is not None)


def _profile_help_route_note() -> str:
    """The ``--help`` sentence about profiles a route cannot prepare.

    ``--help`` listed eight profiles with no marker while two of them
    were refused unconditionally on ``--source gfs`` -- which is the
    DEFAULT source -- so a reader choosing from the list had a 1-in-4
    chance of picking something that could never work, and learned it
    only from the refusal.  The refusal itself is honest and precise;
    the advertisement was not.

    Empty when every listed profile is preparable on every source,
    which is what this note existing at all is waiting for.

    It walks EVERY plannable source rather than the three the door used
    to offer, so widening the door cannot leave a route's refusals
    unadvertised -- the same one-in-four gap, one layer up.
    """

    notes = []
    for source in planable_sources():
        blocked = profiles_blocked_on_source(source)
        if blocked:
            notes.append(f"--source {source} cannot prepare "
                         + " or ".join(blocked))
    if not notes:
        return ""
    return ("  NOT every profile runs on every route: " + "; ".join(notes)
            + " -- the wizard refuses those pairings and names the "
              "missing component rather than emitting a config the "
              "front door would reject.  ")


#: The source ``--source`` binds when none is named.  One constant, read
#: by the flag and by the help sentence below, because the help used to
#: state two sources' defaults as literal text and a route gate moving
#: would have left it lying.
DEFAULT_WIZARD_SOURCE = "era5"


def _profile_help_default_note() -> str:
    """The ``--help`` sentence about what a bare run binds.

    DERIVED.  This sentence used to read "(gfs/era5 default: <id> ...;
    hrrr default: <id> ...)" -- two source names and two profile ids
    typed into help text, correct on the day they were written and
    silently wrong the moment a route gate moves or a third gated source
    is registered.  Every source has its own computed default now, so
    help names the DEFAULT source's and points at the door that answers
    for the rest rather than pretending to enumerate them.
    """

    default = resolved_physics_profile(DEFAULT_WIZARD_SOURCE, None)
    return (f"(--source {DEFAULT_WIZARD_SOURCE}, the default source, "
            f"binds {default}; every source has its own computed default "
            "and its own admissible set -- `gpuwm run-plan "
            "--physics-profiles` prints the whole table)")


def _refuse_profile_its_source_cannot_prepare(profile, source) -> None:
    """Do not emit a config the named source's front door will refuse.

    The wizard prints, of a profile-bound config, that it "passes the
    prepared single-domain forecast runner's physics guard exactly as
    emitted".  That sentence has to stay true.  When a route withdraws a
    component -- GFS and RUC, whose forecast cannot complete its first
    step -- the wizard must stop offering the pairing rather than write
    the file and let the front door refuse it later, which is the same
    selectable-but-not-usable shape one surface earlier.

    Scoped by the same registry declaration the front door enforces, so
    the two cannot disagree, and silent for every pairing that
    declaration still offers.
    """

    blocker = profile_route_blocker(profile, source)
    if blocker is not None:
        # One sentence at the boundary; the registry pointer and the
        # failure mechanism ride the --explain layer.
        from gpuwm.explain import layered
        head, _, detail = str(blocker).partition(": ")
        raise ValueError(layered(
            f"--physics-profile {profile} cannot be prepared with "
            f"--source {source}: {head}", detail))


def resolve_sizing_card(card: str | None, vram_gib: float | None):
    """The VRAM the wizard sizes against, the CARD, and where they came from.

    Returns ``(vram_gib, device_profile, sentence)``.  ``device_profile``
    is the measured local card when nothing was declared and ``None``
    when the caller declared one -- a declaration says "size for a
    machine that need not be this one", and that machine's shader count
    is unknown, so it keeps the conservative reference profile.

    THE PROFILE IS THE POINT.  The capacity was already measured here;
    what was thrown away was the rest of the probe's answer, so a wizard
    that knew it was looking at a 68-SM RTX 3080 priced that card's
    local-memory backing store against a 170-SM reference and charged it
    1.49 GiB it does not have (task 206).
    """

    return _resolve_vram_budget(card, vram_gib)


def _resolve_vram_budget(card: str | None,
                         vram_gib: float | None):
    """The VRAM the wizard sizes against, and where the number came from.

    Three sources, in the only defensible order:

    1. A DECLARATION (``--card`` tier or ``--vram-gib``) wins outright and
       nothing local is probed -- the caller said "size for a machine that
       need not be this one", and a probe would at best be ignored.
    2. Nothing declared: the local card is MEASURED, through the same
       short-lived subprocess probe the `go` memory gate uses (the
       measured-thresholds rule -- a number this box can produce beats an
       assumed tier).  The capacity is what is read; free VRAM is a
       moment's answer and this file outlives the moment.
    3. Nothing declared and nothing measurable: a refusal that names the
       real choice.  This replaces two prior behaviors, both wrong: a
       silent 24 GiB assumption (a config sized for a card nobody has),
       and -- on CPU-only installs -- a cupy package check for a command
       that integrates nothing on a card (the 2.5.0 persona walks'
       finding).

    Returns ``(vram_gib, device_profile, sentence)``.  ``sentence`` is
    the measurement announcement to print, or ``None`` for a
    declaration; ``device_profile`` is the MEASURED card, or ``None``
    for a declaration -- which prices against the conservative reference
    profile, because the machine being sized for is somewhere else.
    """

    if card is not None:
        return CARD_VRAM_GIB[card], None, None
    if vram_gib is not None:
        return float(vram_gib), None, None
    probe = device_memory_probe_subprocess()
    total = probe.get("total_bytes") if isinstance(probe, dict) else None
    if isinstance(total, int) and not isinstance(total, bool) and total > 0:
        measured = total / GIB
        # The SAME probe answer, used WHOLE.  Reading the capacity out of
        # it and dropping the shader census beside it is exactly how a
        # measured 68-SM card came to be priced on a 170-SM profile.
        device_profile = profile_from_device_probe(probe)
        name = device_profile.name if device_profile is not None else None
        card_words = f"{name}, " if name else ""
        basis = ("" if device_profile is None else
                 f"; grid-independent terms {non_pool_basis(device_profile)}")
        return measured, device_profile, (
            f"domain: no --card/--vram-gib declared, so the budget is the "
            f"measured local card ({card_words}{measured:g} GiB total); "
            f"declare --card or --vram-gib to size for another machine"
            f"{basis}")
    reason = (device_memory_probe_reason()
              or "the local card could not be measured")
    tiers = "/".join(sorted(CARD_VRAM_GIB))
    if "CuPy" in reason or "cupy" in reason:
        way_back = ("or install cupy (pip install 'gpuwm[gpu-cu12]', or "
                    "'gpuwm[gpu-cu13]' on a CUDA-13 box) so the wizard "
                    "can measure the local card")
    else:
        way_back = ("or make the local card readable so the wizard can "
                    "measure it")
    from gpuwm.explain import layered
    raise ValueError(layered(
        f"this wizard sizes every emitted level against a VRAM budget, "
        f"and there is none: no --card/--vram-gib was declared, and "
        f"{reason}.  Declare the target card -- --card {tiers} or "
        f"--vram-gib N -- {way_back}.",
        "The budget decides every grid dimension in the emitted file, so "
        "the wizard needs one of three sources: a declared tier, a "
        "declared GiB figure, or a measurement of the card in this "
        "machine.  It used to assume a 24 GiB card when nothing was "
        "declared, which sized domains for hardware nobody stated exists; "
        "an assumption is not a budget.  The measurement runs in a "
        "short-lived subprocess (no CUDA context survives in this "
        "process) and GPUWM_NO_LOCAL_GPU suppresses it entirely."))


def domain_main(args) -> int:
    # FIRST, before any geometry: the source name becomes a registry row,
    # or the run stops with the registry's own words.  Everything below
    # reads the row -- coverage, cadence, forecast horizon -- so a name
    # that never resolved would have been re-guessed at four later points.
    args.source = resolve_source(args.source)
    polygon = None
    level_buffer_values = None
    polygon_path = getattr(args, "polygon", None)
    if polygon_path is None:
        lat, lon = _parse_point(args.point)
        if getattr(args, "buffer_km", None) is not None:
            raise ValueError("--buffer-km requires --polygon")
    else:
        polygon = load_polygon_footprint(polygon_path)
        lat, lon = polygon.center_lat, polygon.center_lon
        level_buffer_values = parse_level_buffers(
            getattr(args, "buffer_km", None))
    if args.card is not None and args.vram_gib is not None:
        raise ValueError("--card and --vram-gib are mutually exclusive")
    _refuse_profile_its_source_cannot_prepare(
        getattr(args, "physics_profile", None), args.source)
    vram_gib, device_profile, budget_sentence = _resolve_vram_budget(
        args.card, args.vram_gib)
    if budget_sentence is not None:
        print(budget_sentence)
    # The floor is the reserve NOTHING can be sized below: one CUDA
    # context plus the external margin.  Deliberately not the flat
    # `vram_reserve_gib` any more -- that figure is retired from the
    # sizing path, and using it here refused cards the suite-priced
    # reserve would have sized.  Everything above this floor goes to
    # the fit loop, whose refusal names the layout and the
    # arithmetic.
    reserve_floor_gib = (CUDA_CONTEXT_BYTES + EXTERNAL_MARGIN_BYTES) / GIB
    if not math.isfinite(vram_gib):
        # Separated from the arithmetic branch below, because that branch
        # DESCRIBES the card it was given and there is no such card here.
        # card_assumed_free_gib launders a non-finite value through
        # max(), which returns the finite operand, so the shared message
        # reported `--vram-gib nan` as a card presenting "about 0.00 GiB
        # free" -- a specific, false, plausible number invented for an
        # input that has no meaning.
        raise ValueError(
            f"--vram-gib {vram_gib:g} is not a size: it names no amount "
            f"of memory, so there is nothing to size a domain against.  "
            f"Pass the card's capacity in GiB, for example --vram-gib 16.")
    if card_assumed_free_gib(vram_gib) <= reserve_floor_gib:
        raise ValueError(
            f"--vram-gib {vram_gib:g} leaves no budget: a card that size "
            f"presents about {card_assumed_free_gib(vram_gib):.2f} GiB "
            f"free, and one CUDA context plus the external margin is "
            f"already {reserve_floor_gib:.2f} GiB of it")
    # FREE, not a budget.  The reserve belongs to the candidate experiment
    # -- it carries that suite's local-memory backing store -- so it is
    # subtracted inside the fit loop, by the same call `gpuwm check`
    # makes.  A card also never hands over its nominal capacity: see
    # CARD_UNAVAILABLE_VRAM_GIB.
    free_gib = card_assumed_free_gib(vram_gib)
    free_bytes = int(free_gib * GIB)
    if args.hours < 1:
        raise ValueError("--hours must be at least 1")
    # The model's time zero.  It is the cycle when the run is initialized
    # from the analysis, and cycle + K when it is initialized from the
    # f{K} forecast lead -- which is a routine thing to want and used to
    # be impossible to ask for on this door.  Checked BEFORE the cycle is
    # resolved, because `--cycle latest` probes the mirrors for a cycle
    # complete through lead + length and a bad lead should not spend a
    # network round trip to be refused.
    start_hour = getattr(args, "forecast_start_hour", None) or 0
    if start_hour < 0:
        raise ValueError(
            "--forecast-start-hour must be a nonnegative forecast lead")
    if start_hour and not source_reaches_forecast_leads(args.source):
        raise ValueError(
            f"--forecast-start-hour: {args.source} publishes no forecast "
            "leads (its registry row declares max_forecast_hour = 0), so "
            "every time in it is an analysis at its own valid time and "
            "there is no lead to begin at; name the analysis time you want "
            "with --cycle")
    horizon = get_source_adapter(args.source).max_forecast_hour
    if horizon and start_hour + args.hours > horizon:
        # Named here, from the source's own declared horizon, rather than
        # after the acquisition.  gdas stops at f009 and rap at f051; a
        # window that walks past either is a window the product never
        # published, and it used to be discovered by whichever stage first
        # went looking for the missing lead.
        raise ValueError(
            f"--hours {args.hours} beginning at f{start_hour:03d} reaches "
            f"f{start_hour + args.hours:03d}, past {args.source}'s declared "
            f"f{horizon:03d} horizon; shorten the window, start earlier, or "
            "choose a source with a longer forecast")
    cycle = _resolve_cycle(
        args.cycle, source=args.source, hours=args.hours,
        start_hour=start_hour)
    if args.source == "hrrr":
        # The cycle horizon is a property of the cycle hour (48 h at
        # 00/06/12/18Z, 18 h otherwise), so a lead can walk a window off
        # the end of what NOAA published.  Refused HERE, with the horizon
        # named, rather than at the fetch after the file has been written.
        from gpuwm.hrrr_forecast import hrrr_source_window
        hrrr_source_window(cycle=cycle, start_hour=start_hour,
                           run_seconds=args.hours * 3600.0)
    start_time = cycle + timedelta(hours=start_hour)
    name = args.name or _default_name(lat, lon)
    projection = _projection_entries(
        lat, lon, getattr(args, 'projection', 'auto'))
    out: Path = args.out

    profile = resolved_physics_profile(
        args.source, getattr(args, "physics_profile", None))
    # NAMING THE SUITE IS THE EXPLICIT CUMULUS REQUEST.  It is the only
    # cumulus statement this door takes, and it is a strong one: naming
    # --physics-profile asserts the config IS that shipped suite, which
    # both prepared routes then enforce switch for switch
    # (gpuwm.gfs_direct.front_door_physics_selection).  Emitting it with
    # a switch retired would hand the user a file the runner they were
    # steered to refuses.  Without the flag the suite is DERIVED, nobody
    # asserted its cumulus scheme, and the grid decides
    # (see _domain_tables).
    #
    # The interactive session composes --physics-profile with the same
    # DERIVED default and prints the command for re-running, so on that
    # door the flag is not a person's assertion.  It costs nothing
    # today: that door pins --ladder 12, whose root is three times the
    # convection-permitting bound, so neither branch can differ there.
    # A future interactive --root-dx has to decide this deliberately
    # rather than inherit it.
    cumulus_requested = getattr(args, "physics_profile", None) is not None

    # THE NOCTURNAL DECLARATION IS THE USER'S TO MAKE, AND THIS IS WHERE
    # THEY MAKE IT (2026-08-09).
    #
    # Until today this door wrote
    # acknowledgements = [ASYMMETRIC_RADIATION_NOCTURNAL_ACK] into the
    # emitted [experiment] by itself whenever an asymmetric suite met a
    # window with local night in it.  Every downstream door reads that
    # line and falls silent: `gpuwm check`, `gpuwm run`, `gpuwm go`,
    # run-plan and both prepared runners.  So the wizard was manufacturing
    # the user's consent, into a file that outlives the session, for the
    # exact failure v1.7.1 shipped a guard for -- and the audit reproduced
    # a real user's journey ending in a config that could never be
    # refused by anything.  Speaking the declaration aloud (the advisory
    # below, from lane/advisory) was necessary and is not sufficient: the
    # FILE still carried a statement nobody made.
    #
    # Refuse instead, here, before any fitting or fetching, and name the
    # two ways forward.  --ack is the project's existing idiom for
    # exactly this (gpuwm/cli.py, gfs_direct, prepared_single_domain_
    # forecast, source_cli), so the user types the token themselves and
    # the emitted file records a decision that was actually taken.
    acknowledgements = tuple(getattr(args, "ack", None) or ())
    declared_night = declared_nocturnal_night(
        profile, start_time=start_time, hours=args.hours,
        projection=projection)
    if (declared_night is not None
            and ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in acknowledgements):
        from gpuwm.explain import layered
        from gpuwm.physics_menu import nocturnal_remedy

        # THE REMEDY IS THIS SOURCE'S, NOT A FIXED ID (2026-08-20).
        #
        # This sentence used to name MORRISON_PROFILE_ID on every
        # source.  On --source hrrr that is a Kain-Fritsch suite, and
        # the pairing refusal above (_refuse_profile_its_source_cannot_
        # prepare) then refuses it for cu_physics=1, because the route's
        # 3 km grid resolves its own convection.  So the user was handed
        # a remedy that leads to the next refusal, which names nothing.
        # The remedy comes off the computed per-source menu now -- the
        # same table `gpuwm run-plan --physics-profiles` serves -- so it
        # cannot name a suite this source's route refuses.
        remedy = nocturnal_remedy(args.source)
        raise ValueError(layered(
            f"profile {profile} runs shortwave with longwave OFF and this "
            f"window includes local night (first at "
            f"{declared_night:%Y-%m-%dT%H:%M}Z at "
            f"{projection['ref_lat']:.4g}, {projection['ref_lon']:.4g}), so "
            f"the config this would emit is one every front door refuses "
            f"at load.  Choose a nocturnally valid profile with both "
            f"radiation streams on that --source {args.source} can "
            f"actually prepare -- {remedy['instruction']} -- or, if you "
            f"mean the daytime validation suite and accept the night, "
            f"declare it yourself with --ack "
            f"{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}.  `gpuwm run-plan "
            f"--physics-profiles` lists every suite this source admits",
            "Shortwave heats the surface by day while no longwave scheme "
            "runs, so after sunset the surface radiates to space with no "
            "downward longwave to balance it: skin temperature craters, "
            "the surface saturation humidity collapses with it, and 2 m "
            "dewpoints read far below the airmass.  This wizard used to "
            "write that acknowledgement into the emitted [experiment] for "
            "you, which silenced the guard at every later command for the "
            "life of the file.  A declaration nobody made is not a "
            "declaration.  See docs/public/PHYSICS.md, 'Nocturnal "
            "validity'."))

    if args.ladder is None:
        # Absence, resolved: bare means the single-domain `go` shape
        # (DEFAULT_LADDER); with --root-dx/--chain it means the custom
        # form, which has always ridden on the permissive "auto" value
        # -- the guard below refuses only a --ladder someone TYPED next
        # to the custom flags.
        args.ladder = ("auto"
                       if (getattr(args, "root_dx", None) is not None
                           or getattr(args, "chain", None) is not None)
                       else DEFAULT_LADDER)
    custom = parse_custom_ladder(
        root_dx_km=getattr(args, "root_dx", None),
        chain=getattr(args, "chain", None),
        ladder=args.ladder)
    level_buffers = None
    if custom is not None:
        root_dx_m, ratios = custom
        if polygon is None:
            dims, _ = fit_ladder(
                ratios=ratios, root_dx_m=root_dx_m, free_bytes=free_bytes,
                hours=args.hours, start_time=start_time,
                projection=projection, source=args.source, name=name,
                profile=profile, cumulus_requested=cumulus_requested,
                vram_gib=vram_gib,
                device_profile=device_profile,
                acknowledgements=acknowledgements)
        else:
            level_buffers = _buffers_for_levels(
                level_buffer_values, len(ratios) + 1)
            dims, _ = fit_polygon_ladder(
                footprint=polygon, buffers_km=level_buffers,
                ratios=ratios, root_dx_m=root_dx_m, free_bytes=free_bytes,
                hours=args.hours, start_time=start_time,
                projection=projection, source=args.source, name=name,
                profile=profile, cumulus_requested=cumulus_requested,
                vram_gib=vram_gib,
                device_profile=device_profile,
                acknowledgements=acknowledgements)
        ladder = "-".join(f"{v:g}" for v in _ladder_dx_km(ratios, root_dx_m))
    else:
        root_dx_m = ROOT_DX_M
        ladders = ([args.ladder] if args.ladder != "auto"
                   else list(_LADDERS_DEEPEST_FIRST))
        fixed_buffer_depth = False
        if polygon is not None and level_buffer_values is not None \
                and len(level_buffer_values) > 1 and args.ladder == "auto":
            fixed_buffer_depth = True
            ladders = [candidate for candidate in ladders
                       if len(LADDER_RATIOS[candidate]) + 1
                       == len(level_buffer_values)]
            if not ladders:
                raise ValueError(
                    f"--buffer-km supplies {len(level_buffer_values)} "
                    "per-level distances, but no preset ladder has that "
                    "many levels; use --root-dx / --chain for a custom "
                    "ladder")
        chosen = None
        for candidate_ladder in ladders:
            try:
                candidate_ratios = LADDER_RATIOS[candidate_ladder]
                if polygon is None:
                    dims, _ = fit_ladder(
                        ladder=candidate_ladder, free_bytes=free_bytes,
                        hours=args.hours, start_time=start_time,
                        projection=projection, source=args.source, name=name,
                        profile=profile,
                        cumulus_requested=cumulus_requested,
                        vram_gib=vram_gib,
                        device_profile=device_profile,
                        acknowledgements=acknowledgements)
                    candidate_buffers = None
                else:
                    candidate_buffers = _buffers_for_levels(
                        level_buffer_values, len(candidate_ratios) + 1)
                    dims, _ = fit_polygon_ladder(
                        footprint=polygon, buffers_km=candidate_buffers,
                        ratios=candidate_ratios, root_dx_m=root_dx_m,
                        free_bytes=free_bytes, hours=args.hours,
                        start_time=start_time, projection=projection,
                        source=args.source, name=name, profile=profile,
                        cumulus_requested=cumulus_requested,
                        vram_gib=vram_gib, device_profile=device_profile,
                        acknowledgements=acknowledgements)
            except DomainFitError as error:
                if args.ladder != "auto" or fixed_buffer_depth:
                    raise
                # One line per candidate; the full envelope arithmetic
                # is the same for every ladder and prints with the
                # final refusal if none fits.
                first_sentence = str(error).split(":", 1)[0]
                print(f"ladder {candidate_ladder}: does not fit "
                      f"({first_sentence}); trying the next shallower one")
                continue
            chosen = (candidate_ladder, dims, candidate_buffers)
            break
        if chosen is None:
            raise DomainFitError(
                "no ladder fits the requested card; even the shallowest "
                f"ladder's smallest layout exceeds the budget a "
                f"{vram_gib:g} GiB card leaves (about {free_gib:g} GiB "
                "free, minus this suite's reserve)")
        ladder, dims, level_buffers = chosen
        ratios = LADDER_RATIOS[ladder]
    # Genuine-limit refusal first (its message names the real problem;
    # a pole-containing footprint would otherwise also trip the
    # 180-degree fetch-span refusal below with a less useful message).
    target_option = "--polygon" if polygon is not None else "--point"
    _pole_clearance_refusal(
        projection, *dims[0], root_dx_m,
        target_option=target_option)

    # Fetch hints from the fitted root footprint.  The default data
    # directory lives beside the emitted TOML so the declared forcing
    # paths stay short and the config directory stays relocatable.  The
    # area hint is bounded by the SOURCE's own coverage envelope -- the
    # same grid-derived data the fetch guard enforces -- so the printed
    # next command cannot name ground the source does not carry.
    area_notes: list[str] = []
    coverage_notes: list[str] = []
    area_hint = fetch_area_hint(
        projection, *dims[0], source=args.source, root_dx_m=root_dx_m,
        target_option=target_option, notes=area_notes,
        coverage_notes=coverage_notes)
    cadence = _fetch_cadence_h(args.source, start_hour)
    data_dir = (Path(args.data_dir) if args.data_dir
                else out.parent / "data" / name)
    fetch_hints = {
        # The RESOLVED cycle, never the literal "latest": the emitted
        # config is a record of one start time, not of a query.  And the
        # CYCLE, never the start time: they differ by the forecast lead,
        # and a fetch aimed at start_time would ask for the wrong cycle
        # entirely.
        "source": args.source, "cycle": cycle.strftime("%Y-%m-%dT%H"),
        "hours": (args.hours if cadence is None else
                  max(cadence, math.ceil(args.hours / cadence) * cadence)),
        "area": area_hint,
        "out": _relative_or_absolute(data_dir, Path.cwd()),
    }
    if cadence is not None:
        fetch_hints["cadence"] = cadence
    if start_hour:
        fetch_hints["forecast_start_hour"] = start_hour
    # A [fetch] table is a claim that `gpuwm fetch` can go and get these
    # bytes.  For a source with no download route that claim is false, and
    # the table would be refused at every later config load anyway
    # (validate_fetch_hints checks it against the routes that exist), so
    # the emission carries the geometry and states the acquisition gap in
    # the file's own header instead of advertising a step that refuses.
    emitted_fetch_hints = (fetch_hints
                           if source_has_fetch_front_door(args.source)
                           else None)
    # And the crop key comes out for a source whose fetch takes whole
    # published objects.  The area is still COMPUTED (the advisories and
    # the hand-staging note below both say what window this config needs)
    # -- it is not ADVERTISED as a download flag the fetch would refuse.
    if (emitted_fetch_hints is not None
            and not source_fetch_takes_a_crop_box(args.source)):
        emitted_fetch_hints = {k: v for k, v in fetch_hints.items()
                               if k not in {"area", "point", "radius_km"}}
    # Prove every emitted hint against the REAL fetch validators before
    # anything is written.  A config the wizard cannot fetch is a config
    # whose printed step 1 exits 2, and that shipped twice: any lead not
    # a multiple of three was written with `cadence = 3` and refused as
    # "not on the 3 h cadence", and an HRRR emission's --area was
    # refused by the coverage guard the wizard had never consulted.
    # Running the fetch's own parsers and planners here is the only
    # check that cannot drift from what the fetch will do; the whole
    # [fetch] table (area included, through parse_area and the
    # per-source coverage gate) is round-tripped again through
    # `validate_fetch_hints` when `experiment_from_text` re-loads the
    # rendered bytes below, still before the file lands on disk.
    parse_cycle(fetch_hints["cycle"], args.source)
    if args.source in {"gfs", "gdas"}:
        from gpuwm.fetch import gfs_forecast_hours
        gfs_forecast_hours(int(fetch_hints["hours"]), cadence, start_hour)

    # [case_data] only where the config-driven front door can honestly
    # consume the fetched data (the native GRIB1 = ERA5 route).
    case_data = None
    vtable_path = None
    if args.source == "era5":
        if args.vtable is not None:
            vtable_path = Path(args.vtable)
            vtable_text = _posix(vtable_path.resolve())
        else:
            vtable_path = out.parent / _PACKAGED_VTABLE.name
            vtable_text = _PACKAGED_VTABLE.name
        forcing = ([_posix(Path(p).resolve()) for p in args.forcing]
                   if args.forcing
                   else [_relative_or_absolute(
                       data_dir / "era5-combined.grib", out.parent)])
        case_data = {
            "forcing": forcing,
            "vtable": vtable_text,
            "forcing_interval_s": source_forcing_interval_seconds("era5"),
            "wps_namelist": f"{out.stem}.namelist.wps",
            "geog_root": (_posix(Path(args.geog_root).resolve())
                          if args.geog_root
                          else "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG"),
            "sfcp_to_sfcp": True,
            "output_domain": 1,
            "output_title": f"gpuwm {name}",
        }

    text = render_config(
        name=name, start_time=start_time, hours=args.hours,
        projection=projection, dims=dims, ratios=ratios,
        fetch_hints=emitted_fetch_hints, case_data=case_data,
        root_dx_m=root_dx_m, profile=profile,
        cumulus_requested=cumulus_requested,
        interactive=getattr(args, "interactive", False),
        level_buffers_km=level_buffers,
        history_interval_s=args.history_interval,
        nest_history_interval_s=args.nest_history_interval,
        acknowledgements=acknowledgements)
    # Round-trip the exact bytes through the real loader before writing.
    exp = experiment_from_text(text, source=str(out))
    interval = source_forcing_interval_seconds(args.source)
    estimate = estimate_experiment(
        exp, forcing_interval_seconds=interval, vram_gib=vram_gib,
        profile=device_profile)
    phases = estimate_phases(
        exp, source=args.source, forcing_interval_seconds=interval,
        vram_gib=vram_gib, profile=device_profile)
    envelope = phases.peak_envelope_bytes
    # The budget the EMITTED config gets -- its own reserve, not a flat
    # one, which is the number `gpuwm check --budget-gib` is handed below
    # so the two commands cannot disagree about the same file.
    budget = sizing_budget_bytes(
        exp, free_bytes=free_bytes, vram_gib=vram_gib,
        forcing_interval_seconds=interval, profile=device_profile)
    budget_gib = budget / GIB
    if envelope > budget:
        raise DomainFitError(
            "internal fit regression: emitted config's "
            f"{phases.binding_phase} envelope "
            f"{envelope / GIB:.2f} GiB exceeds the budget "
            f"{budget_gib:.2f} GiB")
    if polygon is not None:
        verify_polygon_containment(exp, polygon, level_buffers)

    _write_atomic(out, text)
    wps_text = render_wps_namelist(
        projection, dims, ratios, root_dx_m=root_dx_m, source=args.source)
    if args.source == "hrrr":
        # The HRRR routes read namelists and a target-domain document,
        # not this TOML.  Emitting only the TOML left every one of them
        # to be hand-authored -- which is why the route's own gate had
        # to borrow a proof harness to run at all.
        written = [out] + write_hrrr_route_inputs(
            out, exp, wps_text=wps_text, writer=_write_atomic)
    else:
        wps_path = out.parent / f"{out.stem}.namelist.wps"
        _write_atomic(wps_path, wps_text)
        written = [out, wps_path]
    if args.source == "era5" and args.vtable is None:
        if vtable_path.exists():
            if vtable_path.read_bytes() != _PACKAGED_VTABLE.read_bytes():
                warn(f"kept your existing {vtable_path.name} (it differs "
                     "from the packaged Vtable.ERA5_CDO); pass --vtable "
                     "to name one explicitly")
        else:
            shutil.copyfile(_PACKAGED_VTABLE, vtable_path)
            written.append(vtable_path)

    explain = explain_enabled(args)
    if polygon is None:
        print(f"gpuwm domain: {name!r} at ({lat:g}, {lon:g}), ladder {ladder} "
              f"({'-'.join(f'{v:g}' for v in _ladder_dx_km(ratios, root_dx_m))} km), "
              f"card {vram_gib:g} GiB")
    else:
        buffers = ",".join(f"{value:g}" for value in level_buffers)
        print(f"gpuwm domain: {name!r}, polygon center ({lat:g}, {lon:g}), "
              f"ladder {ladder} "
              f"({'-'.join(f'{v:g}' for v in _ladder_dx_km(ratios, root_dx_m))} km), "
              f"buffers {buffers} km, card {vram_gib:g} GiB")
    # The spoken half of snap_cadences_to_clock: what the author moved
    # to keep its own two derivations compatible, one line each, or
    # nothing (UX finding N14 -- the silent alternative was exit 2).
    for note in snap_cadences_to_clock(
            root_time_step_s(projection["ref_lat"], root_dx_m),
            {key: profile_switches(profile)[key]
             for key in _PER_DOMAIN_PHYSICS})[1]:
        print(f"domain: {note}")
    if explain:
        _print_sizing_table(exp, estimate, budget, vram_gib,
                            phases=phases)
    else:
        print(sizing_summary(exp, estimate, budget, vram_gib,
                             phases=phases))
    print(f"wrote {out}"
          + (f" (+ {', '.join(p.name for p in written[1:])})"
             if len(written) > 1 else ""))
    # The printed command must be pasteable as-is from THIS directory.
    # A relative data path that climbs out of the cwd silently targets
    # the wrong place when pasted from anywhere else, so it is printed
    # resolved; the TOML keeps the relative form for relocatability.
    printed_out = fetch_hints["out"]
    if printed_out.startswith("../") or args.data_dir:
        printed_out = _posix(Path(data_dir).resolve())
    # `--area=-58.58,...` -- the "=" form is what a leading minus needs
    # in every argument parser, including shells' own.
    area_flag = (f"--area={fetch_hints['area']}"
                 if fetch_hints["area"].startswith("-")
                 else f"--area {fetch_hints['area']}")
    # --cadence is printed whenever it is not the source's own default:
    # the emitted [fetch] table carries it, and a printed command that
    # omits it downloads a different window from the one this config was
    # written for.  At a lead not on the default grid it does not merely
    # differ -- it is refused.
    cadence_flag = (
        f"--cadence {cadence} "
        if cadence is not None and cadence != _SOURCE_CADENCE_H.get(args.source)
        else "")
    if not source_fetch_takes_a_crop_box(args.source):
        # No crop flag for a route that publishes whole objects: `gpuwm
        # fetch` refuses --area on one, so printing it made step 1 exit 2
        # for every table-driven model.  The window is stated after the
        # command instead, where it belongs -- it is a prep fact.
        area_flag = ""
    if emitted_fetch_hints is not None:
        fetch_command = ("gpuwm fetch "
                         f"--source {args.source} "
                         f"--cycle {fetch_hints['cycle']} "
                         f"--hours {fetch_hints['hours']} "
                         + (f"{area_flag} " if area_flag else "")
                         + cadence_flag
                         + (f"--forecast-start-hour {start_hour} "
                            if start_hour else "")
                         + f"--out {_printed_path(printed_out)}")
    else:
        # NOT a `gpuwm fetch` line.  This source has a runnable profile
        # and no download route, and printing a command that refuses is
        # how a reader concludes the tool is broken -- the 1.3.0 field
        # exhibit.  Say what is true, and say what the config is FOR: the
        # geometry, levels, physics and boundary cadence below are
        # complete, so a hand-staged directory of this cycle's files runs
        # the same chain every other mapped source runs.
        spacing_h = int(
            source_forcing_interval_seconds(args.source)) // 3600
        fetch_command = "\n".join((
            f"# stage {args.source}'s bytes for cycle "
            f"{fetch_hints['cycle']} yourself into",
            f"#   {_printed_path(printed_out)}",
            f"#   (`gpuwm fetch` has no download route for {args.source} "
            f"yet; the window this config",
            f"#    needs is {fetch_hints['area']}, "
            f"{fetch_hints['hours']} h at {spacing_h} h spacing)",
            f"#   `gpuwm prep --show-source {args.source}` names the exact "
            f"products this route requires."))
    # The BARE form is the next step, because it is the one that measures
    # this machine.  v1.4.0 printed only the declared form, and on a real
    # 16 GB card the two returned opposite verdicts on the same file --
    # rc 0 for the declared budget, rc 4 for the measured one -- because
    # the tier had sized against free VRAM the card does not have.  They
    # agree now (the tier is conservative against its class), and where
    # they cannot -- sizing for a card that is not in this machine -- the
    # declared form is printed beside it and labelled as such.
    check_command = f"gpuwm check {_printed_path(out)}"
    check_command_declared = (f"gpuwm check {_printed_path(out)} "
                              f"--budget-gib {budget_gib:.2f} "
                              f"--vram-gib {vram_gib:g}")
    run_command = final_step_command(
        out, source=args.source, profile=profile,
        domain_count=len(dims), data_dir=printed_out,
        case_data=case_data, exp=exp, cycle=cycle,
        forecast_start_hour=start_hour)

    # The nocturnal declaration, SPOKEN.  `render_config` writes
    # acknowledgements = [ASYMMETRIC_RADIATION_NOCTURNAL_ACK] into the
    # emitted [experiment] whenever an explicitly selected asymmetric
    # suite meets a window with local night in it, and that line disarms
    # the load-time guard at every other front door for this file -- so
    # `gpuwm check`, `gpuwm run`, `gpuwm go` and run-plan all fall
    # silent on a run PHYSICS.md classes as nocturnally invalid.  Until
    # now the only statement of that was a comment inside the emitted
    # TOML, which is not a warning: this door was measured emitting the
    # declaration with nothing on stdout or stderr, not even under
    # --explain.  warn() is the right voice twice over -- it never
    # blocks a deliberately selected validation suite, and it reaches
    # run-plan's structured warning sink, so a front end driving the
    # intent route sees the same sentence as a field.
    if declared_night is not None:
        warn(f"{out.name} is NOT NOCTURNALLY VALID: profile {profile} runs "
             f"shortwave with longwave OFF and this window includes local "
             f"night (first at {declared_night:%Y-%m-%dT%H:%M}Z).  The "
             f"emitted [experiment] declares "
             f'acknowledgements = ["{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}", '
             f'"{CONSTANT_DOWNWARD_LONGWAVE_ACK}"] -- the first because '
             f"you passed --ack, the second because this suite fabricates "
             f"its downward longwave and the file would not load without "
             f"it -- so no later command will stop this run.  Re-emit "
             f"with a full lw+sw "
             f"profile (e.g. {MORRISON_PROFILE_ID}) for a forecast.",
             why="Shortwave heats the surface by day while no longwave "
                 "scheme runs, so after sunset the surface radiates to "
                 "space with no downward longwave to balance it: skin "
                 "temperature craters, the surface saturation humidity "
                 "collapses with it, and 2 m dewpoints read far below "
                 "the airmass.  See docs/public/PHYSICS.md, 'Nocturnal "
                 "validity'.")
    for note in area_notes:
        warn(note,
             why="lat-lon source interpolation and static-tile windowing "
                 "are not pole-capable, so a box touching the pole is "
                 "not a box the pipeline can honour.")
    for note in coverage_notes:
        # Advisory, never a refusal: the DOMAIN was already bounded by
        # source coverage during fitting, so only the margined bbox --
        # a projection artifact plus the fetch margin -- overran.
        warn(note,
             why="`gpuwm fetch` validates --area against the source "
                 "grid's own lat/lon envelope and refuses a box naming "
                 "ground the grid does not carry; the clamped box still "
                 "contains the whole fitted domain, whose source "
                 "coverage was proven during fitting.")
    for note in oversized_footprint_advisory(fetch_hints["area"]):
        print(f"advisory: {note}")
    if args.source == "hrrr":
        for note in coverage_advisory(exp):
            print(f"advisory: {note}")
    # The cadence statement is LAYERED like every other advisory here:
    # the default screen gets a clause on the physics line it already
    # prints (zero lines -- the one-screen cap is a measured gate, a
    # user's next command went unfound under a 20-line wall), and
    # --explain gets the full one-line advisory with the mechanism and
    # the override path.
    chain_km = _ladder_dx_km(ratios, root_dx_m)
    shared = shared_physics(profile)
    cu_by_domain = cumulus_by_domain(
        dims, ratios, profile=profile, root_dx_m=root_dx_m,
        cumulus_requested=cumulus_requested)
    # The line describes the FILE, not the catalogue entry: the root's
    # emitted cumulus switch, which the grid may have retired.
    summary = physics_summary(profile, cu_physics=cu_by_domain[0])
    if len(dims) > 1:
        summary += (f" -- one radiation cadence: all {len(dims)} domains "
                    "run the root's radt, nests inherit it")
    print(f"physics: {summary}")
    # A changed switch is reported on the DEFAULT screen, not behind
    # --explain: the user asked for a suite by not naming one, and the
    # emission moved one of its switches.
    speak = cumulus_retired_note if explain else cumulus_retired_headline
    for note in speak(profile, root_dx_m / 1000.0,
                      cumulus_requested=cumulus_requested):
        print(f"advisory: {note}")
    if explain:
        for line in prepared_route_physics_notice(profile, args.source):
            print(line)
        for note in radiation_cadence_advisory(profile, len(dims)):
            print(f"advisory: {note}")
        for note in gray_zone_advisory(chain_km, shared):
            print(f"advisory: {note}")
        for note in cumulus_gray_zone_advisory(chain_km, cu_by_domain):
            print(f"advisory: {note}")
    else:
        # The headline says what was found; the emitted config carries
        # the whole advisory in its header comment either way, so the
        # reasoning is never only one flag away -- it is also in the
        # file, next to the settings it is about.
        for note in gray_zone_headline(chain_km, shared):
            print(f"advisory: {note}")
        for note in cumulus_gray_zone_headline(chain_km, cu_by_domain):
            print(f"advisory: {note}")

    # ---- gpuwm check, where it can honestly run ----------------------
    #
    # Run it BEFORE the next-steps block rather than after.  The block
    # is the last thing printed, on purpose: the field exhibit's whole
    # failure was a correct next command with output after it, which
    # reads as "and then this other thing happened" rather than "do
    # this".
    deferred: list[str] = []
    if case_data is None:
        if explain:
            # "fetched" only where a fetch exists; every other source's
            # bytes arrive by hand, and calling that a fetch is the same
            # false claim the omitted [fetch] table exists to avoid.
            arrival = ("fetched" if emitted_fetch_hints is not None
                       else "hand-staged")
            print(
                f"note: {arrival} {args.source.upper()} data feeds the "
                "rw-wps/gpuwm-wrf-init native initialization front door, "
                "not the [case_data] run path (the config-driven route "
                "decodes native GRIB1 = ERA5 today); this TOML's "
                "[experiment]/[[domain]]/[projection] tables are what "
                "that front door consumes.  `gpuwm check` validates the "
                "geometry and memory preflight for this file; the native "
                "front door validates its own inputs.")
        from gpuwm.cli import main as cli_main
        # The card size travels with the budget: --budget-gib alone lets
        # check re-derive a notional free larger than the whole card.
        rc = cli_main(["check", str(out), "--budget-gib",
                       f"{budget_gib:.2f}", "--vram-gib", f"{vram_gib:g}"])
        if rc != 0:
            print(f"gpuwm check FAILED (rc {rc}) on the emitted config.  "
                  "The files above were still written, so nothing is "
                  "lost: fix the gap that check names, then re-run "
                  "`gpuwm check` on the same file.", flush=True)
            return rc
        print("gpuwm check: PASS (rc 0)")
    else:
        deferred = _missing_case_inputs(out, case_data)
        if deferred:
            # Not a stanza of its own.  "gpuwm check: deferred" followed
            # by an indented inventory reads as a failure report in the
            # middle of a success, which is how the field exhibit's
            # reader met it.  What it actually means is "step 2 comes
            # after step 1", and that is where it now says it.
            if explain:
                print("gpuwm check: deferred -- declared inputs not on "
                      "disk yet:")
                for item in deferred:
                    print(f"  missing {item}")
                _print_geog_help()
        else:
            from gpuwm.cli import main as cli_main
            rc = cli_main(["check", str(out), "--budget-gib",
                           f"{budget_gib:.2f}", "--vram-gib",
                           f"{vram_gib:g}"])
            if rc != 0:
                print(f"gpuwm check FAILED (rc {rc}) on the emitted "
                      "config.  The files above were still written, so "
                      "nothing is lost: fix the gap that check names, "
                      "then re-run `gpuwm check` on the same file.",
                      flush=True)
                return rc
            print("gpuwm check: PASS (rc 0)")

    _print_next_steps(fetch_command, check_command, run_command,
                      source=args.source, deferred=bool(deferred),
                      explain=explain,
                      check_command_declared=check_command_declared)
    return 0


def _print_next_steps(fetch_command: str, check_command: str,
                      run_command: str, *, source: str, deferred: bool,
                      explain: bool,
                      check_command_declared: str | None = None) -> None:
    """The last thing the wizard prints: three commands, in order.

    Three, numbered, and nothing after them.  Everything above this
    block is a report on what was just written; this block is the only
    part that asks the reader to do something, and the field exhibit
    showed what happens when it is not visually distinct -- the correct
    ``gpuwm fetch`` line sat at line 15 of 20 and the reader concluded
    the tool did not work.

    A source whose acquisition needs something CONFIGURED first earns
    one extra line under step 1, and only when it is not configured: a
    key file nothing in this project can create otherwise surfaces as
    the provider client's own exception several commands later, with
    nothing pointing back here.  The line is the registry row's
    CREDENTIAL column, not an arm for a particular source -- a row that
    declares one gets the pointer with nothing edited here.
    """

    if not explain:
        # Before the block, never after it: the numbered steps are the
        # last thing on the screen, so the eye lands on step 1.
        print("Re-run with --explain for the sizing table, the physics "
              "notes and the full advisories.")
    print("")
    print("next:")
    # Step 1 is a command where `gpuwm fetch` has a route and a short
    # acquisition note where it does not.  Printing a `gpuwm fetch
    # --source <x>` line for a source the fetch door refuses is the exact
    # numbered-list-to-a-refusal shape this block exists to end.
    acquire = fetch_command.splitlines()
    print(f"  1. {acquire[0]}")
    for line in acquire[1:]:
        print(f"     {line}")
    for note in source_credential_notes(source):
        print(f"     {note}")
    print(f"  2. {check_command}"
          + ("   # after the fetch lands" if deferred else ""))
    if check_command_declared:
        # The bare form above measures THIS machine.  Say, once, what the
        # other form is for -- v1.4.0 printed only the declared form and
        # a reader who ran the documented bare one got the opposite exit
        # code with no explanation of why two commands disagreed.
        print(f"     # that measures THIS machine's free VRAM.  Sizing "
              f"for a card that is not in it (or no GPU here at all):")
        print(f"     #   {check_command_declared}")
    # Step 3 is a command for most configs and a short route note for
    # the ones no single command finishes; either way it is indented
    # into the numbered block rather than trailing off the end of it.
    lines = run_command.split("\n")
    print(f"  3. {lines[0]}")
    for line in lines[1:]:
        print(f"     {line}")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "domain",
        help="wizard: emit an experiment TOML for a point or polygon + GPU budget, "
             "sized by the in-process VRAM estimator")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--point", metavar="LAT,LON",
                        help="domain center in decimal degrees. |lat| 90 "
                             "is refused, and so is any center whose "
                             "FITTED domain reaches the pole -- a domain "
                             "containing one is unsupported -- so the "
                             "usable limit is set by the domain's size, "
                             "not by the center, and lands well short of "
                             "90 (near |lat| 72 on the default card and "
                             "ladder, further equatorward as either "
                             "grows). The refusal names the fitted size "
                             "when it fires; the projection is "
                             "auto-selected from |lat| (<25 Mercator, "
                             "25-60 Lambert conformal, >60 polar "
                             "stereographic) unless --projection is set. "
                             "Negative (southern/western) values work in "
                             "both forms: --point -33.87,151.21 and "
                             "--point=-33.87,151.21")
    target.add_argument(
        "--polygon", type=Path, metavar="GEOJSON",
        help="local GeoJSON Polygon, MultiPolygon, Feature, or "
             "FeatureCollection; the minimum antimeridian-aware bounds "
             "supply the center and every emitted level is fitted around "
             "the geometry")
    parser.add_argument(
        "--buffer-km", default=None, metavar="KM[,KM...]",
        help="with --polygon, nonnegative geometry buffer in kilometres; "
             "one value applies to every domain, or supply exactly one "
             "outer-to-inner value per level. With --ladder auto, a "
             "multi-value list selects the preset of that depth "
             "(default: zero)")
    parser.add_argument("--projection", default="auto",
                        choices=("auto", "lambert", "mercator", "polar"),
                        help="map projection override (default: auto by "
                             "center latitude; all three are oracle-gated "
                             "against WRF v4.6.1 module_llxy)")
    parser.add_argument("--name", default=None,
                        help="experiment name (default derived from the "
                             "center)")
    parser.add_argument("--card", choices=sorted(CARD_VRAM_GIB),
                        default=None,
                        help="GPU tier; sets the VRAM budget with no "
                             "local probe.  With neither --card nor "
                             "--vram-gib the wizard MEASURES the local "
                             "card's capacity (short-lived probe, "
                             "suppressed by GPUWM_NO_LOCAL_GPU) and "
                             "refuses, naming both flags, when there is "
                             "nothing to measure")
    parser.add_argument("--vram-gib", type=float, default=None,
                        metavar="N",
                        help="total VRAM in GiB (alternative to --card)")
    parser.add_argument("--ladder", default=None,
                        choices=(*LADDER_RATIOS, "auto"),
                        help="preset nest dx chain in km (default: 12 -- "
                             "one 12 km domain, the shape `gpuwm go` runs "
                             "end to end, same as the interactive "
                             "session).  Nest trees are explicit opt-in: "
                             "a deeper preset, `auto` (the deepest preset "
                             "that fits the card), or --root-dx / --chain "
                             "for anything else; their closing block "
                             "names the tree runner they route to")
    parser.add_argument("--physics-profile", default=None,
                        choices=WIZARD_PHYSICS_PROFILES,
                        help="shipped physics suite to emit; taken verbatim "
                             "from the registry the prepared-forecast "
                             "runner validates against, so the emitted "
                             "config passes its guard as written.  Read "
                             "the names: the *-no-radiation-* and "
                             "*-validation-* profiles run reduced physics "
                             "with longwave OFF and are NOT nocturnally "
                             "valid -- selecting one for a window that "
                             "includes local night is REFUSED unless you "
                             "declare it yourself with --ack.  "
                             + _profile_help_route_note()
                             + _profile_help_default_note())
    parser.add_argument(
        "--ack", action="append", default=[], metavar="ID",
        help="declare a governed experiment, written verbatim into the "
             "emitted [experiment].acknowledgements.  Repeatable.  This "
             "door used to write the nocturnal declaration for you, which "
             "silenced the load guard at check/run/go/run-plan and both "
             "prepared runners for the life of the file; it no longer "
             "does, and refuses instead.  The id it accepts is "
             + ASYMMETRIC_RADIATION_NOCTURNAL_ACK
             + ": a longwave-OFF suite over a window that includes local "
             "night, which you are running deliberately as a daytime "
             "validation experiment")
    parser.add_argument("--root-dx", type=float, default=None,
                        metavar="KM",
                        help="custom root grid spacing in km "
                             f"[{MIN_ROOT_DX_KM:g}, {MAX_ROOT_DX_KM:g}]; "
                             "use with --chain instead of --ladder")
    parser.add_argument("--chain", default=None, metavar="R1,R2,...",
                        help="custom nest refinement ratios, integers in "
                             f"[{MIN_CHAIN_RATIO}, {MAX_CHAIN_RATIO}] "
                             "(e.g. --root-dx 3 --chain 4 for 3 km -> "
                             "750 m); omit for a single domain at "
                             "--root-dx.  Sized by the same estimator fit "
                             "loop as the presets")
    parser.add_argument(
        "--history-interval", type=float, default=None, metavar="SECONDS",
        help="how often the ROOT domain writes a wrfout, in seconds "
             f"(default {DEFAULT_ROOT_HISTORY_INTERVAL_S:g}).  Must be a "
             "whole number of seconds and a whole number of that "
             "domain's time steps -- the loader checks both against the "
             "exact rational dt and refuses the emitted file otherwise, "
             "before it is written")
    parser.add_argument(
        "--nest-history-interval", type=float, default=None,
        metavar="SECONDS",
        help="the same, for every NESTED domain (default "
             f"{DEFAULT_NEST_HISTORY_INTERVAL_S:g}).  Nests write more "
             "often than the root by default because resolving what the "
             "root cannot, over a shorter window, is the point of "
             "running one.  Ignored for a single-domain ladder")
    parser.add_argument("--hours", type=int, default=6, metavar="N",
                        help="forecast length (run_seconds = N*3600)")
    parser.add_argument(
        "--source", default=DEFAULT_WIZARD_SOURCE, metavar="SOURCE",
        # NO `choices=`, deliberately.  This used to be
        # choices=("gfs", "hrrr", "era5") while the product shipped
        # sixteen runnable sources, so argparse answered `--source rap`
        # with "invalid choice" -- a refusal that says nothing about RAP.
        # The registry answers instead (`resolve_source`), which lets an
        # alias resolve, a registered-but-unrunnable row explain itself,
        # and a new row reach this door with no edit here.
        help="forcing source: any registered source id or alias -- "
             + ", ".join(wizard_planable_source_ids())
             + " today (`gpuwm prep --list-sources` lists the whole "
               "registry).  It "
               "sets the boundary cadence written into the companion "
               "namelist.wps, bounds the domain by the source's own grid "
               "where that grid is regional, and (era5) declares "
               "[case_data].  A source `gpuwm fetch` cannot download yet "
               "emits the same geometry with the acquisition step named "
               "instead of a [fetch] table")
    parser.add_argument("--cycle", required=True,
                        metavar="YYYY-MM-DDTHH|latest",
                        help="the forcing CYCLE (UTC), which is the run's "
                             "start time unless --forecast-start-hour "
                             "moves it; 'latest' probes the public mirrors "
                             "for the newest complete gfs/hrrr cycle "
                             "covering the whole window and prints what it "
                             "picked (needs network; era5 must name an "
                             "explicit time)")
    parser.add_argument("--forecast-start-hour", type=int, default=None,
                        metavar="K",
                        help="gfs/gdas/hrrr: initialize the run from the "
                             "cycle's f{K} FORECAST lead instead of its "
                             "analysis, so start_time = cycle + K h and "
                             "the boundaries come from f{K+i}.  This is "
                             "how a window deep in a forecast (say "
                             "f174..f240) is reached without integrating "
                             "from f000.  The initial condition is then "
                             "itself a K-hour forecast, and every receipt "
                             "says so")
    parser.add_argument("--out", type=Path, required=True, metavar="TOML",
                        help="emitted experiment TOML path")
    parser.add_argument("--data-dir", default=None, metavar="DIR",
                        help="where fetched forcing lives/will live "
                             "(default data/<name>)")
    parser.add_argument("--forcing", nargs="+", default=None,
                        metavar="GRIB",
                        help="era5: explicit forcing GRIB path(s) already "
                             "on disk (default <data-dir>/"
                             "era5-combined.grib)")
    parser.add_argument("--vtable", type=Path, default=None,
                        help="era5: Vtable override (default: the "
                             "packaged Vtable.ERA5_CDO, copied beside "
                             "the TOML)")
    parser.add_argument("--geog-root", type=Path, default=None,
                        metavar="DIR",
                        help="staged WPS_GEOG tree (default "
                             "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG)")
    parser.set_defaults(func=domain_main)
    return parser


__all__ = [
    "CARD_VRAM_GIB", "CUMULUS_CONVECTION_PERMITTING_DX_KM",
    "CUMULUS_GRAY_ZONE_TOP_DX_KM",
    "DEFAULT_LADDER", "DEFAULT_WIZARD_SOURCE",
    "DomainFitError", "GEOG_DATASETS", "GRAY_ZONE_DX_KM",
    "LADDER_RATIOS", "MAX_FETCH_ABS_LAT", "POLE_CLEARANCE_DEG",
    "ROOT_DX_M", "ROOT_TIME_STEP_S", "TROPICAL_ROOT_TIME_STEP_S",
    "convection_permitting",
    "cumulus_by_domain", "cumulus_gray_zone_advisory",
    "cumulus_gray_zone_headline", "cumulus_retired_headline",
    "cumulus_retired_note", "declared_nocturnal_night",
    "domain_main", "experiment_from_text", "final_step_command",
    "fit_ladder", "fit_polygon_ladder", "gray_zone_advisory",
    "load_polygon_footprint", "max_fetch_abs_lat",
    "oversized_footprint_advisory", "parse_chain",
    "parse_custom_ladder", "parse_level_buffers", "pole_clearance_deg",
    "polygon_ladder_dims", "radiation_cadence_advisory",
    "radt_ladder_minutes", "register_cli", "render_config",
    "render_wps_namelist", "root_time_step_s", "seconds_per_km",
    "verify_polygon_containment", "vram_reserve_gib",
    "CARD_UNAVAILABLE_VRAM_GIB", "CARD_UNAVAILABLE_VRAM_FRACTION",
    "FIT_HEADROOM_FRACTION",
    "FIT_HEADROOM_MIN_BYTES", "card_assumed_free_gib",
    "fit_headroom_bytes", "sizing_budget_bytes",
]
