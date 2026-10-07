"""EXPERIMENTAL: the real-radar ``assimilate()`` for the cycle driver.

Every piece of a real radar analysis already ships -- the NEXRAD front
door and superob writer produce ``gpuwm-obs.radar-grid.v1`` files, the
:mod:`gpuwm.da.obs_radar` adapter turns one into filter batches, the
LETKF returns increments, and :mod:`gpuwm.ensemble.cycle` owns the seam
those increments go through.  What did not ship is the callable the seam
takes: something that reads REAL member checkpoints, evaluates H(x) on
them, and hands the driver increments in the checkpoint's own field
vocabulary.  The synthetic gate (:mod:`gpuwm.da.synthetic_cycle`) is not
that callable -- its members are mass-point dicts a stand-in dycore wrote,
while a real ``gpuwmrst`` checkpoint carries ARW-staggered winds -- and
the one real-data LETKF cycle run to date had to be assembled by hand on
the node because of exactly that gap.  This module is the missing brain.

Three decisions here shape the module:

**The filter analyses mass-point, grid-relative winds; the checkpoint
receives face-point increments.**  The LETKF requires every analysis
field on one ``(R, nz, ny, nx)`` grid, and a checkpoint's ``u`` is
``(nz, ny, nx+1)``.  So winds are destaggered to mass points for the
prior, and the returned wind increments are linearly interpolated back
onto the faces (interior face = mean of its two mass neighbours, rim
face = its one neighbour -- constant fields survive the round trip
exactly).  The analysed pair stays GRID-relative: the filter relates
state to observations only through ensemble covariances with H(x), so
the frame of the analysed field is free, and choosing the state's own
frame means the increments add to ``state/u`` and ``state/v`` with no
rotation on the way back.  The rotation lives inside H(x), where the
beam is.

**H(x) uses the grid's own rotation, not the checkpoint's.**
``SINALPHA``/``COSALPHA`` are setup arrays and a restart deliberately
does not carry them.  The observation file is bound to the caller's
:class:`~gpuwm.obs.target_grid.TargetGrid`, whose projection is verified
against the wrfout's own XLAT/XLONG -- so ``grid.projection.rotation``
is the same authority the observations were placed with, and the one
this module uses to rotate member winds into the beam's earth frame.

**Radial velocity projects onto the file's beam vectors.**  Straight
from :mod:`gpuwm.da.obs_radar`: the superob writer shipped the
normalised per-cell beam direction precisely so the assimilating side
does not re-derive geometry.  The vertical component is ``w - vt`` when
a reflectivity provider is configured (Sun and Crook fall speed from the
scheme's own dBZ, surface pressure taken as the lowest mass level's full
pressure -- within the bottom half-layer of the true surface value,
versus the +4-15% systematic error of the ``P0`` substitution the
operator refuses), and plain ``w`` under the explicit
``fall_speed="none"`` simplification.

Reflectivity H(x) needs the scheme and the base state.  A checkpoint
carries ``thp`` but not ``thb`` (base state is SETUP, rebuilt at
restore), so temperature is not derivable from the checkpoint alone.
:func:`scheme_reflectivity_provider` takes the run config and the base
theta explicitly and refuses nothing silently; without a provider,
``reflectivity=True`` and ``fall_speed="reflectivity"`` are refusals
that name what is missing.

Nothing here is wired into a default route.  EXPERIMENTAL.
"""

from __future__ import annotations

import gc
import time
import types
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


_ANALYSIS_EXECUTION = ContextVar("analysis_execution", default=None)


@contextmanager
def analysis_execution_options(*, scratch_budget=None, progress=None):
    """Execution scheduling for one calling context, with automatic restoration.

    The scope does not change input configuration or the analysis method.
    Direct solver calls and concurrent contexts retain their own settings.
    """
    token = _ANALYSIS_EXECUTION.set((scratch_budget, progress))
    try:
        yield
    finally:
        _ANALYSIS_EXECUTION.reset(token)


def _execution_settings(memory_budget_mib, progress):
    options = _ANALYSIS_EXECUTION.get()
    if options is None:
        return memory_budget_mib, progress, None
    resolve, relay = options
    budget, receipt = (resolve(memory_budget_mib) if resolve else
                       (memory_budget_mib, None))
    return budget, progress if progress is not None else relay, receipt


from gpuwm.da.letkf import (RELAXATION_MODES, GriddedObs, LetkfConfig,
                            LetkfDiagnostics, Localization, analyze)
from gpuwm.da.moments import (MOMENT_POLICIES, DEFAULT_MOMENT_POLICY,
                              validate_analysis_fields)
from gpuwm.da.obs_goes import goes_grid_to_gridded_obs
from gpuwm.da.obs_radar import (Z_SOURCES, beam_unit_vectors,
                                observed_radial_velocity,
                                letkf_grid_geometry, radar_grid_to_gridded_obs,
                                read_document, simulated_radial_velocity)
from gpuwm.da.obsop import (CLEAR_AIR_FLOOR_DBZ, clear_air_floor_dbz,
                            destagger_u, destagger_v,
                            destagger_w, earth_relative_winds,
                            precipitating_activity_mask,
                            reflectivity_fall_speed)
from gpuwm.da.velocity_dispersion import (
    DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO, DEFAULT_VELOCITY_DISPERSION_RATIO,
    DispersionGateError, check_ratio, column_plan, velocity_dispersion,
    withhold)
from gpuwm.da.letkf_device import ColumnWithhold
from gpuwm.da.reflectivity_echo import (
    DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS, DEFAULT_REFLECTIVITY_FLOOR_DBZ,
    ReflectivityEchoError, check_floor, check_sigmas,
    condition_reflectivity_batch)
from gpuwm.da import field_rules as _fr
from gpuwm.da import radar_classes as _rc
from gpuwm.da.positivity import (BOUNDING_POLICIES,
                                 NON_NEGATIVE_FIELDS, POLICIES,
                                 DevicePositivity, apply_positivity,
                                 constrained_fields, verify_non_negative)

#: Provenance schema for the analysis receipt this module emits.
METHOD_SCHEMA = "gpuwm-da.radar-assimilation.v1"

#: Checkpoint fields that live on staggered grids.  Everything else in the
#: restart prognostic contract is mass-shaped and passes through unchanged.
WIND_FIELDS = ("u", "v", "w")

#: Prefix of the state arrays inside a checkpoint npz.
CHECKPOINT_STATE_PREFIX = "state/"

#: The fall-speed policies this module will express.  "reflectivity" is Sun
#: and Crook from the scheme's own dBZ; "none" is the documented
#: air-motion-only simplification.  There is no default-by-omission: the
#: config names one or the other.
FALL_SPEED_POLICIES = ("none", "reflectivity")

#: Where the batched LETKF may be asked to run.  "auto" is a REQUEST, not
#: a device: it resolves to one of the other two before the solve, and the
#: resolution is recorded.  See :func:`resolve_solve_device`.
SOLVE_DEVICES = ("auto", "host", "cuda")

#: The radar precipitation analysis after the solve
#: (:mod:`gpuwm.da.hydrometeor_analysis`).  "off" leaves the analysis exactly
#: as the filter, positivity and the saturation bound make it; "clear" is
#: what HRRRDAS applies to every member (gsdcloudanalysis.F90:910-927,
#: l_precip_clear_only at parm/hrrrdas/hrrrdas_gsiparm.anl:170).  The module's
#: "trim-build" rule (gsdcloudanalysis.F90:928-1049) is a deterministic
#: analysis's rule and is refused here: every analysis this function makes
#: is an ensemble member's (see :data:`PRECIP_ANALYSIS_MEMBER_REFUSAL`).
PRECIP_ANALYSIS_MODES = ("off", "clear")

#: Why ``precip_analysis="trim-build"`` is refused on the ensemble.  NOAA
#: runs the trim and build rule on the deterministic analysis only and gives
#: every HRRRDAS member the clear step (gsdcloudanalysis.F90:910-927 under
#: l_precip_clear_only, parm/hrrrdas/hrrrdas_gsiparm.anl:170).  This seam
#: analyses ensemble members and nothing else (the cycle has no analysed
#: deterministic member, tools/da_cycle_prepared.py module header).  Run on
#: every member, the rule scales each member's column so its largest rain
#: plus snow equals one retrieval and writes that same retrieval at the
#: strongest echo level of every member that had less: the members' rain
#: under echo converges on one value, and the filter's next cycle has no
#: hydrometeor spread there to weigh the radar against.
PRECIP_ANALYSIS_MEMBER_REFUSAL = (
    "precip_analysis='trim-build' is refused for an ensemble analysis. The "
    "trim and build rule (NOAA's gsdcloudanalysis.F90:928-1049) is a "
    "deterministic analysis's rule: NOAA gives every ensemble member the "
    "clear step only (l_precip_clear_only, hrrrdas_gsiparm.anl:170), and "
    "this analysis has no deterministic member. On every member it scales "
    "each column to one retrieval maximum and inserts one retrieval at the "
    "strongest echo, so the members' rain under echo converges on one value "
    "and the next cycle's filter has no spread there (measured on a planted "
    "ensemble: tests/test_da_hydrometeor_analysis.py, "
    "test_trim_build_on_every_member_collapses_rain_spread). Use 'clear'; "
    "the rule stays in gpuwm.da.hydrometeor_analysis for a deterministic "
    "member")


class RadarAssimilationError(ValueError):
    """The observations, the checkpoints and the config cannot be
    reconciled.  Never a warning."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def analysis_sources(cfg, *, extra_batches: int | None) -> tuple[str, ...]:
    """Which observation families one analysis actually carries.

    ONE FUNCTION, BOTH DOORS.  The config's own refusal and the analysis
    call both ask this, so a configuration cannot be admitted at one door
    and called empty at the other.  ``extra_batches`` is the COUNT of
    attributed extra batches when it is known (the analysis call) and
    ``None`` when it is not yet (construction), where the config's
    ``extra_observations`` statement stands in: ``None`` there means "the
    batch list will say", which is admitted, and ``False`` means "there
    will be none", which is a fact the construction can already act on.
    """

    names = [name for name, enabled in (
        ("velocity", cfg.velocity), ("reflectivity", cfg.reflectivity),
        ("clear_air", cfg.clear_air), ("cwp", cfg.cwp)) if enabled]
    declared = getattr(cfg, "extra_observations", None)
    if extra_batches is None:
        if declared is not False:
            names.append("extra_observations")
    elif int(extra_batches) > 0:
        names.append("extra_observations")
    return tuple(names)


@dataclass(frozen=True)
class RadarAssimilationConfig:
    """Everything one radar analysis needs that is not data.

    ``analysis_fields`` are CHECKPOINT spellings (``u``/``v``/``w`` are the
    staggered prognostics; the mass-point projection is internal).  A field
    with no ensemble spread is the filter's refusal, not this module's:
    naming ``w`` against an initial-condition perturbation that does not
    perturb ``w`` is a configuration error the filter reports precisely.

    ``rtps_alpha`` is required with no default, for the filter's own
    reason: 0.0 silently disables relaxation and the ensemble collapses
    several cycles later, so the choice must be stated where it can be
    seen.

    ``fall_speed`` and ``reflectivity`` both need a reflectivity provider
    (see :func:`scheme_reflectivity_provider`); configuring either without
    one is refused at call time, naming the gap.
    """

    localization: Localization
    rtps_alpha: float
    analysis_fields: tuple[str, ...] = ("u", "v")
    prior_inflation: float = 1.0
    #: Named Lane 8 replay policy, retained by dataclasses.replace in da-tune.
    spread_repair: str = "off"
    spread_repair_z_threshold: float = 25.0
    spread_repair_seed: int = 0
    spread_repair_adaptive: bool = False
    #: Assimilate the per-radar radial-velocity batches.
    velocity: bool = True
    #: Assimilate the merged reflectivity batch.  Needs a provider.
    reflectivity: bool = False
    #: Which reflectivity reduction to difference against; see
    #: :data:`gpuwm.da.obs_radar.Z_SOURCES`: the in-cell linear-Z mean by
    #: default, the maximum as an arm.
    z_source: str = "z_mean"
    #: Restrict velocity batches to these radar ids (None = all in file).
    radars: tuple[str, ...] | None = None
    #: "reflectivity" (Sun & Crook from the provider's dBZ) or "none"
    #: (air motion only, an explicit simplification).
    fall_speed: str = "none"
    #: Per-type localization overrides; None falls back to ``localization``.
    velocity_localization: Localization | None = None
    reflectivity_localization: Localization | None = None
    #: Keep at most one velocity observation per ``s x s`` horizontal block
    #: per level per radar (the cell with the most contributing gates; ties
    #: to the smaller error, then the first index).  1 keeps everything.
    #: The shakedown case that motivated this saw up to 513 observations
    #: inside one localisation lens against R-1 = 9 ensemble degrees of
    #: freedom -- a rank-9 fit to 513 correlated numbers, which is how a
    #: 17 m/s increment gets extracted from a 1.4 m/s-spread ensemble.
    velocity_thinning_cells: int = 1
    #: The same rank argument, applied to the merged reflectivity batch.
    #: A dBZ field is smooth on the scale of a storm, so a localisation
    #: lens holds many more reflectivity observations than the ensemble
    #: has degrees of freedom, and they are far more correlated with each
    #: other than a velocity superob pair is.
    reflectivity_thinning_cells: int = 1
    #: Multiplies the file's reflectivity error standard deviations, for
    #: the same reason the velocity knob exists: representativeness error
    #: the ensemble cannot carry belongs in sigma_o, not in an increment.
    reflectivity_error_inflation: float = 1.0
    #: Assimilate clear-air ("zero") observations: cells the radar
    #: measured and found free of significant echo.  Needs the same
    #: reflectivity provider ``reflectivity`` needs, because a zero is
    #: differenced against the same H(x).
    #:
    #: Off by default and separate from ``reflectivity`` on purpose.  The
    #: two do opposite things -- echo places and maintains storms, zeroes
    #: erase them -- and they fail in opposite directions, so being able
    #: to run one without the other is what makes an ablation possible.
    #:
    #: **On the zero-variance background.**  The known limitation of this
    #: lane is that the ensemble cannot invent echo it never made:
    #: :mod:`gpuwm.da.perturb` applies its species factor only where the
    #: background pair is jointly active (:1346-1355), so where every
    #: member is clear the prior spread is zero and no observation can
    #: move the state.  That limit binds the ECHO half of reflectivity DA
    #: -- radar sees a storm, model has none, filter cannot create one.
    #:
    #: It does **not** bind clear-air zeroes, and no additive mechanism is
    #: introduced for them here.  A zero only carries information where
    #: H(x) exceeds the clear-air floor, which is exactly where the model
    #: HAS condensate -- and where the model has condensate the species
    #: perturbation has been applied and the prior spread is non-zero.
    #: Where model and radar are both clear the innovation is identically
    #: zero and a zero-variance background is the correct and harmless
    #: answer.  So suppression works with the machinery that already
    #: exists; it is initiation that needs something this filter is not.
    #:
    #: The mitigation for the echo half stays where the lane already put
    #: it -- ``gpuwm.da.perturb``'s ``SpeciesPerturbation``, configured by
    #: the caller's cycle policy, off unless asked for -- rather than
    #: being baked in here.  :mod:`gpuwm.da.letkf` (:119-131) assigns
    #: ensemble construction to the caller on purpose, and additive
    #: inflation, the one family that restores rank rather than
    #: amplitude, is documented there as absent.
    clear_air: bool = False
    #: Localization for the clear-air batch; None falls back to
    #: ``localization``.  A tighter radius than echo is the usual choice:
    #: a zero is evidence about the cell that was measured, and spreading
    #: it far means one clear gate erasing condensate it never sampled.
    clear_air_localization: Localization | None = None
    #: Keep at most one clear-air observation per ``s x s`` horizontal
    #: block per level.  Defaults to 4 rather than 1 because clear air is
    #: the overwhelming majority of any volume and is far smoother than
    #: echo -- every cell in a lens carries nearly the same number, so
    #: they add rank-1 information and rank-starvation cost in proportion
    #: to their count.  See the rank note on
    #: ``reflectivity_thinning_cells``.
    clear_air_thinning_cells: int = 4
    #: Multiplies the file's clear-air error standard deviations.  The
    #: file's own ``clear_air_error_dbz`` is already larger than the echo
    #: error; this is the cycle-side knob on top of it.
    clear_air_error_inflation: float = 1.0
    #: The dBZ value a clear-air observation carries.  None derives it
    #: from ``mp_physics`` via
    #: :func:`gpuwm.da.obsop.clear_air_floor_dbz`, which is the sanctioned
    #: route.  An explicit number is honoured and recorded, and is refused
    #: unless it is finite AND equal to the active scheme's floor when
    #: that floor has been read: the two disagreeing is the silent
    #: innovation this pair exists to prevent, not a setting.
    clear_air_value_dbz: float | None = None
    #: Multiplies the file's velocity error standard deviations.  The
    #: defensible setting is diagnosed from the innovation statistics this
    #: module itself reports: when mean(d^2) far exceeds
    #: spread^2 + sigma_o^2, the surplus is background error the ensemble
    #: does not represent (storm displacement, unrepresented scales), and
    #: absorbing it into sigma_o is the standard single-knob correction.
    velocity_error_inflation: float = 1.0
    #: Assimilate the ``gpuwm-obs.goes-grid.v1`` cloud-water-path batch
    #: beside the radar ones.  Needs a provider; see
    #: :func:`gpuwm.da.obsop_cwp.checkpoint_cwp_provider`.
    cwp: bool = False
    #: Whether this analysis carries attributed extra batches beside (or
    #: instead of) the radar and satellite ones.  ``None``, the DEFAULT,
    #: means the config does not claim to know: the batch list handed to
    #: :func:`assimilate_radar_grid` is the fact, and that is where the
    #: emptiness of an analysis is decided.  ``True`` and ``False`` are
    #: statements, kept because a caller that KNOWS it will pass none
    #: gets its "this would assimilate nothing" refusal at construction,
    #: which is earlier.
    #:
    #: It is not an admission ticket.  A flag defaulting False meant a
    #: caller with real attributed point observations and no radar was
    #: refused at construction for not having declared them, while a
    #: caller that declared them and then passed an empty list was
    #: admitted; both are the declaration disagreeing with the data.
    #: :func:`analysis_sources` is the one function both doors call.
    extra_observations: bool | None = None
    #: Per-type localization for CWP.  This one is not decoration: CWP is a
    #: column integral carried at one level (see :mod:`gpuwm.da.obs_goes`),
    #: so its vertical radius is what decides whether the observation acts
    #: on the column it integrated or on a slab.  Falls back to
    #: ``localization``, which is tuned for radar and is very probably too
    #: shallow for this.
    cwp_localization: Localization | None = None
    #: The same rank argument as the reflectivity knob, applied to CWP.
    #: A satellite cloud field is smoother than a dBZ field and the pixels
    #: are 2 km, so a localisation lens holds far more CWP observations
    #: than the ensemble has degrees of freedom.
    cwp_thinning_cells: int = 1
    #: Multiplies the file's CWP error standard deviations.  Those errors
    #: are UNCALIBRATED by construction (there is no measured CWP error
    #: covariance for this system), so this knob is inflating a number that
    #: was already a stated assumption -- which is a reason to record it,
    #: not a reason to avoid it.
    cwp_error_inflation: float = 1.0
    #: Which posterior relaxation ``rtps_alpha`` drives, threaded to
    #: :class:`gpuwm.da.letkf.LetkfConfig` unchanged.
    relaxation: str = "rtps"
    #: What to do where ``prior + increment`` would be negative in a
    #: physically non-negative field.  ``None`` means "not stated", which
    #: is refused as soon as any constrained field is analysed: the
    #: filter deliberately owns no positivity policy (see
    #: :mod:`gpuwm.da.positivity`), so somebody has to, and the config is
    #: where a choice is visible.  When nothing constrained is analysed
    #: ``None`` is correct and is recorded as not applicable.
    positivity_policy: str | None = None
    #: The moment policy the analysed field set is validated against.
    #: ``full-moment`` refuses an update that moves a species' mass while
    #: leaving the paired number moment the background carries -- the
    #: defect that made :mod:`gpuwm.da.moments` necessary -- BEFORE the
    #: solve, rather than leaving it to the applier after it.
    moment_policy: str = DEFAULT_MOMENT_POLICY
    #: The scheme whose moment structure the field set is checked against.
    #: ``None`` detects the pairs from the checkpoint's own spellings,
    #: which is weaker but never wrong about a state it can see, and stays
    #: accepted for every arm: a cycle that STATES its scheme additionally
    #: has that scheme's radar_da route checked here (an unrouted or
    #: native-Z scheme is refused at configuration instead of per member
    #: inside the first analysis), and a cycle that does not is answered by
    #: the caller's own reflectivity provider, as it always was.
    mp_physics: int | None = None
    #: Where the batched LETKF runs.  "host" solves on numpy; "cuda"
    #: moves the prior and the observation batches to CuPy for the batched
    #: LETKF and brings the increments back; "auto", the DEFAULT, takes
    #: cuda when this process can reach a device and host when it cannot,
    #: and records which it took.
    #:
    #: auto is the default because the alternative is a workaround.  The
    #: analysis is the majority of a DA cycle's wall clock, the device
    #: path has been reachable the whole time, and a bare run that gets
    #: the slow one is a run that shows the defect at its defaults.  A
    #: host-only box still works, unchanged, and says so in the receipt
    #: rather than silently.  Naming "host" or "cuda" explicitly is
    #: honoured verbatim and never second-guessed -- a run pinned to host
    #: for reproducibility must stay on host even beside a working card,
    #: and a run that asked for cuda must fail loudly if the card is gone
    #: rather than quietly becoming a different, slower experiment.
    solve_device: str = "auto"
    #: Threaded to :class:`gpuwm.da.letkf.LetkfConfig` unchanged.
    solve_dtype: str = "float64"
    #: Also threaded unchanged; see
    #: :data:`gpuwm.da.letkf.EIGENSOLVER_MODES`.  The default needs no
    #: linear-algebra library on either device, which is the whole reason
    #: it is a setting rather than a fact.
    eigensolver: str = "auto"
    memory_budget_mib: float = 512.0
    chunk_points: int | None = None
    #: The cards the device analysis spreads its chunks over: "all", the
    #: DEFAULT, every card this process can see, or a tuple of device
    #: ordinals.  Every point's increment is one warp's fixed-order
    #: arithmetic on its own inputs, so the bytes are the same on one card
    #: or eight (gpuwm.da.letkf_device); on the 9 km CONUS case one card
    #: ran all 51 chunks while seven sat idle.
    solve_cards: object = "all"
    #: The radial-velocity dispersion gate
    #: (:mod:`gpuwm.da.velocity_dispersion`): a Vr batch is withheld from
    #: theta and vapour in the columns where its innovation variance
    #: exceeds this many times its ensemble plus observation error
    #: variance, inside a batch gated by the batch ratio below.  ``None``
    #: switches the gate off (Vr updates theta and vapour everywhere); the
    #: ratios are recorded in every analysis's receipt either way.
    velocity_dispersion_ratio: float | None = DEFAULT_VELOCITY_DISPERSION_RATIO
    #: The gate's batch condition: a Vr batch is gated at all only when its
    #: ratio over all its gates exceeds this.  ``None`` gates every batch on
    #: its columns alone.
    velocity_dispersion_batch_ratio: float | None = (
        DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO)
    #: The radar half of the GSD cloud analysis, run on the device after the
    #: solve and positivity and before the saturation bound, as the HRRR
    #: orders it (its cloud analysis runs after the last outer loop).  One of
    #: :data:`PRECIP_ANALYSIS_MODES`.  OFF by default today: it is opt-in
    #: until a rain-scored real case shows what it does to rain after the
    #: analysis.  The water it removes is a declared sink, as in the HRRR.
    #: "clear" fits any scheme and is the step NOAA gives every ensemble
    #: member; "trim-build" is refused here
    #: (:data:`PRECIP_ANALYSIS_MEMBER_REFUSAL`).
    precip_analysis: str = "off"
    #: Which observations may move which fields (gpuwm.da.field_rules,
    #: design D1): "design" keeps reflectivity and clear air out of the
    #: winds, adds only ``z_thermo_weight`` of their thermodynamic increment
    #: inside observed echo (|dqv| capped at ``z_qv_cap``), lets only them
    #: move hydrometeor mass (none when ``z_hydrometeors`` is off), and
    #: analyses no number or aerosol; "joint" is every batch on every field.
    field_rules: str = _fr.DEFAULT_RULES
    z_thermo_weight: float = _fr.DEFAULT_Z_THERMO_WEIGHT
    z_hydrometeors: bool = _fr.DEFAULT_Z_HYDROMETEORS
    z_qv_cap: float | None = _fr.DEFAULT_Z_QV_CAP
    #: Rain and ice number after the update: the scheme's own relation from
    #: analysed mass ("scheme"), the background size kept ("preserve-size"),
    #: or left alone ("off").
    number_rediagnosis: str = _fr.DEFAULT_NUMBER_MODE
    #: Common echo floor, dBZ, applied to the reflectivity batch's
    #: observations AND its H(x) before differencing
    #: (:mod:`gpuwm.da.reflectivity_echo`).  ``None`` switches it off and
    #: differences raw returns from -15 dBZ against the scheme's -35 dBZ
    #: H(x) floor, which on the CONUS first-light analysis doubled the echo
    #: area an hour later.  The clear-air batch is not affected.
    reflectivity_floor_dbz: float | None = DEFAULT_REFLECTIVITY_FLOOR_DBZ
    #: Outlier tempering on the reflectivity batch, in standard deviations
    #: of sqrt(spread^2 + sigma_o^2): an observation beyond it has its error
    #: raised just enough to sit at it, so an innovation the ensemble cannot
    #: reach is not extrapolated past the members.  ``None`` switches it
    #: off.  Observations inside it are untouched.
    reflectivity_outlier_sigmas: float | None = (
        DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS)
    #: Radar observation classes (gpuwm.da.radar_classes, design A1): the
    #: clear-air floor shared by observation and H(x) (None = the legacy
    #: unclassified batches), whether weak echo between it and the 15 dBZ
    #: echo floor is a dead band, the per-class errors (None = the file's),
    #: the level strides, the 11 km top and Huber weighting.
    reflectivity_clear_floor_dbz: float | None = _rc.DEFAULT_CLEAR_FLOOR_DBZ
    reflectivity_dead_band: bool = _rc.DEFAULT_DEAD_BAND
    reflectivity_error_dbz: float | None = _rc.DEFAULT_ECHO_ERROR_DBZ
    clear_air_error_dbz: float | None = _rc.DEFAULT_CLEAR_ERROR_DBZ
    reflectivity_level_stride: int = _rc.DEFAULT_ECHO_LEVEL_STRIDE
    clear_air_level_stride: int = _rc.DEFAULT_CLEAR_LEVEL_STRIDE
    radar_top_pa: float | None = _rc.DEFAULT_TOP_PA
    reflectivity_huber_c: float | None = None
    #: Which analysed fields each KIND of observation may move
    #: (gpuwm.da.field_rules): cloud water path never moves vapour or heat,
    #: a surface report never moves a hydrometeor moment.  On by default
    #: (audit S9); False lets every batch update every analysed field.
    kind_field_rules: bool = True

    def __post_init__(self) -> None:
        if self.spread_repair != "off":
            from gpuwm.da.spread_repair import SpreadRepairConfig
            SpreadRepairConfig(policy=self.spread_repair,
                observed_threshold_dbz=self.spread_repair_z_threshold,
                adaptive=self.spread_repair_adaptive)
            if not self.reflectivity:
                raise RadarAssimilationError("spread repair needs precipitation reflectivity to distinguish observed clear air")
        elif self.spread_repair_adaptive:
            raise RadarAssimilationError("adaptive spread needs an enabled spread-repair policy to carry its inflation state")
        try:
            _fr.check_settings(self.field_rules, self.z_thermo_weight,
                               self.z_qv_cap, self.number_rediagnosis)
        except _fr.FieldRuleError as exc:
            raise RadarAssimilationError(str(exc)) from None
        try:
            check_floor(self.reflectivity_floor_dbz)
            check_sigmas(self.reflectivity_outlier_sigmas)
            _rc.check_settings(
                clear_floor=self.reflectivity_clear_floor_dbz,
                echo_error=self.reflectivity_error_dbz,
                clear_error=self.clear_air_error_dbz,
                echo_stride=self.reflectivity_level_stride,
                clear_stride=self.clear_air_level_stride,
                top_pa=self.radar_top_pa, huber_c=self.reflectivity_huber_c)
        except (ReflectivityEchoError, _rc.RadarClassError) as exc:
            raise RadarAssimilationError(str(exc)) from None
        if not self.analysis_fields:
            raise RadarAssimilationError(
                "analysis_fields is empty: an analysis that updates nothing "
                "is a bug, not a configuration")
        if len(set(self.analysis_fields)) != len(self.analysis_fields):
            raise RadarAssimilationError(
                f"analysis_fields has duplicates: {self.analysis_fields!r}")
        if not analysis_sources(self, extra_batches=None):
            raise RadarAssimilationError(
                "none of velocity, reflectivity, clear_air or cwp is "
                "enabled and extra_observations is declared False, so this "
                "config would assimilate nothing. An intentional "
                "no-observation cycle is a run_cycles call with "
                "assimilate=None, not an empty analysis here; leave "
                "extra_observations unset to let the batch list decide")
        if self.z_source not in Z_SOURCES:
            raise RadarAssimilationError(
                f"z_source must be one of {Z_SOURCES}, got "
                f"{self.z_source!r}")
        if self.fall_speed not in FALL_SPEED_POLICIES:
            raise RadarAssimilationError(
                f"fall_speed must be one of {FALL_SPEED_POLICIES}, got "
                f"{self.fall_speed!r}; an array-valued closure belongs in "
                "a custom velocity operator, not in this config")
        for label, cells in (
                ("velocity_thinning_cells", self.velocity_thinning_cells),
                ("reflectivity_thinning_cells",
                 self.reflectivity_thinning_cells),
                ("clear_air_thinning_cells", self.clear_air_thinning_cells),
                ("cwp_thinning_cells", self.cwp_thinning_cells)):
            if int(cells) < 1:
                raise RadarAssimilationError(
                    f"{label} must be >= 1, got {cells!r} (1 keeps every "
                    "observation)")
        for label, value in (
                ("velocity_error_inflation", self.velocity_error_inflation),
                ("reflectivity_error_inflation",
                 self.reflectivity_error_inflation),
                ("clear_air_error_inflation",
                 self.clear_air_error_inflation),
                ("cwp_error_inflation", self.cwp_error_inflation)):
            inflation = float(value)
            if not np.isfinite(inflation) or inflation < 1.0:
                raise RadarAssimilationError(
                    f"{label} must be finite and >= 1, got {value!r}; "
                    "deflating stated observation errors is a claim of skill "
                    "nobody measured")
        if self.relaxation not in RELAXATION_MODES:
            raise RadarAssimilationError(
                f"relaxation must be one of {RELAXATION_MODES}, got "
                f"{self.relaxation!r}")
        for label, value in (
                ("velocity_dispersion_ratio", self.velocity_dispersion_ratio),
                ("velocity_dispersion_batch_ratio",
                 self.velocity_dispersion_batch_ratio)):
            try:
                check_ratio(value, label)
            except DispersionGateError as exc:
                raise RadarAssimilationError(str(exc)) from None
        if self.moment_policy not in MOMENT_POLICIES:
            raise RadarAssimilationError(
                f"moment_policy must be one of {MOMENT_POLICIES}, got "
                f"{self.moment_policy!r}")
        constrained = constrained_fields(self.analysis_fields)
        if self.positivity_policy is None:
            if constrained:
                raise RadarAssimilationError(
                    "this analysis updates the physically non-negative "
                    f"field(s) {list(constrained)} and states no "
                    "positivity_policy. A Gaussian filter applied to a "
                    "bounded, zero-inflated variable routinely proposes a "
                    "negative mixing ratio, and mean-preserving / clip / "
                    "reject / none are not equivalent -- clipping at zero "
                    "ADDS mass and is "
                    "biased wetward, rejecting conserves the background and "
                    "invents gradients, and none lets the microphysics meet "
                    f"the negatives. Choose one of {POLICIES}; "
                    "gpuwm.da.positivity documents what each costs")
        elif self.positivity_policy not in POLICIES:
            raise RadarAssimilationError(
                f"positivity_policy must be one of {POLICIES} or None, got "
                f"{self.positivity_policy!r}")
        # PLAN REVIEW FOR THE DA DOOR, for ALL THREE arms that evaluate a
        # reflectivity operator.  A RunConfig carries no DA fields, so
        # validate_run_config cannot ask whether the active scheme has an
        # H(x); this configuration is where the cycle is decided, so the
        # registry's radar_da row is read here rather than by
        # simulated_reflectivity inside the first analysis, per member
        # (audit R-051).
        #
        # WHERE THAT IS EARLY ENOUGH IS THE CALLER'S HALF, and this class
        # cannot buy it alone: a driver that builds this configuration at
        # its first analysis seam has already spent an ensemble
        # integration whatever this __post_init__ does.  So the cycling
        # driver plans the configuration where the cycle is planned --
        # tools/da_cycle_prepared.py's plan_radar_assimilation, called
        # above the leg loop, before a member takes a step -- and builds
        # the leg's real one through the same function.  A driver that
        # skips that call gets its refusal at its first analysis, which is
        # the defect R-051 named, so the call is pinned by
        # tests/test_da_cycle_prepared.py rather than by this sentence.
        #
        # fall_speed="reflectivity" is the third arm and was not gated at
        # all: it calls the same operator (gpuwm.da.obsop
        # .reflectivity_fall_speed reads the simulated dBZ) whether or not
        # reflectivity observations are assimilated.
        #
        # ASKED ONLY OF A CYCLE THAT STATES ITS SCHEME.  A first pass at
        # this gate also refused mp_physics=None, and that was a refusal of
        # configurations that run: nothing in this module derives the
        # operator from this field.  The provider is the CALLER's
        # (assimilate_radar_grid takes reflectivity_provider, built by
        # scheme_reflectivity_provider off the RUN config, which dispatches
        # on run_cfg.mp_physics), and cfg.mp_physics is read only by the
        # moment policy -- which documents None as "detect the pairs from
        # the checkpoint's own spellings" -- by the clear-air floor, which
        # is separately guarded below, and by a provenance label.  Refusing
        # None also contradicted the unimplemented-selector sentence below,
        # which offers None as its way out; the two were reachable from one
        # configuration and instructed opposite actions.  What is left is
        # the half that is sound: a STATED scheme with no route is refused
        # at configuration, which the cycling driver reaches before leg 0.
        #
        # ASKED BEFORE THE CLEAR-AIR FLOOR, and the order matters.
        # The floor's own refusal offers "state clear_air_value_dbz", and
        # for a scheme with no H(x) at all that sentence reaches a sibling
        # refusal rather than a running configuration.  With the route
        # asked first, a scheme the operator cannot simulate is refused by
        # the arm-composed way out, and the floor refusal is reached only
        # by a scheme that HAS an operator -- where stating the value is a
        # real way out.
        needs_reflectivity_operator = bool(
            self.reflectivity or self.clear_air
            or self.fall_speed == "reflectivity")
        if needs_reflectivity_operator and self.mp_physics is not None:
            self._require_reflectivity_route(int(self.mp_physics))
        if self.clear_air:
            if self.clear_air_value_dbz is not None:
                floor = float(self.clear_air_value_dbz)
                if not np.isfinite(floor):
                    raise RadarAssimilationError(
                        f"clear_air_value_dbz is "
                        f"{self.clear_air_value_dbz!r}; it must be a finite "
                        "dBZ value or None to derive it from mp_physics")
                # STATED AND DERIVED MUST AGREE.  Honouring the explicit
                # number while the scheme's own H(x) floors somewhere else
                # is the defect the table was built to prevent, arrived at
                # from the other side: the operator writes the scheme's
                # floor in every clear cell, the observation carries this
                # one, and the difference is an innovation in air both
                # sides agree is empty.  mp=9 against -35.0 is 64 dB of it.
                # Only a scheme whose floor has been READ is compared --
                # one with no single floor (P3) has nothing to compare
                # against and keeps the documented "explicit value is
                # honoured" route.
                if self.mp_physics is not None:
                    scheme = int(self.mp_physics)
                    recorded = CLEAR_AIR_FLOOR_DBZ.get(scheme)
                    if recorded is not None and floor != recorded:
                        raise RadarAssimilationError(
                            f"clear_air_value_dbz={floor} dBZ disagrees "
                            f"with mp_physics={scheme}, whose H(x) floors "
                            f"at {recorded} dBZ "
                            "(gpuwm.da.obsop.CLEAR_AIR_FLOOR_DBZ). Every "
                            "clear-air observation would then be "
                            f"differenced against a background "
                            f"{abs(floor - recorded):g} dB away from it and "
                            "the analysis would build or erase condensate "
                            "in air both sides call empty. Drop "
                            "clear_air_value_dbz to derive the floor from "
                            f"the scheme, or state {recorded}; if the "
                            "file's zeroes genuinely mean something else, "
                            "they are not this operator's clear air and "
                            "belong out of the batch (clear_air=False)")
            elif self.mp_physics is None:
                raise RadarAssimilationError(
                    "clear_air is enabled with neither clear_air_value_dbz "
                    "nor mp_physics. A clear-air observation is differenced "
                    "against H(x), so it must carry the ACTIVE scheme's "
                    "clear-air floor -- -35 dBZ for the refl10cm family "
                    "(mp 1/6/8/10/16/28), 0 dBZ for NSSL mp18, -99 dBZ for "
                    "Milbrandt-Yau mp9. With neither the scheme nor the "
                    "value stated there is nothing to derive it from, and "
                    "the wrong floor is silent: two agreeing clear skies "
                    "produce a 35, 64 or 99 dB innovation -- whichever "
                    "pair of those three floors was crossed -- and the "
                    "analysis removes condensate to chase it")
            else:
                # Raises for a scheme whose floor nobody has read, here at
                # config time rather than mid-cycle.  Re-raised as this
                # module's own error: everything else __post_init__ refuses
                # is a RadarAssimilationError, and a caller that catches
                # the configuration's error type must not miss this one.
                # The message carries its own two ways out (state
                # clear_air_value_dbz, or clear_air=False) and both are
                # reachable from here, because the route question above
                # has already refused a scheme with no operator at all.
                try:
                    clear_air_floor_dbz(int(self.mp_physics))
                except ValueError as exc:
                    raise RadarAssimilationError(str(exc)) from exc
        if (self.clear_air or self.reflectivity) and not any(
                name in NON_NEGATIVE_FIELDS or name == "thp"
                for name in self.analysis_fields):
            raise RadarAssimilationError(
                "reflectivity is enabled but no analysed field is a "
                "thermodynamic or hydrometeor variable "
                f"({list(self.analysis_fields)}). Reflectivity constrains "
                "condensate; assimilating it against a wind-only state "
                "vector relies entirely on wind-hydrometeor sampling "
                "covariance, which at storm scale is noise. Analyse the "
                "scheme's moisture and hydrometeor set -- "
                "gpuwm.da.moments.analysis_fields derives it -- or turn "
                "reflectivity off")
        if self.cwp and not any(
                name in NON_NEGATIVE_FIELDS or name == "thp"
                for name in self.analysis_fields):
            raise RadarAssimilationError(
                "cwp is enabled but no analysed field is a thermodynamic or "
                f"hydrometeor variable ({list(self.analysis_fields)}). Cloud "
                "water path IS the column condensate; assimilating it "
                "against a wind-only state vector relies entirely on "
                "wind-condensate sampling covariance, which at storm scale "
                "is noise. Analyse the scheme's moisture and hydrometeor "
                "set -- gpuwm.da.moments.analysis_fields derives it -- or "
                "turn cwp off")
        if self.solve_device not in SOLVE_DEVICES:
            raise RadarAssimilationError(
                f"solve_device must be one of {SOLVE_DEVICES}, got "
                f"{self.solve_device!r}")
        if self.solve_cards != "all":
            try:
                cards = tuple(int(card) for card in self.solve_cards)
            except (TypeError, ValueError):
                cards = ()
            if not cards or any(card < 0 for card in cards) \
                    or len(set(cards)) != len(cards):
                raise RadarAssimilationError(
                    "solve_cards must be 'all' or distinct non-negative "
                    f"device ordinals, got {self.solve_cards!r}")
            object.__setattr__(self, "solve_cards", cards)
        if self.precip_analysis == "trim-build":
            raise RadarAssimilationError(PRECIP_ANALYSIS_MEMBER_REFUSAL)
        if self.precip_analysis not in PRECIP_ANALYSIS_MODES:
            raise RadarAssimilationError(
                f"precip_analysis must be one of {PRECIP_ANALYSIS_MODES}, "
                f"got {self.precip_analysis!r}")
        if self.precip_analysis != "off":
            if not (self.velocity or self.reflectivity or self.clear_air):
                raise RadarAssimilationError(
                    f"precip_analysis={self.precip_analysis!r} reads the "
                    "radar file's echo and clear-air masks, and with none "
                    "of velocity, reflectivity or clear_air enabled this "
                    "analysis reads no radar file: the stage would have no "
                    "observation to clear or build from. Enable a radar "
                    "source, or set precip_analysis='off'")
        object.__setattr__(self, "analysis_fields",
                           tuple(self.analysis_fields))
        if self.radars is not None:
            object.__setattr__(self, "radars",
                               tuple(str(r) for r in self.radars))

    def _reflectivity_arms_way_out(self) -> str:
        """How THIS cycle stops evaluating a reflectivity operator.

        Composed from the arms actually enabled.  A static sentence named
        the two arms the gate was written for and told a cycle whose only
        reflectivity-evaluating arm is ``fall_speed="reflectivity"`` to set
        two flags it had already set -- following it verbatim left the run
        refused, and the reachable way out (``fall_speed="none"``) was
        never named.  A refusal whose way out does not reach a running
        configuration is not a refusal.
        """
        arms = []
        if self.reflectivity:
            arms.append("reflectivity=False")
        if self.clear_air:
            arms.append("clear_air=False")
        if self.fall_speed == "reflectivity":
            arms.append("fall_speed='none' (radial velocity then projects "
                        "plain w instead of w - vt)")
        return ("Turn off the arm(s) this scheme has no operator for -- "
                + ", ".join(arms) + " -- which leaves the rest of the "
                "cycle, velocity included, exactly as configured.")

    def _require_reflectivity_route(self, mp_physics: int) -> None:
        """Refuse, at configuration, a scheme the operator cannot simulate.

        Reads ``consumers.radar_da.reflectivity_route`` off the physics
        registry for the active scheme.  ``operator`` and
        ``scheme-diagnostic`` have an H(x); ``native-not-separable`` and
        ``unrouted`` do not, and each is refused with the recorded reason
        and the way out.  A selector no implemented option carries is
        refused too: the operator has no H(x) for a scheme the model
        does not run.
        """

        from gpuwm.da.obsop import NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP
        from gpuwm.physics_registry import (
            consumer_row_for_selector, consumer_rows_by_selector)

        rows = consumer_rows_by_selector("microphysics", "radar_da")
        routed = sorted(
            mp for mp, row in rows.items()
            if isinstance(row, dict)
            and row.get("reflectivity_route") in ("operator", "scheme-diagnostic"))
        way_out = (self._reflectivity_arms_way_out()
                   + f" Or run a scheme with a routed H(x): {routed}.")
        row = consumer_row_for_selector("microphysics", "radar_da", mp_physics)
        if not isinstance(row, dict):
            raise RadarAssimilationError(
                f"mp_physics={mp_physics} is not an implemented microphysics "
                "option in gpuwm/physics_registry_v2.json (implemented: "
                f"{sorted(rows)}), so the radar operator has no H(x) for it; "
                "state the ACTIVE scheme's mp_physics, or leave it None to "
                "detect the moment structure from the checkpoint. " + way_out)
        route = row.get("reflectivity_route")
        if route == "native-not-separable":
            raise RadarAssimilationError(
                f"{NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP[mp_physics]} " + way_out)
        if route == "unrouted":
            raise RadarAssimilationError(
                f"mp_physics={mp_physics} produces its reflectivity natively "
                "and gpuwm/da/obsop.py names no H(x) route for it "
                f"(registry radar_da.reflectivity_route={route!r}: "
                f"{row.get('reflectivity_route_reason')}), so no "
                "reflectivity or clear-air observation can be simulated for "
                "any member. " + way_out)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def member_background_checkpoint(member_dir: str | Path) -> Path:
    """The newest ``gpuwmrst_*.npz`` in a member directory.  Fails closed.

    The SAME rule the cycle driver applies when it writes the analysis
    (``gpuwm.ensemble.cycle._member_background_checkpoint``): the state
    this module reads must be the state the increments are added to, or
    the analysis is an increment against one background applied to
    another.  ``tests/test_radar_assimilation.py`` binds the two rules to
    each other so they cannot drift apart silently.
    """
    member_dir = Path(member_dir)
    candidates = sorted(member_dir.glob("gpuwmrst_*.npz"))
    if not candidates:
        raise RadarAssimilationError(
            f"member directory {member_dir} carries no gpuwmrst_*.npz "
            "checkpoint, so there is no background to assimilate against. "
            "Set restart_interval_s in the base experiment config so each "
            "leg checkpoints at its end.")
    return candidates[-1]


def read_checkpoint_state(path: str | Path,
                          fields: Sequence[str] | None = None) -> dict:
    """``{field: ndarray}`` from a checkpoint's ``state/`` arrays.

    ``fields=None`` reads every state array the file carries (the forward
    operators read hydrometeors the analysis may not update).  Naming a
    field the file does not carry is a refusal: guessing zeros for a
    missing prognostic would manufacture an ensemble member that never
    existed.
    """
    path = Path(path)
    if not path.is_file():
        raise RadarAssimilationError(f"no checkpoint at {path}")
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        available = {key[len(CHECKPOINT_STATE_PREFIX):]: key
                     for key in data.files
                     if key.startswith(CHECKPOINT_STATE_PREFIX)}
        wanted = list(available) if fields is None else list(fields)
        missing = [name for name in wanted if name not in available]
        if missing:
            raise RadarAssimilationError(
                f"checkpoint {path} does not carry state field(s) "
                f"{missing}; it has {sorted(available)}")
        for name in wanted:
            out[name] = np.asarray(data[available[name]])
    return out


class CheckpointStateView(Mapping):
    """A member's ``state/`` arrays, read from its checkpoint on demand.

    The analysis touches a fraction of what a restart carries (the analysed
    fields plus what H(x) and the bounds read), and the cycle driver used to
    mirror the WHOLE serialized state of every member into host memory, save
    that mirror to a second ``.npz`` and read it back here with
    ``fields=None``.  At 241 x 241 x 49 and 32 members that was three whole
    copies of the ensemble crossing the host per analysis.  This view reads
    one array the first time it is asked for, keeps it, and never reads the
    rest.  The bytes are the checkpoint's own ``.npy`` payload, so a field
    read through the view is identical to the same field read by
    :func:`read_checkpoint_state`.

    The archive stays open until :meth:`close`; a consumed restart that is
    unlinked while the analysis still holds the view remains readable on
    POSIX, and every field the analysis needs is cached by then anyway.

    A restart member is an UNCOMPRESSED zip of raw ``.npy`` payloads, so a
    stored field is mapped read-only straight from the file rather than
    read through ``zipfile`` (which copies it in chunks and CRCs it on one
    core): the analysis then reads the page cache the worker's write left
    behind.  The barrier already hashed the whole file.  A compressed or
    unexpected member falls back to the ordinary read.  The mapped arrays
    are read-only, so an analysis step that tried to write a background
    in place would fail loudly instead of corrupting the restart.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise RadarAssimilationError(f"no checkpoint at {self.path}")
        self._archive = np.load(self.path, allow_pickle=False)
        self._keys = {key[len(CHECKPOINT_STATE_PREFIX):]: key
                      for key in self._archive.files
                      if key.startswith(CHECKPOINT_STATE_PREFIX)}
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return self.path.name

    def __getitem__(self, field: str) -> np.ndarray:
        value = self._cache.get(field)
        if value is not None:
            return value
        key = self._keys.get(field)
        if key is None:
            raise KeyError(field)
        if self._archive is None:
            raise RadarAssimilationError(
                f"checkpoint view {self.path} was closed before field "
                f"{field!r} was read")
        value = self._mapped(key)
        if value is None:
            value = np.asarray(self._archive[key])
        self._cache[field] = value
        return value

    def _mapped(self, key: str):
        """The stored ``.npy`` payload of ``key`` mapped read-only, or None."""
        import zipfile

        try:
            info = self._archive.zip.getinfo(key + ".npy")
        except (AttributeError, KeyError):
            return None
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        try:
            with open(self.path, "rb") as stream:
                stream.seek(info.header_offset)
                header = stream.read(30)
                if len(header) != 30 or header[:4] != b"PK\x03\x04":
                    return None
                start = (info.header_offset + 30
                         + int.from_bytes(header[26:28], "little")
                         + int.from_bytes(header[28:30], "little"))
                stream.seek(start)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = (
                        np.lib.format.read_array_header_1_0(stream))
                elif version == (2, 0):
                    shape, fortran, dtype = (
                        np.lib.format.read_array_header_2_0(stream))
                else:
                    return None
                offset = stream.tell()
        except (OSError, ValueError):
            return None
        count = int(np.prod(shape, dtype=np.int64))
        if (dtype.hasobject or count == 0 or not shape
                or offset - start + count * dtype.itemsize != info.file_size):
            return None
        mapped = np.memmap(self.path, dtype=dtype, mode="r", offset=offset,
                           shape=tuple(shape), order="F" if fortran else "C")
        return np.asarray(mapped)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, field) -> bool:
        return field in self._keys

    @property
    def cached_bytes(self) -> int:
        return int(sum(value.nbytes for value in self._cache.values()))

    def close(self) -> None:
        """Close the archive and drop every cached array."""
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        self._cache.clear()


def member_states(checkpoints: Mapping[int, object]) -> dict:
    """``{index: Mapping}`` for the analysis, without copying a state.

    A value may be a checkpoint path (read whole, as before) or an
    already-open mapping of ``state/`` arrays -- a :class:`CheckpointStateView`
    or the in-memory mirror of a live member -- which is used as it is.
    """
    out = {}
    for index in sorted(int(key) for key in checkpoints):
        source = checkpoints[index]
        out[index] = (source if isinstance(source, Mapping)
                      else read_checkpoint_state(source))
    return out


def checkpoint_label(source, index: int) -> str:
    """How the provenance names one member's background."""
    if isinstance(source, (str, Path)):
        return Path(source).name
    name = getattr(source, "name", None)
    return str(name) if name else f"in-memory member {int(index):03d}"


# ---------------------------------------------------------------------------
# staggering: mass-point analysis, face-point increments
# ---------------------------------------------------------------------------


def mass_to_u_faces(increment: np.ndarray) -> np.ndarray:
    """``(nz, ny, nx)`` mass increment -> ``(nz, ny, nx+1)`` u faces."""
    inc = np.asarray(increment)
    if inc.ndim != 3:
        raise RadarAssimilationError(
            f"mass increment must be 3-D, got {inc.shape}")
    nz, ny, nx = inc.shape
    out = np.empty((nz, ny, nx + 1), dtype=inc.dtype)
    out[:, :, 1:nx] = 0.5 * (inc[:, :, :-1] + inc[:, :, 1:])
    out[:, :, 0] = inc[:, :, 0]
    out[:, :, nx] = inc[:, :, -1]
    return out


def mass_to_v_faces(increment: np.ndarray) -> np.ndarray:
    """``(nz, ny, nx)`` mass increment -> ``(nz, ny+1, nx)`` v faces."""
    inc = np.asarray(increment)
    if inc.ndim != 3:
        raise RadarAssimilationError(
            f"mass increment must be 3-D, got {inc.shape}")
    nz, ny, nx = inc.shape
    out = np.empty((nz, ny + 1, nx), dtype=inc.dtype)
    out[:, 1:ny, :] = 0.5 * (inc[:, :-1, :] + inc[:, 1:, :])
    out[:, 0, :] = inc[:, 0, :]
    out[:, ny, :] = inc[:, -1, :]
    return out


def mass_to_w_faces(increment: np.ndarray) -> np.ndarray:
    """``(nz, ny, nx)`` mass increment -> ``(nz+1, ny, nx)`` w faces."""
    inc = np.asarray(increment)
    if inc.ndim != 3:
        raise RadarAssimilationError(
            f"mass increment must be 3-D, got {inc.shape}")
    nz, ny, nx = inc.shape
    out = np.empty((nz + 1, ny, nx), dtype=inc.dtype)
    out[1:nz, :, :] = 0.5 * (inc[:-1, :, :] + inc[1:, :, :])
    out[0, :, :] = inc[0, :, :]
    out[nz, :, :] = inc[-1, :, :]
    return out


_RESTAGGER = {"u": mass_to_u_faces, "v": mass_to_v_faces,
              "w": mass_to_w_faces}
_DESTAGGER = {"u": destagger_u, "v": destagger_v, "w": destagger_w}


def _saturation_bound(prior, increments, states, indices):
    """``(increments, receipt)`` with the analysed vapour held under each
    member's saturation limit and the ensemble mean kept
    (:func:`gpuwm.ensemble.increments.mean_preserving_saturation_bound`).
    The limit reads each member's full pressure and inverse dry density; a
    background without them is left to the applier's cap, which refuses to
    evaluate the same way, and the receipt says so."""
    from gpuwm.ensemble.increments import (SATURATION_BOUND_SCHEMA,
                                           mean_preserving_saturation_bound)

    missing = sorted({name for index in indices for name in ("p", "alt")
                      if name not in states[index]})
    if missing or len(indices) < 2:
        return increments, {
            "schema": SATURATION_BOUND_SCHEMA, "evaluated": False,
            "reason": (f"the members carry no {missing}; the applier's cap "
                       "is the only saturation bound" if missing else
                       "one member has no ensemble mean to keep")}
    pressure = np.stack([np.asarray(states[index]["p"]) for index in indices])
    inverse_density = np.stack([np.asarray(states[index]["alt"])
                                for index in indices])
    theta = (np.asarray(increments["thp"]) if "thp" in increments else None)
    bounded, receipt = mean_preserving_saturation_bound(
        np.asarray(prior["qv"]), np.asarray(increments["qv"]), pressure,
        inverse_density, theta)
    out = dict(increments)
    out["qv"] = bounded
    return out, {**receipt, "evaluated": True}


def _by_member(function, indices):
    """``[function(index) for index in indices]`` on threads, in order.

    Each call reads its own member and returns a new array, so threads
    change when it runs, never what; an exception is the first member's
    in order, as in the loop.
    """
    from concurrent.futures import ThreadPoolExecutor

    indices = list(indices)
    if len(indices) < 2:
        return [function(index) for index in indices]
    with ThreadPoolExecutor(max_workers=min(32, len(indices))) as pool:
        return list(pool.map(function, indices))


class MemberStack:
    """One analysis field's prior, ``(R, nz, ny, nx)`` float64, not stacked.

    The analysis used to stack every analysed field of every member into a
    private float64 prior before the filter (about 37 GB on the 9 km CONUS
    case, 32 members, fourteen fields; with the float64 increments beside
    it about 95 GB of controller memory per analysis).  The filter on the
    card never needed it whole: it stages the prior chunk by chunk.  This
    holds each member's own field -- the background array itself, or the
    mass-point projection of a staggered wind -- and hands out spans of
    points; ``np.asarray`` builds the whole float64 stack when a host
    stage needs one.  Every value is what :func:`_mass_field` and
    ``np.stack`` gave: a float32 background widens to float64 exactly
    wherever it is copied.
    """

    def __init__(self, name, states, indices, shape):
        self.name = name
        self.shape = (len(indices),) + tuple(int(v) for v in shape)
        self.ndim = 4
        self.dtype = np.dtype(np.float64)
        def member_field(index):
            where = f"member {index} checkpoint"
            if name in _DESTAGGER:
                field = _mass_field(name, states[index], where)
            else:
                if name not in states[index]:
                    _mass_field(name, states[index], where)   # its refusal
                field = np.asarray(states[index][name])
                if field.dtype.kind != "f":
                    field = np.asarray(field, dtype=np.float64)
            if field.shape != tuple(shape):
                raise RadarAssimilationError(
                    f"member {index} field {name!r} is {field.shape} at "
                    f"mass points but the observation file's grid is "
                    f"{tuple(shape)}; these checkpoints are not from this "
                    "domain")
            return np.ascontiguousarray(field).reshape(-1)

        # Members side by side (a staggered wind's projection, or the first
        # touch of a mapped restart); a refusal is the first member's.
        self._members = _by_member(member_field, indices)
        self._whole = None

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * 8

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        points = int(np.prod(self.shape[1:]))
        if tuple(shape) in ((self.shape[0], -1), (self.shape[0], points)):
            return _MemberSpans(self)
        return np.asarray(self).reshape(*shape)

    def __array__(self, dtype=None, copy=None):
        if self._whole is None:
            from concurrent.futures import ThreadPoolExecutor

            out = np.empty(self.shape, dtype=np.float64)
            flat = out.reshape(self.shape[0], -1)

            def fill(slot):
                flat[slot] = self._members[slot]

            with ThreadPoolExecutor(max_workers=max(1, min(
                    32, self.shape[0]))) as pool:
                list(pool.map(fill, range(self.shape[0])))
            self._whole = out
        whole = self._whole
        return whole if dtype is None else whole.astype(dtype, copy=False)

    def __getitem__(self, key):
        return np.asarray(self)[key]


class _MemberSpans:
    """``(R, points)`` view of a :class:`MemberStack`: ``[:, a:b]`` stacks
    the members' points ``a..b`` in their own dtype."""

    def __init__(self, stack):
        self._stack = stack
        self.shape = (stack.shape[0], int(np.prod(stack.shape[1:])))
        self.dtype = stack.dtype
        self.ndim = 2

    def __getitem__(self, key):
        rows, cols = key if isinstance(key, tuple) else (key, slice(None))
        if rows != slice(None) or not isinstance(cols, slice):
            return np.asarray(self._stack).reshape(self.shape)[key]
        return np.stack([member[cols] for member in self._stack._members])


def materialized(prior, names=None):
    """``{name: ndarray}`` of ``prior``'s fields (all, or ``names``)."""
    names = tuple(prior) if names is None else tuple(names)
    return {name: np.asarray(prior[name]) for name in names}


def _mass_field(name: str, state: Mapping[str, np.ndarray],
                where: str) -> np.ndarray:
    """One analysis field on mass points, float64, from checkpoint arrays."""
    if name not in state:
        raise RadarAssimilationError(
            f"{where} carries no state field {name!r}; the analysis "
            "vocabulary is the restart prognostic contract")
    value = np.asarray(state[name], dtype=np.float64)
    if name in _DESTAGGER:
        return np.asarray(_DESTAGGER[name](value))
    return value


# ---------------------------------------------------------------------------
# forward operators on checkpoint arrays
# ---------------------------------------------------------------------------


def grid_rotation(grid) -> tuple[np.ndarray, np.ndarray]:
    """``(SINALPHA, COSALPHA)`` at the grid's mass points.

    From the grid's own projection -- the authority the observations were
    placed with -- because a restart deliberately does not serialize the
    rotation (it is SETUP, rebuilt at restore).
    """
    sina, cosa = grid.projection.rotation(np.asarray(grid.lon))
    return (np.asarray(sina, dtype=np.float64),
            np.asarray(cosa, dtype=np.float64))


def member_earth_winds(state: Mapping[str, np.ndarray], rotation,
                       *, where: str) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray]:
    """``(u_east, v_north, w)`` mass-point earth-relative winds."""
    for name in WIND_FIELDS:
        if name not in state:
            raise RadarAssimilationError(
                f"{where} carries no {name!r}; a radial-velocity operator "
                "needs all three wind components")
    u_mass = np.asarray(destagger_u(np.asarray(state["u"], np.float64)))
    v_mass = np.asarray(destagger_v(np.asarray(state["v"], np.float64)))
    w_mass = np.asarray(destagger_w(np.asarray(state["w"], np.float64)))
    if rotation is not None:
        sina, cosa = rotation
        u_mass, v_mass = earth_relative_winds(u_mass, v_mass, sina, cosa)
    return u_mass, v_mass, w_mass


def reflectivity_route_available(mp_physics) -> bool:
    """Whether the registry routes an H_Z(x) for ``mp_physics``.

    The question :meth:`RadarAssimilationConfig._require_reflectivity_route`
    refuses on, asked without refusing, so a door can default the fall
    speed on where the scheme has an operator and say why it is off where
    it has none.
    """
    from gpuwm.physics_registry import consumer_row_for_selector

    if mp_physics is None:
        return False
    row = consumer_row_for_selector("microphysics", "radar_da",
                                    int(mp_physics))
    return (isinstance(row, dict) and row.get("reflectivity_route")
            in ("operator", "scheme-diagnostic"))


def _member_fall_speed(state: Mapping[str, np.ndarray], dbz: np.ndarray,
                       *, where: str) -> np.ndarray:
    """Sun & Crook vt (m/s downward) on mass points, from the member's own
    dBZ and full pressure.

    Surface pressure is the lowest mass level's full pressure: within the
    bottom half-layer of the true surface value (a <0.5% density-factor
    difference), while the ``P0`` substitution the operator refuses is a
    +4-15% systematic.  A checkpoint carries no separate surface-pressure
    field, and rebuilding one here would be a second hydrostatic authority.
    """
    if "p" not in state:
        raise RadarAssimilationError(
            f"{where} carries no full pressure 'p'; the Sun & Crook fall "
            "speed needs it. Use fall_speed='none' to accept the air-"
            "motion-only operator explicitly")
    pressure = np.asarray(state["p"], dtype=np.float64)
    hydrometeors = types.SimpleNamespace(
        **{name: state.get(name) for name in ("qr", "qs", "qg", "qh")})
    active = precipitating_activity_mask(hydrometeors)
    # Each frozen species at its own fall speed, not the rain relation
    # (gpuwm.da.obsop.species_blended_fall_speed); the checkpoint carries
    # alt, the inverse density the frozen speeds need.
    species = {name: np.asarray(state[name], dtype=np.float64)
               for name in ("qr", "qs", "qg", "qh")
               if state.get(name) is not None}
    alt = state.get("alt")
    return np.asarray(reflectivity_fall_speed(
        np.asarray(dbz, dtype=np.float64), pressure, active,
        surface_pressure=pressure[0],
        species=species if alt is not None else None,
        inverse_density=(None if alt is None
                         else np.asarray(alt, dtype=np.float64))))


def scheme_reflectivity_provider(run_cfg, *, base_theta):
    """A reflectivity H(x) provider bound to the scheme and the base state.

    ``run_cfg`` is the run's own config (``mp_physics`` selects the Z
    authority exactly as :func:`gpuwm.da.obsop.simulated_reflectivity`
    dispatches it).  ``base_theta`` is the base-state potential temperature
    ``thb`` -- a ``(nz,)`` column or the full ``(nz, ny, nx)`` field --
    supplied explicitly because a checkpoint does not serialize it (base
    state is SETUP) and temperature is not derivable without it.

    Returns ``provider(member_index, state_arrays) -> (nz, ny, nx) dBZ``.
    """
    theta_base = np.asarray(base_theta, dtype=np.float64)
    if theta_base.ndim not in (1, 3):
        raise RadarAssimilationError(
            f"base_theta must be (nz,) or (nz, ny, nx), got "
            f"{theta_base.shape}")
    from gpuwm.da.obsop import simulated_reflectivity  # noqa: PLC0415

    moisture = ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
                "nr", "ns", "ng", "nc", "ni")

    def provider(member_index: int, state: Mapping[str, np.ndarray]):
        for name in ("p", "thp", "qv"):
            if name not in state:
                raise RadarAssimilationError(
                    f"member {member_index}: checkpoint carries no {name!r}, "
                    "which the reflectivity operator needs")
        namespace = types.SimpleNamespace(
            p=np.asarray(state["p"], dtype=np.float64),
            thp=np.asarray(state["thp"], dtype=np.float64),
            thb=theta_base,
            **{name: (None if state.get(name) is None
                      else np.asarray(state[name], dtype=np.float64))
               for name in moisture})
        return np.asarray(simulated_reflectivity(namespace, run_cfg),
                          dtype=np.float64)

    return provider


# ---------------------------------------------------------------------------
# observation thinning
# ---------------------------------------------------------------------------


def thin_mask(mask, counts, errors, cells: int):
    """One survivor per ``cells x cells`` horizontal block, per level.

    The survivor is the observation with the most contributing gates
    (``counts``); ties go to the smaller error, then to the first cell in
    block order -- deterministic, so a thinned analysis is reproducible.
    Blocks with no observation keep none.  Fully vectorised: two argmax
    passes over a block-reshaped view, no Python loop over blocks.
    """
    cells = int(cells)
    mask = np.asarray(mask).astype(bool)
    if cells < 1:
        raise RadarAssimilationError(
            f"thinning block must be >= 1 cell, got {cells}")
    if cells == 1:
        return mask.copy()
    if mask.ndim != 3:
        raise RadarAssimilationError(
            f"thin_mask takes one (nz, ny, nx) mask, got {mask.shape}")
    nz, ny, nx = mask.shape
    counts = np.asarray(counts, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if counts.shape != mask.shape or errors.shape != mask.shape:
        raise RadarAssimilationError(
            f"counts {counts.shape} and errors {errors.shape} must match "
            f"the mask {mask.shape}")
    pad_j = (-ny) % cells
    pad_i = (-nx) % cells

    def padded(array, fill):
        return np.pad(array, ((0, 0), (0, pad_j), (0, pad_i)),
                      constant_values=fill)

    def blocked(array):
        blocks_j = (ny + pad_j) // cells
        blocks_i = (nx + pad_i) // cells
        return (array.reshape(nz, blocks_j, cells, blocks_i, cells)
                .transpose(0, 1, 3, 2, 4)
                .reshape(nz, blocks_j, blocks_i, cells * cells))

    m4 = blocked(padded(mask, False))
    c4 = blocked(padded(counts, 0.0))
    e4 = blocked(padded(errors, np.inf))

    count_key = np.where(m4, c4, -np.inf)
    best_count = count_key.max(axis=-1, keepdims=True)
    candidates = m4 & (count_key == best_count)
    error_key = np.where(candidates, -e4, -np.inf)
    winner = error_key.argmax(axis=-1)

    keep4 = np.zeros(m4.shape, dtype=bool)
    np.put_along_axis(keep4, winner[..., None], True, axis=-1)
    keep4 &= m4
    blocks_j = (ny + pad_j) // cells
    blocks_i = (nx + pad_i) // cells
    keep = (keep4.reshape(nz, blocks_j, blocks_i, cells, cells)
            .transpose(0, 1, 3, 2, 4)
            .reshape(nz, blocks_j * cells, blocks_i * cells))
    return keep[:, :ny, :nx]


def _thinned_reflectivity_document(document, cfg: RadarAssimilationConfig):
    """A working copy with the merged reflectivity batch thinned/inflated.

    The reflectivity batch has no radar axis -- the superob writer merges
    it across radars -- so this is :func:`thin_mask` applied once, with
    ``z_count`` as the "most gates" key.  Same rule, same determinism,
    same receipt shape as the velocity path; separate function because
    conflating a per-radar stack with a merged field is how one of them
    silently gets indexed as the other.
    """
    cells = int(cfg.reflectivity_thinning_cells)
    inflation = float(cfg.reflectivity_error_inflation)
    if cells == 1 and inflation == 1.0:
        return document, None
    variables = dict(document["variables"])
    z_mask = np.asarray(variables["z_mask"]).astype(bool)
    z_err = np.asarray(variables["z_err"], dtype=np.float64)
    counts = variables.get("z_count")
    counts = (np.ones_like(z_err) if counts is None
              else np.asarray(counts, dtype=np.float64))
    kept = thin_mask(z_mask, counts, z_err, cells) if cells > 1 else z_mask
    variables["z_mask"] = kept.astype(np.int8)
    if inflation != 1.0:
        variables["z_err"] = z_err * inflation
    copied = dict(document)
    copied["variables"] = variables
    receipt = {
        "cells": cells,
        "error_inflation": inflation,
        "points_before": int(z_mask.sum()),
        "points_after": int(kept.sum()),
        "rule": "one obs per block: most gates, then smallest error, "
                "then first cell; blocks with no obs keep none",
        "merged_across_radars": True,
    }
    return copied, receipt


def _thinned_clear_air_document(document, cfg: RadarAssimilationConfig):
    """A working copy with the clear-air batch thinned/inflated.

    Same rule and same determinism as the reflectivity path, keyed on
    ``z0_count`` -- the cell in each block with the most supporting
    measured gates wins, which is the cell whose clear-air claim rests on
    the most evidence.

    Thinning matters more here than anywhere else in the lane.  Clear air
    is most of any volume and it is smooth, so an unthinned zero batch
    puts thousands of near-identical numbers inside one localisation lens
    against an ensemble with a few tens of degrees of freedom.  That is
    the rank starvation the velocity precedent already measured, with a
    field that is far more correlated with itself than velocity is.
    """
    cells = int(cfg.clear_air_thinning_cells)
    inflation = float(cfg.clear_air_error_inflation)
    if cells == 1 and inflation == 1.0:
        return document, None
    variables = dict(document["variables"])
    if "z0_mask" not in variables:
        # The adapter raises the explanatory error; thinning a batch that
        # is not there must not pre-empt it with a KeyError.
        return document, None
    z0_mask = np.asarray(variables["z0_mask"]).astype(bool)
    z0_err = np.asarray(variables["z0_err"], dtype=np.float64)
    counts = variables.get("z0_count")
    counts = (np.ones_like(z0_err) if counts is None
              else np.asarray(counts, dtype=np.float64))
    kept = thin_mask(z0_mask, counts, z0_err, cells) if cells > 1 else z0_mask
    variables["z0_mask"] = kept.astype(np.int8)
    if inflation != 1.0:
        variables["z0_err"] = z0_err * inflation
    copied = dict(document)
    copied["variables"] = variables
    receipt = {
        "cells": cells,
        "error_inflation": inflation,
        "points_before": int(z0_mask.sum()),
        "points_after": int(kept.sum()),
        "rule": "one obs per block: most supporting gates, then smallest "
                "error, then first cell; blocks with no obs keep none",
        "merged_across_radars": True,
    }
    return copied, receipt


def _thinned_cwp_document(document, cfg: RadarAssimilationConfig):
    """A working copy with the CWP column batch thinned.

    The CWP product is 2-D -- one column integral per column -- so the
    shared :func:`thin_mask` is fed a single-level volume rather than
    given a second, nearly identical implementation.  The "most gates"
    key is ``cwp_count``, the valid satellite pixels averaged into the
    cell, which is the same kind of quantity ``z_count`` is.

    Unlike the two radar helpers this one does **not** apply the error
    inflation.  :func:`gpuwm.da.obs_goes.goes_grid_to_gridded_obs` does,
    and doing it in both places would square it silently -- an inflation
    of 2 would arrive at the filter as 4, which is a factor-of-four change
    in observation weight that no receipt would name.
    """
    cells = int(cfg.cwp_thinning_cells)
    if cells == 1:
        return document, None
    variables = dict(document["variables"])
    mask = np.asarray(variables["cwp_mask"]).astype(bool)
    errors = np.asarray(variables["cwp_err"], dtype=np.float64)
    counts = np.asarray(variables["cwp_count"], dtype=np.float64)
    kept = thin_mask(mask[None], counts[None], errors[None], cells)[0]
    variables["cwp_mask"] = kept.astype(np.int8)
    # cwp_class must follow the mask: the file's own consistency rule is
    # that a class is set exactly where an observation is, and a thinned
    # copy that kept the classes of dropped columns would fail the very
    # check that proves it is still a valid product.
    classes = np.asarray(variables["cwp_class"], dtype=np.int8)
    variables["cwp_class"] = np.where(kept, classes, np.int8(-1)).astype(
        np.int8)
    variables["obs_level"] = np.where(
        kept, np.asarray(variables["obs_level"], dtype=np.int32),
        np.int32(-1)).astype(np.int32)
    copied = dict(document)
    copied["variables"] = variables
    receipt = {
        "cells": cells,
        "points_before": int(mask.sum()),
        "points_after": int(kept.sum()),
        "rule": "one obs per block: most satellite pixels, then smallest "
                "error, then first cell; blocks with no obs keep none",
        "note": "applied to the column product, not per level: there is "
                "one CWP observation per column by construction. Error "
                "inflation is applied by the adapter, not here",
    }
    return copied, receipt


def _thinned_velocity_document(document, cfg: RadarAssimilationConfig):
    """A working copy of the document with thinned/inflated velocities.

    Returns ``(document, receipt)``.  The file on disk is never touched;
    the copy is shallow except for the two arrays this rewrites, and the
    grid-binding digests ride through unchanged so the copy still proves
    itself against the caller's grid.
    """
    cells = int(cfg.velocity_thinning_cells)
    inflation = float(cfg.velocity_error_inflation)
    if cells == 1 and inflation == 1.0:
        return document, None
    variables = dict(document["variables"])
    vr_mask = np.asarray(variables["vr_mask"]).astype(bool)
    vr_err = np.asarray(variables["vr_err"], dtype=np.float64)
    counts = variables.get("vr_count")
    counts = (np.ones_like(vr_err) if counts is None
              else np.asarray(counts, dtype=np.float64))
    per_radar = []
    thinned = np.zeros_like(vr_mask)
    for index in range(vr_mask.shape[0]):
        kept = thin_mask(vr_mask[index], counts[index], vr_err[index],
                         cells) if cells > 1 else vr_mask[index]
        thinned[index] = kept
        per_radar.append({"radar_index": index,
                          "points_before": int(vr_mask[index].sum()),
                          "points_after": int(kept.sum())})
    variables["vr_mask"] = thinned.astype(np.int8)
    if inflation != 1.0:
        variables["vr_err"] = vr_err * inflation
    copied = dict(document)
    copied["variables"] = variables
    receipt = {
        "cells": cells,
        "error_inflation": inflation,
        "radars": per_radar,
        "rule": "one obs per block: most gates, then smallest error, "
                "then first cell; blocks with no obs keep none",
    }
    return copied, receipt


# ---------------------------------------------------------------------------
# observation-space diagnostics
# ---------------------------------------------------------------------------


def resolve_solve_device(requested: str) -> tuple[str, str]:
    """``(device, reason)`` -- what "auto" becomes, and why.

    "host" and "cuda" are returned verbatim.  A caller that named a device
    named it for a reason this function cannot see: a run pinned to host
    for reproducibility must stay on host beside a working card, and a run
    that asked for cuda must meet the card's absence as an error where it
    happens, not as a silent demotion to a slower and differently-rounded
    experiment.

    "auto" is answered by IMPORTING cupy and asking the runtime for a
    device count, which is the only question whose answer is the one that
    matters.  An installed cupy that cannot reach a driver, a wheel built
    against the wrong CUDA major, a box with the libraries and no card:
    all three present as an exception here and all three resolve to host
    with the exception's type in the reason.  The reason is a sentence
    because it is going into a receipt, where "why is this run on the
    host" has to be answerable months later.
    """

    if requested in ("host", "cuda"):
        return requested, f"requested explicitly as {requested!r}"
    if requested != "auto":
        raise RadarAssimilationError(
            f"solve_device must be one of {SOLVE_DEVICES}, got "
            f"{requested!r}")
    # The project-wide "keep off the local card" switch outranks the
    # probe.  A CPU-only suite run, or a box whose card belongs to another
    # run, must not have a default quietly open a CUDA context on it --
    # which is exactly what a probe that only asked "is a device visible"
    # would do.
    from gpuwm.local_gpu import no_local_gpu
    if no_local_gpu():
        return "host", ("auto: GPUWM_NO_LOCAL_GPU is set, so the analysis "
                        "runs on numpy and never opens a device context")
    try:
        import cupy as cp                                 # noqa: PLC0415

        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return "host", (
            "auto: no usable CUDA device from this process "
            f"({type(exc).__name__}: {exc}); the analysis runs on numpy")
    if count < 1:
        return "host", ("auto: cupy imported but the runtime reports zero "
                        "devices; the analysis runs on numpy")
    return "cuda", (f"auto: {count} CUDA device(s) visible; the analysis "
                    "runs the batched device solve")


def innovation_summary(batches: Sequence[GriddedObs]) -> list[dict]:
    """Observation-space statistics per batch, JSON-serialisable.

    ``d = y - mean_k H(x_k)`` over the masked points: the number a real-
    data run is judged by before any increment exists.  Everything here is
    computed from the batch the filter itself consumes, so what is
    reported is what was assimilated, not a parallel reading of the file.
    """
    out = []
    for batch in batches:
        points = getattr(batch, "points", None)
        if points is not None:
            # A point batch states its observations directly; the dense
            # H(x) it would otherwise build is (R, nz, ny, nx).
            n = int(np.size(points.flat_index))
        else:
            mask = np.asarray(batch.mask, dtype=bool)
            n = int(np.count_nonzero(mask))
        entry: dict = {"name": batch.name, "observations": n}
        if n and points is not None:
            y = np.asarray(points.values, dtype=np.float64)
            sim = np.asarray(points.simulated, dtype=np.float64)
            err = np.asarray(points.errors, dtype=np.float64)
        elif n:
            y = np.asarray(batch.values, dtype=np.float64)[mask]
            sim = np.asarray(batch.simulated, dtype=np.float64)[:, mask]
            err = np.asarray(batch.errors, dtype=np.float64)
            err = (np.full(y.shape, float(err)) if err.ndim == 0
                   else err[mask])
        if n:
            hx = sim.mean(axis=0)
            d = y - hx
            spread = sim.std(axis=0, ddof=1) if sim.shape[0] > 1 else \
                np.zeros_like(hx)
            entry.update({
                "obs_mean": float(y.mean()),
                "obs_min": float(y.min()),
                "obs_max": float(y.max()),
                "hx_mean": float(hx.mean()),
                "hx_min": float(hx.min()),
                "hx_max": float(hx.max()),
                "innovation_mean": float(d.mean()),
                "innovation_rms": float(np.sqrt(np.mean(d ** 2))),
                "ensemble_spread_mean": float(spread.mean()),
                "obs_error_mean": float(err.mean()),
            })
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------



def _letkf_config(cfg: RadarAssimilationConfig,
                  memory_budget_mib: float) -> LetkfConfig:
    """The filter configuration this analysis solves with."""
    return LetkfConfig(
        localization=cfg.localization,
        analysis_fields=tuple(cfg.analysis_fields),
        rtps_alpha=cfg.rtps_alpha,
        prior_inflation=cfg.prior_inflation,
        relaxation=cfg.relaxation,
        chunk_points=cfg.chunk_points,
        memory_budget_mib=memory_budget_mib,
        solve_dtype=cfg.solve_dtype,
        eigensolver=cfg.eigensolver)


def analysis_device_price(cfg: RadarAssimilationConfig, *, members: int,
                          grid, document=None,
                          extra_localizations: Sequence = ()):
    """What :func:`assimilate_radar_grid` will hold on the card, or None.

    None when the solve resolves to the host.  Otherwise the price of the
    batches this configuration builds: the merged reflectivity and
    clear-air batches and each extra and satellite batch over the whole
    grid, and one radial-velocity batch per radar on the window
    ``document`` stores for it.  ``extra_localizations`` holds one entry
    per extra batch the caller will pass, its localisation or None.  The
    prior and every batch reach the card as float64: ``_mass_field``
    builds the prior that way and ``_analysis_attempt`` uploads the
    batches that way.  The filter configuration and its budget are the
    ones the solve takes (:func:`_letkf_config`, :func:`_execution_settings`).
    """
    from gpuwm.da.letkf import \
        analysis_device_price as _price  # noqa: PLC0415
    from gpuwm.da.obs_radar import velocity_batch_points  # noqa: PLC0415

    solve_device, _reason = resolve_solve_device(cfg.solve_device)
    if solve_device != "cuda":
        return None
    budget, _progress, _receipt = _execution_settings(cfg.memory_budget_mib,
                                                      None)
    shape = (int(grid.nz), int(grid.ny), int(grid.nx))
    whole = shape[0] * shape[1] * shape[2]
    batches = []
    if cfg.reflectivity:
        batches.append((whole, cfg.reflectivity_localization))
    if cfg.clear_air:
        batches.append((whole, cfg.clear_air_localization))
    if cfg.velocity and document is not None:
        batches.extend((points, cfg.velocity_localization)
                       for points in velocity_batch_points(
                           document, radars=cfg.radars))
    batches.extend((whole, loc) for loc in extra_localizations)
    if cfg.cwp:
        batches.append((whole, cfg.cwp_localization))
    float64 = np.dtype(np.float64).itemsize
    return _price(members=int(members), shape=shape,
                  fields=len(cfg.analysis_fields), prior_itemsize=float64,
                  batches=batches, grid=letkf_grid_geometry(grid),
                  config=_letkf_config(cfg, budget), obs_itemsize=float64)


def _resident_memory_failure(exc):
    """Only allocation failures or the owner's explicit capacity condition."""
    from gpuwm.da.letkf import LetkfCapacityError, LetkfError, _is_device_memory_error
    if isinstance(exc, (MemoryError, LetkfCapacityError)):
        return True
    if isinstance(exc, LetkfError):
        return exc.__cause__ is not None and _resident_memory_failure(exc.__cause__)
    if type(exc).__module__.split('.')[0] in ('cupy', 'cupy_backends'):
        return _is_device_memory_error(exc)
    # A numerical error or cancellation with an incidental OOM context is
    # still that error, not a request to try another storage mode.
    return False


def _analysis_attempt(solver, prior, batches, geometry, config, *, namespace,
                      storage, supports_staging, progress, diagnostics=None,
                      device_options=None):
    """Own every device reference until the result is back on the host."""
    if diagnostics is None:
        diagnostics = LetkfDiagnostics()
    if storage != 'cuda-obs-sparse' and any(
            isinstance(value, MemberStack) for value in prior.values()):
        # Only the device route stages a MemberStack itself.
        prior = materialized(prior)
    solve_prior, solve_batches = prior, batches
    stage_seconds = unstage_seconds = 0.
    if storage == 'cuda-resident':
        started = time.perf_counter()
        solve_prior = {name: namespace.asarray(value) for name, value in prior.items()}
        solve_batches = [replace(batch,
            values=namespace.asarray(np.asarray(batch.values, np.float64)),
            errors=namespace.asarray(np.asarray(batch.errors, np.float64)),
            simulated=namespace.asarray(np.asarray(batch.simulated, np.float64)),
            mask=namespace.asarray(np.asarray(batch.mask, bool))) for batch in batches]
        if hasattr(namespace, 'cuda'):
            namespace.cuda.runtime.deviceSynchronize()
        stage_seconds = time.perf_counter()-started
    if storage == 'cuda-obs-sparse':
        from gpuwm.da.letkf_device import analyze_device  # noqa: PLC0415
        increments = analyze_device(prior, batches, geometry, config,
                                    diagnostics, progress=progress,
                                    **(device_options or {}))
        return (increments, diagnostics,
                float(getattr(diagnostics, 'stage_seconds', 0.0)),
                float(getattr(diagnostics, 'unstage_seconds', 0.0)))
    options = dict(progress=progress) if supports_staging else {}
    if supports_staging and storage == 'host':
        # The host route stages on the host and solves there: the forked
        # chunk loop runs only when the solve namespace is numpy.
        options['solve_namespace'] = np
    elif supports_staging and storage != 'cuda-resident':
        options['solve_namespace'] = namespace
    increments = solver(solve_prior, solve_batches, geometry, config, diagnostics, **options)
    if storage == 'cuda-resident':
        started = time.perf_counter()
        increments = {name: namespace.asnumpy(value) for name, value in increments.items()}
        unstage_seconds = time.perf_counter()-started
    return increments, diagnostics, stage_seconds, unstage_seconds


def _obs_sparse_takes(solver, prior, config) -> tuple[bool, str | None]:
    """``(takes, reason)``: whether the observation-sparse device route
    solves this analysis, and why not when it does not.

    Only the ensemble transform itself (a static-covariance runner keeps its
    own routes), and only shapes the route's warp-per-point kernels take.
    A missing CUDA stack is a capability answer and is recorded; any other
    failure to import the route is a defect and raises, because swallowing
    it silently returned every cycle to a 23-minute host solve with nothing
    in the receipt to say why.
    """
    from gpuwm.da import letkf as _letkf  # noqa: PLC0415

    if solver is not _letkf.analyze:
        return False, None
    try:
        from gpuwm.da.letkf_device import supported  # noqa: PLC0415
    except ImportError as exc:
        return False, f'device LETKF unavailable: {type(exc).__name__}: {exc}'
    members = int(next(iter(prior.values())).shape[0])
    if not supported(members, config.solve_dtype):
        return False, (f'{members} members at {config.solve_dtype} is outside '
                       'the device LETKF kernels')
    return True, 'observation-sparse device LETKF'


#: Breakage these routes remove: on the 241 x 241 x 49, 32-member
#: storm-scale case the resident route could not fit the dense stencil
#: block and the host-staged route spent 23 minutes per analysis in
#: single-threaded numpy.  The device route (gpuwm.da.letkf_device) puts
#: weights, transform and diagnostics on the card and sizes its own chunk
#: from free memory.  Its fallback is the host route, which runs the chunk
#: loop on forked workers (gpuwm.da.letkf_host_parallel) and needs no card
#: memory at all, so it cannot fail the way the device route just did.
OBS_SPARSE_ROUTES = ('cuda-obs-sparse', 'host')


def _execute_analysis(solver, prior, batches, geometry, config, *, namespace,
                      device, progress=None, diagnostics=None,
                      device_options=None):
    supports_staging = bool(getattr(solver, 'supports_host_staging', False))
    routes = (['cuda-resident', 'host-staged-cuda'] if device == 'cuda' and supports_staging
              else ['cuda-resident'] if device == 'cuda' else ['host'])
    route_note = None
    if device == 'cuda':
        takes, why = _obs_sparse_takes(solver, prior, config)
        if takes:
            routes = list(OBS_SPARSE_ROUTES)
        elif why is not None:
            route_note = dict(attempt=0, storage='cuda-obs-sparse',
                              status='not-taken', reason=why,
                              wall_seconds=0.0, committed=False)
    attempts = [] if route_note is None else [route_note]
    last_progress = None
    for number, storage in enumerate(routes, 1):
        def relay(value):
            nonlocal last_progress
            last_progress = dict(value, attempt=number, storage=storage)
            if progress is not None:
                progress(last_progress)
        callback = (relay if supports_staging and device == 'cuda' else progress)
        started = time.perf_counter()
        try:
            result = _analysis_attempt(solver, prior, batches, geometry, config,
                namespace=namespace, storage=storage, supports_staging=supports_staging,
                progress=callback, diagnostics=diagnostics if len(routes) == 1 else None,
                device_options=device_options)
        except Exception as exc:
            if number == len(routes) or not _resident_memory_failure(exc):
                raise
            # Retain only scalar evidence, never the exception/traceback:
            # its frames own the arrays that must die before the retry.
            attempts.append(dict(attempt=number, storage=storage, status='memory-failed',
                error_type=type(exc).__name__, reason=str(exc),
                wall_seconds=time.perf_counter()-started, last_progress=last_progress,
                committed=False))
        else:
            attempts.append(dict(attempt=number, storage=storage, status='computed',
                wall_seconds=time.perf_counter()-started, committed=False))
            return (*result, storage, attempts)
        # Outside the exception handler: no failed frame should retain
        # resident arrays when the bounded mode takes its first allocation.
        gc.collect()
        if hasattr(namespace, 'cuda'):
            namespace.cuda.runtime.deviceSynchronize()
        namespace.get_default_memory_pool().free_all_blocks()
        if progress is not None:
            progress(dict(schema='gpuwm-da.analysis-progress.v1', phase='retry',
                attempt=number+1, storage=routes[number], gridpoints_done=0,
                gridpoints_total=int(np.prod(next(iter(prior.values())).shape[1:])),
                chunks=0, active_points=0, elapsed_seconds=time.perf_counter()-started,
                reason=attempts[-1]['reason'], previous_attempt_committed=False))
        last_progress = None
    raise AssertionError('Analysis execution ended without a result')


def _precip_analysis(cfg: RadarAssimilationConfig, document, increments,
                     states, indices, grid):
    """``(increments, receipt)`` after the radar precipitation analysis.

    Per member, on the device: the analysed fields the applier would write
    (each background plus the filter's increment cast to the state's
    float32, the arithmetic of
    :func:`gpuwm.ensemble.increments.apply_increments`) go through
    :func:`gpuwm.da.hydrometeor_analysis.hydrometeor_analysis`.  Where the
    stage moved a cell, the increment becomes analysed minus background;
    everywhere else it is the filter's increment, byte for byte, and for a
    field the filter did not analyse it is zero there.  Cells the radar did
    not sample are never moved (scope ``covered``).

    The field set comes from :mod:`gpuwm.da.moments`: ``clear`` takes every
    precipitating mass the checkpoint carries with its paired moments, and
    the set is validated under the full-moment policy, so no pair is split.
    ``clear`` is the only mode an ensemble member takes
    (:data:`PRECIP_ANALYSIS_MEMBER_REFUSAL`).
    """
    from gpuwm.da import hydrometeor_analysis as ha  # noqa: PLC0415
    from gpuwm.da.moments import validate_analysis_fields  # noqa: PLC0415

    cp = ha.require_device()
    if document is None:
        raise RadarAssimilationError(
            "the precipitation analysis needs the radar file, and none was "
            "read for this analysis")
    reflectivity, reflectivity_receipt = ha.radar_grid_reflectivity(
        document, z_source=cfg.z_source)
    shape = tuple(int(n) for n in reflectivity.shape)
    available = tuple(sorted(states[indices[0]]))
    mode = cfg.precip_analysis
    fields = ha.precipitating_fields(available, mp_physics=cfg.mp_physics)
    moment_receipt = validate_analysis_fields(
        fields, available=available, mp_physics=cfg.mp_physics,
        policy="full-moment")
    stage_cfg = ha.HydrometeorAnalysisConfig(mode=mode, scope="covered")

    volume = None
    z_w = np.asarray(getattr(grid, "z_w", np.zeros(0)), dtype=np.float64)
    if "alt" in available and z_w.ndim in (1, 3) \
            and z_w.shape[0] == shape[0] + 1:
        thickness = cp.diff(cp.asarray(z_w), axis=0)
        if thickness.ndim == 1:
            thickness = thickness[:, None, None]
        volume = cp.broadcast_to(thickness, shape) * (
            float(grid.dx_m) * float(grid.dy_m))

    out = dict(increments)
    for name in fields:
        if name not in out:
            out[name] = np.zeros((len(indices),) + shape, dtype=np.float64)
        else:
            out[name] = np.array(out[name], dtype=np.float64, copy=True)
    members = {}
    totals: dict[str, dict] = {}
    noaa_source = None
    for slot, index in enumerate(indices):
        state = states[index]
        backgrounds, analysed = {}, {}
        for name in fields:
            background = cp.asarray(np.ascontiguousarray(
                state[name], dtype=np.float32))
            if tuple(background.shape) != shape:
                raise RadarAssimilationError(
                    f"member {index} field {name!r} is "
                    f"{tuple(background.shape)}, the radar grid is {shape}")
            backgrounds[name] = background
            if name in increments:
                analysed[name] = background + cp.asarray(
                    np.asarray(increments[name][slot])).astype(cp.float32)
            else:
                analysed[name] = background
        result = ha.hydrometeor_analysis(analysed, reflectivity, stage_cfg)
        noaa_source = result.receipt["noaa_source"]
        air = None
        if volume is not None:
            air = volume / cp.asarray(np.ascontiguousarray(
                state["alt"], dtype=np.float64))
        per_field = {}
        for name in fields:
            after = result.analysed[name]
            moved = after.view(cp.int32) != analysed[name].view(cp.int32)
            filter_inc = (cp.asarray(np.asarray(increments[name][slot],
                                                dtype=np.float64))
                          if name in increments else
                          cp.zeros(shape, dtype=cp.float64))
            final = cp.where(moved, after.astype(cp.float64)
                             - backgrounds[name].astype(cp.float64),
                             filter_inc)
            out[name][slot] = cp.asnumpy(final)
            row = dict(result.receipt["per_field"][name])
            if air is not None:
                change = (after.astype(cp.float64)
                          - analysed[name].astype(cp.float64))
                row["removed_kg"] = float(cp.sum(
                    cp.where(change < 0.0, -change, 0.0) * air))
                row["added_kg"] = float(cp.sum(
                    cp.where(change > 0.0, change, 0.0) * air))
            per_field[name] = row
            total = totals.setdefault(name, {
                "units": row["units"], "removed_sum": 0.0, "added_sum": 0.0})
            total["removed_sum"] += row["removed_sum"]
            total["added_sum"] += row["added_sum"]
            if "removed_kg" in row:
                total["removed_kg"] = (total.get("removed_kg", 0.0)
                                       + row["removed_kg"])
                total["added_kg"] = (total.get("added_kg", 0.0)
                                     + row["added_kg"])
        members[int(index)] = {
            "cells": result.receipt["cells"],
            "per_field": per_field,
        }
    receipt = {
        "schema": ha.SCHEMA,
        "mode": mode,
        "status": ("opt-in; default-on waits on a rain-scored real case"),
        "stage": ("after the solve and positivity, before the saturation "
                  "bound and the applier's moment repair"),
        "noaa_source": noaa_source,
        "settings_source": {name: row["source"]
                            for name, row in ha.NOAA_SETTINGS.items()},
        "scope": stage_cfg.scope,
        "fields": list(fields),
        "filter_analysed_fields": [name for name in fields
                                   if name in increments],
        "moment_policy": moment_receipt,
        "reflectivity": reflectivity_receipt,
        "members_take": ("the clear step only, as NOAA gives every HRRRDAS "
                         "member (l_precip_clear_only); the trim and build "
                         "rule is a deterministic analysis's and is refused "
                         "for members"),
        "mass_units": ("removed_kg and added_kg are dry-air mass times "
                       "mixing ratio over dx*dy*dz, without map factors"
                       if volume is not None else None),
        "removed_water": ("removed, not moved to another field: a declared "
                          "sink, as in the HRRR's cloud analysis"),
        "members": members,
        "totals": totals,
    }
    return out, receipt


def assimilate_radar_grid(checkpoints: Mapping[int, str | Path | Mapping],
                          observations, grid,
                          cfg: RadarAssimilationConfig, *,
                          reflectivity_provider=None,
                          extra_obs=None,
                          extra_obs_provenance=None,
                          cwp_observations=None,
                          cwp_provider=None,
                          analysis_runner=None, progress=None,
                          diagnostics: LetkfDiagnostics | None = None
                          ) -> tuple[dict, dict]:
    """One LETKF analysis over member checkpoints, radar and/or satellite.

    The name is kept because every call site uses it; the function is now
    the multi-source analysis.  Radar observations and GOES cloud water
    path go into the same solve as separate batches with their own errors,
    thinning and localisation, which is the only arrangement in which a
    satellite column integral and a radar gate can constrain the same
    state without one being re-expressed as the other.

    Parameters
    ----------
    checkpoints
        ``{member_index: checkpoint path}`` -- each member's background,
        the file the driver will add the increments to.  A value may
        instead be the member's ``{field: ndarray}`` state itself, exactly
        what :func:`read_checkpoint_state` would return for its file; it is
        read, never written.
    observations
        A ``gpuwm-obs.radar-grid.v1`` path or an already-read document.
        May be ``None`` when none of ``cfg.velocity``, ``cfg.reflectivity``
        or ``cfg.clear_air`` is set -- a satellite-only analysis.
    grid
        The caller's own :class:`~gpuwm.obs.target_grid.TargetGrid`; every
        observation file is bound to its arrays (``z_w`` included), never
        to an identity string.
    cfg
        :class:`RadarAssimilationConfig`.
    reflectivity_provider
        ``(member_index, state_arrays) -> (nz, ny, nx) dBZ``; required
        when ``cfg.reflectivity`` or ``cfg.fall_speed == "reflectivity"``.
        :func:`scheme_reflectivity_provider` builds the scheme-true one.
    extra_obs
        Additional, already-validated :class:`~gpuwm.da.letkf.GriddedObs`
        batches to solve in the SAME analysis (surface observations from
        :mod:`gpuwm.da.obs_surface`, for instance).  The filter batches
        observation types independently, so appending them here is the
        whole integration -- but they must be on this very grid, with
        member axis in ascending checkpoint-index order, and their
        simulated H(x) already evaluated; this function cannot check the
        physics that produced them, only their shapes (the filter does
        that).  Their innovation statistics join the radar ones in the
        provenance.
    extra_obs_provenance
        JSON-serialisable provenance for ``extra_obs``, recorded verbatim
        under ``extra_observations``.  Required whenever ``extra_obs`` is
        non-empty: unattributed observations do not enter an analysis.
    cwp_observations
        A ``gpuwm-obs.goes-grid.v1`` path or document; required when
        ``cfg.cwp``.
    cwp_provider
        ``(member_index, state_arrays, obs_class) -> (ny, nx) g m-2``;
        required when ``cfg.cwp``.
        :func:`gpuwm.da.obsop_cwp.checkpoint_cwp_provider` builds it.  It
        takes the observation's phase class because the operator composes
        model condensate under the phase the retrieval saw, never the
        model's own -- see :mod:`gpuwm.da.obsop_cwp`.

    analysis_runner
        Optional transform with the same contract as letkf.analyze. Local
        deterministic cycling uses the static covariance mean transform;
        the default remains the existing ensemble transform.

    Returns
    -------
    ``(increments_by_member, provenance)`` in exactly the shape the cycle
    driver's seam consumes: ``{member_index: {field: ndarray}}`` with
    every array matching the CHECKPOINT shape of its field (winds on
    faces), and a JSON-serialisable method-provenance mapping carrying
    the adapter provenance, the observation-space innovation statistics
    and the filter diagnostics.
    """
    # The batch list is the fact about what this analysis carries, so the
    # emptiness of an analysis is decided here and not from a flag set
    # before anybody knew. It is the FIRST check: refusing before a
    # checkpoint is opened keeps the refusal at plan review rather than
    # partway into one.
    extra_batches = [] if extra_obs is None else list(extra_obs)
    if not analysis_sources(cfg, extra_batches=len(extra_batches)):
        raise RadarAssimilationError(
            "this analysis carries no velocity, reflectivity, clear_air or "
            "cwp source and no extra observation batch arrived, so it would "
            "update the background with nothing. Enable a source, pass the "
            "attributed extra batches, or run the cycle with "
            "assimilate=None, which is what a deliberate forecast-only "
            "cycle is")
    if not checkpoints:
        raise RadarAssimilationError("no member checkpoints were given")
    needs_dbz = (cfg.reflectivity or cfg.clear_air
                 or cfg.fall_speed == "reflectivity")
    if needs_dbz and reflectivity_provider is None:
        raise RadarAssimilationError(
            "this config needs member reflectivity (reflectivity="
            f"{cfg.reflectivity}, clear_air={cfg.clear_air}, "
            f"fall_speed={cfg.fall_speed!r}) but no "
            "reflectivity_provider was given. Build one with "
            "scheme_reflectivity_provider(run_cfg, base_theta=...) -- the "
            "checkpoint alone cannot supply it, because thb is setup "
            "state and temperature is not derivable without it")
    if cfg.cwp and cwp_provider is None:
        raise RadarAssimilationError(
            "cwp is enabled but no cwp_provider was given. Build one with "
            "gpuwm.da.obsop_cwp.checkpoint_cwp_provider(run_cfg, c1h=..., "
            "c2h=..., dnw=..., mub2d=...) -- the checkpoint alone cannot "
            "supply it, because the eta coordinate arrays and the base "
            "column mass are setup state")
    if cfg.cwp and cwp_observations is None:
        raise RadarAssimilationError(
            "cwp is enabled but no cwp_observations were given; a "
            "gpuwm-obs.goes-grid.v1 path or document is required")

    # Wall clock per stage of the observation side and around the filter,
    # in the order they run, so a receipt says where an analysis that is
    # not the filter spent its time (the filter splits its own).
    stage_clock = {}
    stage_mark = [time.perf_counter()]

    def _stage(name):
        now = time.perf_counter()
        stage_clock[name] = round(stage_clock.get(name, 0.0)
                                  + now - stage_mark[0], 3)
        stage_mark[0] = now

    needs_radar = cfg.velocity or cfg.reflectivity or cfg.clear_air
    document = None
    observed_document = None
    thinning_receipt = None
    z_thinning_receipt = None
    z0_thinning_receipt = None
    if needs_radar:
        if observations is None:
            raise RadarAssimilationError(
                f"velocity={cfg.velocity} / reflectivity={cfg.reflectivity} "
                f"/ clear_air={cfg.clear_air} are enabled but no radar "
                "observations were given")
        document = read_document(
            observations, expected_grid=grid,
            expected_grid_identity=grid.identity_sha256())
        # The precipitation analysis reads the masks as observed, never the
        # copies thinned for the filter's batches below.
        observed_document = document
        if cfg.velocity:
            document, thinning_receipt = _thinned_velocity_document(document,
                                                                    cfg)
        if cfg.reflectivity:
            document, z_thinning_receipt = _thinned_reflectivity_document(
                document, cfg)
        if cfg.clear_air:
            document, z0_thinning_receipt = _thinned_clear_air_document(
                document, cfg)
        dims = document["dims"]
        shape = (int(dims["level"]), int(dims["south_north"]),
                 int(dims["west_east"]))
    else:
        shape = (int(grid.nz), int(grid.ny), int(grid.nx))
    _stage("observations_read_and_thinned")

    indices = sorted(int(index) for index in checkpoints)
    states = member_states(checkpoints)
    _stage("checkpoints_read")

    # The moment policy is checked against what the BACKGROUND carries,
    # before a single H(x) is evaluated.  Doing it here rather than
    # leaving it to gpuwm.ensemble.increments is not redundancy: the
    # applier can only refuse an analysis that already cost an hour of
    # solve, and a field set that truncates a pair is a configuration
    # error, not a numerical one.
    available = tuple(sorted(states[indices[0]]))
    moment_receipt = validate_analysis_fields(
        tuple(cfg.analysis_fields), available=available,
        mp_physics=cfg.mp_physics, policy=cfg.moment_policy)

    # -- prior: analysis fields at mass points ------------------------------
    # Each field's prior is the members' own arrays, staged into the
    # filter's float64 chunks as the chunks run (MemberStack); a whole
    # float64 stack is built only where a host stage needs one.
    prior = {name: MemberStack(name, states, indices, shape)
             for name in cfg.analysis_fields}
    _stage("prior_stacked")

    # -- H(x) ----------------------------------------------------------------
    rotation = grid_rotation(grid)
    dbz_by_member = None
    if needs_dbz:
        dbz_by_member = {}
        for index in indices:
            dbz = np.asarray(reflectivity_provider(index, states[index]),
                             dtype=np.float64)
            if dbz.shape != shape:
                raise RadarAssimilationError(
                    f"reflectivity_provider returned {dbz.shape} for "
                    f"member {index}; the observation grid is {shape}")
            dbz_by_member[index] = dbz

    reflectivity_simulated = None
    if cfg.reflectivity:
        reflectivity_simulated = np.stack(
            [dbz_by_member[index] for index in indices])

    # A clear-air observation is differenced against exactly the same
    # H(x) an echo observation is -- the same scheme, the same members,
    # the same array.  What differs is where it applies and what value it
    # carries, not how the model side is computed.
    clear_air_simulated = None
    clear_air_value = None
    if cfg.clear_air:
        # The SAME array, not a second stack of the same members: nothing
        # downstream writes into a batch's H(x), and one whole-domain
        # member cube is 0.7 GB on the recent case.
        clear_air_simulated = (
            reflectivity_simulated if reflectivity_simulated is not None
            else np.stack([dbz_by_member[index] for index in indices]))
        clear_air_value = (
            float(cfg.clear_air_value_dbz)
            if cfg.clear_air_value_dbz is not None
            else clear_air_floor_dbz(int(cfg.mp_physics)))

    velocity_simulated = None
    if cfg.velocity:
        def member_winds(index):
            u_e, v_n, w_m = member_earth_winds(
                states[index], rotation, where=f"member {index} checkpoint")
            if cfg.fall_speed == "reflectivity":
                w_m = w_m - _member_fall_speed(
                    states[index], dbz_by_member[index],
                    where=f"member {index} checkpoint")
            return (u_e, v_n, w_m)

        winds = dict(zip(indices, _by_member(member_winds, indices)))

        def velocity_simulated(radar_index, radar):
            # Evaluated only where this radar's batch observes, on the
            # batch's own extent: the same products of the same numbers
            # at those points, zero elsewhere (where the filter zeroes
            # H(x) anyway).  The whole-domain evaluation cost a member
            # cube per radar, held alive by the cropped batch view.
            return observed_radial_velocity(
                document, radar_index, [winds[index] for index in indices])

    _stage("forward_operator_inputs")
    batches = []
    adapter_provenance = None
    if needs_radar:
        batches, adapter_provenance = radar_grid_to_gridded_obs(
            document, expected_grid=grid,
            expected_grid_identity=grid.identity_sha256(),
            reflectivity_simulated=reflectivity_simulated,
            velocity_simulated=velocity_simulated,
            z_source=cfg.z_source,
            reflectivity_localization=cfg.reflectivity_localization,
            velocity_localization=cfg.velocity_localization,
            clear_air_simulated=clear_air_simulated,
            clear_air_value_dbz=clear_air_value,
            clear_air_localization=cfg.clear_air_localization,
            # Already applied to z0_err in the thinned document above, the
            # same way velocity and reflectivity inflate theirs. Passing it
            # again here would square it.
            clear_air_error_inflation=1.0,
            radars=None if cfg.radars is None else list(cfg.radars))

    # -- reflectivity echo conditioning --------------------------------------
    # The common echo floor and the outlier tempering, on the echo batch only
    # (gpuwm.da.reflectivity_echo names the breakage: weak returns
    # differenced against the -35 dBZ H(x) floor doubled the echo area an
    # hour after the CONUS first-light analysis).  Recorded every analysis.
    reflectivity_echo_receipt = None
    spread_gate_batch = None
    radar_class_receipt = None
    if cfg.reflectivity or cfg.clear_air:
        from gpuwm.da.obs_radar import (  # noqa: PLC0415
            CLEAR_AIR_NAME, REFLECTIVITY_NAME)
        pressure = (np.asarray(states[indices[0]]["p"])
                    if "p" in states[indices[0]] else None)
        if pressure is not None and pressure.shape != shape:
            pressure = None
        z_keep = (None if pressure is None else _rc.level_keep(
            pressure, cfg.reflectivity_level_stride, cfg.radar_top_pa))
        z0_keep = (None if pressure is None else _rc.level_keep(
            pressure, cfg.clear_air_level_stride, cfg.radar_top_pa))
        classed = cfg.reflectivity_clear_floor_dbz is not None
        radar_class_receipt = {
            "schema": _rc.SCHEMA, "z_source": cfg.z_source,
            "vertical_thinning_evaluated": pressure is not None}
        conditioned = []
        for batch in batches:
            if batch.name == REFLECTIVITY_NAME:
                # Preserve raw dry-member H(x) before radar class floors.
                spread_gate_batch = batch
                batch, floor, radar_class_receipt["echo"] = (
                    _rc.classify_echo_batch(
                        batch, clear_floor=cfg.reflectivity_clear_floor_dbz,
                        dead_band=cfg.reflectivity_dead_band,
                        echo_error=cfg.reflectivity_error_dbz,
                        clear_error=cfg.clear_air_error_dbz, keep=z_keep))
                batch = _rc.apply_floor(batch, floor)
                # With classes on, the per-class floor above IS the common
                # floor; the A7 conditioning keeps only its tempering.
                batch, reflectivity_echo_receipt = (
                    condition_reflectivity_batch(
                        batch,
                        floor_dbz=(None if classed
                                   else cfg.reflectivity_floor_dbz),
                        outlier_sigmas=cfg.reflectivity_outlier_sigmas))
                batch, radar_class_receipt["echo_huber"] = _rc.huber_errors(
                    batch, cfg.reflectivity_huber_c)
                _rc.assert_shared_floor(batch, floor)
            elif batch.name == CLEAR_AIR_NAME:
                batch, floor, radar_class_receipt["clear_air"] = (
                    _rc.classify_clear_batch(
                        batch, clear_floor=cfg.reflectivity_clear_floor_dbz,
                        clear_error=cfg.clear_air_error_dbz, keep=z0_keep))
                batch = _rc.apply_floor(batch, floor)
                _rc.assert_shared_floor(batch, floor)
            conditioned.append(batch)
        batches = conditioned

    _stage("radar_batches")
    if extra_batches:
        if extra_obs_provenance is None:
            raise RadarAssimilationError(
                f"{len(extra_batches)} extra observation batch(es) were "
                "given with no extra_obs_provenance; unattributed "
                "observations do not enter an analysis")
        radar_names = {batch.name for batch in batches}
        for batch in extra_batches:
            if not isinstance(batch, GriddedObs):
                raise RadarAssimilationError(
                    "extra_obs entries must be GriddedObs, got "
                    f"{type(batch).__name__}")
            if batch.name in radar_names:
                raise RadarAssimilationError(
                    f"extra observation batch {batch.name!r} collides "
                    "with a radar batch name; two batches with one name "
                    "would be indistinguishable in every receipt")
            expected = np.shape(batch.mask)
            if tuple(expected) != shape:
                raise RadarAssimilationError(
                    f"extra observation batch {batch.name!r} is on a "
                    f"{tuple(expected)} grid, this analysis is on {shape}")
            simulated_members = int(np.shape(batch.simulated)[0])
            if simulated_members != len(indices):
                raise RadarAssimilationError(
                    f"extra observation batch {batch.name!r} carries H(x) "
                    f"for {simulated_members} member(s), the checkpoint "
                    f"set has {len(indices)}; the member axis must be the "
                    "ascending checkpoint-index order")
        batches = batches + extra_batches

    # -- the satellite batch -------------------------------------------------
    # Read, thin, THEN evaluate H(x): the operator composes model
    # condensate under the observation's phase class, so the class it is
    # handed has to be the one that survived thinning, not the one the
    # file started with.
    cwp_thinning_receipt = None
    cwp_provenance = None
    if cfg.cwp:
        from gpuwm.da.obs_goes import (  # noqa: PLC0415
            read_document as read_goes_document)

        cwp_document = read_goes_document(
            cwp_observations, expected_grid=grid,
            expected_grid_identity=grid.identity_sha256())
        cwp_document, cwp_thinning_receipt = _thinned_cwp_document(
            cwp_document, cfg)
        obs_class = np.asarray(cwp_document["variables"]["cwp_class"],
                               dtype=np.int8)
        cwp_stack = []
        for index in indices:
            simulated = np.asarray(
                cwp_provider(index, states[index], obs_class),
                dtype=np.float64)
            if simulated.shape != shape[1:]:
                raise RadarAssimilationError(
                    f"cwp_provider returned {simulated.shape} for member "
                    f"{index}; a column integral on this grid is "
                    f"{shape[1:]}")
            cwp_stack.append(simulated)
        cwp_batches, cwp_provenance = goes_grid_to_gridded_obs(
            cwp_document, expected_grid=grid,
            expected_grid_identity=grid.identity_sha256(),
            cwp_simulated=np.stack(cwp_stack),
            localization=cfg.cwp_localization,
            error_inflation=cfg.cwp_error_inflation)
        batches = list(batches) + list(cwp_batches)

    if not batches:
        raise RadarAssimilationError(
            "no observation batch was built, so this analysis would move "
            "nothing. The config enabled a type whose adapter returned "
            "nothing, which is a bug rather than an empty cycle")

    _stage("extra_and_satellite_batches")
    innovations = innovation_summary(batches)
    _stage("innovation_summary")

    # -- the radial-velocity dispersion gate ---------------------------------
    # Where a Vr batch's innovations outrun its ensemble and error, as a
    # whole and in a column, it is withheld from theta and vapour there
    # (gpuwm.da.velocity_dispersion names the breakage and the measurement
    # behind both thresholds).  The ratios are recorded for every analysis,
    # gate on or off.
    dispersion_geometry = letkf_grid_geometry(grid)
    dispersion_gates, dispersion_receipt = velocity_dispersion(
        batches, dx_m=float(dispersion_geometry.dx_m),
        dy_m=float(dispersion_geometry.dy_m),
        localization=cfg.localization, shape=shape,
        ratio=cfg.velocity_dispersion_ratio,
        prior_inflation=float(cfg.prior_inflation),
        batch_ratio=cfg.velocity_dispersion_batch_ratio)
    dispersion_gates = tuple(
        gate for gate in dispersion_gates
        if set(gate.fields) & set(cfg.analysis_fields))
    _stage("velocity_dispersion_gate")

    # -- the filter ----------------------------------------------------------
    if diagnostics is None:
        diagnostics = LetkfDiagnostics()
    execution_budget, progress, execution_receipt = _execution_settings(
        cfg.memory_budget_mib, progress)
    letkf_cfg = _letkf_config(cfg, execution_budget)
    solve_device, solve_device_reason = resolve_solve_device(cfg.solve_device)
    namespace = np
    if solve_device == 'cuda':
        import cupy as namespace
    solver = analyze if analysis_runner is None else analysis_runner
    spread_controller = None
    if cfg.spread_repair != "off" and analysis_runner is None:
        from gpuwm.da.spread_repair import SpreadRepairConfig, SpreadRepairController
        spread_controller = SpreadRepairController(SpreadRepairConfig(
            policy=cfg.spread_repair, observed_threshold_dbz=cfg.spread_repair_z_threshold,
            adaptive=cfg.spread_repair_adaptive), seed=cfg.spread_repair_seed)
        solver = spread_controller.analysis_runner(
            pressure=tuple(states[index]["p"] for index in indices),
            gate_radar=spread_gate_batch)
    if hasattr(solver,"gate_radar"):
        solver.gate_radar = spread_gate_batch
        spread_controller = solver.spread_controller
    if spread_controller is not None:
        # Reconstructed replay factories and the persistent cycle runner
        # must draw the same noise for the same validated observation slot.
        spread_controller.set_analysis_time(
            document.get("valid_time") if document is not None else None)
    # On the device route the gated columns are re-solved, and the
    # positivity policy applied, inside the filter's own pass over the
    # grid, on the card each chunk is on (gpuwm.da.letkf_device): the same
    # bytes as the host stages below, which run only when another route
    # solved.  On the 9 km CONUS case those two host stages were 114 s of
    # 391 small solves and 35 to 265 s of one host core.
    # -- which fields each kind of observation may move ---------------------
    # Gates grouped by identical field sets (gpuwm.da.field_rules): the
    # dispersion gate's (thp, qv) and the cloud water path rule's share one
    # group, the surface rule's hydrometeors are another.  On the device
    # route every group is re-solved inside the filter's pass; otherwise one
    # exact local withhold per group runs after it.
    from gpuwm.da import field_rules

    field_rules_receipt = None
    kind = []
    if cfg.kind_field_rules:
        kind, field_rules_receipt = field_rules.kind_gates(
            batches, tuple(cfg.analysis_fields), shape=shape,
            dx_m=float(dispersion_geometry.dx_m),
            dy_m=float(dispersion_geometry.dy_m),
            localization=cfg.localization)
    # Lane 3's radar rules as gates in the same groups (gpuwm.da.field_rules
    # .radar_rule_gates): one pass, no extra full solve.
    radar_rule_receipt = {"rules": cfg.field_rules}
    radar_gates, echo_gates, echo = [], None, None
    if cfg.field_rules == "design" and any(
            b.name in _fr.RADAR_BATCHES for b in batches):
        echo = _fr.echo_columns(
            batches, echo_floor=15.0,
            dilate=max(1, int(cfg.reflectivity_thinning_cells)))
        keep0 = np.zeros(np.shape(echo), dtype=bool)
        for gate in list(dispersion_gates) + kind:
            if set(gate.fields) == {"thp", "qv"}:
                keep0 |= np.asarray(_fr._host(gate.columns), dtype=bool)
        radar_gates, echo_gates, radar_rule_receipt = _fr.radar_rule_gates(
            batches, tuple(cfg.analysis_fields), shape=shape, echo=echo,
            z_thermo_weight=cfg.z_thermo_weight,
            z_hydrometeors=cfg.z_hydrometeors,
            in_pass_keep=cfg.z_qv_cap is None, keep0_thermo=keep0)
    gate_groups = field_rules.group_by_fields(
        list(dispersion_gates) + kind + radar_gates)
    # A group none of whose fields is analysed has nothing to re-solve
    # (column_plan gives None, and withhold would return the increments).
    in_filter = [plan for plan in (
        column_plan(group, tuple(cfg.analysis_fields))
        for group in gate_groups) if plan is not None]
    device_options = {"devices": cfg.solve_cards}
    if in_filter:
        device_options["withhold"] = tuple(
            ColumnWithhold(fields=fields, column_zone=column_zone,
                           zones=zones,
                           keep=tuple(entry["keep"] for entry in _receipt))
            for fields, column_zone, zones, _receipt in in_filter)
    if cfg.positivity_policy is not None:
        device_options["chunk_hook"] = DevicePositivity(
            cfg.positivity_policy, fields=tuple(cfg.analysis_fields))
    increments, completed_diagnostics, stage_seconds, unstage_seconds, storage, attempts = _execute_analysis(
        solver, prior, batches, letkf_grid_geometry(grid), letkf_cfg,
        namespace=namespace, device=solve_device, progress=progress, diagnostics=diagnostics,
        device_options=device_options)
    # Failed attempts never contaminate the caller's success diagnostics.
    vars(diagnostics).update(vars(completed_diagnostics))
    _stage("filter")

    def _gated_solve(gated_prior, gated_batches, fields, gated_geometry):
        gated_cfg = replace(letkf_cfg, analysis_fields=tuple(fields))
        solved, *_ = _execute_analysis(
            getattr(solver,"subsolve",solver), gated_prior, gated_batches, gated_geometry,
            gated_cfg, namespace=namespace, device=solve_device,
            progress=None, diagnostics=LetkfDiagnostics())
        return solved

    in_filter_withheld = getattr(completed_diagnostics, "withheld", None)
    if in_filter and in_filter_withheld is not None:
        dispersion_solves = {
            "gates": [gate.payload() for gate in dispersion_gates],
            "route": in_filter_withheld["route"],
            "groups": in_filter_withheld.get("groups"),
            "points": in_filter_withheld["points"],
            "active_points": in_filter_withheld["active_points"],
            "seconds": in_filter_withheld["seconds"],
            "solves": [entry for plan in in_filter for entry in plan[3]]}
    else:
        dispersion_solves = {"gates": [gate.payload()
                                       for gate in dispersion_gates],
                             "solves": []}
        for group in gate_groups:
            increments, solves = withhold(
                _gated_solve, materialized(prior, tuple(
                    name for name in cfg.analysis_fields
                    if any(name in gate.fields for gate in group))),
                batches, increments, group,
                tuple(cfg.analysis_fields), geometry=dispersion_geometry,
                localization=cfg.localization)
            dispersion_solves["solves"].extend(solves["solves"])
    if field_rules_receipt is not None:
        field_rules_receipt["solves"] = [
            entry for entry in dispersion_solves["solves"]
            if set(entry["withheld"]) & {
                row["batch"] for row in field_rules_receipt["gates"]}]
    dispersion_receipt["withheld"] = dispersion_solves
    _stage("withheld_solves")

    # -- radar field rules: the inside-echo blend --------------------------
    # The gates above already gave winds, masses and theta/qv outside echo
    # their rules in the filter's own pass.  Inside observed echo the design
    # keeps z_thermo_weight of the radar's theta/qv increment: one local
    # re-solve without the radar over the echo columns only (its cost
    # follows the echo area), then a blend.  Numbers and aerosols take no
    # filter increment; they are rediagnosed below.
    field_rule_receipt = radar_rule_receipt
    rule_fields = ()
    if cfg.field_rules == "design" and radar_gates:
        thermo = tuple(f for f in ("thp", "qv") if f in cfg.analysis_fields)
        if echo_gates and thermo:
            from gpuwm.da.velocity_dispersion import DispersionGate
            radar_names = set(_fr.RADAR_BATCHES)
            same = [DispersionGate(batch=g.batch, fields=g.fields,
                                   columns=np.asarray(g.columns, bool) & echo)
                    for g in list(dispersion_gates) + kind
                    if set(g.fields) == {"thp", "qv"}
                    and g.batch not in radar_names]
            withheld, blend_solves = withhold(
                _gated_solve, materialized(prior, thermo), batches,
                {f: increments[f] for f in thermo}, same + echo_gates,
                thermo, geometry=dispersion_geometry,
                localization=cfg.localization)
            increments = _fr.blend_echo(
                increments, withheld, columns=echo,
                weight=cfg.z_thermo_weight, qv_cap=cfg.z_qv_cap)
            field_rule_receipt["echo_blend_solves"] = len(
                blend_solves.get("solves", []))
        increments = _fr.zero_unanalysed(increments,
                                         tuple(cfg.analysis_fields))
        rule_fields = thermo
    _stage("field_rules")

    # -- positivity ----------------------------------------------------------
    # On the MASS-POINT increments, before restaggering, because the
    # constraint is "prior + increment >= 0" and the prior it must be
    # evaluated against is the mass-point prior the filter analysed.
    # Doing it after the wind restagger would be evaluating a constraint
    # against a field that no longer lines up with it -- and every
    # constrained field is mass-shaped anyway, so nothing is lost.
    positivity_receipt = None
    in_filter_positivity = getattr(completed_diagnostics,
                                   "chunk_hook_result", None)
    if cfg.positivity_policy is not None and in_filter_positivity is not None:
        # Applied and verified chunk by chunk on the card, before the
        # increments left it (gpuwm.da.positivity.DevicePositivity).
        positivity_receipt = in_filter_positivity
        redo = tuple(f for f in rule_fields if f in NON_NEGATIVE_FIELDS)
        if redo:
            # The field rules replaced these after the device clipped the
            # joint solve, so they are bounded again here.
            increments, positivity_receipt_rules = apply_positivity(
                prior, increments, policy=cfg.positivity_policy, fields=redo)
            positivity_receipt = {"device": in_filter_positivity,
                                  "after_field_rules": positivity_receipt_rules}
    elif cfg.positivity_policy is not None:
        increments, positivity_receipt = apply_positivity(
            prior, increments, policy=cfg.positivity_policy,
            fields=tuple(cfg.analysis_fields))
        if cfg.positivity_policy in BOUNDING_POLICIES:
            # The post-condition that catches a policy applied to the
            # wrong mapping: a receipt claiming N clipped points beside
            # increments that were never clipped.
            verify_non_negative(prior, increments,
                                fields=tuple(cfg.analysis_fields))

    # Full-domain maintenance follows the caller's positivity step exactly
    # once. Cropped field-rule re-solves do not redraw or move its gates.
    if hasattr(solver,"finish"):
        increments = solver.finish(prior,increments,diagnostics=diagnostics,
                                   solve_namespace=namespace)
        if cfg.positivity_policy in BOUNDING_POLICIES:
            verify_non_negative(prior,increments,fields=tuple(cfg.analysis_fields))
    _stage("spread_maintenance")

    # -- the radar precipitation analysis ------------------------------------
    # After the solve and positivity, before the saturation bound and the
    # applier's moment repair: the order the HRRR runs its cloud analysis in
    # (after the last outer loop).  Off, it is not called and nothing here
    # changes a byte (gpuwm.da.hydrometeor_analysis).
    _stage("positivity")
    precip_receipt = None
    output_fields = tuple(cfg.analysis_fields)
    if cfg.precip_analysis != "off":
        increments, precip_receipt = _precip_analysis(
            cfg, observed_document, increments, states, indices, grid)
        output_fields = output_fields + tuple(
            name for name in precip_receipt["fields"]
            if name not in output_fields)

    # -- vapour at or below saturation, the ensemble mean kept ---------------
    # On the same whole-ensemble mass-point arrays, after positivity: the
    # applier's per-member saturation cap alone cuts the upper tail of the
    # members' analysed vapour at every analysis and keeps the lower one,
    # a one-signed sink of 30 to 44 g m-2 per member per analysis on a
    # storm-scale child (gpuwm.ensemble.increments
    # .mean_preserving_saturation_bound).  It never makes vapour negative:
    # a member over its limit ends at it, and a member under its limit
    # keeps a share of its headroom, so it only gains vapour.
    _stage("precip_analysis")
    saturation_bound_receipt = None
    if "qv" in increments:
        increments, saturation_bound_receipt = _saturation_bound(
            prior, increments, states, indices)
    _stage("saturation_bound")

    number_receipt = None
    if cfg.field_rules == "design" and cfg.number_rediagnosis != "off":
        increments, number_receipt = _fr.rediagnose_numbers(
            materialized(prior), increments, states, indices,
            mode=cfg.number_rediagnosis, mp_physics=cfg.mp_physics)
        output_fields = output_fields + tuple(
            name for name in number_receipt["species"]
            if name not in output_fields)
    _stage("number_rediagnosis")

    def restaggered(slot):
        index = indices[slot]
        member: dict[str, np.ndarray] = {}
        for name in output_fields:
            mass_inc = np.asarray(increments[name][slot])
            if name in _RESTAGGER:
                member[name] = _RESTAGGER[name](mass_inc)
            else:
                member[name] = mass_inc
            expected = tuple(np.shape(states[index][name]))
            if member[name].shape != expected:
                raise RadarAssimilationError(
                    f"member {index} increment for {name!r} came out "
                    f"{member[name].shape}, checkpoint field is {expected}")
        return member

    increments_by_member: dict[int, dict[str, np.ndarray]] = dict(zip(
        indices, _by_member(restaggered, range(len(indices)))))
    _stage("restaggered")

    provenance = {
        "schema": METHOD_SCHEMA,
        "stability": "experimental",
        "method": "LETKF (Hunt, Kostelich & Szunyogh 2007 sec 2.3)",
        "analysis_fields": list(cfg.analysis_fields),
        "wind_fields_restaggered": [name for name in cfg.analysis_fields
                                    if name in _RESTAGGER],
        "wind_frame": "grid-relative; H(x) rotates with the grid "
                      "projection's own SINALPHA/COSALPHA",
        "fall_speed": cfg.fall_speed,
        "kind_field_rules": (field_rules_receipt if cfg.kind_field_rules
                             else {"enabled": False}),
        "localization_horizontal_m": float(cfg.localization.horizontal_m),
        "localization_vertical_m": float(cfg.localization.vertical_m),
        "rtps_alpha": float(cfg.rtps_alpha),
        "relaxation": cfg.relaxation,
        "prior_inflation": float(cfg.prior_inflation),
        "spread_repair": (spread_controller.last_receipt
                          if spread_controller is not None else None),
        "velocity_thinning": thinning_receipt,
        "velocity_error_inflation": float(cfg.velocity_error_inflation),
        "reflectivity_thinning": z_thinning_receipt,
        "reflectivity_echo_conditioning": reflectivity_echo_receipt,
        "radar_field_rules": field_rule_receipt,
        "number_rediagnosis": number_receipt,
        "radar_classes": radar_class_receipt,
        "reflectivity_error_inflation": float(
            cfg.reflectivity_error_inflation),
        "clear_air": {
            "enabled": bool(cfg.clear_air),
            "thinning": z0_thinning_receipt,
            "error_inflation": float(cfg.clear_air_error_inflation),
            "value_dbz": clear_air_value,
            "value_source": (
                None if not cfg.clear_air
                else ("explicit" if cfg.clear_air_value_dbz is not None
                      else f"mp_physics={cfg.mp_physics} H(x) floor")),
        },
        "cwp_assimilated": bool(cfg.cwp),
        "extra_observation_batches": len(extra_batches),
        "analysis_sources": list(analysis_sources(
            cfg, extra_batches=len(extra_batches))),
        "cwp_thinning": cwp_thinning_receipt,
        "cwp_error_inflation": float(cfg.cwp_error_inflation),
        "cwp_localization_horizontal_m": (
            None if cfg.cwp_localization is None
            else float(cfg.cwp_localization.horizontal_m)),
        "cwp_localization_vertical_m": (
            None if cfg.cwp_localization is None
            else float(cfg.cwp_localization.vertical_m)),
        "cwp_observations": cwp_provenance,
        "moment_policy": moment_receipt,
        "positivity": positivity_receipt,
        "saturation_bound": saturation_bound_receipt,
        "velocity_dispersion": dispersion_receipt,
        # What was ASKED for, and what RAN.  The two differ whenever the
        # default "auto" is in effect, and a receipt that recorded only
        # the request could not tell a cycle that used the card from one
        # that fell back to numpy on a box whose cupy was broken -- which
        # is a 20x difference in the leg's wall clock and a different set
        # of last bits in the increments.
        "solve_device": cfg.solve_device,
        "solve_device_resolved": solve_device,
        "analysis_storage": storage,
        "analysis_attempts": attempts,
        "solve_timing_scope": "successful attempt; failed attempt wall time is recorded separately",
        "analysis_execution": execution_receipt,
        "solve_device_reason": solve_device_reason,
        # Host-to-device staging of the prior and the observation batches,
        # and the copy of the increments back.  Both are exactly zero on
        # the host arm, which is what makes them comparable.
        "solve_stage_seconds": round(float(stage_seconds), 3),
        # Wall clock of each stage of this call, in run order; "filter" is
        # the whole storage attempt the filter split above belongs to.
        "stage_wall_seconds": dict(stage_clock),
        "solve_unstage_seconds": round(float(unstage_seconds), 3),
        "members": len(indices),
        "checkpoints": {int(index): checkpoint_label(checkpoints[index],
                                                     index)
                        for index in indices},
        "observations": adapter_provenance,
        "extra_observations": (extra_obs_provenance if extra_batches
                               else None),
        "innovations": innovations,
        "filter": {
            "active_points": int(diagnostics.active_points),
            "total_points": int(diagnostics.total_points),
            "max_local_obs": int(diagnostics.max_local_obs),
            "batches": int(diagnostics.batches),
            "host_staging": bool(getattr(diagnostics, "host_staging", False)),
            # How many processes ran a host solve's chunk loop and why
            # (gpuwm.da.letkf_host_parallel); 1 on a device solve.
            "host_workers": int(getattr(diagnostics, "host_workers", 1)),
            "host_workers_reason": str(
                getattr(diagnostics, "host_workers_reason", "")),
            "host_row_blocks": int(getattr(diagnostics, "host_row_blocks", 0)),
            "host_transform_pieces": int(
                getattr(diagnostics, "host_transform_pieces", 0)),
            "host_geometry_bytes_per_point": int(getattr(diagnostics, "host_geometry_bytes_per_point", 0)),
            "device_chunks": int(getattr(diagnostics, "device_chunks", 0)),
            "staging_bytes": int(getattr(diagnostics, "staging_bytes", 0)),
            "staging_peak_bytes": int(getattr(diagnostics, "staging_peak_bytes", 0)),
            "geometry_evaluations": int(getattr(diagnostics, "geometry_evaluations", 0)),
            "geometry_reuses": int(getattr(diagnostics, "geometry_reuses", 0)),
            "driver_free_bytes": getattr(diagnostics, "driver_free_bytes", None),
            "pool_reusable_bytes": getattr(diagnostics, "pool_reusable_bytes", None),
            "chunk_points": int(diagnostics.chunk_points),
            # The chunk the sizing chose up front, and how many times a
            # device allocation failure halved it mid-analysis.  Equal
            # initial/final with 0 shrinks is the healthy case; a nonzero
            # shrink count means the analysis survived on the degradation
            # path and the memory model under-read the card that ran it.
            "chunk_points_initial": int(
                getattr(diagnostics, "chunk_points_initial", 0)),
            "chunk_oom_shrinks": int(
                getattr(diagnostics, "chunk_oom_shrinks", 0)),
            "solve_bytes_per_point": int(
                getattr(diagnostics, "solve_bytes_per_point", 0)),
            # WHERE the analysis spent its wall clock, split three ways by
            # gpuwm.da.letkf.analyze.  A cycle report already recorded that
            # the analysis was most of the leg; without this split it could
            # not say whether that was the eigensolve (which runs at the
            # ACTIVE points) or the localisation weighting (which runs at
            # EVERY point), and those two have different remedies.
            "setup_seconds": round(float(
                getattr(diagnostics, "setup_seconds", 0.0)), 3),
            "solve_seconds": round(float(
                getattr(diagnostics, "solve_seconds", 0.0)), 3),
            "finish_seconds": round(float(
                getattr(diagnostics, "finish_seconds", 0.0)), 3),
            # And the split INSIDE the chunk loop: localisation weights at
            # every gridpoint, versus the transform at the active ones.
            # This is the pair that says whether a faster eigensolver
            # could have helped this analysis at all.
            "weights_seconds": round(float(
                getattr(diagnostics, "weights_seconds", 0.0)), 3),
            "transform_seconds": round(float(
                getattr(diagnostics, "transform_seconds", 0.0)), 3),
            # Device bytes in use against pool bytes held, at the end of
            # the chunk loop.  The GAP is the part of the card the LETKF
            # budget does not govern and the pool cap cannot reclaim: the
            # whole-domain arrays the budget excludes by design, plus
            # anything allocated outside the pool.  It is the term that
            # explains a RAW cudaErrorMemoryAllocation, which cupy cannot
            # raise -- cupy releases and retries before it gives up -- so
            # a card that dies with one died on memory this number counts
            # and the budget does not.
            "device_used_mib": round(float(
                getattr(diagnostics, "device_used_mib", 0.0)), 1),
            "pool_total_mib": round(float(
                getattr(diagnostics, "pool_total_mib", 0.0)), 1),
            "device_pool_gap_mib": round(float(
                getattr(diagnostics, "device_pool_gap_mib", 0.0)), 1),
            "relaxation": str(getattr(diagnostics, "relaxation", "rtps")),
            "rtps_alpha": float(getattr(diagnostics, "rtps_alpha", 0.0)),
            # WHICH eigensolver factored the R x R matrix, and how hard it
            # had to work.  This is not decoration.  The bundled Jacobi
            # kernel became the default the moment it landed, so every
            # --solve-device cuda run silently stopped calling cuSOLVER;
            # the two agree to ~1e-11 relative, NOT bitwise.  Every receipt
            # banked before that change was produced by the library solver
            # and will not reproduce byte-for-byte under the default today.
            # Without these two keys a cycle report cannot say which side
            # of that line it sits on, and "reproduce this receipt" becomes
            # a claim about a setting that has no CLI knob.
            #
            # max_jacobi_sweeps is the early warning that goes with it:
            # climbing toward gpuwm.core.jacobi_eigh.SWEEP_CAP means the
            # localised matrix is worse conditioned than expected.  0 under
            # the library solver.
            "eigensolver": str(getattr(diagnostics, "eigensolver", "")),
            # Which neighbour search found the localised observations, and
            # the roster's own receipt when it was the once-built index.
            "neighbor_search": str(getattr(diagnostics, "neighbor_search", "")),
            "neighbor_index": dict(getattr(diagnostics, "neighbor_index", {}) or {}),
            "max_jacobi_sweeps": int(
                getattr(diagnostics, "max_jacobi_sweeps", 0)),
            "mean_increment_rms": {
                name: float(value) for name, value
                in getattr(diagnostics, "mean_increment_rms", {}).items()},
            # PRIOR spread beside the posterior, per field.  Without the
            # pair a cycling run cannot tell an analysis that is
            # over-confident from a forecast step that is losing spread on
            # its own: the posterior alone falls in both cases and says
            # nothing about which.  The ratio posterior/prior at one leg,
            # against prior[leg+1]/posterior[leg] across the forecast, is
            # exactly the split, and it is what decides whether the
            # answer is a bigger alpha or additive inflation.
            "prior_spread": {
                name: float(value) for name, value
                in getattr(diagnostics, "prior_spread", {}).items()},
            "posterior_spread": {
                name: float(value) for name, value
                in getattr(diagnostics, "posterior_spread", {}).items()},
        },
    }
    if precip_receipt is not None:
        # Only when the stage ran: with it off the provenance is the bytes
        # it always was.
        provenance["precip_analysis"] = precip_receipt
    return increments_by_member, provenance


# ---------------------------------------------------------------------------
# the seam callable
# ---------------------------------------------------------------------------


def make_assimilate(observations, grid, cfg: RadarAssimilationConfig, *,
                    reflectivity_provider=None) -> Callable:
    """Bind observations to the cycle driver's assimilation seam.

    ``observations`` is either one ``gpuwm-obs.radar-grid.v1`` path (every
    cycle assimilates the same file -- a single-analysis run), a mapping
    ``{cycle_index: path}``, or a callable ``cycle_index -> path``.  A
    cycle the mapping does not name is a refusal, not a silent forecast-
    only leg: the driver calls ``assimilate`` on EVERY cycle it runs, so
    an operator who wants unassimilated legs must say so by running them
    with ``assimilate=None``, where the manifest records the seam as
    deliberately empty.

    Returns ``assimilate(cycle_index, member_states) -> (increments,
    provenance)`` in exactly the shape
    :func:`gpuwm.ensemble.cycle.run_cycles` consumes.
    """
    if callable(observations):
        resolve = observations
    elif isinstance(observations, Mapping):
        table = {int(key): value for key, value in observations.items()}

        def resolve(cycle_index: int):
            if cycle_index not in table:
                raise RadarAssimilationError(
                    f"no observation file is bound to cycle {cycle_index}; "
                    f"bound cycles: {sorted(table)}. Run unassimilated "
                    "legs with assimilate=None rather than leaving a "
                    "cycle to guess")
            return table[cycle_index]
    else:
        path = observations

        def resolve(cycle_index: int):
            return path

    def assimilate(cycle_index: int, member_states: Mapping[int, Mapping]):
        checkpoints = {
            int(index): member_background_checkpoint(info["member_dir"])
            for index, info in member_states.items()}
        increments, provenance = assimilate_radar_grid(
            checkpoints, resolve(cycle_index), grid, cfg,
            reflectivity_provider=reflectivity_provider)
        provenance["cycle"] = int(cycle_index)
        return increments, provenance

    return assimilate
