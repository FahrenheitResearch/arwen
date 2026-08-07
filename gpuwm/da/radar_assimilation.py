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

Three decisions here are load-bearing:

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

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from gpuwm.da.letkf import (RELAXATION_MODES, GriddedObs, LetkfConfig,
                            LetkfDiagnostics, Localization, analyze)
from gpuwm.da.moments import (MOMENT_POLICIES, DEFAULT_MOMENT_POLICY,
                              validate_analysis_fields)
from gpuwm.da.obs_goes import goes_grid_to_gridded_obs
from gpuwm.da.obs_radar import (Z_SOURCES, beam_unit_vectors,
                                letkf_grid_geometry, radar_grid_to_gridded_obs,
                                read_document, simulated_radial_velocity)
from gpuwm.da.obsop import (clear_air_floor_dbz, destagger_u, destagger_v,
                            destagger_w, earth_relative_winds,
                            precipitating_activity_mask,
                            reflectivity_fall_speed)
from gpuwm.da.positivity import (NON_NEGATIVE_FIELDS, POLICIES,
                                 apply_positivity, constrained_fields,
                                 verify_non_negative)

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


class RadarAssimilationError(ValueError):
    """The observations, the checkpoints and the config cannot be
    reconciled.  Never a warning."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


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
    #: Assimilate the per-radar radial-velocity batches.
    velocity: bool = True
    #: Assimilate the merged reflectivity batch.  Needs a provider.
    reflectivity: bool = False
    #: Which reflectivity reduction to difference against; see
    #: :data:`gpuwm.da.obs_radar.Z_SOURCES` for why ``z_mean`` is absent.
    z_source: str = "z_obs"
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
    #: route; an explicit number is honoured and recorded, and is refused
    #: unless it is finite.
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
    #: which is weaker but never wrong about a state it can see.
    mp_physics: int | None = None
    #: "host" solves on numpy; "cuda" moves the prior and the observation
    #: batches to CuPy for the batched LETKF and brings increments back.
    solve_device: str = "host"
    #: Threaded to :class:`gpuwm.da.letkf.LetkfConfig` unchanged.
    solve_dtype: str = "float64"
    #: Also threaded unchanged; see
    #: :data:`gpuwm.da.letkf.EIGENSOLVER_MODES`.  The default needs no
    #: linear-algebra library on either device, which is the whole reason
    #: it is a setting rather than a fact.
    eigensolver: str = "auto"
    memory_budget_mib: float = 512.0
    chunk_points: int | None = None

    def __post_init__(self) -> None:
        if not self.analysis_fields:
            raise RadarAssimilationError(
                "analysis_fields is empty: an analysis that updates nothing "
                "is a bug, not a configuration")
        if len(set(self.analysis_fields)) != len(self.analysis_fields):
            raise RadarAssimilationError(
                f"analysis_fields has duplicates: {self.analysis_fields!r}")
        if not (self.velocity or self.reflectivity or self.clear_air
                or self.cwp):
            raise RadarAssimilationError(
                "none of velocity, reflectivity, clear_air or cwp is "
                "enabled, so this "
                "config would assimilate nothing. An intentional "
                "no-observation cycle is a run_cycles call with "
                "assimilate=None, not an empty analysis here")
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
                    "negative mixing ratio, and clip / reject / none are "
                    "not equivalent -- clipping at zero ADDS mass and is "
                    "biased wetward, rejecting conserves the background and "
                    "invents gradients, and none lets the microphysics meet "
                    f"the negatives. Choose one of {POLICIES}; "
                    "gpuwm.da.positivity documents what each costs")
        elif self.positivity_policy not in POLICIES:
            raise RadarAssimilationError(
                f"positivity_policy must be one of {POLICIES} or None, got "
                f"{self.positivity_policy!r}")
        if self.clear_air:
            if self.clear_air_value_dbz is not None:
                floor = float(self.clear_air_value_dbz)
                if not np.isfinite(floor):
                    raise RadarAssimilationError(
                        f"clear_air_value_dbz is "
                        f"{self.clear_air_value_dbz!r}; it must be a finite "
                        "dBZ value or None to derive it from mp_physics")
            elif self.mp_physics is None:
                raise RadarAssimilationError(
                    "clear_air is enabled with neither clear_air_value_dbz "
                    "nor mp_physics. A clear-air observation is differenced "
                    "against H(x), so it must carry the ACTIVE scheme's "
                    "clear-air floor -- -35 dBZ for the refl10cm family, "
                    "0 dBZ for NSSL mp18. With neither the scheme nor the "
                    "value stated there is nothing to derive it from, and "
                    "the wrong floor is silent: two agreeing clear skies "
                    "produce a 35 dB innovation and the analysis removes "
                    "condensate to chase it")
            else:
                # Raises for a scheme whose floor nobody has read, here at
                # config time rather than mid-cycle.
                clear_air_floor_dbz(int(self.mp_physics))
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
        if self.solve_device not in ("host", "cuda"):
            raise RadarAssimilationError(
                f"solve_device must be 'host' or 'cuda', got "
                f"{self.solve_device!r}")
        object.__setattr__(self, "analysis_fields",
                           tuple(self.analysis_fields))
        if self.radars is not None:
            object.__setattr__(self, "radars",
                               tuple(str(r) for r in self.radars))


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
    return np.asarray(reflectivity_fall_speed(
        np.asarray(dbz, dtype=np.float64), pressure, active,
        surface_pressure=pressure[0]))


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


def innovation_summary(batches: Sequence[GriddedObs]) -> list[dict]:
    """Observation-space statistics per batch, JSON-serialisable.

    ``d = y - mean_k H(x_k)`` over the masked points: the number a real-
    data run is judged by before any increment exists.  Everything here is
    computed from the batch the filter itself consumes, so what is
    reported is what was assimilated, not a parallel reading of the file.
    """
    out = []
    for batch in batches:
        mask = np.asarray(batch.mask, dtype=bool)
        n = int(np.count_nonzero(mask))
        entry: dict = {"name": batch.name, "observations": n}
        if n:
            y = np.asarray(batch.values, dtype=np.float64)[mask]
            sim = np.asarray(batch.simulated, dtype=np.float64)[:, mask]
            err = np.asarray(batch.errors, dtype=np.float64)
            err = (np.full(y.shape, float(err)) if err.ndim == 0
                   else err[mask])
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


def assimilate_radar_grid(checkpoints: Mapping[int, str | Path],
                          observations, grid,
                          cfg: RadarAssimilationConfig, *,
                          reflectivity_provider=None,
                          extra_obs=None,
                          extra_obs_provenance=None,
                          cwp_observations=None,
                          cwp_provider=None,
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
        the file the driver will add the increments to.
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

    Returns
    -------
    ``(increments_by_member, provenance)`` in exactly the shape the cycle
    driver's seam consumes: ``{member_index: {field: ndarray}}`` with
    every array matching the CHECKPOINT shape of its field (winds on
    faces), and a JSON-serialisable method-provenance mapping carrying
    the adapter provenance, the observation-space innovation statistics
    and the filter diagnostics.
    """
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

    needs_radar = cfg.velocity or cfg.reflectivity or cfg.clear_air
    document = None
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

    indices = sorted(int(index) for index in checkpoints)
    states = {index: read_checkpoint_state(checkpoints[index])
              for index in indices}

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
    prior = {}
    for name in cfg.analysis_fields:
        stack = []
        for index in indices:
            field = _mass_field(name, states[index],
                                f"member {index} checkpoint")
            if field.shape != shape:
                raise RadarAssimilationError(
                    f"member {index} field {name!r} is {field.shape} at "
                    f"mass points but the observation file's grid is "
                    f"{shape}; these checkpoints are not from this domain")
            stack.append(field)
        prior[name] = np.stack(stack)

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
        clear_air_simulated = np.stack(
            [dbz_by_member[index] for index in indices])
        clear_air_value = (
            float(cfg.clear_air_value_dbz)
            if cfg.clear_air_value_dbz is not None
            else clear_air_floor_dbz(int(cfg.mp_physics)))

    velocity_simulated = None
    if cfg.velocity:
        winds = {}
        for index in indices:
            u_e, v_n, w_m = member_earth_winds(
                states[index], rotation, where=f"member {index} checkpoint")
            if cfg.fall_speed == "reflectivity":
                w_m = w_m - _member_fall_speed(
                    states[index], dbz_by_member[index],
                    where=f"member {index} checkpoint")
            winds[index] = (u_e, v_n, w_m)

        def velocity_simulated(radar_index, radar):
            beam = beam_unit_vectors(document, radar_index)
            return np.stack([
                simulated_radial_velocity(*winds[index], beam)
                for index in indices])

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

    extra_batches = list(extra_obs) if extra_obs else []
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

    innovations = innovation_summary(batches)

    # -- the filter ----------------------------------------------------------
    if diagnostics is None:
        diagnostics = LetkfDiagnostics()
    letkf_cfg = LetkfConfig(
        localization=cfg.localization,
        analysis_fields=tuple(cfg.analysis_fields),
        rtps_alpha=cfg.rtps_alpha,
        prior_inflation=cfg.prior_inflation,
        relaxation=cfg.relaxation,
        chunk_points=cfg.chunk_points,
        memory_budget_mib=cfg.memory_budget_mib,
        solve_dtype=cfg.solve_dtype,
        eigensolver=cfg.eigensolver)
    solve_prior, solve_batches = prior, batches
    if cfg.solve_device == "cuda":
        import cupy as cp  # noqa: PLC0415

        solve_prior = {name: cp.asarray(values)
                       for name, values in prior.items()}
        solve_batches = [
            GriddedObs(name=batch.name,
                       values=cp.asarray(np.asarray(batch.values,
                                                    np.float64)),
                       errors=cp.asarray(np.asarray(batch.errors,
                                                    np.float64)),
                       simulated=cp.asarray(np.asarray(batch.simulated,
                                                       np.float64)),
                       mask=cp.asarray(np.asarray(batch.mask, bool)),
                       localization=batch.localization)
            for batch in batches]
    increments = analyze(solve_prior, solve_batches,
                         letkf_grid_geometry(grid), letkf_cfg, diagnostics)
    if cfg.solve_device == "cuda":
        import cupy as cp  # noqa: PLC0415

        increments = {name: cp.asnumpy(values)
                      for name, values in increments.items()}

    # -- positivity ----------------------------------------------------------
    # On the MASS-POINT increments, before restaggering, because the
    # constraint is "prior + increment >= 0" and the prior it must be
    # evaluated against is the mass-point prior the filter analysed.
    # Doing it after the wind restagger would be evaluating a constraint
    # against a field that no longer lines up with it -- and every
    # constrained field is mass-shaped anyway, so nothing is lost.
    positivity_receipt = None
    if cfg.positivity_policy is not None:
        increments, positivity_receipt = apply_positivity(
            prior, increments, policy=cfg.positivity_policy,
            fields=tuple(cfg.analysis_fields))
        if cfg.positivity_policy in ("clip", "reject"):
            # The post-condition that catches a policy applied to the
            # wrong mapping: a receipt claiming N clipped points beside
            # increments that were never clipped.
            verify_non_negative(prior, increments,
                                fields=tuple(cfg.analysis_fields))

    # -- back to the checkpoint vocabulary -----------------------------------
    increments_by_member: dict[int, dict[str, np.ndarray]] = {}
    for slot, index in enumerate(indices):
        member: dict[str, np.ndarray] = {}
        for name in cfg.analysis_fields:
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
        increments_by_member[index] = member

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
        "localization_horizontal_m": float(cfg.localization.horizontal_m),
        "localization_vertical_m": float(cfg.localization.vertical_m),
        "rtps_alpha": float(cfg.rtps_alpha),
        "relaxation": cfg.relaxation,
        "prior_inflation": float(cfg.prior_inflation),
        "velocity_thinning": thinning_receipt,
        "velocity_error_inflation": float(cfg.velocity_error_inflation),
        "reflectivity_thinning": z_thinning_receipt,
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
        "solve_device": cfg.solve_device,
        "members": len(indices),
        "checkpoints": {int(index): Path(checkpoints[index]).name
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
