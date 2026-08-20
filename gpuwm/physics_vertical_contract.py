"""Lightweight first-call vertical bounds for physics front doors.

This module deliberately lives at package top level and imports no forecast
executor.  The standalone preprocessing distribution needs to reject an
impossible vertical grid without staging CUDA-backed component modules.
Component tests bind these values and layer-count helpers to their runtime
counterparts.
"""

from __future__ import annotations

import math

import numpy as np


class PhysicsVerticalPreflightError(ValueError):
    """Resolved physics cannot execute on the requested vertical grid.

    Lives here rather than beside the preflight because the LAUNCHERS raise
    it too: the preparation-time gate and the first-call rejection inside a
    physics module are the same refusal met at two moments, and a user who
    reaches the second one has already paid for fetch and preparation.  A
    ``ValueError`` subclass so every existing ``except ValueError`` around a
    launcher keeps working.
    """


def vertical_bounds_wording(bounds: tuple[int | None, int | None]) -> str:
    """The one spelling of a level-count bound, shared by every refusal."""

    minimum, maximum = bounds
    if minimum is None and maximum is None:
        raise ValueError("a vertical bound must constrain at least one end")
    if minimum is None:
        return f"nz <= {maximum}"
    if maximum is None:
        return f"nz >= {minimum}"
    return f"{minimum} <= nz <= {maximum}"


def outside_vertical_bounds(nz: int,
                            bounds: tuple[int | None, int | None]) -> bool:
    """True when ``nz`` violates either end of a component's bound."""

    minimum, maximum = bounds
    return ((minimum is not None and nz < minimum)
            or (maximum is not None and nz > maximum))


def refuse_vertical_levels(label: str,
                           bounds: tuple[int | None, int | None],
                           nz: int, *, breakage: str,
                           ) -> PhysicsVerticalPreflightError:
    """Build the launcher-side refusal in the preflight's own grammar.

    ``breakage`` names what would go wrong and what to do instead, because a
    refusal that only restates the number tells a user nothing they can act
    on.  The sentence before it is byte-identical to the line
    ``validate_resolved_physics_vertical_levels`` emits, so the two surfaces
    read as one gate rather than two.
    """

    return PhysicsVerticalPreflightError(
        f"{label} requires {vertical_bounds_wording(bounds)}, got nz={nz}: "
        f"{breakage}  `gpuwm check` refuses this configuration before a run "
        "starts; reaching it here means the preparation gate was bypassed.")


MYNN_VERTICAL_LEVEL_BOUNDS = (5, None)
#: YSU.  The ceiling is ``YSU_KMAX`` in ``kernels/ysu.cu``: one CUDA thread
#: owns a whole column and holds it in per-thread local memory at that fixed
#: depth.  The floor is the launcher's own ``nz >= 4`` -- the counter-gradient
#: and entrainment layers the scheme indexes need four levels to exist.
YSU_VERTICAL_LEVEL_BOUNDS = (4, 128)
#: Shin-Hong, the same pair for the same two reasons (``SHINHONG_KMAX``, and
#: the scheme shares YSU's column indexing).  Held as its own name rather
#: than aliased to YSU's: the two kernels are separate translation units a
#: future change could size differently.
SHINHONG_VERTICAL_LEVEL_BOUNDS = (4, 128)
#: MYJ.  ``MYJ_KMAX`` in ``kernels/myjpbl.cu`` for the ceiling; the floor is
#: where VDIFQ's tridiagonal chain still has interior rows and MIXLEN's
#: profile smoothing can read ``REL(K-1)``/``REL(K+1)``.
MYJ_VERTICAL_LEVEL_BOUNDS = (4, 128)
#: SASE.  ``SASE_KMAX`` in ``kernels/sase.cu``, which is the same number as
#: ``gpuwm.core.sase_limits.MAX_COLUMN_LEVELS`` and ``gpuwm.config
#: .SASE_MAX_NZ``: the implicit vertical solve keeps three FP64 columns of
#: that depth in per-thread local memory.  The floor is where the solve stops
#: being an identity -- one interior face needs two levels.
SASE_VERTICAL_LEVEL_BOUNDS = (2, 128)
KF_VERTICAL_LEVEL_BOUNDS = (8, 128)
# Grell-Freitas: the inversion-layer search clamps kend to ktf-8 and its
# second-derivative stencil then reaches ktf-1, so a column shorter than 12
# levels leaves no search window at all; the certified fixture ran nz=40 and
# the kernel recompiles its level bound above GF_KMAX=40 through the
# integer-define tier, so there is no fixed upper bound to declare.
GF_VERTICAL_LEVEL_BOUNDS = (12, None)
KESSLER_VERTICAL_LEVEL_BOUNDS = (None, 256)
WSM6_VERTICAL_LEVEL_BOUNDS = (2, 80)
THOMPSON_VERTICAL_LEVEL_BOUNDS = (2, 256)
#: Aerosol-aware Thompson (mp_physics=28).  Identical to the classic
#: Thompson bound and NOT a copy of a number: the mp=28 sedimentation
#: translation unit declares the same ``THOMPSON_AA_KMAX_GENERIC 256``
#: per-thread column bound as ``kernels/thompson.cu``, and the runtime
#: counterpart ``gpuwm.core.thompson_aerosol.VERTICAL_LEVEL_BOUNDS`` is
#: bound to this constant by
#: ``tests/test_mp28_runnable.py::test_mp28_vertical_bounds_are_the_runtime_bounds``
#: so the two cannot drift.  Held as its own name rather than aliased to
#: THOMPSON_VERTICAL_LEVEL_BOUNDS because the two schemes' kernels are
#: separate translation units that a future change could size differently.
THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS = (2, 256)
MORRISON_VERTICAL_LEVEL_BOUNDS = (2, 256)
NSSL2_VERTICAL_LEVEL_BOUNDS = (3, 256)
#: The dynamical core's own vertical bound, from the top ``WPHI_MAX_LEV``
#: tier ``gpuwm.core.acoustic`` compiles the implicit w''-phi'' solve at.
#: Bound to the launcher's ladder by
#: ``tests/test_acoustic_nz_tiers.py::test_the_contract_bound_is_the_launcher_bound``.
ACOUSTIC_VERTICAL_LEVEL_BOUNDS = (1, 256)

RRTMGP_TOA_PRESSURE_PA = 1.005183574463
WRF_LW_UPPER_DELTA_P_PA = 400.0
MAX_RRTMGP_LAYERS = 128
MAX_LEGACY_LONGWAVE_LAYERS = 128
MAX_LEGACY_SHORTWAVE_LAYERS = 64


def rrtmgp_above_model_layer_counts(
        p_top: float, *,
        pressure_floor: float = RRTMGP_TOA_PRESSURE_PA,
) -> tuple[int, int]:
    """Return WRF's RTE+RRTMGP ``(LW, SW)`` cap-layer counts."""

    if (not math.isfinite(p_top) or not math.isfinite(pressure_floor)
            or p_top < 0.0 or pressure_floor <= 0.0):
        raise ValueError("radiation top pressures must be finite and nonnegative")
    if p_top <= pressure_floor:
        return 0, 0
    lw = int(np.floor(p_top / WRF_LW_UPPER_DELTA_P_PA + 0.5))
    sw = int(0.5 * p_top >= pressure_floor)
    return max(0, lw), sw


def legacy_radiation_layer_counts(
        nz: int, p_top: float,
) -> tuple[int, int]:
    """Return exact first-call WRF legacy-RRTMG ``(LW, SW)`` counts."""

    value = np.float32(
        np.float32(np.float32(p_top) * np.float32("0.01"))
        / np.float32("4.0"))
    raw = float(value)
    rounded = int(raw)
    if raw - rounded >= 0.5:
        rounded += 1
    elif raw - rounded <= -0.5:
        rounded -= 1
    return int(nz) + rounded, int(nz) + 1
