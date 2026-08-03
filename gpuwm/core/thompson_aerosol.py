"""Launcher facade for WRF v4.6.1 Thompson aerosol-aware (``mp_physics=28``).

This module is **re-exports only**.  It contains no arithmetic, no argument
marshalling and no state: every name below is bound directly from the module
that owns it, so ``from gpuwm.core.thompson_aerosol import X`` and
``from gpuwm.core.thompson_aerosol_<owner> import X`` are the same object.

Why it exists
-------------
``gpuwm/core/thompson.py`` is a single module that owns every classic
(``mp_physics=8``) launcher, and ``gpuwm/core/microphysics.py::_apply_thompson``
imports from exactly one place.  mp=28's launchers are spread over six owner
modules because six agents built them in parallel against six separate CUDA
translation units (MP28_PORT_SPEC.md, "Strategy").  Without a facade the
forecast adapter would have to name all six, and the *set* of modules would
become an implicit part of the adapter's contract -- so moving one launcher
between owners would edit the adapter, which is exactly the coupling the
per-file ownership rule exists to prevent.

What it deliberately does NOT do
--------------------------------
* It does not re-export anything from :mod:`gpuwm.core.thompson`.  The mp=28
  adapter calls four classic launchers (graupel-number init/finalize, the two
  column-mask helpers) and four classic sedimentation launchers **by their own
  names, from their own module**, so a reader of the adapter can see at a
  glance which lines are frozen mp=8 code being reused unchanged.  Aliasing
  them through here would hide that.
* It does not re-export the ``probe_*`` diagnostics.  Those are test
  instruments; a forecast path that can reach one is a forecast path that can
  accidentally depend on one.
* Importing it compiles nothing, uploads nothing, and does not make
  ``mp_physics=28`` selectable.

The tuples at the bottom are the published inventory.  ``AEROSOL_LAUNCHERS``
is what ``tests/test_thompson_aerosol_adapter.py`` uses to install call
recorders, so a launcher that is not listed here cannot be pinned by the
call-order gate.
"""

from __future__ import annotations

from gpuwm.core.thompson_aerosol_cold import (
    launch_aa_cold_network,
    launch_aa_cold_network_from_owner,
)
from gpuwm.core.thompson_aerosol_launch import (
    AEROSOL_COMMON_HEADER,
    AEROSOL_KERNEL_MODULES,
    CCN_ACTIVATION_SHAPE,
    CLASSIC_MODULE,
    COLD_MODULE,
    DEFAULT_THREADS,
    PROBE_MODULE,
    SAT_MODULE,
    SED_MODULE,
    STATE_MODULE,
    WARM_MODULE,
    launch_grid,
    validate_fields,
    validate_fp64_fortran_table,
    validate_int_fields,
)
from gpuwm.core.thompson_aerosol_runtime import (
    device_drop_evaporation_number_table,
    load_aerosol_device_tables,
)
from gpuwm.core.thompson_aerosol_sat import (
    launch_aerosol_rain_evaporation,
    launch_aerosol_saturation_adjust,
)
from gpuwm.core.thompson_aerosol_sed import (
    VERTICAL_LEVEL_BOUNDS,
    launch_aa_cloud_sedimentation,
    launch_aa_final_phase_cleanup,
)
from gpuwm.core.thompson_aerosol_state import (
    AEROSOL_CEILING,
    INIT_PROFILE_HEIGHT_FIELD,
    NC_FLOOR_M3,
    NIFA_FLOOR,
    NT_C_MAX,
    NWFA_FLOOR,
    PROFILE_FILL_EPS,
    R1,
    aerosol_profile_needs_fill,
    launch_aerosol_effective_radius,
    launch_aerosol_entry_cloud_number,
    launch_aerosol_entry_snapshot,
    launch_aerosol_init_profile,
    launch_aerosol_state_finalize,
    launch_aerosol_surface_emission,
    launch_aerosol_working_cloud,
    launch_aerosol_working_number,
    launch_tau1_density,
    zero_aerosol_accumulators,
)
from gpuwm.core.thompson_aerosol_warm import (
    launch_aerosol_warm_source_network,
    launch_aerosol_warm_source_network_from_owner,
    launch_ncten_balance,
)

#: Every device launcher mp=28 owns, grouped by the module that owns it.
#: The forecast adapter calls a subset; the rest are reachable for focused
#: gates.  ``tests/test_thompson_aerosol_adapter.py`` installs a recorder on
#: each of these names in its owner module.
AEROSOL_LAUNCHERS: dict[str, tuple[str, ...]] = {
    "gpuwm.core.thompson_aerosol_state": (
        # Not a device launch, but part of the same contract and recorded by
        # the same gate: the accumulator zeroing is a call the adapter must
        # make, in a position the order test pins.
        "zero_aerosol_accumulators",
        "launch_aerosol_entry_snapshot",
        "launch_aerosol_entry_cloud_number",
        "launch_tau1_density",
        "launch_aerosol_working_number",
        "launch_aerosol_working_cloud",
        "launch_aerosol_state_finalize",
        "launch_aerosol_surface_emission",
        "launch_aerosol_init_profile",
        "launch_aerosol_effective_radius",
    ),
    "gpuwm.core.thompson_aerosol_cold": (
        "launch_aa_cold_network",
        "launch_aa_cold_network_from_owner",
    ),
    "gpuwm.core.thompson_aerosol_warm": (
        "launch_aerosol_warm_source_network",
        "launch_aerosol_warm_source_network_from_owner",
        "launch_ncten_balance",
    ),
    "gpuwm.core.thompson_aerosol_sat": (
        "launch_aerosol_saturation_adjust",
        "launch_aerosol_rain_evaporation",
    ),
    "gpuwm.core.thompson_aerosol_sed": (
        "launch_aa_cloud_sedimentation",
        "launch_aa_final_phase_cleanup",
    ),
}

#: The classic (frozen mp=8) launchers the mp=28 adapter reuses UNCHANGED.
#: Listed here for the call-order gate only; they are NOT re-exported.
#:
#: Every name is verified above to contain no aerosol reference:
#: module_mp_thompson.F:3790-3936 (the four reused fallout blocks) has no
#: ``is_aerosol_aware`` branch and no nc/nwfa/nifa reference at all, and the
#: classic graupel-number diagnostic is identical because ``is_hail_aware``
#: is false for mp=8 and mp=28 alike.
REUSED_CLASSIC_LAUNCHERS: tuple[str, ...] = (
    "launch_classic_graupel_number_init",
    "launch_classic_graupel_number_finalize",
    "launch_hydrometeor_column_mask",
    "launch_graupel_fallout_column_mask",
    "launch_ice_sedimentation",
    "launch_snow_sedimentation",
    "launch_graupel_sedimentation",
    "launch_rain_sedimentation",
)

#: Classic launchers that carry WRF's **Cooper (1986)** deposition nucleation
#: and therefore MUST NOT appear on the mp=28 path.
#:
#: mp=28 replaces Cooper with ``iceDeMott`` -- module_mp_thompson.F:2537-2551
#: selects one or the other on ``is_aerosol_aware``; it never runs both.  The
#: three CUDA kernels below carry the literal Cooper expression
#: ``MIN(250.E3, 5.0*EXP(0.304*(T_0-temp)))`` at gpuwm/core/kernels/
#: thompson.cu:5931, :6208, :6641, :7190 and :7635, so launching any of them
#: on an mp=28 column would ADD a second ice-nucleation source on top of
#: DeMott rather than substitute for it -- a silent, stable, wrong result.
#:
#: ``tests/test_thompson_aerosol_adapter.py::
#: test_no_cooper_bearing_classic_launcher_is_ever_called`` pins this both at
#: the launcher level and at the CUDA-symbol level.
COOPER_BEARING_CLASSIC_LAUNCHERS: tuple[str, ...] = (
    "launch_frozen_vapor_network",
    "launch_frozen_vapor_network_from_owner",
    "launch_ice_nucleation",
)

#: The CUDA symbols those launchers resolve to, pinned so the gate survives a
#: launcher rename.
COOPER_BEARING_CLASSIC_KERNELS: tuple[str, ...] = (
    "thompson_frozen_vapor_network",
    "thompson_frozen_vapor_cloud_network",
    "thompson_ice_nucleation",
)

__all__ = [
    "AEROSOL_CEILING",
    "AEROSOL_COMMON_HEADER",
    "AEROSOL_KERNEL_MODULES",
    "AEROSOL_LAUNCHERS",
    "CCN_ACTIVATION_SHAPE",
    "CLASSIC_MODULE",
    "COLD_MODULE",
    "COOPER_BEARING_CLASSIC_KERNELS",
    "COOPER_BEARING_CLASSIC_LAUNCHERS",
    "DEFAULT_THREADS",
    "INIT_PROFILE_HEIGHT_FIELD",
    "NC_FLOOR_M3",
    "NIFA_FLOOR",
    "NT_C_MAX",
    "NWFA_FLOOR",
    "PROBE_MODULE",
    "PROFILE_FILL_EPS",
    "R1",
    "REUSED_CLASSIC_LAUNCHERS",
    "SAT_MODULE",
    "SED_MODULE",
    "STATE_MODULE",
    "VERTICAL_LEVEL_BOUNDS",
    "WARM_MODULE",
    "aerosol_profile_needs_fill",
    "device_drop_evaporation_number_table",
    "launch_aa_cloud_sedimentation",
    "launch_aa_cold_network",
    "launch_aa_cold_network_from_owner",
    "launch_aa_final_phase_cleanup",
    "launch_aerosol_effective_radius",
    "launch_aerosol_entry_cloud_number",
    "launch_aerosol_entry_snapshot",
    "launch_aerosol_init_profile",
    "launch_aerosol_rain_evaporation",
    "launch_aerosol_saturation_adjust",
    "launch_aerosol_state_finalize",
    "launch_aerosol_surface_emission",
    "launch_aerosol_warm_source_network",
    "launch_aerosol_warm_source_network_from_owner",
    "launch_aerosol_working_cloud",
    "launch_aerosol_working_number",
    "launch_grid",
    "launch_ncten_balance",
    "launch_tau1_density",
    "load_aerosol_device_tables",
    "validate_fields",
    "validate_fp64_fortran_table",
    "validate_int_fields",
    "zero_aerosol_accumulators",
]
