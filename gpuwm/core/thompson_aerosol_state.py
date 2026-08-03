"""Aerosol state launchers for WRF v4.6.1 Thompson ``mp_physics=28``.

This module owns the **accumulator contract** for the aerosol-aware port:
the entry snapshot, the working-value refresh helpers every other aerosol
package calls, the single terminal apply/clamp, the surface emission,
``thompson_init``'s synthetic CCN/IN profile fill, and the effective-radius
diagnostic.

Numerical authority is ``wrf461-pristine/phys/module_mp_thompson.F``
(WRF v4.6.1, commit ``d66e442``).  Bare line numbers below refer to that file.
Nothing here imports from :mod:`gpuwm.core.thompson`, which is read-only for
this port, and nothing here compiles or touches ``thompson.cu``.

The accumulator contract, restated
----------------------------------
``state.nc`` / ``state.nwfa`` / ``state.nifa`` are **read-only entry state for
the whole call**.  Three device scratch accumulators ``ncten`` / ``nwfaten`` /
``nifaten`` (per kilogram per second, matching WRF's ``orho``-scaled increment
sites at :2964, :2975, :3008, :3833) are zeroed at adapter entry by
:func:`zero_aerosol_accumulators`, written by every aerosol kernel, and applied
**exactly once** by :func:`launch_aerosol_state_finalize`, which carries WRF's
only set of clamps (:3972-4021).

Any kernel that needs a working per-m3 value recomputes it locally the way WRF
does -- :func:`launch_aerosol_working_number` for the aerosol,
:func:`launch_aerosol_working_cloud` for the droplets -- instead of applying
its own delta to state.  WRF clamps once; clamping four times is a silent
physics change.

.. _wp04-height-field:

The height field for the profile fill  (blocking unknown #4, RESOLVED)
----------------------------------------------------------------------
The spec's WP-04 text says "hgt is the MASS-level height (WRF passes z_at_q),
so use ArWen's mass-level height array, not z8w."  **That reading is wrong,
and the correction matters over terrain.**  Traced end to end:

* ``phys/module_physics_init.F:4517-4544`` calls ``thompson_init(HGT=z_at_q, ...)``.
* ``dyn_em/start_em.F:870-876`` is the only assignment to ``z_at_q``::

      DO k = kts,kte
         z_at_q(i,k,j) = (grid%ph_2(i,k,j)+grid%phb(i,k,j))/g

* ``Registry/Registry.EM_COMMON:198-200`` declares both ``ph`` and ``phb``
  with ``Z`` staggering, i.e. on FULL (w) levels.

So ``z_at_q`` is the **w-level geopotential height above sea level**, sampled
at the lowest ``kte`` full levels -- the array's name is misleading.
``hgt(i,1,j)`` is therefore the terrain elevation, which is exactly what the
ABSOLUTE 1000 m / 2500 m ``h_01`` thresholds are meant to test, and the profile
body uses the AGL difference ``hgt(k)-hgt(1)``.

ArWen's exact analogue is ``z8w[:nz]`` where
``z8w = (state.phb + state.php) / G``, which
``gpuwm/core/microphysics.py::_apply_thompson`` already materializes at
lines 470-474 (and ``launch_kessler`` at :389 shows the mass-level form
``0.5*(z8w[:nz]+z8w[1:])`` for contrast -- that is what must NOT be used
here).  :func:`launch_aerosol_init_profile` therefore documents and validates
an ``(nz, ny, nx)`` w-level height; see :data:`INIT_PROFILE_HEIGHT_FIELD`.

Passing the mass-level height instead would raise ``hgt(i,1,j)`` by half a
layer everywhere, which flips the ``h_01`` branch for any column whose terrain
sits within half a layer of 1000 m or 2500 m and rescales ``niCCN3`` for every
column in between -- silently reshaping the entire synthetic CCN profile.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE
from gpuwm.core.thompson_aerosol_launch import (
    STATE_MODULE,
    launch_grid,
    validate_fields,
    validate_int_fields,
)

#: ``thompson_init``'s fill test, module_mp_thompson.F:185 and :493.
PROFILE_FILL_EPS = 1.0e-15

#: Which ArWen array :func:`launch_aerosol_init_profile` expects for ``hgt``.
#: See :ref:`the module docstring <wp04-height-field>` for the derivation.
INIT_PROFILE_HEIGHT_FIELD = "z8w[:nz], i.e. (state.phb + state.php)/G on the "\
    "lowest nz FULL (w) levels, above sea level"

#: WRF's aerosol scalars, pinned with their line numbers so a caller can
#: assert against them without importing the CUDA header.
NWFA_FLOOR = 11.1e6      # :1805, :3211, :3980
NIFA_FLOOR = 5.0e3       # :1806, :3982  (naIN1*0.01)
AEROSOL_CEILING = 9999.0e6   # :1805-1806, :3980-3982
NC_FLOOR_M3 = 2.0        # :1830, :3217, :3486
NT_C_MAX = 1999.0e6      # :89
R1 = 1.0e-12             # :183
NA_CCN0 = 300.0e6        # :96
NA_CCN1 = 50.0e6         # :97
NA_IN0 = 1.5e6           # :94
NA_IN1 = 0.5e6           # :95


def _kernel(name: str):
    return get_kernel(STATE_MODULE, name)


def _launch(name: str, size: int, args: tuple) -> None:
    grid, block = launch_grid(size)
    _kernel(name)(grid, block, args)


# ---------------------------------------------------------------------------
# Accumulators.
# ---------------------------------------------------------------------------

def zero_aerosol_accumulators(ncten, nwfaten, nifaten) -> None:
    """Zero the three device scratch accumulators, module_mp_thompson.F:1679-1681.

    Called ONCE at adapter entry, before any aerosol kernel runs.  The
    accumulators carry per-kilogram-per-second tendencies, matching WRF's
    ``orho``-scaled increment sites, and are consumed only by
    :func:`launch_aerosol_state_finalize` and the two working-refresh
    launchers.
    """
    validate_fields(
        {"ncten": ncten, "nwfaten": nwfaten, "nifaten": nifaten})
    ncten.fill(0)
    nwfaten.fill(0)
    nifaten.fill(0)


# ---------------------------------------------------------------------------
# 1.  Entry snapshot.
# ---------------------------------------------------------------------------

def launch_aerosol_entry_snapshot(temperature, pressure, qv, nwfa, nifa,
                                  rho_out, nwfa_entry_m3, nifa_entry_m3
                                  ) -> None:
    """Freeze the read-only per-m3 aerosol entry state, :1795-1812.

    ``nwfa``/``nifa`` are the per-kilogram state fields and are NOT modified.
    ``rho_out`` receives WRF's entry density (:1802), formed from the
    vapour already floored at 1e-10 by :1801.

    This snapshot -- both bounds applied -- is what scavenging, ``iceDeMott``
    and ``iceKoop`` consume.  It is a different quantity from
    :func:`launch_aerosol_working_number`'s output; see that docstring.
    """
    _, size = validate_fields({
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "nwfa": nwfa, "nifa": nifa, "rho_out": rho_out,
        "nwfa_entry_m3": nwfa_entry_m3, "nifa_entry_m3": nifa_entry_m3,
    })
    _launch("thompson_aa_entry_snapshot", size,
            (temperature, pressure, qv, nwfa, nifa, rho_out,
             nwfa_entry_m3, nifa_entry_m3, np.int32(size)))


def launch_aerosol_entry_cloud_number(qc, nc, rho, rc_out, nc_entry_m3,
                                      nu_c_out, l_qc_out) -> None:
    """Entry droplet-distribution diagnosis, :1826-1848.

    ``qc`` and ``nc`` are modified in place ONLY on WRF's ``qc1d <= R1``
    branch, which zeroes both (:1844-1845).  ``rc_out`` is the cloud content
    in kg m^-3 (floored at R1), ``nc_entry_m3`` the rediagnosed droplet number
    in m^-3, ``nu_c_out``/``l_qc_out`` int32.

    The arithmetic is ``thompson_aa_cloud_dist`` from the shared device
    header, so a package that prefers to recompute it inline gets the
    identical value.
    """
    shape, size = validate_fields({
        "qc": qc, "nc": nc, "rho": rho, "rc_out": rc_out,
        "nc_entry_m3": nc_entry_m3,
    })
    validate_int_fields({"nu_c_out": nu_c_out, "l_qc_out": l_qc_out}, shape)
    _launch("thompson_aa_entry_cloud_number", size,
            (qc, nc, rho, rc_out, nc_entry_m3, nu_c_out, l_qc_out,
             np.int32(size)))


# ---------------------------------------------------------------------------
# 2.  Working refresh.
# ---------------------------------------------------------------------------

def launch_tau1_density(temperature, pressure, qv, rho_out) -> None:
    """TAU+1 density, :3189-3193.

    ``temperature``/``qv`` must be the already-updated (post-tendency) fields;
    ArWen's networks write them in place.  The vapour floor of :3192 is
    re-applied here, so the caller need not.
    """
    _, size = validate_fields({
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "rho_out": rho_out,
    })
    _launch("thompson_aa_tau1_density", size,
            (temperature, pressure, qv, rho_out, np.int32(size)))


def launch_aerosol_working_number(nwfa, nwfaten, rho, dt, nwfa_work_m3
                                  ) -> None:
    """The working water-friendly aerosol number, :3211.

    ``nwfa(k) = MAX(11.1E6, (nwfa1d(k) + nwfaten(k)*DT)*rho(k))``

    This is a SECOND, DISTINCT snapshot from
    :func:`launch_aerosol_entry_snapshot`, and the differences are
    deliberate in WRF:

    * no ``9999.E6`` ceiling,
    * no ``nifa`` counterpart anywhere in the scheme,
    * ``rho`` is the TAU+1 density recomputed at :3193, not the entry density
      of :1802 (use :func:`launch_tau1_density`).

    It is consumed by ``activ_ncloud`` and by nothing else (:3416-3421).
    Feeding the entry snapshot to activation instead changes activated droplet
    number wherever scavenging was significant.

    ``nwfa`` is the read-only entry per-kilogram state; nothing here writes
    state.
    """
    _, size = validate_fields({
        "nwfa": nwfa, "nwfaten": nwfaten, "rho": rho,
        "nwfa_work_m3": nwfa_work_m3,
    })
    _launch("thompson_aa_working_number", size,
            (nwfa, nwfaten, rho, DTYPE(dt), nwfa_work_m3, np.int32(size)))


def launch_aerosol_working_cloud(qc, qcten, nc, ncten, rho, dt,
                                 rc_work, nc_work_m3, l_qc_out) -> None:
    """The working cloud content and droplet number, :3213-3221 / :3484-3488.

    A plain clamp of the accumulated values -- unlike the entry diagnosis
    there is no ``lamc`` / ``D0c`` / ``2*D0r`` rediagnosis here.  WRF runs this
    twice with the same code and different densities (:3193 and :3490); pass
    whichever applies.

    ``qc``/``nc`` are the read-only entry per-kilogram fields.
    """
    shape, size = validate_fields({
        "qc": qc, "qcten": qcten, "nc": nc, "ncten": ncten, "rho": rho,
        "rc_work": rc_work, "nc_work_m3": nc_work_m3,
    })
    validate_int_fields({"l_qc_out": l_qc_out}, shape)
    _launch("thompson_aa_working_cloud", size,
            (qc, qcten, nc, ncten, rho, DTYPE(dt), rc_work, nc_work_m3,
             l_qc_out, np.int32(size)))


# ---------------------------------------------------------------------------
# 3.  Terminal apply and clamp -- the ONE clamp point.
# ---------------------------------------------------------------------------

def launch_aerosol_state_finalize(qc, nc, nwfa, nifa, ncten, nwfaten,
                                  nifaten, rho, dt,
                                  nc_out, nwfa_out, nifa_out) -> None:
    """Apply the three accumulators to state exactly once, :3972-4021.

    This is the ONLY place mp=28 writes ``nc``/``nwfa``/``nifa`` from the
    accumulators, and it carries WRF's only clamps.  Four other packages
    depend on that being true.

    ``qc`` is the FINAL per-kilogram cloud mixing ratio (the networks have
    already written it) and is zeroed in place on WRF's ``qc1d <= R1`` branch.
    ``nc``/``nwfa``/``nifa`` are the ENTRY per-kilogram state.

    ``rho`` IS NOT THE ENTRY DENSITY, and this used to say that it was.
    :3976, :4011, :4019 and :4020 all read ``rho(k)``, and by :3972 that is
    the TAU+1 value written at :3193, then per level at :3490 (condensation)
    and :3572 (rain evaporation); nothing after :3574 touches it.  MEASURED
    from an instrumented build of pristine ``module_mp_thompson.F`` that
    reproduces all 22 committed column fixtures byte for byte: the entry
    density is up to **7.4672e-03** away from it (aero-cold-overlap), and
    substituting it inside WRF itself moves ``nc1d`` at one of 504 fixture
    levels by **3.9271e-03** -- 1963x the 2.0e-06 end-to-end gate -- through
    :3976's ``2./rho(k)`` floor.

    The buffer to pass is the one :func:`launch_tau1_density` writes when it
    is called immediately AFTER the rain-evaporation launcher and BEFORE
    sedimentation: ArWen's ``temperature``/``qv`` hold WRF's ``temp(k)``/
    ``qv(k)`` exactly at that moment, and doing so reproduces WRF's terminal
    ``rho(k)`` bitwise at 501 of those 504 levels (worst 1.2334e-07).
    Recomputing it at the finalize call instead is 5.39e-05 off, because
    ArWen's ``temperature`` keeps absorbing the melt/freeze cleanup's ``tten``
    while WRF's ``temp(k)`` snapshot does not.  See
    ``tests/test_thompson_aerosol_state_gpu.py::
    test_terminal_apply_matches_wrfs_own_terminal_loop_on_every_fixture``,
    which drives this launcher on WRF's own terminal-loop state both ways.

    Each ``*_out`` may alias its corresponding input -- every thread reads its
    own element before writing it -- so the adapter can legally pass
    ``state.nc`` for both ``nc`` and ``nc_out``.

    Three of WRF's unit inconsistencies are reproduced literally and must not
    be "fixed":

    * :3976 compares the PER-KILOGRAM ``nc1d`` against the volumetric
      ``Nt_c_max``, while converting only the lower bound (``2./rho``);
    * :3979-3982 clamp per-kilogram ``nwfa1d``/``nifa1d`` against the per-m3
      constants ``11.1E6`` / ``5.0E3`` / ``9999.E6`` with no density at all;
    * :4020 caps the rediagnosed droplet number at ``Nt_c_max/rho``, i.e. the
      same constant, converted.

    So is one thing about the ARITHMETIC, and it is easy to lose.  :4012's
    lambda base, :4015/:4017's ``cce(2,nu_c)/D0c`` quotient and :4019's
    ``ccg(1,nu_c)*ocg2(nu_c)*qc1d/am_r`` prefactor are all REAL(4)
    sub-expressions whose results meet the DOUBLE ``lamc``.  nvrtc widens such
    a chain to double -- MEASURED, and a named ``float`` local does not stop
    it -- so the kernel pins them with ``__fmul_rn``/``__fdiv_rn``.  Without
    the pins the rediagnosed droplet number ran 2 to 3.5 float32 ulps hot on
    nearly every fixture, because :4019 CUBES the lambda.  With them, the
    kernel is BITWISE against a Fortran-faithful host transcription on all
    456 fixture states.
    """
    _, size = validate_fields({
        "qc": qc, "nc": nc, "nwfa": nwfa, "nifa": nifa, "ncten": ncten,
        "nwfaten": nwfaten, "nifaten": nifaten, "rho": rho,
        "nc_out": nc_out, "nwfa_out": nwfa_out, "nifa_out": nifa_out,
    })
    _launch("thompson_aa_state_finalize", size,
            (qc, nc, nwfa, nifa, ncten, nwfaten, nifaten, rho, DTYPE(dt),
             nc_out, nwfa_out, nifa_out, np.int32(size)))


# ---------------------------------------------------------------------------
# 4.  Surface emission.
# ---------------------------------------------------------------------------

def launch_aerosol_surface_emission(nwfa, nifa, nwfa2d, nifa2d, dt) -> None:
    """Fake surface aerosol source, mp_gt_driver:1310-1327.

    ``nwfa1d(kts) += nwfa2d*dt`` and ``nifa1d(kts) += nifa2d*dt``.  Lowest
    model level only, and DELIBERATELY NOT CLAMPED: WRF applies this after
    ``mp_thompson`` has already run its terminal ceiling, so between here and
    the next call's entry pack (:1805-1806) the aerosol fields may legitimately
    exceed ``9999.E6``.  Adding a ceiling here would silently change the
    boundary-layer aerosol budget on every step.

    ``nwfa2d``/``nifa2d`` are number tendencies in per-kilogram per second --
    the field was redefined from a concentration to a tendency on 13 May 2013
    (:1313-1315).

    Must be called AFTER :func:`launch_aerosol_state_finalize`, matching WRF's
    order.
    """
    shape, _ = validate_fields({"nwfa": nwfa, "nifa": nifa})
    if len(shape) != 3:
        raise ValueError(f"nwfa/nifa must be (nz, ny, nx), got {shape}")
    surface_shape = shape[1:]
    _, ncolumns = validate_fields({"nwfa2d": nwfa2d, "nifa2d": nifa2d})
    if nwfa2d.shape != surface_shape:
        raise ValueError(
            f"nwfa2d must have shape {surface_shape}, got {nwfa2d.shape}")
    grid, block = launch_grid(ncolumns)
    _kernel("thompson_aa_surface_emission")(
        grid, block,
        (nwfa, nifa, nwfa2d, nifa2d, DTYPE(dt), np.int32(ncolumns)))


# ---------------------------------------------------------------------------
# 5.  thompson_init's synthetic CCN / IN profile.
# ---------------------------------------------------------------------------

def aerosol_profile_needs_fill(field) -> bool:
    """WRF's domain-wide ``MAXVAL(...) .lt. eps`` test, :493 and :530.

    WRF reduces over the whole (distributed) domain via ``wrf_dm_max_real``,
    so this is one decision per domain per field, not per column.  The CCN and
    IN decisions are INDEPENDENT -- a domain can get one fill and not the
    other.
    """
    return bool(float(field.max()) < PROFILE_FILL_EPS)


def launch_aerosol_init_profile(hgt, nwfa, nifa, nwfa2d, *,
                                fill_ccn: bool, fill_in: bool) -> None:
    """thompson_init's synthetic CCN/IN profile fill, thompson_init:493-551.

    Run ONCE at domain construction, gated by
    :func:`aerosol_profile_needs_fill` on each of ``nwfa`` and ``nifa``
    separately.

    ``hgt`` must be ArWen's ``z8w[:nz]`` -- the FULL (w) level height above sea
    level, ``(state.phb + state.php)/G`` -- NOT the mass-level height.  See
    :ref:`the module docstring <wp04-height-field>`: WRF's argument is named
    ``z_at_q`` but ``dyn_em/start_em.F:873`` fills it from the Z-staggered
    ``ph_2 + phb``, so ``hgt(i,1,j)`` is the terrain elevation the absolute
    1000 m / 2500 m ``h_01`` thresholds test.

    Four things this reproduces that are easy to get wrong:

    * the ``k=1`` level uses the LEVEL-2 height difference, not zero;
    * ``h_01`` reads an ABSOLUTE height while the profile body uses the AGL
      difference ``hgt(k)-hgt(1)``;
    * ``nifa`` gets the same shape with ``naIN0``/``naIN1`` but there is NO
      2-D flux -- WRF never derives a ``nifa2d`` and it stays exactly zero;
    * ``nc`` is never touched by ``thompson_init``.  It stays 0 and is
      bootstrapped by the first call's terminal rediagnosis.
    """
    shape, _ = validate_fields({"hgt": hgt, "nwfa": nwfa, "nifa": nifa})
    if len(shape) != 3:
        raise ValueError(f"hgt/nwfa/nifa must be (nz, ny, nx), got {shape}")
    nz = shape[0]
    if nz < 2:
        raise ValueError(
            "thompson_init's profile fill reads level 2; nz must be >= 2")
    surface_shape = shape[1:]
    _, ncolumns = validate_fields({"nwfa2d": nwfa2d})
    if nwfa2d.shape != surface_shape:
        raise ValueError(
            f"nwfa2d must have shape {surface_shape}, got {nwfa2d.shape}")
    grid, block = launch_grid(ncolumns)
    _kernel("thompson_aa_init_profile")(
        grid, block,
        (hgt, nwfa, nifa, nwfa2d, np.int32(1 if fill_ccn else 0),
         np.int32(1 if fill_in else 0), np.int32(nz), np.int32(ncolumns)))


# ---------------------------------------------------------------------------
# 6.  Effective radius.
# ---------------------------------------------------------------------------

def launch_aerosol_effective_radius(temperature, pressure, qv, qc, nc,
                                    qi, ni, qs, effc, effi, effs,
                                    *, metres: bool = False) -> None:
    """``calc_effectRad``:5594-5699 with a PROGNOSTIC droplet number.

    ``nc``/``ni`` are per-kilogram.  Outputs are MICRONS by default, matching
    gpuwm's radiation-facing state contract and ``thompson.cu:373-381``: WRF's
    own ``mp_gt_driver:1476-1478`` clamps are applied first and the
    metre->micron multiply is the last operation.  Pass ``metres=True`` for
    raw ``calc_effectRad`` output in metres, which is what
    ``oracle-aero/probe-effectrad.csv`` records.

    The cloud branch is the only THREE-way shape selector in the scheme
    (:5637-5643) and the only consumer of the exact-integer ``g_ratio``
    PARAMETER (:5611-5613) rather than the runtime ``ccg(2,n)*ocg1(n)``
    product.  ``thompson_aa_inu_c_effrad`` and ``THOMPSON_AA_G_RATIO`` come
    from the shared device header; do not substitute ``thompson_aa_nu_c`` or
    ``2730.0f`` here.

    THE THREE BRANCH BODIES ARE NOW THE SHARED HEADER'S.
    ``thompson_aerosol_state.cu`` used to carry private pinned copies named
    ``thompson_aa_state_eff_rad_cloud``/``_ice``/``_snow``, because the shared
    ones left their float32 chains unpinned (nvrtc is free to evaluate such a
    chain in double, and WP-04 measured it doing exactly that elsewhere in
    this file).  The header pins all three now, with the identical
    ``__fmul_rn``/``__fdiv_rn``/``__fadd_rn`` spelling, so the copies are
    deleted.  PROVED INERT BEFORE DELETION: the three deleted bodies compiled
    alongside the shared ones in ONE translation unit agree BITWISE on 400 000
    randomized states.  ``thompson_field_a``/``thompson_field_b`` were already
    shared with the cold network's snow moments and stay that way.

    MEASURED against WRF, driving the kernel with the oracle's own post-step
    columns so no upstream residual is in the way:

    * all 19 committed fixtures x 24 levels x 3 fields: BITWISE, except FIVE
      levels that are proved to be the harness's own float32 temperature
      round trip (``run_column_aero.F90`` records ``theta*exner`` where
      ``mp_gt_driver:1357`` stored ``t1d/pii``) -- down from eight, because
      the harness repaired its own ``pii``, and each survivor is now proved at
      exactly ONE ulp where the search used to have to run to +-4;
    * ``probe-effectrad.csv``, all 50 rows: BITWISE (the probe grew from 14
      rows to 50 and now carries 14 temperatures and 22 distinct ``effs_m``
      values, where the 14-row version was one snow state repeated);
    * the REAL ``calc_effectRad``, called not transcribed, over 960 states
      (40 columns x 24 levels sweeping T 233-260 K, qc/nc/qi/ni/qs over four
      decades each): ``effc`` and ``effs`` BITWISE, ``effi`` 958/960 with the
      two survivors at 1.094592e-07 -- the double-rounding limit of
      ``(float)pow(double,double)`` against glibc's singly-rounded ``powf``,
      which plain CUDA ``powf`` does not fix either;
    * a 2400-state wide-range sweep against a float32 host transcription of
      :5624-5695: ``effc``/``effi`` BITWISE, ``effs`` bitwise at 2397 of 2400
      with the three exceptions attributed to the ~0.09% of arguments where
      glibc's ``powf`` and an evaluate-in-double-round-once ``powf`` disagree.

    Whatever an end-to-end gate sees in ``effc_m``/``effi_m`` is therefore
    INHERITED from ``qc``/``nc``/``qi``/``ni``, at one third of their relative
    error (``re ~ (r/n)**(1/3)``); that ratio is itself pinned by
    ``test_effective_radius_residual_is_inherited_from_qc_and_nc``.
    """
    _, size = validate_fields({
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "qc": qc, "nc": nc, "qi": qi, "ni": ni, "qs": qs,
        "effc": effc, "effi": effi, "effs": effs,
    })
    name = ("thompson_aa_effective_radius_metres" if metres
            else "thompson_aa_effective_radius")
    _launch(name, size,
            (temperature, pressure, qv, qc, nc, qi, ni, qs,
             effc, effi, effs, np.int32(size)))


__all__ = [
    "AEROSOL_CEILING",
    "INIT_PROFILE_HEIGHT_FIELD",
    "NA_CCN0",
    "NA_CCN1",
    "NA_IN0",
    "NA_IN1",
    "NC_FLOOR_M3",
    "NIFA_FLOOR",
    "NT_C_MAX",
    "NWFA_FLOOR",
    "PROFILE_FILL_EPS",
    "R1",
    "aerosol_profile_needs_fill",
    "launch_aerosol_effective_radius",
    "launch_aerosol_entry_cloud_number",
    "launch_aerosol_entry_snapshot",
    "launch_aerosol_init_profile",
    "launch_aerosol_state_finalize",
    "launch_aerosol_surface_emission",
    "launch_aerosol_working_cloud",
    "launch_aerosol_working_number",
    "launch_tau1_density",
    "zero_aerosol_accumulators",
]
