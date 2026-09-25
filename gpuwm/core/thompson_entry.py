"""Thompson's own entry moment-consistency block, on the host.

WRF v4.6.1 ``phys/module_mp_thompson.F:1827-1899`` is the first thing the
scheme does to any state handed to it: before a single process rate is
computed, ``mp_thompson`` walks the column and makes every hydrometeor
mass and number pair self-consistent.  Where a species has mass and no
number, the block SETS the number from the mass under the scheme's own
assumed size distribution; where a species has number and no mass, it
zeroes both.  That is the scheme's answer -- not a port's answer -- to
the exact question an analysis increment raises.

It lives in ``gpuwm.core`` rather than beside the other host mirrors in
``gpuwm.verify`` because the mp=28 real-data cold start runs it once over
the analyzed mass (``gpuwm.ingest.real``), and the preparation-only
distribution stages ``real.py`` while omitting the verification package;
``gpuwm.verify.thompson_entry`` re-exports it for the tests that read it
there.

This module is the float32/float64 host mirror of those three arms, and
it exists for two callers, :func:`gpuwm.da.moments.repair_moments` and
the mp=28 cold start in :mod:`gpuwm.ingest.real`; the first
needs the numbers on the host, for a checkpoint or a member's analysis,
with no device and no column driver in reach.  It is the Thompson
counterpart of ``npref._np_morrison_slopes``, which plays the same part
for Morrison.

WHAT THE SCHEME SAYS, ARM BY ARM
--------------------------------
``cloud`` (:1827-1842).  ``nc`` floors at 2 m^-3, which for a cell the
    analysis left empty IS the entry value; ``nu_c`` is then 15, the
    lambda from the mass is far too small, and the ``xDc > 2*D0r`` clamp
    fires, so the rediagnosed droplet number is the scheme's
    largest-droplet limit at a 100 um mean diameter.

``ice`` (:1851-1869).  ``ni`` floors at ``R2``; the ``ni <= R2`` branch
    sets lambda from a 5 um crystal and caps the result at 999e3 m^-3.
    Any ice mass above ~5e-9 kg m^-3 hits that cap, so the repaired ice
    number is 999e3 m^-3 for every cell a radar analysis creates.

``rain`` (:1878-1898).  ``nr`` floors at ``R2``; the ``nr <= R2`` branch
    sets the median volume diameter to 1 mm and back-computes the number
    from it.  The recomputed lambda reproduces that 1 mm, so neither
    size clamp fires afterwards.

DENSITY
-------
Every arm works in per-volume units and the caller's state is per
kilogram, so a density is needed.  It very nearly cancels: at a cell
with ``q > 0`` and ``N <= 0`` the repaired per-kilogram number is
exactly proportional to the mass and INDEPENDENT of density for cloud
and rain, because the size clamp that fires replaces the only lambda the
density reached.  Ice is the one exception, and only through its 999e3
m^-3 cap: there the per-kilogram answer is ``999e3 / rho``.
``tests/test_da_moments.py`` pins both halves of that statement rather
than asserting them.

The density used is ``rho = 1/alt``, the same dry density
``gpuwm.core.microphysics`` hands every scheme (its module docstring,
line 25, and ``launch_kessler``'s call site).  WRF's ``mp_thompson``
forms its own entry density from pressure, temperature and vapour at
:1802 instead, and the two differ by order one percent in moist air.
That difference reaches NOTHING except the ice cap, where it moves the
repaired number by the same order one percent -- on a cell that had no
number at all.  It is recorded here because a limit that is written down
is a limit; a limit that is not is a surprise.

PRECISION
---------
Every intermediate below carries the precision the Fortran DECLARES for
it, not the precision Python would pick: ``rc``/``ri``/``rr`` are REAL
(:1579), ``nc``/``ni``/``nr`` are REAL (:1580), ``mvd_r`` is REAL
(:1589), ``lamc``/``lami``/``ilami``/``lamr`` are DOUBLE (:1597-1598)
and ``xDc``/``xDi`` are REAL (:1599).  The association matters as much as
the width, because Fortran's ``*`` and ``/`` share a precedence and
associate to the left: the ice number divides by ``am_i`` BEFORE its
power (:1857) and the rain number divides by ``am_r`` AFTER its power
(:1885), so the two arms cannot share a helper however alike they read.
``tests/test_thompson_entry_wrf_order.py`` holds the whole block against
a line-by-line transcription of the Fortran, bitwise, and pins the
numbers that transcription produces.  This is not decoration: the
``nu_c`` quotient is REAL, and formed in float64 it reads a different
gamma-table row at an ordinary droplet concentration.

CONSTANTS
---------
Nothing is restated.  The gamma moments come from
:mod:`gpuwm.core.thompson_aerosol_contract`, which derives them at
import from WRF's own ``WGAMMA`` and SHA-256 pins the result, and the
scalars are the values ``gpuwm/core/kernels/thompson_aerosol_common.cuh``
already pins with their WRF line numbers.
``tests/test_da_moments.py`` parses that header and asserts every scalar
below equals the ``#define`` it names, so there is one spelling of each
and a drift in either copy is a test failure rather than a quiet
disagreement between the analysis and the forecast.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.thompson_aerosol_contract import (
    AM_R, BM_R, CCE2, CCG1, CCG2, NT_C_MAX, OCG1, OCG2,
)

#: Source of every line number in this module.
THOMPSON_ENTRY_SOURCE = (
    "WRF v4.6.1 phys/module_mp_thompson.F:1827-1899 "
    "(commit d66e442fccc04111067e29274c9f9eaccc3cef28)")

#: The one authority string every receipt that ran this block carries:
#: the assimilation repair (gpuwm.da.moments.repair_moments) and the
#: mp=28 real-data cold start (gpuwm.ingest.real) both name it, so a
#: receipt from either can be matched to the other.
THOMPSON_ENTRY_AUTHORITY = (
    "WRF v4.6.1 module_mp_thompson.F:1827-1899 entry moment-consistency "
    "block, via its host mirror gpuwm.core.thompson_entry")

#: Scalars, each with the ``THOMPSON_AA_*`` define in
#: ``gpuwm/core/kernels/thompson_aerosol_common.cuh`` that carries the same
#: value and the WRF line it is transcribed from.  The test that parses the
#: header keys on these names.
CUH_SCALARS = {
    "R1": ("THOMPSON_AA_R1", 1.0e-12),            # :183
    "R2": ("THOMPSON_AA_R2", 1.0e-6),             # :184
    "AM_R": ("THOMPSON_AA_AM_R", None),           # :128 PI*rho_w/6
    "BM_R": ("THOMPSON_AA_BM_R", 3.0),            # :129
    "AM_I": ("THOMPSON_AA_AM_I", None),           # :137 PI*rho_i/6
    "MU_I": ("THOMPSON_AA_MU_I", 0.0),            # :105
    "D0C": ("THOMPSON_AA_D0C", 1.0e-6),           # :224
    "D0R": ("THOMPSON_AA_D0R", 50.0e-6),          # :225
    "NT_C_MAX": ("THOMPSON_AA_NT_C_MAX", 1999.0e6),   # :89
    "NC_FLOOR_M3": ("THOMPSON_AA_NC_FLOOR", 2.0),     # :1830
}

#: ``R1``, the scheme's activity gate: at or below it the entry block
#: zeroes the species' mass AND its number (:1844-1848, :1871-1875,
#: :1900-1904).  This is the threshold above which Thompson reads a
#: number moment at all.
R1 = 1.0e-12
#: ``R2`` (:184), the number floor the three arms apply before deciding
#: whether the number needs setting from the mass.
R2 = 1.0e-6

#: :137, ``am_i = PI*rho_i/6`` with ``rho_i = 890``.  Written as the
#: float32 product the header pins, not as a decimal literal.
AM_I = float(np.float32(np.float32(3.1415926536) * np.float32(890.0)
                        / np.float32(6.0)))
BM_I = 3.0                      # :138
MU_I = 0.0                      # :105
MU_R = 0.0                      # :103
D0C = 1.0e-6                    # :224
D0R = 50.0e-6                   # :225

#: ``cie(2) = bm_i + mu_i + 1`` (:688), an exact small integer, and the
#: same 4 the ported terminal ice bound spells as ``4.0 / lambda``.
CIE2 = BM_I + MU_I + 1.0
#: ``cig(1)*oig2 = WGAMMA(mu_i+1)/WGAMMA(bm_i+mu_i+1) = 1/6`` exactly
#: (:694-695, :701-702), which the ported bound also spells as ``1/6``.
CIG1_OIG2 = 1.0 / 6.0
#: ``cig(2)*oig1 = WGAMMA(bm_i+mu_i+1)/WGAMMA(mu_i+1) = 6`` exactly.
CIG2_OIG1 = 6.0
#: ``crg(2)*org3 = WGAMMA(mu_r+1)/WGAMMA(bm_r+mu_r+1) = 1/6`` and
#: ``crg(3)*org2 = 6``, both proved exact in the device header's comment
#: above ``thompson_aa_entry_rain_distribution``.
CRG2_ORG3 = 1.0 / 6.0
CRG3_ORG2 = 6.0

#: ``(3.0 + mu_r + 0.672)``, the median-volume-diameter factor (:1884).
MVD_FACTOR = 3.0 + MU_R + 0.672
#: The median volume diameter the scheme assumes for rain mass that
#: arrives without a number (:1883).
RAIN_INITIAL_MVD_M = 1.0e-3
#: The rain size clamps (:1887, :1891).
RAIN_MVD_MAX_M = 2.5e-3
#: ``D0r*0.75`` (:1894), formed as WRF forms it: a REAL parameter times a
#: default-real literal.  The same product in float64 is one float32 ULP
#: higher (3.7500001781154424e-05 against 3.749999814317562e-05), which
#: moves the rebuilt lambda from 97920.0 to 97919.9921875 and the rain
#: number it rebuilds by two float32 ULPs.
RAIN_MVD_MIN_M = float(np.float32(np.float32(D0R) * np.float32(0.75)))
#: The ice size clamps and the number ceiling (:1855-1856, :1864-1869),
#: the same three the ported ``thompson_aa_bound_ice_number`` carries.
ICE_MIN_DIAMETER_M = 5.0e-6
ICE_MAX_DIAMETER_M = 300.0e-6
ICE_NUMBER_CEILING_M3 = 999.0e3

#: Highest ``nu_c`` the scheme's gamma tables carry (:1832).
NU_C_MAX = 15

#: Which species this mirror answers for, and the state spellings
#: :mod:`gpuwm.da.moments` pairs them with.
ENTRY_SPECIES = ("cloud", "rain", "ice")


def _f32(value):
    return np.asarray(value, dtype=np.float32)


def _nu_c(nc_m3: np.ndarray) -> np.ndarray:
    """``nu_c = MIN(15, NINT(1000.E6/nc) + 2)`` (:1832).

    Fortran's ``NINT`` rounds half away from zero, which for positive
    arguments is ``floor(x + 0.5)``; numpy's ``rint`` rounds half to
    even, and the two disagree on exact halves.  The floor form is used
    so a tie lands where the scheme puts it.

    The quotient is formed in SINGLE precision because that is where WRF
    forms it: ``1000.E6`` is a default-real literal and ``nc`` is a REAL
    array (:1580), so ``NINT`` is handed a single-precision value.  This
    is not a rounding nicety, it moves whole gamma-table rows.  At
    ``nc = 666666688`` m^-3, a 666 cm^-3 droplet concentration and
    nothing exotic, the REAL quotient is exactly 1.5 and ``NINT`` gives
    2, so the scheme reads ``nu_c = 4``; the same division in float64
    gives 1.4999999520000016, ``NINT`` gives 1, and a float64 mirror
    reads row 3.  The float32 quotient is widened exactly before the
    ``+ 0.5``, so the tie survives the widening.
    """
    ratio = np.float32(1000.0e6) / np.asarray(nc_m3, dtype=np.float32)
    nearest = np.floor(ratio.astype(np.float64) + 0.5) + 2.0
    return np.clip(nearest, 1.0, float(NU_C_MAX)).astype(np.int64)


def cloud_number_m3(cloud_mass_m3: np.ndarray,
                    cloud_number_m3: np.ndarray) -> np.ndarray:
    """The entry block's rediagnosed droplet number, :1829-1841.

    Both arguments are per volume.  ``cloud_mass_m3`` must be above
    ``R1``; the caller owns that branch, exactly as the device helper
    ``thompson_aa_cloud_dist`` says of its own ``rc``.

    :1842, ``IF (.NOT. is_aerosol_aware) nc(k) = Nt_c``, is deliberately
    absent: it overrides the rediagnosed number with the scheme's fixed
    droplet concentration on mp 8, and mp 8 carries no prognostic ``nc``
    for an analysis to break, so this arm is only ever reached for mp 28
    where the Fortran does not take that branch.
    """
    rc = _f32(cloud_mass_m3)
    nc = np.maximum(
        np.float32(CUH_SCALARS["NC_FLOOR_M3"][1]),
        np.minimum(_f32(cloud_number_m3), np.float32(NT_C_MAX)))
    nu_c = _nu_c(nc)
    ccg2 = np.float32(CCG2[nu_c])
    ocg1 = np.float32(OCG1[nu_c])
    cce2 = np.float32(CCE2[nu_c])
    ccg1 = np.float32(CCG1[nu_c])
    ocg2 = np.float32(OCG2[nu_c])
    obmr = np.float32(np.float32(1.0) / np.float32(BM_R))
    # :1833  lamc = (nc*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc)**obmr
    lamc = np.asarray(
        np.power(_f32(nc * np.float32(AM_R) * ccg2 * ocg1 / rc), obmr),
        dtype=np.float64)
    # :1834  xDc = (bm_r + nu_c + 1.)/lamc.  A REAL numerator over the
    # DOUBLE lamc is a DOUBLE quotient ASSIGNED TO A REAL (:1599 declares
    # xDc single), so the two comparisons below are made on the rounded
    # value, and a lambda that puts xDc within half an ULP of a threshold
    # takes the branch the scheme takes.
    x_dc = ((np.float32(BM_R) + nu_c.astype(np.float32)
             + np.float32(1.0)).astype(np.float64)
            / lamc).astype(np.float32)
    # :1836 / :1838  a REAL quotient assigned to the DOUBLE lambda.
    small = x_dc < np.float32(D0C)
    large = x_dc > np.float32(D0R) * np.float32(2.0)
    lamc = np.where(small, np.float64(cce2 / np.float32(D0C)), lamc)
    lamc = np.where(
        large & ~small,
        np.float64(cce2 / (np.float32(D0R) * np.float32(2.0))), lamc)
    # :1840-1841  MIN(DBLE(Nt_c_max), ccg(1,nu_c)*ocg2(nu_c)*rc/am_r
    #                                 * lamc**bm_r)
    prefactor = np.float64(_f32(ccg1 * ocg2 * rc / np.float32(AM_R)))
    return np.minimum(np.float64(NT_C_MAX),
                      prefactor * lamc ** np.float64(BM_R))


def rain_number_m3(rain_mass_m3: np.ndarray,
                   rain_number_m3: np.ndarray) -> np.ndarray:
    """The entry block's bounded rain number, :1880-1898."""
    rr = _f32(rain_mass_m3)
    nr = np.maximum(np.float32(R2), _f32(rain_number_m3))
    obmr = np.float32(np.float32(1.0) / np.float32(BM_R))

    def _number_from_lambda(lam):
        """``nr = crg(2)*org3*rr*lamr**bm_r / am_r`` (:1885, :1893, :1897).

        Fortran's ``*`` and ``/`` share a precedence and associate to the
        left, so the division by ``am_r`` comes AFTER the power and is
        therefore done in DOUBLE: the REAL product ``crg(2)*org3*rr`` is
        widened, multiplied by the DOUBLE ``lamr**bm_r``, divided, and
        only the result is rounded back into the REAL ``nr``.  The ice
        arm writes the same three factors in a different order and
        divides in REAL, which is why the two arms cannot share a helper.
        """
        return (_f32(np.float32(CRG2_ORG3) * rr).astype(np.float64)
                * lam ** np.float64(BM_R)
                / np.float64(np.float32(AM_R))).astype(np.float32)

    # :1882-1886  mass with no number takes the 1 mm median volume drop.
    lam_repair = np.float64(np.float32(MVD_FACTOR)
                          / np.float32(RAIN_INITIAL_MVD_M))
    nr = np.where(nr <= np.float32(R2), _number_from_lambda(lam_repair),
                  nr).astype(np.float32)
    # :1888  lamr = (am_r*crg(3)*org2*nr/rr)**obmr
    lamr = np.asarray(
        np.power(_f32(np.float32(AM_R) * np.float32(CRG3_ORG2) * nr / rr),
                 obmr),
        dtype=np.float64)
    mvd = (np.float64(np.float32(MVD_FACTOR)) / lamr).astype(np.float32)
    for condition, clamped in (
            (mvd > np.float32(RAIN_MVD_MAX_M), np.float32(RAIN_MVD_MAX_M)),
            (mvd < np.float32(RAIN_MVD_MIN_M), np.float32(RAIN_MVD_MIN_M))):
        lam_clamped = np.float64(np.float32(MVD_FACTOR) / clamped)
        nr = np.where(condition, _number_from_lambda(lam_clamped),
                      nr).astype(np.float32)
    return np.asarray(nr, dtype=np.float64)


def ice_number_m3(ice_mass_m3: np.ndarray,
                  ice_number_m3: np.ndarray) -> np.ndarray:
    """The entry block's bounded ice number, :1853-1869."""
    ri = _f32(ice_mass_m3)
    ni = np.maximum(np.float32(R2), _f32(ice_number_m3))
    obmi = np.float32(np.float32(1.0) / np.float32(BM_I))

    def _from_diameter(diameter):
        """``cig(1)*oig2*ri/am_i*lami**bm_i`` (:1857, :1865, :1868).

        Left-associative, so the division by ``am_i`` lands BEFORE the
        power and is done in REAL; only the multiply by the DOUBLE
        ``lami**bm_i`` widens the expression.  Returned in float64
        because the two callers that cap it do their ``MIN`` against a
        DOUBLE literal (``999.D3``) and round to REAL after it, not
        before.
        """
        lami = np.float64(np.float32(CIE2) / np.float32(diameter))
        return (_f32(np.float32(CIG1_OIG2) * ri / np.float32(AM_I))
                .astype(np.float64) * lami ** np.float64(BM_I))

    # :1854-1857  mass with no number takes the 5 um crystal, capped.
    repaired = np.minimum(
        np.float64(ICE_NUMBER_CEILING_M3),
        _from_diameter(ICE_MIN_DIAMETER_M)).astype(np.float32)
    ni = np.where(ni <= np.float32(R2), repaired, ni).astype(np.float32)
    # :1859-1861  lami from the bounded number, then the size clamps.
    lami = np.asarray(
        np.power(_f32(np.float32(AM_I) * np.float32(CIG2_OIG1) * ni / ri),
                 obmi),
        dtype=np.float64)
    # :1861-1862  ilami = 1./lami, then xDi = (bm_i + mu_i + 1.)*ilami.
    # WRF MULTIPLIES by the reciprocal it just formed and assigns the
    # DOUBLE product to a REAL (:1599); dividing by lami in float64 and
    # comparing the unrounded quotient is a different number on both
    # counts.
    ilami = np.float64(1.0) / lami
    x_di = ((np.float32(BM_I) + np.float32(MU_I) + np.float32(1.0))
            * ilami).astype(np.float32)
    small = x_di < np.float32(ICE_MIN_DIAMETER_M)
    large = x_di > np.float32(ICE_MAX_DIAMETER_M)
    ni = np.where(small,
                  np.minimum(
                      np.float64(ICE_NUMBER_CEILING_M3),
                      _from_diameter(ICE_MIN_DIAMETER_M)
                  ).astype(np.float32),
                  ni).astype(np.float32)
    ni = np.where(large & ~small,
                  _from_diameter(ICE_MAX_DIAMETER_M).astype(np.float32),
                  ni).astype(np.float32)
    return np.asarray(ni, dtype=np.float64)


#: The per-volume arm for each species this mirror answers for.
ENTRY_ARMS = {
    "cloud": cloud_number_m3,
    "rain": rain_number_m3,
    "ice": ice_number_m3,
}


def np_thompson_entry_numbers(species: str, mass_per_kg, number_per_kg,
                              density) -> np.ndarray:
    """The entry block's number moment, per kilogram, for one species.

    ``mass_per_kg`` and ``number_per_kg`` are the state's own arrays;
    ``density`` is the density the scheme is handed (see the module
    docstring).  The return is per kilogram, the unit the state carries,
    with the scheme's mass-below-``R1`` arm ALREADY applied: those cells
    come back at zero, because that is what the entry block writes back
    to ``nc1d``/``ni1d``/``nr1d`` there.
    """
    try:
        arm = ENTRY_ARMS[species]
    except KeyError:
        raise ValueError(
            f"Thompson's entry block has no arm for {species!r}; it "
            f"diagnoses {', '.join(ENTRY_SPECIES)} and nothing else -- "
            "snow and graupel are single-moment in this scheme") from None
    mass = np.asarray(mass_per_kg, dtype=np.float64)
    number = np.asarray(number_per_kg, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    active = mass > R1
    # The inactive cells are not merely uninteresting: the entry block
    # divides by the species' mass, so they must be kept away from the
    # arithmetic entirely rather than filtered out of its result.
    safe_mass = np.where(active, mass, 1.0e-6)
    per_volume = arm(safe_mass * rho, number * rho)
    return np.where(active, np.asarray(per_volume) / rho, 0.0)


__all__ = [
    "CUH_SCALARS",
    "ENTRY_ARMS",
    "ENTRY_SPECIES",
    "ICE_NUMBER_CEILING_M3",
    "R1",
    "R2",
    "THOMPSON_ENTRY_AUTHORITY",
    "THOMPSON_ENTRY_SOURCE",
    "cloud_number_m3",
    "ice_number_m3",
    "np_thompson_entry_numbers",
    "rain_number_m3",
]
