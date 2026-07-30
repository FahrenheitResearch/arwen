"""Noah-MP ``NOAHMP_SFLX``: the whole-column composition, in FP32.

Ports ``MODULE_SF_NOAHMPLSM``'s NOAHMP_SFLX (450-1079) and ERROR (1560-1737)
from ``phys/module_sf_noahmplsm.F`` at WRF commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``
(``sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282``).

``kind_phys == kind(1.0)`` in that build, so every quantity here is IEEE
binary32 and every arithmetic boundary is pinned to ``numpy.float32``.

What this module is
-------------------
NOAHMP_SFLX owns no physics.  It is six calls -- ATM, PHENOLOGY, PRECIP_HEAT,
ENERGY, WATER, ERROR -- plus about thirty statements of marshalling around
them.  Every callee is pinned bitwise elsewhere in this package and is
**imported**, never re-transcribed:

* :func:`gpuwm.core.noahmp_leaves.atm`
* :func:`gpuwm.core.noahmp_vegprecip.phenology`
* :func:`gpuwm.core.noahmp_vegprecip.precip_heat`
* :func:`gpuwm.core.noahmp_water.water`
* ``gpuwm.core.noahmp_energy.energy`` -- see "The ENERGY seam" below.

ERROR is transcribed here because it has no caller other than NOAHMP_SFLX and
no callee at all: nine arithmetic statements and three abort gates.

No transcendental is evaluated in this module.  ATM's ``expf``/``powf`` and
PRECIP_HEAT's ``expf`` live inside those leaves, which own
:mod:`gpuwm.core.noahmp_libm`.

Pinned option identity
----------------------
``dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0``, plus
``soiltstep=0.0``, which ``module_sf_noahmpdrv.F`` (drivers/wrf, 648-676)
turns into ``soil_update_steps = 1`` and ``calculate_soil = .true.``.
``WRF_HYDRO`` is not defined.

What that identity kills, and why each kill is a fact rather than a hope
-----------------------------------------------------------------------
* **DVEG = 4** selects ``FVEG = SHDMAX`` at 863-864 and leaves
  ``dveg_active = .false.`` at 1016, so CARBON (1021-1028) is never called and
  ``NEE``/``GPP``/``NPP`` keep the ``0.0`` written at 806-808.  The six carbon
  pools ``LFMASS RTMASS STMASS WOOD STBLCP FASTCP`` are INOUT and only CARBON
  writes them, so all six are pass-through.  ``LAI``/``SAI`` are rewritten by
  PHENOLOGY from the monthly table.
* **OPT_CROP = 0** with ``CROPTYPE = 0`` kills CARBON_CROP (1030-1036), the
  crop ``FVEG`` override (877-879) and the gecros state vector.  ``GRAIN``,
  ``GDD`` and ``PGS`` are therefore pass-through.
* **OPT_IRR = 0** kills the whole irrigation region -- 878-936 and 1042-1045 --
  through a chain that does not depend on ``parameters%IRR_FRAC``.  That
  matters: ``IRR_FRAC`` is one of the components
  :func:`gpuwm.core.noahmp.transfer_mp_parameters` deliberately refuses to
  transfer, so a kill that needed its value could not be checked here at all.
  The chain instead runs: TRIGGER_IRRIGATION takes its ``OPT_IRR .LT. 1``
  branch at 9291 and zeroes ``IRAMTSI/IRAMTMI/IRAMTFI`` at 9344-9346, so
  whether or not the guard at 912-919 admits the call, all three amounts leave
  that block at zero.  ``IRAMTSI == 0.0`` makes the SPRINKLER_IRRIGATION guard
  at 928 false, so ``IRSIRATE`` and ``FIRR`` keep their entry zeros and
  ``RAIN`` is not incremented.  With ``IRSIRATE == 0.0`` and ``FIRR == 0.0``
  the pre-ERROR block at 1042-1045 is ``PRCP = PRCP + 0.0`` and
  ``FSH = FSH - 0.0``, both exact identities on binary32 for the finite,
  non-negative ``PRCP`` and finite ``FSH`` that reach it.
  ``SIFAC``/``MIFAC``/``FIFAC`` (884-907) are read only by the three
  irrigation routines and are dead with them.  :func:`sflx_pre` asserts
  ``IRRFRA`` and the three ``IRAMT*`` entry zeros rather than assuming them.
* **OPT_TDRN = 0** makes ``QTLDRN`` identically zero out of WATER.
* **ICE = 0** in every admitted column.  NOAHMP_SFLX has no ``ICE`` branch of
  its own -- it only forwards it to ENERGY -- and NOAHMP_GLACIER is a separate
  entry point this module does not compose.

Arguments the pinned identity does not consume
----------------------------------------------
``SMCEQ`` reaches WATER and dies inside its ``OPT_RUN == 5`` branch.
``ILOC``/``JLOC`` appear only in diagnostic ``WRITE`` statements and in
argument lists whose callees ignore them.  ``LLANDUSE`` is read only to set
``CROPLU`` (878-882), whose only consumers are the dead irrigation gates.
``SIFRA``/``MIFRA``/``FIFRA``, the three ``IRCNT*`` counters, ``IRSIRATE``,
``IRMIRATE``, ``IRFIRATE``, ``EIRR`` and ``GECROS1D`` are irrigation/crop
state.  ``TDFRACMP`` reaches WATER's dead tile drain.  ``SHDFAC`` is read only
by the ``DVEG == 1/6/7`` arm at 858.  None of them is an argument here.

The undefined slots, reproduced as the DEFINED value and named
--------------------------------------------------------------
``DZSNSO`` is a local of NOAHMP_SFLX and the loop at 827-833 writes only
``ISNOW+1 .. NSOIL``.  On a column with fewer than NSNOW snow layers the
buried slots are whatever the stack held.  Every consumer -- ENERGY's
THERMOPROP / CSNOW / TSNOSOI / PHASECHANGE loops, WATER, and ERROR's soil sum
-- indexes from ``ISNOW+1`` or from ``1``, so no defined value depends on them.
:func:`sflx_pre` writes ``0.0`` there, which is the defined behaviour, and
:mod:`tests.test_noahmp_sflx` proves the whole-column answer does not move when
those slots are seeded with something else.

The ENERGY seam
---------------
:func:`sflx` is split at the ENERGY call, and the split is not a convenience:

* :func:`sflx_pre` -- entry through the ENERGY call boundary.  Returns the
  complete ENERGY argument bundle as :class:`EnergyCall`, whose field names are
  ENERGY's own dummy-argument names, so the bundle can be compared field for
  field against the ``in`` role of ``noahmp-energy.csv``.
* :func:`sflx_post` -- ENERGY's return through the end of the subroutine,
  taking ENERGY's results as an :class:`EnergyResult`.

That lets the composition be held to ``max_ulp 0`` from two directions at once:
the bundle against ENERGY's own byte-unmodified fixture, and the exit state
against the byte-unmodified whole-column fixture with ENERGY *replayed out of
that fixture* rather than computed.  A divergence is then attributable to this
module rather than absorbed into whatever the ENERGY port happens to do, and
the two halves cannot hide compensating errors in each other.

``gpuwm.core.noahmp_energy`` landed while this module was being written and
:func:`sflx` closes the loop, so the whole column runs end to end from ported
code.  ``ist`` is nonetheless still admitted as an argument here even though
that port refuses ``ist /= 1``: ERROR's lake branch (1733-1734) and WATER's
lake balance (6209-6212) are both live, transcribed and reachable through
:func:`sflx_post`.

Layer conventions
-----------------
:class:`gpuwm.core.noahmp_snow.SnowColumn` carries the snow/soil state in WRF's
own ``-NSNOW+1..NSOIL`` convention and is mutated in place, the contract
:func:`gpuwm.core.noahmp_water.water` already uses.  ``smc`` is handled
functionally, as it is there.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from typing import Sequence

import numpy as np

from gpuwm.core.noahmp_energy import (EnergyParameters, EnergyState,
                                      LeafRequest,
                                      drive_on_host as _drive_on_host,
                                      energy_steps as _energy_steps,
                                      register_host_leaf as _register_leaf)
from gpuwm.core.noahmp_leaves import atm
from gpuwm.core.noahmp_snow import SnowColumn
from gpuwm.core.noahmp_vegprecip import phenology, precip_heat
from gpuwm.core.noahmp_water import WaterParameters, water

#: ENERGY's own result record.  Aliased rather than re-declared: the ENERGY
#: lane owns the field inventory, and a second copy here would be one more
#: thing that can drift.
EnergyResult = EnergyState

__all__ = [
    "NSNOW_DEFAULT",
    "NSOIL_DEFAULT",
    "NIGHT_ALBEDO",
    "DVEG",
    "SflxParameters",
    "EnergyCall",
    "EnergyResult",
    "ErrorResult",
    "PreEnergy",
    "SflxResult",
    "NoahmpBalanceError",
    "ShortwaveBalanceError",
    "EnergyBalanceError",
    "WaterBalanceError",
    "error",
    "sflx_pre",
    "sflx_post",
    "sflx",
    "sflx_steps",
    "sflx_post_steps",
]


def _f(x) -> np.float32:
    return np.float32(x)


NSNOW_DEFAULT = 3
NSOIL_DEFAULT = 4

#: The pinned dynamic-vegetation option.  Only the ``DVEG == 4`` arm of
#: 857-870 is transcribed; the others are refused, not approximated.
DVEG = 4

#: NOAHMP_SFLX's own sentinel for a column with no incident shortwave, 1076.
NIGHT_ALBEDO = _f(-999.9)

_ZERO = _f(0.0)
_THOUSAND = _f(1000.0)
_FVEG_FLOOR = _f(0.05)
_SNOW_EPS = _f(1.0e-6)

# ERROR's three abort thresholds, as the binary32 literals the Fortran holds.
ERRSW_TOL = _f(0.01)        # :1641
ERRENG_TOL = _f(0.01)       # :1665
ERRWAT_TOL = _f(0.1)        # :1713


def _fmax(a: np.float32, b: np.float32) -> np.float32:
    """gfortran ``MAX(a, b)``: ``maxss`` keeps the *second* operand on a tie."""
    return a if a > b else b


def _fmin(a: np.float32, b: np.float32) -> np.float32:
    """gfortran ``MIN(a, b)``: ``minss`` keeps the *second* operand on a tie."""
    return a if a < b else b


class _View:
    """A Fortran-indexed view of a 0-based array, so subscripts match source."""

    __slots__ = ("_a", "_lo")

    def __init__(self, a: np.ndarray, lo: int):
        self._a = a
        self._lo = lo

    def __getitem__(self, j: int) -> np.float32:
        return self._a[j - self._lo]

    def __setitem__(self, j: int, v) -> None:
        self._a[j - self._lo] = v


# ---------------------------------------------------------------------------
# The abort gates, as exceptions rather than as a process abort
# ---------------------------------------------------------------------------

class NoahmpBalanceError(RuntimeError):
    """Base for the three ``wrf_error_fatal`` calls ERROR can reach."""


class ShortwaveBalanceError(NoahmpBalanceError):
    """``ABS(ERRSW) > 0.01`` -- ERROR 1641-1660, "Stop in Noah-MP"."""


class EnergyBalanceError(NoahmpBalanceError):
    """``ABS(ERRENG) > 0.01`` -- "Energy budget problem in NOAHMP LSM"."""


class WaterBalanceError(NoahmpBalanceError):
    """``ABS(ERRWAT) > 0.1`` -- "Water budget problem in NOAHMP LSM"."""


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class SflxParameters(WaterParameters):
    """Every ``noahmp_parameters`` component the composition itself reads.

    Extends :class:`gpuwm.core.noahmp_water.WaterParameters` -- which already
    carries the soil block, ``NROOT``, ``CH2OP``, ``SSI``, ``SNOW_RET_FAC`` and
    ``URBAN_FLAG`` -- with PHENOLOGY's seven reads plus ``ISBARREN``, which
    NOAHMP_SFLX reads directly at 874.

    Every added field carries a ``None`` default only because Python forbids a
    non-default field after a defaulted one in a derived dataclass.  All are
    required and ``__post_init__`` refuses a handle that omits one; a silently
    zero soil or vegetation parameter is exactly what this project cannot
    afford.
    """

    isbarren: int = None
    iswater: int = None
    isice: int = None
    hvt: np.float32 = None
    hvb: np.float32 = None
    tmin: np.float32 = None
    laim: Sequence[float] = None
    saim: Sequence[float] = None
    #: ENERGY's own parameter record, which the ENERGY lane owns.
    energy: EnergyParameters = None

    _REQUIRED = ("isbarren", "iswater", "isice", "hvt", "hvb", "tmin",
                 "laim", "saim")

    def __post_init__(self) -> None:
        super().__post_init__()
        missing = [n for n in self._REQUIRED if getattr(self, n) is None]
        if missing:
            raise TypeError("SflxParameters requires " + ", ".join(missing))
        for name in ("isbarren", "iswater", "isice"):
            setattr(self, name, int(getattr(self, name)))
        for name in ("hvt", "hvb", "tmin"):
            setattr(self, name, _f(getattr(self, name)))
        for name in ("laim", "saim"):
            v = np.asarray(getattr(self, name), dtype=np.float32)
            if v.shape != (12,):
                raise ValueError(f"{name} must carry 12 monthly values")
            setattr(self, name, v)

    @classmethod
    def from_transferred(cls, p, nsoil: int = NSOIL_DEFAULT
                         ) -> "SflxParameters":
        """Build the whole handle from one TRANSFER_MP_PARAMETERS column.

        ``p`` is a
        :class:`gpuwm.core.noahmp.NoahmpTransferredParameters`, whose own
        output is pinned in ``gpuwm/data/noahmp/oracle/noahmp-parameters.csv``.
        Every component below is read from it by name, so a table change moves
        this handle rather than being silently absorbed.
        """
        def sc(name):
            return _f(float(p.scalar(name)))

        def vec(name, n):
            return np.asarray([_f(float(p.array(name)[i])) for i in range(n)],
                              dtype=np.float32)

        return cls(
            smcmax=vec("SMCMAX", nsoil), smcwlt=vec("SMCWLT", nsoil),
            bexp=vec("BEXP", nsoil), dksat=vec("DKSAT", nsoil),
            dwsat=vec("DWSAT", nsoil), kdt=sc("KDT"), frzx=sc("FRZX"),
            slope=sc("SLOPE"), ch2op=sc("CH2OP"), urban_flag=p.urban_flag,
            timean=sc("TIMEAN"), fsatmx=sc("FSATMX"),
            nroot=int(p.scalar("NROOT")), ssi=sc("SSI"),
            snow_ret_fac=sc("SNOW_RET_FAC"),
            isbarren=int(p.scalar("ISBARREN")),
            iswater=int(p.scalar("ISWATER")), isice=int(p.scalar("ISICE")),
            hvt=sc("HVT"), hvb=sc("HVB"), tmin=sc("TMIN"),
            laim=vec("LAIM", 12), saim=vec("SAIM", 12),
            energy=EnergyParameters.from_transferred(p, nsoil=nsoil))


# ---------------------------------------------------------------------------
# The ENERGY boundary
# ---------------------------------------------------------------------------

@dataclass
class EnergyCall:
    """Exactly the arguments NOAHMP_SFLX hands ENERGY at 950-976.

    Field names are ENERGY's own dummy-argument names, lower-cased, so a test
    can compare this bundle field for field against the ``in`` role of
    ``gpuwm/data/noahmp/oracle/noahmp-energy.csv``.  Arrays are 0-based numpy
    copies of WRF's ``-NSNOW+1..NSOIL`` / ``1..NSOIL`` / ``1..2`` extents, in
    the same order.

    The INOUT members are carried here as their *entry* values; their exit
    values arrive separately in :class:`EnergyResult`.  ``LAISUN``, ``LAISHA``
    and ``RB`` are INOUT in ENERGY's declaration but NOAHMP_SFLX never assigns
    any of the three before the call, so what reaches ENERGY is the caller's
    own uninitialised local.  WRF's driver zeroes them; this bundle carries
    ``0.0`` and :func:`sflx_pre` names them as such rather than pretending they
    are meaningful state.
    """

    # in
    ice: int
    vegtyp: int
    ist: int
    isnow: int
    dt: np.float32
    rhoair: np.float32
    sfcprs: np.float32
    qair: np.float32
    sfctmp: np.float32
    thair: np.float32
    lwdn: np.float32
    uu: np.float32
    vv: np.float32
    zref: np.float32
    co2air: np.float32
    o2air: np.float32
    solad: np.ndarray
    solai: np.ndarray
    cosz: np.float32
    igs: np.float32
    eair: np.float32
    tbot: np.float32
    zsnso: np.ndarray
    zsoil: np.ndarray
    elai: np.float32
    esai: np.float32
    fwet: np.float32
    foln: np.float32
    fveg: np.float32
    pahv: np.float32
    pahg: np.float32
    pahb: np.float32
    qsnow: np.float32
    dzsnso: np.ndarray
    lat: np.float32
    canliq: np.float32
    canice: np.float32
    # inout, at entry
    tv: np.float32
    tg: np.float32
    stc: np.ndarray
    snowh: np.float32
    eah: np.float32
    tah: np.float32
    sneqvo: np.float32
    sneqv: np.float32
    sh2o: np.ndarray
    smc: np.ndarray
    snice: np.ndarray
    snliq: np.ndarray
    albold: np.float32
    cm: np.float32
    ch: np.float32
    dx: np.float32
    dz8w: np.float32
    q2: np.float32
    tauss: np.float32
    laisun: np.float32
    laisha: np.float32
    rb: np.float32
    qc: np.float32
    qsfc: np.float32
    psfc: np.float32
    acc_ssoil: np.float32
    julian: np.float32
    swdown: np.float32
    prcp: np.float32
    fb: np.float32

    def to_energy_kwargs(self) -> dict:
        """The bundle as ``gpuwm.core.noahmp_energy.energy``'s keywords.

        Seven members are dropped, and both reasons are properties of ENERGY
        rather than conveniences here:

        * ``laisun``, ``laisha`` and ``rb`` -- ENERGY declares all three INOUT
          and overwrites each before any read, so
          :mod:`gpuwm.core.noahmp_energy` refuses them as arguments.
        * ``julian``, ``swdown``, ``prcp`` and ``fb`` -- ENERGY's body
          (2023-2396) mentions each exactly once, in the gecros tail of the
          VEGE_FLUX call at 2260.  ``opt_crop = 0`` kills gecros, so all four
          are inert and that port refuses them too.

        All seven stay members of this bundle because WRF's argument list
        carries them and ``noahmp-energy.csv`` pins them, which is what makes
        the claim checkable instead of asserted.
        """
        drop = {"laisun", "laisha", "rb", "julian", "swdown", "prcp", "fb"}
        return {f.name: getattr(self, f.name)
                for f in dc_fields(self) if f.name not in drop}



# ---------------------------------------------------------------------------
# ERROR -- module_sf_noahmplsm.F:1560-1737
# ---------------------------------------------------------------------------

@dataclass
class ErrorResult:
    """ERROR's five INOUT accumulators, plus the three residuals it tests.

    NOAHMP_SFLX declares ``ERRWAT`` as a local (625) and never reads it after
    the call, so it reaches no WRF output.  It is returned anyway, together
    with ``ERRSW`` and ``ERRENG``: they are what the abort gates test, and a
    port that computed them wrongly but never showed them would pass every
    whole-column check there is.
    """

    errwat: np.float32
    acc_dwater: np.float32
    acc_prcp: np.float32
    acc_ecan: np.float32
    acc_etran: np.float32
    acc_edir: np.float32
    errsw: np.float32
    erreng: np.float32
    end_wb: np.float32


def error(*, swdown, fsa, fsr, fira, fsh, fcev, fgev, fctr, ssoil,
          sav, sag, beg_wb, canliq, canice, sneqv, wa, smc, dzsnso, prcp,
          ecan, etran, edir, runsrf, runsub, dt, qtldrn, ist,
          acc_dwater, acc_prcp, acc_ecan, acc_etran, acc_edir,
          pah=_ZERO, firr=_ZERO, canhs=_ZERO,
          irmirate=_ZERO, irfirate=_ZERO,
          nsoil: int = NSOIL_DEFAULT, nsnow: int = NSNOW_DEFAULT,
          calculate_soil: bool = True) -> ErrorResult:
    """WRF's own shortwave, surface-energy and water balance check.

    Every failure is raised, not printed and survived: all three WRF failure
    paths end in ``wrf_error_fatal``, which stops the process.  Returning
    instead would make this port strictly more permissive than the module it
    claims to reproduce, and the accumulators it would return are the ones the
    aborted step never wrote.

    ``FSRV``, ``FSRG``, ``ZWT``, ``PAHV``, ``PAHG``, ``PAHB``, ``ILOC`` and
    ``JLOC`` are dummy arguments of the Fortran and are deliberately **not**
    arguments here: every one appears only inside a diagnostic ``WRITE`` in a
    block that ends in ``wrf_error_fatal``, so no value surviving the call can
    depend on them.  ``PARAMETERS`` is read nowhere in the body at all.

    ``nsnow`` is accepted because WRF's argument list carries it; the body uses
    it only to declare ``DZSNSO``'s lower bound.  ``dzsnso`` here is the
    ``1..NSOIL`` slice, matching what ERROR's only loop indexes.

    One INTENT(OUT) hazard, reproduced as the defined value: with ``ist == 1``
    and ``calculate_soil`` false, the only statement that assigns ``ERRWAT``
    (1709-1710) is skipped and the lake ``ERRWAT = 0.0`` at 1734 is on the
    other branch, so WRF leaves it holding whatever the caller's storage held.
    ``gpuwm/data/noahmp/oracle/noahmp-sflx-error.csv`` records that directly --
    its ``err_no_soil_substep`` case poisons ``ERRWAT`` before the call and the
    poison comes back out -- and this port returns ``0.0``, the defined value.
    Under the pinned ``soiltstep = 0.0`` the path is unreachable anyway;
    ``calculate_soil`` is an argument here only so that claim can be tested.
    """
    swdown = _f(swdown)
    fsa = _f(fsa)
    fsr = _f(fsr)
    dt = _f(dt)
    prcp = _f(prcp)
    ecan = _f(ecan)
    etran = _f(etran)
    edir = _f(edir)
    runsrf = _f(runsrf)
    runsub = _f(runsub)
    qtldrn = _f(qtldrn)
    beg_wb = _f(beg_wb)
    acc_dwater = _f(acc_dwater)
    acc_prcp = _f(acc_prcp)
    acc_ecan = _f(acc_ecan)
    acc_etran = _f(acc_etran)
    acc_edir = _f(acc_edir)
    smc = np.asarray(smc, dtype=np.float32)
    dzsnso = np.asarray(dzsnso, dtype=np.float32)
    if smc.shape[0] < nsoil or dzsnso.shape[0] < nsoil:
        raise ValueError("smc and dzsnso must cover 1..NSOIL")

    # ERRSW = SWDOWN - (FSA + FSR)                                    :1638
    errsw = _f(swdown - _f(fsa + fsr))
    if abs(errsw) > ERRSW_TOL:                                      # :1641
        raise ShortwaveBalanceError(
            f"Stop in Noah-MP: ERRSW = {float(errsw)!r} W/m2")

    # ERRENG = SAV+SAG-(FIRA+FSH+FCEV+FGEV+FCTR+SSOIL+FIRR+CANHS)+PAH :1662
    sink = _f(_f(fira) + _f(fsh))
    sink = _f(sink + _f(fcev))
    sink = _f(sink + _f(fgev))
    sink = _f(sink + _f(fctr))
    sink = _f(sink + _f(ssoil))
    sink = _f(sink + _f(firr))
    sink = _f(sink + _f(canhs))
    erreng = _f(_f(_f(_f(sav) + _f(sag)) - sink) + _f(pah))
    if abs(erreng) > ERRENG_TOL:                                    # :1665
        raise EnergyBalanceError(
            f"Energy budget problem in NOAHMP LSM: ERRENG = "
            f"{float(erreng)!r} W/m2")

    if int(ist) == 1:                                               # :1696  soil
        end_wb = _f(_f(_f(_f(canliq) + _f(canice)) + _f(sneqv)) + _f(wa))
        for iz in range(nsoil):                                     # :1698
            end_wb = _f(end_wb
                        + _f(_f(smc[iz] * dzsnso[iz]) * _THOUSAND))  # :1699

        acc_dwater = _f(acc_dwater + _f(end_wb - beg_wb))            # :1702
        acc_prcp = _f(acc_prcp + _f(prcp * dt))                      # :1703
        acc_ecan = _f(acc_ecan + _f(ecan * dt))                      # :1704
        acc_etran = _f(acc_etran + _f(etran * dt))                   # :1705
        acc_edir = _f(acc_edir + _f(edir * dt))                      # :1706

        errwat = _ZERO
        if calculate_soil:                                           # :1708
            # ACC_PRCP + IRFIRATE*1000 + IRMIRATE*1000 - ACC_ECAN
            #          - ACC_ETRAN - ACC_EDIR - RUNSRF - RUNSUB - QTLDRN
            inner = _f(acc_prcp + _f(_f(irfirate) * _THOUSAND))       # :1709
            inner = _f(inner + _f(_f(irmirate) * _THOUSAND))
            inner = _f(inner - acc_ecan)
            inner = _f(inner - acc_etran)                             # :1710
            inner = _f(inner - acc_edir)
            inner = _f(inner - runsrf)
            inner = _f(inner - runsub)
            inner = _f(inner - qtldrn)
            errwat = _f(acc_dwater - inner)

            # WRF_HYDRO is not defined, so 1712-1728 is live.
            if abs(errwat) > ERRWAT_TOL:                             # :1713
                direction = ("gaining" if errwat > _ZERO else "losing")
                raise WaterBalanceError(
                    "Water budget problem in NOAHMP LSM: the model is "
                    f"{direction} water, ERRWAT = {float(errwat)!r} "
                    "kg m{-2} timestep{-1}")
    else:                                                            # :1733  lake
        errwat = _ZERO                                               # :1734
        end_wb = _ZERO

    return ErrorResult(
        errwat=errwat, acc_dwater=acc_dwater, acc_prcp=acc_prcp,
        acc_ecan=acc_ecan, acc_etran=acc_etran, acc_edir=acc_edir,
        errsw=errsw, erreng=erreng, end_wb=end_wb)


# ---------------------------------------------------------------------------
# NOAHMP_SFLX, first half -- 806 up to the ENERGY call at 950
# ---------------------------------------------------------------------------

@dataclass
class PreEnergy:
    """What :func:`sflx_pre` computes: the ENERGY bundle, plus the state
    NOAHMP_SFLX holds across the call and the diagnostics it later emits."""

    call: EnergyCall
    dzsnso: np.ndarray
    troot: np.float32
    beg_wb: np.float32
    fveg: np.float32
    elai: np.float32
    esai: np.float32
    igs: np.float32
    fb: np.float32
    lai: np.float32
    sai: np.float32
    bdfall: np.float32
    rain: np.float32
    snow: np.float32
    fp: np.float32
    fpice: np.float32
    prcp: np.float32
    swdown: np.float32
    qprecc: np.float32
    qprecl: np.float32
    qair: np.float32
    rhoair: np.float32
    canliq: np.float32
    canice: np.float32
    fwet: np.float32
    cmc: np.float32
    qintr: np.float32
    qdripr: np.float32
    qthror: np.float32
    qints: np.float32
    qdrips: np.float32
    qthros: np.float32
    pahv: np.float32
    pahg: np.float32
    pahb: np.float32
    qrain: np.float32
    qsnow: np.float32
    snowhin: np.float32


def sflx_pre(parameters: SflxParameters, col: SnowColumn, smc, *,
             lat, yearlen, julian, cosz, dt, dx, dz8w, zsoil,
             shdmax, vegtyp, ice, ist, croptype,
             sfctmp, sfcprs, psfc, uu, vv, q2, qc, soldn, lwdn,
             prcpconv, prcpnonc, prcpshcv, prcpsnow, prcpgrpl, prcphail,
             tbot, co2air, o2air, foln, zlvl,
             albold, sneqvo, tah, eah, canliq, canice, tv, tg, qsfc,
             lai, sai, cm, ch, tauss, pgs, wa, acc_ssoil,
             irrfra=_ZERO, iramtsi=_ZERO, iramtmi=_ZERO,
             iramtfi=_ZERO) -> PreEnergy:
    """NOAHMP_SFLX from its first executable statement to the ENERGY call.

    ``col`` is read, not mutated: everything the first half changes travels in
    the returned :class:`PreEnergy`.

    ``fwet`` is not an argument.  NOAHMP_SFLX declares it INOUT and forwards it
    to ENERGY, but PRECIP_HEAT (940-946) unconditionally assigns it at 1554
    before ENERGY ever sees it, so its entry value is dead.  The whole-column
    fixture drives it from 0.0 and the ENERGY fixture's ``in`` role records
    PRECIP_HEAT's value, which is what makes the omission checkable.
    """
    if int(croptype) != 0:
        raise ValueError(
            "croptype must be 0: opt_crop=0 pins it there and the crop "
            "branches are not transcribed")
    for name, value in (("irrfra", irrfra), ("iramtsi", iramtsi),
                        ("iramtmi", iramtmi), ("iramtfi", iramtfi)):
        if _f(value) != _ZERO:
            raise ValueError(
                f"{name} must be 0.0 under opt_irr=0; the irrigation region "
                "(878-936, 1042-1045) is proved dead, not transcribed")

    nsoil, nsnow = col.nsoil, col.nsnow
    zsoil = np.asarray(zsoil, dtype=np.float32)
    smc = np.asarray(smc, dtype=np.float32)
    vegtyp = int(vegtyp)
    isnow = int(col.isnow)

    # 806-813.  NEE/NPP/GPP and PAH stay at these zeros under DVEG = 4;
    # PAHV/PAHG/PAHB are overwritten by PRECIP_HEAT and CANHS by ENERGY.
    # CROPLU = .FALSE. (813) has no live consumer; see the module docstring.

    # ---- ATM, 819-823 -----------------------------------------------------
    (thair, qair, eair, rhoair, qprecc, qprecl, solad1, solad2,
     solai1, solai2, swdown, bdfall, rain, snow, fp, fpice, prcp) = atm(
        sfcprs, sfctmp, q2, prcpconv, prcpnonc, prcpshcv,
        prcpsnow, prcpgrpl, prcphail, soldn, cosz)
    solad = np.asarray([solad1, solad2], dtype=np.float32)
    solai = np.asarray([solai1, solai2], dtype=np.float32)

    # ---- snow/soil layer thickness, 827-833 -------------------------------
    # Slots below ISNOW+1 are never written by WRF; see the module docstring.
    dzsnso = np.zeros(nsnow + nsoil, dtype=np.float32)
    DZ = _View(dzsnso, -nsnow + 1)
    ZS = col.ZSNSO
    for iz in range(isnow + 1, nsoil + 1):                          # :827
        if iz == isnow + 1:                                         # :828
            DZ[iz] = _f(-ZS[iz])                                    # :829
        else:
            DZ[iz] = _f(ZS[iz - 1] - ZS[iz])                        # :831

    # ---- root-zone temperature, 837-840 -----------------------------------
    # Dead under this identity -- PHENOLOGY never reads TROOT and CARBON is
    # not called -- but computed, because the ENERGY fixture pins it and a
    # silently-wrong dead value is still a wrong transcription.
    STC = col.STC
    nroot = parameters.nroot
    troot = _ZERO                                                   # :837
    denom = _f(-zsoil[nroot - 1])
    for iz in range(1, nroot + 1):                                  # :838
        troot = _f(troot + _f(_f(STC[iz] * DZ[iz]) / denom))        # :839

    # ---- total water storage at step entry, 844-849 -----------------------
    beg_wb = _ZERO
    if int(ist) == 1:                                               # :844
        beg_wb = _f(_f(_f(_f(canliq) + _f(canice)) + _f(col.sneqv))
                    + _f(wa))                                       # :845
        for iz in range(1, nsoil + 1):                              # :846
            beg_wb = _f(beg_wb
                        + _f(_f(smc[iz - 1] * DZ[iz]) * _THOUSAND))  # :847

    # ---- PHENOLOGY, 853-854 -----------------------------------------------
    ph = phenology(
        dveg=DVEG, vegtyp=vegtyp, croptype=int(croptype),
        snowh=float(col.snowh), tv=float(_f(tv)), lat=float(_f(lat)),
        yearlen=int(yearlen), julian=float(_f(julian)),
        lai=float(_f(lai)), sai=float(_f(sai)), troot=float(troot),
        pgs=int(pgs), iswater=parameters.iswater,
        isbarren=parameters.isbarren, isice=parameters.isice,
        urban_flag=bool(parameters.urban_flag),
        hvt=float(parameters.hvt), hvb=float(parameters.hvb),
        tmin=float(parameters.tmin),
        laim=[float(v) for v in parameters.laim],
        saim=[float(v) for v in parameters.saim])
    lai, sai = _f(ph.lai), _f(ph.sai)
    elai, esai = _f(ph.elai), _f(ph.esai)
    igs, fb = _f(ph.igs), _f(ph.fb)

    # ---- FVEG, 857-875 ----------------------------------------------------
    # Only the DVEG == 4 arm is live; 858-862 and 866-870 belong to option
    # values this module does not admit, and 871-872 aborts on the rest.
    fveg = _f(shdmax)                                               # :863
    if fveg <= _FVEG_FLOOR:                                         # :864
        fveg = _FVEG_FLOOR
    # 877-879 (OPT_CROP > 0 .and. CROPTYPE > 0) is dead: opt_crop = 0.
    if parameters.urban_flag or vegtyp == parameters.isbarren:      # :874
        fveg = _ZERO
    if _f(elai + esai) == _ZERO:                                    # :875
        fveg = _ZERO

    # 878-936 -- CROPLU, SIFAC/MIFAC/FIFAC, TRIGGER_IRRIGATION and
    # SPRINKLER_IRRIGATION.  Proved dead in the module docstring; the guards at
    # the top of this function assert the inputs that make it so.

    # ---- PRECIP_HEAT, 940-946 ---------------------------------------------
    pcp = precip_heat(
        iloc=1, jloc=1, vegtyp=vegtyp, ist=int(ist), dt=float(_f(dt)),
        uu=float(_f(uu)), vv=float(_f(vv)), elai=float(elai),
        esai=float(esai), fveg=float(fveg), bdfall=float(bdfall),
        rain=float(rain), snow=float(snow), fp=float(fp),
        canliq=float(_f(canliq)), canice=float(_f(canice)),
        tv=float(_f(tv)), sfctmp=float(_f(sfctmp)), tg=float(_f(tg)),
        ch2op=float(parameters.ch2op))
    canliq, canice = _f(pcp.canliq), _f(pcp.canice)
    fwet, cmc = _f(pcp.fwet), _f(pcp.cmc)
    pahv, pahg, pahb = _f(pcp.pahv), _f(pcp.pahg), _f(pcp.pahb)
    qrain, qsnow, snowhin = _f(pcp.qrain), _f(pcp.qsnow), _f(pcp.snowhin)

    call = EnergyCall(
        ice=int(ice), vegtyp=vegtyp, ist=int(ist), isnow=isnow,
        dt=_f(dt), rhoair=rhoair, sfcprs=_f(sfcprs), qair=qair,
        sfctmp=_f(sfctmp), thair=thair, lwdn=_f(lwdn), uu=_f(uu), vv=_f(vv),
        zref=_f(zlvl), co2air=_f(co2air), o2air=_f(o2air),
        solad=solad, solai=solai, cosz=_f(cosz), igs=igs, eair=eair,
        tbot=_f(tbot), zsnso=col.zsnso.copy(), zsoil=zsoil.copy(),
        elai=elai, esai=esai, fwet=fwet, foln=_f(foln), fveg=fveg,
        pahv=pahv, pahg=pahg, pahb=pahb, qsnow=qsnow,
        dzsnso=dzsnso.copy(), lat=_f(lat), canliq=canliq, canice=canice,
        tv=_f(tv), tg=_f(tg), stc=col.stc.copy(), snowh=_f(col.snowh),
        eah=_f(eah), tah=_f(tah), sneqvo=_f(sneqvo), sneqv=_f(col.sneqv),
        sh2o=col.sh2o.copy(), smc=smc.copy(), snice=col.snice.copy(),
        snliq=col.snliq.copy(), albold=_f(albold), cm=_f(cm), ch=_f(ch),
        dx=_f(dx), dz8w=_f(dz8w), q2=_f(q2), tauss=_f(tauss),
        laisun=_ZERO, laisha=_ZERO, rb=_ZERO,
        qc=_f(qc), qsfc=_f(qsfc), psfc=_f(psfc), acc_ssoil=_f(acc_ssoil),
        julian=_f(julian), swdown=swdown, prcp=prcp, fb=fb)

    return PreEnergy(
        call=call, dzsnso=dzsnso, troot=troot, beg_wb=beg_wb, fveg=fveg,
        elai=elai, esai=esai, igs=igs, fb=fb, lai=lai, sai=sai,
        bdfall=bdfall, rain=rain, snow=snow, fp=fp, fpice=fpice, prcp=prcp,
        swdown=swdown, qprecc=qprecc, qprecl=qprecl, qair=qair,
        rhoair=rhoair, canliq=canliq, canice=canice, fwet=fwet, cmc=cmc,
        qintr=_f(pcp.qintr), qdripr=_f(pcp.qdripr), qthror=_f(pcp.qthror),
        qints=_f(pcp.qints), qdrips=_f(pcp.qdrips), qthros=_f(pcp.qthros),
        pahv=pahv, pahg=pahg, pahb=pahb,
        qrain=qrain, qsnow=qsnow, snowhin=snowhin)


# ---------------------------------------------------------------------------
# NOAHMP_SFLX, second half -- 979 through 1076
# ---------------------------------------------------------------------------

@dataclass
class SflxResult:
    """Everything NOAHMP_SFLX leaves behind that the column does not carry.

    The snow/soil column itself (``ISNOW SNOWH SNEQV SNICE SNLIQ STC ZSNSO
    SH2O``) is mutated in place on the :class:`SnowColumn` handed in, matching
    :func:`gpuwm.core.noahmp_water.water`.  ``SMC`` is functional and comes back
    here, as it does there.
    """

    # prognostic pass-through / update
    smc: np.ndarray
    albold: np.float32
    sneqvo: np.float32
    tah: np.float32
    eah: np.float32
    fwet: np.float32
    canliq: np.float32
    canice: np.float32
    tv: np.float32
    tg: np.float32
    qsfc: np.float32
    qsnow: np.float32
    qrain: np.float32
    wslake: np.float32
    lai: np.float32
    sai: np.float32
    cm: np.float32
    ch: np.float32
    tauss: np.float32
    qtldrn: np.float32
    acc_ssoil: np.float32
    acc_qinsur: np.float32
    acc_qseva: np.float32
    acc_etrani: np.ndarray
    acc_dwater: np.float32
    acc_prcp: np.float32
    acc_ecan: np.float32
    acc_etran: np.float32
    acc_edir: np.float32
    # INTENT(OUT)
    z0wrf: np.float32
    fsa: np.float32
    fsr: np.float32
    fira: np.float32
    fsh: np.float32
    ssoil: np.float32
    fcev: np.float32
    fgev: np.float32
    fctr: np.float32
    ecan: np.float32
    etran: np.float32
    edir: np.float32
    trad: np.float32
    tgb: np.float32
    tgv: np.float32
    t2mv: np.float32
    t2mb: np.float32
    q2v: np.float32
    q2b: np.float32
    runsrf: np.float32
    runsub: np.float32
    apar: np.float32
    psn: np.float32
    sav: np.float32
    sag: np.float32
    fsno: np.float32
    nee: np.float32
    gpp: np.float32
    npp: np.float32
    fveg: np.float32
    albedo: np.float32
    qsnbot: np.float32
    ponding: np.float32
    ponding1: np.float32
    ponding2: np.float32
    rssun: np.float32
    rssha: np.float32
    albsnd: np.ndarray
    albsni: np.ndarray
    bgap: np.float32
    wgap: np.float32
    chv: np.float32
    chb: np.float32
    emissi: np.float32
    shg: np.float32
    shc: np.float32
    shb: np.float32
    evg: np.float32
    evb: np.float32
    ghv: np.float32
    ghb: np.float32
    irg: np.float32
    irc: np.float32
    irb: np.float32
    tr: np.float32
    evc: np.float32
    chleaf: np.float32
    chuc: np.float32
    chv2: np.float32
    chb2: np.float32
    fpice: np.float32
    pahv: np.float32
    pahg: np.float32
    pahb: np.float32
    pah: np.float32
    laisun: np.float32
    laisha: np.float32
    rb: np.float32
    qints: np.float32
    qintr: np.float32
    qdrips: np.float32
    qdripr: np.float32
    qthros: np.float32
    qthror: np.float32
    qsnsub: np.float32
    qsnfro: np.float32
    qsubc: np.float32
    qfroc: np.float32
    qfrzc: np.float32
    qmeltc: np.float32
    qevac: np.float32
    qdewc: np.float32
    qmelt: np.float32
    rain: np.float32
    snow: np.float32
    hcpct: np.ndarray
    eflxb: np.float32
    canhs: np.float32
    # locals the fixtures do not carry but a port must still be held to
    errwat: np.float32
    beg_wb: np.float32
    end_wb: np.float32
    troot: np.float32
    qvap: np.float32
    qdew: np.float32
    cmc: np.float32
    prcp: np.float32
    swdown: np.float32


def sflx_post(*args, **kwargs) -> SflxResult:
    """NOAHMP_SFLX after ENERGY, host-driven.

    Identical arithmetic to :func:`sflx_post_steps`; the paused WATER call is
    answered here by the CPython routine.
    """
    return _drive_on_host(sflx_post_steps(*args, **kwargs))


def sflx_post_steps(parameters: SflxParameters, col: SnowColumn,
                    pre: PreEnergy, en: EnergyResult, *, dt, ist, zsoil,
                    ficeold, wa, wslake, acc_qinsur, acc_qseva, acc_etrani,
                    acc_dwater, acc_prcp, acc_ecan, acc_etran, acc_edir,
                    calculate_soil: bool = True):
    """NOAHMP_SFLX from ENERGY's return (979) to the end of the routine (1076).

    ``col`` is updated in place with ENERGY's exit column and then handed to
    WATER, which mutates it further; that is WRF's own aliasing, since
    NOAHMP_SFLX passes the same ``STC``/``SNICE``/``SNLIQ``/``ZSNSO``/``SH2O``
    arrays to both.  The device WATER batch reproduces that aliasing exactly --
    it writes the same nine arrays of the same ``col`` object -- so a caller
    cannot tell the two apart by identity either.

    The generator yields one :class:`~gpuwm.core.noahmp_energy.LeafRequest` at
    WATER and returns the :class:`SflxResult`.
    """
    nsoil, nsnow = col.nsoil, col.nsnow
    dt = _f(dt)
    ist = int(ist)

    # ENERGY's INOUT column lands back on the caller's arrays.
    col.stc[:] = np.asarray(en.stc, dtype=np.float32)
    col.snice[:] = np.asarray(en.snice, dtype=np.float32)
    col.snliq[:] = np.asarray(en.snliq, dtype=np.float32)
    col.sh2o[:] = np.asarray(en.sh2o, dtype=np.float32)
    col.snowh = _f(en.snowh)
    col.sneqv = _f(en.sneqv)
    col.dzsnso[:] = pre.dzsnso
    smc = np.asarray(en.smc, dtype=np.float32).copy()

    # ---- 979-984 ----------------------------------------------------------
    sice = np.empty(nsoil, dtype=np.float32)
    for iz in range(nsoil):
        sice[iz] = _fmax(_ZERO, _f(smc[iz] - col.sh2o[iz]))          # :979
    col.sice[:] = sice
    sneqvo = _f(col.sneqv)                                           # :980

    ratio = _f(_f(en.fgev) / _f(en.latheag))
    qvap = _fmax(ratio, _ZERO)                                       # :982
    qdew = abs(_fmin(ratio, _ZERO))                                  # :983
    edir = _f(qvap - qdew)                                           # :984

    # ---- WATER, 988-1009 --------------------------------------------------
    # IMELT is ENERGY's ``-NSNOW+1:NSOIL``; SNOWWATER's COMPACT indexes only
    # the snow slots, which is the extent noahmp_snow.compact declares.
    wat = yield LeafRequest("water", (
        parameters, col, smc, dt, ist,
        [int(v) for v in np.asarray(en.imelt)[:nsnow]],
        np.asarray(ficeold, dtype=np.float32),
        en.fcev, en.fctr, pre.elai, pre.esai, pre.fveg, pre.bdfall,
        bool(en.frozen_canopy), bool(en.frozen_ground),
        pre.call.sfctmp, qvap, qdew, zsoil, np.asarray(en.btrani),
        pre.qsnow, pre.qrain, pre.snowhin, en.ponding,
        pre.canliq, pre.canice, en.tv, wslake,
        acc_qinsur, acc_qseva, acc_etrani), {})

    # ---- CARBON, 1014-1036 ------------------------------------------------
    # dveg_active is .false. under DVEG = 4 and crop_active is .false. under
    # OPT_CROP = 0, so neither CARBON nor CARBON_CROP is reached and
    # NEE/GPP/NPP keep the zeros written at 806-808.
    nee = gpp = npp = _ZERO

    # ---- 1042-1045 --------------------------------------------------------
    # PRCP += IRSIRATE*1000/DT and FSH -= FIRR, both exactly zero-valued under
    # OPT_IRR = 0; see the module docstring.
    prcp = pre.prcp
    fsh = _f(en.fsh)

    # ---- ERROR, 1049-1058 -------------------------------------------------
    err = error(
        swdown=pre.swdown, fsa=en.fsa, fsr=en.fsr, fira=en.fira, fsh=fsh,
        fcev=en.fcev, fgev=en.fgev, fctr=en.fctr, ssoil=en.ssoil,
        sav=en.sav, sag=en.sag, beg_wb=pre.beg_wb,
        canliq=wat.canliq, canice=wat.canice, sneqv=col.sneqv, wa=wa,
        smc=wat.smc, dzsnso=col.dzsnso[nsnow:], prcp=prcp,
        ecan=wat.ecan, etran=wat.etran, edir=edir,
        runsrf=wat.runsrf, runsub=wat.runsub, dt=dt, qtldrn=wat.qtldrn,
        ist=ist, acc_dwater=acc_dwater, acc_prcp=acc_prcp,
        acc_ecan=acc_ecan, acc_etran=acc_etran, acc_edir=acc_edir,
        pah=en.pah, canhs=en.canhs,
        nsoil=nsoil, nsnow=nsnow, calculate_soil=calculate_soil)

    # ---- 1061-1076 --------------------------------------------------------
    qsfc = _f(en.qsfc)
    q2b = _f(en.q2b)
    if parameters.urban_flag:                                        # :1062
        qfx = _f(_f(wat.etran + wat.ecan) + edir)                    # :1061
        qsfc = _f(_f(qfx / _f(pre.rhoair * _f(en.ch))) + pre.qair)   # :1063
        q2b = qsfc                                                   # :1064

    snowh = _f(col.snowh)
    sneqv = _f(col.sneqv)
    if snowh <= _SNOW_EPS or sneqv <= _SNOW_EPS:                     # :1067
        snowh = _ZERO                                                # :1068
        sneqv = _ZERO                                                # :1069
    col.snowh = snowh
    col.sneqv = sneqv

    if pre.swdown != _ZERO:                                          # :1072
        albedo = _f(_f(en.fsr) / pre.swdown)                         # :1073
    else:
        albedo = NIGHT_ALBEDO                                        # :1075

    return SflxResult(
        smc=wat.smc, albold=_f(en.albold), sneqvo=sneqvo, tah=_f(en.tah),
        eah=_f(en.eah), fwet=wat.fwet, canliq=wat.canliq,
        canice=wat.canice, tv=wat.tv, tg=_f(en.tg), qsfc=qsfc,
        qsnow=pre.qsnow, qrain=pre.qrain, wslake=wat.wslake,
        lai=pre.lai, sai=pre.sai, cm=_f(en.cm), ch=_f(en.ch),
        tauss=_f(en.tauss), qtldrn=wat.qtldrn, acc_ssoil=_f(en.acc_ssoil),
        acc_qinsur=wat.acc_qinsur, acc_qseva=wat.acc_qseva,
        acc_etrani=wat.acc_etrani, acc_dwater=err.acc_dwater,
        acc_prcp=err.acc_prcp, acc_ecan=err.acc_ecan,
        acc_etran=err.acc_etran, acc_edir=err.acc_edir,
        z0wrf=_f(en.z0wrf), fsa=_f(en.fsa), fsr=_f(en.fsr),
        fira=_f(en.fira), fsh=fsh, ssoil=_f(en.ssoil), fcev=_f(en.fcev),
        fgev=_f(en.fgev), fctr=_f(en.fctr), ecan=wat.ecan,
        etran=wat.etran, edir=edir, trad=_f(en.trad), tgb=_f(en.tgb),
        tgv=_f(en.tgv), t2mv=_f(en.t2mv), t2mb=_f(en.t2mb),
        q2v=_f(en.q2v), q2b=q2b, runsrf=wat.runsrf, runsub=wat.runsub,
        apar=_f(en.apar), psn=_f(en.psn), sav=_f(en.sav), sag=_f(en.sag),
        fsno=_f(en.fsno), nee=nee, gpp=gpp, npp=npp, fveg=pre.fveg,
        albedo=albedo, qsnbot=wat.qsnbot, ponding=_f(en.ponding),
        ponding1=wat.ponding1, ponding2=wat.ponding2,
        rssun=_f(en.rssun), rssha=_f(en.rssha),
        albsnd=np.asarray(en.albsnd, dtype=np.float32),
        albsni=np.asarray(en.albsni, dtype=np.float32),
        bgap=_f(en.bgap), wgap=_f(en.wgap), chv=_f(en.chv), chb=_f(en.chb),
        emissi=_f(en.emissi), shg=_f(en.shg), shc=_f(en.shc),
        shb=_f(en.shb), evg=_f(en.evg), evb=_f(en.evb), ghv=_f(en.ghv),
        ghb=_f(en.ghb), irg=_f(en.irg), irc=_f(en.irc), irb=_f(en.irb),
        tr=_f(en.tr), evc=_f(en.evc), chleaf=_f(en.chleaf),
        chuc=_f(en.chuc), chv2=_f(en.chv2), chb2=_f(en.chb2),
        fpice=pre.fpice, pahv=pre.pahv, pahg=pre.pahg, pahb=pre.pahb,
        pah=_f(en.pah), laisun=_f(en.laisun), laisha=_f(en.laisha),
        rb=_f(en.rb), qints=pre.qints, qintr=pre.qintr,
        qdrips=pre.qdrips, qdripr=pre.qdripr, qthros=pre.qthros,
        qthror=pre.qthror, qsnsub=wat.qsnsub, qsnfro=wat.qsnfro,
        qsubc=wat.qsubc, qfroc=wat.qfroc, qfrzc=wat.qfrzc,
        qmeltc=wat.qmeltc, qevac=wat.qevac, qdewc=wat.qdewc,
        qmelt=_f(en.qmelt), rain=pre.rain, snow=pre.snow,
        hcpct=np.asarray(en.hcpct, dtype=np.float32), eflxb=_f(en.eflxb),
        canhs=_f(en.canhs),
        errwat=err.errwat, beg_wb=pre.beg_wb, end_wb=err.end_wb,
        troot=pre.troot, qvap=qvap, qdew=qdew, cmc=wat.cmc,
        prcp=prcp, swdown=pre.swdown)


# ---------------------------------------------------------------------------
# The whole column
# ---------------------------------------------------------------------------

def sflx(*args, **kwargs) -> SflxResult:
    """One NOAHMP_SFLX column, host-driven.

    Identical arithmetic to :func:`sflx_steps`; the two paused leaf calls are
    answered here by the CPython routines.  This is the signature every
    fixture, oracle and non-batched caller uses.
    """
    return _drive_on_host(sflx_steps(*args, **kwargs))


def sflx_steps(parameters: SflxParameters, col: SnowColumn, smc, *,
               ficeold, wa, wslake, ponding=_ZERO,
               acc_ssoil=_ZERO, acc_qinsur=_ZERO, acc_qseva=_ZERO,
               acc_etrani=None, acc_dwater=_ZERO, acc_prcp=_ZERO,
               acc_ecan=_ZERO, acc_etran=_ZERO, acc_edir=_ZERO,
               calculate_soil: bool = True, **forcing):
    """One NOAHMP_SFLX column under the pinned option identity, as a generator.

    It yields :class:`gpuwm.core.noahmp_energy.LeafRequest` objects from inside
    ENERGY and returns the :class:`SflxResult`; see :func:`sflx` for the
    ordinary call, and :func:`gpuwm.core.noahmp_energy.energy_steps` for what
    the seam is and is not.

    ``forcing`` is :func:`sflx_pre`'s keyword set verbatim; it is forwarded
    rather than re-declared so the two signatures cannot drift apart.

    ``col`` is mutated in place, as it is in :func:`sflx_post`.  ``ponding`` is
    an argument because NOAHMP_SFLX declares it INTENT(OUT) and ENERGY assigns
    it before WATER reads it -- the entry value is dead and this argument
    exists only so a caller can prove that, which
    ``tests/test_noahmp_sflx.py::test_entry_ponding_is_dead`` does.
    """
    if parameters.energy is None:
        raise TypeError(
            "SflxParameters.energy is required to run the whole column; build "
            "the handle with SflxParameters.from_transferred()")
    if acc_etrani is None:
        acc_etrani = np.zeros(col.nsoil, dtype=np.float32)

    # NOAHMP_SFLX's first half pauses here for the same reason every leaf
    # inside ENERGY does: so a scheduler can answer it for many columns at
    # once.  The host answer is :func:`sflx_pre` itself, unchanged, which is
    # what makes ``evaluate_leaf_batch_on_host`` the paired authority for the
    # device batch rather than a second transcription of the prefix.
    pre = yield LeafRequest("sflx_pre", (parameters, col, smc),
                            dict(acc_ssoil=acc_ssoil, wa=wa, **forcing))
    en = yield from _energy_steps(parameters.energy, nsnow=col.nsnow,
                                  nsoil=col.nsoil,
                                  **pre.call.to_energy_kwargs())
    return (yield from sflx_post_steps(
        parameters, col, pre, en,
        dt=forcing["dt"], ist=forcing["ist"], zsoil=forcing["zsoil"],
        ficeold=ficeold, wa=wa, wslake=wslake,
        acc_qinsur=acc_qinsur, acc_qseva=acc_qseva, acc_etrani=acc_etrani,
        acc_dwater=acc_dwater, acc_prcp=acc_prcp, acc_ecan=acc_ecan,
        acc_etran=acc_etran, acc_edir=acc_edir,
        calculate_soil=calculate_soil))


#: See the note on :data:`gpuwm.core.noahmp_energy.energy.__wrapped__`: the
#: drivers forward their whole argument list, and introspection must say so or
#: an "argument is absent" assertion becomes unfailable.
sflx.__wrapped__ = sflx_steps
sflx_post.__wrapped__ = sflx_post_steps

#: WATER is paused outside ENERGY, so ENERGY cannot know its host answer
#: without importing this module.  Bind it here instead.  The same is true of
#: NOAHMP_SFLX's own first half: ``sflx_pre`` is the host answer for the
#: ``sflx_pre`` request :func:`sflx_steps` yields, so the host authority runs
#: the identical routine the WRF fixtures pin.
_register_leaf("water", water)
_register_leaf("sflx_pre", sflx_pre)
