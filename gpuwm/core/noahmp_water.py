"""Noah-MP WATER, the whole-column water assembly, in FP32.

Ports ``MODULE_SF_NOAHMPLSM``'s WATER (5954-6261) from
``phys/module_sf_noahmplsm.F`` at WRF commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``
(``sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282``).

``kind_phys == kind(1.0)`` in that build, so every quantity here is IEEE
binary32 and every arithmetic boundary is pinned to ``numpy.float32``.

WATER contains no physics of its own beyond nine statements: it calls
CANWATER, splits the surface vapour flux between the snowpack and the soil,
calls SNOWWATER, moves the frozen-ground surface exchange onto ``SICE(1)``,
converts mm/s to m/s, accumulates the soil-timestep averages and calls
SOILWATER (or the two-line lake balance).  Every one of those callees is
already pinned bitwise elsewhere in this package and is **imported**, not
re-transcribed:

* :func:`gpuwm.core.noahmp_soilwater.canwater` and
  :func:`gpuwm.core.noahmp_soilwater.soilwater` (which reaches INFIL, SRT,
  SSTEP, WDFCND1, WDFCND2 and ROSR12);
* :func:`gpuwm.core.noahmp_snow.snowwater` (which reaches SNOWFALL, COMPACT,
  COMBINE, DIVIDE, COMBO and SNOWH2O).

No transcendental is evaluated in this module.

Pinned option identity
----------------------
``dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0``, plus
``soiltstep=0.0`` which ``module_sf_noahmpdrv.F`` (drivers/wrf, 648-676)
turns into ``soil_update_steps = 1`` and ``calculate_soil = .true.``.

Everything those kill is absent here, not stubbed:

* ``opt_run=3`` selects ``RUNSUB = RUNSUB + QDRAIN`` (6233-6236) as the only
  live baseflow form and kills GROUNDWATER (6225-6231) and SHALLOWWATERTABLE
  (6242-6250), together with ZWTEQ, COMPUTE_VIC_SURFRUNOFF,
  COMPUTE_XAJ_SURFRUNOFF and DYNAMIC_VIC, which nothing else reaches.
* ``opt_irr=0`` kills FLOOD_IRRIGATION (6188-6193) and MICRO_IRRIGATION
  (6196-6202).  Their gates read ``IRAMTFI``/``IRAMTMI``, not ``OPT_IRR``, so
  the kill runs through the caller: TRIGGER_IRRIGATION takes its
  ``OPT_IRR .LT. 1`` branch at 9291, sets ``IRR_ACTIVE = .false.`` and zeroes
  all three irrigation amounts at 9344-9346, and NOAHMP_SFLX zeroes them again
  at 919-923 whenever ``IRRFRA < parameters%IRR_FRAC``.  WATER therefore cannot
  be reached with either amount above zero.  ``CROPLU``, ``IRRFRA``, ``MIFAC``,
  ``FIFAC``, ``IRFIRATE`` and ``IRMIRATE`` are consequently not arguments here;
  ``wat_inert_probe`` and ``wat_lake_inert_probe`` hold that claim.
* ``opt_tdrn=0`` kills the tile drain, so ``QTLDRN`` is zeroed at 6112 and
  scaled by ``DT_soil`` at 6254 and is identically ``0.0``.
* ``WRF_HYDRO`` is not defined, so the ``sfcheadrt`` term at 6174 does not
  exist.
* ``soil_update_steps == 1`` makes ``DT_soil == DT`` (6185) and the three
  divisions at 6204-6206 the identity.  They are written as such below,
  which is exact rather than an approximation: ``x / 1.0`` is ``x`` for every
  finite binary32 ``x``.

Arguments the pinned identity does not consume
----------------------------------------------
Never referenced anywhere in WATER's body: ``UU``, ``VV``, ``QPRECC``,
``QPRECL``, ``FP``, ``RAIN``, ``SNOW``, ``LATHEAV``, ``LATHEAG``, ``IRRFRA``.
Referenced only inside dead branches: ``SMCEQ`` (6244), ``WA`` (6228, 6249),
``WT`` (6228), ``RECH`` (6245).  Forwarded to callees where they are inert in
turn: ``VEGTYP``, ``TG``, ``ILOC``, ``JLOC``, ``DX``, ``TDFRACMP``, ``ZWT``,
``SMCWTD``, ``DEEPRECH``.  None of them is an argument here.

The INTENT(OUT) hazards, reproduced rather than tidied
------------------------------------------------------
* ``QIN`` (6052) and ``QDIS`` (6053) are INTENT(OUT) and the only statement
  that assigns either is inside the ``OPT_RUN==1`` GROUNDWATER call.  Under
  ``opt_run=3`` they are never written.  gfortran passes scalar dummies by
  reference, so the caller's value stands unchanged; NOAHMP_SFLX declares both
  as uninitialised locals (671-672) and reads neither afterwards, so nothing
  downstream consumes them.  They are therefore not outputs of this function.
  ``wat_out_entry_probe`` pins the pass-through, and
  :mod:`tests.test_noahmp_water` asserts entry equals exit on every case, which
  is what makes the omission checkable rather than convenient.
* ``QSNSUB`` and ``QSNFRO`` are dummy arguments declared without INTENT in
  WATER's local block (6086-6087).  They are unconditionally assigned at
  6126/6132, so their entry values are dead and they are pure outputs here.
* SOILWATER's ``RUNSUB`` aliasing (7549 reads before assigning) is invisible
  from WATER because 6109 zeroes ``RUNSUB`` first.  That is exactly the fact
  which makes SOILWATER's forecast path defined, and it is why ``runsub`` is
  not an argument here.

Layer conventions
-----------------
:class:`gpuwm.core.noahmp_snow.SnowColumn` carries the snow/soil state in
WRF's own ``-NSNOW+1..NSOIL`` convention and is mutated in place, the same
contract :func:`gpuwm.core.noahmp_snow.snowwater` already uses.  ``smc`` and
the accumulators are handled functionally and come back in :class:`WaterFluxes`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gpuwm.core.noahmp_snow import SnowColumn, snowwater
from gpuwm.core.noahmp_soilwater import SoilParameters, canwater, soilwater

__all__ = [
    "NSNOW_DEFAULT",
    "NSOIL_DEFAULT",
    "WSLMAX",
    "WaterParameters",
    "WaterFluxes",
    "water",
]


def _f(x) -> np.float32:
    return np.float32(x)


NSNOW_DEFAULT = 3
NSOIL_DEFAULT = 4

_ZERO = _f(0.0)
_THOUSAND = _f(1000.0)
_MILLI = _f(0.001)

# --- WATER's own PARAMETER, line 6098 --------------------------------------
WSLMAX = _f(5000.0)     # maximum lake water storage (mm)


@dataclass
class WaterParameters(SoilParameters):
    """The ``noahmp_parameters`` components the whole assembly consumes.

    Extends :class:`gpuwm.core.noahmp_soilwater.SoilParameters` with the three
    WATER and SNOWWATER read that the soil-water group does not: ``NROOT``
    (6169), and ``SSI``/``SNOW_RET_FAC``, which SNOWH2O reads at 7169 and 7189.

    The three carry ``None`` defaults purely because Python forbids a
    non-default field after a defaulted one in a derived dataclass; every one
    is required and ``__post_init__`` refuses a handle that omits it.  A
    silently-zero soil parameter is exactly the kind of thing this project
    cannot afford.
    """

    nroot: int = None
    ssi: np.float32 = None
    snow_ret_fac: np.float32 = None

    def __post_init__(self) -> None:
        super().__post_init__()
        missing = [n for n in ("nroot", "ssi", "snow_ret_fac")
                   if getattr(self, n) is None]
        if missing:
            raise TypeError(
                "WaterParameters requires " + ", ".join(missing))
        self.nroot = int(self.nroot)
        if not 1 <= self.nroot <= len(self.smcmax):
            raise ValueError(f"nroot out of range: {self.nroot}")
        self.ssi = _f(self.ssi)
        self.snow_ret_fac = _f(self.snow_ret_fac)


@dataclass
class WaterFluxes:
    """Everything WATER returns that is not carried by the column itself."""

    smc: np.ndarray
    canliq: np.float32
    canice: np.float32
    tv: np.float32
    wslake: np.float32
    acc_qinsur: np.float32
    acc_qseva: np.float32
    acc_etrani: np.ndarray
    cmc: np.float32
    ecan: np.float32
    etran: np.float32
    fwet: np.float32
    runsrf: np.float32
    runsub: np.float32
    qtldrn: np.float32
    ponding1: np.float32
    ponding2: np.float32
    qsnbot: np.float32
    qsnsub: np.float32
    qsnfro: np.float32
    qsubc: np.float32
    qfroc: np.float32
    qfrzc: np.float32
    qmeltc: np.float32
    qevac: np.float32
    qdewc: np.float32
    # Locals that never reach a WRF output but decide a branch; the oracle's
    # probe stage pins them, so they are returned rather than recomputed by a
    # test that would then be checking itself.
    qseva: np.float32 = field(default=_ZERO)
    qsdew: np.float32 = field(default=_ZERO)
    qinsur: np.float32 = field(default=_ZERO)
    snoflow: np.float32 = field(default=_ZERO)
    etrani: np.ndarray = field(default=None)


# ---------------------------------------------------------------------------
# WATER -- module_sf_noahmplsm.F:5954-6261
# ---------------------------------------------------------------------------

def water(parameters, col: SnowColumn, smc, dt, ist, imelt, ficeold,
          fcev, fctr, elai, esai, fveg, bdfall, frozen_canopy, frozen_ground,
          sfctmp, qvap, qdew, zsoil, btrani, qsnow, qrain, snowhin, ponding,
          canliq, canice, tv, wslake, acc_qinsur, acc_qseva,
          acc_etrani) -> WaterFluxes:
    """One WATER call under the pinned option identity.

    ``col`` is mutated in place -- ``isnow``, ``snowh``, ``sneqv``, ``snice``,
    ``snliq``, ``stc``, ``zsnso``, ``dzsnso``, ``sh2o`` and ``sice`` are all
    INOUT in WRF and :func:`gpuwm.core.noahmp_snow.snowwater` already owns that
    contract.  Everything else comes back in the returned :class:`WaterFluxes`.

    ``ist`` is 1 for soil and 2 for lake; both are live under this identity,
    because it is a land-use property and not an option.
    """
    nsoil, nsnow = col.nsoil, col.nsnow

    dt = _f(dt)
    qvap = _f(qvap)
    qdew = _f(qdew)
    qrain = _f(qrain)
    ponding = _f(ponding)
    wslake = _f(wslake)
    acc_qinsur = _f(acc_qinsur)
    acc_qseva = _f(acc_qseva)
    smc = np.asarray(smc, dtype=np.float32).copy()
    zsoil = np.asarray(zsoil, dtype=np.float32)
    btrani = np.asarray(btrani, dtype=np.float32)
    acc_etrani = np.asarray(acc_etrani, dtype=np.float32).copy()

    etrani = np.zeros(nsoil, dtype=np.float32)                      # :6107
    snoflow = _ZERO                                                 # :6108
    runsub = _ZERO                                                  # :6109
    runsrf = _ZERO                                                  # :6110
    qinsur = _ZERO                                                  # :6111
    qtldrn = _ZERO                                                  # :6112

    # canopy-intercepted snowfall/rainfall, drips, and throughfall     :6116
    (canliq, canice, tv, cmc, ecan, etran, fwet,
     qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc) = canwater(
        parameters, dt, fcev, fctr, elai, esai, fveg, bdfall,
        frozen_canopy, canliq, canice, tv)

    # sublimation, frost, evaporation, and dew
    qsnsub = _ZERO                                                  # :6126
    if col.sneqv > _ZERO:                                           # :6127
        qsnsub = min(qvap, _f(col.sneqv / dt))                      # :6128
    qseva = _f(qvap - qsnsub)                                       # :6130

    qsnfro = _ZERO                                                  # :6132
    if col.sneqv > _ZERO:                                           # :6133
        qsnfro = qdew                                               # :6134
    qsdew = _f(qdew - qsnfro)                                       # :6136

    qsnbot, snoflow, ponding1, ponding2 = snowwater(                # :6138
        col, dt, zsoil, imelt, ficeold, sfctmp, snowhin, qsnow,
        qsnfro, qsnsub, qrain, parameters.ssi, parameters.snow_ret_fac)

    # SNOWWATER restores DZSNSO(1:NSOIL) to a positive thickness before it
    # returns, which is what 6146 below indexes.
    dz = col.dzsnso[nsnow:]
    sh2o = col.sh2o
    sice = col.sice

    if frozen_ground:                                               # :6145
        sice[0] = _f(sice[0] + _f(_f(_f(qsdew - qseva) * dt)
                                  / _f(dz[0] * _THOUSAND)))         # :6146
        qsdew = _ZERO                                               # :6147
        qseva = _ZERO                                               # :6148
        if sice[0] < _ZERO:                                         # :6149
            sh2o[0] = _f(sh2o[0] + sice[0])                         # :6150
            sice[0] = _ZERO                                         # :6151
        smc[0] = _f(sh2o[0] + sice[0])                              # :6153

    # convert units (mm/s -> m/s).  PONDING is melt water from snow when there
    # is no layer; WATER reads it and never writes it.
    qinsur = _f(_f(_f(_f(ponding + ponding1) + ponding2) / dt) * _MILLI)  # :6159

    if col.isnow == 0:                                              # :6161
        qinsur = _f(qinsur + _f(_f(_f(qsnbot + qsdew) + qrain) * _MILLI))  # :6162
    else:                                                           # :6163
        qinsur = _f(qinsur + _f(_f(qsnbot + qsdew) * _MILLI))       # :6164

    qseva = _f(qseva * _MILLI)                                      # :6167

    for iz in range(parameters.nroot):                              # :6169
        etrani[iz] = _f(_f(etran * btrani[iz]) * _MILLI)            # :6170

    # added soil timestep capability
    acc_qinsur = _f(acc_qinsur + qinsur)                            # :6178
    acc_qseva = _f(acc_qseva + qseva)                               # :6179
    acc_etrani = acc_etrani + etrani                                # :6180

    # `if (calculate_soil)` at 6183 is always true under soiltstep=0.0.
    dt_soil = dt                                                    # :6185

    # The two irrigation blocks at 6188-6202 are unreachable; see the module
    # docstring.

    # soil_update_steps == 1, so all three divisions are the identity.
    qseva_avg = acc_qseva                                           # :6204
    qinsur_avg = acc_qinsur                                         # :6205
    etrani_avg = acc_etrani                                         # :6206

    if ist == 2:                                                    # :6209  lake
        runsrf = _ZERO                                              # :6210
        if wslake >= WSLMAX:                                        # :6211
            runsrf = _f(_f(qinsur_avg * _THOUSAND) * dt_soil)
        wslake = _f(_f(wslake
                       + _f(_f(_f(qinsur_avg - qseva_avg) * _THOUSAND)
                            * dt_soil))
                    - runsrf)                                       # :6212
        # QDRAIN, WCND and FCRMAX stay undefined on this branch and nothing
        # reads them: 6235 and 6252-6254 are inside the ELSE.
    else:                                                           # :6213  soil
        sh2o_new, smc, runsrf, qdrain, runsub, _wcnd, _fcrmax = soilwater(
            parameters, dt_soil, zsoil, col.dzsnso, qinsur_avg, qseva_avg,
            etrani_avg, sice, sh2o, smc, runsub,
            nsoil=nsoil, nsnow=nsnow)                               # :6214
        sh2o[:] = sh2o_new

        # OPT_RUN==1 GROUNDWATER at 6225-6231 is dead.
        runsub = _f(runsub + qdrain)                                # :6235

        for iz in range(nsoil):                                     # :6238
            smc[iz] = _f(sh2o[iz] + sice[iz])                       # :6239

        # OPT_RUN==5 SHALLOWWATERTABLE at 6242-6250 is dead.

        runsrf = _f(runsrf * dt_soil)                               # :6252
        runsub = _f(runsub * dt_soil)                               # :6253
        qtldrn = _f(qtldrn * dt_soil)                               # :6254

    runsub = _f(runsub + _f(snoflow * dt))                          # :6259

    return WaterFluxes(
        smc=smc, canliq=canliq, canice=canice, tv=tv, wslake=wslake,
        acc_qinsur=acc_qinsur, acc_qseva=acc_qseva, acc_etrani=acc_etrani,
        cmc=cmc, ecan=ecan, etran=etran, fwet=fwet,
        runsrf=runsrf, runsub=runsub, qtldrn=qtldrn,
        ponding1=ponding1, ponding2=ponding2, qsnbot=qsnbot,
        qsnsub=qsnsub, qsnfro=qsnfro,
        qsubc=qsubc, qfroc=qfroc, qfrzc=qfrzc, qmeltc=qmeltc,
        qevac=qevac, qdewc=qdewc,
        qseva=qseva, qsdew=qsdew, qinsur=qinsur, snoflow=snoflow,
        etrani=etrani,
    )
