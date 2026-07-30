"""NOAHMP_SFLX's first half, as one CUDA batch across many land columns.

What this is
------------
:func:`gpuwm.core.noahmp_sflx.sflx_pre` is everything ``NOAHMP_SFLX`` does
between its first executable statement and the ``ENERGY`` call: ``ATM``
(819-823), the ``DZSNSO``/``TROOT``/``BEG_WB`` marshalling (827-849),
``PHENOLOGY`` (853-854), the ``FVEG`` selection (857-875) and ``PRECIP_HEAT``
(940-946).  Measured on one RTX 5090 at 352 land columns it was **22% of the
whole land-surface call** and every bit of it ran in CPython, once per column.

:func:`evaluate_sflx_pre_calls` answers the same physical calls with the
kernels that are already oracle-matched against the unmodified WRF fixtures --
``noahmp_leaf_atm``, ``k_sflx_marshal``, ``noahmp_phenology`` and
``noahmp_precip_heat`` -- launched once for the whole batch.

**No arithmetic is transcribed here.**  The three device leaves and the
marshalling kernel already exist and are each gated against their own oracle
CSV; the only new code is the packing, the ``FVEG`` selection (four statements
of comparison, in NumPy over the column axis, which carries no rounding), and
the reconstruction of :class:`~gpuwm.core.noahmp_sflx.PreEnergy`.  That is
deliberate: the one thing a composition must not do is re-transcribe a leaf.

Why the packing is the thing to gate
------------------------------------
Everything here can be wrong in exactly one way -- a value landing in the wrong
slot, the wrong column or the wrong keyword -- and the arithmetic would still
be bitwise.  ``tests/test_noahmp_sflx_pre_batch_cuda.py`` therefore compares
this batch against :func:`gpuwm.core.noahmp_sflx.sflx_pre` itself, on the
physical calls the four whole-column WRF fixtures actually pause at, field by
field over the whole :class:`PreEnergy` including its nested
:class:`~gpuwm.core.noahmp_sflx.EnergyCall`, and shows that comparison
rejecting a misroute and a transposition first.

NumPy float32 and WRF float32
-----------------------------
The ``FVEG`` block is the only place this file evaluates anything.  It is four
comparisons and two constants -- ``FVEG = SHDMAX``; ``IF (FVEG <= 0.05) FVEG =
0.05``; the urban/barren kill; the ``ELAI+ESAI == 0`` kill -- so the only
arithmetic in it is ``ELAI + ESAI``, an IEEE binary32 add that NumPy performs
in the same rounding mode as :func:`gpuwm.core.noahmp_libm.f32` does.  Nothing
else in this module adds, multiplies or divides.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.noahmp_sflx import (DVEG, EnergyCall, PreEnergy,
                                    NSNOW_DEFAULT, NSOIL_DEFAULT)

#: One thread per column, the launch shape every other Noah-MP batch uses.
_THREADS = 64

#: ``noahmp_leaf_atm``'s flat input slots, in the kernel's own order
#: (``gpuwm/core/kernels/noahmp_leaves.cu``).  Slots 6-8 are
#: PRCPSNOW/PRCPGRPL/PRCPHAIL, which OPT_SNF=1 never reads; they are packed
#: because WRF's argument list carries them.
_ATM_IN = ("sfcprs", "sfctmp", "q2", "prcpconv", "prcpnonc", "prcpshcv",
           "prcpsnow", "prcpgrpl", "prcphail", "soldn", "cosz")

#: ``noahmp_leaf_atm``'s flat output slots, matching
#: :func:`gpuwm.core.noahmp_leaves.atm`'s return tuple position for position.
_ATM_OUT = ("thair", "qair", "eair", "rhoair", "qprecc", "qprecl",
            "solad1", "solad2", "solai1", "solai2", "swdown", "bdfall",
            "rain", "snow", "fp", "fpice", "prcp")

#: PRECIP_HEAT's outputs, in the kernel's argument order.
_PRCP_OUT = ("canliq", "canice", "qintr", "qdripr", "qthror", "qints",
             "qdrips", "qthros", "pahv", "pahg", "pahb", "qrain", "qsnow",
             "snowhin", "fwet", "cmc")

#: PHENOLOGY's outputs, in the kernel's argument order.
_PHEN_OUT = ("lai", "sai", "elai", "esai", "igs", "fb")

#: ``_FVEG_FLOOR``, :864.  Held here as the binary32 word, not as 0.05.
_FVEG_FLOOR = np.float32(0.05)

_ZERO = np.float32(0.0)


def _blocks(n: int) -> int:
    return (n + _THREADS - 1) // _THREADS


def _scalars(kwargs_list, name, n, dtype=np.float32):
    """One keyword, over the batch, as a contiguous array."""
    return np.fromiter((kw[name] for kw in kwargs_list), dtype=dtype, count=n)


def evaluate_sflx_pre_calls(calls):
    """Answer a batch of physical ``sflx_pre`` calls with CUDA.

    ``calls`` is the ``[(args, kwargs), ...]`` list
    :func:`gpuwm.core.noahmp_sflx.sflx_steps` yields, so ``args`` is
    ``(parameters, col, smc)`` and ``kwargs`` is ``sflx_pre``'s keyword set.
    Returns one :class:`PreEnergy` per call, in order.
    """
    import cupy as cp

    n = len(calls)
    if n == 0:
        return []

    pars = [call[0][0] for call in calls]
    cols = [call[0][1] for call in calls]
    smcs = [np.ascontiguousarray(np.asarray(call[0][2], dtype=np.float32))
            for call in calls]
    kws = [call[1] for call in calls]

    # ---- the guards sflx_pre itself applies, applied per call -------------
    # They are the assertions that make the dead irrigation and crop regions
    # dead; a batch that skipped them would run the same admitted arithmetic
    # on an inadmissible column and say nothing about it.
    for kw in kws:
        if int(kw["croptype"]) != 0:
            raise ValueError(
                "croptype must be 0: opt_crop=0 pins it there and the crop "
                "branches are not transcribed")
        for name in ("irrfra", "iramtsi", "iramtmi", "iramtfi"):
            if np.float32(kw.get(name, 0.0)) != _ZERO:
                raise ValueError(
                    f"{name} must be 0.0 under opt_irr=0; the irrigation "
                    "region (878-936, 1042-1045) is proved dead, not "
                    "transcribed")

    nsoil = cols[0].nsoil
    nsnow = cols[0].nsnow
    if nsnow != NSNOW_DEFAULT or nsoil != NSOIL_DEFAULT:
        raise ValueError(
            f"the sflx_pre batch is compiled for nsnow={NSNOW_DEFAULT} / "
            f"nsoil={NSOIL_DEFAULT}, got {nsnow} / {nsoil}")
    for col in cols:
        if col.nsoil != nsoil or col.nsnow != nsnow:
            raise ValueError(
                "every column in one sflx_pre batch must carry the same "
                "snow/soil layer counts")
    nfull = nsnow + nsoil

    # ---- ATM, 819-823 ----------------------------------------------------
    atm_x = np.empty((n, len(_ATM_IN)), dtype=np.float32)
    for slot, name in enumerate(_ATM_IN):
        atm_x[:, slot] = _scalars(kws, name, n)
    atm_y = cp.zeros((n, len(_ATM_OUT)), dtype=cp.float32)
    get_kernel("noahmp_leaves", "noahmp_leaf_atm")(
        (_blocks(n),), (_THREADS,),
        (cp.asarray(atm_x), cp.zeros((n, 1), dtype=cp.int32), atm_y,
         np.int32(n)))
    # ``atm_y[:, slot]`` is a *strided* view -- 17 floats apart -- and a raw
    # kernel argument carries a pointer, not a stride.  PRECIP_HEAT reads
    # BDFALL/RAIN/SNOW/FP as ``a[c]``, so handing it the view feeds column 0's
    # neighbouring slots to every column.  Measured: it left PAHV/PAHG/PAHB at
    # zero and CANLIQ untouched on the one fixture column that has rain.
    atm = {name: cp.ascontiguousarray(atm_y[:, slot])
           for slot, name in enumerate(_ATM_OUT)}

    # ---- DZSNSO / TROOT / BEG_WB, 827-849 --------------------------------
    zsnso = np.array([col.zsnso for col in cols], dtype=np.float32)
    stc = np.array([col.stc for col in cols], dtype=np.float32)
    smc = np.array(smcs, dtype=np.float32)
    zsoil = np.array([np.asarray(kw["zsoil"], dtype=np.float32)
                      for kw in kws], dtype=np.float32)
    isnow = np.fromiter((col.isnow for col in cols), dtype=np.int32, count=n)
    nroot = np.fromiter((p.nroot for p in pars), dtype=np.int32, count=n)
    ist = _scalars(kws, "ist", n, dtype=np.int32)

    d_dzsnso = cp.zeros((n, nfull), dtype=cp.float32)
    d_troot = cp.zeros(n, dtype=cp.float32)
    d_beg_wb = cp.zeros(n, dtype=cp.float32)
    get_kernel("noahmp_sflx", "k_sflx_marshal")(
        (_blocks(n),), (_THREADS,),
        (cp.asarray(zsnso), cp.asarray(stc), cp.asarray(smc),
         cp.asarray(zsoil),
         cp.asarray(_scalars(kws, "canliq", n)),
         cp.asarray(_scalars(kws, "canice", n)),
         cp.asarray(np.fromiter((col.sneqv for col in cols),
                                dtype=np.float32, count=n)),
         cp.asarray(_scalars(kws, "wa", n)),
         cp.asarray(isnow), cp.asarray(nroot), cp.asarray(ist),
         d_dzsnso, d_troot, d_beg_wb, np.int32(n)))

    # ---- PHENOLOGY, 853-854 ----------------------------------------------
    # DVEG is not a kernel argument: noahmp_phenology transcribes the
    # DVEG in {1,3,4} arm only, which is the arm this identity takes.
    if int(DVEG) not in (1, 3, 4):                # pragma: no cover - pinned
        raise ValueError(f"the phenology kernel is not written for dveg={DVEG}")
    vegtyp = _scalars(kws, "vegtyp", n, dtype=np.int32)
    phen_out = {name: cp.zeros(n, dtype=cp.float32) for name in _PHEN_OUT}
    get_kernel("noahmp_vegprecip", "noahmp_phenology")(
        (_blocks(n),), (_THREADS,),
        (np.int32(n),
         cp.asarray(vegtyp),
         cp.asarray(_scalars(kws, "yearlen", n, dtype=np.int32)),
         cp.asarray(np.fromiter((p.iswater for p in pars),
                                dtype=np.int32, count=n)),
         cp.asarray(np.fromiter((p.isbarren for p in pars),
                                dtype=np.int32, count=n)),
         cp.asarray(np.fromiter((p.isice for p in pars),
                                dtype=np.int32, count=n)),
         cp.asarray(np.fromiter((bool(p.urban_flag) for p in pars),
                                dtype=np.int32, count=n)),
         cp.asarray(np.fromiter((col.snowh for col in cols),
                                dtype=np.float32, count=n)),
         cp.asarray(_scalars(kws, "tv", n)),
         cp.asarray(_scalars(kws, "lat", n)),
         cp.asarray(_scalars(kws, "julian", n)),
         cp.asarray(np.fromiter((p.hvt for p in pars),
                                dtype=np.float32, count=n)),
         cp.asarray(np.fromiter((p.hvb for p in pars),
                                dtype=np.float32, count=n)),
         cp.asarray(np.fromiter((p.tmin for p in pars),
                                dtype=np.float32, count=n)),
         cp.asarray(np.array([p.laim for p in pars],
                             dtype=np.float32).ravel()),
         cp.asarray(np.array([p.saim for p in pars],
                             dtype=np.float32).ravel()),
         *(phen_out[name] for name in _PHEN_OUT)))

    elai = cp.asnumpy(phen_out["elai"])
    esai = cp.asnumpy(phen_out["esai"])

    # ---- FVEG, 857-875 ----------------------------------------------------
    # Only the DVEG == 4 arm is live; the others belong to option values this
    # port does not admit.  Written as three masked assignments in the source
    # order of the three IF statements, because 874 and 875 both overwrite 864
    # and the order they do it in is not observable but the source order is
    # what a reader will check this against.
    isbarren = np.fromiter((p.isbarren for p in pars), dtype=np.int32, count=n)
    urban = np.fromiter((bool(p.urban_flag) for p in pars),
                        dtype=bool, count=n)
    fveg = _scalars(kws, "shdmax", n)                            # :863
    fveg[fveg <= _FVEG_FLOOR] = _FVEG_FLOOR                      # :864
    fveg[urban | (vegtyp == isbarren)] = _ZERO                   # :874
    fveg[(elai + esai) == _ZERO] = _ZERO                         # :875

    # ---- PRECIP_HEAT, 940-946 --------------------------------------------
    prcp_out = {name: cp.zeros(n, dtype=cp.float32) for name in _PRCP_OUT}
    get_kernel("noahmp_vegprecip", "noahmp_precip_heat")(
        (_blocks(n),), (_THREADS,),
        (np.int32(n),
         cp.asarray(ist),
         cp.asarray(_scalars(kws, "dt", n)),
         cp.asarray(_scalars(kws, "uu", n)),
         cp.asarray(_scalars(kws, "vv", n)),
         phen_out["elai"], phen_out["esai"], cp.asarray(fveg),
         atm["bdfall"], atm["rain"], atm["snow"], atm["fp"],
         cp.asarray(_scalars(kws, "canliq", n)),
         cp.asarray(_scalars(kws, "canice", n)),
         cp.asarray(_scalars(kws, "tv", n)),
         cp.asarray(_scalars(kws, "sfctmp", n)),
         cp.asarray(_scalars(kws, "tg", n)),
         cp.asarray(np.fromiter((p.ch2op for p in pars),
                                dtype=np.float32, count=n)),
         *(prcp_out[name] for name in _PRCP_OUT)))

    # ---- back to one PreEnergy per column ---------------------------------
    a = {name: cp.asnumpy(value) for name, value in atm.items()}
    ph = {name: cp.asnumpy(value) for name, value in phen_out.items()}
    pc = {name: cp.asnumpy(value) for name, value in prcp_out.items()}
    dzsnso = cp.asnumpy(d_dzsnso)
    troot = cp.asnumpy(d_troot)
    beg_wb = cp.asnumpy(d_beg_wb)
    solad = np.stack((a["solad1"], a["solad2"]), axis=1)
    solai = np.stack((a["solai1"], a["solai2"]), axis=1)

    out = []
    for k, (par, col, kw) in enumerate(zip(pars, cols, kws)):
        dz = np.ascontiguousarray(dzsnso[k])
        call = EnergyCall(
            ice=int(kw["ice"]), vegtyp=int(kw["vegtyp"]), ist=int(kw["ist"]),
            isnow=int(col.isnow),
            dt=np.float32(kw["dt"]), rhoair=a["rhoair"][k],
            sfcprs=np.float32(kw["sfcprs"]), qair=a["qair"][k],
            sfctmp=np.float32(kw["sfctmp"]), thair=a["thair"][k],
            lwdn=np.float32(kw["lwdn"]), uu=np.float32(kw["uu"]),
            vv=np.float32(kw["vv"]), zref=np.float32(kw["zlvl"]),
            co2air=np.float32(kw["co2air"]), o2air=np.float32(kw["o2air"]),
            solad=np.ascontiguousarray(solad[k]),
            solai=np.ascontiguousarray(solai[k]),
            cosz=np.float32(kw["cosz"]), igs=ph["igs"][k], eair=a["eair"][k],
            tbot=np.float32(kw["tbot"]), zsnso=col.zsnso.copy(),
            zsoil=zsoil[k].copy(),
            elai=elai[k], esai=esai[k], fwet=pc["fwet"][k],
            foln=np.float32(kw["foln"]), fveg=fveg[k],
            pahv=pc["pahv"][k], pahg=pc["pahg"][k], pahb=pc["pahb"][k],
            qsnow=pc["qsnow"][k], dzsnso=dz.copy(),
            lat=np.float32(kw["lat"]),
            canliq=pc["canliq"][k], canice=pc["canice"][k],
            tv=np.float32(kw["tv"]), tg=np.float32(kw["tg"]),
            stc=col.stc.copy(), snowh=np.float32(col.snowh),
            eah=np.float32(kw["eah"]), tah=np.float32(kw["tah"]),
            sneqvo=np.float32(kw["sneqvo"]), sneqv=np.float32(col.sneqv),
            sh2o=col.sh2o.copy(), smc=smcs[k].copy(),
            snice=col.snice.copy(), snliq=col.snliq.copy(),
            albold=np.float32(kw["albold"]), cm=np.float32(kw["cm"]),
            ch=np.float32(kw["ch"]), dx=np.float32(kw["dx"]),
            dz8w=np.float32(kw["dz8w"]), q2=np.float32(kw["q2"]),
            tauss=np.float32(kw["tauss"]),
            laisun=_ZERO, laisha=_ZERO, rb=_ZERO,
            qc=np.float32(kw["qc"]), qsfc=np.float32(kw["qsfc"]),
            psfc=np.float32(kw["psfc"]),
            acc_ssoil=np.float32(kw["acc_ssoil"]),
            julian=np.float32(kw["julian"]), swdown=a["swdown"][k],
            prcp=a["prcp"][k], fb=ph["fb"][k])
        out.append(PreEnergy(
            call=call, dzsnso=dz, troot=troot[k], beg_wb=beg_wb[k],
            fveg=fveg[k], elai=elai[k], esai=esai[k], igs=ph["igs"][k],
            fb=ph["fb"][k], lai=ph["lai"][k], sai=ph["sai"][k],
            bdfall=a["bdfall"][k], rain=a["rain"][k], snow=a["snow"][k],
            fp=a["fp"][k], fpice=a["fpice"][k], prcp=a["prcp"][k],
            swdown=a["swdown"][k], qprecc=a["qprecc"][k],
            qprecl=a["qprecl"][k], qair=a["qair"][k], rhoair=a["rhoair"][k],
            canliq=pc["canliq"][k], canice=pc["canice"][k],
            fwet=pc["fwet"][k], cmc=pc["cmc"][k], qintr=pc["qintr"][k],
            qdripr=pc["qdripr"][k], qthror=pc["qthror"][k],
            qints=pc["qints"][k], qdrips=pc["qdrips"][k],
            qthros=pc["qthros"][k], pahv=pc["pahv"][k], pahg=pc["pahg"][k],
            pahb=pc["pahb"][k], qrain=pc["qrain"][k], qsnow=pc["qsnow"][k],
            snowhin=pc["snowhin"][k]))
    return out


__all__ = ["evaluate_sflx_pre_calls"]
