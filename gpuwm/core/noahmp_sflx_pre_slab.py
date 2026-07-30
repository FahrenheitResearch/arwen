"""NOAHMP_SFLX's prefix as whole-slab CuPy arrays, with no per-column Python.

What this is, and what it is not
--------------------------------
:func:`gpuwm.core.noahmp_sflx_pre_gpu.evaluate_sflx_pre_calls` already answers
the ``sflx_pre`` leaf on the GPU, but it takes a Python list of per-column
``(args, kwargs)`` tuples and returns a Python list of
:class:`~gpuwm.core.noahmp_sflx.PreEnergy` records.  Between and around its four
kernel launches it runs a ``for`` loop over the batch: ``_scalars`` list
comprehensions per keyword, the ``FVEG`` block in host NumPy, eight
``ndarray.copy()`` per column and the reconstruction of ``PreEnergy`` and its
nested :class:`~gpuwm.core.noahmp_sflx.EnergyCall`.  At d04 that CPython is
essentially the whole cost of the scheme.

:func:`evaluate_sflx_pre_slab` answers the *same* batch from whole-slab CuPy
arrays and returns whole-slab CuPy arrays.  Nothing in it is per-column: there
is no loop over ``n`` anywhere in this file.

**Zero new arithmetic.**  The four kernels are the ones that are already
oracle-matched -- ``noahmp_leaf_atm`` (``noahmp_leaves.cu``), ``k_sflx_marshal``
(``noahmp_sflx.cu``), ``noahmp_phenology`` and ``noahmp_precip_heat``
(``noahmp_vegprecip.cu``) -- launched unchanged, in the same order, with the
same slot tables.  The slot tables are read off
:mod:`gpuwm.core.noahmp_sflx_pre_gpu` at call time rather than copied here, so
the two packers cannot drift apart and a table that is transposed in one is
transposed in both.  The only expression this module evaluates is the ``FVEG``
selection (``module_sf_noahmplsm.F:857-875``), which is four comparisons and the
single binary32 add ``ELAI + ESAI``; it is written with :func:`cupy.where` over
the column axis, which carries no rounding of its own.

There is no ``MAX``/``min`` anywhere in the prefix's host glue -- every one of
them lives inside a kernel -- so the ``numpy.maximum(-0.0, 0.0) == +0.0`` versus
``max(-0.0, 0.0) == -0.0`` hazard has nothing to bite here.  If a future edit
introduces one, write it as ``where(b > a, b, a)``: that is the form that
reproduces Fortran ``MAX`` on signed zeros, and this project has lost bugs to
the other one.

Strided views are a live defect, not a style note
-------------------------------------------------
``atm_y[:, slot]`` is a *strided* CuPy view -- 17 floats apart -- and a raw
kernel argument carries a pointer, not a stride.  ``noahmp_precip_heat`` reads
BDFALL/RAIN/SNOW/FP as ``a[c]``, so handing it such a view feeds column 0's
neighbouring output slots to every column; measured, it left PAHV/PAHG/PAHB at
zero and CANLIQ untouched on the one fixture column that has rain.  Every array
that reaches a kernel here therefore goes through :func:`cupy.ascontiguousarray`
first, and so does every array that leaves this function, so that a consumer
which forwards one to a kernel of its own cannot inherit the same trap.

The input contract
------------------
``fields`` maps the keyword names the per-column ``sflx_pre`` call already
carries to CuPy arrays, layer axis **last**:

* the forcing keyword set :func:`gpuwm.core.noahmp_sflx.sflx_steps` forwards
  (``sfctmp``, ``sfcprs``, ``zsoil``, ...), plus ``wa`` and ``acc_ssoil``;
* ``smc``, the third positional argument of ``sflx_pre``, as ``(n, NSOIL)``;
* the :class:`~gpuwm.core.noahmp_snow.SnowColumn` state under the *attribute*
  names: ``isnow snowh sneqv snice snliq stc zsnso sh2o``.  ``dzsnso`` and
  ``sice`` are not read by the prefix and are not required;
* the per-column values the batch reads off the ``SflxParameters`` handle:
  ``nroot iswater isbarren isice urban_flag hvt hvb tmin laim saim ch2op``.

``lai``, ``sai`` and ``pgs`` may be present and are ignored, exactly as they are
by the existing batch: under ``DVEG = 4`` PHENOLOGY overwrites LAI and SAI from
the monthly tables before reading them, and ``pgs`` is crop state that
``opt_crop = 0`` kills.

The output contract
-------------------
One flat ``dict`` of CuPy arrays.  Every field of :class:`PreEnergy` appears
under its own attribute name, and every field of the nested
:class:`EnergyCall` appears under ``"call." + name``.  The flat prefix is chosen
over a nested dict so the whole bundle can be indexed, compared and forwarded
with one mapping and one key convention; the key set is asserted against the two
dataclasses on the way out, so a field added to either fails here rather than
being silently dropped.

Shapes are ``(n,)`` wherever the record held a scalar and ``(n, width)``
wherever it held a layer vector, layer axis last.  Dtypes are ``float32``
throughout except ``call.ice``, ``call.vegtyp``, ``call.ist`` and
``call.isnow``, which the record declares ``int`` and which come back
``int32``.  A value that appears in both records -- ``fveg``, ``elai``,
``dzsnso``, ... -- is the *same array object* under both keys, and so are
``call.laisun``, ``call.laisha`` and ``call.rb``; treat the bundle as
read-only.  The seven arrays that pass straight through from the input slab
(``zsnso zsoil stc sh2o smc snice snliq``) are copied on the way out, as
``sflx_pre`` copies them per column, so the bundle survives the caller
overwriting its inputs.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields

import numpy as np

from gpuwm.core import noahmp_sflx_pre_gpu as _batch
from gpuwm.core.kernels import get_kernel
from gpuwm.core.noahmp_sflx import (DVEG, EnergyCall, PreEnergy,
                                    NSNOW_DEFAULT, NSOIL_DEFAULT)

#: WRF's ``-NSNOW+1 : NSOIL`` extent, as a 0-based width.
_NFULL = NSNOW_DEFAULT + NSOIL_DEFAULT

_ZERO = np.float32(0.0)

#: The irrigation amounts whose being zero is what makes 878-936 dead code.
_IRRIGATION = ("irrfra", "iramtsi", "iramtmi", "iramtfi")

#: ``(n,)`` float32 inputs that are not already named by a slot table.
_SCALAR_FIELDS = (
    # k_sflx_marshal
    "canliq", "canice", "sneqv", "wa",
    # noahmp_phenology
    "snowh", "tv", "lat", "julian", "hvt", "hvb", "tmin",
    # FVEG, and noahmp_precip_heat
    "shdmax", "dt", "uu", "vv", "tg", "ch2op",
    # carried through to the ENERGY bundle and nowhere else
    "lwdn", "zlvl", "co2air", "o2air", "tbot", "foln", "eah", "tah",
    "sneqvo", "albold", "cm", "ch", "dx", "dz8w", "tauss", "qc", "qsfc",
    "psfc", "acc_ssoil",
)

#: ``(n,)`` int32 inputs.
_INT_FIELDS = ("isnow", "nroot", "ist", "vegtyp", "yearlen", "iswater",
               "isbarren", "isice", "urban_flag", "ice", "croptype")

#: ``(n, width)`` inputs, layer axis last.
_LAYER_FIELDS = {"zsnso": _NFULL, "stc": _NFULL,
                 "smc": NSOIL_DEFAULT, "zsoil": NSOIL_DEFAULT,
                 "sh2o": NSOIL_DEFAULT,
                 "snice": NSNOW_DEFAULT, "snliq": NSNOW_DEFAULT,
                 "laim": 12, "saim": 12}


def _missing(name: str) -> "KeyError":
    return KeyError(
        f"evaluate_sflx_pre_slab: no {name!r} in fields; the slab carries the "
        "sflx_pre keyword set, the SnowColumn attributes and the per-column "
        "parameter values, each as one array over the column axis")


def _column(cp, fields, name, n, dtype):
    """One ``(n,)`` input, contiguous and in the dtype the kernel declares."""
    value = fields.get(name)
    if value is None:
        raise _missing(name)
    value = cp.asarray(value)
    if value.shape != (n,):
        raise ValueError(
            f"evaluate_sflx_pre_slab: {name!r} must have shape ({n},), got "
            f"{value.shape}")
    return cp.ascontiguousarray(value, dtype=dtype)


def _layers(cp, fields, name, n, width):
    """One ``(n, width)`` input, contiguous, layer axis last."""
    value = fields.get(name)
    if value is None:
        raise _missing(name)
    value = cp.asarray(value)
    if value.shape != (n, width):
        raise ValueError(
            f"evaluate_sflx_pre_slab: {name!r} must have shape ({n}, {width}) "
            f"with the layer axis last, got {value.shape}")
    return cp.ascontiguousarray(value, dtype=cp.float32)


def _launch(kernel, n, args) -> None:
    """One launch over the column axis, or nothing at all for an empty slab.

    A grid of zero blocks is a CUDA configuration error, so a slab with no land
    columns skips the launch; every array it would have written is already
    allocated at length zero and the bundle comes back well-formed and empty.
    """
    if n:
        kernel((_batch._blocks(n),), (_batch._THREADS,), args)


def evaluate_sflx_pre_slab(fields: dict, n: int) -> dict:
    """Answer ``sflx_pre`` for ``n`` land columns from whole-slab arrays.

    See the module docstring for the input and output contracts.  Returns a
    flat ``dict`` of CuPy arrays: :class:`PreEnergy`'s fields under their own
    names, :class:`EnergyCall`'s under ``"call." + name``.
    """
    import cupy as cp

    if int(n) < 0:
        raise ValueError(f"evaluate_sflx_pre_slab: n must not be negative: {n}")
    n = int(n)

    # ---- every input, once, over the whole column axis --------------------
    # The shape checks are load-bearing: k_sflx_marshal hard-codes NSNOW and
    # NSOIL, so a slab of another layer count would be read with the wrong
    # stride and answer silently.
    x = {name: _column(cp, fields, name, n, cp.float32)
         for name in _SCALAR_FIELDS}
    x.update({name: _column(cp, fields, name, n, cp.int32)
              for name in _INT_FIELDS})
    x.update({name: _layers(cp, fields, name, n, width)
              for name, width in _LAYER_FIELDS.items()})
    for name in _batch._ATM_IN:
        x[name] = _column(cp, fields, name, n, cp.float32)

    # ---- the guards sflx_pre itself applies, over the whole slab ----------
    # They are what makes the dead irrigation and crop regions dead.  One
    # device-to-host transfer answers all five, so the slab pays one
    # synchronisation for the batch rather than one per column.
    probes = [("croptype", x["croptype"] != 0)]
    for name in _IRRIGATION:
        value = fields.get(name)
        if value is not None:
            probes.append((name, _column(cp, fields, name, n, cp.float32)
                           != _ZERO))
    tripped = cp.asnumpy(cp.stack([flag.any() for _name, flag in probes]))
    for (name, _flag), bad in zip(probes, tripped):
        if not bad:
            continue
        if name == "croptype":
            raise ValueError(
                "croptype must be 0: opt_crop=0 pins it there and the crop "
                "branches are not transcribed")
        raise ValueError(
            f"{name} must be 0.0 under opt_irr=0; the irrigation region "
            "(878-936, 1042-1045) is proved dead, not transcribed")

    # ---- ATM, 819-823 -----------------------------------------------------
    atm_x = cp.empty((n, len(_batch._ATM_IN)), dtype=cp.float32)
    for slot, name in enumerate(_batch._ATM_IN):
        atm_x[:, slot] = x[name]
    atm_y = cp.zeros((n, len(_batch._ATM_OUT)), dtype=cp.float32)
    _launch(get_kernel("noahmp_leaves", "noahmp_leaf_atm"), n,
            (atm_x, cp.zeros((n, 1), dtype=cp.int32), atm_y, np.int32(n)))
    # Contiguous, not the strided column view; see the module docstring.
    atm = {name: cp.ascontiguousarray(atm_y[:, slot])
           for slot, name in enumerate(_batch._ATM_OUT)}

    # ---- DZSNSO / TROOT / BEG_WB, 827-849 ---------------------------------
    dzsnso = cp.zeros((n, _NFULL), dtype=cp.float32)
    troot = cp.zeros(n, dtype=cp.float32)
    beg_wb = cp.zeros(n, dtype=cp.float32)
    _launch(get_kernel("noahmp_sflx", "k_sflx_marshal"), n,
            (x["zsnso"], x["stc"], x["smc"], x["zsoil"],
             x["canliq"], x["canice"], x["sneqv"], x["wa"],
             x["isnow"], x["nroot"], x["ist"],
             dzsnso, troot, beg_wb, np.int32(n)))

    # ---- PHENOLOGY, 853-854 -----------------------------------------------
    # DVEG is not a kernel argument: noahmp_phenology transcribes the
    # DVEG in {1,3,4} arm only, which is the arm this identity takes.
    if int(DVEG) not in (1, 3, 4):                # pragma: no cover - pinned
        raise ValueError(f"the phenology kernel is not written for dveg={DVEG}")
    urban = cp.ascontiguousarray((x["urban_flag"] != 0).astype(cp.int32))
    phen = {name: cp.zeros(n, dtype=cp.float32) for name in _batch._PHEN_OUT}
    _launch(get_kernel("noahmp_vegprecip", "noahmp_phenology"), n,
            (np.int32(n), x["vegtyp"], x["yearlen"], x["iswater"],
             x["isbarren"], x["isice"], urban,
             x["snowh"], x["tv"], x["lat"], x["julian"],
             x["hvt"], x["hvb"], x["tmin"],
             x["laim"].ravel(), x["saim"].ravel(),
             *(phen[name] for name in _batch._PHEN_OUT)))
    elai, esai = phen["elai"], phen["esai"]

    # ---- FVEG, 857-875 ----------------------------------------------------
    # Only the DVEG == 4 arm is live.  Three masked selections in the source
    # order of the three IF statements, because 874 and 875 both overwrite 864
    # and the source order is what a reader will check this against.  No loop:
    # each line is one comparison over the column axis.
    floor = _batch._FVEG_FLOOR
    fveg = x["shdmax"]                                              # :863
    fveg = cp.where(fveg <= floor, floor, fveg)                     # :864
    fveg = cp.where((urban != 0) | (x["vegtyp"] == x["isbarren"]),
                    _ZERO, fveg)                                    # :874
    fveg = cp.where((elai + esai) == _ZERO, _ZERO, fveg)            # :875
    fveg = cp.ascontiguousarray(fveg, dtype=cp.float32)

    # ---- PRECIP_HEAT, 940-946 ---------------------------------------------
    prcp = {name: cp.zeros(n, dtype=cp.float32) for name in _batch._PRCP_OUT}
    _launch(get_kernel("noahmp_vegprecip", "noahmp_precip_heat"), n,
            (np.int32(n), x["ist"], x["dt"], x["uu"], x["vv"],
             elai, esai, fveg,
             atm["bdfall"], atm["rain"], atm["snow"], atm["fp"],
             x["canliq"], x["canice"], x["tv"], x["sfctmp"], x["tg"],
             x["ch2op"],
             *(prcp[name] for name in _batch._PRCP_OUT)))

    # ---- the two records, as columns --------------------------------------
    solad = cp.ascontiguousarray(
        cp.stack((atm["solad1"], atm["solad2"]), axis=1))
    solai = cp.ascontiguousarray(
        cp.stack((atm["solai1"], atm["solai2"]), axis=1))
    zeros = cp.zeros(n, dtype=cp.float32)

    # The seven arrays the ENERGY bundle carries straight through from the
    # input slab.  ``sflx_pre`` copies each of them per column, for the reason
    # ENERGY's declarations give: SNICE/SNLIQ/STC/SH2O/SMC are INOUT there.
    # One slab copy each keeps that property without one copy per column,
    # which is eight of the ``ndarray.copy()`` calls this file exists to
    # delete -- and it keeps the returned bundle independent of an input slab
    # the caller may go on to overwrite.
    through = {name: x[name].copy()
               for name in ("zsnso", "zsoil", "stc", "sh2o", "smc",
                            "snice", "snliq")}

    out = {
        "dzsnso": dzsnso, "troot": troot, "beg_wb": beg_wb, "fveg": fveg,
        "elai": elai, "esai": esai, "igs": phen["igs"], "fb": phen["fb"],
        "lai": phen["lai"], "sai": phen["sai"],
        "bdfall": atm["bdfall"], "rain": atm["rain"], "snow": atm["snow"],
        "fp": atm["fp"], "fpice": atm["fpice"], "prcp": atm["prcp"],
        "swdown": atm["swdown"], "qprecc": atm["qprecc"],
        "qprecl": atm["qprecl"], "qair": atm["qair"], "rhoair": atm["rhoair"],
        "canliq": prcp["canliq"], "canice": prcp["canice"],
        "fwet": prcp["fwet"], "cmc": prcp["cmc"], "qintr": prcp["qintr"],
        "qdripr": prcp["qdripr"], "qthror": prcp["qthror"],
        "qints": prcp["qints"], "qdrips": prcp["qdrips"],
        "qthros": prcp["qthros"], "pahv": prcp["pahv"], "pahg": prcp["pahg"],
        "pahb": prcp["pahb"], "qrain": prcp["qrain"], "qsnow": prcp["qsnow"],
        "snowhin": prcp["snowhin"],
    }
    call = {
        "ice": x["ice"], "vegtyp": x["vegtyp"], "ist": x["ist"],
        "isnow": x["isnow"],
        "dt": x["dt"], "rhoair": atm["rhoair"], "sfcprs": x["sfcprs"],
        "qair": atm["qair"], "sfctmp": x["sfctmp"], "thair": atm["thair"],
        "lwdn": x["lwdn"], "uu": x["uu"], "vv": x["vv"], "zref": x["zlvl"],
        "co2air": x["co2air"], "o2air": x["o2air"],
        "solad": solad, "solai": solai, "cosz": x["cosz"],
        "igs": phen["igs"], "eair": atm["eair"], "tbot": x["tbot"],
        "zsnso": through["zsnso"], "zsoil": through["zsoil"],
        "elai": elai, "esai": esai, "fwet": prcp["fwet"], "foln": x["foln"],
        "fveg": fveg, "pahv": prcp["pahv"], "pahg": prcp["pahg"],
        "pahb": prcp["pahb"], "qsnow": prcp["qsnow"], "dzsnso": dzsnso,
        "lat": x["lat"], "canliq": prcp["canliq"], "canice": prcp["canice"],
        "tv": x["tv"], "tg": x["tg"], "stc": through["stc"],
        "snowh": x["snowh"],
        "eah": x["eah"], "tah": x["tah"], "sneqvo": x["sneqvo"],
        "sneqv": x["sneqv"], "sh2o": through["sh2o"], "smc": through["smc"],
        "snice": through["snice"], "snliq": through["snliq"],
        "albold": x["albold"],
        "cm": x["cm"], "ch": x["ch"], "dx": x["dx"], "dz8w": x["dz8w"],
        "q2": x["q2"], "tauss": x["tauss"],
        # LAISUN/LAISHA/RB are the caller's uninitialised locals; WRF's driver
        # zeroes them and sflx_pre names them as such.
        "laisun": zeros, "laisha": zeros, "rb": zeros,
        "qc": x["qc"], "qsfc": x["qsfc"], "psfc": x["psfc"],
        "acc_ssoil": x["acc_ssoil"], "julian": x["julian"],
        "swdown": atm["swdown"], "prcp": atm["prcp"], "fb": phen["fb"],
    }
    out.update({f"call.{name}": value for name, value in call.items()})

    # A field added to either record must fail here, not vanish from the slab.
    want = {f.name for f in dc_fields(PreEnergy) if f.name != "call"}
    want |= {f"call.{f.name}" for f in dc_fields(EnergyCall)}
    if set(out) != want:
        raise AssertionError(
            "the sflx_pre slab and the PreEnergy/EnergyCall records disagree: "
            f"missing {sorted(want - set(out))}, extra {sorted(set(out) - want)}")
    return out


__all__ = ["evaluate_sflx_pre_slab"]
