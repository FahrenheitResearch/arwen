"""The whole Noah-MP column, assembled over the column axis.

Every piece of NOAHMP_SFLX except this file already exists in slab form and is
gated bitwise on its own: the eight leaf batches
(:mod:`gpuwm.core.noahmp_sflx_pre_slab`, :mod:`gpuwm.core.noahmp_thermal_slab`,
:mod:`gpuwm.core.noahmp_flux_slab`, :mod:`gpuwm.core.noahmp_water_slab`),
ENERGY's four composition segments (:mod:`gpuwm.core.noahmp_energy_slab`) and
``sflx_post``'s two (:mod:`gpuwm.core.noahmp_sflx_post_slab`).  This module is
the orchestration those conversions were built for: the straight-line launch
sequence that runs one whole land-surface call for ``n`` columns with **no
Python executed per column** -- the same statements
:func:`gpuwm.core.noahmp_sflx.sflx_steps` drives one frame at a time, in the
same order, with the array spelling of each hand-off.

Nothing here computes physics.  Every value is produced by a slab module that
is itself gated against the scalar authority; what this file owns is the
*hand-offs* -- which array reaches which leaf, and when.  That is exactly the
defect class the device-column report has warned about for five sections (a
per-column value reaching the wrong slot survives every leaf's own oracle), so
``tests/test_noahmp_column_slab.py`` gates this file the only way that can
catch it: the assembled slab against :func:`sflx_steps` driven per column on
the host, every output field, bitwise.

The hand-offs that are easy to get wrong, written down
------------------------------------------------------
* **VEGE_FLUX runs on the vegetated subset only**, exactly as the scalar
  column only pauses there when ``tile_veg``.  The subset is compacted with
  one gather, evaluated, and scattered back; the lanes VEGE_FLUX never wrote
  are read nowhere downstream except through ``where(tile_veg, ...)``, and the
  five INOUT scalars (``tv tg->tgv cm->cmv ch->chv eah``, plus ``tah`` and
  ``qsfc``) fall back to their entry values on the bare lanes -- which is what
  the scalar path holds there.
* **BARE_FLUX's ``qsfc`` argument is the post-VEGE value** on vegetated
  columns: the scalar path assigns ``qsfc = vf.QSFC`` before BARE_FLUX reads
  it.  Handing BARE_FLUX the entry ``qsfc`` everywhere would be bitwise wrong
  on every vegetated column and invisible to both leaves' own oracles.
* **TSNOSOI reads the entry ``stc``; PHASECHANGE reads TSNOSOI's.**  The
  scalar generator rebinds ``stc`` between the two calls.
* **WATER reads ENERGY's exit column through** :func:`sflx_post_seg1`, which
  lands PHASECHANGE's arrays and copies them -- WRF's own aliasing, where
  NOAHMP_SFLX hands one set of arrays to both ENERGY and WATER.
* **ERROR reads the post-WATER column**: ``k_sflx_error``'s DZSNSO and the
  snow clamp's SNOWH/SNEQV are WATER's outputs, not ENERGY's.

Parameters arrive as per-column slabs
-------------------------------------
The scalar path reads parameters off one memoised ``SflxParameters`` handle
per (vegetation, soil) identity.  A slab cannot chase Python attributes per
column, so :func:`parameter_rows` flattens one handle into one row of named
values and the caller gathers rows into per-column arrays with one indexed
read (:func:`parameter_fields`).  The row layout is asserted against the
consuming slab modules' own slot tables where those exist, and the whole-column
gate is what holds the rest.

What this module refuses
------------------------
``n == 0`` answers an empty, well-formed bundle.  Everything the scalar path
refuses -- ``ICE != 0``, ``IST != 1``, a non-zero crop or irrigation state,
the three balance aborts, emitted longwave <= 0 -- is refused by the slab
modules themselves, for the slab, naming the first offending column.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core import noahmp_vegeflux_gpu as _vege
from gpuwm.core.noahmp_energy import PINNED_OPTIONS
from gpuwm.core.noahmp_energy_slab import (energy_seg1, energy_seg2,
                                           energy_seg3, energy_seg4)
from gpuwm.core.noahmp_flux_slab import (evaluate_bare_flux_slab,
                                         evaluate_vege_flux_slab)
from gpuwm.core.noahmp_radiation_gpu import PARAMETER_SLOTS as _RAD_PARAMETERS
from gpuwm.core.noahmp_sflx import NSNOW_DEFAULT, NSOIL_DEFAULT
from gpuwm.core.noahmp_sflx_post_slab import sflx_post_seg1, sflx_post_seg2
from gpuwm.core.noahmp_sflx_pre_slab import evaluate_sflx_pre_slab
from gpuwm.core.noahmp_thermal_slab import (RADIATION_PARAMETER_WIDTH,
                                            evaluate_phasechange_slab,
                                            evaluate_radiation_slab,
                                            evaluate_thermoprop_slab,
                                            evaluate_tsnosoi_slab)
from gpuwm.core.noahmp_water_slab import evaluate_water_slab

NSNOW = NSNOW_DEFAULT
NSOIL = NSOIL_DEFAULT
_NFULL = NSNOW + NSOIL

_ZERO = np.float32(0.0)


# ---------------------------------------------------------------------------
# the parameter rows: one SflxParameters handle, flattened by name
# ---------------------------------------------------------------------------

#: Integer per-identity values, and where each is read from the handle.
PARAMETER_INTS = ("nroot", "iswater", "isbarren", "isice", "urban_flag")

#: Float per-identity values that are read by name somewhere in the slab
#: composition, mapped to their per-column width.  The RADIATION widths are
#: imported from the slab module that consumes them rather than restated;
#: everything else is a scalar, a soil-layer vector or a monthly vector.
PARAMETER_WIDTHS: dict[str, int] = {
    # the SFLX prefix (PHENOLOGY, PRECIP_HEAT)
    "hvt": 1, "hvb": 1, "tmin": 1, "ch2op": 1, "laim": 12, "saim": 12,
    # ENERGY's own composition (seg1, seg2)
    "mfsno": 1, "scffac": 1, "z0sno": 1, "z0mvt": 1, "cwpvt": 1,
    "snow_emis": 1, "rsurf_exp": 1, "eg": 1,
    "smcmax": NSOIL, "smcref": NSOIL, "smcwlt": NSOIL,
    "psisat": NSOIL, "bexp": NSOIL,
    # THERMOPROP
    "quartz": NSOIL, "csoil": 1,
    # TSNOSOI
    "zbot": 1,
    # RADIATION
    **dict(RADIATION_PARAMETER_WIDTH),
    # VEGE_FLUX (kept in WRF's upper-case spelling, as the batch reads them)
    **{name: 1 for name in _vege.PARAMETER_NAMES},
    # WATER
    "dksat": NSOIL, "dwsat": NSOIL,
    "kdt": 1, "frzx": 1, "slope": 1, "ssi": 1, "snow_ret_fac": 1,
}

if tuple(name for name in _RAD_PARAMETERS if name not in PARAMETER_WIDTHS):
    raise AssertionError("a RADIATION parameter slot has no row entry")


def parameter_row(parameters) -> dict:
    """One ``SflxParameters`` handle as ``{name: numpy value}``.

    ``eg`` is ``EG(IST)`` with ``IST == 1`` -- the only surface type this port
    admits, refused everywhere else -- so the row carries the scalar the
    column would read, not the two-entry table.
    """
    p = parameters
    e = p.energy
    if e is None:
        raise TypeError("parameter_row needs a handle with .energy bound")
    row = {
        "nroot": int(p.nroot), "iswater": int(p.iswater),
        "isbarren": int(p.isbarren), "isice": int(p.isice),
        "urban_flag": int(bool(p.urban_flag)),
        "hvt": e.hvt, "hvb": p.hvb, "tmin": p.tmin, "ch2op": p.ch2op,
        "laim": np.asarray(p.laim, np.float32),
        "saim": np.asarray(p.saim, np.float32),
        "mfsno": e.mfsno, "scffac": e.scffac, "z0sno": e.z0sno,
        "z0mvt": e.z0mvt, "cwpvt": e.cwpvt, "snow_emis": e.snow_emis,
        "rsurf_exp": e.rsurf_exp, "eg": np.float32(e.eg[0]),
        "smcmax": np.asarray(e.smcmax, np.float32),
        "smcref": np.asarray(e.smcref, np.float32),
        "smcwlt": np.asarray(e.smcwlt, np.float32),
        "psisat": np.asarray(e.psisat, np.float32),
        "bexp": np.asarray(e.bexp, np.float32),
        "quartz": np.asarray(e.quartz, np.float32), "csoil": e.csoil,
        "zbot": e.zbot,
        "tau0": e.tau0, "grain_growth": e.grain_growth,
        "extra_growth": e.extra_growth, "dirt_soot": e.dirt_soot,
        "swemx": e.swemx,
        "albsat": np.asarray(e.albsat, np.float32),
        "albdry": np.asarray(e.albdry, np.float32),
        "alblak": np.asarray(e.alblak, np.float32),
        "rhol": np.asarray(e.rhol, np.float32),
        "rhos": np.asarray(e.rhos, np.float32),
        "taul": np.asarray(e.taul, np.float32),
        "taus": np.asarray(e.taus, np.float32),
        "xl": e.xl, "omegas": np.asarray(e.omegas, np.float32),
        "betads": e.betads, "betais": e.betais,
        "dksat": np.asarray(p.dksat, np.float32),
        "dwsat": np.asarray(p.dwsat, np.float32),
        "kdt": p.kdt, "frzx": p.frzx, "slope": p.slope, "ssi": p.ssi,
        "snow_ret_fac": p.snow_ret_fac,
    }
    row.update({name: np.float32(getattr(e.vege, name))
                for name in _vege.PARAMETER_NAMES})
    return row


def parameter_fields(rows, index) -> dict:
    """Per-column parameter slabs from per-identity rows and an index.

    ``rows`` is a sequence of :func:`parameter_row` results, one per distinct
    (vegetation, soil) identity; ``index`` maps each column to its row.  The
    gather is one indexed read per field over the whole slab, so the Python
    here is proportional to the number of parameter names and never to the
    number of columns.
    """
    import cupy as cp

    if not rows:
        raise ValueError("parameter_fields needs at least one identity row")
    idx = cp.asarray(np.ascontiguousarray(index, dtype=np.int32))
    out = {}
    for name, width in PARAMETER_WIDTHS.items():
        table = np.array([np.broadcast_to(
            np.asarray(row[name], np.float32), (width,)) for row in rows],
            dtype=np.float32)
        gathered = cp.asarray(table)[idx]
        out[name] = cp.ascontiguousarray(
            gathered[:, 0] if width == 1 else gathered)
    for name in PARAMETER_INTS:
        table = np.array([int(row[name]) for row in rows], dtype=np.int32)
        out[name] = cp.ascontiguousarray(cp.asarray(table)[idx])
    return out


# ---------------------------------------------------------------------------
# the composition
# ---------------------------------------------------------------------------

#: VEGE_FLUX's outputs under the names ENERGY's composition reads them by,
#: exactly the table ``tests/test_noahmp_energy_slab.py`` gates seg3 with.
#: The five INOUT names fall back to the column's entry value on bare lanes;
#: everything else falls back to the zeros :2033-2053 writes.
_VEGE_TO_ENERGY = {
    "tauxv": "TAUXV", "tauyv": "TAUYV", "irg": "IRG", "irc": "IRC",
    "shg": "SHG", "shc": "SHC", "evg": "EVG", "evc": "EVC", "tr": "TR",
    "ghv": "GH", "t2mv": "T2MV", "psnsun": "PSNSUN", "psnsha": "PSNSHA",
    "rssun": "RSSUN", "rssha": "RSSHA", "q2v": "Q2V",
    "tgv": "TG", "cmv": "CM", "chv": "CH", "eah": "EAH", "tv": "TV",
}
_VEGE_ENTRY_FALLBACK = {"tgv": "tg", "cmv": "cm", "chv": "ch",
                        "eah": "eah", "tv": "tv"}

#: The names VEGE_FLUX reads straight off the ENERGY bundle -- the same value
#: under the same name on both sides of the seam.
_VEGE_CALL_THROUGH = (
    "dt", "lwdn", "uu", "vv", "sfctmp", "qair", "eair", "rhoair", "snowh",
    "fwet", "canliq", "canice", "igs", "foln", "co2air", "o2air", "sfcprs",
    "pahv", "pahg", "eah", "tah", "tv", "tg", "cm", "ch", "psfc", "fveg",
)

#: Everything :func:`evaluate_sflx_slab` returns, so a consumer that stops
#: receiving a field fails by name instead of silently writing stale state.
OUTPUT_NAMES = (
    # the SflxResult scalars the driver's write-back reads, same spelling
    "trad", "fsh", "ecan", "edir", "etran", "fcev", "fgev", "fctr", "ssoil",
    "runsrf", "runsub", "albedo", "fsno", "canliq", "canice", "fpice",
    "qmelt", "ponding", "ponding1", "ponding2", "emissi", "qsfc", "tv", "tg",
    "eah", "tah", "cm", "ch", "fwet", "sneqvo", "albold", "qsnow", "qrain",
    "wslake", "lai", "sai", "tauss", "z0wrf", "t2mv", "t2mb", "q2v", "q2b",
    "fveg", "fsa", "fira", "rssun", "rssha", "chv", "chb", "canhs", "qsnbot",
    "eflxb", "laisun", "laisha", "rb",
    # the mutated column state
    "isnow", "snowh", "sneqv", "snice", "snliq", "stc", "zsnso", "sh2o",
    "smc", "hcpct",
    # ERROR's residuals, which no WRF output carries and every gate wants
    "errwat", "end_wb", "errsw", "erreng", "beg_wb",
)


def _shaped(cp, fields, name, n, width):
    """One extra input (beyond the pre slab's own contract), shape-checked."""
    try:
        value = fields[name]
    except KeyError:
        raise KeyError(
            f"evaluate_sflx_slab: no {name!r} in fields; the slab carries the "
            "sflx_pre keyword set, the SnowColumn attributes, the per-column "
            "parameter values, wslake and ficeold") from None
    value = cp.asarray(value)
    want = (n,) if width == 1 else (n, width)
    if value.shape != want:
        raise ValueError(
            f"evaluate_sflx_slab: {name!r} must have shape {want}, got "
            f"{tuple(value.shape)}")
    return cp.ascontiguousarray(value, dtype=cp.float32)


def evaluate_sflx_slab(fields: dict, n: int) -> dict:
    """One NOAHMP_SFLX land-surface call for ``n`` columns, on the device.

    ``fields`` carries, each as one CuPy (or uploadable) array over the column
    axis with the layer axis last:

    * everything :func:`gpuwm.core.noahmp_sflx_pre_slab.evaluate_sflx_pre_slab`
      documents -- the ``sflx_pre`` keyword set, the ``SnowColumn`` attributes
      and the per-column parameter values;
    * the rest of :data:`PARAMETER_WIDTHS` and :data:`PARAMETER_INTS`, which
      :func:`parameter_fields` produces;
    * ``wslake`` ``(n,)`` and ``ficeold`` ``(n, NSNOW)``.

    Returns :data:`OUTPUT_NAMES`, each ``(n,)`` float32 unless it is column
    state (layer axis last) or ``isnow`` (int32).  Raises exactly where the
    scalar column raises: the glacier/lake refusals, the emitted-longwave
    check, and the three balance aborts, each naming the first offending
    column.
    """
    import cupy as cp

    n = int(n)
    if n < 0:
        raise ValueError(f"evaluate_sflx_slab: n must not be negative: {n}")
    if n == 0:
        empty = cp.empty(0, dtype=cp.float32)
        out = {name: empty for name in OUTPUT_NAMES}
        out["isnow"] = cp.empty(0, dtype=cp.int32)
        for name, width in (("snice", NSNOW), ("snliq", NSNOW),
                            ("stc", _NFULL), ("zsnso", _NFULL),
                            ("sh2o", NSOIL), ("smc", NSOIL),
                            ("hcpct", _NFULL)):
            out[name] = cp.empty((0, width), dtype=cp.float32)
        return out

    def par(name):
        return _shaped(cp, fields, name, n, PARAMETER_WIDTHS[name])

    urban = cp.ascontiguousarray(cp.asarray(fields["urban_flag"]) != 0)

    # ---- NOAHMP_SFLX's first half, 806-946 --------------------------------
    pre = evaluate_sflx_pre_slab(fields, n)

    def c(name):
        return pre["call." + name]

    # ---- ENERGY's entry state, spelled the way the segments read it -------
    s = {
        "ice": c("ice"), "ist": c("ist"), "isnow": c("isnow"),
        "uu": c("uu"), "vv": c("vv"), "elai": c("elai"), "esai": c("esai"),
        "snowh": c("snowh"), "sneqv": c("sneqv"), "zref": c("zref"),
        "fveg": c("fveg"), "sfcprs": c("sfcprs"), "tv": c("tv"),
        "tg": c("tg"), "lwdn": c("lwdn"), "cosz": c("cosz"), "dt": c("dt"),
        "acc_ssoil": c("acc_ssoil"),
        "pahv": c("pahv"), "pahg": c("pahg"), "pahb": c("pahb"),
        "sh2o": c("sh2o"), "zsoil": c("zsoil"), "dzsnso": c("dzsnso"),
        "mfsno": par("mfsno"), "scffac": par("scffac"),
        "z0sno": par("z0sno"), "z0mvt": par("z0mvt"), "hvt": par("hvt"),
        "cwpvt": par("cwpvt"), "snow_emis": par("snow_emis"),
        "rsurf_exp": par("rsurf_exp"),
        "nroot": cp.ascontiguousarray(cp.asarray(fields["nroot"],
                                                 dtype=cp.int32)),
        "urban_flag": urban,
        "smcmax": par("smcmax"), "smcref": par("smcref"),
        "smcwlt": par("smcwlt"), "psisat": par("psisat"),
        "bexp": par("bexp"), "eg": par("eg"),
        "soil_update_steps": PINNED_OPTIONS.soil_update_steps,
    }

    # ---- ENERGY seg1, then the two thermal-side leaves --------------------
    g = energy_seg1(s)

    thermo = evaluate_thermoprop_slab({
        "dzsnso": c("dzsnso"), "snice": c("snice"), "snliq": c("snliq"),
        "smc": c("smc"), "sh2o": c("sh2o"), "stc": c("stc"),
        "snowh": c("snowh"), "dt": c("dt"),
        "smcmax": s["smcmax"], "csoil": par("csoil"), "quartz": par("quartz"),
        "isnow": c("isnow"), "ist": c("ist"), "urban_flag": urban}, n)

    rad_fields = {name: par(name) for name in _RAD_PARAMETERS}
    rad_fields.update({
        "dt": c("dt"), "cosz": c("cosz"), "elai": c("elai"),
        "esai": c("esai"), "tg": c("tg"), "tv": c("tv"),
        "snowh": c("snowh"), "fsno": g["fsno"], "fwet": c("fwet"),
        "sneqvo": c("sneqvo"), "sneqv": c("sneqv"), "qsnow": c("qsnow"),
        "fveg": c("fveg"), "smc": c("smc"), "albold": c("albold"),
        "tauss": c("tauss"), "solad": c("solad"), "solai": c("solai"),
        "ist": c("ist")})
    rad = evaluate_radiation_slab(rad_fields, n)

    # ---- ENERGY seg2, then the two flux leaves -----------------------------
    e = energy_seg2(s, g)
    tile_veg = e["tile_veg"]

    # VEGE_FLUX answers the vegetated subset, exactly as the scalar column
    # only pauses there under tile_veg.  One gather in, one scatter out.
    veg_index = cp.nonzero(tile_veg)[0]
    nveg = int(veg_index.size)

    vf_fields = {name: c(name)[veg_index] for name in _VEGE_CALL_THROUGH}
    vf_fields["qsfc"] = c("qsfc")[veg_index]
    vf_fields.update({name: rad[name][veg_index]
                      for name in ("sav", "sag", "laisun", "laisha",
                                   "parsun", "parsha", "fsr")})
    vf_fields.update({name: g[name][veg_index]
                      for name in ("ur", "vai", "cwp", "zlvl", "zpd",
                                   "z0m", "z0mg")})
    vf_fields.update({name: e[name][veg_index]
                      for name in ("gammav", "gammag", "emv", "emg",
                                   "rsurf", "latheav", "btran", "rhsur")})
    vf_fields["latheag"] = e["latheag"][veg_index]
    vf_fields.update({name: par(name)[veg_index]
                      for name in _vege.PARAMETER_NAMES})
    vf_fields.update({"isnow": c("isnow")[veg_index],
                      "dzsnso": c("dzsnso")[veg_index],
                      "stc": c("stc")[veg_index],
                      "df": thermo["df"][veg_index]})
    vf = evaluate_vege_flux_slab(vf_fields, nveg)

    def spread(sub):
        full = cp.zeros(n, dtype=cp.float32)
        full[veg_index] = sub
        return full

    v = {}
    for name, upper in _VEGE_TO_ENERGY.items():
        scattered = spread(vf[upper])
        fallback = _VEGE_ENTRY_FALLBACK.get(name)
        if fallback is not None:
            scattered = cp.where(tile_veg, scattered, c(fallback))
        v[name] = scattered
    tah = cp.where(tile_veg, spread(vf["TAH"]), c("tah"))
    canhs = cp.where(tile_veg, spread(vf["CANHS"]), _ZERO)
    # :2262 -- the scalar path assigns qsfc = vf.QSFC before BARE_FLUX reads
    # it, so on a vegetated column BARE_FLUX's qsfc argument is VEGE_FLUX's.
    qsfc_after_vege = cp.where(tile_veg, spread(vf["QSFC"]), c("qsfc"))

    bf = evaluate_bare_flux_slab({
        "dt": c("dt"), "sag": rad["sag"], "lwdn": c("lwdn"), "ur": g["ur"],
        "uu": c("uu"), "vv": c("vv"), "sfctmp": c("sfctmp"),
        "thair": c("thair"), "qair": c("qair"), "eair": c("eair"),
        "rhoair": c("rhoair"), "snowh": c("snowh"), "zlvl": g["zlvl"],
        "zpd": g["zpdg"], "z0m": g["z0mg"], "fsno": g["fsno"],
        "emg": e["emg"], "rsurf": e["rsurf"], "lathea": e["latheag"],
        "gamma": e["gammag"], "rhsur": e["rhsur"], "q2": c("q2"),
        "pahb": c("pahb"), "dx": c("dx"), "dz8w": c("dz8w"), "qc": c("qc"),
        "psfc": c("psfc"), "sfcprs": c("sfcprs"),
        "tgb": c("tg"), "cm": c("cm"), "ch": c("ch"),
        "qsfc": qsfc_after_vege,
        "dzsnso": c("dzsnso"), "stc": c("stc"), "df": thermo["df"],
        "isnow": c("isnow"), "ivgtyp": c("vegtyp"),
        "iloc": np.int32(1), "jloc": np.int32(1),
        "urban_flag": urban}, n)
    b = dict(bf)
    b["cmb"] = b.pop("cm")
    b["chb"] = b.pop("ch")

    # ---- ENERGY seg3, the soil column, seg4 --------------------------------
    s3 = dict(s)
    s3.update({name: rad[name]
               for name in ("laisun", "laisha", "parsun", "parsha")})
    seg3 = energy_seg3(s3, g, e, v, b)

    tsn = evaluate_tsnosoi_slab({
        "zsnso": c("zsnso"), "stc": c("stc"), "df": thermo["df"],
        "hcpct": thermo["hcpct"], "tbot": c("tbot"),
        "ssoil": seg3["ssoil_avg"], "dt": seg3["dt_soil"],
        "snowh": c("snowh"), "zbot": par("zbot"), "isnow": c("isnow")}, n)

    pc = evaluate_phasechange_slab({
        "fact": thermo["fact"], "dzsnso": c("dzsnso"), "stc": tsn["stc"],
        "snice": c("snice"), "snliq": c("snliq"), "smc": c("smc"),
        "sh2o": c("sh2o"), "sneqv": c("sneqv"), "snowh": c("snowh"),
        "dt": c("dt"), "smcmax": s["smcmax"], "psisat": s["psisat"],
        "bexp": s["bexp"], "isnow": c("isnow"), "ist": c("ist")}, n)

    seg4 = energy_seg4(s, g, {**e, **seg3}, {
        "eflxb": tsn["eflxb"], "hcpct": thermo["hcpct"], "imelt": pc["imelt"],
        "snicev": thermo["snicev"], "snliqv": thermo["snliqv"],
        "epore": thermo["epore"], "fsrv": rad["fsrv"], "fsrg": rad["fsrg"],
        "bgap": rad["bgap"], "wgap": rad["wgap"]})

    # ---- NOAHMP_SFLX's second half, 979-1076 -------------------------------
    p1 = sflx_post_seg1({
        "stc": pc["stc"], "dzsnso": pre["dzsnso"], "snice": pc["snice"],
        "snliq": pc["snliq"], "sh2o": pc["sh2o"], "smc": pc["smc"],
        "snowh": pc["snowh"], "sneqv": pc["sneqv"],
        "fgev": seg3["fgev"], "latheag": e["latheag"]})

    # The ten ACC_* accumulators are zero on entry to every call under the
    # pinned soiltstep = 0 -- see NOAHMP_RUNTIME_RESTRICTIONS -- and every
    # consumer below only reads them, so one zero buffer per shape serves all.
    acc_zero = cp.zeros(n, dtype=cp.float32)
    acc_zero_soil = cp.zeros((n, NSOIL), dtype=cp.float32)
    wat = evaluate_water_slab({
        "smcmax": s["smcmax"], "smcwlt": s["smcwlt"], "bexp": s["bexp"],
        "dksat": par("dksat"), "dwsat": par("dwsat"),
        "kdt": par("kdt"), "frzx": par("frzx"), "slope": par("slope"),
        "ch2op": par("ch2op"), "ssi": par("ssi"),
        "snow_ret_fac": par("snow_ret_fac"),
        "snowh": p1["snowh"], "sneqv": p1["sneqv"], "snice": p1["snice"],
        "snliq": p1["snliq"], "stc": p1["stc"], "zsnso": c("zsnso"),
        "dzsnso": p1["dzsnso"], "sh2o": p1["sh2o"], "sice": p1["sice"],
        "dt": c("dt"), "fcev": seg3["fcev"], "fctr": seg3["fctr"],
        "elai": c("elai"), "esai": c("esai"), "fveg": c("fveg"),
        "bdfall": pre["bdfall"], "sfctmp": c("sfctmp"),
        "qvap": p1["qvap"], "qdew": p1["qdew"],
        "qsnow": c("qsnow"), "qrain": pre["qrain"],
        "snowhin": pre["snowhin"], "ponding": pc["ponding"],
        "canliq": c("canliq"), "canice": c("canice"), "tv": v["tv"],
        "wslake": _shaped(cp, fields, "wslake", n, 1),
        "acc_qinsur": acc_zero, "acc_qseva": acc_zero,
        "zsoil": c("zsoil"), "btrani": seg4["btrani"],
        "acc_etrani": acc_zero_soil, "smc": p1["smc"],
        "ficeold": _shaped(cp, fields, "ficeold", n, NSNOW),
        "isnow": c("isnow"), "urban_flag": urban, "nroot": s["nroot"],
        "ist": c("ist"), "frozen_canopy": e["frozen_canopy"],
        "frozen_ground": e["frozen_ground"],
        "imelt": seg4["imelt"][:, :NSNOW]}, n)

    p2 = sflx_post_seg2({
        "edir": p1["edir"],
        "fsa": rad["fsa"], "fsr": rad["fsr"], "fira": seg3["fira"],
        "fsh": seg3["fsh"], "fcev": seg3["fcev"], "fgev": seg3["fgev"],
        "fctr": seg3["fctr"], "ssoil": seg3["ssoil"], "sav": rad["sav"],
        "sag": rad["sag"], "pah": seg3["pah"], "canhs": canhs,
        "qsfc": b["qsfc"], "q2b": b["q2b"], "ch": seg3["ch"],
        "swdown": pre["swdown"], "beg_wb": pre["beg_wb"],
        "prcp": pre["prcp"], "rhoair": pre["rhoair"], "qair": pre["qair"],
        "canliq": wat["canliq"], "canice": wat["canice"],
        "ecan": wat["ecan"], "etran": wat["etran"], "runsrf": wat["runsrf"],
        "runsub": wat["runsub"], "qtldrn": wat["qtldrn"], "smc": wat["smc"],
        "sneqv": wat["sneqv"], "snowh": wat["snowh"],
        "dzsnso": wat["dzsnso"],
        "wa": _shaped(cp, fields, "wa", n, 1), "dt": c("dt"),
        "acc_dwater": acc_zero, "acc_prcp": acc_zero,
        "acc_ecan": acc_zero, "acc_etran": acc_zero, "acc_edir": acc_zero,
        "ist": c("ist"), "urban_flag": urban})

    # ---- the answer, under the SflxResult names ----------------------------
    return {
        "trad": seg3["trad"], "fsh": p2["fsh"], "ecan": wat["ecan"],
        "edir": p1["edir"], "etran": wat["etran"], "fcev": seg3["fcev"],
        "fgev": seg3["fgev"], "fctr": seg3["fctr"], "ssoil": seg3["ssoil"],
        "runsrf": wat["runsrf"], "runsub": wat["runsub"],
        "albedo": p2["albedo"], "fsno": g["fsno"], "canliq": wat["canliq"],
        "canice": wat["canice"], "fpice": pre["fpice"], "qmelt": pc["qmelt"],
        "ponding": pc["ponding"], "ponding1": wat["ponding1"],
        "ponding2": wat["ponding2"], "emissi": seg3["emissi"],
        "qsfc": p2["qsfc"], "tv": wat["tv"], "tg": seg3["tg"],
        "eah": v["eah"], "tah": tah, "cm": seg3["cm"], "ch": seg3["ch"],
        "fwet": wat["fwet"], "sneqvo": p1["sneqvo"], "albold": rad["albold"],
        "qsnow": c("qsnow"), "qrain": pre["qrain"], "wslake": wat["wslake"],
        "lai": pre["lai"], "sai": pre["sai"], "tauss": rad["tauss"],
        "z0wrf": seg3["z0wrf"], "t2mv": v["t2mv"], "t2mb": b["t2mb"],
        "q2v": v["q2v"], "q2b": p2["q2b"], "fveg": pre["fveg"],
        "fsa": rad["fsa"], "fira": seg3["fira"], "rssun": seg3["rssun"],
        "rssha": seg3["rssha"], "chv": seg3["chv"], "chb": b["chb"],
        "canhs": canhs, "qsnbot": wat["qsnbot"], "eflxb": seg4["eflxb"],
        "laisun": rad["laisun"], "laisha": rad["laisha"],
        "rb": cp.zeros(n, dtype=cp.float32),
        "isnow": wat["isnow"], "snowh": p2["snowh"], "sneqv": p2["sneqv"],
        "snice": wat["snice"], "snliq": wat["snliq"], "stc": wat["stc"],
        "zsnso": wat["zsnso"], "sh2o": wat["sh2o"], "smc": wat["smc"],
        "hcpct": seg4["hcpct"],
        "errwat": p2["errwat"], "end_wb": p2["end_wb"],
        "errsw": p2["errsw"], "erreng": p2["erreng"], "beg_wb": pre["beg_wb"],
    }


__all__ = [
    "OUTPUT_NAMES",
    "PARAMETER_INTS",
    "PARAMETER_WIDTHS",
    "evaluate_sflx_slab",
    "parameter_fields",
    "parameter_row",
]
