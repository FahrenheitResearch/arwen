"""Noah land surface model: parameter tables and the column-kernel launcher.

The physics is transcribed line-faithfully from the bundle's WRF v4.6.1
``phys/module_sf_noahdrv.F`` (subroutine ``lsm``: the per-column input
prep and the post-``SFLX`` state/flux updates) and
``phys/module_sf_noahlsm.F`` (``SFLX`` and its full subtree).  One CUDA
thread integrates one land column (kernels/noah.cu); the float64
verification mirror is :func:`gpuwm.verify.npref.np_noah_column`.

Scope (Phase 3 Task 10): 4 soil layers, snowpack (SWE/depth/density,
melt, cover fraction, snow albedo aging), canopy water, the full surface
energy balance producing TSK/HFX/QFX/LH/GRDFLX, runoff, and the WRF
driver diagnostics (SMSTAV/SMSTOT/SMCREL/ACSNOM/ACSNOW/SNOPCX/POTEVP/
NOAHRES).  NOT ported, matching the plan's authority-file scope: UA_PHYS,
FASDAS, WRF-Hydro, the urban canopy models (the plain ``VEGTYP==ISURBAN``
parameter overrides inside SFLX/HRT ARE ported), ``SFCDIF_off`` (the
exchange coefficient CH comes from the surface-layer scheme), and
``SFLX_GLACIAL`` (module_sf_noahlsm_glacial_only.F): land-ice columns
(``ivgtyp == isice``) are skipped like water points.  The bundle's d01
has no land-ice points (pinned in tests/test_noah.py).

Parameter tables: VEGPARM.TBL / SOILPARM.TBL / GENPARM.TBL are shipped
verbatim from the WRF v4.6.1 ``run/`` directory in
``gpuwm/data/noah_tables`` (see PROVENANCE.md there) and parsed here by
transcribing the list-directed READ sequence of ``SOIL_VEG_GEN_PARM``
(module_sf_noahdrv.F).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TBL_DIR = Path(__file__).resolve().parent.parent / "data" / "noah_tables"

#: Noah's soil geometry, stated once, beside the scheme that owns it.
#:
#: WRF resolves the soil-layer count from ``sf_surface_physics`` rather than
#: honouring the namelist request: ``share/module_check_a_mundo.F`` subroutine
#: ``set_physics_rconfigs`` sets ``num_soil_layers = 4`` for LSMSCHEME (:3553)
#: and for NOAHMPSCHEME (:3555), and both schemes take their layer depths from
#: the *same* generator, ``init_soil_depth_2``
#: (``share/module_soil_pre.F:795`` for Noah, ``:807`` for Noah-MP).  That
#: shared generator -- not a coincidence of two literals -- is why
#: :data:`gpuwm.config.LAND_SURFACE_SOIL_LAYERS` sources the Noah-MP entry
#: from this constant.  ``init_soil_depth_2`` itself carries WRF's fatal for
#: any other request ("The Noah and NoahMP LSMs use 4 layers.",
#: ``share/module_soil_pre.F:1140-1143``), so 4 is not a gpuwm choice.
#: ``init_soil_depth_2`` emits dzs, whose accumulation is the layer-bound
#: table in :mod:`gpuwm.ingest.soil_contract`; that module checks itself
#: against this count so the two cannot drift.
NUM_SOIL_LAYERS = 4
#: ``init_soil_depth_2``'s dzs (share/module_soil_pre.F:1138).
SOIL_LAYER_THICKNESS_M = (0.10, 0.30, 0.60, 1.00)

# Packed-table column layouts shared by the CUDA kernel and the mirror.
VEG_COLS = ("nroot", "rsmin", "rgl", "hs", "snup", "laimin", "laimax",
            "emissmin", "emissmax", "albedomin", "albedomax",
            "z0min", "z0max", "shdtbl", "maxalb")
SOIL_COLS = ("bexp", "smcdry", "f1", "smcmax", "smcref", "psisat",
             "dksat", "dwsat", "smcwlt", "quartz")
GEN = {name: i for i, name in enumerate(
    ("topt", "cmcmax", "cfactr", "rsmax", "sbeta", "fxexp", "csoil",
     "salp", "refdk", "refkdt", "frzk", "zbot", "lvcoef", "slope",
     "bare", "natural"))}

_TPB = 64          # threads per block (one thread per land column)


# ---------------------------------------------------------------------------
# table parsing (SOIL_VEG_GEN_PARM transcription)
# ---------------------------------------------------------------------------

def _tokens(line: str) -> list[str]:
    """Fortran list-directed items of one record: comma/space separated,
    quoted strings kept whole."""
    out = []
    for m in re.finditer(r"'[^']*'|[^,\s]+", line):
        tok = m.group(0)
        out.append(tok)
    return out


@dataclass
class NoahTables:
    """Raw parsed tables (float64), one attribute per Fortran table array."""
    # VEGPARM
    lutype: str = ""
    lucats: int = 0
    shdtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nrotbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rstbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rgltbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hstbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    snuptbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    maxalb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    laimintbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    laimaxtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    emissmintbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    emissmaxtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    albedomintbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    albedomaxtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z0mintbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z0maxtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ztopvtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    zbotvtbl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    topt: float = 0.0
    cmcmax: float = 0.0
    cfactr: float = 0.0
    rsmax: float = 0.0
    bare: int = 0
    natural: int = 0
    # SOILPARM
    sltype: str = ""
    slcats: int = 0
    bb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    drysmc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f11: np.ndarray = field(default_factory=lambda: np.zeros(0))
    maxsmc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    refsmc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    satpsi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    satdk: np.ndarray = field(default_factory=lambda: np.zeros(0))
    satdw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wltsmc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    qtz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # GENPARM
    slope_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sbeta: float = 0.0
    fxexp: float = 0.0
    csoil: float = 0.0
    salp: float = 0.0
    refdk: float = 0.0
    refkdt: float = 0.0
    frzk: float = 0.0
    zbot: float = 0.0
    czil: float = 0.0
    smlow: float = 0.0
    smhigh: float = 0.0
    lvcoef: float = 0.0


def _parse_vegparm(text: str, mminlu: str, t: NoahTables) -> None:
    lines = iter(text.splitlines())
    next(lines)                                     # header line
    while True:                                     # FIND_LUTYPE
        lutype = _tokens(next(lines))[0].strip("'")
        cats_line = _tokens(next(lines))
        lucats = int(cats_line[0])
        if lutype == mminlu:
            break
        # skip to the next 'Vegetation Parameters' flag line
        for line in lines:
            if line.startswith("Vegetation Parameters"):
                break
        else:
            raise ValueError(f"landuse dataset {mminlu!r} not found "
                             f"in VEGPARM.TBL")
    t.lutype, t.lucats = lutype, lucats
    names = ("shdtbl nrotbl rstbl rgltbl hstbl snuptbl maxalb laimintbl "
             "laimaxtbl emissmintbl emissmaxtbl albedomintbl albedomaxtbl "
             "z0mintbl z0maxtbl ztopvtbl zbotvtbl").split()
    cols = {n: np.zeros(lucats) for n in names}
    for _ in range(lucats):
        toks = _tokens(next(lines))
        lc = int(float(toks[0])) - 1
        for n, tok in zip(names, toks[1:1 + len(names)]):
            cols[n][lc] = float(tok)
    for n in names:
        setattr(t, n, cols[n])
    # scalar block: label line then value line, in the READ order
    def val():
        next(lines)
        return _tokens(next(lines))[0]
    t.topt = float(val())
    t.cmcmax = float(val())
    t.cfactr = float(val())
    t.rsmax = float(val())
    t.bare = int(val())
    t.natural = int(val())
    # the Fortran skips two records (CROP label + value) then checks the
    # next record is not a new section before the LCZ block; the LCZ
    # indices are urban-model-only and unused here.


def _parse_soilparm(text: str, mminsl: str, t: NoahTables) -> None:
    lines = iter(text.splitlines())
    while True:
        line = next(lines)                          # 'Soil Parameters'
        sltype = _tokens(next(lines))[0].strip("'")
        cats = int(_tokens(next(lines))[0])
        if sltype == mminsl:
            break
        for _ in range(cats):                       # skip this section
            next(lines)
    t.sltype, t.slcats = sltype, cats
    names = "bb drysmc f11 maxsmc refsmc satpsi satdk satdw wltsmc qtz".split()
    cols = {n: np.zeros(cats) for n in names}
    for _ in range(cats):
        toks = _tokens(next(lines))
        lc = int(float(toks[0])) - 1
        for n, tok in zip(names, toks[1:1 + len(names)]):
            cols[n][lc] = float(tok)
    for n in names:
        setattr(t, n, cols[n])


def _parse_genparm(text: str, t: NoahTables) -> None:
    lines = iter(text.splitlines())
    next(lines)                                     # 'General Parameters'
    next(lines)                                     # 'SLOPE_DATA'
    n = int(_tokens(next(lines))[0])
    t.slope_data = np.array([float(_tokens(next(lines))[0])
                             for _ in range(n)])
    for name in ("sbeta", "fxexp", "csoil", "salp", "refdk", "refkdt",
                 "frzk", "zbot", "czil", "smlow", "smhigh", "lvcoef"):
        next(lines)                                 # label record
        setattr(t, name, float(_tokens(next(lines))[0]))


def load_tables(mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
                mminsl: str = "STAS",
                tbl_dir: Path | None = None) -> NoahTables:
    """Parse the shipped WRF run-directory tables (SOIL_VEG_GEN_PARM)."""
    d = Path(tbl_dir) if tbl_dir is not None else TBL_DIR
    t = NoahTables()
    _parse_vegparm((d / "VEGPARM.TBL").read_text(), mminlu, t)
    _parse_soilparm((d / "SOILPARM.TBL").read_text(), mminsl, t)
    _parse_genparm((d / "GENPARM.TBL").read_text(), t)
    return t


# ---------------------------------------------------------------------------
# packing for the kernel / mirror
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoahParams:
    """Packed float64 parameter tables (rows = category-1; columns per
    VEG_COLS / SOIL_COLS / GEN)."""
    veg: np.ndarray
    soil: np.ndarray
    gen: np.ndarray
    lucats: int
    slcats: int
    bare: int
    natural: int
    lutype: str
    sltype: str


def pack_params(t: NoahTables) -> NoahParams:
    veg = np.zeros((t.lucats, len(VEG_COLS)))
    src = dict(nroot=t.nrotbl, rsmin=t.rstbl, rgl=t.rgltbl, hs=t.hstbl,
               snup=t.snuptbl, laimin=t.laimintbl, laimax=t.laimaxtbl,
               emissmin=t.emissmintbl, emissmax=t.emissmaxtbl,
               albedomin=t.albedomintbl, albedomax=t.albedomaxtbl,
               z0min=t.z0mintbl, z0max=t.z0maxtbl, shdtbl=t.shdtbl,
               maxalb=t.maxalb)
    for i, name in enumerate(VEG_COLS):
        veg[:, i] = src[name]
    soil = np.zeros((t.slcats, len(SOIL_COLS)))
    ssrc = dict(bexp=t.bb, smcdry=t.drysmc, f1=t.f11, smcmax=t.maxsmc,
                smcref=t.refsmc, psisat=t.satpsi, dksat=t.satdk,
                dwsat=t.satdw, smcwlt=t.wltsmc, quartz=t.qtz)
    for i, name in enumerate(SOIL_COLS):
        soil[:, i] = ssrc[name]
    gen = np.zeros(len(GEN))
    for name, i in GEN.items():
        if name == "slope":
            gen[i] = t.slope_data[0]     # driver: SLOPETYP = 1 always
        elif name == "bare":
            gen[i] = t.bare
        elif name == "natural":
            gen[i] = t.natural
        else:
            gen[i] = getattr(t, name)
    return NoahParams(veg=veg, soil=soil, gen=gen, lucats=t.lucats,
                      slcats=t.slcats, bare=t.bare, natural=t.natural,
                      lutype=t.lutype, sltype=t.sltype)


# ---------------------------------------------------------------------------
# LSMINIT soil-liquid-water helper (CPU, float64)
# ---------------------------------------------------------------------------

def noah_frh2o(tkelv: float, smc: float, sh2o: float, smcmax: float,
                bexp: float, psis: float) -> float:
    """WRF Noah ``FRH2O`` supercooled-liquid-water solve in float64.

    This is the setup-time CPU twin of ``noah_frh2o`` in ``noah.cu`` and
    follows ``module_sf_noahlsm.F:1447-1585``: the CK=8 log-form Newton
    iteration is bounded to ten iterations, with the CK=0 explicit fallback.
    """
    ck, blim, error = 8.0, 5.5, 0.005
    hlice, gs, t0 = 3.335e5, 9.81, 273.15
    bx = bexp if bexp <= blim else blim
    nlog = 0
    kcount = 0
    if tkelv > (t0 - 1.0e-3):
        return smc
    swl = smc - sh2o
    if swl > (smc - 0.02):
        swl = smc - 0.02
    if swl < 0.0:
        swl = 0.0
    while (nlog < 10) and (kcount == 0):
        nlog += 1
        df = (math.log((psis * gs / hlice) * ((1.0 + ck * swl) ** 2.0)
                       * (smcmax / (smc - swl)) ** bx)
              - math.log(-(tkelv - t0) / tkelv))
        denom = 2.0 * ck / (1.0 + ck * swl) + bx / (smc - swl)
        swlk = swl - df / denom
        if swlk > (smc - 0.02):
            swlk = smc - 0.02
        if swlk < 0.0:
            swlk = 0.0
        dswl = abs(swlk - swl)
        swl = swlk
        if dswl <= error:
            kcount += 1
    free = smc - swl
    if kcount == 0:
        fk = (((hlice / (gs * (-psis))) * ((tkelv - t0) / tkelv))
              ** (-1.0 / bx)) * smcmax
        if fk < 0.02:
            fk = 0.02
        free = min(fk, smc)
    return free


def sh2o_init(smois, tslb, isltyp, params: NoahParams) -> np.ndarray:
    """SH2O from SMOIS/TSLB exactly as LSMINIT (module_sf_noahdrv.F):
    Flerchinger explicit first guess, then the FRH2O Newton iteration."""
    smois = np.asarray(smois, np.float64)
    tslb = np.asarray(tslb, np.float64)
    if smois.shape != tslb.shape or smois.ndim == 0:
        raise ValueError("smois and tslb must be same-shape soil profiles")
    column_shape = smois.shape[1:]
    soil_type = np.asarray(isltyp)
    try:
        soil_type = np.broadcast_to(soil_type, column_shape)
    except ValueError as exc:
        raise ValueError("isltyp must match the soil-profile columns") from exc
    if (not np.isfinite(soil_type).all()
            or np.any(soil_type != np.floor(soil_type))):
        raise ValueError("isltyp must contain finite integer categories")

    out = smois.copy()
    blim, hlice, grav, t0 = 5.5, 3.335e5, 9.81, 273.15
    for column in np.ndindex(column_shape):
        category = int(soil_type[column])
        if category < 1 or category > params.slcats:
            raise ValueError(f"isltyp category {category} is outside table")
        row = params.soil[category - 1]
        bx = row[SOIL_COLS.index("bexp")]
        smcmax = row[SOIL_COLS.index("smcmax")]
        psisat = row[SOIL_COLS.index("psisat")]
        if not (bx > 0.0 and smcmax > 0.0 and psisat > 0.0):
            continue
        bx = min(bx, blim)
        for k in range(smois.shape[0]):
            index = (k, *column)
            if tslb[index] >= 273.149:
                continue
            fk = (((hlice / (grav * (-psisat)))
                   * ((tslb[index] - t0) / tslb[index]))
                  ** (-1.0 / bx)) * smcmax
            if fk < 0.02:
                fk = 0.02
            guess = min(fk, smois[index])
            out[index] = noah_frh2o(
                tslb[index], smois[index], guess, smcmax, bx, psisat)
    return out


# ---------------------------------------------------------------------------
# kernel launcher
# ---------------------------------------------------------------------------

# ordered 2-D real field names, matching the noah_column kernel signature
_F2D = ("psfc", "sfcprs", "sfctmp", "qv1", "qgh", "dz8w1", "glw",
        "swdown", "rainbl", "sr", "chs", "cqs2", "chs2", "rib",
        "vegfra", "shdmin", "shdmax", "tmn", "xland", "xice", "snoalb",
        "embck",
        "tsk", "hfx", "qfx", "lh", "grdflx", "qsfc",
        "canwat", "snow", "snowc", "snowh",
        "albedo", "albbck", "emiss", "znt", "z0", "snotime", "lai",
        "smstav", "smstot",
        "sfcrunoff", "udrunoff", "acsnow", "acsnom", "snopcx", "potevp",
        "noahres", "reslin", "chklowq")
_F3D = ("smois", "tslb", "sh2o", "smcrel")


def launch_noah(dev: dict, params: NoahParams, dt: float, dzs,
                isurban: int = 13, isice: int = 15,
                xice_threshold: float = 0.5, frpcpn: bool = False,
                usemonalb: bool = False, rdlai2d: bool = False,
                opt_thcnd: int = 1, itimestep: int = 2) -> None:
    """Run the ``noah_column`` kernel on device arrays (in place).

    ``dev`` maps field names to CuPy arrays: every name in ``_F2D`` as
    ``(ny, nx)`` float32 (``ivgtyp``/``isltyp`` int32), every name in
    ``_F3D`` as ``(4, ny, nx)`` float32, plus ``ebal`` (ny, nx) int32.
    Water (``xland >= 1.5``), sea-ice (``xice >= xice_threshold``) and
    land-ice (``ivgtyp == isice``) columns are skipped exactly as the
    WRF driver skips them.
    """
    import cupy as cp

    from gpuwm.core.kernels import get_kernel
    from gpuwm.core.state import DTYPE

    ny, nx = dev["tsk"].shape
    if len(dzs) != NUM_SOIL_LAYERS:
        raise ValueError(
            f"Noah port is fixed at {NUM_SOIL_LAYERS} soil layers")
    if (isinstance(itimestep, bool) or not isinstance(itimestep, int)
            or itimestep < 1):
        raise ValueError("Noah itimestep must be a positive integer")
    args = [dev["ivgtyp"], dev["isltyp"]]
    for name in _F2D:
        a = dev[name]
        if a.shape != (ny, nx) or a.dtype != cp.float32:
            raise ValueError(f"{name}: expected ({ny},{nx}) float32")
        args.append(a)
    for name in _F3D:
        a = dev[name]
        if a.shape != (4, ny, nx) or a.dtype != cp.float32:
            raise ValueError(f"{name}: expected (4,{ny},{nx}) float32")
        args.append(a)
    args.append(dev["ebal"])
    vegtbl = cp.asarray(params.veg.astype(np.float32).ravel())
    soiltbl = cp.asarray(params.soil.astype(np.float32).ravel())
    genp = cp.asarray(params.gen.astype(np.float32))
    dzs4 = cp.asarray(np.asarray(dzs, np.float32))
    args += [vegtbl, soiltbl, genp, dzs4,
             DTYPE(dt), np.int32(params.lucats), np.int32(params.slcats),
             np.int32(isurban), np.int32(isice), DTYPE(xice_threshold),
             np.int32(itimestep),
             np.int32(1 if frpcpn else 0), np.int32(1 if usemonalb else 0),
             np.int32(1 if rdlai2d else 0), np.int32(opt_thcnd),
             np.int32(ny), np.int32(nx)]
    kern = get_kernel("noah", "noah_column")
    blocks = (ny * nx + _TPB - 1) // _TPB
    kern((blocks,), (_TPB,), tuple(args))
