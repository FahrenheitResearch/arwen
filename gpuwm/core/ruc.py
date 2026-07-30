"""RUC LSM parameter loading, geometry, cold start, and surface setup.

The routines here transcribe executable pieces from WRF v4.6.1
``share/module_soil_pre.F:init_soil_depth_3`` and
``phys/module_sf_ruclsm.F`` (``ruclsminit``, ``soilvegin``, ``soilprop``,
``transf``, the ``qsn`` saturation lookup, the implicit ``soilmoist`` state
solve, the snow-free ``soiltemp`` heat/surface solve, and the snow-free
``sice`` sea-ice column).  They intentionally do not route
``sf_surface_physics=3`` yet; snow, top-level fluxes, and driver coupling
remain required for selector admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from functools import lru_cache
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gpuwm.core.noahmp import (
    NoahmpGeneralTable,
    NoahmpSoilTable,
    parse_genparm,
    parse_soilparm,
)
from gpuwm.core.ruc_contract import (
    NUM_SOIL_LAYERS,
    RUC_SOIL_LEVELS_M,
    RUC_TABLE_ASSETS,
)

#: numpy under a private name, so a function can shadow ``np`` with a caller's
#: array namespace and still reach the default.
#:
#: :func:`ruc_surface_temperature_step` takes an ``arrays=`` namespace and
#: rebinds ``np`` to it for the whole body.  That is what lets ONE
#: transcription of the ``sfctmp`` dispatch run either on host numpy arrays --
#: which is what the oracle suite compares against WRF -- or on device arrays
#: with the four leaves and the preparation stage as CUDA kernels, without a
#: second copy of the masking and the mosaic recombination to keep in step.
#: The alternative was a device mirror of the dispatch, and this lane has
#: already recorded what mirrors cost: a mirror is not an oracle.
#:
#: ``arrays=None`` means numpy, and then ``np`` inside the body IS this module
#: object, so the host path is not a configuration of the new mechanism -- it
#: is unchanged code.
_NUMPY = np


_RUC_PROVISIONAL_TRANSCENDENTALS = """\
PROVISIONAL float32 transcendentals, pending a verified glibc transcription.

gfortran lowers ``**``, ``exp``, ``log`` and ``log10`` on default REAL to
glibc's powf/expf/logf/log10f, and glibc's are NOT correctly rounded -- 2.39
still ships the 1993 SunPro float32 reduction for log10f.  Evaluating in
float64 and rounding once therefore produces a THIRD function, not a more
accurate glibc.  Measured against the live glibc 2.39 (a C program built
-O0 -fno-builtin, calling the real libm) over the 30720 distinct float32
arguments the RUC snow-covered land column reaches:

    float64-then-round-once (what this module uses)   783 / 30720 =  2.55 %
    CUDA device libm  (powf/expf/logf/log10f)        3030 / 30720 =  9.86 %
    numpy 2.2.6 float32 loop                         4420 / 30720 = 14.4 %
    numpy 2.4.3 float32 loop                         8901 / 30720 = 29.0 %

All deviations are 1 ULP for the float64 path and up to 2 ULP for the other
three.  The float64 path is the closest of the four and, unlike the others,
is identical on the host and the device.  Of the residual 783, 723 are the
freezing-curve ``log`` whose only consumer is a ``tln < 0.`` sign test (zero
sign flips measured) and 34 are the ``log10`` feeding WRF's discarded legacy
McCumber conductivity, leaving ~26 consequential 1-ULP deviations.

At the column level the cost is smaller still.  Running the whole snowsoil
CPU lane with these ufuncs routed through ctypes to the live libm gives a
gfortran-faithful reference; over an 84-column perturbed ensemble (7224
output values) the Windows CPU, the WSL CPU and the CUDA lanes are identical
to each other and each leaves that reference in exactly ONE value -- 2 ULP of
``smfrkeep`` at one level of one column, from the freezing-curve
``pow(2446.8157, -0.12903225)``, which propagates to nothing else.  Before
these substitutions the CPU and CUDA lanes disagreed with EACH OTHER on 13 of
38 columns, with 3 ULP of ``thdif`` amplified to 53248 ULP of ``fltot``.

Every pinned RUC fixture reproduces bit for bit under all four
implementations, including the ctypes-glibc reference, so fixture parity does
not discriminate between them; that is precisely why the ensemble exists.
Replace all of these with the verified glibc transcription when it lands.
"""

RUC_TABLE_DIR = Path(__file__).resolve().parent.parent / "data" / "noah_tables"
RUC_ORACLE_DIR = Path(__file__).resolve().parent.parent / "data" / "ruc" / "oracle"
RUC_VEGETATION_DATASETS = MappingProxyType({
    "USGS": "USGS-RUC",
    "MODIS": "MODI-RUC",
    "MODIFIED_IGBP_MODIS_NOAH": "MODI-RUC",
})
RUC_VEGETATION_FIELDS = (
    "albedo",
    "z0",
    "lemi",
    "pc",
    "shdfac",
    "ifor",
    "rs",
    "rgl",
    "hs",
    "snup",
    "lai",
    "maxalb",
)
RUC_VEGETATION_SCALARS = (
    "TOPT_DATA",
    "CMCMAX_DATA",
    "CFACTR_DATA",
    "RSMAX_DATA",
    "BARE",
    "NATURAL",
    "CROP",
    "URBAN",
)


@dataclass(frozen=True)
class RucVegetationRow:
    category: int
    albedo: float
    z0: float
    lemi: float
    pc: float
    shdfac: float
    ifor: int
    rs: float
    rgl: float
    hs: float
    snup: float
    lai: float
    maxalb: float
    description: str


@dataclass(frozen=True)
class RucVegetationTable:
    name: str
    rows: tuple[RucVegetationRow, ...]
    scalars: Mapping[str, int | float]

    def row(self, category: int) -> RucVegetationRow:
        if type(category) is not int or not 1 <= category <= len(self.rows):
            raise ValueError(
                f"{self.name} vegetation category {category!r} is outside "
                f"1..{len(self.rows)}"
            )
        return self.rows[category - 1]


@dataclass(frozen=True)
class RucParameterBundle:
    vegetation: Mapping[str, RucVegetationTable]
    soil: NoahmpSoilTable
    general: NoahmpGeneralTable
    receipt: Mapping[str, object]

    def vegetation_for(self, mminlu: str) -> RucVegetationTable:
        try:
            name = RUC_VEGETATION_DATASETS[mminlu]
        except KeyError as exc:
            raise ValueError(
                "RUC MMINLU must be USGS, MODIS, or "
                "MODIFIED_IGBP_MODIS_NOAH"
            ) from exc
        return self.vegetation[name]


@dataclass(frozen=True)
class RucInitialization:
    """Cold-start fields returned in gpuwm soil-first array order."""

    sh2o: np.ndarray
    smfr3d: np.ndarray
    mavail: np.ndarray
    znt: np.ndarray


@dataclass(frozen=True)
class RucSurfaceParameters:
    """Dominant-category outputs from WRF ``soilvegin``."""

    iforest: np.ndarray
    emiss: np.ndarray
    pc: np.ndarray
    znt: np.ndarray
    lai: np.ndarray
    qwrtz: np.ndarray
    rhocs: np.ndarray
    bclh: np.ndarray
    dqm: np.ndarray
    ksat: np.ndarray
    psis: np.ndarray
    qmin: np.ndarray
    ref: np.ndarray
    wilt: np.ndarray


@dataclass(frozen=True)
class RucSoilProperties:
    """Thermal and hydraulic coefficients from WRF ``soilprop``."""

    thdif: np.ndarray
    diffu: np.ndarray
    hydro: np.ndarray
    cap: np.ndarray


@dataclass(frozen=True)
class RucTranspiration:
    """Root-zone transpiration weights from WRF ``transf``."""

    tranf: np.ndarray
    transum: np.ndarray


@dataclass(frozen=True)
class RucSoilMoisture:
    """Nine-level water state and fluxes returned by WRF ``soilmoist``."""

    soilmois: np.ndarray
    soiliqw: np.ndarray
    mavail: np.ndarray
    runoff: np.ndarray
    runoff2: np.ndarray
    infiltrp: np.ndarray
    infmax: np.ndarray


@dataclass(frozen=True)
class RucSoilTemperature:
    """Nine-level heat state and skin diagnostics returned by ``soiltemp``."""

    tso: np.ndarray
    soilt: np.ndarray
    qvg: np.ndarray
    qsg: np.ndarray
    qcg: np.ndarray
    storage: np.ndarray


@dataclass(frozen=True)
class RucSoilStep:
    """Complete snow-free land state and fluxes from WRF ``soil``."""

    soilmois: np.ndarray
    tso: np.ndarray
    smfrkeep: np.ndarray
    keepfr: np.ndarray
    soilice: np.ndarray
    soiliqw: np.ndarray
    cst: np.ndarray
    dew: np.ndarray
    soilt: np.ndarray
    qvg: np.ndarray
    qsg: np.ndarray
    qcg: np.ndarray
    edir1: np.ndarray
    ec1: np.ndarray
    ett1: np.ndarray
    eeta: np.ndarray
    qfx: np.ndarray
    hfx: np.ndarray
    s: np.ndarray
    evapl: np.ndarray
    prcpl: np.ndarray
    fltot: np.ndarray
    runoff1: np.ndarray
    runoff2: np.ndarray
    mavail: np.ndarray
    infiltrp: np.ndarray
    smf: np.ndarray


@dataclass(frozen=True)
class RucSeaIceStep:
    """Snow-free sea-ice column state and fluxes from WRF ``sice``.

    ``soilmois``/``soiliqw``/``soilice``/``smfrkeep``/``keepfr`` are not
    written by ``sice`` itself; ``sfctmp`` forces them to 1/0/1/1/0
    immediately after both call sites and this result carries that forcing.
    """

    tso: np.ndarray
    soilmois: np.ndarray
    soiliqw: np.ndarray
    soilice: np.ndarray
    smfrkeep: np.ndarray
    keepfr: np.ndarray
    dew: np.ndarray
    soilt: np.ndarray
    qvg: np.ndarray
    qsg: np.ndarray
    qcg: np.ndarray
    eeta: np.ndarray
    qfx: np.ndarray
    hfx: np.ndarray
    s: np.ndarray
    evapl: np.ndarray
    prcpl: np.ndarray
    fltot: np.ndarray


RUC_SOIL_PROPERTY_PROFILE_INPUTS = (
    "fwsat",
    "lwsat",
    "tav",
    "keepfr",
    "soilmois",
    "soiliqw",
    "soilice",
    "soilmoism",
    "soiliqwm",
    "soilicem",
)
RUC_SOIL_PROPERTY_COLUMN_INPUTS = (
    "qwrtz",
    "rhocs",
    "dqm",
    "qmin",
    "psis",
    "bclh",
    "ksat",
)
RUC_SOIL_MOISTURE_PROFILE_INPUTS = (
    "diffu",
    "hydro",
    "transp",
    "soilice",
    "soilmois",
    "soiliqw",
)
RUC_SOIL_MOISTURE_COLUMN_INPUTS = (
    "qsg",
    "qvg",
    "qcg",
    "qcatm",
    "qvatm",
    "prcp",
    "qkms",
    "drip",
    "dew",
    "smelt",
    "vegfrac",
    "snowfrac",
    "soilres",
    "dqm",
    "qmin",
    "ref",
    "ksat",
    "ras",
)
RUC_SOIL_TEMPERATURE_PROFILE_INPUTS = (
    "thdif",
    "cap",
    "tso",
)
RUC_SOIL_TEMPERATURE_COLUMN_INPUTS = (
    "prcpms",
    "rainf",
    "patm",
    "tabs",
    "qvatm",
    "qcatm",
    "emiss",
    "rnet",
    "qkms",
    "tkms",
    "pc",
    "rho",
    "vegfrac",
    "lai",
    "drycan",
    "wetcan",
    "transum",
    "dew",
    "mavail",
    "soilres",
    "alfa",
    "dqm",
    "qmin",
    "bclh",
    "soilt",
    "qvg",
    "qsg",
    "qcg",
)
RUC_SOIL_STEP_PROFILE_INPUTS = (
    "soilmois",
    "tso",
    "smfrkeep",
    "keepfr",
)
RUC_SOIL_STEP_COLUMN_INPUTS = (
    "prcpms",
    "rainf",
    "patm",
    "qvatm",
    "qcatm",
    "glw",
    "gsw",
    "gswin",
    "emiss",
    "rnet",
    "qkms",
    "tkms",
    "pc",
    "cst",
    "drip",
    "infwater",
    "rho",
    "vegfrac",
    "lai",
    "qwrtz",
    "rhocs",
    "dqm",
    "qmin",
    "ref",
    "wilt",
    "psis",
    "bclh",
    "ksat",
    "sat",
    "cn",
    "tabs",
    "soilt",
    "qvg",
    "qsg",
    "qcg",
    "mavail",
)
RUC_SEA_ICE_PROFILE_INPUTS = (
    "capice",
    "thdifice",
    "tso",
)
RUC_SEA_ICE_COLUMN_INPUTS = (
    "prcpms",
    "rainf",
    "patm",
    "qvatm",
    "emiss",
    "rnet",
    "qkms",
    "tkms",
    "rho",
    "tabs",
    "soilt",
    "qvg",
    "qsg",
)


def _list_tokens(line: str) -> list[str]:
    """Fortran list-directed tokens, retaining quoted descriptions."""

    return [match.group(0) for match in re.finditer(r"'[^']*'|[^,\s]+", line)]


def _parse_ruc_vegetation_row(line: str) -> RucVegetationRow:
    tokens = _list_tokens(line)
    if len(tokens) < 14:
        raise ValueError(f"truncated RUC VEGPARM row: {line!r}")
    description = tokens[13]
    if not (description.startswith("'") and description.endswith("'")):
        raise ValueError(f"missing RUC VEGPARM description: {line!r}")
    values = [float(token) for token in tokens[1:13]]
    ifor = values[5]
    if not ifor.is_integer():
        raise ValueError(f"non-integral RUC forest class in {line!r}")
    return RucVegetationRow(
        category=int(tokens[0]),
        albedo=values[0],
        z0=values[1],
        lemi=values[2],
        pc=values[3],
        shdfac=values[4],
        ifor=int(ifor),
        rs=values[6],
        rgl=values[7],
        hs=values[8],
        snup=values[9],
        lai=values[10],
        maxalb=values[11],
        description=description[1:-1],
    )


def parse_ruc_vegparm(text: str) -> Mapping[str, RucVegetationTable]:
    """Parse the two RUC sections using WRF's list-directed READ order."""

    lines = text.splitlines()
    wanted = frozenset(RUC_VEGETATION_DATASETS.values())
    tables: dict[str, RucVegetationTable] = {}
    cursor = 0
    while cursor < len(lines):
        if lines[cursor].strip() != "Vegetation Parameters":
            cursor += 1
            continue
        if cursor + 2 >= len(lines):
            raise ValueError("truncated VEGPARM section header")
        name = lines[cursor + 1].strip()
        header = _list_tokens(lines[cursor + 2])
        if not header:
            raise ValueError(f"invalid VEGPARM header for {name!r}")
        count = int(header[0])
        start = cursor + 3
        end = start + count
        if end > len(lines):
            raise ValueError(f"truncated VEGPARM section {name!r}")
        if name not in wanted:
            cursor = end
            continue
        if name in tables:
            raise ValueError(f"duplicate RUC VEGPARM section {name}")
        rows = tuple(_parse_ruc_vegetation_row(line) for line in lines[start:end])
        if tuple(row.category for row in rows) != tuple(range(1, count + 1)):
            raise ValueError(f"nonsequential RUC VEGPARM categories in {name}")
        scalar_cursor = end
        scalars: dict[str, int | float] = {}
        for scalar_name in RUC_VEGETATION_SCALARS:
            if scalar_cursor + 1 >= len(lines):
                raise ValueError(f"truncated RUC VEGPARM scalar block {name}")
            if lines[scalar_cursor].strip() != scalar_name:
                raise ValueError(
                    f"RUC VEGPARM {name} expected {scalar_name}, got "
                    f"{lines[scalar_cursor]!r}"
                )
            token = _list_tokens(lines[scalar_cursor + 1])[0]
            if scalar_name in ("BARE", "NATURAL", "CROP", "URBAN"):
                scalars[scalar_name] = int(token)
            else:
                scalars[scalar_name] = float(token)
            scalar_cursor += 2
        tables[name] = RucVegetationTable(
            name=name,
            rows=rows,
            scalars=MappingProxyType(scalars),
        )
        cursor = scalar_cursor
    if tuple(tables) != ("USGS-RUC", "MODI-RUC"):
        raise ValueError(
            f"RUC VEGPARM inventory {tuple(tables)!r}; expected "
            "('USGS-RUC', 'MODI-RUC')"
        )
    return MappingProxyType(tables)


def _read_canonical_table(root: Path, filename: str) -> tuple[str, dict[str, object]]:
    """Read a legacy CRLF checkout and verify its canonical Git blob."""

    asset = next(
        item for item in RUC_TABLE_ASSETS
        if Path(item.relative_path).name == filename
    )
    path = root / filename
    payload = path.read_bytes()
    if b"\r" in payload.replace(b"\r\n", b""):
        raise ValueError(f"RUC table {path} contains a non-CRLF carriage return")
    canonical = payload.replace(b"\r\n", b"\n")
    digest = hashlib.sha256(canonical).hexdigest()
    if len(canonical) != asset.bytes or digest != asset.sha256:
        raise ValueError(
            f"RUC table {path} canonical identity is "
            f"{len(canonical)} bytes/{digest}; expected "
            f"{asset.bytes}/{asset.sha256}"
        )
    try:
        text = canonical.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"RUC table {path} is not ASCII") from exc
    return text, {
        "path": str(path.resolve()),
        "canonical_bytes": len(canonical),
        "canonical_sha256": digest,
        "checkout_line_endings": "CRLF" if b"\r\n" in payload else "LF",
    }


def load_ruc_parameters(root: str | Path | None = None) -> RucParameterBundle:
    """Load byte-validated RUC vegetation, soil, and general parameters."""

    table_root = Path(root) if root is not None else RUC_TABLE_DIR
    texts: dict[str, str] = {}
    receipts: dict[str, object] = {}
    for filename in ("VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        text, receipt = _read_canonical_table(table_root, filename)
        texts[filename] = text
        receipts[filename] = receipt
    soil_tables = parse_soilparm(texts["SOILPARM.TBL"])
    return RucParameterBundle(
        vegetation=parse_ruc_vegparm(texts["VEGPARM.TBL"]),
        soil=soil_tables["STAS-RUC"],
        general=parse_genparm(texts["GENPARM.TBL"]),
        receipt=MappingProxyType({
            "schema": "gpuwm.ruc-parameter-bundle/v1",
            "status": "PASS",
            "assets": MappingProxyType(receipts),
        }),
    )


def ruc_soil_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return WRF's nine RUC level depths and derived ``DZS`` values."""

    zs = np.asarray(RUC_SOIL_LEVELS_M, dtype=np.float32)
    zs2 = np.empty(NUM_SOIL_LAYERS, dtype=np.float32)
    dzs = np.empty(NUM_SOIL_LAYERS, dtype=np.float32)
    half = np.float32(0.5)
    zs2[0] = np.float32(0.0)
    zs2[1] = np.float32((zs[1] + zs[0]) * half)
    dzs[0] = np.float32(zs2[1] - zs2[0])
    for level in range(1, NUM_SOIL_LAYERS - 1):
        # This intentionally overwrites ZS2(2), exactly as WRF does.
        zs2[level] = np.float32((zs[level + 1] + zs[level]) * half)
        dzs[level] = np.float32(zs2[level] - zs2[level - 1])
    zs2[-1] = zs[-1]
    dzs[-1] = np.float32(zs2[-1] - zs2[-2])
    return zs, dzs


def _horizontal_integer_field(
    value, shape: tuple[int, ...], name: str, *, arrays=None
) -> np.ndarray:
    np = arrays if arrays is not None else _NUMPY
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{name} shape {raw.shape}; expected {shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integer WRF categories")
    return raw.astype(np.int32, copy=False)


def _horizontal_float_field(
    value, shape: tuple[int, ...], name: str, *, arrays=None
) -> np.ndarray:
    np = arrays if arrays is not None else _NUMPY
    raw = np.asarray(value, dtype=np.float32)
    if raw.shape != shape:
        try:
            raw = np.broadcast_to(raw, shape)
        except ValueError as exc:
            raise ValueError(
                f"{name} shape {raw.shape} is not broadcastable to {shape}"
            ) from exc
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be finite")
    return raw


def _root_count_field(value, shape: tuple[int, ...], *, arrays=None) -> np.ndarray:
    np = arrays if arrays is not None else _NUMPY
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("nroot must contain integer root-zone level counts")
    if raw.shape != shape:
        try:
            raw = np.broadcast_to(raw, shape)
        except ValueError as exc:
            raise ValueError(
                f"nroot shape {raw.shape} is not broadcastable to {shape}"
            ) from exc
    roots = raw.astype(np.int32, copy=False)
    if np.any((roots < 1) | (roots >= NUM_SOIL_LAYERS)):
        bad = int(roots[(roots < 1) | (roots >= NUM_SOIL_LAYERS)][0])
        raise ValueError(f"RUC nroot {bad} is outside 1..8")
    return roots


def ruc_surface_parameters(
    isltyp,
    ivgtyp,
    shdmin,
    shdmax,
    vegfrac,
    znt,
    lai,
    *,
    rdlai2d: bool = False,
    iswater: int | None = None,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    mosaic_lu: int = 0,
    mosaic_soil: int = 0,
    parameters: RucParameterBundle | None = None,
    arrays=None,
) -> RucSurfaceParameters:
    """Transcribe WRF ``soilvegin`` for the first dominant-category lane.

    All arithmetic follows WRF's default-real (float32) evaluation order.
    The two mosaic modes remain fail-closed because they are outside the
    pinned first RUC option lane.
    """

    np = arrays if arrays is not None else _NUMPY
    if type(mosaic_lu) is not int or mosaic_lu != 0:
        raise ValueError("RUC surface setup currently requires mosaic_lu=0")
    if type(mosaic_soil) is not int or mosaic_soil != 0:
        raise ValueError("RUC surface setup currently requires mosaic_soil=0")
    if type(rdlai2d) is not bool:
        raise TypeError("rdlai2d must be bool")

    soil_raw = np.asarray(isltyp)
    if soil_raw.ndim < 1:
        raise ValueError("RUC surface fields must have at least one dimension")
    shape = soil_raw.shape
    soil_type = _horizontal_integer_field(
        soil_raw, shape, "isltyp", arrays=arrays)
    vegetation_type = _horizontal_integer_field(
        ivgtyp, shape, "ivgtyp", arrays=arrays)
    minimum_green = _horizontal_float_field(
        shdmin, shape, "shdmin", arrays=arrays)
    maximum_green = _horizontal_float_field(
        shdmax, shape, "shdmax", arrays=arrays)
    green_fraction = _horizontal_float_field(
        vegfrac, shape, "vegfrac", arrays=arrays)
    incoming_roughness = _horizontal_float_field(
        znt, shape, "znt", arrays=arrays)
    incoming_lai = _horizontal_float_field(lai, shape, "lai", arrays=arrays)

    bundle = parameters or load_ruc_parameters()
    vegetation = bundle.vegetation_for(mminlu)
    if np.any((soil_type < 1) | (soil_type > len(bundle.soil.rows))):
        bad = int(soil_type[(soil_type < 1) | (soil_type > len(bundle.soil.rows))][0])
        raise ValueError(f"RUC isltyp {bad} is outside 1..{len(bundle.soil.rows)}")
    if np.any((vegetation_type < 1) | (vegetation_type > len(vegetation.rows))):
        bad = int(
            vegetation_type[
                (vegetation_type < 1) | (vegetation_type > len(vegetation.rows))
            ][0]
        )
        raise ValueError(
            f"RUC ivgtyp {bad} is outside 1..{len(vegetation.rows)} "
            f"for {vegetation.name}"
        )
    if iswater is None:
        water_category = 16 if vegetation.name == "USGS-RUC" else 17
    elif type(iswater) is int and 1 <= iswater <= len(vegetation.rows):
        water_category = iswater
    else:
        raise ValueError(
            f"RUC iswater {iswater!r} is outside 1..{len(vegetation.rows)}"
        )

    float_names = (
        "emiss", "pc", "qwrtz", "rhocs", "bclh", "dqm", "ksat",
        "psis", "qmin", "ref", "wilt",
    )
    output = {
        name: np.zeros(shape, dtype=np.float32) for name in float_names
    }
    roughness = np.array(incoming_roughness, dtype=np.float32, copy=True)
    leaf_area = np.array(incoming_lai, dtype=np.float32, copy=True)
    forest = np.empty(shape, dtype=np.int32)

    soil_flat = soil_type.reshape(-1)
    vegetation_flat = vegetation_type.reshape(-1)
    shdmin_flat = minimum_green.reshape(-1)
    shdmax_flat = maximum_green.reshape(-1)
    vegfrac_flat = green_fraction.reshape(-1)
    roughness_flat = roughness.reshape(-1)
    leaf_flat = leaf_area.reshape(-1)
    forest_flat = forest.reshape(-1)
    output_flat = {name: values.reshape(-1) for name, values in output.items()}

    zero = np.float32(0.0)
    one = np.float32(1.0)

    # Vectorised over the horizontal.  ``soilvegin`` is a per-column table
    # lookup and five float32 operations, so this used to be a
    # ``for column in range(ncolumn)`` loop -- 0.0069 ms per land column of
    # pure Python, which is 32% of what a RUC land-surface call still cost
    # once the leaf arithmetic had moved to the card.  Every operation below
    # keeps WRF's default-real evaluation order and operand pairing, so the
    # answer is bit-identical to the loop; that identity is the gate in
    # ``tests/test_ruc_batching.py`` and the ``soilvegin`` oracle, not an
    # assumption.
    veg_index = vegetation_flat.astype(np.intp) - 1
    forest_class = np.asarray(
        [row.ifor for row in vegetation.rows], dtype=np.int32)[veg_index]
    forest_flat[:] = forest_class
    table_lai = np.asarray(
        [row.lai for row in vegetation.rows], dtype=np.float32)[veg_index]
    table_z0 = np.asarray(
        [row.z0 for row in vegetation.rows], dtype=np.float32)[veg_index]

    green_range = (shdmax_flat - shdmin_flat).astype(np.float32)
    numerator = (vegfrac_flat - shdmin_flat).astype(np.float32)
    denominator = np.maximum(one, green_range).astype(np.float32)
    ratio = (numerator / denominator).astype(np.float32)
    bounded = np.maximum(zero, np.minimum(one, ratio)).astype(np.float32)
    factor = np.where(
        green_range < one, one, (one - bounded).astype(np.float32)
    ).astype(np.float32)

    scaled_lai = (np.float32(0.8) * table_lai).astype(np.float32)
    delta_lai = np.zeros(forest_class.shape, dtype=np.float32)
    for klass, cap in ((1, 0.2), (2, 0.5), (3, 0.45), (4, 0.75),
                       (5, 0.86), (7, 0.5)):
        selected = forest_class == klass
        delta_lai[selected] = np.minimum(
            np.float32(cap), scaled_lai[selected]).astype(np.float32)

    water = vegetation_flat == water_category
    land = ~water
    if not rdlai2d:
        leaf_flat[water] = table_lai[water]
        leaf_flat[land] = (
            table_lai[land]
            - (delta_lai[land] * factor[land]).astype(np.float32)
        ).astype(np.float32)
    roughness_flat[land] = np.where(
        forest_class[land] == 7,
        (table_z0[land]
         - (np.float32(0.125) * factor[land]).astype(np.float32)
         ).astype(np.float32),
        table_z0[land],
    ).astype(np.float32)

    output_flat["emiss"][:] = np.asarray(
        [row.lemi for row in vegetation.rows], dtype=np.float32)[veg_index]
    output_flat["pc"][:] = np.asarray(
        [row.pc for row in vegetation.rows], dtype=np.float32)[veg_index]

    # ``isltyp == 14`` is WRF's water soil class: :5652 cycles the column and
    # leaves every soil property at the zero this routine initialised.
    solid = soil_flat != 14
    soil_table = np.asarray(
        [row.values for row in bundle.soil.rows], dtype=np.float32)
    selected = soil_flat.astype(np.intp)[solid] - 1
    drysmc = soil_table[selected, 1]
    output_flat["rhocs"][solid] = (
        soil_table[selected, 2] * np.float32(1.0e6)).astype(np.float32)
    output_flat["bclh"][solid] = soil_table[selected, 0]
    output_flat["dqm"][solid] = (
        soil_table[selected, 3] - drysmc).astype(np.float32)
    output_flat["ksat"][solid] = soil_table[selected, 6]
    output_flat["psis"][solid] = (-soil_table[selected, 5]).astype(np.float32)
    output_flat["qmin"][solid] = drysmc
    output_flat["ref"][solid] = soil_table[selected, 4]
    output_flat["wilt"][solid] = soil_table[selected, 8]
    output_flat["qwrtz"][solid] = soil_table[selected, 9]

    return RucSurfaceParameters(
        iforest=forest,
        emiss=output["emiss"],
        pc=output["pc"],
        znt=roughness,
        lai=leaf_area,
        qwrtz=output["qwrtz"],
        rhocs=output["rhocs"],
        bclh=output["bclh"],
        dqm=output["dqm"],
        ksat=output["ksat"],
        psis=output["psis"],
        qmin=output["qmin"],
        ref=output["ref"],
        wilt=output["wilt"],
    )


def ruc_soil_properties(
    values: Mapping[str, object],
    *,
    riw: float = 0.9,
) -> RucSoilProperties:
    """Transcribe deterministic WRF ``soilprop`` in float32.

    Profile arrays use gpuwm soil-first order
    ``(9, ...horizontal...)``.  WRF initializes all four output profiles to
    zero immediately before this call; ``soilprop`` deliberately leaves the
    deepest ``thdif``, ``diffu``, and ``cap`` entries at that initialized
    value while computing hydraulic conductivity at all nine levels.
    """

    ice_water_ratio = np.float32(riw)
    if not np.isfinite(ice_water_ratio) or ice_water_ratio <= 0.0:
        raise ValueError("RUC soilprop riw must be finite and positive")
    required = RUC_SOIL_PROPERTY_PROFILE_INPUTS + RUC_SOIL_PROPERTY_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC soil-property inputs: {', '.join(missing)}")
    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SOIL_PROPERTY_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC soil-property profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SOIL_PROPERTY_COLUMN_INPUTS
    }
    if np.any(columns["bclh"] <= np.float32(0.0)):
        raise ValueError("RUC bclh must be positive")
    if np.any(columns["psis"] >= np.float32(0.0)):
        raise ValueError("RUC psis must be negative")
    if np.any(columns["ksat"] < np.float32(0.0)):
        raise ValueError("RUC ksat must be nonnegative")

    outputs = {
        name: np.zeros(shape, dtype=np.float32)
        for name in ("thdif", "diffu", "hydro", "cap")
    }
    ncolumn = int(np.prod(horizontal_shape))
    profile_flat = {
        name: array.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, array in profiles.items()
    }
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    output_flat = {
        name: array.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, array in outputs.items()
    }

    zero = np.float32(0.0)
    one = np.float32(1.0)
    minimum = np.float32(1.0e-8)
    riw = ice_water_ratio
    xlmelt = np.float32(3.35e5)
    heat_air = np.float32(1004.5)
    gravity = np.float32(9.81)
    heat_water = np.float32(4.183e6)
    heat_ice = np.float32(1.89e6)
    conductivity_quartz = np.float32(7.7)
    conductivity_ice = np.float32(2.2)
    conductivity_water = np.float32(0.57)

    for column in range(ncolumn):
        qwrtz = column_flat["qwrtz"][column]
        rhocs = column_flat["rhocs"][column]
        dqm = column_flat["dqm"][column]
        qmin = column_flat["qmin"][column]
        psis = column_flat["psis"][column]
        bclh = column_flat["bclh"][column]
        ksat = column_flat["ksat"][column]
        ws = np.float32(dqm + qmin)
        x1 = np.float32(xlmelt / np.float32(gravity * psis))
        x2 = np.float32(np.float32(x1 / bclh) * ws)
        x4 = np.float32(np.float32(bclh + one) / bclh)
        gamd = np.float32(np.float32(one - ws) * np.float32(2700.0))
        kdry = np.float32(
            np.float32(
                np.float32(np.float32(0.135) * gamd) + np.float32(64.7)
            )
            / np.float32(
                np.float32(2700.0) - np.float32(np.float32(0.947) * gamd)
            )
        )
        mineral = np.float32(2.0 if qwrtz > np.float32(0.2) else 3.0)
        # Every transcendental in soilprop rounds a float64 evaluation once.
        # See _RUC_PROVISIONAL_TRANSCENDENTALS: this is NOT glibc, it is the
        # closest of the available float32 kernels and the only one that is
        # identical on the host and the device.  numpy's own float32 kernels
        # drift between releases (1-3 ULP of thdif/diffu/hydro between 2.2 and
        # 2.4), which propagated into every RUC lane.
        kas = np.float32(
            np.float32(
                np.power(np.float64(conductivity_quartz), np.float64(qwrtz))
            )
            * np.float32(
                np.power(np.float64(mineral), np.float64(np.float32(one - qwrtz)))
            )
        )

        for level in range(NUM_SOIL_LAYERS - 1):
            tav = profile_flat["tav"][level, column]
            tn = np.float32(tav - np.float32(273.15))
            soilicem = profile_flat["soilicem"][level, column]
            wd = np.float32(ws - np.float32(riw * soilicem))
            liquid_middle = profile_flat["soiliqwm"][level, column]
            first_ratio = np.float32(
                wd / np.float32(liquid_middle + qmin)
            )
            psif = np.float32(np.float32(psis * np.float32(100.0)) * np.float32(
                np.power(np.float64(first_ratio), np.float64(bclh))
            ))
            psif = np.float32(
                psif
                * np.float32(
                    np.power(
                        np.float64(np.float32(ws / wd)), np.float64(3.0)
                    )
                )
            )
            pf = np.float32(np.log10(np.float64(np.float32(abs(psif)))))
            fact = np.float32(one + np.float32(riw * soilicem))
            if pf <= np.float32(5.2):
                _ = np.float32(
                    np.float32(420.0)
                    * np.float32(np.exp(np.float32(-(pf + np.float32(2.7)))))
                    * fact
                )
            else:
                _ = np.float32(np.float32(0.1744) * fact)

            detal = zero
            if soilicem != zero and tn < zero:
                detal = np.float32(
                    np.float32(
                        np.float32(np.float32(273.15) * x2)
                        / np.float32(tav * tav)
                    )
                    * np.float32(np.power(
                        np.float64(np.float32(tav / np.float32(x1 * tn))),
                        np.float64(x4),
                    ))
                )
                if profile_flat["keepfr"][level, column] == one:
                    detal = zero

            kasat = np.float32(
                np.power(np.float64(kas), np.float64(np.float32(one - ws)))
            )
            kasat = np.float32(
                kasat * np.float32(np.power(
                    np.float64(conductivity_ice),
                    np.float64(profile_flat["fwsat"][level, column]),
                ))
            )
            kasat = np.float32(
                kasat * np.float32(np.power(
                    np.float64(conductivity_water),
                    np.float64(profile_flat["lwsat"][level, column]),
                ))
            )
            soilmoism = profile_flat["soilmoism"][level, column]
            x5 = np.float32(np.float32(soilmoism + qmin) / ws)
            if soilicem == zero:
                sr = np.float32(max(np.float32(0.101), x5))
                ke = np.float32(
                    np.float32(np.log10(np.float64(sr))) + one
                )
            else:
                ke = x5
            kjpl = np.float32(
                np.float32(ke * np.float32(kasat - kdry)) + kdry
            )

            cap = np.float32(np.float32(one - ws) * rhocs)
            cap = np.float32(
                cap + np.float32(np.float32(liquid_middle + qmin) * heat_water)
            )
            cap = np.float32(cap + np.float32(soilicem * heat_ice))
            cap = np.float32(
                cap
                + np.float32(
                    np.float32(np.float32(dqm - soilmoism) * heat_air)
                    * np.float32(1.2)
                )
            )
            cap = np.float32(
                cap
                - np.float32(
                    np.float32(detal * np.float32(1.0e3)) * xlmelt
                )
            )
            output_flat["cap"][level, column] = cap

            ice = np.float32(riw * soilicem)
            if np.float32(ws - ice) < np.float32(0.12):
                diffu = zero
            else:
                h = np.float32(max(
                    zero,
                    np.float32(soilmoism + qmin - ice)
                    / np.float32(max(minimum, np.float32(ws - ice))),
                ))
                facd = one
                if ice != zero:
                    facd = np.float32(
                        one - ice / np.float32(max(minimum, soilmoism))
                    )
                ame = np.float32(max(minimum, np.float32(ws - ice)))
                diffu = np.float32(np.float32(-bclh * ksat) * psis)
                diffu = np.float32(diffu / ame)
                diffu = np.float32(
                    diffu * np.float32(np.power(
                        np.float64(np.float32(ws / ame)), np.float64(3.0)
                    ))
                )
                diffu = np.float32(
                    diffu * np.float32(np.power(
                        np.float64(h),
                        np.float64(np.float32(bclh + np.float32(2.0))),
                    ))
                )
                diffu = np.float32(diffu * facd)
            output_flat["diffu"][level, column] = diffu
            output_flat["thdif"][level, column] = np.float32(kjpl / cap)

        for level in range(NUM_SOIL_LAYERS):
            soilice = profile_flat["soilice"][level, column]
            ice = np.float32(riw * soilice)
            if np.float32(ws - ice) < np.float32(0.12):
                hydro = zero
            else:
                fach = one
                if soilice != zero:
                    fach = np.float32(
                        one
                        - ice
                        / np.float32(max(
                            minimum, profile_flat["soilmois"][level, column]
                        ))
                    )
                am = np.float32(max(minimum, np.float32(ws - ice)))
                hydro = np.float32(ksat / am)
                hydro = np.float32(
                    hydro * np.float32(np.power(
                        np.float64(np.float32(
                            profile_flat["soiliqw"][level, column] / am
                        )),
                        np.float64(np.float32(
                            np.float32(2.0) * bclh + np.float32(2.0)
                        )),
                    ))
                )
                hydro = np.float32(hydro * fach)
                hydro = np.float32(min(ksat, hydro))
                if hydro < np.float32(1.0e-10):
                    hydro = zero
            output_flat["hydro"][level, column] = hydro

    for name, array in outputs.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"RUC soilprop produced non-finite {name}")
    return RucSoilProperties(**outputs)


def ruc_transpiration(
    soiliqw,
    tabs,
    lai,
    gswin,
    dqm,
    qmin,
    ref,
    wilt,
    pc,
    iland,
    *,
    nroot: object = 4,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    parameters: RucParameterBundle | None = None,
) -> RucTranspiration:
    """Transcribe WRF ``transf`` for per-column 1..8-level root zones."""
    liquid = np.asarray(soiliqw, dtype=np.float32)
    if liquid.ndim < 2 or liquid.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC soiliqw must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {liquid.shape}"
        )
    if not np.all(np.isfinite(liquid)):
        raise ValueError("soiliqw must be finite")
    horizontal_shape = liquid.shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    column_values = {
        name: _horizontal_float_field(value, horizontal_shape, name)
        for name, value in (
            ("tabs", tabs),
            ("lai", lai),
            ("gswin", gswin),
            ("dqm", dqm),
            ("qmin", qmin),
            ("ref", ref),
            ("wilt", wilt),
            ("pc", pc),
        )
    }
    land_type = _horizontal_integer_field(iland, horizontal_shape, "iland")
    bundle = parameters or load_ruc_parameters()
    vegetation = bundle.vegetation_for(mminlu)
    if np.any((land_type < 1) | (land_type > len(vegetation.rows))):
        bad = int(land_type[(land_type < 1) | (land_type > len(vegetation.rows))][0])
        raise ValueError(
            f"RUC iland {bad} is outside 1..{len(vegetation.rows)} "
            f"for {vegetation.name}"
        )
    if np.any(column_values["ref"] <= column_values["wilt"]):
        raise ValueError("RUC ref must exceed wilt")

    zs, _ = ruc_soil_geometry()
    zshalf = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
    half = np.float32(0.5)
    for level in range(1, NUM_SOIL_LAYERS):
        zshalf[level] = np.float32(
            np.float32(zs[level - 1] + zs[level]) * half
        )

    ncolumn = int(np.prod(horizontal_shape))
    liquid_flat = liquid.reshape(NUM_SOIL_LAYERS, ncolumn)
    columns = {
        name: value.reshape(ncolumn) for name, value in column_values.items()
    }
    land_flat = land_type.reshape(ncolumn)
    roots_flat = roots.reshape(ncolumn)
    weights = np.zeros_like(liquid)
    totals = np.zeros(horizontal_shape, dtype=np.float32)
    weights_flat = weights.reshape(NUM_SOIL_LAYERS, ncolumn)
    totals_flat = totals.reshape(ncolumn)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    rsmax = np.float32(vegetation.scalars["RSMAX_DATA"])

    for column in range(ncolumn):
        root_count = int(roots_flat[column])
        qmin_value = columns["qmin"][column]
        reference = columns["ref"][column]
        wilting = columns["wilt"][column]
        for level in range(root_count):
            total_liquid = np.float32(
                liquid_flat[level, column] + qmin_value
            )
            if level == 0:
                depth = zshalf[1]
                saturated = total_liquid > reference
            else:
                depth = np.float32(zshalf[level + 1] - zshalf[level])
                saturated = total_liquid >= reference
            if saturated:
                weight = depth
            elif total_liquid <= wilting:
                weight = zero
            else:
                weight = np.float32(
                    np.float32(
                        np.float32(total_liquid - wilting)
                        / np.float32(reference - wilting)
                    )
                    * depth
                )
            weights_flat[level, column] = weight

        leaf_area = columns["lai"][column]
        pctot = np.float32(0.8) if leaf_area > np.float32(4.0) \
            else columns["pc"][column]
        temperature = columns["tabs"][column]
        if temperature <= np.float32(302.15):
            exponent = np.float32(
                np.float32(-0.41)
                * np.float32(temperature - np.float32(282.05))
            )
        else:
            exponent = np.float32(
                np.float32(0.5)
                * np.float32(temperature - np.float32(314.0))
            )
        ftem = np.float32(
            one
            / np.float32(one + np.float32(np.exp(np.float64(exponent))))
        )

        veg = vegetation.rows[int(land_flat[column]) - 1]
        resistance = np.float32(veg.rs)
        light_threshold = np.float32(veg.rgl)
        cmin = np.float32(one / rsmax)
        cmax = np.float32(one / resistance)
        if leaf_area > one:
            cmax = np.float32(leaf_area / resistance)
        incoming_solar = columns["gswin"][column]
        if incoming_solar < light_threshold:
            light_exponent = np.float32(
                np.float32(-0.034)
                * np.float32(incoming_solar - np.float32(3.5))
            )
            fsol = np.float32(
                one
                / np.float32(
                    one + np.float32(np.exp(np.float64(light_exponent)))
                )
            )
        else:
            fsol = one
        conductance = np.float32(cmax - cmin)
        conductance = np.float32(conductance * pctot)
        conductance = np.float32(conductance * ftem)
        conductance = np.float32(conductance * fsol)
        conductance = np.float32(cmin + conductance)
        conductance = np.float32(conductance / cmax)

        total = zero
        for level in range(root_count):
            weight = np.float32(
                max(
                    cmin,
                    np.float32(weights_flat[level, column] * conductance),
                )
            )
            weights_flat[level, column] = weight
            total = np.float32(total + weight)
        totals_flat[column] = total

    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(totals)):
        raise ValueError("RUC transf produced non-finite output")
    return RucTranspiration(tranf=weights, transum=totals)


def _ruc_soil_solver_geometry(
    delt: np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct WRF RUC ``zsmain/zshalf/dtdzs/dtdzs2`` in float32."""

    zsmain, _ = ruc_soil_geometry()
    zshalf = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
    for level in range(1, NUM_SOIL_LAYERS):
        zshalf[level] = np.float32(
            np.float32(zsmain[level - 1] + zsmain[level])
            * np.float32(0.5)
        )
    dtdzs = np.zeros(2 * (NUM_SOIL_LAYERS - 2), dtype=np.float32)
    dtdzs2 = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
    for fortran_level in range(2, NUM_SOIL_LAYERS):
        first = 2 * fortran_level - 3
        second = first + 1
        level = fortran_level - 1
        x = np.float32(
            np.float32(np.float32(delt / np.float32(2.0)))
            / np.float32(zshalf[level + 1] - zshalf[level])
        )
        dtdzs[first - 1] = np.float32(
            x / np.float32(zsmain[level] - zsmain[level - 1])
        )
        dtdzs2[level - 1] = x
        dtdzs[second - 1] = np.float32(
            x / np.float32(zsmain[level + 1] - zsmain[level])
        )
    return zsmain, zshalf, dtdzs, dtdzs2


def ruc_soil_moisture_step(
    values: Mapping[str, object],
    *,
    delt: float,
) -> RucSoilMoisture:
    """Transcribe the nine-level WRF ``soilmoist`` implicit water solve."""

    timestep = np.float32(delt)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC soilmoist delt must be finite and positive")
    required = RUC_SOIL_MOISTURE_PROFILE_INPUTS + RUC_SOIL_MOISTURE_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC soilmoist inputs: {', '.join(missing)}")

    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SOIL_MOISTURE_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC soilmoist profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SOIL_MOISTURE_COLUMN_INPUTS
    }
    if np.any(columns["dqm"] <= np.float32(0.0)):
        raise ValueError("RUC soilmoist dqm must be positive")
    if np.any(columns["ref"] <= columns["qmin"]):
        raise ValueError("RUC soilmoist ref must exceed qmin")
    if np.any(columns["ksat"] < np.float32(0.0)):
        raise ValueError("RUC soilmoist ksat must be nonnegative")

    zsmain, zshalf, dtdzs, dtdzs2 = _ruc_soil_solver_geometry(timestep)
    ncolumn = int(np.prod(horizontal_shape))
    profile_flat = {
        name: array.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, array in profiles.items()
    }
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    moisture = np.array(profiles["soilmois"], dtype=np.float32, copy=True)
    liquid = np.array(profiles["soiliqw"], dtype=np.float32, copy=True)
    moisture_flat = moisture.reshape(NUM_SOIL_LAYERS, ncolumn)
    liquid_flat = liquid.reshape(NUM_SOIL_LAYERS, ncolumn)
    horizontal_outputs = {
        name: np.empty(horizontal_shape, dtype=np.float32)
        for name in ("mavail", "runoff", "runoff2", "infiltrp", "infmax")
    }
    output_flat = {
        name: array.reshape(ncolumn) for name, array in horizontal_outputs.items()
    }
    zero = np.float32(0.0)
    one = np.float32(1.0)
    minimum = np.float32(1.0e-8)

    for column in range(ncolumn):
        cosmc = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhsmc = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        # WRF evaluates several experimental lower-boundary forms and then
        # intentionally overwrites them with this zero-gradient condition.
        cosmc[0] = zero
        rhsmc[0] = moisture_flat[-1, column]
        for step in range(1, NUM_SOIL_LAYERS - 1):
            kn = NUM_SOIL_LAYERS - step
            first = 2 * kn - 3
            x4 = np.float32(
                np.float32(np.float32(2.0) * dtdzs[first - 1])
                * profile_flat["diffu"][kn - 2, column]
            )
            x2 = np.float32(
                np.float32(np.float32(2.0) * dtdzs[first])
                * profile_flat["diffu"][kn - 1, column]
            )
            q4 = np.float32(
                x4
                + np.float32(
                    profile_flat["hydro"][kn - 2, column]
                    * dtdzs2[kn - 2]
                )
            )
            q2 = np.float32(
                x2
                - np.float32(
                    profile_flat["hydro"][kn, column]
                    * dtdzs2[kn - 2]
                )
            )
            denominator = np.float32(one + x2)
            denominator = np.float32(denominator + x4)
            denominator = np.float32(
                denominator - np.float32(q2 * cosmc[step - 1])
            )
            cosmc[step] = np.float32(q4 / denominator)
            rhs = np.float32(
                moisture_flat[kn - 1, column]
                + np.float32(q2 * rhsmc[step - 1])
            )
            root_flux = np.float32(
                profile_flat["transp"][kn - 1, column]
                / np.float32(zshalf[kn] - zshalf[kn - 1])
            )
            rhs = np.float32(rhs + np.float32(root_flux * timestep))
            rhsmc[step] = np.float32(rhs / denominator)

        vegfrac = column_flat["vegfrac"][column]
        soilres = column_flat["soilres"][column]
        umveg = np.float32(np.float32(one - vegfrac) * soilres)
        runoff = zero
        runoff2 = zero
        dzs = zsmain[1]
        r1 = cosmc[NUM_SOIL_LAYERS - 2]
        r2 = rhsmc[NUM_SOIL_LAYERS - 2]
        r3 = np.float32(profile_flat["diffu"][0, column] / dzs)
        r4 = np.float32(
            r3 + np.float32(profile_flat["hydro"][0, column] * np.float32(0.5))
        )
        r5 = np.float32(
            r3 - np.float32(profile_flat["hydro"][1, column] * np.float32(0.5))
        )
        r6 = np.float32(
            column_flat["qkms"][column] * column_flat["ras"][column]
        )
        total_liquid = column_flat["prcp"][column]
        total_liquid = np.float32(
            total_liquid - np.float32(column_flat["drip"][column] / timestep)
        )
        total_liquid = np.float32(
            total_liquid
            - np.float32(
                np.float32(umveg * column_flat["dew"][column])
                * column_flat["ras"][column]
            )
        )
        total_liquid = np.float32(
            total_liquid - column_flat["smelt"][column]
        )
        flux = total_liquid

        dqm = column_flat["dqm"][column]
        qmin = column_flat["qmin"][column]
        reference = column_flat["ref"][column]
        ksat = column_flat["ksat"][column]
        delt1 = np.float32(timestep / np.float32(86400.0))
        f1max = np.float32(dqm * zshalf[1])
        free_storage = np.float32(
            f1max
            * np.float32(one - np.float32(moisture_flat[0, column] / dqm))
        )
        frozen_depth = np.float32(
            profile_flat["soilice"][0, column] * zshalf[1]
        )
        for level in range(1, NUM_SOIL_LAYERS - 1):
            thickness = np.float32(zshalf[level + 1] - zshalf[level])
            frozen_depth = np.float32(
                frozen_depth
                + np.float32(profile_flat["soilice"][level, column] * thickness)
            )
            maximum = np.float32(dqm * thickness)
            available = np.float32(
                maximum
                * np.float32(one - np.float32(moisture_flat[level, column] / dqm))
            )
            free_storage = np.float32(free_storage + available)
        kdt = np.float32(
            np.float32(np.float32(3.0) * ksat) / np.float32(3.4341e-6)
        )
        # This 1 - exp(x) with x near zero amplifies a single-ULP exp error by
        # 1/|x| ~ 230, which showed as 112 ULP of runoff1.  Rounding a float64
        # exp once agrees with gfortran here and on every argument the pinned
        # fixtures reach; see _RUC_PROVISIONAL_TRANSCENDENTALS for what it does
        # not guarantee.
        infiltration_fraction = np.float32(
            one
            - np.float32(np.exp(np.float64(np.float32(-kdt * delt1))))
        )
        ddt = np.float32(free_storage * infiltration_fraction)
        water_input = np.float32(np.float32(-total_liquid) * timestep)
        if water_input < zero:
            water_input = zero
        if water_input > zero:
            infmax1 = np.float32(
                np.float32(
                    water_input
                    * np.float32(ddt / np.float32(water_input + ddt))
                )
                / timestep
            )
        else:
            infmax1 = zero

        frzx = np.float32(
            np.float32(0.15)
            * np.float32(np.float32(dqm + qmin) / reference)
        )
        frzx = np.float32(
            frzx * np.float32(np.float32(0.412) / np.float32(0.468))
        )
        frozen_factor = one
        if frozen_depth > np.float32(1.0e-2):
            acrt = np.float32(
                np.float32(np.float32(3.0) * frzx) / frozen_depth
            )
            series = one
            series = np.float32(
                series
                + np.float32(
                    np.float32(
                        np.power(np.float64(acrt), np.float64(2.0))
                    )
                    / np.float32(2.0)
                )
            )
            series = np.float32(series + acrt)
            frozen_factor = np.float32(
                one
                - np.float32(
                    np.float32(np.exp(np.float64(-acrt))) * series
                )
            )
        infmax1 = np.float32(infmax1 * frozen_factor)
        infmax = np.float32(max(
            infmax1,
            np.float32(
                profile_flat["hydro"][0, column]
                * moisture_flat[0, column]
            ),
        ))
        infmax = np.float32(min(infmax, np.float32(-total_liquid)))
        if np.float32(-total_liquid) > infmax:
            runoff = np.float32(np.float32(-total_liquid) - infmax)
            flux = np.float32(-infmax)
        infiltrp = flux

        r7 = np.float32(
            np.float32(np.float32(0.5) * dzs) / timestep
        )
        r4 = np.float32(r4 + r7)
        flux = np.float32(
            flux - np.float32(moisture_flat[0, column] * r7)
        )
        r8 = np.float32(umveg * r6)
        r8 = np.float32(
            r8 * np.float32(one - column_flat["snowfrac"][column])
        )
        qtot = np.float32(
            column_flat["qvatm"][column] + column_flat["qcatm"][column]
        )
        r9 = profile_flat["transp"][0, column]
        r10 = np.float32(qtot - column_flat["qsg"][column])
        if r10 <= zero:
            denominator = np.float32(r4 - np.float32(r5 * r1))
            denominator = np.float32(
                denominator
                - np.float32(
                    np.float32(r10 * r8)
                    / np.float32(reference - qmin)
                )
            )
            numerator = np.float32(np.float32(r5 * r2) - flux)
            numerator = np.float32(numerator + r9)
            candidate = np.float32(numerator / denominator)
            saturated_flux = np.float32(np.float32(-dqm) * denominator)
            saturated_flux = np.float32(
                saturated_flux + np.float32(r5 * r2)
            )
            saturated_flux = np.float32(saturated_flux + r9)
        else:
            denominator = np.float32(r4 - np.float32(r1 * r5))
            humidity = np.float32(qtot - column_flat["qcg"][column])
            humidity = np.float32(humidity - column_flat["qvg"][column])
            numerator = np.float32(np.float32(r2 * r5) - flux)
            numerator = np.float32(numerator + np.float32(r8 * humidity))
            numerator = np.float32(numerator + r9)
            candidate = np.float32(numerator / denominator)
            saturated_flux = np.float32(
                np.float32(-dqm) * denominator
            )
            saturated_flux = np.float32(
                saturated_flux + np.float32(r2 * r5)
            )
            saturated_flux = np.float32(
                saturated_flux + np.float32(r8 * humidity)
            )
            saturated_flux = np.float32(saturated_flux + r9)
        if candidate < zero:
            moisture_flat[0, column] = minimum
        elif candidate > dqm:
            moisture_flat[0, column] = dqm
            runoff = np.float32(
                runoff + np.float32(saturated_flux - flux)
            )
        else:
            moisture_flat[0, column] = np.float32(
                min(dqm, max(minimum, candidate))
            )

        for level in range(1, NUM_SOIL_LAYERS):
            coefficient = NUM_SOIL_LAYERS - level - 1
            candidate = np.float32(
                np.float32(
                    cosmc[coefficient] * moisture_flat[level - 1, column]
                )
                + rhsmc[coefficient]
            )
            if candidate < zero:
                moisture_flat[level, column] = minimum
            elif candidate > dqm:
                moisture_flat[level, column] = dqm
                if level == NUM_SOIL_LAYERS - 1:
                    thickness = np.float32(zsmain[level] - zshalf[level])
                else:
                    thickness = np.float32(
                        zshalf[level + 1] - zshalf[level]
                    )
                runoff2 = np.float32(
                    runoff2
                    + np.float32(
                        np.float32(candidate - dqm) * thickness
                    )
                    / timestep
                )
            else:
                moisture_flat[level, column] = np.float32(
                    min(dqm, max(minimum, candidate))
                )
        availability = np.float32(
            moisture_flat[0, column] / np.float32(reference - qmin)
        )
        availability = np.float32(
            availability
            * np.float32(one - column_flat["snowfrac"][column])
        )
        availability = np.float32(
            availability + column_flat["snowfrac"][column]
        )
        output_flat["mavail"][column] = np.float32(
            max(np.float32(0.00001), min(one, availability))
        )
        output_flat["runoff"][column] = runoff
        output_flat["runoff2"][column] = runoff2
        output_flat["infiltrp"][column] = infiltrp
        output_flat["infmax"][column] = infmax

    for name, array in (("soilmois", moisture), ("soiliqw", liquid), *horizontal_outputs.items()):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"RUC soilmoist produced non-finite {name}")
    return RucSoilMoisture(
        soilmois=moisture,
        soiliqw=liquid,
        **horizontal_outputs,
    )


@lru_cache(maxsize=1)
def _load_ruc_saturation_table() -> np.ndarray:
    """Load the float32 ``tbq`` emitted by unmodified WRF v4.6.1."""

    raw = np.loadtxt(
        RUC_ORACLE_DIR / "tbq.csv",
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )
    if raw.shape != (5001, 2):
        raise ValueError(f"RUC tbq oracle has unexpected shape {raw.shape}")
    indices = raw[:, 0].astype(np.int32)
    if not np.array_equal(indices, np.arange(1, 5002, dtype=np.int32)):
        raise ValueError("RUC tbq oracle indices are not exactly 1..5001")
    table = raw[:, 1].astype(np.float32)
    if not np.all(np.isfinite(table)) or not np.all(np.diff(table) > 0.0):
        raise ValueError("RUC tbq oracle must be finite and strictly increasing")
    table.flags.writeable = False
    return table


def ruc_saturation_table() -> np.ndarray:
    """Return a copy of WRF RUC's 5001-entry ``tbq`` table."""

    return _load_ruc_saturation_table().copy()


def ruc_qsn(tn, table: np.ndarray | None = None, *, arrays=None) -> np.ndarray:
    """Transcribe WRF ``qsn``: the ``tbq`` saturation lookup at ``tn``.

    ``qsn`` returns ``0.62198 * es(tn)`` in the table's pressure units; the
    callers divide by the surface pressure to obtain a mixing ratio.  The
    table spans 173.15 K to 423.15 K in 0.05 K steps; below the first node
    WRF clamps the index to 1 and the interpolation weight to zero (so the
    result is ``tbq[0]``), and at or above the last node it clamps the index
    to 5000 with a weight of one (so the result is ``tbq[5000]``).  ``INT``
    truncates toward zero, which is reproduced by the float32-to-int32 cast.
    """

    np = arrays if arrays is not None else _NUMPY
    values = np.asarray(tn, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("RUC qsn temperatures must be finite")
    saturation = (
        _load_ruc_saturation_table()
        if table is None
        else np.asarray(table, dtype=np.float32)
    )
    if saturation.shape != (5001,):
        raise ValueError(
            f"RUC qsn table must have shape (5001,), got {saturation.shape}"
        )

    raw = (values - np.float32(173.15)).astype(np.float32)
    raw = (raw / np.float32(0.05)).astype(np.float32)
    raw = (raw + np.float32(1.0)).astype(np.float32)
    if not np.all(np.abs(raw) < np.float32(2.0 ** 31)):
        raise ValueError("RUC qsn index expression overflows Fortran INT")
    index = raw.astype(np.int32)

    below = index < np.int32(1)
    index = np.where(below, np.int32(1), index).astype(np.int32)
    raw = np.where(below, np.float32(1.0), raw).astype(np.float32)
    above = index > np.int32(5000)
    index = np.where(above, np.int32(5000), index).astype(np.int32)
    raw = np.where(above, np.float32(5001.0), raw).astype(np.float32)

    lower = saturation[index - 1]
    weight = (raw - index.astype(np.float32)).astype(np.float32)
    result = (saturation[index] - lower).astype(np.float32)
    result = (result * weight).astype(np.float32)
    result = (result + lower).astype(np.float32)
    if result.dtype != np.float32:
        raise AssertionError("RUC qsn lost float32 arithmetic")
    return result.reshape(values.shape)


def _ruc_vilka(
    tn: np.float32,
    d1: np.float32,
    d2: np.float32,
    pp: np.float32,
    table: np.ndarray,
) -> tuple[np.float32, np.float32]:
    """Solve WRF ``vilka`` with Fortran integer truncation semantics."""

    raw_index = np.float32(tn - np.float32(173.15))
    raw_index = np.float32(raw_index / np.float32(0.05))
    raw_index = np.float32(raw_index + np.float32(1.0))
    index = int(raw_index)
    if not 1 <= index <= 5000:
        raise ValueError(f"RUC vilka table index {index} is outside 1..5000")

    t1 = np.float32(
        np.float32(173.1)
        + np.float32(np.float32(index) * np.float32(0.05))
    )
    delta = np.float32(table[index] - table[index - 1])
    f1 = np.float32(t1 + np.float32(d1 * table[index - 1]))
    f1 = np.float32(f1 - d2)
    denominator = np.float32(
        np.float32(0.05) + np.float32(d1 * delta)
    )
    ratio = np.float32(f1 / denominator)
    index = int(np.float32(np.float32(index) - ratio))
    if not 1 <= index <= 5000:
        raise ValueError(f"RUC vilka table index {index} is outside 1..5000")

    for _ in range(5001):
        previous = index
        t1 = np.float32(
            np.float32(173.1)
            + np.float32(np.float32(index) * np.float32(0.05))
        )
        delta = np.float32(table[index] - table[index - 1])
        f1 = np.float32(t1 + np.float32(d1 * table[index - 1]))
        f1 = np.float32(f1 - d2)
        denominator = np.float32(
            np.float32(0.05) + np.float32(d1 * delta)
        )
        rn = np.float32(f1 / denominator)
        index = index - int(rn)
        if not 1 <= index <= 5000:
            raise ValueError(
                f"RUC vilka table index {index} is outside 1..5000"
            )
        if index == previous:
            break
    else:
        raise RuntimeError("RUC vilka failed to converge")

    ts = np.float32(t1 - np.float32(np.float32(0.05) * rn))
    numerator = np.float32(
        table[index - 1]
        + np.float32(
            np.float32(table[index - 1] - table[index]) * rn
        )
    )
    qs = np.float32(numerator / pp)
    return qs, ts


def _ruc_constant_flux_depth(conflx, label: str, *, arrays=None) -> np.ndarray:
    """``conflx`` as a float32 column field, accepting a scalar or an array.

    WRF's ``LSMRUC`` passes ``0.5*dz8w(i,1,j)`` -- HALF THE LOWEST MODEL
    LAYER, which is a per-column quantity on a terrain-following coordinate
    -- into a scalar ``sfctmp`` argument, because ``sfctmp`` is called from
    inside ``DO j / DO i``.  This port evaluates the whole column axis in one
    call, so the argument has to carry the column axis with it; a scalar
    still means "the same depth in every column" and is broadcast.

    Returned as an ARRAY in both cases rather than as a scalar in one and an
    array in the other, so the four ``r211`` sites have a single spelling and
    cannot silently take a whole-field value where they meant one column's.
    """

    np = arrays if arrays is not None else _NUMPY
    depth = np.asarray(conflx, dtype=np.float32)
    if depth.ndim > 1:
        raise ValueError(f"RUC {label} conflx must be scalar or 1-D")
    if not np.all(np.isfinite(depth)) or np.any(depth < np.float32(0.0)):
        raise ValueError(f"RUC {label} conflx must be finite and nonnegative")
    return np.atleast_1d(depth)


def ruc_soil_temperature_step(
    values: Mapping[str, object],
    *,
    delt: float,
    conflx: float = 0.5,
    nroot: object = 4,
    cvw: float = 4183.0,
) -> RucSoilTemperature:
    """Transcribe WRF ``soiltemp`` for snow-free nine-level land columns."""

    timestep = np.float32(delt)
    constant_flux_depth = _ruc_constant_flux_depth(conflx, "soiltemp")
    water_heat_capacity = np.float32(cvw)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC soiltemp delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC soiltemp cvw must be finite and positive")
    required = (
        RUC_SOIL_TEMPERATURE_PROFILE_INPUTS
        + RUC_SOIL_TEMPERATURE_COLUMN_INPUTS
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC soiltemp inputs: {', '.join(missing)}")

    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SOIL_TEMPERATURE_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC soiltemp profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SOIL_TEMPERATURE_COLUMN_INPUTS
    }
    if np.any(profiles["thdif"][0] <= np.float32(0.0)):
        raise ValueError("RUC soiltemp top-level thdif must be positive")
    if np.any(profiles["cap"][0] <= np.float32(0.0)):
        raise ValueError("RUC soiltemp top-level cap must be positive")
    if np.any(columns["patm"] <= np.float32(0.0)):
        raise ValueError("RUC soiltemp patm must be positive")
    if np.any(columns["rho"] <= np.float32(0.0)):
        raise ValueError("RUC soiltemp rho must be positive")
    if np.any((columns["mavail"] < 0.0) | (columns["mavail"] > 1.0)):
        raise ValueError("RUC soiltemp mavail must be within 0..1")

    zsmain, zshalf, dtdzs, _ = _ruc_soil_solver_geometry(timestep)
    table = _load_ruc_saturation_table()
    ncolumn = int(np.prod(horizontal_shape))
    profile_flat = {
        name: array.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, array in profiles.items()
    }
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    roots_flat = roots.reshape(ncolumn)
    temperature = np.array(profiles["tso"], dtype=np.float32, copy=True)
    temperature_flat = temperature.reshape(NUM_SOIL_LAYERS, ncolumn)
    outputs = {
        name: np.empty(horizontal_shape, dtype=np.float32)
        for name in ("soilt", "qvg", "qsg", "qcg", "storage")
    }
    output_flat = {name: array.reshape(ncolumn) for name, array in outputs.items()}

    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    cp_air = np.float32(1004.5)
    cvw = water_heat_capacity
    xlv = np.float32(2.5e6)
    stbolt = np.float32(5.67051e-8)
    freeze = np.float32(273.15)
    dzstop = np.float32(one / np.float32(zsmain[1] - zsmain[0]))

    # ``conflx`` may be one depth for the whole field or one per column;
    # bind it to the column axis once rather than at every read.
    flux_depth = np.broadcast_to(constant_flux_depth, (ncolumn,))

    for column in range(ncolumn):
        root_count = int(roots_flat[column])
        cotso = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhtso = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhtso[0] = temperature_flat[-1, column]
        for step in range(NUM_SOIL_LAYERS - 2):
            kn = NUM_SOIL_LAYERS - step - 1
            first = 2 * kn - 3
            x1 = np.float32(
                dtdzs[first - 1]
                * profile_flat["thdif"][kn - 2, column]
            )
            x2 = np.float32(
                dtdzs[first]
                * profile_flat["thdif"][kn - 1, column]
            )
            ft = np.float32(
                temperature_flat[kn - 1, column]
                + np.float32(
                    x1
                    * np.float32(
                        temperature_flat[kn - 2, column]
                        - temperature_flat[kn - 1, column]
                    )
                )
            )
            ft = np.float32(
                ft
                - np.float32(
                    x2
                    * np.float32(
                        temperature_flat[kn - 1, column]
                        - temperature_flat[kn, column]
                    )
                )
            )
            denominator = np.float32(one + x1)
            denominator = np.float32(denominator + x2)
            denominator = np.float32(
                denominator - np.float32(x2 * cotso[step])
            )
            cotso[step + 1] = np.float32(x1 / denominator)
            numerator = np.float32(
                ft + np.float32(x2 * rhtso[step])
            )
            rhtso[step + 1] = np.float32(numerator / denominator)

        rhcs = profile_flat["cap"][0, column]
        h = column_flat["mavail"][column]
        trans = np.float32(
            np.float32(
                column_flat["transum"][column]
                * column_flat["drycan"][column]
            )
            / zshalf[root_count]
        )
        can = np.float32(column_flat["wetcan"][column] + trans)
        umveg = np.float32(
            np.float32(one - column_flat["vegfrac"][column])
            * column_flat["soilres"][column]
        )
        d1 = cotso[NUM_SOIL_LAYERS - 2]
        d2 = rhtso[NUM_SOIL_LAYERS - 2]
        tn = column_flat["soilt"][column]
        qgold = column_flat["qvg"][column]
        d9 = np.float32(
            np.float32(profile_flat["thdif"][0, column] * rhcs) * dzstop
        )
        d10 = np.float32(
            np.float32(column_flat["tkms"][column] * cp_air)
            * column_flat["rho"][column]
        )
        r211 = np.float32(
            np.float32(half * flux_depth[column]) / timestep
        )
        r21 = np.float32(
            np.float32(r211 * cp_air) * column_flat["rho"][column]
        )
        depth_square = np.float32(dzstop * dzstop)
        denominator = np.float32(
            np.float32(profile_flat["thdif"][0, column] * timestep)
            * depth_square
        )
        r22 = np.float32(half / denominator)
        tn2 = np.float32(tn * tn)
        tn4 = np.float32(tn2 * tn2)
        r6 = np.float32(
            np.float32(
                np.float32(column_flat["emiss"][column] * stbolt) * half
            )
            * tn4
        )
        r7 = np.float32(r6 / tn)
        d11 = np.float32(column_flat["rnet"][column] + r6)
        tdenom = np.float32(
            d9 * np.float32(np.float32(one - d1) + r22)
        )
        tdenom = np.float32(tdenom + d10)
        tdenom = np.float32(tdenom + r21)
        tdenom = np.float32(tdenom + r7)
        rain_heat_capacity = np.float32(
            np.float32(column_flat["rainf"][column] * cvw)
            * column_flat["prcpms"][column]
        )
        tdenom = np.float32(tdenom + rain_heat_capacity)
        fkq = np.float32(
            column_flat["qkms"][column] * column_flat["rho"][column]
        )
        r210 = np.float32(r211 * column_flat["rho"][column])
        c = np.float32(
            np.float32(column_flat["vegfrac"][column] * fkq) * can
        )
        cc = np.float32(np.float32(c * xlv) / tdenom)
        aa_inner = np.float32(np.float32(fkq * umveg) + r210)
        aa = np.float32(np.float32(xlv * aa_inner) / tdenom)

        humidity_inner = np.float32(np.float32(fkq * umveg) + c)
        humidity_inner = np.float32(
            column_flat["qvatm"][column] * humidity_inner
        )
        humidity_inner = np.float32(humidity_inner + np.float32(r210 * qgold))
        numerator = np.float32(d10 * column_flat["tabs"][column])
        numerator = np.float32(numerator + np.float32(r21 * tn))
        numerator = np.float32(numerator + np.float32(xlv * humidity_inner))
        numerator = np.float32(numerator + d11)
        conduction = np.float32(d2 + np.float32(r22 * tn))
        numerator = np.float32(numerator + np.float32(d9 * conduction))
        rain_temperature = np.float32(
            max(freeze, column_flat["tabs"][column])
        )
        numerator = np.float32(
            numerator + np.float32(rain_heat_capacity * rain_temperature)
        )
        bb = np.float32(numerator / tdenom)
        aa1 = np.float32(aa + cc)
        pp = np.float32(column_flat["patm"][column] * np.float32(1.0e3))
        aa1 = np.float32(aa1 / pp)
        qs1, ts1 = _ruc_vilka(tn, aa1, bb, pp, table)
        tx2 = np.float32(
            column_flat["qvatm"][column] * np.float32(one - h)
        )
        q1 = np.float32(tx2 + np.float32(h * qs1))
        if q1 >= qs1:
            qvg = qs1
            qsg = qs1
            qcg = np.float32(max(zero, np.float32(q1 - qs1)))
        else:
            bb = np.float32(bb - np.float32(aa * tx2))
            aa = np.float32(np.float32(np.float32(aa * h) + cc) / pp)
            qs1, ts1 = _ruc_vilka(tn, aa, bb, pp, table)
            q1 = np.float32(tx2 + np.float32(h * qs1))
            if q1 >= qs1:
                qvg = qs1
                qsg = qs1
                qcg = np.float32(max(zero, np.float32(q1 - qs1)))
            else:
                qsg = qs1
                qvg = q1
                qcg = zero

        temperature_flat[0, column] = ts1
        for level in range(1, NUM_SOIL_LAYERS):
            coefficient = NUM_SOIL_LAYERS - level - 1
            temperature_flat[level, column] = np.float32(
                rhtso[coefficient]
                + np.float32(cotso[coefficient] * temperature_flat[level - 1, column])
            )
        storage_coefficient = np.float32(
            np.float32(np.float32(cp_air * column_flat["rho"][column]) * r211)
            + np.float32(
                np.float32(np.float32(rhcs * zsmain[1]) * half) / timestep
            )
        )
        storage = np.float32(storage_coefficient * np.float32(ts1 - tn))
        storage = np.float32(
            storage
            + np.float32(
                np.float32(np.float32(xlv * column_flat["rho"][column]) * r211)
                * np.float32(qvg - qgold)
            )
        )
        storage = np.float32(
            storage
            - np.float32(
                rain_heat_capacity * np.float32(rain_temperature - ts1)
            )
        )
        output_flat["soilt"][column] = ts1
        output_flat["qvg"][column] = qvg
        output_flat["qsg"][column] = qsg
        output_flat["qcg"][column] = qcg
        output_flat["storage"][column] = storage

    for name, array in (("tso", temperature), *outputs.items()):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"RUC soiltemp produced non-finite {name}")
    return RucSoilTemperature(tso=temperature, **outputs)


def ruc_sea_ice_step(
    values: Mapping[str, object],
    *,
    delt: float,
    conflx: float = 40.0,
    myj: bool = False,
    cw: float = 4.183e6,
) -> RucSeaIceStep:
    """Transcribe WRF ``sice``: the snow-free sea-ice column.

    Heat diffusion through the nine ice levels plus the surface energy
    balance closed by ``vilka``.  Every returned ice temperature is clipped
    at the 271.4 K sea-ice melting cap; melt itself is not modelled, the
    excess is absorbed by ``sice``'s local ``icemelt``.

    ``sice`` does not touch the soil water arrays.  ``sfctmp`` forces
    ``soilmois=1``, ``soiliqw=0``, ``soilice=1``, ``smfrkeep=1`` and
    ``keepfr=0`` in an identical nine-level loop immediately after both of
    its ``sice`` call sites (WRF v4.6.1 ``phys/module_sf_ruclsm.F``
    ``:1871-1877`` and ``:2184-2190``).  That forcing is reproduced here so
    callers do not have to supply or repeat it; the remaining post-call
    assignments in ``sfctmp`` (``edir1``, ``ec1``, ``ett1``, ``runoff1``,
    ``runoff2``, ``mavail``, ``infiltr``, ``cst``) belong to the dispatcher
    and are deliberately left out.

    Arguments WRF passes but ``sice`` never reads are omitted from this
    signature: ``qcatm``, ``gsw``, ``tice``, ``rhosice``, ``zshalf``,
    ``dtdzs2``, ``nroot`` and ``xlv`` are dead, ``glw`` reaches only the
    local ``xinet`` which is assigned and never used, and the incoming
    ``qcg`` is overwritten with zero before it is read.

    ``fltot`` is identically zero: ``sice`` computes ``icemelt`` as the full
    residual ``rnet-xls*eeta-hft-s-x`` and then subtracts it from the same
    expression, so the FP32 cancellation is exact.
    """

    timestep = np.float32(delt)
    constant_flux_depth = _ruc_constant_flux_depth(conflx, "sice")
    water_heat_capacity = np.float32(cw)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC sice delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC sice cw must be finite and positive")
    if not isinstance(myj, bool):
        raise TypeError("RUC sice myj must be a bool")
    required = RUC_SEA_ICE_PROFILE_INPUTS + RUC_SEA_ICE_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC sice inputs: {', '.join(missing)}")

    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SEA_ICE_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC sice profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SEA_ICE_COLUMN_INPUTS
    }
    if np.any(profiles["thdifice"][0] <= np.float32(0.0)):
        raise ValueError("RUC sice top-level thdifice must be positive")
    if np.any(profiles["capice"][0] <= np.float32(0.0)):
        raise ValueError("RUC sice top-level capice must be positive")
    if np.any(columns["patm"] <= np.float32(0.0)):
        raise ValueError("RUC sice patm must be positive")
    if np.any(columns["rho"] <= np.float32(0.0)):
        raise ValueError("RUC sice rho must be positive")

    zsmain, _, dtdzs, _ = _ruc_soil_solver_geometry(timestep)
    table = _load_ruc_saturation_table()
    ncolumn = int(np.prod(horizontal_shape))
    profile_flat = {
        name: array.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, array in profiles.items()
    }
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    temperature = np.array(profiles["tso"], dtype=np.float32, copy=True)
    temperature_flat = temperature.reshape(NUM_SOIL_LAYERS, ncolumn)
    scalar_names = (
        "dew", "soilt", "qvg", "qsg", "qcg", "eeta", "qfx", "hfx",
        "s", "evapl", "prcpl", "fltot",
    )
    outputs = {
        name: np.empty(horizontal_shape, dtype=np.float32)
        for name in scalar_names
    }
    output_flat = {name: array.reshape(ncolumn) for name, array in outputs.items()}

    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    cp_air = np.float32(1004.5)
    rovcp = np.float32(np.float32(287.0) / cp_air)
    cvw = water_heat_capacity
    xls = np.float32(2.85e6)
    p1000mb = np.float32(100000.0)
    reference = np.float32(p1000mb * np.float32(0.00001))
    stbolt = np.float32(5.67051e-8)
    freeze = np.float32(273.15)
    ice_cap = np.float32(271.4)
    dzstop = np.float32(one / np.float32(zsmain[1] - zsmain[0]))

    # ``conflx`` may be one depth for the whole field or one per column;
    # bind it to the column axis once rather than at every read.
    flux_depth = np.broadcast_to(constant_flux_depth, (ncolumn,))

    for column in range(ncolumn):
        prcpl = column_flat["prcpms"][column]
        ras = np.float32(column_flat["rho"][column] * np.float32(1.0e-3))
        cotso = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhtso = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhtso[0] = temperature_flat[-1, column]
        for step in range(NUM_SOIL_LAYERS - 2):
            kn = NUM_SOIL_LAYERS - step - 1
            first = 2 * kn - 3
            x1 = np.float32(
                dtdzs[first - 1]
                * profile_flat["thdifice"][kn - 2, column]
            )
            x2 = np.float32(
                dtdzs[first]
                * profile_flat["thdifice"][kn - 1, column]
            )
            ft = np.float32(
                temperature_flat[kn - 1, column]
                + np.float32(
                    x1
                    * np.float32(
                        temperature_flat[kn - 2, column]
                        - temperature_flat[kn - 1, column]
                    )
                )
            )
            ft = np.float32(
                ft
                - np.float32(
                    x2
                    * np.float32(
                        temperature_flat[kn - 1, column]
                        - temperature_flat[kn, column]
                    )
                )
            )
            denominator = np.float32(one + x1)
            denominator = np.float32(denominator + x2)
            denominator = np.float32(
                denominator - np.float32(x2 * cotso[step])
            )
            cotso[step + 1] = np.float32(x1 / denominator)
            numerator = np.float32(ft + np.float32(x2 * rhtso[step]))
            rhtso[step + 1] = np.float32(numerator / denominator)

        rhcs = profile_flat["capice"][0, column]
        d1 = cotso[NUM_SOIL_LAYERS - 2]
        d2 = rhtso[NUM_SOIL_LAYERS - 2]
        tn = column_flat["soilt"][column]
        d9 = np.float32(
            np.float32(profile_flat["thdifice"][0, column] * rhcs) * dzstop
        )
        d10 = np.float32(
            np.float32(column_flat["tkms"][column] * cp_air)
            * column_flat["rho"][column]
        )
        r211 = np.float32(np.float32(half * flux_depth[column]) / timestep)
        r21 = np.float32(
            np.float32(r211 * cp_air) * column_flat["rho"][column]
        )
        depth_square = np.float32(dzstop * dzstop)
        denominator = np.float32(
            np.float32(profile_flat["thdifice"][0, column] * timestep)
            * depth_square
        )
        r22 = np.float32(half / denominator)
        tn2 = np.float32(tn * tn)
        tn4 = np.float32(tn2 * tn2)
        r6 = np.float32(
            np.float32(
                np.float32(column_flat["emiss"][column] * stbolt) * half
            )
            * tn4
        )
        r7 = np.float32(r6 / tn)
        d11 = np.float32(column_flat["rnet"][column] + r6)
        tdenom = np.float32(d9 * np.float32(np.float32(one - d1) + r22))
        tdenom = np.float32(tdenom + d10)
        tdenom = np.float32(tdenom + r21)
        tdenom = np.float32(tdenom + r7)
        rain_heat_capacity = np.float32(
            np.float32(column_flat["rainf"][column] * cvw)
            * column_flat["prcpms"][column]
        )
        tdenom = np.float32(tdenom + rain_heat_capacity)
        fkq = np.float32(
            column_flat["qkms"][column] * column_flat["rho"][column]
        )
        r210 = np.float32(r211 * column_flat["rho"][column])
        aa = np.float32(np.float32(xls * np.float32(fkq + r210)) / tdenom)
        humidity_inner = np.float32(column_flat["qvatm"][column] * fkq)
        humidity_inner = np.float32(
            humidity_inner
            + np.float32(r210 * column_flat["qvg"][column])
        )
        numerator = np.float32(d10 * column_flat["tabs"][column])
        numerator = np.float32(numerator + np.float32(r21 * tn))
        numerator = np.float32(numerator + np.float32(xls * humidity_inner))
        numerator = np.float32(numerator + d11)
        conduction = np.float32(d2 + np.float32(r22 * tn))
        numerator = np.float32(numerator + np.float32(d9 * conduction))
        rain_temperature = np.float32(
            max(freeze, column_flat["tabs"][column])
        )
        numerator = np.float32(
            numerator + np.float32(rain_heat_capacity * rain_temperature)
        )
        bb = np.float32(numerator / tdenom)
        aa1 = aa
        pp = np.float32(column_flat["patm"][column] * np.float32(1.0e3))
        aa1 = np.float32(aa1 / pp)
        qgold = column_flat["qsg"][column]
        qs1, ts1 = _ruc_vilka(tn, aa1, bb, pp, table)
        qvg = qs1
        qsg = qs1
        temperature_flat[0, column] = min(ice_cap, ts1)
        qcg = zero
        soilt = temperature_flat[0, column]
        for level in range(1, NUM_SOIL_LAYERS):
            coefficient = NUM_SOIL_LAYERS - level - 1
            temperature_flat[level, column] = min(
                ice_cap,
                np.float32(
                    rhtso[coefficient]
                    + np.float32(
                        cotso[coefficient] * temperature_flat[level - 1, column]
                    )
                ),
            )
        dew = zero

        # WRF's t3/upflux/xinet block here is dead: xinet is assigned and
        # never read, so nothing downstream depends on it.
        hft = np.float32(
            -np.float32(
                np.float32(
                    np.float32(column_flat["tkms"][column] * cp_air)
                    * column_flat["rho"][column]
                )
                * np.float32(column_flat["tabs"][column] - soilt)
            )
        )
        # float64 rounded once, matching the CUDA kernel; see
        # _RUC_PROVISIONAL_TRANSCENDENTALS and the identical lift in
        # ruc_soil_step and ruc_snow_sea_ice_step.
        exner = np.float32(
            np.power(
                np.float64(
                    np.float32(reference / column_flat["patm"][column])
                ),
                np.float64(rovcp),
            )
        )
        hfx = np.float32(
            -np.float32(
                np.float32(
                    np.float32(
                        np.float32(column_flat["tkms"][column] * cp_air)
                        * column_flat["rho"][column]
                    )
                    * np.float32(column_flat["tabs"][column] - soilt)
                )
                * exner
            )
        )
        q1 = np.float32(
            -np.float32(
                np.float32(column_flat["qkms"][column] * ras)
                * np.float32(column_flat["qvatm"][column] - qsg)
            )
        )
        if q1 <= zero:
            if myj:
                eeta = np.float32(
                    -np.float32(
                        np.float32(
                            np.float32(column_flat["qkms"][column] * ras)
                            * np.float32(
                                np.float32(
                                    column_flat["qvatm"][column]
                                    / np.float32(
                                        one + column_flat["qvatm"][column]
                                    )
                                )
                                - np.float32(qsg / np.float32(one + qsg))
                            )
                        )
                        * np.float32(1.0e3)
                    )
                )
            else:
                dew = np.float32(
                    column_flat["qkms"][column]
                    * np.float32(column_flat["qvatm"][column] - qsg)
                )
                eeta = np.float32(
                    -np.float32(column_flat["rho"][column] * dew)
                )
            qfx = np.float32(xls * eeta)
            eeta = np.float32(-np.float32(column_flat["rho"][column] * dew))
        else:
            if myj:
                eeta = np.float32(
                    -np.float32(
                        np.float32(
                            np.float32(column_flat["qkms"][column] * ras)
                            * np.float32(
                                np.float32(
                                    column_flat["qvatm"][column]
                                    / np.float32(
                                        one + column_flat["qvatm"][column]
                                    )
                                )
                                - np.float32(qvg / np.float32(one + qvg))
                            )
                        )
                        * np.float32(1.0e3)
                    )
                )
            else:
                eeta = np.float32(q1 * np.float32(1.0e3))
            qfx = np.float32(xls * eeta)
            eeta = np.float32(q1 * np.float32(1.0e3))
        evapl = eeta

        storage = np.float32(
            np.float32(
                np.float32(
                    profile_flat["thdifice"][0, column] * rhcs
                )
                * dzstop
            )
            * np.float32(
                temperature_flat[0, column] - temperature_flat[1, column]
            )
        )
        x = np.float32(
            np.float32(
                np.float32(
                    np.float32(cp_air * column_flat["rho"][column]) * r211
                )
                + np.float32(
                    np.float32(np.float32(rhcs * zsmain[1]) * half) / timestep
                )
            )
            * np.float32(soilt - tn)
        )
        x = np.float32(
            x
            + np.float32(
                np.float32(np.float32(xls * column_flat["rho"][column]) * r211)
                * np.float32(qsg - qgold)
            )
        )
        x = np.float32(
            x
            - np.float32(
                rain_heat_capacity * np.float32(rain_temperature - soilt)
            )
        )
        residual = np.float32(column_flat["rnet"][column] - np.float32(xls * eeta))
        residual = np.float32(residual - hft)
        residual = np.float32(residual - storage)
        residual = np.float32(residual - x)
        icemelt = residual
        fltot = np.float32(residual - icemelt)

        output_flat["dew"][column] = dew
        output_flat["soilt"][column] = soilt
        output_flat["qvg"][column] = qvg
        output_flat["qsg"][column] = qsg
        output_flat["qcg"][column] = qcg
        output_flat["eeta"][column] = eeta
        output_flat["qfx"][column] = qfx
        output_flat["hfx"][column] = hfx
        output_flat["s"][column] = storage
        output_flat["evapl"][column] = evapl
        output_flat["prcpl"][column] = prcpl
        output_flat["fltot"][column] = fltot

    for name, array in (("tso", temperature), *outputs.items()):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"RUC sice produced non-finite {name}")
    forced = {
        "soilmois": np.float32(1.0),
        "soiliqw": np.float32(0.0),
        "soilice": np.float32(1.0),
        "smfrkeep": np.float32(1.0),
        "keepfr": np.float32(0.0),
    }
    return RucSeaIceStep(
        tso=temperature,
        **{
            name: np.full(shape, value, dtype=np.float32)
            for name, value in forced.items()
        },
        **outputs,
    )


def _ruc_phase_partition(
    soilmois: np.ndarray,
    tso: np.ndarray,
    smfrkeep: np.ndarray,
    keepfr: np.ndarray,
    columns: Mapping[str, np.ndarray],
    *,
    update_smfrkeep: bool,
) -> dict[str, np.ndarray]:
    """Apply the freezing curve used before and after WRF ``soiltemp``."""

    shape = soilmois.shape
    horizontal_shape = shape[1:]
    ncolumn = int(np.prod(horizontal_shape))
    moisture = soilmois.reshape(NUM_SOIL_LAYERS, ncolumn)
    temperature = tso.reshape(NUM_SOIL_LAYERS, ncolumn)
    frozen_memory = smfrkeep.reshape(NUM_SOIL_LAYERS, ncolumn)
    freeze_flag = keepfr.reshape(NUM_SOIL_LAYERS, ncolumn)
    column_flat = {name: value.reshape(ncolumn) for name, value in columns.items()}
    fields = {
        name: np.zeros(shape, dtype=np.float32)
        for name in (
            "soiliqw", "soilice", "tav", "soilmoism", "soiliqwm",
            "soilicem", "lwsat", "fwsat",
        )
    }
    flat = {
        name: value.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, value in fields.items()
    }
    zero = np.float32(0.0)
    half = np.float32(0.5)
    one = np.float32(1.0)
    freeze = np.float32(273.15)
    xlmelt = np.float32(3.35e5)
    gravity = np.float32(9.81)
    # ``soil`` forms this as rhoice*1.e-3, which is one ULP above the
    # decimal float32 literal 0.9 used by the isolated soilmoist harness.
    riw = np.float32(np.float32(900.0) * np.float32(1.0e-3))

    for column in range(ncolumn):
        dqm = column_flat["dqm"][column]
        qmin = column_flat["qmin"][column]
        psis = column_flat["psis"][column]
        bclh = column_flat["bclh"][column]
        maximum = np.float32(dqm + qmin)
        exponent = np.float32(np.float32(-one) / bclh)
        for level in range(NUM_SOIL_LAYERS):
            level_temperature = temperature[level, column]
            tln = np.float32(np.log(np.float32(level_temperature / freeze)))
            if tln < zero:
                base = np.float32(
                    xlmelt * np.float32(level_temperature - freeze)
                )
                base = np.float32(base / level_temperature)
                base = np.float32(base / gravity)
                base = np.float32(base / psis)
                # Round a float64 power once, so the freezing curve does not
                # depend on the numpy build (2.2 and 2.4 disagree on 0.008 %
                # and 24 % of these arguments respectively).  See
                # _RUC_PROVISIONAL_TRANSCENDENTALS.
                liquid = np.float32(
                    np.float32(
                        maximum
                        * np.float32(
                            np.power(np.float64(base), np.float64(exponent))
                        )
                    )
                    - qmin
                )
                liquid = np.float32(max(zero, liquid))
                liquid = np.float32(min(liquid, moisture[level, column]))
                ice = np.float32(
                    np.float32(moisture[level, column] - liquid) / riw
                )
                if freeze_flag[level, column] == one:
                    ice = np.float32(min(ice, frozen_memory[level, column]))
                    liquid = np.float32(
                        max(
                            zero,
                            np.float32(
                                moisture[level, column]
                                - np.float32(ice * riw)
                            ),
                        )
                    )
                flat["soiliqw"][level, column] = liquid
                flat["soilice"][level, column] = ice
            else:
                flat["soiliqw"][level, column] = moisture[level, column]

        for level in range(NUM_SOIL_LAYERS - 1):
            middle_temperature = np.float32(
                half
                * np.float32(
                    temperature[level, column]
                    + temperature[level + 1, column]
                )
            )
            middle_moisture = np.float32(
                half
                * np.float32(
                    moisture[level, column]
                    + moisture[level + 1, column]
                )
            )
            flat["tav"][level, column] = middle_temperature
            flat["soilmoism"][level, column] = middle_moisture
            tavln = np.float32(np.log(np.float32(middle_temperature / freeze)))
            if tavln < zero:
                base = np.float32(
                    xlmelt * np.float32(middle_temperature - freeze)
                )
                base = np.float32(base / middle_temperature)
                base = np.float32(base / gravity)
                base = np.float32(base / psis)
                # Same float64-rounded power substitution as the full-level
                # loop above.
                liquid = np.float32(
                    np.float32(
                        maximum
                        * np.float32(
                            np.power(np.float64(base), np.float64(exponent))
                        )
                    )
                    - qmin
                )
                flat["fwsat"][level, column] = np.float32(dqm - liquid)
                flat["lwsat"][level, column] = np.float32(liquid + qmin)
                liquid = np.float32(max(zero, liquid))
                liquid = np.float32(min(liquid, middle_moisture))
                ice = np.float32(
                    np.float32(middle_moisture - liquid) / riw
                )
                if freeze_flag[level, column] == one:
                    memory = np.float32(
                        half
                        * np.float32(
                            frozen_memory[level, column]
                            + frozen_memory[level + 1, column]
                        )
                    )
                    ice = np.float32(min(ice, memory))
                    liquid = np.float32(
                        max(
                            zero,
                            np.float32(middle_moisture - np.float32(ice * riw)),
                        )
                    )
                    flat["fwsat"][level, column] = np.float32(dqm - liquid)
                    flat["lwsat"][level, column] = np.float32(liquid + qmin)
                flat["soiliqwm"][level, column] = liquid
                flat["soilicem"][level, column] = ice
            else:
                flat["soiliqwm"][level, column] = middle_moisture
                flat["lwsat"][level, column] = maximum

        if update_smfrkeep:
            for level in range(NUM_SOIL_LAYERS):
                ice = flat["soilice"][level, column]
                if ice > zero:
                    frozen_memory[level, column] = ice
                else:
                    frozen_memory[level, column] = np.float32(
                        moisture[level, column] / riw
                    )
    return fields


def ruc_soil_step(
    values: Mapping[str, object],
    iland,
    *,
    nroot: object,
    delt: float,
    conflx: float,
    myj: bool = False,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    parameters: RucParameterBundle | None = None,
) -> RucSoilStep:
    """Run the complete deterministic snow-free WRF RUC land column."""

    if myj is not False:
        raise ValueError("RUC first soil lane supports myj=False only")
    timestep = np.float32(delt)
    constant_flux_depth = _ruc_constant_flux_depth(conflx, "soil")
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC soil delt must be finite and positive")
    required = RUC_SOIL_STEP_PROFILE_INPUTS + RUC_SOIL_STEP_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC soil inputs: {', '.join(missing)}")

    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SOIL_STEP_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC soil profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    land_type = _horizontal_integer_field(iland, horizontal_shape, "iland")
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SOIL_STEP_COLUMN_INPUTS
    }
    if np.any(columns["dqm"] <= 0.0):
        raise ValueError("RUC soil dqm must be positive")
    if np.any(columns["psis"] >= 0.0):
        raise ValueError("RUC soil psis must be negative")
    if np.any(columns["bclh"] <= 0.0):
        raise ValueError("RUC soil bclh must be positive")
    if np.any(columns["sat"] <= 0.0):
        raise ValueError("RUC soil canopy saturation must be positive")
    if np.any(columns["rho"] <= 0.0) or np.any(columns["patm"] <= 0.0):
        raise ValueError("RUC soil rho and patm must be positive")
    if np.any((columns["mavail"] < 0.0) | (columns["mavail"] > 1.0)):
        raise ValueError("RUC soil mavail must be within 0..1")

    soilmois = np.array(profiles["soilmois"], dtype=np.float32, copy=True)
    tso = np.array(profiles["tso"], dtype=np.float32, copy=True)
    smfrkeep = np.array(profiles["smfrkeep"], dtype=np.float32, copy=True)
    keepfr = np.array(profiles["keepfr"], dtype=np.float32, copy=True)
    told = tso.copy()
    smold = soilmois.copy()
    phase = _ruc_phase_partition(
        soilmois, tso, smfrkeep, keepfr, columns, update_smfrkeep=True
    )
    properties = ruc_soil_properties({
        **{name: phase[name] for name in (
            "fwsat", "lwsat", "tav", "soilmoism", "soiliqwm", "soilicem"
        )},
        "keepfr": keepfr,
        "soilmois": soilmois,
        "soiliqw": phase["soiliqw"],
        "soilice": phase["soilice"],
        **{name: columns[name] for name in (
            "qwrtz", "rhocs", "dqm", "qmin", "psis", "bclh", "ksat"
        )},
    }, riw=float(np.float32(np.float32(900.0) * np.float32(1.0e-3))))

    ncolumn = int(np.prod(horizontal_shape))
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    roots_flat = roots.reshape(ncolumn)
    zero = np.float32(0.0)
    one = np.float32(1.0)
    quarter = np.float32(0.25)
    dew = np.zeros(horizontal_shape, dtype=np.float32)
    wetcan = np.empty(horizontal_shape, dtype=np.float32)
    drycan = np.empty(horizontal_shape, dtype=np.float32)
    dew_flat = dew.reshape(ncolumn)
    wet_flat = wetcan.reshape(ncolumn)
    dry_flat = drycan.reshape(ncolumn)
    for column in range(ncolumn):
        if column_flat["qvatm"][column] >= column_flat["qsg"][column]:
            dew_flat[column] = np.float32(
                column_flat["qkms"][column]
                * np.float32(
                    column_flat["qvatm"][column] - column_flat["qsg"][column]
                )
            )
        ratio = np.float32(
            column_flat["cst"][column] / column_flat["sat"][column]
        )
        ratio = np.float32(max(zero, ratio))
        # Round a float64 power once, matching the CUDA canopy kernel and
        # ruc_snow_soil_step's copy of this same block.  numpy's float32
        # power and the CUDA device libm's powf are two different non-glibc
        # functions; see _RUC_PROVISIONAL_TRANSCENDENTALS.
        fraction = np.float32(
            np.power(
                np.float64(ratio), np.float64(column_flat["cn"][column])
            )
        )
        wet_flat[column] = np.float32(min(quarter, fraction))
        dry_flat[column] = np.float32(one - wet_flat[column])

    transpiration = ruc_transpiration(
        phase["soiliqw"],
        columns["tabs"], columns["lai"], columns["gswin"],
        columns["dqm"], columns["qmin"], columns["ref"], columns["wilt"],
        columns["pc"], land_type, nroot=roots, mminlu=mminlu,
        parameters=parameters,
    )
    soilres = np.empty(horizontal_shape, dtype=np.float32)
    soilres_flat = soilres.reshape(ncolumn)
    moisture_flat = soilmois.reshape(NUM_SOIL_LAYERS, ncolumn)
    piconst = np.float32(3.141592653589793)
    for column in range(ncolumn):
        fc = np.float32(
            max(
                column_flat["qmin"][column],
                np.float32(column_flat["ref"][column] * np.float32(0.5)),
            )
        )
        total_top = np.float32(
            moisture_flat[0, column] + column_flat["qmin"][column]
        )
        if (
            total_top > fc
            or np.float32(
                column_flat["qvatm"][column] - column_flat["qvg"][column]
            ) > zero
        ):
            soilres_flat[column] = one
        else:
            fex = np.float32(total_top / fc)
            fex = np.float32(max(np.float32(0.01), min(one, fex)))
            cosine = np.float32(np.cos(np.float32(piconst * fex)))
            resistance = np.float32(one - cosine)
            resistance = np.float32(resistance * resistance)
            soilres_flat[column] = np.float32(quarter * resistance)

    temperature = ruc_soil_temperature_step(
        {
            "thdif": properties.thdif,
            "cap": properties.cap,
            "tso": tso,
            **{name: columns[name] for name in (
                "prcpms", "rainf", "patm", "tabs", "qvatm", "qcatm",
                "emiss", "rnet", "qkms", "tkms", "pc", "rho", "vegfrac",
                "lai", "dqm", "qmin", "bclh",
            )},
            "drycan": drycan,
            "wetcan": wetcan,
            "transum": transpiration.transum,
            "dew": dew,
            "mavail": columns["mavail"],
            "soilres": soilres,
            "alfa": np.ones(horizontal_shape, dtype=np.float32),
            "soilt": columns["soilt"],
            "qvg": columns["qvg"],
            "qsg": columns["qsg"],
            "qcg": columns["qcg"],
        },
        delt=float(timestep), conflx=constant_flux_depth, nroot=roots,
        cvw=4.183e6,
    )
    tso = temperature.tso
    phase = _ruc_phase_partition(
        soilmois, tso, smfrkeep, keepfr, columns, update_smfrkeep=False
    )

    zsmain, zshalf, _, _ = _ruc_soil_solver_geometry(timestep)
    transp = np.zeros_like(soilmois)
    transp_flat = transp.reshape(NUM_SOIL_LAYERS, ncolumn)
    weights_flat = transpiration.tranf.reshape(NUM_SOIL_LAYERS, ncolumn)
    ett1 = np.zeros(horizontal_shape, dtype=np.float32)
    dew = np.zeros(horizontal_shape, dtype=np.float32)
    ett_flat = ett1.reshape(ncolumn)
    dew_flat = dew.reshape(ncolumn)
    qsg_flat = temperature.qsg.reshape(ncolumn)
    for column in range(ncolumn):
        if column_flat["qvatm"][column] >= qsg_flat[column]:
            dew_flat[column] = np.float32(
                column_flat["qkms"][column]
                * np.float32(column_flat["qvatm"][column] - qsg_flat[column])
            )
            continue
        root_count = int(roots_flat[column])
        ras = np.float32(column_flat["rho"][column] * np.float32(1.0e-3))
        for level in range(root_count):
            flux = np.float32(
                column_flat["vegfrac"][column] * ras
            )
            flux = np.float32(flux * column_flat["qkms"][column])
            flux = np.float32(
                flux * np.float32(column_flat["qvatm"][column] - qsg_flat[column])
            )
            flux = np.float32(flux * weights_flat[level, column])
            flux = np.float32(flux * dry_flat[column])
            flux = np.float32(flux / zshalf[root_count])
            if flux > zero:
                flux = zero
            transp_flat[level, column] = flux
            ett_flat[column] = np.float32(ett_flat[column] - flux)

    precipitation = np.empty(horizontal_shape, dtype=np.float32)
    ras = np.empty(horizontal_shape, dtype=np.float32)
    precipitation_flat = precipitation.reshape(ncolumn)
    ras_flat = ras.reshape(ncolumn)
    for column in range(ncolumn):
        precipitation_flat[column] = np.float32(-column_flat["infwater"][column])
        ras_flat[column] = np.float32(
            column_flat["rho"][column] * np.float32(1.0e-3)
        )
    moisture = ruc_soil_moisture_step(
        {
            "diffu": properties.diffu,
            "hydro": properties.hydro,
            "transp": transp,
            "soilice": phase["soilice"],
            "soilmois": soilmois,
            "soiliqw": phase["soiliqw"],
            "qsg": temperature.qsg,
            "qvg": temperature.qvg,
            "qcg": temperature.qcg,
            "qcatm": columns["qcatm"],
            "qvatm": columns["qvatm"],
            "prcp": precipitation,
            "qkms": columns["qkms"],
            "drip": columns["drip"],
            "dew": dew,
            "smelt": np.zeros(horizontal_shape, dtype=np.float32),
            "vegfrac": columns["vegfrac"],
            "snowfrac": np.zeros(horizontal_shape, dtype=np.float32),
            "soilres": soilres,
            "dqm": columns["dqm"],
            "qmin": columns["qmin"],
            "ref": columns["ref"],
            "ksat": columns["ksat"],
            "ras": ras,
        },
        delt=float(timestep),
    )
    soilmois = moisture.soilmois
    for column in range(ncolumn):
        for level in range(NUM_SOIL_LAYERS):
            if phase["soilice"].reshape(NUM_SOIL_LAYERS, ncolumn)[level, column] > zero:
                if (
                    tso.reshape(NUM_SOIL_LAYERS, ncolumn)[level, column]
                    > told.reshape(NUM_SOIL_LAYERS, ncolumn)[level, column]
                    and soilmois.reshape(NUM_SOIL_LAYERS, ncolumn)[level, column]
                    > smold.reshape(NUM_SOIL_LAYERS, ncolumn)[level, column]
                ):
                    keepfr.reshape(NUM_SOIL_LAYERS, ncolumn)[level, column] = one
                else:
                    keepfr.reshape(NUM_SOIL_LAYERS, ncolumn)[level, column] = zero

    scalar_names = (
        "cst", "dew", "soilt", "qvg", "qsg", "qcg", "edir1", "ec1",
        "ett1", "eeta", "qfx", "hfx", "s", "evapl", "prcpl", "fltot",
        "runoff1", "runoff2", "mavail", "infiltrp", "smf",
    )
    output = {
        name: np.zeros(horizontal_shape, dtype=np.float32) for name in scalar_names
    }
    output["cst"][...] = columns["cst"]
    output["dew"][...] = dew
    output["soilt"][...] = temperature.soilt
    output["qvg"][...] = temperature.qvg
    output["qsg"][...] = temperature.qsg
    output["qcg"][...] = temperature.qcg
    output["ett1"][...] = ett1
    output["prcpl"][...] = columns["prcpms"]
    output["runoff1"][...] = moisture.runoff
    output["runoff2"][...] = moisture.runoff2
    output["mavail"][...] = moisture.mavail
    output["infiltrp"][...] = moisture.infiltrp
    output_flat = {name: value.reshape(ncolumn) for name, value in output.items()}
    qvg_flat = temperature.qvg.reshape(ncolumn)
    soilt_flat = temperature.soilt.reshape(ncolumn)
    thdif_flat = properties.thdif.reshape(NUM_SOIL_LAYERS, ncolumn)
    cap_flat = properties.cap.reshape(NUM_SOIL_LAYERS, ncolumn)
    tso_flat = tso.reshape(NUM_SOIL_LAYERS, ncolumn)
    xlv = np.float32(2.5e6)
    cp_air = np.float32(1004.5)
    stbolt = np.float32(5.67051e-8)
    rovcp = np.float32(np.float32(287.0) / cp_air)
    dzstop = np.float32(one / np.float32(zsmain[1] - zsmain[0]))
    for column in range(ncolumn):
        hft = np.float32(
            -np.float32(
                np.float32(column_flat["tkms"][column] * cp_air)
                * column_flat["rho"][column]
            )
            * np.float32(column_flat["tabs"][column] - soilt_flat[column])
        )
        # Round a float64 power once, matching the CUDA kernel and
        # ruc_snow_soil_step's copy of this block.  This is the site that was
        # MEASURED to diverge: with numpy's float32 power here and the device
        # libm's powf there, hfx moved 1 ULP on 1 of 512 warm columns and
        # 2 ULP on 13 of 4,096.  See _RUC_PROVISIONAL_TRANSCENDENTALS.
        pressure_factor = np.float32(
            np.power(
                np.float64(np.float32(one / column_flat["patm"][column])),
                np.float64(rovcp),
            )
        )
        output_flat["hfx"][column] = np.float32(hft * pressure_factor)
        q1 = np.float32(
            -np.float32(
                np.float32(column_flat["qkms"][column] * ras_flat[column])
                * np.float32(column_flat["qvatm"][column] - qsg_flat[column])
            )
        )
        if q1 <= zero:
            output_flat["ec1"][column] = zero
            output_flat["edir1"][column] = zero
            output_flat["ett1"][column] = zero
            eeta = np.float32(-np.float32(column_flat["rho"][column] * dew_flat[column]))
            output_flat["cst"][column] = np.float32(
                output_flat["cst"][column]
                + np.float32(
                    np.float32(
                        np.float32(timestep * dew_flat[column]) * ras_flat[column]
                    )
                    * column_flat["vegfrac"][column]
                )
            )
        else:
            edir1 = np.float32(
                -np.float32(
                    np.float32(
                        np.float32(
                            soilres_flat[column]
                            * np.float32(one - column_flat["vegfrac"][column])
                        )
                        * column_flat["qkms"][column]
                    )
                    * ras_flat[column]
                )
                * np.float32(column_flat["qvatm"][column] - qvg_flat[column])
            )
            ec1 = np.float32(
                np.float32(q1 * wet_flat[column])
                * column_flat["vegfrac"][column]
            )
            output_flat["edir1"][column] = edir1
            output_flat["ec1"][column] = ec1
            output_flat["cst"][column] = np.float32(
                max(
                    zero,
                    np.float32(
                        output_flat["cst"][column]
                        - np.float32(ec1 * timestep)
                    ),
                )
            )
            total_evaporation = np.float32(edir1 + ec1)
            total_evaporation = np.float32(total_evaporation + ett_flat[column])
            eeta = np.float32(total_evaporation * np.float32(1.0e3))
        output_flat["eeta"][column] = eeta
        output_flat["qfx"][column] = np.float32(xlv * eeta)
        output_flat["evapl"][column] = eeta
        soil_heat = np.float32(
            np.float32(
                np.float32(thdif_flat[0, column] * cap_flat[0, column])
                * dzstop
            )
            * np.float32(tso_flat[0, column] - tso_flat[1, column])
        )
        output_flat["s"][column] = soil_heat
        balance = np.float32(column_flat["rnet"][column] - hft)
        balance = np.float32(balance - np.float32(xlv * eeta))
        balance = np.float32(balance - soil_heat)
        balance = np.float32(balance - temperature.storage.reshape(ncolumn)[column])
        output_flat["fltot"][column] = balance

    for name, array in (
        ("soilmois", soilmois), ("tso", tso), ("smfrkeep", smfrkeep),
        ("keepfr", keepfr), ("soilice", phase["soilice"]),
        ("soiliqw", moisture.soiliqw), *output.items(),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"RUC soil produced non-finite {name}")
    return RucSoilStep(
        soilmois=soilmois,
        tso=tso,
        smfrkeep=smfrkeep,
        keepfr=keepfr,
        soilice=phase["soilice"],
        soiliqw=moisture.soiliqw,
        **output,
    )


def ruc_initialize_cold_start(
    tslb,
    smois,
    isltyp,
    ivgtyp,
    xice,
    *,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    parameters: RucParameterBundle | None = None,
) -> RucInitialization:
    """Transcribe WRF ``ruclsminit`` for a non-restart domain.

    Soil arrays use gpuwm's existing ``(soil_level, ...horizontal...)``
    order.  Inputs are copied to float32; this function never mutates caller
    arrays.  WRF restart behavior is deliberately outside this cold-start
    API because restart fields must be restored, not recomputed.
    """

    temperature = np.asarray(tslb, dtype=np.float32)
    total_water = np.asarray(smois, dtype=np.float32)
    if temperature.shape != total_water.shape:
        raise ValueError(
            f"tslb shape {temperature.shape} differs from smois "
            f"{total_water.shape}"
        )
    if temperature.ndim < 2 or temperature.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            f"RUC soil arrays must have shape ({NUM_SOIL_LAYERS}, ...), "
            f"got {temperature.shape}"
        )
    if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(total_water)):
        raise ValueError("RUC tslb and smois must be finite")
    if np.any((temperature < np.float32(170.0)) | (temperature > np.float32(400.0))):
        raise ValueError("RUC tslb must remain within WRF's 170..400 K preflight")
    if np.any((total_water < np.float32(0.0)) | (total_water > np.float32(1.0))):
        raise ValueError("RUC smois must remain within 0..1 m3 m-3")

    horizontal_shape = temperature.shape[1:]
    soil_type = _horizontal_integer_field(isltyp, horizontal_shape, "isltyp")
    vegetation_type = _horizontal_integer_field(
        ivgtyp, horizontal_shape, "ivgtyp"
    )
    ice_fraction = np.asarray(xice, dtype=np.float32)
    if ice_fraction.shape != horizontal_shape:
        raise ValueError(
            f"xice shape {ice_fraction.shape}; expected {horizontal_shape}"
        )
    if not np.all(np.isfinite(ice_fraction)) or np.any(
        (ice_fraction < np.float32(0.0)) | (ice_fraction > np.float32(1.0))
    ):
        raise ValueError("RUC xice must be finite and within 0..1")

    bundle = parameters or load_ruc_parameters()
    vegetation = bundle.vegetation_for(mminlu)
    if np.any((soil_type < 1) | (soil_type > len(bundle.soil.rows))):
        bad = int(soil_type[(soil_type < 1) | (soil_type > len(bundle.soil.rows))][0])
        raise ValueError(f"RUC isltyp {bad} is outside 1..{len(bundle.soil.rows)}")
    if np.any((vegetation_type < 1) | (vegetation_type > len(vegetation.rows))):
        bad = int(
            vegetation_type[
                (vegetation_type < 1) | (vegetation_type > len(vegetation.rows))
            ][0]
        )
        raise ValueError(
            f"RUC ivgtyp {bad} is outside 1..{len(vegetation.rows)} "
            f"for {vegetation.name}"
        )

    liquid = np.empty_like(total_water)
    frozen = np.empty_like(total_water)
    availability = np.empty(horizontal_shape, dtype=np.float32)
    roughness = np.empty(horizontal_shape, dtype=np.float32)

    t_flat = temperature.reshape(NUM_SOIL_LAYERS, -1)
    w_flat = total_water.reshape(NUM_SOIL_LAYERS, -1)
    liquid_flat = liquid.reshape(NUM_SOIL_LAYERS, -1)
    frozen_flat = frozen.reshape(NUM_SOIL_LAYERS, -1)
    soil_flat = soil_type.reshape(-1)
    vegetation_flat = vegetation_type.reshape(-1)
    ice_flat = ice_fraction.reshape(-1)
    availability_flat = availability.reshape(-1)
    roughness_flat = roughness.reshape(-1)

    riw = np.float32(np.float32(900.0) * np.float32(1.0e-3))
    xlmelt = np.float32(3.35e5)
    t_freeze = np.float32(273.15)
    gravity = np.float32(9.81)
    one = np.float32(1.0)
    minimum_availability = np.float32(0.00001)

    for column in range(soil_flat.size):
        soil = bundle.soil.rows[int(soil_flat[column]) - 1]
        veg = vegetation.rows[int(vegetation_flat[column]) - 1]
        bb = np.float32(soil.values[0])
        drysmc = np.float32(soil.values[1])
        maxsmc = np.float32(soil.values[3])
        refsmc = np.float32(soil.values[4])
        satpsi = np.float32(soil.values[5])
        dqm = np.float32(maxsmc - drysmc)
        psis = np.float32(-satpsi)
        roughness_flat[column] = np.float32(veg.z0)

        if ice_flat[column] > np.float32(0.0):
            frozen_flat[:, column] = one
            liquid_flat[:, column] = np.float32(0.0)
            availability_flat[column] = one
            continue
        if soil_flat[column] == 14:
            frozen_flat[:, column] = np.float32(0.0)
            liquid_flat[:, column] = one
            availability_flat[column] = one
            continue

        numerator = np.float32(w_flat[0, column] - drysmc)
        denominator = np.float32(refsmc - drysmc)
        raw_availability = np.float32(numerator / denominator)
        availability_flat[column] = np.float32(
            max(minimum_availability, min(one, raw_availability))
        )
        for level in range(NUM_SOIL_LAYERS):
            soil_temperature = t_flat[level, column]
            tln = np.float32(np.log(np.float32(soil_temperature / t_freeze)))
            if tln < np.float32(0.0):
                base = np.float32(xlmelt * np.float32(soil_temperature - t_freeze))
                base = np.float32(base / soil_temperature)
                base = np.float32(base / gravity)
                base = np.float32(base / psis)
                exponent = np.float32(np.float32(-one) / bb)
                equilibrium = np.float32(
                    np.float32(dqm + drysmc)
                    * np.float32(np.power(base, exponent))
                )
                equilibrium = np.float32(max(np.float32(0.0), equilibrium))
                equilibrium = np.float32(
                    min(equilibrium, w_flat[level, column])
                )
                liquid_flat[level, column] = equilibrium
                frozen_flat[level, column] = np.float32(
                    np.float32(w_flat[level, column] - equilibrium) / riw
                )
            else:
                frozen_flat[level, column] = np.float32(0.0)
                liquid_flat[level, column] = w_flat[level, column]

    return RucInitialization(
        sh2o=liquid,
        smfr3d=frozen,
        mavail=availability,
        znt=roughness,
    )


__all__ = [
    "RUC_TABLE_DIR",
    "RUC_ORACLE_DIR",
    "RUC_VEGETATION_DATASETS",
    "RUC_VEGETATION_FIELDS",
    "RUC_SOIL_PROPERTY_COLUMN_INPUTS",
    "RUC_SOIL_PROPERTY_PROFILE_INPUTS",
    "RUC_SOIL_MOISTURE_COLUMN_INPUTS",
    "RUC_SOIL_MOISTURE_PROFILE_INPUTS",
    "RUC_SOIL_TEMPERATURE_COLUMN_INPUTS",
    "RUC_SOIL_TEMPERATURE_PROFILE_INPUTS",
    "RUC_SOIL_STEP_COLUMN_INPUTS",
    "RUC_SOIL_STEP_PROFILE_INPUTS",
    "RucInitialization",
    "RucParameterBundle",
    "RucSoilProperties",
    "RucSoilMoisture",
    "RucSoilTemperature",
    "RucSoilStep",
    "RucSurfaceParameters",
    "RucTranspiration",
    "RucVegetationRow",
    "RucVegetationTable",
    "load_ruc_parameters",
    "parse_ruc_vegparm",
    "ruc_initialize_cold_start",
    "ruc_soil_geometry",
    "ruc_soil_properties",
    "ruc_soil_moisture_step",
    "ruc_soil_temperature_step",
    "ruc_soil_step",
    "ruc_saturation_table",
    "ruc_surface_parameters",
    "ruc_transpiration",
]


# --------------------------------------------------------------------------
# WRF v4.6.1 sfctmp snow preparation (phys/module_sf_ruclsm.F:1400-1766)
# --------------------------------------------------------------------------

# module_sf_ruclsm.F:63-69.  Only read when isncovr_opt==3.
RUC_SNOW_COVER_FACTOR = (
    0.030, 0.030, 0.030, 0.030, 0.030,
    0.016, 0.016, 0.020, 0.020, 0.020,
    0.020, 0.014, 0.042, 0.026, 0.030,
    0.016, 0.030, 0.030, 0.030, 0.030,
    0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000,
)
# module_sf_ruclsm.F:78 pins the snow-cover formulation at compile time.
RUC_SNOW_COVER_OPTION = 2


def _f32_exp(value) -> np.float32:
    """Single-rounded float32 ``exp``.  NOT glibc's ``expf``: a third function.

    ``numpy``'s float32 ``exp`` loop is not correctly rounded and lands up to
    2 ULP away from glibc's ``expf``.  That is invisible in most places, but
    ``sfctmp``'s snow compaction evaluates ``(exp(x)-1)/x`` for
    ``x ~ 2e-4``, where the cancellation amplifies a 1 ULP input error into
    ~7600 ULP of ``rhosn``.  So the numpy loop is not usable here.

    What this returns is a float64 ``exp`` rounded once.  An earlier version
    of this docstring claimed that reproduces "the correctly rounded float32
    result, which is what glibc delivers".  Both halves are wrong.  glibc
    2.39 is not correctly rounded -- its ``expf`` is a float32 reduction with
    its own error -- so rounding once yields a *third* function, not a more
    accurate one.  ``gpuwm/core/noahmp_libm.py`` measures the gap directly:
    the round-once shim misses glibc ``expf`` on 21,750 of 34,902,602
    arguments.  ``codex/ruc-snowsoil``'s ``b303e61`` retracted the identical
    claim on its own branch; this is the same retraction.

    It is kept because it is the closest available and because the CUDA half
    (``ruc_expf_glibc``) computes exactly the same thing, so host and device
    agree with each other.  ``noahmp_libm.expf`` is the verified glibc
    transcription and is where this should land; see
    ``_RUC_PROVISIONAL_TRANSCENDENTALS``.  Switching is a measurement, not an
    edit -- at ``x ~ 2e-4`` the compaction amplifies 1 ULP by ~7600, so if
    the two disagree here the fixture may be what is wrong.
    """

    return np.float32(np.exp(np.float64(value)))


def _f32_expm1(value) -> np.float32:
    """Single-rounded float32 ``expm1``.  NOT glibc's ``expm1f``.

    The previous docstring claimed this is "as glibc's ``expm1f`` gives".
    Measured counter-example: at ``x = 1.0`` this returns ``0x3FDBF0A9``
    where glibc returns ``0x3FDBF0A8``.  Against the 24 glibc words dumped
    in ``tests/test_mynn_pbl.py`` this shim misses 1; ``mynn_pbl._expm1f``,
    which spells out the fdlibm reduction, misses 0 and is the verified
    transcription.
    """

    return np.float32(np.expm1(np.float64(value)))


def _f32_tanh(value) -> np.float32:
    """The fdlibm ``s_tanhf.c`` reduction.  NOT glibc ``tanhf``.

    **This docstring used to claim "glibc ``tanhf`` bit for bit".  That is
    measurably false and the claim is retracted here.**  gfortran emits a call
    to ``tanhf`` for ``tanh`` on a default REAL, but glibc 2.39's ``tanhf`` is
    no longer fdlibm's ``expm1``-based reduction.  Rebuilding the fdlibm form
    from glibc's OWN ``expm1f`` gives ``0.760541856`` where glibc ``tanhf``
    returns ``0.760541916``, at ``x = 0.9974991083145142`` -- so the gap is
    the algorithm, not the ``expm1``.

    Measured over the arguments WRF ``:1520-1521`` reaches for air
    temperatures from 220 K to 310 K in 0.01 K steps (18,000 distinct float32
    arguments, both the ``0.15`` and the ``0.3333`` forms), this function
    misses glibc 2.39 ``tanhf`` on 231 of them -- 1.28% -- by up to 3 ULP.

    ``oracle/lsmruc.csv`` is the first RUC fixture to land on one of those
    arguments, which is why the snow lanes were bitwise without noticing.
    Twenty-five of that fixture's twenty-six residue cells are this function;
    substituting a measured glibc ``tanhf`` table collapses them to one cell of
    1 ULP.  See ``gpuwm/data/ruc/PROVENANCE.md``.

    Fixing it means transcribing glibc 2.39's actual ``tanhf`` here and
    identically in ``ruc.cu``, then re-verifying ``oracle/sfctmp_prep.csv`` and
    ``oracle/sfctmp.csv``.  Until then the deviation is pinned, not hidden:
    every fixture that reaches it fails closed on any cell that is not in its
    residue table.

    ``expm1`` here is a float64 ``expm1`` rounded once, which is what glibc's
    ``expm1f`` delivers on every argument this lane's fixtures reach.

    Non-finite input is out of contract; callers validate finiteness first.
    """

    x = np.float32(value)
    one = np.float32(1.0)
    two = np.float32(2.0)
    magnitude = np.float32(abs(x))
    if magnitude < np.float32(22.0):
        if magnitude < np.float32(2.0 ** -28):
            # tanh(tiny) == tiny, with the fdlibm inexact-flag form.
            return np.float32(x * np.float32(one + x))
        doubled = np.float32(two * magnitude)
        if magnitude >= one:
            t = _f32_expm1(doubled)
            z = np.float32(one - np.float32(two / np.float32(t + two)))
        else:
            t = _f32_expm1(np.float32(-doubled))
            z = np.float32(np.float32(-t) / np.float32(t + two))
    else:
        # fdlibm returns one-tiny here, which rounds to exactly one.
        z = one
    return z if x >= np.float32(0.0) else np.float32(-z)

RUC_SNOW_PREP_PROFILE_INPUTS = ("ts1d",)
RUC_SNOW_PREP_COLUMN_INPUTS = (
    "seaice",
    "gsw",
    "tabs",
    "tsnav",
    "prcpms",
    "newsnms",
    "vegfra",
    "lai",
    "sat",
    "soilt",
    "snowfallac",
    "alb_snow",
    "alb_snow_free",
    "snowrat",
    "grauprat",
    "icerat",
    "curat",
    "snwe",
    "snhei",
    "snowfrac",
    "rhosn",
    "rhosnfall",
    "cst",
    "alb",
    "emiss",
    "znt",
)
RUC_SNOW_PREP_PROFILE_OUTPUTS = ("tice", "rhosice", "capice", "thdifice")
RUC_SNOW_PREP_COLUMN_OUTPUTS = (
    "snhei_crit",
    "snhei_crit_newsn",
    "zntsn",
    "snow_mosaic",
    "snfr",
    "newsn",
    "newsnowratio",
    "snowfracnewsn",
    "rhonewsn",
    "smelt",
    "rainf",
    "rsm",
    "dd1",
    "infiltr",
    "vegfrac",
    "drip",
    "dripsn",
    "dripliq",
    "smf",
    "interw",
    "intersn",
    "infwater",
    "intwratio",
    "gswnew",
    "gswin",
    "albice",
    "albsn",
    "emissn",
    "emiss_snowfree",
    "keep_snow_albedo",
    "snowfrac2",
    "snwe",
    "snhei",
    "snowfrac",
    "rhosn",
    "rhosnfall",
    "cst",
    "alb",
    "emiss",
    "znt",
)


@dataclass(frozen=True)
class RucSnowPreparation:
    """State left by WRF ``sfctmp``'s snow-preparation block.

    Every field is written by ``phys/module_sf_ruclsm.F:1400-1766``, the
    straight-line prologue that runs before ``sfctmp`` dispatches to
    ``soil``/``snowsoil``/``sice``/``snowseaice``.  Names match the Fortran
    exactly; ``iland`` is the only integer.

    ``keep_snow_albedo`` and ``snowfrac2`` are locals WRF only assigns inside
    the ``snhei>0.`` branch.  When that branch does not run WRF leaves them
    undefined and nothing reads them, so this result reports 0 rather than
    stack residue -- the one value here that is a defined-behaviour choice
    instead of a transcription.
    """

    tice: np.ndarray
    rhosice: np.ndarray
    capice: np.ndarray
    thdifice: np.ndarray
    snhei_crit: np.ndarray
    snhei_crit_newsn: np.ndarray
    zntsn: np.ndarray
    snow_mosaic: np.ndarray
    snfr: np.ndarray
    newsn: np.ndarray
    newsnowratio: np.ndarray
    snowfracnewsn: np.ndarray
    rhonewsn: np.ndarray
    smelt: np.ndarray
    rainf: np.ndarray
    rsm: np.ndarray
    dd1: np.ndarray
    infiltr: np.ndarray
    vegfrac: np.ndarray
    drip: np.ndarray
    dripsn: np.ndarray
    dripliq: np.ndarray
    smf: np.ndarray
    interw: np.ndarray
    intersn: np.ndarray
    infwater: np.ndarray
    intwratio: np.ndarray
    gswnew: np.ndarray
    gswin: np.ndarray
    albice: np.ndarray
    albsn: np.ndarray
    emissn: np.ndarray
    emiss_snowfree: np.ndarray
    keep_snow_albedo: np.ndarray
    snowfrac2: np.ndarray
    snwe: np.ndarray
    snhei: np.ndarray
    snowfrac: np.ndarray
    rhosn: np.ndarray
    rhosnfall: np.ndarray
    cst: np.ndarray
    alb: np.ndarray
    emiss: np.ndarray
    znt: np.ndarray
    iland: np.ndarray


def ruc_snow_preparation(
    values: Mapping[str, object],
    *,
    delt: float,
    ivgtyp,
    iland,
    isice: int = 15,
    c1sn: float = 0.026,
    c2sn: float = 21.0,
    isncovr_opt: int = RUC_SNOW_COVER_OPTION,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    bundle: RucParameterBundle | None = None,
) -> RucSnowPreparation:
    """Transcribe WRF ``sfctmp``'s snow-preparation block.

    ``phys/module_sf_ruclsm.F:1400-1766``: the straight-line prologue of
    ``sfctmp``.  It zeroes the per-step accumulators, builds the Zubov
    sea-ice column (``:1471-1485``), ages the snow pack (``:1493-1501``),
    mixes freshly fallen snow/graupel/ice into the pack density
    (``:1508-1540``), intercepts precipitation in the canopy
    (``:1553-1598``), and then -- only when snow survives -- sets the snow
    fraction, the mosaic flag, the roughness blend and the snow albedo and
    emissivity (``:1600-1766``).

    This routine stops immediately before ``:1767``, where ``sfctmp`` begins
    dispatching to ``soil``, ``snowsoil``, ``sice`` and ``snowseaice``; the
    dispatch and the mosaic recombination that follows it are not part of
    this port.

    ``rnet`` and the mosaic ``gswnew`` are formed inside the dispatch, so the
    ``gswnew`` returned here is WRF's ``:1460`` value, the raw absorbed solar
    flux.

    Arguments WRF passes to ``sfctmp`` but the preparation block never reads
    are omitted: ``conflx``, ``nroot``, ``meltfactor``, ``isltyp``,
    ``xland``, ``patm``, ``qvatm``, ``qcatm``, ``rho``, ``glw``, ``qkms``,
    ``tkms``, ``pc``, ``mavail``, every soil-property scalar except ``sat``,
    the solver geometry, ``tbq`` and every constant except ``c1sn``/``c2sn``.
    The incoming ``rhonewsn`` is dead too: ``:1428`` overwrites it before any
    read.

    ``isncovr_opt`` is ``integer, parameter :: isncovr_opt=2`` at
    ``module_sf_ruclsm.F:78``, so only option 2 can be exercised against the
    pinned build.  Options 1 and 3 are transcribed for completeness and are
    NOT oracle-verified.
    """

    timestep = np.float32(delt)
    density_a = np.float32(c1sn)
    density_b = np.float32(c2sn)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC snow preparation delt must be finite and positive")
    if not np.isfinite(density_a) or not np.isfinite(density_b):
        raise ValueError("RUC snow preparation c1sn/c2sn must be finite")
    if isncovr_opt not in (1, 2, 3):
        raise ValueError("RUC isncovr_opt must be 1, 2, or 3")
    if type(isice) is not int:
        raise TypeError("RUC snow preparation isice must be an int")
    missing = [
        name
        for name in RUC_SNOW_PREP_PROFILE_INPUTS + RUC_SNOW_PREP_COLUMN_INPUTS
        if name not in values
    ]
    if missing:
        raise TypeError(
            f"missing RUC snow preparation inputs: {', '.join(missing)}"
        )

    parameters = bundle if bundle is not None else load_ruc_parameters()
    vegetation = parameters.vegetation_for(mminlu)
    urban = int(vegetation.scalars["URBAN"])
    roughness_table = np.asarray(
        [row.z0 for row in vegetation.rows], dtype=np.float32
    )
    emissivity_table = np.asarray(
        [row.lemi for row in vegetation.rows], dtype=np.float32
    )
    cover_factor = np.asarray(RUC_SNOW_COVER_FACTOR, dtype=np.float32)

    profile = np.asarray(values["ts1d"], dtype=np.float32)
    if profile.ndim < 2 or profile.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC snow preparation ts1d must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {profile.shape}"
        )
    if not np.all(np.isfinite(profile)):
        raise ValueError("ts1d must be finite")
    shape = profile.shape
    horizontal_shape = shape[1:]
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_PREP_COLUMN_INPUTS
    }
    vegetation_category = _horizontal_integer_field(
        ivgtyp, horizontal_shape, "ivgtyp"
    )
    land_category = _horizontal_integer_field(iland, horizontal_shape, "iland")
    for name, category in (
        ("ivgtyp", vegetation_category), ("iland", land_category)
    ):
        if np.any((category < 1) | (category > len(vegetation.rows))):
            raise ValueError(
                f"RUC {name} is outside 1..{len(vegetation.rows)}"
            )
    if not 1 <= isice <= len(vegetation.rows):
        raise ValueError(f"RUC isice is outside 1..{len(vegetation.rows)}")
    if np.any(columns["rhosn"] <= np.float32(0.0)):
        raise ValueError("RUC snow preparation rhosn must be positive")
    if np.any(columns["alb"] >= np.float32(1.0)):
        raise ValueError("RUC snow preparation alb must be below 1")

    ncolumn = int(np.prod(horizontal_shape))
    profile_flat = profile.reshape(NUM_SOIL_LAYERS, ncolumn)
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    veg_flat = vegetation_category.reshape(ncolumn)
    land_flat = land_category.reshape(ncolumn)

    profiles = {
        name: np.zeros(shape, dtype=np.float32)
        for name in RUC_SNOW_PREP_PROFILE_OUTPUTS
    }
    profile_out = {
        name: array.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, array in profiles.items()
    }
    outputs = {
        name: np.empty(horizontal_shape, dtype=np.float32)
        for name in RUC_SNOW_PREP_COLUMN_OUTPUTS
    }
    output_flat = {name: array.reshape(ncolumn) for name, array in outputs.items()}
    land_result = np.empty(horizontal_shape, dtype=np.int32)
    land_result_flat = land_result.reshape(ncolumn)

    zero = np.float32(0.0)
    one = np.float32(1.0)
    rhowater = np.float32(1000.0)
    # module_sf_ruclsm.F:1418-1419 and :1493 fold their leading products at
    # compile time in default REAL, so they are folded here the same way.
    critical_depth_coefficient = np.float32(np.float32(0.01601) * rhowater)
    new_snow_depth_coefficient = np.float32(np.float32(0.0005) * rhowater)
    aging_depth_coefficient = np.float32(np.float32(0.0081) * np.float32(1.0e3))
    freeze = np.float32(273.15)
    snow_emissivity = np.float32(0.98)
    # (273.15-263.15) in the :1735/:1761 albedo forms.
    albedo_span = np.float32(np.float32(273.15) - np.float32(263.15))

    for column in range(ncolumn):
        seaice = column_flat["seaice"][column]
        gsw = column_flat["gsw"][column]
        tabs = column_flat["tabs"][column]
        tsnav = column_flat["tsnav"][column]
        prcpms = column_flat["prcpms"][column]
        newsnms = column_flat["newsnms"][column]
        vegfra = column_flat["vegfra"][column]
        lai = column_flat["lai"][column]
        sat = column_flat["sat"][column]
        soilt = column_flat["soilt"][column]
        snowfallac = column_flat["snowfallac"][column]
        alb_snow = column_flat["alb_snow"][column]
        alb_snow_free = column_flat["alb_snow_free"][column]
        snowrat = column_flat["snowrat"][column]
        grauprat = column_flat["grauprat"][column]
        icerat = column_flat["icerat"][column]
        curat = column_flat["curat"][column]
        snwe = column_flat["snwe"][column]
        snhei = column_flat["snhei"][column]
        snowfrac = column_flat["snowfrac"][column]
        rhosn = column_flat["rhosn"][column]
        rhosnfall = column_flat["rhosnfall"][column]
        cst = column_flat["cst"][column]
        alb = column_flat["alb"][column]
        emiss = column_flat["emiss"][column]
        znt = column_flat["znt"][column]
        ivgtyp_column = int(veg_flat[column])
        iland_column = int(land_flat[column])

        # :1418-1419
        snhei_crit = np.float32(critical_depth_coefficient / rhosn)
        snhei_crit_newsn = np.float32(new_snow_depth_coefficient / rhosn)
        # :1421 - assigned and never read again by sfctmp.
        zntsn = roughness_table[isice - 1]
        # :1423-1434
        snow_mosaic = zero
        snfr = one
        newsn = zero
        newsnowratio = zero
        snowfracnewsn = zero
        rhonewsn = np.float32(100.0)
        if snhei == zero:
            snowfrac = zero
        smelt = zero
        rainf = zero
        rsm = zero
        dd1 = zero
        infiltr = zero
        # :1442-1449
        vegfrac = np.float32(np.float32(0.01) * vegfra)
        drip = zero
        dripsn = zero
        dripliq = zero
        smf = zero
        interw = zero
        intersn = zero
        infwater = zero
        # :1452-1458 - the ice column starts at zero on every point.
        for level in range(NUM_SOIL_LAYERS):
            profile_out["tice"][level, column] = zero
            profile_out["rhosice"][level, column] = zero
            profile_out["capice"][level, column] = zero
            profile_out["thdifice"][level, column] = zero
        # :1460-1465
        gswnew = gsw
        gswin = np.float32(gsw / np.float32(one - alb))
        albice = alb_snow_free
        albsn = alb_snow
        emissn = snow_emissivity
        emiss_snowfree = emissivity_table[ivgtyp_column - 1]

        # :1471-1485 sea-ice column properties (Zubov) and the ice albedo.
        if seaice >= np.float32(0.5):
            for level in range(NUM_SOIL_LAYERS):
                tice = np.float32(profile_flat[level, column] - freeze)
                rhosice = np.float32(
                    np.float32(917.6)
                    / np.float32(one - np.float32(np.float32(0.000165) * tice))
                )
                cice = np.float32(
                    np.float32(2115.85) + np.float32(np.float32(7.7948) * tice)
                )
                capice = np.float32(cice * rhosice)
                profile_out["tice"][level, column] = tice
                profile_out["rhosice"][level, column] = rhosice
                profile_out["capice"][level, column] = capice
                profile_out["thdifice"][level, column] = np.float32(
                    np.float32(2.260872) / capice
                )
            surface_tice = profile_out["tice"][0, column]
            albice = min(
                alb_snow_free,
                max(
                    np.float32(alb_snow_free - np.float32(0.05)),
                    np.float32(
                        alb_snow_free
                        - np.float32(
                            np.float32(
                                np.float32(0.1)
                                * np.float32(surface_tice + np.float32(10.0))
                            )
                            / np.float32(10.0)
                        )
                    ),
                ),
            )

        # :1493-1501 Koren et al. (1999) compaction of the existing pack.
        if snhei > np.float32(aging_depth_coefficient / rhosn):
            bsn = np.float32(
                np.float32(
                    np.float32(timestep / np.float32(3600.0)) * density_a
                )
                * _f32_exp(np.float32(
                    np.float32(np.float32(0.08) * min(zero, tsnav))
                    - np.float32(
                        np.float32(density_b * rhosn) * np.float32(1.0e-3)
                    )
                ))
            )
            compaction = np.float32(
                np.float32(bsn * snwe) * np.float32(100.0)
            )
            # :1496 - goto 777 leaves rhosn untouched.
            if not compaction < np.float32(1.0e-4):
                xsn = np.float32(
                    np.float32(
                        rhosn
                        * np.float32(_f32_exp(compaction) - one)
                    )
                    / compaction
                )
                rhosn = min(max(np.float32(58.8), xsn), np.float32(500.0))

        # :1504 - the mosaic flag from the previous step's snow fraction.
        if snowfrac < np.float32(0.75):
            snow_mosaic = one

        # :1506-1540 fresh snowfall and its density.
        newsn = np.float32(newsnms * timestep)
        if newsn > zero:
            newsnowratio = min(
                one, np.float32(newsn / np.float32(snwe + newsn))
            )
            rhonewsn = min(
                np.float32(125.0),
                np.float32(
                    np.float32(1000.0)
                    / max(
                        np.float32(8.0),
                        np.float32(
                            np.float32(17.0)
                            * _f32_tanh(np.float32(
                                np.float32(np.float32(276.65) - tabs)
                                * np.float32(0.15)
                            ))
                        ),
                    )
                ),
            )
            rhonewgr = min(
                np.float32(500.0),
                np.float32(
                    rhowater
                    / max(
                        np.float32(2.0),
                        np.float32(
                            np.float32(3.5)
                            * _f32_tanh(np.float32(
                                np.float32(np.float32(274.15) - tabs)
                                * np.float32(0.3333)
                            ))
                        ),
                    )
                ),
            )
            rhonewice = rhonewsn
            weighted = np.float32(rhonewsn * snowrat)
            weighted = np.float32(weighted + np.float32(rhonewgr * grauprat))
            weighted = np.float32(weighted + np.float32(rhonewice * icerat))
            weighted = np.float32(weighted + np.float32(rhonewgr * curat))
            rhosnfall = min(
                np.float32(500.0), max(np.float32(58.8), weighted)
            )
            # :1531 - from here rhonewsn is the density of the falling
            # frozen precipitation, not of fresh snow alone.
            rhonewsn = rhosnfall
            xsn = np.float32(
                np.float32(
                    np.float32(rhosn * snwe) + np.float32(rhonewsn * newsn)
                )
                / np.float32(snwe + newsn)
            )
            rhosn = min(max(np.float32(58.8), xsn), np.float32(500.0))

        # :1542-1551
        if prcpms != zero:
            rainf = one

        # :1553-1578 canopy interception (Lawrence et al. 2006 CLM eq. 1).
        drip = zero
        intwratio = zero
        if vegfrac > np.float32(0.01):
            canopy_gap = np.float32(
                one
                - _f32_exp(np.float32(np.float32(-0.5) * lai))
            )
            interw = np.float32(
                np.float32(
                    np.float32(
                        np.float32(np.float32(0.25) * timestep) * prcpms
                    )
                    * canopy_gap
                )
                * vegfrac
            )
            intersn = np.float32(
                np.float32(
                    np.float32(np.float32(0.25) * newsn) * canopy_gap
                )
                * vegfrac
            )
            infwater = np.float32(prcpms - np.float32(interw / timestep))
            intercepted = np.float32(interw + intersn)
            if intercepted > zero:
                intwratio = np.float32(interw / intercepted)
            dd1 = np.float32(np.float32(cst + interw) + intersn)
            cst = dd1
            if cst > sat:
                cst = sat
                drip = np.float32(dd1 - sat)
        else:
            cst = zero
            drip = zero
            interw = zero
            intersn = zero
            infwater = prcpms

        # :1580-1598 fresh snow onto the ground.
        if newsn > zero:
            snwe = max(zero, np.float32(np.float32(snwe + newsn) - intersn))
            if drip > zero:
                if snow_mosaic == one:
                    dripliq = np.float32(drip * intwratio)
                    dripsn = np.float32(drip - dripliq)
                    snwe = np.float32(snwe + dripsn)
                    infwater = np.float32(infwater + dripliq)
                    dripliq = zero
                    dripsn = zero
                else:
                    snwe = np.float32(snwe + drip)
            snhei = np.float32(np.float32(snwe * rhowater) / rhosn)
            newsn = np.float32(np.float32(newsn * rhowater) / rhonewsn)

        # WRF leaves both undefined when the snhei>0. branch does not run.
        keep_snow_albedo = zero
        snowfrac2 = zero

        if snhei > zero:
            # :1603 - snow-covered points use the snow/ice land-use class.
            iland_column = isice
            if isncovr_opt == 1:
                snowfrac = min(
                    one,
                    np.float32(
                        snhei / np.float32(np.float32(2.0) * snhei_crit)
                    ),
                )
            elif isncovr_opt == 2:
                snowfrac = min(
                    one,
                    np.float32(
                        snhei / np.float32(np.float32(2.0) * snhei_crit)
                    ),
                )
                # (rhosn/rhonewsn)**1. is the quotient itself; a real
                # exponent of exactly 1 is an identity in both gfortran and
                # CUDA, so no pow is issued.
                snowfrac2 = _f32_tanh(np.float32(
                    snhei
                    / np.float32(
                        np.float32(
                            np.float32(2.5)
                            * min(np.float32(0.2), znt)
                        )
                        * np.float32(rhosn / rhonewsn)
                    )
                ))
                snowfrac = np.float32(
                    np.float32(0.5) * np.float32(snowfrac + snowfrac2)
                )
            else:
                # :1633 - m is the Noah-MP facsnf exponent, held at 1.
                snowfrac = _f32_tanh(np.float32(
                    snhei
                    / np.float32(
                        np.float32(
                            np.float32(10.0)
                            * cover_factor[ivgtyp_column - 1]
                        )
                        * np.float32(rhosn / rhonewsn)
                    )
                ))
            if newsn > zero:
                snowfracnewsn = min(
                    one,
                    np.float32(
                        np.float32(snowfallac * np.float32(1.0e-3))
                        / snhei_crit_newsn
                    ),
                )
            # :1645
            if ivgtyp_column == urban:
                snowfrac = min(np.float32(0.75), snowfrac)
            # :1656
            if snowfrac < np.float32(0.75):
                snow_mosaic = one
            # :1658-1663
            keep_snow_albedo = zero
            if snowfracnewsn > np.float32(0.99) and rhosnfall < np.float32(450.0):
                keep_snow_albedo = one
                snow_mosaic = zero
            # :1672-1680 roughness blend toward the snow/ice class.
            if (
                newsn == zero
                and znt <= np.float32(0.2)
                and ivgtyp_column != isice
            ):
                snow_roughness = roughness_table[iland_column - 1]
                if snhei <= np.float32(np.float32(2.0) * znt):
                    znt = np.float32(
                        np.float32(np.float32(0.55) * znt)
                        + np.float32(np.float32(0.45) * snow_roughness)
                    )
                elif (
                    snhei > np.float32(np.float32(2.0) * znt)
                    and snhei <= np.float32(np.float32(4.0) * znt)
                ):
                    znt = np.float32(
                        np.float32(np.float32(0.2) * znt)
                        + np.float32(np.float32(0.8) * snow_roughness)
                    )
                elif snhei > np.float32(np.float32(4.0) * znt):
                    znt = snow_roughness

            if seaice < np.float32(0.5):
                # :1682-1737 snow on soil.
                if snow_mosaic == one:
                    albsn = alb_snow
                    # :1690-1696 is unreachable: :1662 clears snow_mosaic
                    # whenever keep_snow_albedo is set, so this test can
                    # never pass.  Transcribed to stay faithful.
                    if (
                        keep_snow_albedo > np.float32(0.9)
                        and albsn < np.float32(0.4)
                    ):
                        albsn = np.float32(0.7)
                    emiss = emissn
                else:
                    albsn = max(
                        np.float32(keep_snow_albedo * alb_snow),
                        min(
                            np.float32(
                                alb_snow_free
                                + np.float32(
                                    np.float32(alb_snow - alb_snow_free)
                                    * snowfrac
                                )
                            ),
                            alb_snow,
                        ),
                    )
                    if (
                        newsn > zero
                        and keep_snow_albedo > np.float32(0.9)
                        and albsn < np.float32(0.4)
                    ):
                        albsn = np.float32(0.7)
                    emiss = max(
                        np.float32(keep_snow_albedo * emissn),
                        min(
                            np.float32(
                                emiss_snowfree
                                + np.float32(
                                    np.float32(emissn - emiss_snowfree)
                                    * snowfrac
                                )
                            ),
                            emissn,
                        ),
                    )
                if albsn < np.float32(0.4) or keep_snow_albedo == one:
                    alb = albsn
                else:
                    alb = min(
                        albsn,
                        max(
                            np.float32(
                                albsn
                                - np.float32(
                                    np.float32(
                                        np.float32(
                                            np.float32(0.1)
                                            * np.float32(
                                                soilt - np.float32(263.15)
                                            )
                                        )
                                        / albedo_span
                                    )
                                    * albsn
                                )
                            ),
                            np.float32(albsn - np.float32(0.05)),
                        ),
                    )
            else:
                # :1738-1765 snow on ice.
                if snow_mosaic == one:
                    albsn = alb_snow
                    emiss = emissn
                else:
                    albsn = max(
                        np.float32(keep_snow_albedo * alb_snow),
                        min(
                            np.float32(
                                albice
                                + np.float32(
                                    np.float32(alb_snow - albice) * snowfrac
                                )
                            ),
                            alb_snow,
                        ),
                    )
                    emiss = max(
                        np.float32(keep_snow_albedo * emissn),
                        min(
                            np.float32(
                                emiss_snowfree
                                + np.float32(
                                    np.float32(emissn - emiss_snowfree)
                                    * snowfrac
                                )
                            ),
                            emissn,
                        ),
                    )
                if albsn < alb_snow or keep_snow_albedo == one:
                    alb = albsn
                else:
                    alb = min(
                        albsn,
                        max(
                            np.float32(
                                albsn
                                - np.float32(
                                    np.float32(
                                        np.float32(
                                            np.float32(0.15) * albsn
                                        )
                                        * np.float32(
                                            soilt - np.float32(263.15)
                                        )
                                    )
                                    / albedo_span
                                )
                            ),
                            np.float32(albsn - np.float32(0.1)),
                        ),
                    )

        results = {
            "snhei_crit": snhei_crit,
            "snhei_crit_newsn": snhei_crit_newsn,
            "zntsn": zntsn,
            "snow_mosaic": snow_mosaic,
            "snfr": snfr,
            "newsn": newsn,
            "newsnowratio": newsnowratio,
            "snowfracnewsn": snowfracnewsn,
            "rhonewsn": rhonewsn,
            "smelt": smelt,
            "rainf": rainf,
            "rsm": rsm,
            "dd1": dd1,
            "infiltr": infiltr,
            "vegfrac": vegfrac,
            "drip": drip,
            "dripsn": dripsn,
            "dripliq": dripliq,
            "smf": smf,
            "interw": interw,
            "intersn": intersn,
            "infwater": infwater,
            "intwratio": intwratio,
            "gswnew": gswnew,
            "gswin": gswin,
            "albice": albice,
            "albsn": albsn,
            "emissn": emissn,
            "emiss_snowfree": emiss_snowfree,
            "keep_snow_albedo": keep_snow_albedo,
            "snowfrac2": snowfrac2,
            "snwe": snwe,
            "snhei": snhei,
            "snowfrac": snowfrac,
            "rhosn": rhosn,
            "rhosnfall": rhosnfall,
            "cst": cst,
            "alb": alb,
            "emiss": emiss,
            "znt": znt,
        }
        for name, value in results.items():
            output_flat[name][column] = value
        land_result_flat[column] = iland_column

    for name, array in (*profiles.items(), *outputs.items()):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"RUC snow preparation produced non-finite {name}")
    return RucSnowPreparation(**profiles, **outputs, iland=land_result)


__all__ += [
    "RUC_SNOW_COVER_FACTOR",
    "RUC_SNOW_COVER_OPTION",
    "RUC_SNOW_PREP_COLUMN_INPUTS",
    "RUC_SNOW_PREP_COLUMN_OUTPUTS",
    "RUC_SNOW_PREP_PROFILE_INPUTS",
    "RUC_SNOW_PREP_PROFILE_OUTPUTS",
    "RucSnowPreparation",
    "ruc_snow_preparation",
]


# ---------------------------------------------------------------------------
# WRF v4.6.1 ``phys/module_sf_ruclsm.F:3789-4526`` subroutine ``snowseaice``:
# the snow energy budget plus coupled snow/sea-ice heat diffusion.
# ---------------------------------------------------------------------------

RUC_SNOW_SEA_ICE_PROFILE_INPUTS = (
    "capice",
    "thdifice",
    "tso",
)
RUC_SNOW_SEA_ICE_COLUMN_INPUTS = (
    "meltfactor",
    "rhonewsn",
    "prcpms",
    "rainf",
    "newsnow",
    "snhei",
    "snwe",
    "snowfrac",
    "rhosn",
    "patm",
    "qvatm",
    "emiss",
    "rnet",
    "qkms",
    "tkms",
    "rho",
    "alb",
    "znt",
    "tabs",
    "soilt",
    "soilt1",
    "tsnav",
    "qvg",
    "qsg",
    "snom",
    "s",
)
RUC_SNOW_SEA_ICE_INTEGER_INPUTS = ("ilnb",)


@dataclass(frozen=True)
class RucSnowSeaIceStep:
    """Snow-on-sea-ice column state and fluxes from WRF ``snowseaice``.

    Every ``intent(inout)`` and ``intent(out)`` argument of the subroutine is
    returned.  ``snweprint``/``snheiprint`` are WRF's diagnostic copies of
    ``snwe``/``snhei`` taken before the final ``snhei`` update at
    ``module_sf_ruclsm.F:4486``, so they are not redundant with ``snwe`` and
    ``snhei``.
    """

    tso: np.ndarray
    ilnb: np.ndarray
    snweprint: np.ndarray
    snheiprint: np.ndarray
    rsm: np.ndarray
    dew: np.ndarray
    soilt: np.ndarray
    soilt1: np.ndarray
    tsnav: np.ndarray
    qvg: np.ndarray
    qsg: np.ndarray
    qcg: np.ndarray
    smelt: np.ndarray
    snoh: np.ndarray
    snflx: np.ndarray
    snom: np.ndarray
    eeta: np.ndarray
    qfx: np.ndarray
    hfx: np.ndarray
    s: np.ndarray
    sublim: np.ndarray
    prcpl: np.ndarray
    fltot: np.ndarray
    snwe: np.ndarray
    snhei: np.ndarray
    rhosn: np.ndarray
    emiss: np.ndarray
    alb: np.ndarray
    znt: np.ndarray


RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS = (
    "snweprint",
    "snheiprint",
    "rsm",
    "dew",
    "soilt",
    "soilt1",
    "tsnav",
    "qvg",
    "qsg",
    "qcg",
    "smelt",
    "snoh",
    "snflx",
    "snom",
    "eeta",
    "qfx",
    "hfx",
    "s",
    "sublim",
    "prcpl",
    "fltot",
    "snwe",
    "snhei",
    "rhosn",
    "emiss",
    "alb",
    "znt",
)


def ruc_snow_sea_ice_step(
    values: Mapping[str, object],
    *,
    delt: float,
    conflx: float = 40.0,
    myj: bool = False,
    cw: float = 4.183e6,
    xlv: float = 2.5e6,
) -> RucSnowSeaIceStep:
    """Transcribe WRF ``snowseaice``: snow over a nine-level sea-ice column.

    One, two or blended snow layers sit on the ice column; ``vilka`` closes
    the skin energy balance, a single non-iterated melt pass follows (WRF
    disables ``snowseaice``'s second ``vilka`` iteration at
    ``module_sf_ruclsm.F:4374``), and every ice temperature is clipped at the
    271.4 K melting cap.

    Arguments WRF passes but ``snowseaice`` never reads are omitted from this
    signature: ``snhei_crit``, ``qcatm``, ``gsw``, ``tice``, ``rhosice`` and
    ``dtdzs2`` are dead, ``glw`` reaches only the local ``xinet`` which is
    assigned and never used, ``ktau``/``i``/``j``/``iland``/``isoil`` only
    reach ``vilka``'s error print, and the incoming ``qcg`` is overwritten
    with zero at ``:4197`` before it is read.  ``cp``, ``rovcp`` and
    ``stbolt`` are pinned to the ``module_model_constants`` values the RUC
    lane fixes; ``xlv`` and ``cw`` stay adjustable because ``snowseaice``
    reads them into ``xlvm`` and ``cvw``.

    Both ``snhei`` and ``snwe`` are required: ``:4004`` rebuilds ``snhei``
    from ``snwe``, but ``:3950-3951`` has already used the *incoming*
    ``snhei`` to decide whether to halve ``deltsn``.
    """

    timestep = np.float32(delt)
    constant_flux_depth = _ruc_constant_flux_depth(conflx, "snowseaice")
    water_heat_capacity = np.float32(cw)
    vaporization = np.float32(xlv)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC snowseaice delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC snowseaice cw must be finite and positive")
    if not np.isfinite(vaporization) or vaporization <= 0.0:
        raise ValueError("RUC snowseaice xlv must be finite and positive")
    if not isinstance(myj, bool):
        raise TypeError("RUC snowseaice myj must be a bool")
    required = (
        RUC_SNOW_SEA_ICE_PROFILE_INPUTS
        + RUC_SNOW_SEA_ICE_COLUMN_INPUTS
        + RUC_SNOW_SEA_ICE_INTEGER_INPUTS
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(
            f"missing RUC snowseaice inputs: {', '.join(missing)}"
        )

    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SNOW_SEA_ICE_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC snowseaice profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_SEA_ICE_COLUMN_INPUTS
    }
    integers = {
        name: _horizontal_integer_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_SEA_ICE_INTEGER_INPUTS
    }
    if np.any(profiles["thdifice"][0] <= np.float32(0.0)):
        raise ValueError("RUC snowseaice top-level thdifice must be positive")
    if np.any(profiles["capice"][0] <= np.float32(0.0)):
        raise ValueError("RUC snowseaice top-level capice must be positive")
    if np.any(columns["patm"] <= np.float32(0.0)):
        raise ValueError("RUC snowseaice patm must be positive")
    if np.any(columns["rho"] <= np.float32(0.0)):
        raise ValueError("RUC snowseaice rho must be positive")
    if np.any(columns["rhosn"] <= np.float32(0.0)):
        raise ValueError("RUC snowseaice rhosn must be positive")
    if np.any(columns["snwe"] < np.float32(0.0)):
        raise ValueError("RUC snowseaice snwe must be nonnegative")

    zsmain, zshalf, dtdzs, _ = _ruc_soil_solver_geometry(timestep)
    table = _load_ruc_saturation_table()
    ncolumn = int(np.prod(horizontal_shape))
    profile_flat = {
        name: array.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, array in profiles.items()
    }
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    integer_flat = {name: array.reshape(ncolumn) for name, array in integers.items()}
    temperature = np.array(profiles["tso"], dtype=np.float32, copy=True)
    temperature_flat = temperature.reshape(NUM_SOIL_LAYERS, ncolumn)
    layer_count = np.empty(horizontal_shape, dtype=np.int32)
    layer_count_flat = layer_count.reshape(ncolumn)
    outputs = {
        name: np.empty(horizontal_shape, dtype=np.float32)
        for name in RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS
    }
    output_flat = {name: array.reshape(ncolumn) for name, array in outputs.items()}

    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    two = np.float32(2.0)
    cp_air = np.float32(1004.5)
    rovcp = np.float32(np.float32(287.0) / cp_air)
    stbolt = np.float32(5.67051e-8)
    freeze = np.float32(273.15)
    ice_cap = np.float32(271.4)
    # module_sf_ruclsm.F:3930 latent heat of fusion, :3932 sublimation.
    xlmelt = np.float32(3.35e5)
    xlvm = np.float32(vaporization + xlmelt)
    cvw = water_heat_capacity
    thousand = np.float32(1.0e3)
    milli = np.float32(1.0e-3)
    snow_heat = np.float32(2090.0)
    snow_cond = np.float32(0.265)
    # module_sf_ruclsm.F:3945-3946 two-layer and blending depth thresholds.
    deltsn_scale = np.float32(np.float32(0.05) * thousand)
    snth_scale = np.float32(np.float32(0.01) * thousand)
    # module_sf_ruclsm.F:4318 Egglston melt-rate limit.
    egglston = np.float32(5.6e-8)
    # module_sf_ruclsm.F:4341 Koren et al. (1999) retained-liquid fraction.
    rsm_low = np.float32(0.08)
    rsm_high = np.float32(0.18)
    rsm_depth = np.float32(0.10)
    rsm_share = np.float32(0.13)
    rsm_snhei = np.float32(0.01)
    rhosn_low = np.float32(58.8)
    rhosn_high = np.float32(500.0)
    epdt_floor = np.float32(1.0e-8)
    # (p1000mb*0.00001) from the :4438 Exner factor, folded in float32.
    reference = np.float32(np.float32(100000.0) * np.float32(0.00001))
    # module_sf_ruclsm.F:4519-4521 bare sea-ice surface properties.
    bare_emiss = np.float32(0.98)
    bare_znt = np.float32(0.011)
    bare_alb = np.float32(0.55)

    # ``conflx`` may be one depth for the whole field or one per column;
    # bind it to the column axis once rather than at every read.
    flux_depth = np.broadcast_to(constant_flux_depth, (ncolumn,))

    for column in range(ncolumn):
        rhosn = column_flat["rhosn"][column]
        rhonewsn = column_flat["rhonewsn"][column]
        snhei = column_flat["snhei"][column]
        snwe = column_flat["snwe"][column]
        soilt = column_flat["soilt"][column]
        soilt1 = column_flat["soilt1"][column]
        tsnav = column_flat["tsnav"][column]
        qvg = column_flat["qvg"][column]
        qsg = column_flat["qsg"][column]
        emiss = column_flat["emiss"][column]
        alb = column_flat["alb"][column]
        znt = column_flat["znt"][column]
        snom = column_flat["snom"][column]
        storage = column_flat["s"][column]
        ilnb = int(integer_flat["ilnb"][column])
        rho = column_flat["rho"][column]
        qkms = column_flat["qkms"][column]
        tkms = column_flat["tkms"][column]
        qvatm = column_flat["qvatm"][column]
        tabs = column_flat["tabs"][column]
        patm = column_flat["patm"][column]
        rnet = column_flat["rnet"][column]
        prcpms = column_flat["prcpms"][column]
        rainf = column_flat["rainf"][column]
        newsnow = column_flat["newsnow"][column]
        meltfactor = column_flat["meltfactor"][column]
        snowfrac = column_flat["snowfrac"][column]

        # module_sf_ruclsm.F:3945-3956.
        deltsn = np.float32(deltsn_scale / rhosn)
        snth = np.float32(snth_scale / rhosn)
        if snhei >= np.float32(deltsn + snth):
            if np.float32(np.float32(snhei - deltsn) - snth) < snth:
                deltsn = np.float32(half * np.float32(snhei - snth))

        # module_sf_ruclsm.F:3958-3993.
        ras = np.float32(rho * milli)
        rhocsn = np.float32(snow_heat * rhosn)
        rhonewcsn = np.float32(snow_heat * rhonewsn)
        thdifsn = np.float32(snow_cond / rhocsn)
        smelt = zero
        snoh = zero
        rsm = zero
        fsn = one
        fso = zero
        qgold = qsg
        tnold = soilt
        dzstop = np.float32(one / np.float32(zsmain[1] - zsmain[0]))
        prcpl = prcpms

        # module_sf_ruclsm.F:4003-4014.
        fq = qkms
        snhei = np.float32(np.float32(snwe * thousand) / rhosn)
        snwepr = snwe
        beta = one
        epot = np.float32(-np.float32(fq * np.float32(qvatm - qsg)))
        epdt = np.float32(np.float32(epot * ras) * timestep)
        if epdt > zero and snwepr <= epdt:
            beta = np.float32(snwepr / max(epdt_floor, epdt))
            snwe = zero

        # module_sf_ruclsm.F:4020-4032 upward tridiagonal sweep in the ice.
        cotso = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhtso = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhtso[0] = temperature_flat[NUM_SOIL_LAYERS - 1, column]
        for step in range(NUM_SOIL_LAYERS - 2):
            kn = NUM_SOIL_LAYERS - step - 1
            first = 2 * kn - 3
            x1 = np.float32(
                dtdzs[first - 1] * profile_flat["thdifice"][kn - 2, column]
            )
            x2 = np.float32(
                dtdzs[first] * profile_flat["thdifice"][kn - 1, column]
            )
            ft = np.float32(
                temperature_flat[kn - 1, column]
                + np.float32(
                    x1
                    * np.float32(
                        temperature_flat[kn - 2, column]
                        - temperature_flat[kn - 1, column]
                    )
                )
            )
            ft = np.float32(
                ft
                - np.float32(
                    x2
                    * np.float32(
                        temperature_flat[kn - 1, column]
                        - temperature_flat[kn, column]
                    )
                )
            )
            denominator = np.float32(one + x1)
            denominator = np.float32(denominator + x2)
            denominator = np.float32(denominator - np.float32(x2 * cotso[step]))
            cotso[step + 1] = np.float32(x1 / denominator)
            rhtso[step + 1] = np.float32(
                np.float32(ft + np.float32(x2 * rhtso[step])) / denominator
            )

        # WRF leaves snprim/tsob/cotsn/rhtsn undefined when snhei == 0; no
        # consumer is reached on that path, so zero stands in for them here.
        snprim = zero
        tsob = zero
        cotsn = zero
        rhtsn = zero
        thin = bool(snhei < snth and snhei > zero)
        thdifice_top = profile_flat["thdifice"][0, column]
        if snhei >= snth:
            if snhei <= np.float32(deltsn + snth):
                # module_sf_ruclsm.F:4037-4055 one-layer snow.
                ilnb = 1
                snprim = max(snth, snhei)
                soilt1 = temperature_flat[0, column]
                tsob = temperature_flat[0, column]
                xsn = np.float32(
                    np.float32(timestep / two)
                    / np.float32(zshalf[1] + np.float32(half * snprim))
                )
                ddzsn = np.float32(xsn / snprim)
                x1sn = np.float32(ddzsn * thdifsn)
                x2 = np.float32(dtdzs[0] * thdifice_top)
                ft = np.float32(
                    temperature_flat[0, column]
                    + np.float32(
                        x1sn * np.float32(soilt - temperature_flat[0, column])
                    )
                )
                ft = np.float32(
                    ft
                    - np.float32(
                        x2
                        * np.float32(
                            temperature_flat[0, column]
                            - temperature_flat[1, column]
                        )
                    )
                )
                denominator = np.float32(one + x1sn)
                denominator = np.float32(denominator + x2)
                denominator = np.float32(
                    denominator
                    - np.float32(x2 * cotso[NUM_SOIL_LAYERS - 2])
                )
                cotso[NUM_SOIL_LAYERS - 1] = np.float32(x1sn / denominator)
                rhtso[NUM_SOIL_LAYERS - 1] = np.float32(
                    np.float32(
                        ft
                        + np.float32(x2 * rhtso[NUM_SOIL_LAYERS - 2])
                    )
                    / denominator
                )
                cotsn = cotso[NUM_SOIL_LAYERS - 1]
                rhtsn = rhtso[NUM_SOIL_LAYERS - 1]
                tsnav = np.float32(
                    np.float32(
                        half
                        * np.float32(soilt + temperature_flat[0, column])
                    )
                    - freeze
                )
            else:
                # module_sf_ruclsm.F:4058-4082 two-layer snow.
                ilnb = 2
                snprim = deltsn
                tsob = soilt1
                xsn = np.float32(
                    np.float32(timestep / two) / np.float32(half * snhei)
                )
                xsn1 = np.float32(
                    np.float32(timestep / two)
                    / np.float32(
                        zshalf[1]
                        + np.float32(half * np.float32(snhei - deltsn))
                    )
                )
                ddzsn = np.float32(xsn / deltsn)
                ddzsn1 = np.float32(xsn1 / np.float32(snhei - deltsn))
                x1sn = np.float32(ddzsn * thdifsn)
                x1sn1 = np.float32(ddzsn1 * thdifsn)
                x2 = np.float32(dtdzs[0] * thdifice_top)
                ft = np.float32(
                    temperature_flat[0, column]
                    + np.float32(
                        x1sn1 * np.float32(soilt1 - temperature_flat[0, column])
                    )
                )
                ft = np.float32(
                    ft
                    - np.float32(
                        x2
                        * np.float32(
                            temperature_flat[0, column]
                            - temperature_flat[1, column]
                        )
                    )
                )
                denominator = np.float32(one + x1sn1)
                denominator = np.float32(denominator + x2)
                denominator = np.float32(
                    denominator
                    - np.float32(x2 * cotso[NUM_SOIL_LAYERS - 2])
                )
                cotso[NUM_SOIL_LAYERS - 1] = np.float32(x1sn1 / denominator)
                rhtso[NUM_SOIL_LAYERS - 1] = np.float32(
                    np.float32(
                        ft
                        + np.float32(x2 * rhtso[NUM_SOIL_LAYERS - 2])
                    )
                    / denominator
                )
                ftsnow = np.float32(
                    soilt1 + np.float32(x1sn * np.float32(soilt - soilt1))
                )
                ftsnow = np.float32(
                    ftsnow
                    - np.float32(
                        x1sn1
                        * np.float32(soilt1 - temperature_flat[0, column])
                    )
                )
                denomsn = np.float32(one + x1sn)
                denomsn = np.float32(denomsn + x1sn1)
                denomsn = np.float32(
                    denomsn
                    - np.float32(x1sn1 * cotso[NUM_SOIL_LAYERS - 1])
                )
                cotsn = np.float32(x1sn / denomsn)
                rhtsn = np.float32(
                    np.float32(
                        ftsnow
                        + np.float32(x1sn1 * rhtso[NUM_SOIL_LAYERS - 1])
                    )
                    / denomsn
                )
                tsnav = np.float32(
                    np.float32(
                        np.float32(half / snhei)
                        * np.float32(
                            np.float32(
                                np.float32(soilt + soilt1) * deltsn
                            )
                            + np.float32(
                                np.float32(
                                    soilt1 + temperature_flat[0, column]
                                )
                                * np.float32(snhei - deltsn)
                            )
                        )
                    )
                    - freeze
                )

        if thin:
            # module_sf_ruclsm.F:4089-4108 snow blended into the top ice layer.
            snprim = np.float32(snhei + zsmain[1])
            fsn = np.float32(snhei / snprim)
            fso = np.float32(one - fsn)
            soilt1 = temperature_flat[0, column]
            tsob = temperature_flat[1, column]
            xsn = np.float32(
                np.float32(timestep / two)
                / np.float32(
                    np.float32(zshalf[2] - zsmain[1])
                    + np.float32(half * snprim)
                )
            )
            ddzsn = np.float32(xsn / snprim)
            x1sn = np.float32(
                ddzsn
                * np.float32(
                    np.float32(fsn * thdifsn)
                    + np.float32(fso * thdifice_top)
                )
            )
            x2 = np.float32(
                dtdzs[1] * profile_flat["thdifice"][1, column]
            )
            ft = np.float32(
                temperature_flat[1, column]
                + np.float32(
                    x1sn * np.float32(soilt - temperature_flat[1, column])
                )
            )
            ft = np.float32(
                ft
                - np.float32(
                    x2
                    * np.float32(
                        temperature_flat[1, column]
                        - temperature_flat[2, column]
                    )
                )
            )
            denominator = np.float32(one + x1sn)
            denominator = np.float32(denominator + x2)
            denominator = np.float32(
                denominator - np.float32(x2 * cotso[NUM_SOIL_LAYERS - 3])
            )
            cotso[NUM_SOIL_LAYERS - 2] = np.float32(x1sn / denominator)
            rhtso[NUM_SOIL_LAYERS - 2] = np.float32(
                np.float32(
                    ft + np.float32(x2 * rhtso[NUM_SOIL_LAYERS - 3])
                )
                / denominator
            )
            tsnav = np.float32(
                np.float32(
                    half * np.float32(soilt + temperature_flat[0, column])
                )
                - freeze
            )
            cotso[NUM_SOIL_LAYERS - 1] = cotso[NUM_SOIL_LAYERS - 2]
            rhtso[NUM_SOIL_LAYERS - 1] = rhtso[NUM_SOIL_LAYERS - 2]
            cotsn = cotso[NUM_SOIL_LAYERS - 1]
            rhtsn = rhtso[NUM_SOIL_LAYERS - 1]

        # module_sf_ruclsm.F:4114-4131 heat balance coefficients.
        epot = np.float32(-np.float32(qkms * np.float32(qvatm - qsg)))
        rhcs = profile_flat["capice"][0, column]
        d1 = cotso[NUM_SOIL_LAYERS - 2]
        d2 = rhtso[NUM_SOIL_LAYERS - 2]
        tn = soilt
        d9 = np.float32(np.float32(thdifice_top * rhcs) * dzstop)
        d10 = np.float32(np.float32(tkms * cp_air) * rho)
        r211 = np.float32(np.float32(half * flux_depth[column]) / timestep)
        r21 = np.float32(np.float32(r211 * cp_air) * rho)
        depth_square = np.float32(dzstop * dzstop)
        r22 = np.float32(
            half
            / np.float32(np.float32(thdifice_top * timestep) * depth_square)
        )
        tn2 = np.float32(tn * tn)
        tn4 = np.float32(tn2 * tn2)
        r6 = np.float32(
            np.float32(np.float32(emiss * stbolt) * half) * tn4
        )
        r7 = np.float32(r6 / tn)
        d11 = np.float32(rnet + r6)

        # module_sf_ruclsm.F:4133-4163 snow-side coefficients.
        if snhei >= snth:
            if snhei <= np.float32(deltsn + snth):
                d1sn = cotso[NUM_SOIL_LAYERS - 1]
                d2sn = rhtso[NUM_SOIL_LAYERS - 1]
            else:
                d1sn = cotsn
                d2sn = rhtsn
            d9sn = np.float32(np.float32(thdifsn * rhocsn) / snprim)
            r22sn = np.float32(
                np.float32(np.float32(snprim * snprim) * half)
                / np.float32(thdifsn * timestep)
            )
        if thin:
            d1sn = d1
            d2sn = d2
            d9sn = np.float32(
                np.float32(
                    np.float32(np.float32(fsn * thdifsn) * rhocsn)
                    + np.float32(np.float32(fso * thdifice_top) * rhcs)
                )
                / snprim
            )
            r22sn = np.float32(
                np.float32(np.float32(snprim * snprim) * half)
                / np.float32(
                    np.float32(
                        np.float32(fsn * thdifsn)
                        + np.float32(fso * thdifice_top)
                    )
                    * timestep
                )
            )
        if snhei == zero:
            d9sn = d9
            r22sn = r22
            d1sn = d1
            d2sn = d2

        # module_sf_ruclsm.F:4167-4182.
        rain_heat_capacity = np.float32(np.float32(rainf * cvw) * prcpms)
        newsnow_heat_capacity = np.float32(
            np.float32(rhonewcsn * newsnow) / timestep
        )
        tdenom = np.float32(
            d9sn * np.float32(np.float32(one - d1sn) + r22sn)
        )
        tdenom = np.float32(tdenom + d10)
        tdenom = np.float32(tdenom + r21)
        tdenom = np.float32(tdenom + r7)
        tdenom = np.float32(tdenom + rain_heat_capacity)
        tdenom = np.float32(tdenom + newsnow_heat_capacity)
        fkq = np.float32(qkms * rho)
        r210 = np.float32(r211 * rho)
        beta_fkq = np.float32(beta * fkq)
        aa = np.float32(
            np.float32(xlvm * np.float32(beta_fkq + r210)) / tdenom
        )
        humidity_inner = np.float32(qvatm * beta_fkq)
        humidity_inner = np.float32(humidity_inner + np.float32(r210 * qvg))
        rain_temperature = np.float32(max(freeze, tabs))
        newsnow_temperature = np.float32(min(freeze, tabs))
        numerator = np.float32(d10 * tabs)
        numerator = np.float32(numerator + np.float32(r21 * tn))
        numerator = np.float32(numerator + np.float32(xlvm * humidity_inner))
        numerator = np.float32(numerator + d11)
        conduction = np.float32(d2sn + np.float32(r22sn * tn))
        numerator = np.float32(numerator + np.float32(d9sn * conduction))
        numerator = np.float32(
            numerator + np.float32(rain_heat_capacity * rain_temperature)
        )
        numerator = np.float32(
            numerator
            + np.float32(newsnow_heat_capacity * newsnow_temperature)
        )
        bb = np.float32(numerator / tdenom)
        aa1 = aa
        pp = np.float32(patm * thousand)
        aa1 = np.float32(aa1 / pp)

        # module_sf_ruclsm.F:4184-4200.  snoh is still zero on the only pass
        # WRF makes; the :4374 second iteration is commented out upstream.
        bb = np.float32(bb - np.float32(snoh / tdenom))
        qs1, ts1 = _ruc_vilka(tn, aa1, bb, pp, table)
        qvg = qs1
        qsg = qs1
        qcg = zero
        soilt = ts1

        # module_sf_ruclsm.F:4207-4230 snow interior and top ice temperature.
        if snhei >= snth:
            if snhei > np.float32(deltsn + snth):
                soilt1 = np.float32(
                    min(
                        freeze,
                        np.float32(rhtsn + np.float32(cotsn * soilt)),
                    )
                )
                temperature_flat[0, column] = np.float32(
                    min(
                        ice_cap,
                        np.float32(
                            rhtso[NUM_SOIL_LAYERS - 1]
                            + np.float32(
                                cotso[NUM_SOIL_LAYERS - 1] * soilt1
                            )
                        ),
                    )
                )
                tsob = soilt1
            else:
                temperature_flat[0, column] = np.float32(
                    min(
                        ice_cap,
                        np.float32(
                            rhtso[NUM_SOIL_LAYERS - 1]
                            + np.float32(
                                cotso[NUM_SOIL_LAYERS - 1] * soilt
                            )
                        ),
                    )
                )
                soilt1 = temperature_flat[0, column]
                tsob = temperature_flat[0, column]
        elif thin:
            temperature_flat[1, column] = np.float32(
                min(
                    ice_cap,
                    np.float32(
                        rhtso[NUM_SOIL_LAYERS - 2]
                        + np.float32(cotso[NUM_SOIL_LAYERS - 2] * soilt)
                    ),
                )
            )
            temperature_flat[0, column] = np.float32(
                min(
                    ice_cap,
                    np.float32(
                        temperature_flat[1, column]
                        + np.float32(
                            np.float32(soilt - temperature_flat[1, column])
                            * fso
                        )
                    ),
                )
            )
            soilt1 = temperature_flat[0, column]
            tsob = temperature_flat[1, column]
        else:
            temperature_flat[0, column] = np.float32(min(ice_cap, soilt))
            soilt1 = np.float32(min(ice_cap, soilt))
            tsob = temperature_flat[0, column]

        # module_sf_ruclsm.F:4232-4243 downward substitution through the ice.
        start = 2 if thin else 1
        for level in range(start, NUM_SOIL_LAYERS):
            coefficient = NUM_SOIL_LAYERS - level - 1
            temperature_flat[level, column] = np.float32(
                min(
                    ice_cap,
                    np.float32(
                        rhtso[coefficient]
                        + np.float32(
                            cotso[coefficient]
                            * temperature_flat[level - 1, column]
                        )
                    ),
                )
            )

        # module_sf_ruclsm.F:4257 melting test on the pre-vilka epot.
        melt_supply = np.float32(
            snwepr
            - np.float32(
                np.float32(np.float32(beta * epot) * ras) * timestep
            )
        )
        if soilt > freeze and melt_supply > zero and snhei > zero:
            # module_sf_ruclsm.F:4259-4360 the single melt pass.
            soiltfrac = np.float32(
                np.float32(snowfrac * freeze)
                + np.float32(
                    np.float32(one - snowfrac)
                    * np.float32(min(ice_cap, soilt))
                )
            )
            qsg = np.float32(ruc_qsn(soiltfrac, table) / pp)
            epot = np.float32(-np.float32(qkms * np.float32(qvatm - qsg)))
            q1 = np.float32(epot * ras)
            if q1 <= zero:
                dew = np.float32(-epot)
                qfx = np.float32(np.float32(xlvm * rho) * dew)
                eeta = np.float32(qfx / xlvm)
            else:
                eeta = np.float32(np.float32(q1 * beta) * thousand)
                qfx = np.float32(-np.float32(xlvm * eeta))
            hfx = np.float32(d10 * np.float32(tabs - soiltfrac))
            if snhei >= snth:
                soh = np.float32(
                    np.float32(
                        np.float32(thdifsn * rhocsn)
                        * np.float32(soiltfrac - tsob)
                    )
                    / snprim
                )
            else:
                soh = np.float32(
                    np.float32(
                        np.float32(
                            np.float32(np.float32(fsn * thdifsn) * rhocsn)
                            + np.float32(
                                np.float32(fso * thdifice_top) * rhcs
                            )
                        )
                        * np.float32(soiltfrac - tsob)
                    )
                    / snprim
                )
            snflx = soh
            x = np.float32(
                np.float32(r21 + np.float32(d9sn * r22sn))
                * np.float32(soiltfrac - tnold)
            )
            x = np.float32(
                x
                + np.float32(
                    np.float32(xlvm * r210) * np.float32(qsg - qgold)
                )
            )
            snoh = np.float32(rnet + qfx)
            snoh = np.float32(snoh + hfx)
            snoh = np.float32(
                snoh
                + np.float32(
                    newsnow_heat_capacity
                    * np.float32(newsnow_temperature - soiltfrac)
                )
            )
            snoh = np.float32(snoh - soh)
            snoh = np.float32(snoh - x)
            snoh = np.float32(
                snoh
                + np.float32(
                    rain_heat_capacity
                    * np.float32(rain_temperature - soiltfrac)
                )
            )
            snoh = np.float32(max(zero, snoh))
            smelt = np.float32(np.float32(snoh / xlmelt) * milli)
            potential = np.float32(
                np.float32(snwepr / timestep)
                - np.float32(np.float32(beta * epot) * ras)
            )
            smelt = np.float32(min(smelt, potential))
            smelt = np.float32(max(zero, smelt))
            smelt = np.float32(
                min(
                    smelt,
                    np.float32(
                        np.float32(egglston * meltfactor)
                        * np.float32(max(one, np.float32(soilt - freeze)))
                    ),
                )
            )
            rr = np.float32(
                np.float32(snwepr / timestep)
                - np.float32(np.float32(beta * epot) * ras)
            )
            smelt = np.float32(min(smelt, rr))
            snohgnew = np.float32(np.float32(smelt * xlmelt) * thousand)
            # WRF's snodif here only feeds debug prints.
            snoh = snohgnew
            rsmfrac = np.float32(
                min(
                    rsm_high,
                    np.float32(
                        max(
                            rsm_low,
                            np.float32(
                                np.float32(snwepr / rsm_depth) * rsm_share
                            ),
                        )
                    ),
                )
            )
            if snhei > rsm_snhei:
                rsm = np.float32(np.float32(rsmfrac * smelt) * timestep)
            else:
                rsm = zero
            smelt = np.float32(
                max(zero, np.float32(smelt - np.float32(rsm / timestep)))
            )
            snwe = np.float32(
                max(
                    zero,
                    np.float32(
                        snwepr
                        - np.float32(
                            np.float32(
                                smelt
                                + np.float32(np.float32(beta * epot) * ras)
                            )
                            * timestep
                        )
                    ),
                )
            )
            soilt = soiltfrac
        else:
            # module_sf_ruclsm.F:4363-4370 evaporation-only update.
            if snhei != zero:
                epot = np.float32(-np.float32(qkms * np.float32(qvatm - qsg)))
                snwe = np.float32(
                    max(
                        zero,
                        np.float32(
                            snwepr
                            - np.float32(
                                np.float32(
                                    np.float32(beta * epot) * ras
                                )
                                * timestep
                            )
                        ),
                    )
                )

        # module_sf_ruclsm.F:4377-4396 melt-driven snow densification.
        if smelt > zero and rsm > zero:
            if snwe > rsm:
                xsn = np.float32(
                    np.float32(
                        np.float32(rhosn * np.float32(snwe - rsm))
                        + np.float32(thousand * rsm)
                    )
                    / snwe
                )
                rhosn = np.float32(
                    min(np.float32(max(rhosn_low, xsn)), rhosn_high)
                )
                rhocsn = np.float32(snow_heat * rhosn)
                thdifsn = np.float32(snow_cond / rhocsn)

        # module_sf_ruclsm.F:4398-4403 diagnostic copies.
        snweprint = snwe
        snheiprint = np.float32(np.float32(snweprint * thousand) / rhosn)

        # module_sf_ruclsm.F:4409-4417 snow-pack mean temperature.
        if snhei > zero:
            if ilnb > 1:
                tsnav = np.float32(
                    np.float32(
                        np.float32(half / snhei)
                        * np.float32(
                            np.float32(
                                np.float32(soilt + soilt1) * deltsn
                            )
                            + np.float32(
                                np.float32(
                                    soilt1 + temperature_flat[0, column]
                                )
                                * np.float32(snhei - deltsn)
                            )
                        )
                    )
                    - freeze
                )
            else:
                tsnav = np.float32(
                    np.float32(
                        half
                        * np.float32(soilt + temperature_flat[0, column])
                    )
                    - freeze
                )

        # module_sf_ruclsm.F:4419-4428.
        dew = zero
        pp = np.float32(patm * thousand)
        qsg = np.float32(ruc_qsn(soilt, table) / pp)
        epot = np.float32(-np.float32(fq * np.float32(qvatm - qsg)))
        if epot < zero:
            dew = np.float32(-epot)
        snom = np.float32(
            snom + np.float32(np.float32(smelt * timestep) * thousand)
        )

        # module_sf_ruclsm.F:4432-4466 surface flux diagnostics.  WRF's
        # t3/upflux/xinet block here is dead: xinet is assigned, never read.
        sensible = np.float32(np.float32(tkms * cp_air) * rho)
        sensible = np.float32(sensible * np.float32(tabs - soilt))
        hft = np.float32(-sensible)
        # float64 rounded once, matching the CUDA kernel; see
        # _RUC_PROVISIONAL_TRANSCENDENTALS.
        exner = np.float32(
            np.power(np.float64(np.float32(reference / patm)),
                     np.float64(rovcp))
        )
        hfx = np.float32(-np.float32(sensible * exner))
        q1 = np.float32(
            -np.float32(np.float32(fq * ras) * np.float32(qvatm - qsg))
        )
        if q1 < zero:
            if myj:
                eeta = np.float32(
                    -np.float32(
                        np.float32(
                            np.float32(qkms * ras)
                            * np.float32(
                                np.float32(qvatm / np.float32(one + qvatm))
                                - np.float32(qsg / np.float32(one + qsg))
                            )
                        )
                        * thousand
                    )
                )
            else:
                dew = np.float32(qkms * np.float32(qvatm - qsg))
                eeta = np.float32(-np.float32(rho * dew))
            qfx = np.float32(xlvm * eeta)
            eeta = np.float32(-np.float32(rho * dew))
        else:
            if myj:
                eeta = np.float32(
                    -np.float32(
                        np.float32(
                            np.float32(np.float32(qkms * ras) * beta)
                            * np.float32(
                                np.float32(qvatm / np.float32(one + qvatm))
                                - np.float32(qvg / np.float32(one + qvg))
                            )
                        )
                        * thousand
                    )
                )
            else:
                eeta = np.float32(np.float32(q1 * beta) * thousand)
            qfx = np.float32(xlvm * eeta)
            eeta = np.float32(np.float32(q1 * beta) * thousand)
        sublim = eeta

        # module_sf_ruclsm.F:4468-4486.  The snhei tested here is still the
        # pre-melt depth; :4486 rebuilds it from the updated snwe.
        if snhei >= snth:
            storage = np.float32(
                np.float32(
                    np.float32(thdifsn * rhocsn) * np.float32(soilt - tsob)
                )
                / snprim
            )
            snflx = storage
        elif thin:
            storage = np.float32(
                np.float32(
                    np.float32(
                        np.float32(np.float32(fsn * thdifsn) * rhocsn)
                        + np.float32(np.float32(fso * thdifice_top) * rhcs)
                    )
                    * np.float32(soilt - tsob)
                )
                / snprim
            )
            snflx = storage
        else:
            snflx = np.float32(d9sn * np.float32(soilt - tsob))
        snhei = np.float32(np.float32(snwe * thousand) / rhosn)

        # module_sf_ruclsm.F:4492-4509.
        x = np.float32(
            np.float32(r21 + np.float32(d9sn * r22sn))
            * np.float32(soilt - tnold)
        )
        x = np.float32(
            x
            + np.float32(np.float32(xlvm * r210) * np.float32(qsg - qgold))
        )
        x = np.float32(
            x
            - np.float32(
                newsnow_heat_capacity
                * np.float32(newsnow_temperature - soilt)
            )
        )
        x = np.float32(
            x
            - np.float32(
                rain_heat_capacity * np.float32(rain_temperature - soilt)
            )
        )
        residual = np.float32(rnet - hft)
        residual = np.float32(residual - np.float32(xlvm * eeta))
        residual = np.float32(residual - storage)
        residual = np.float32(residual - snoh)
        residual = np.float32(residual - x)
        icemelt = residual
        fltot = np.float32(residual - icemelt)

        # module_sf_ruclsm.F:4517-4522 restore the bare sea-ice surface.
        if snhei == zero:
            tsnav = np.float32(soilt - freeze)
            emiss = bare_emiss
            znt = bare_znt
            alb = bare_alb

        layer_count_flat[column] = np.int32(ilnb)
        output_flat["snweprint"][column] = snweprint
        output_flat["snheiprint"][column] = snheiprint
        output_flat["rsm"][column] = rsm
        output_flat["dew"][column] = dew
        output_flat["soilt"][column] = soilt
        output_flat["soilt1"][column] = soilt1
        output_flat["tsnav"][column] = tsnav
        output_flat["qvg"][column] = qvg
        output_flat["qsg"][column] = qsg
        output_flat["qcg"][column] = qcg
        output_flat["smelt"][column] = smelt
        output_flat["snoh"][column] = snoh
        output_flat["snflx"][column] = snflx
        output_flat["snom"][column] = snom
        output_flat["eeta"][column] = eeta
        output_flat["qfx"][column] = qfx
        output_flat["hfx"][column] = hfx
        output_flat["s"][column] = storage
        output_flat["sublim"][column] = sublim
        output_flat["prcpl"][column] = prcpl
        output_flat["fltot"][column] = fltot
        output_flat["snwe"][column] = snwe
        output_flat["snhei"][column] = snhei
        output_flat["rhosn"][column] = rhosn
        output_flat["emiss"][column] = emiss
        output_flat["alb"][column] = alb
        output_flat["znt"][column] = znt

    for name, array in (("tso", temperature), *outputs.items()):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"RUC snowseaice produced non-finite {name}")
    return RucSnowSeaIceStep(tso=temperature, ilnb=layer_count, **outputs)


__all__ += [
    "RUC_SNOW_SEA_ICE_COLUMN_INPUTS",
    "RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS",
    "RUC_SNOW_SEA_ICE_INTEGER_INPUTS",
    "RUC_SNOW_SEA_ICE_PROFILE_INPUTS",
    "RucSnowSeaIceStep",
    "ruc_snow_sea_ice_step",
]


@dataclass(frozen=True)
class RucSnowTemperature:
    """Snow/soil heat state and snow budget returned by WRF ``snowtemp``.

    ``storage`` is WRF's local ``x`` (the surface heat-storage residual) and
    ``s`` is the top soil-layer conduction it reports last;
    ``snflx`` is the snow-layer flux.  ``ilnb`` is the snow layer count:
    ``snowtemp`` declares it ``intent(out)`` yet reads it back at the final
    ``tsnav`` update (``phys/module_sf_ruclsm.F:5716``), so it is both an
    input and an output of this transcription.
    """

    tso: np.ndarray
    soilt: np.ndarray
    soilt1: np.ndarray
    tsnav: np.ndarray
    qvg: np.ndarray
    qsg: np.ndarray
    qcg: np.ndarray
    dew: np.ndarray
    snwe: np.ndarray
    snhei: np.ndarray
    rhosn: np.ndarray
    beta: np.ndarray
    smelt: np.ndarray
    snoh: np.ndarray
    snflx: np.ndarray
    s: np.ndarray
    rsm: np.ndarray
    snweprint: np.ndarray
    snheiprint: np.ndarray
    storage: np.ndarray
    ilnb: np.ndarray


RUC_SNOW_TEMPERATURE_PROFILE_INPUTS = (
    "cap",
    "thdif",
    "tranf",
    "tso",
)
RUC_SNOW_TEMPERATURE_COLUMN_INPUTS = (
    "snwe",
    "snwepr",
    "snhei",
    "newsnow",
    "snowfrac",
    "beta",
    "deltsn",
    "snth",
    "rhosn",
    "rhonewsn",
    "meltfactor",
    "prcpms",
    "rainf",
    "patm",
    "tabs",
    "qvatm",
    "emiss",
    "rnet",
    "qkms",
    "tkms",
    "rho",
    "vegfrac",
    "drycan",
    "wetcan",
    "transum",
    "dew",
    "soilt",
    "soilt1",
    "qvg",
)


def _ruc_snow_thermal_diffusivity(
    rhosn: np.float32,
    rhocsn: np.float32,
    rhonewsn: np.float32,
    newsnow: np.float32,
    snhei: np.float32,
) -> np.float32:
    """WRF ``snowtemp`` snow thermal diffusivity for ``isncond_opt = 2``.

    ``phys/module_sf_ruclsm.F:49`` fixes ``isncond_opt = 2``, so the constant
    ``0.265/rhocsn`` branch is dead and the Sturm et al. (1997) effective
    conductivity is always used.  The identical block appears twice, at
    ``:5046-5072`` before the solve and at ``:5587-5610`` after a melt-driven
    density update; both call sites share this transcription.
    """

    zero = np.float32(0.0)
    fact = np.float32(1.0)
    if rhosn < np.float32(156.0) or (
        newsnow > zero and rhonewsn < np.float32(156.0)
    ):
        keff = np.float32(
            np.float32(0.023)
            + np.float32(np.float32(np.float32(0.234) * rhosn) * np.float32(1.0e-3))
        )
    else:
        keff = np.float32(
            np.float32(0.138)
            - np.float32(np.float32(np.float32(1.01) * rhosn) * np.float32(1.0e-3))
        )
        keff = np.float32(
            keff
            + np.float32(
                np.float32(np.float32(3.233) * np.float32(rhosn * rhosn))
                * np.float32(1.0e-6)
            )
        )
    if newsnow <= zero and snhei > np.float32(1.0) and rhosn > np.float32(250.0):
        # :5600-5606 - hard slabs under deep packs get a fixed diffusivity.
        return np.float32(4.431718e-7)
    return np.float32(np.float32(keff / rhocsn) * fact)


def ruc_snow_temperature_step(
    values: Mapping[str, object],
    *,
    delt: float,
    conflx: float = 40.0,
    nroot: object = 4,
    ilnb: object = 1,
    xlvm: float = 2.835e6,
    cvw: float = 4.183e6,
) -> RucSnowTemperature:
    """Transcribe WRF ``snowtemp``: the snow energy budget and heat solve.

    ``phys/module_sf_ruclsm.F:4836-5728``.  One nine-level column at a time:
    the upward tridiagonal sweep through the soil (``:5103-5115``), the extra
    snow coefficient row for a one-layer pack (``:5119-5141``), a two-layer
    pack (``:5143-5171``) or snow blended into the top soil layer
    (``:5174-5198``), the surface energy balance closed by ``vilka``
    (``:5277-5329``), the downward substitution (``:5347-5394``), the melt
    iteration (``:5414-5569``) and the bottom-melt, density and flux
    epilogue (``:5572-5725``).

    Arguments WRF passes but ``snowtemp`` never reads are omitted from this
    signature: ``i``, ``j``, ``ktau``, ``iland`` and ``isoil`` only reach
    debug prints and ``vilka``'s fatal handler; ``qcatm``, ``gsw``, ``pc``,
    ``dqm``, ``qmin``, ``psis``, ``bclh``, ``mavail``, ``rovcp`` and ``g0_p``
    are dead; ``glw`` reaches only the local ``xinet`` (``:5421``) which is
    assigned and never used; ``cst`` reaches only ``cmc2ms`` (``:5447``)
    which is likewise dead; and the incoming ``qsg``, ``qcg`` and ``tsnav``
    are all overwritten before they are read.  ``iter`` is set to zero at
    ``:5030`` and never assigned again, so the ``iter==1`` disjunct at
    ``:5425`` is unreachable.

    ``h`` is fixed at 1 (``:5209``), so ``tx2`` is exactly zero, ``q1``
    equals ``qs1`` bit for bit and the unsaturated retry at ``:5315-5328``
    cannot be entered.  It is transcribed anyway.

    ``snhei == 0`` is rejected, but not for the reason first recorded here.
    That reason -- "WRF reads the uninitialised ``snprim`` and ``tsob`` on
    that path (``:5459`` and ``:5626``)" -- does not survive checking:
    ``:5459`` sits inside the melt block opened at ``:5414``
    (``if(soilt.gt.273.15.and.beta==1..and.snhei.gt.0.)``), which
    ``snhei == 0`` cannot enter; ``:5626`` reads ``tsob``, which *is*
    assigned at ``:5371`` on that very path; and ``:5261-5270`` defines all
    four snow-row coefficients precisely for ``snhei == 0``.

    The gate is still correct, on reachability instead: ``LSMRUC`` enters
    the snow branch only at ``:1600`` (``if(snhei.gt.0.0)``), ``snowsoil``
    does not reassign ``snhei`` before ``:3580``, and ``:3580`` is the
    module's only ``call snowtemp``.  So no reachable caller can present
    ``snhei == 0``, and the oracle builder cannot produce a column for it --
    which is what leaves the path without a reference result.
    """

    timestep = np.float32(delt)
    constant_flux_depth = _ruc_constant_flux_depth(conflx, "snowtemp")
    water_heat_capacity = np.float32(cvw)
    latent_heat = np.float32(xlvm)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC snowtemp delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC snowtemp cvw must be finite and positive")
    if not np.isfinite(latent_heat) or latent_heat <= 0.0:
        raise ValueError("RUC snowtemp xlvm must be finite and positive")
    required = (
        RUC_SNOW_TEMPERATURE_PROFILE_INPUTS
        + RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC snowtemp inputs: {', '.join(missing)}")

    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SNOW_TEMPERATURE_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC snowtemp profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    raw_layers = np.asarray(ilnb)
    if not np.issubdtype(raw_layers.dtype, np.integer):
        raise TypeError("RUC snowtemp ilnb must contain integer layer counts")
    try:
        layers = np.ascontiguousarray(
            np.broadcast_to(raw_layers, horizontal_shape), dtype=np.int32
        )
    except ValueError as exc:
        raise ValueError(
            f"ilnb shape {raw_layers.shape} is not broadcastable to "
            f"{horizontal_shape}"
        ) from exc
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    }
    if np.any(profiles["thdif"][0] <= np.float32(0.0)):
        raise ValueError("RUC snowtemp top-level thdif must be positive")
    if np.any(profiles["cap"][0] <= np.float32(0.0)):
        raise ValueError("RUC snowtemp top-level cap must be positive")
    if np.any(columns["patm"] <= np.float32(0.0)):
        raise ValueError("RUC snowtemp patm must be positive")
    if np.any(columns["rho"] <= np.float32(0.0)):
        raise ValueError("RUC snowtemp rho must be positive")
    if np.any(columns["rhosn"] <= np.float32(0.0)):
        raise ValueError("RUC snowtemp rhosn must be positive")
    if np.any(columns["snhei"] <= np.float32(0.0)):
        raise ValueError("RUC snowtemp snhei must be positive")
    if np.any(columns["snth"] <= np.float32(0.0)):
        raise ValueError("RUC snowtemp snth must be positive")
    if np.any(columns["deltsn"] <= np.float32(0.0)):
        raise ValueError("RUC snowtemp deltsn must be positive")

    zsmain, zshalf, dtdzs, _ = _ruc_soil_solver_geometry(timestep)
    table = _load_ruc_saturation_table()
    ncolumn = int(np.prod(horizontal_shape))
    profile_flat = {
        name: array.reshape(NUM_SOIL_LAYERS, ncolumn)
        for name, array in profiles.items()
    }
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    roots_flat = roots.reshape(ncolumn)
    layers_flat = layers.reshape(ncolumn)
    temperature = np.array(profiles["tso"], dtype=np.float32, copy=True)
    temperature_flat = temperature.reshape(NUM_SOIL_LAYERS, ncolumn)
    scalar_names = (
        "soilt", "soilt1", "tsnav", "qvg", "qsg", "qcg", "dew", "snwe",
        "snhei", "rhosn", "beta", "smelt", "snoh", "snflx", "s", "rsm",
        "snweprint", "snheiprint", "storage",
    )
    outputs = {
        name: np.empty(horizontal_shape, dtype=np.float32)
        for name in scalar_names
    }
    output_flat = {name: array.reshape(ncolumn) for name, array in outputs.items()}
    layer_out = np.empty(horizontal_shape, dtype=np.int32)
    layer_out_flat = layer_out.reshape(ncolumn)

    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    # module_model_constants cp and stbolt; the pinned RUC lane fixes them at
    # these values and the oracle carries the same bytes.
    cp_air = np.float32(1004.5)
    stbolt = np.float32(5.67051e-8)
    freeze = np.float32(273.15)
    # :5045 latent heat of fusion.
    xlmelt = np.float32(3.35e5)
    thousand = np.float32(1.0e3)
    milli = np.float32(1.0e-3)
    dzstop = np.float32(one / np.float32(zsmain[1] - zsmain[0]))

    # ``conflx`` may be one depth for the whole field or one per column;
    # bind it to the column axis once rather than at every read.
    flux_depth = np.broadcast_to(constant_flux_depth, (ncolumn,))

    for column in range(ncolumn):
        root_count = int(roots_flat[column])
        snow_layers = int(layers_flat[column])
        thdif = profile_flat["thdif"][:, column]
        cap = profile_flat["cap"][:, column]
        tranf = profile_flat["tranf"][:, column]
        tso = temperature_flat[:, column]
        snwe = column_flat["snwe"][column]
        snwepr = column_flat["snwepr"][column]
        snhei = column_flat["snhei"][column]
        newsnow = column_flat["newsnow"][column]
        snowfrac = column_flat["snowfrac"][column]
        beta = column_flat["beta"][column]
        deltsn = column_flat["deltsn"][column]
        snth = column_flat["snth"][column]
        rhosn = column_flat["rhosn"][column]
        rhonewsn = column_flat["rhonewsn"][column]
        meltfactor = column_flat["meltfactor"][column]
        prcpms = column_flat["prcpms"][column]
        rainf = column_flat["rainf"][column]
        patm = column_flat["patm"][column]
        tabs = column_flat["tabs"][column]
        qvatm = column_flat["qvatm"][column]
        emiss = column_flat["emiss"][column]
        rnet = column_flat["rnet"][column]
        qkms = column_flat["qkms"][column]
        tkms = column_flat["tkms"][column]
        rho = column_flat["rho"][column]
        vegfrac = column_flat["vegfrac"][column]
        drycan = column_flat["drycan"][column]
        wetcan = column_flat["wetcan"][column]
        transum = column_flat["transum"][column]
        dew = column_flat["dew"][column]
        soilt = column_flat["soilt"][column]
        soilt1 = column_flat["soilt1"][column]
        qvg = column_flat["qvg"][column]

        # :5046-5072 snow heat capacity and thermal diffusivity.
        rhocsn = np.float32(np.float32(2090.0) * rhosn)
        rhonewcsn = np.float32(np.float32(2090.0) * rhonewsn)
        thdifsn = _ruc_snow_thermal_diffusivity(
            rhosn, rhocsn, rhonewsn, newsnow, snhei
        )
        # :5074-5091 prologue.
        ras = np.float32(rho * milli)
        soiltfrac = soilt
        smelt = zero
        rsm = zero
        fsn = one
        fso = zero
        qgold = qvg
        snprim = zero
        tsob = zero
        cotsn = zero
        rhtsn = zero
        tsnav = zero

        # :5103-5115 upward tridiagonal sweep through the soil.
        cotso = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhtso = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
        rhtso[0] = tso[NUM_SOIL_LAYERS - 1]
        for step in range(NUM_SOIL_LAYERS - 2):
            kn = NUM_SOIL_LAYERS - step - 1
            first = 2 * kn - 3
            x1 = np.float32(dtdzs[first - 1] * thdif[kn - 2])
            x2 = np.float32(dtdzs[first] * thdif[kn - 1])
            ft = np.float32(
                tso[kn - 1] + np.float32(x1 * np.float32(tso[kn - 2] - tso[kn - 1]))
            )
            ft = np.float32(
                ft - np.float32(x2 * np.float32(tso[kn - 1] - tso[kn]))
            )
            denominator = np.float32(one + x1)
            denominator = np.float32(denominator + x2)
            denominator = np.float32(denominator - np.float32(x2 * cotso[step]))
            cotso[step + 1] = np.float32(x1 / denominator)
            rhtso[step + 1] = np.float32(
                np.float32(ft + np.float32(x2 * rhtso[step])) / denominator
            )

        half_step = np.float32(timestep / np.float32(2.0))
        if snhei >= snth:
            if snhei <= np.float32(deltsn + snth):
                # :5124-5141 one snow layer.
                snow_layers = 1
                snprim = max(snth, snhei)
                tsob = tso[0]
                soilt1 = tso[0]
                xsn = np.float32(
                    half_step / np.float32(zshalf[1] + np.float32(half * snprim))
                )
                ddzsn = np.float32(xsn / snprim)
                x1sn = np.float32(ddzsn * thdifsn)
                x2 = np.float32(dtdzs[0] * thdif[0])
                ft = np.float32(tso[0] + np.float32(x1sn * np.float32(soilt - tso[0])))
                ft = np.float32(ft - np.float32(x2 * np.float32(tso[0] - tso[1])))
                denominator = np.float32(one + x1sn)
                denominator = np.float32(denominator + x2)
                denominator = np.float32(
                    denominator - np.float32(x2 * cotso[NUM_SOIL_LAYERS - 2])
                )
                cotso[NUM_SOIL_LAYERS - 1] = np.float32(x1sn / denominator)
                rhtso[NUM_SOIL_LAYERS - 1] = np.float32(
                    np.float32(ft + np.float32(x2 * rhtso[NUM_SOIL_LAYERS - 2]))
                    / denominator
                )
                cotsn = cotso[NUM_SOIL_LAYERS - 1]
                rhtsn = rhtso[NUM_SOIL_LAYERS - 1]
                tsnav = np.float32(
                    np.float32(half * np.float32(soilt + tso[0])) - freeze
                )
            else:
                # :5148-5171 two snow layers, soilt1 sits at deltsn depth.
                snow_layers = 2
                snprim = deltsn
                tsob = soilt1
                xsn = np.float32(half_step / np.float32(half * deltsn))
                xsn1 = np.float32(
                    half_step
                    / np.float32(
                        zshalf[1] + np.float32(half * np.float32(snhei - deltsn))
                    )
                )
                ddzsn = np.float32(xsn / deltsn)
                ddzsn1 = np.float32(xsn1 / np.float32(snhei - deltsn))
                x1sn = np.float32(ddzsn * thdifsn)
                x1sn1 = np.float32(ddzsn1 * thdifsn)
                x2 = np.float32(dtdzs[0] * thdif[0])
                ft = np.float32(
                    tso[0] + np.float32(x1sn1 * np.float32(soilt1 - tso[0]))
                )
                ft = np.float32(ft - np.float32(x2 * np.float32(tso[0] - tso[1])))
                denominator = np.float32(one + x1sn1)
                denominator = np.float32(denominator + x2)
                denominator = np.float32(
                    denominator - np.float32(x2 * cotso[NUM_SOIL_LAYERS - 2])
                )
                cotso[NUM_SOIL_LAYERS - 1] = np.float32(x1sn1 / denominator)
                rhtso[NUM_SOIL_LAYERS - 1] = np.float32(
                    np.float32(ft + np.float32(x2 * rhtso[NUM_SOIL_LAYERS - 2]))
                    / denominator
                )
                ftsnow = np.float32(
                    soilt1 + np.float32(x1sn * np.float32(soilt - soilt1))
                )
                ftsnow = np.float32(
                    ftsnow - np.float32(x1sn1 * np.float32(soilt1 - tso[0]))
                )
                denomsn = np.float32(one + x1sn)
                denomsn = np.float32(denomsn + x1sn1)
                denomsn = np.float32(
                    denomsn - np.float32(x1sn1 * cotso[NUM_SOIL_LAYERS - 1])
                )
                cotsn = np.float32(x1sn / denomsn)
                rhtsn = np.float32(
                    np.float32(
                        ftsnow + np.float32(x1sn1 * rhtso[NUM_SOIL_LAYERS - 1])
                    )
                    / denomsn
                )
                tsnav = np.float32(np.float32(soilt + soilt1) * deltsn)
                tsnav = np.float32(
                    tsnav
                    + np.float32(
                        np.float32(soilt1 + tso[0]) * np.float32(snhei - deltsn)
                    )
                )
                tsnav = np.float32(
                    np.float32(np.float32(half / snhei) * tsnav) - freeze
                )
        if snhei < snth and snhei > zero:
            # :5174-5198 snow too thin for its own layer: blend it with the
            # top soil layer.
            snprim = np.float32(snhei + zsmain[1])
            fsn = np.float32(snhei / snprim)
            fso = np.float32(one - fsn)
            soilt1 = tso[0]
            tsob = tso[1]
            xsn = np.float32(
                half_step
                / np.float32(
                    np.float32(zshalf[2] - zsmain[1]) + np.float32(half * snprim)
                )
            )
            ddzsn = np.float32(xsn / snprim)
            x1sn = np.float32(
                ddzsn
                * np.float32(np.float32(fsn * thdifsn) + np.float32(fso * thdif[0]))
            )
            x2 = np.float32(dtdzs[1] * thdif[1])
            ft = np.float32(tso[1] + np.float32(x1sn * np.float32(soilt - tso[1])))
            ft = np.float32(ft - np.float32(x2 * np.float32(tso[1] - tso[2])))
            denominator = np.float32(one + x1sn)
            denominator = np.float32(denominator + x2)
            denominator = np.float32(
                denominator - np.float32(x2 * cotso[NUM_SOIL_LAYERS - 3])
            )
            cotso[NUM_SOIL_LAYERS - 2] = np.float32(x1sn / denominator)
            rhtso[NUM_SOIL_LAYERS - 2] = np.float32(
                np.float32(ft + np.float32(x2 * rhtso[NUM_SOIL_LAYERS - 3]))
                / denominator
            )
            tsnav = np.float32(np.float32(half * np.float32(soilt + tso[0])) - freeze)
            cotso[NUM_SOIL_LAYERS - 1] = cotso[NUM_SOIL_LAYERS - 2]
            rhtso[NUM_SOIL_LAYERS - 1] = rhtso[NUM_SOIL_LAYERS - 2]
            cotsn = cotso[NUM_SOIL_LAYERS - 1]
            rhtsn = rhtso[NUM_SOIL_LAYERS - 1]

        # :5203-5224 heat balance (Smirnova et al. 1996, eq. 21, 26).
        nmelt = 0
        snoh = zero
        ett1 = zero
        epot = np.float32(-np.float32(qkms * np.float32(qvatm - qgold)))
        rhcs = cap[0]
        h = one
        trans = np.float32(np.float32(transum * drycan) / zshalf[root_count])
        can = np.float32(wetcan + trans)
        umveg = np.float32(one - vegfrac)
        d1 = cotso[NUM_SOIL_LAYERS - 2]
        d2 = rhtso[NUM_SOIL_LAYERS - 2]
        tn = soilt
        d9 = np.float32(np.float32(thdif[0] * rhcs) * dzstop)
        d10 = np.float32(np.float32(tkms * cp_air) * rho)
        r211 = np.float32(np.float32(half * flux_depth[column]) / timestep)
        r21 = np.float32(np.float32(r211 * cp_air) * rho)
        r22 = np.float32(
            half
            / np.float32(
                np.float32(thdif[0] * timestep) * np.float32(dzstop * dzstop)
            )
        )
        tn2 = np.float32(tn * tn)
        tn4 = np.float32(tn2 * tn2)
        r6 = np.float32(np.float32(np.float32(emiss * stbolt) * half) * tn4)
        r7 = np.float32(r6 / tn)
        d11 = np.float32(rnet + r6)

        # :5226-5270 the snow row of the tridiagonal system.
        if snhei >= snth:
            if snhei <= np.float32(deltsn + snth):
                d1sn = cotso[NUM_SOIL_LAYERS - 1]
                d2sn = rhtso[NUM_SOIL_LAYERS - 1]
            else:
                d1sn = cotsn
                d2sn = rhtsn
            d9sn = np.float32(np.float32(thdifsn * rhocsn) / snprim)
            r22sn = np.float32(
                np.float32(np.float32(snprim * snprim) * half)
                / np.float32(thdifsn * timestep)
            )
        if snhei < snth and snhei > zero:
            d1sn = d1
            d2sn = d2
            d9sn = np.float32(
                np.float32(
                    np.float32(np.float32(fsn * thdifsn) * rhocsn)
                    + np.float32(np.float32(fso * thdif[0]) * rhcs)
                )
                / snprim
            )
            r22sn = np.float32(
                np.float32(np.float32(snprim * snprim) * half)
                / np.float32(
                    np.float32(np.float32(fsn * thdifsn) + np.float32(fso * thdif[0]))
                    * timestep
                )
            )

        # :5279-5280 and :5290-5291.  These rain and new-snow heat terms do
        # not depend on anything the melt iteration updates, so WRF's second
        # pass through :5275 recomputes them identically.
        rain_heat_capacity = np.float32(np.float32(rainf * water_heat_capacity) * prcpms)
        snow_heat_capacity = np.float32(np.float32(rhonewcsn * newsnow) / timestep)
        rain_temperature = max(freeze, tabs)
        snow_temperature = min(freeze, tabs)

        while True:
            # :5275 the melt iteration entry point.
            tdenom = np.float32(d9sn * np.float32(np.float32(one - d1sn) + r22sn))
            tdenom = np.float32(tdenom + d10)
            tdenom = np.float32(tdenom + r21)
            tdenom = np.float32(tdenom + r7)
            tdenom = np.float32(tdenom + rain_heat_capacity)
            tdenom = np.float32(tdenom + snow_heat_capacity)
            fkq = np.float32(qkms * rho)
            r210 = np.float32(r211 * rho)
            c = np.float32(np.float32(vegfrac * fkq) * can)
            cc = np.float32(np.float32(c * latent_heat) / tdenom)
            evaporation = np.float32(np.float32(beta * fkq) * umveg)
            aa = np.float32(
                np.float32(latent_heat * np.float32(evaporation + r210)) / tdenom
            )
            humidity_inner = np.float32(qvatm * np.float32(evaporation + c))
            humidity_inner = np.float32(humidity_inner + np.float32(r210 * qgold))
            numerator = np.float32(d10 * tabs)
            numerator = np.float32(numerator + np.float32(r21 * tn))
            numerator = np.float32(
                numerator + np.float32(latent_heat * humidity_inner)
            )
            numerator = np.float32(numerator + d11)
            numerator = np.float32(
                numerator
                + np.float32(d9sn * np.float32(d2sn + np.float32(r22sn * tn)))
            )
            numerator = np.float32(
                numerator + np.float32(rain_heat_capacity * rain_temperature)
            )
            numerator = np.float32(
                numerator + np.float32(snow_heat_capacity * snow_temperature)
            )
            bb = np.float32(numerator / tdenom)
            aa1 = np.float32(aa + cc)
            pp = np.float32(patm * thousand)
            aa1 = np.float32(aa1 / pp)
            bb = np.float32(bb - np.float32(snoh / tdenom))

            qs1, ts1 = _ruc_vilka(tn, aa1, bb, pp, table)
            tq2 = qvatm
            tx2 = np.float32(tq2 * np.float32(one - h))
            q1 = np.float32(tx2 + np.float32(h * qs1))
            saturated = not (q1 < qs1)
            if not saturated:
                # :5315-5328.  Unreachable while h == 1; transcribed anyway.
                bb = np.float32(bb - np.float32(aa * tx2))
                aa = np.float32(np.float32(np.float32(aa * h) + cc) / pp)
                qs1, ts1 = _ruc_vilka(tn, aa, bb, pp, table)
                q1 = np.float32(tx2 + np.float32(h * qs1))
                saturated = q1 > qs1
            if saturated:
                qvg = qs1
                qsg = qs1
                qcg = max(zero, np.float32(q1 - qs1))
            else:
                qsg = qs1
                qvg = q1
                qcg = zero

            # :5332-5340 skin temperature.
            soilt = ts1
            if (
                nmelt == 1
                and snowfrac == one
                and snwe > zero
                and soilt > freeze
            ):
                soilt = min(freeze, soilt)

            # :5348-5378 the snow-soil interface and the 7.5 cm level.
            if snhei >= snth:
                if snhei > np.float32(deltsn + snth):
                    soilt1 = min(freeze, np.float32(rhtsn + np.float32(cotsn * soilt)))
                    tso[0] = np.float32(
                        rhtso[NUM_SOIL_LAYERS - 1]
                        + np.float32(cotso[NUM_SOIL_LAYERS - 1] * soilt1)
                    )
                    tsob = soilt1
                else:
                    tso[0] = np.float32(
                        rhtso[NUM_SOIL_LAYERS - 1]
                        + np.float32(cotso[NUM_SOIL_LAYERS - 1] * soilt)
                    )
                    soilt1 = tso[0]
                    tsob = tso[0]
            elif snhei > zero and snhei < snth:
                tso[1] = np.float32(
                    rhtso[NUM_SOIL_LAYERS - 2]
                    + np.float32(cotso[NUM_SOIL_LAYERS - 2] * soilt)
                )
                tso[0] = np.float32(
                    tso[1] + np.float32(np.float32(soilt - tso[1]) * fso)
                )
                soilt1 = tso[0]
                tsob = tso[1]
            else:
                tso[0] = soilt
                soilt1 = soilt
                tsob = tso[0]
            if nmelt == 1 and snowfrac == one:
                soilt1 = min(freeze, soilt1)
                tso[0] = min(freeze, tso[0])
                tsob = min(freeze, tsob)

            # :5382-5394 downward substitution.
            start = 2 if (snhei > zero and snhei < snth) else 1
            for level in range(start, NUM_SOIL_LAYERS):
                coefficient = NUM_SOIL_LAYERS - level - 1
                tso[level] = np.float32(
                    rhtso[coefficient]
                    + np.float32(cotso[coefficient] * tso[level - 1])
                )

            if nmelt == 1:
                break

            if soilt > freeze and beta == one and snhei > zero:
                # :5414-5553 top melt.
                nmelt = 1
                soiltfrac = np.float32(
                    np.float32(snowfrac * freeze)
                    + np.float32(np.float32(one - snowfrac) * soilt)
                )
                qsg = min(
                    qsg,
                    np.float32(np.float32(ruc_qsn(soiltfrac, table)) / pp),
                )
                qvg = np.float32(
                    np.float32(snowfrac * qsg)
                    + np.float32(np.float32(one - snowfrac) * qvg)
                )
                # :5419-5421 t3/upflux/xinet are dead: xinet is never read.
                epot = np.float32(-np.float32(qkms * np.float32(qvatm - qsg)))
                q1 = np.float32(epot * ras)
                if q1 <= zero:
                    dew = np.float32(-epot)
                    qfx = np.float32(
                        -np.float32(np.float32(latent_heat * rho) * dew)
                    )
                else:
                    for level in range(root_count):
                        transp = np.float32(
                            -np.float32(
                                np.float32(
                                    np.float32(np.float32(vegfrac * q1) * tranf[level])
                                    * drycan
                                )
                                / zshalf[root_count]
                            )
                        )
                        ett1 = np.float32(ett1 - transp)
                    edir1 = np.float32(np.float32(q1 * umveg) * beta)
                    ec1 = np.float32(np.float32(q1 * wetcan) * vegfrac)
                    eeta = np.float32(
                        np.float32(np.float32(edir1 + ec1) + ett1) * thousand
                    )
                    qfx = np.float32(latent_heat * eeta)
                hfx = np.float32(-np.float32(d10 * np.float32(tabs - soiltfrac)))
                if snhei >= snth:
                    soh = np.float32(
                        np.float32(
                            np.float32(thdifsn * rhocsn)
                            * np.float32(soiltfrac - tsob)
                        )
                        / snprim
                    )
                else:
                    soh = np.float32(
                        np.float32(
                            np.float32(
                                np.float32(np.float32(fsn * thdifsn) * rhocsn)
                                + np.float32(np.float32(fso * thdif[0]) * rhcs)
                            )
                            * np.float32(soiltfrac - tsob)
                        )
                        / snprim
                    )
                snflx = soh
                storage = np.float32(
                    np.float32(r21 + np.float32(d9sn * r22sn))
                    * np.float32(soiltfrac - tn)
                )
                storage = np.float32(
                    storage
                    + np.float32(
                        np.float32(latent_heat * r210) * np.float32(qvg - qgold)
                    )
                )
                snoh = np.float32(rnet - qfx)
                snoh = np.float32(snoh - hfx)
                snoh = np.float32(snoh - soh)
                snoh = np.float32(snoh - storage)
                snoh = np.float32(
                    snoh
                    + np.float32(
                        snow_heat_capacity
                        * np.float32(snow_temperature - soiltfrac)
                    )
                )
                snoh = np.float32(
                    snoh
                    + np.float32(
                        rain_heat_capacity
                        * np.float32(rain_temperature - soiltfrac)
                    )
                )
                snoh = max(zero, snoh)
                smelt = np.float32(np.float32(snoh / xlmelt) * milli)
                potential = np.float32(np.float32(epot * ras) * timestep)
                if epot > zero and snwepr <= potential:
                    # :5483-5491 all the snow can evaporate; jump to :5518.
                    beta = np.float32(snwepr / potential)
                    smelt = min(
                        smelt,
                        np.float32(
                            np.float32(snwepr / timestep)
                            - np.float32(np.float32(beta * epot) * ras)
                        ),
                    )
                    snwe = zero
                else:
                    smelt = max(zero, smelt)
                    # :5499-5501 the Egglston melt limiter.
                    if (
                        rhosn < np.float32(350.0)
                        or (newsnow > zero and rhonewsn < np.float32(450.0))
                    ) and soilt < np.float32(283.0):
                        limit = np.float32(timestep / np.float32(60.0))
                        limit = np.float32(limit * np.float32(5.6e-8))
                        limit = np.float32(limit * meltfactor)
                        limit = np.float32(
                            limit * max(one, np.float32(soilt - freeze))
                        )
                        smelt = min(smelt, limit)
                    rr = max(
                        zero,
                        np.float32(
                            np.float32(snwepr / timestep)
                            - np.float32(np.float32(beta * epot) * ras)
                        ),
                    )
                    if smelt > rr:
                        smelt = min(smelt, rr)
                        snwe = zero
                # :5518-5522.
                snohgnew = np.float32(np.float32(smelt * xlmelt) * thousand)
                snoh = snohgnew
                if smelt > zero:
                    # :5529-5543 Koren et al. (1999) liquid retention.
                    rsmfrac = min(
                        np.float32(0.18),
                        max(
                            np.float32(0.08),
                            np.float32(
                                np.float32(snwepr / np.float32(0.10))
                                * np.float32(0.13)
                            ),
                        ),
                    )
                    if snhei > np.float32(0.01) and rhosn < np.float32(350.0):
                        rsm = np.float32(np.float32(rsmfrac * smelt) * timestep)
                    else:
                        rsm = zero
                    if rsm > zero:
                        smelt = max(
                            zero, np.float32(smelt - np.float32(rsm / timestep))
                        )
                if snwe > zero:
                    snwe = max(
                        zero,
                        np.float32(
                            snwepr
                            - np.float32(
                                np.float32(
                                    smelt + np.float32(np.float32(beta * epot) * ras)
                                )
                                * timestep
                            )
                        ),
                    )
            else:
                # :5557-5567 no melt: sublimation or condensation only.
                if snhei != zero and beta == one:
                    epot = np.float32(-np.float32(qkms * np.float32(qvatm - qsg)))
                    snwe = max(
                        zero,
                        np.float32(
                            snwepr
                            - np.float32(
                                np.float32(np.float32(beta * epot) * ras) * timestep
                            )
                        ),
                    )
                else:
                    snwe = zero

            if nmelt == 1:
                continue
            break

        # :5572-5613 melt water changes the snow density.  WRF only prints a
        # diagnostic when snwe <= rsm and leaves the density alone.
        if smelt > zero and rsm > zero and snwe > rsm:
            xsn = np.float32(
                np.float32(
                    np.float32(rhosn * np.float32(snwe - rsm))
                    + np.float32(thousand * rsm)
                )
                / snwe
            )
            rhosn = min(max(np.float32(58.8), xsn), np.float32(500.0))
            rhocsn = np.float32(np.float32(2090.0) * rhosn)
            thdifsn = _ruc_snow_thermal_diffusivity(
                rhosn, rhocsn, rhonewsn, newsnow, snhei
            )

        # :5616-5629 flux in the top snow layer, then in the top soil layer.
        if snhei >= snth:
            s = np.float32(
                np.float32(
                    np.float32(thdifsn * rhocsn) * np.float32(soilt - tsob)
                )
                / snprim
            )
        elif snhei < snth and snhei > zero:
            s = np.float32(
                np.float32(
                    np.float32(
                        np.float32(np.float32(fsn * thdifsn) * rhocsn)
                        + np.float32(np.float32(fso * thdif[0]) * rhcs)
                    )
                    * np.float32(soilt - tsob)
                )
                / snprim
            )
        else:
            s = np.float32(d9sn * np.float32(soilt - tsob))
        snflx = s
        s = np.float32(d9 * np.float32(tso[0] - tso[1]))

        snhei = np.float32(np.float32(snwe * thousand) / rhosn)

        # :5636-5684 melt from the bottom of the pack on thawed ground.
        if tso[0] > freeze and snhei > zero:
            if snhei > np.float32(deltsn + snth):
                hsn = np.float32(snhei - deltsn)
            else:
                hsn = snhei
            soiltfrac = np.float32(
                np.float32(snowfrac * freeze)
                + np.float32(np.float32(one - snowfrac) * tso[0])
            )
            snohg = np.float32(
                np.float32(cap[0] * zshalf[1])
                + np.float32(np.float32(rhocsn * half) * hsn)
            )
            snohg = np.float32(
                np.float32(np.float32(tso[0] - soiltfrac) * snohg) / timestep
            )
            snohg = max(zero, snohg)
            smeltg = np.float32(np.float32(snohg / xlmelt) * milli)
            # :5658-5660 the Egglston bottom-melt limit.
            if (
                rhosn < np.float32(350.0)
                or (newsnow > zero and rhonewsn < np.float32(450.0))
            ) and soilt < np.float32(283.0):
                smeltg = min(smeltg, np.float32(5.8e-9))
            rr = np.float32(snwe / timestep)
            smeltg = min(smeltg, rr)
            snwe = max(zero, np.float32(snwe - np.float32(smeltg * timestep)))
            snhei = np.float32(np.float32(snwe * thousand) / rhosn)
            smelt = np.float32(smelt + smeltg)
            if snhei > zero:
                tso[0] = soiltfrac

        snweprint = snwe
        snheiprint = np.float32(np.float32(snweprint * thousand) / rhosn)

        # :5697-5708 surface heat storage.
        storage = np.float32(
            np.float32(r21 + np.float32(d9sn * r22sn)) * np.float32(soilt - tn)
        )
        storage = np.float32(
            storage
            + np.float32(np.float32(latent_heat * r210) * np.float32(qsg - qgold))
        )
        storage = np.float32(
            storage
            - np.float32(
                snow_heat_capacity * np.float32(snow_temperature - soilt)
            )
        )
        storage = np.float32(
            storage
            - np.float32(
                rain_heat_capacity * np.float32(rain_temperature - soilt)
            )
        )

        # :5715-5725 mean snow-pack temperature in Celsius.
        if snhei > zero:
            if snow_layers > 1:
                tsnav = np.float32(np.float32(soilt + soilt1) * deltsn)
                tsnav = np.float32(
                    tsnav
                    + np.float32(
                        np.float32(soilt1 + tso[0]) * np.float32(snhei - deltsn)
                    )
                )
                tsnav = np.float32(
                    np.float32(np.float32(half / snhei) * tsnav) - freeze
                )
            else:
                tsnav = np.float32(
                    np.float32(half * np.float32(soilt + tso[0])) - freeze
                )
        else:
            tsnav = np.float32(soilt - freeze)

        output_flat["soilt"][column] = soilt
        output_flat["soilt1"][column] = soilt1
        output_flat["tsnav"][column] = tsnav
        output_flat["qvg"][column] = qvg
        output_flat["qsg"][column] = qsg
        output_flat["qcg"][column] = qcg
        output_flat["dew"][column] = dew
        output_flat["snwe"][column] = snwe
        output_flat["snhei"][column] = snhei
        output_flat["rhosn"][column] = rhosn
        output_flat["beta"][column] = beta
        output_flat["smelt"][column] = smelt
        output_flat["snoh"][column] = snoh
        output_flat["snflx"][column] = snflx
        output_flat["s"][column] = s
        output_flat["rsm"][column] = rsm
        output_flat["snweprint"][column] = snweprint
        output_flat["snheiprint"][column] = snheiprint
        output_flat["storage"][column] = storage
        layer_out_flat[column] = np.int32(snow_layers)

    for name, array in (("tso", temperature), *outputs.items()):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"RUC snowtemp produced non-finite {name}")
    return RucSnowTemperature(tso=temperature, ilnb=layer_out, **outputs)


# Appended rather than folded into the list above so this port stays a pure
# addition alongside the other concurrent RUC snow lanes.
__all__ += [
    "RUC_SNOW_TEMPERATURE_COLUMN_INPUTS",
    "RUC_SNOW_TEMPERATURE_PROFILE_INPUTS",
    "RucSnowTemperature",
    "ruc_snow_temperature_step",
]


# ---------------------------------------------------------------------------
# WRF v4.6.1 phys/module_sf_ruclsm.F:3120-3786 subroutine snowsoil -- the
# snow-covered land column.  Appended as one contiguous block so the three
# parallel RUC snow ports stay mergeable.
#
# The :4836-5728 snowtemp heat and melt solve this drives was transcribed
# twice while those ports ran in parallel.  The two transcriptions were
# executed against each other over every path snowsoil can reach with no
# differing 32-bit word, and the private per-column copy has been deleted:
# snowsoil now calls the public ruc_snow_temperature_step above, deriving
# its deltsn and snth arguments here from :3387-3398.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RucSnowSoilStep:
    """Complete snow-covered land state and fluxes from WRF ``snowsoil``.

    ``soilice``/``soiliqw`` carry the freezing-curve partition rebuilt after
    ``snowtemp`` but before ``soilmoist`` (``:3629-3648``); ``soilmoist``
    never writes ``soiliqw`` back, so they partition the moisture state as it
    stood on entry, not the ``soilmois`` returned beside them.
    """

    soilmois: np.ndarray
    tso: np.ndarray
    smfrkeep: np.ndarray
    keepfr: np.ndarray
    soilice: np.ndarray
    soiliqw: np.ndarray
    cst: np.ndarray
    dew: np.ndarray
    soilt: np.ndarray
    soilt1: np.ndarray
    tsnav: np.ndarray
    qvg: np.ndarray
    qsg: np.ndarray
    qcg: np.ndarray
    snwe: np.ndarray
    snhei: np.ndarray
    rhosn: np.ndarray
    ilnb: np.ndarray
    snweprint: np.ndarray
    snheiprint: np.ndarray
    rsm: np.ndarray
    smelt: np.ndarray
    snoh: np.ndarray
    snflx: np.ndarray
    snom: np.ndarray
    edir1: np.ndarray
    ec1: np.ndarray
    ett1: np.ndarray
    eeta: np.ndarray
    qfx: np.ndarray
    hfx: np.ndarray
    s: np.ndarray
    sublim: np.ndarray
    prcpl: np.ndarray
    fltot: np.ndarray
    runoff1: np.ndarray
    runoff2: np.ndarray
    mavail: np.ndarray
    infiltrp: np.ndarray


RUC_SNOW_SOIL_PROFILE_INPUTS = (
    "soilmois",
    "tso",
    "smfrkeep",
    "keepfr",
)
RUC_SNOW_SOIL_COLUMN_INPUTS = (
    "meltfactor",
    "rhonewsn",
    "prcpms",
    "rainf",
    "newsnow",
    "snhei",
    "snwe",
    "snowfrac",
    "rhosn",
    "patm",
    "qvatm",
    "qcatm",
    "emiss",
    "rnet",
    "qkms",
    "tkms",
    "pc",
    "cst",
    "infwater",
    "rho",
    "vegfrac",
    "lai",
    "gswin",
    "qwrtz",
    "rhocs",
    "dqm",
    "qmin",
    "ref",
    "wilt",
    "psis",
    "bclh",
    "ksat",
    "sat",
    "cn",
    "tabs",
    "soilt",
    "soilt1",
    "qvg",
    "qsg",
    "qcg",
    "snom",
)


def ruc_snow_soil_step(
    values: Mapping[str, object],
    iland,
    *,
    nroot: object,
    delt: float,
    conflx: float,
    ilnb: object = 1,
    myj: bool = False,
    cw: float = 4.183e6,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    parameters: RucParameterBundle | None = None,
) -> RucSnowSoilStep:
    """Run the complete deterministic snow-covered WRF RUC land column.

    Transcribes ``snowsoil`` (``phys/module_sf_ruclsm.F:3120-3786``): the
    freezing-curve partition, ``soilprop``, the canopy water and evaporation
    limiter, ``transf``, ``snowtemp``, the post-melt partition rebuild,
    ``soilmoist``, the ``keepfr`` latch and the surface flux diagnostics.

    ``snhei_crit``, ``gsw``, ``alb``, ``znt``, ``ivgtyp`` and ``drip`` are
    arguments ``snowsoil`` never reads -- ``drip`` because the ``soilmoist``
    call at ``:3654-3666`` passes a literal ``0.`` in its place -- and ``glw``
    reaches only the ``:3701`` ``xinet`` that is assigned and never used, so
    none of them appear in this signature.
    """

    if myj is not False:
        raise ValueError("RUC snow soil lane supports myj=False only")
    timestep = np.float32(delt)
    constant_flux_depth = _ruc_constant_flux_depth(conflx, "snowsoil")
    water_heat_capacity = np.float32(cw)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC snowsoil delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC snowsoil cw must be finite and positive")
    required = RUC_SNOW_SOIL_PROFILE_INPUTS + RUC_SNOW_SOIL_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC snowsoil inputs: {', '.join(missing)}")

    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SNOW_SOIL_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC snowsoil profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    land_type = _horizontal_integer_field(iland, horizontal_shape, "iland")
    snow_layers = np.asarray(ilnb)
    if not np.issubdtype(snow_layers.dtype, np.integer):
        raise TypeError("RUC snowsoil ilnb must be an integer layer count")
    if snow_layers.shape != horizontal_shape:
        try:
            snow_layers = np.broadcast_to(snow_layers, horizontal_shape)
        except ValueError as exc:
            raise ValueError(
                f"ilnb shape {snow_layers.shape} is not broadcastable to "
                f"{horizontal_shape}"
            ) from exc
    columns = {
        name: _horizontal_float_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_SOIL_COLUMN_INPUTS
    }
    if np.any(columns["dqm"] <= 0.0):
        raise ValueError("RUC snowsoil dqm must be positive")
    if np.any(columns["psis"] >= 0.0):
        raise ValueError("RUC snowsoil psis must be negative")
    if np.any(columns["bclh"] <= 0.0):
        raise ValueError("RUC snowsoil bclh must be positive")
    if np.any(columns["sat"] <= 0.0):
        raise ValueError("RUC snowsoil canopy saturation must be positive")
    if np.any(columns["rho"] <= 0.0) or np.any(columns["patm"] <= 0.0):
        raise ValueError("RUC snowsoil rho and patm must be positive")
    if np.any(columns["rhosn"] <= 0.0) or np.any(columns["rhonewsn"] <= 0.0):
        raise ValueError("RUC snowsoil snow densities must be positive")
    if np.any(columns["snwe"] < 0.0) or np.any(columns["snhei"] < 0.0):
        raise ValueError("RUC snowsoil snow depth must be nonnegative")
    if np.any((columns["snowfrac"] < 0.0) | (columns["snowfrac"] > 1.0)):
        raise ValueError("RUC snowsoil snowfrac must be within 0..1")

    zero = np.float32(0.0)
    one = np.float32(1.0)
    quarter = np.float32(0.25)
    half = np.float32(0.5)
    freeze = np.float32(273.15)
    xlv = np.float32(2.5e6)
    xlmelt = np.float32(3.35e5)
    # :3368 snowsoil closes its budget with the sublimation latent heat.
    xlvm = np.float32(xlv + xlmelt)
    cp_air = np.float32(1004.5)
    rovcp = np.float32(np.float32(287.0) / cp_air)
    # :3703 Exner factor; p1000mb*0.00001 is exactly 1 in float32.
    reference = np.float32(np.float32(100000.0) * np.float32(0.00001))
    riw = np.float32(np.float32(900.0) * np.float32(1.0e-3))
    cvw = water_heat_capacity

    soilmois = np.array(profiles["soilmois"], dtype=np.float32, copy=True)
    tso = np.array(profiles["tso"], dtype=np.float32, copy=True)
    smfrkeep = np.array(profiles["smfrkeep"], dtype=np.float32, copy=True)
    keepfr = np.array(profiles["keepfr"], dtype=np.float32, copy=True)
    told = tso.copy()
    smold = soilmois.copy()
    phase = _ruc_phase_partition(
        soilmois, tso, smfrkeep, keepfr, columns, update_smfrkeep=True
    )
    properties = ruc_soil_properties({
        **{name: phase[name] for name in (
            "fwsat", "lwsat", "tav", "soilmoism", "soiliqwm", "soilicem"
        )},
        "keepfr": keepfr,
        "soilmois": soilmois,
        "soiliqw": phase["soiliqw"],
        "soilice": phase["soilice"],
        **{name: columns[name] for name in (
            "qwrtz", "rhocs", "dqm", "qmin", "psis", "bclh", "ksat"
        )},
    }, riw=float(riw))

    ncolumn = int(np.prod(horizontal_shape))
    column_flat = {name: array.reshape(ncolumn) for name, array in columns.items()}
    roots_flat = roots.reshape(ncolumn)
    layers_flat = snow_layers.reshape(ncolumn).astype(np.int32, copy=True)

    # :3535-3554 evaporation limiter and canopy water fractions.
    beta = np.empty(horizontal_shape, dtype=np.float32)
    wetcan = np.empty(horizontal_shape, dtype=np.float32)
    drycan = np.empty(horizontal_shape, dtype=np.float32)
    snwe = np.empty(horizontal_shape, dtype=np.float32)
    ras = np.empty(horizontal_shape, dtype=np.float32)
    beta_flat = beta.reshape(ncolumn)
    wet_flat = wetcan.reshape(ncolumn)
    dry_flat = drycan.reshape(ncolumn)
    snwe_flat = snwe.reshape(ncolumn)
    ras_flat = ras.reshape(ncolumn)
    for column in range(ncolumn):
        ras_flat[column] = np.float32(
            column_flat["rho"][column] * np.float32(1.0e-3)
        )
        umveg = np.float32(one - column_flat["vegfrac"][column])
        epot = np.float32(
            -np.float32(
                column_flat["qkms"][column]
                * np.float32(
                    column_flat["qvatm"][column] - column_flat["qsg"][column]
                )
            )
        )
        snwe_flat[column] = column_flat["snwe"][column]
        beta_flat[column] = one
        epdt = np.float32(epot * ras_flat[column])
        epdt = np.float32(epdt * timestep)
        epdt = np.float32(epdt * umveg)
        if epdt > zero and column_flat["snwe"][column] <= epdt:
            beta_flat[column] = np.float32(
                column_flat["snwe"][column]
                / np.float32(max(np.float32(1.0e-8), epdt))
            )
            snwe_flat[column] = zero
        ratio = np.float32(
            column_flat["cst"][column] / column_flat["sat"][column]
        )
        ratio = np.float32(max(zero, ratio))
        # Round a float64 power once, matching the CUDA canopy kernel; numpy's
        # float32 power drifts between releases.  See
        # _RUC_PROVISIONAL_TRANSCENDENTALS.
        fraction = np.float32(
            np.power(
                np.float64(ratio), np.float64(column_flat["cn"][column])
            )
        )
        wet_flat[column] = np.float32(min(quarter, fraction))
        dry_flat[column] = np.float32(one - wet_flat[column])

    transpiration = ruc_transpiration(
        phase["soiliqw"],
        columns["tabs"], columns["lai"], columns["gswin"],
        columns["dqm"], columns["qmin"], columns["ref"], columns["wilt"],
        columns["pc"], land_type, nroot=roots, mminlu=mminlu,
        parameters=parameters,
    )

    zsmain, zshalf, dtdzs, _ = _ruc_soil_solver_geometry(timestep)
    tranf_flat = transpiration.tranf.reshape(NUM_SOIL_LAYERS, ncolumn)

    # :3387-3398 the snow-layer thresholds, from the entry density.  deltsn
    # is the upper snow layer's thickness and snth the minimum a layer may
    # have; a pack only just deeper than the two together is halved instead.
    deltsn = np.float32(np.float32(0.05) * np.float32(1.0e3)) / columns["rhosn"]
    snth = np.float32(np.float32(0.01) * np.float32(1.0e3)) / columns["rhosn"]
    halve = (columns["snhei"] >= (deltsn + snth)) & (
        ((columns["snhei"] - deltsn) - snth) < snth
    )
    deltsn = np.where(halve, half * (columns["snhei"] - snth), deltsn)

    # :3580 the one call to snowtemp.  ``dew`` enters as the literal 0. of
    # :3535, and snowsoil's own qsg/qcg are not passed because snowtemp
    # overwrites both before it reads them.
    snow_result = ruc_snow_temperature_step(
        {
            "cap": properties.cap,
            "thdif": properties.thdif,
            "tranf": transpiration.tranf,
            "tso": tso,
            "snwe": snwe,
            "snwepr": columns["snwe"],
            "snhei": columns["snhei"],
            "newsnow": columns["newsnow"],
            "snowfrac": columns["snowfrac"],
            "beta": beta,
            "deltsn": deltsn,
            "snth": snth,
            "rhosn": columns["rhosn"],
            "rhonewsn": columns["rhonewsn"],
            "meltfactor": columns["meltfactor"],
            "prcpms": columns["prcpms"],
            "rainf": columns["rainf"],
            "patm": columns["patm"],
            "tabs": columns["tabs"],
            "qvatm": columns["qvatm"],
            "emiss": columns["emiss"],
            "rnet": columns["rnet"],
            "qkms": columns["qkms"],
            "tkms": columns["tkms"],
            "rho": columns["rho"],
            "vegfrac": columns["vegfrac"],
            "drycan": drycan,
            "wetcan": wetcan,
            "transum": transpiration.transum,
            "dew": np.zeros(horizontal_shape, dtype=np.float32),
            "soilt": columns["soilt"],
            "soilt1": columns["soilt1"],
            "qvg": columns["qvg"],
        },
        delt=float(timestep),
        conflx=constant_flux_depth,
        nroot=roots,
        ilnb=layers_flat.reshape(horizontal_shape),
        xlvm=float(xlvm),
        cvw=float(cvw),
    )
    # snowtemp returns tso rather than updating it in place.
    tso = snow_result.tso
    tso_flat = tso.reshape(NUM_SOIL_LAYERS, ncolumn)
    beta_flat[:] = snow_result.beta.reshape(ncolumn)
    layers_flat[:] = snow_result.ilnb.reshape(ncolumn)
    snwe_flat[:] = snow_result.snwe.reshape(ncolumn)
    snow_names = (
        "dew", "qcg", "qsg", "qvg", "rhosn", "rsm", "snflx", "snhei",
        "snheiprint", "snoh", "snweprint", "soilt", "soilt1", "tsnav", "x",
        "smelt",
    )
    # snowsoil's local x is snowtemp's storage; its own s is discarded,
    # because :3768 reports the snow-layer flux in its place.
    snow = {
        name: (
            snow_result.storage if name == "x" else getattr(snow_result, name)
        )
        for name in snow_names
    }
    snow_flat = {name: array.reshape(ncolumn) for name, array in snow.items()}

    # :3604-3626 recompute dew or transpiration against the new qsg.
    transp = np.zeros_like(soilmois)
    transp_flat = transp.reshape(NUM_SOIL_LAYERS, ncolumn)
    dew = np.zeros(horizontal_shape, dtype=np.float32)
    ett1 = np.zeros(horizontal_shape, dtype=np.float32)
    dew_flat = dew.reshape(ncolumn)
    ett_flat = ett1.reshape(ncolumn)
    qsg_flat = snow_flat["qsg"]
    for column in range(ncolumn):
        epot = np.float32(
            -np.float32(
                column_flat["qkms"][column]
                * np.float32(column_flat["qvatm"][column] - qsg_flat[column])
            )
        )
        if epot > zero:
            root_count = int(roots_flat[column])
            for level in range(root_count):
                flux = np.float32(
                    column_flat["vegfrac"][column] * ras_flat[column]
                )
                flux = np.float32(flux * column_flat["qkms"][column])
                flux = np.float32(
                    flux
                    * np.float32(column_flat["qvatm"][column] - qsg_flat[column])
                )
                flux = np.float32(flux * tranf_flat[level, column])
                flux = np.float32(flux * dry_flat[column])
                flux = np.float32(flux / zshalf[root_count])
                transp_flat[level, column] = flux
                ett_flat[column] = np.float32(ett_flat[column] - flux)
        else:
            dew_flat[column] = np.float32(-epot)

    phase = _ruc_phase_partition(
        soilmois, tso, smfrkeep, keepfr, columns, update_smfrkeep=False
    )

    precipitation = np.empty(horizontal_shape, dtype=np.float32)
    precipitation_flat = precipitation.reshape(ncolumn)
    for column in range(ncolumn):
        precipitation_flat[column] = np.float32(
            -column_flat["infwater"][column]
        )
    zeros = np.zeros(horizontal_shape, dtype=np.float32)
    moisture = ruc_soil_moisture_step(
        {
            "diffu": properties.diffu,
            "hydro": properties.hydro,
            "transp": transp,
            "soilice": phase["soilice"],
            "soilmois": soilmois,
            "soiliqw": phase["soiliqw"],
            "qsg": snow["qsg"],
            "qvg": snow["qvg"],
            "qcg": snow["qcg"],
            "qcatm": columns["qcatm"],
            "qvatm": columns["qvatm"],
            "prcp": precipitation,
            "qkms": columns["qkms"],
            "drip": zeros,
            "dew": zeros,
            "smelt": snow["smelt"],
            "vegfrac": columns["vegfrac"],
            "snowfrac": columns["snowfrac"],
            "soilres": np.ones(horizontal_shape, dtype=np.float32),
            "dqm": columns["dqm"],
            "qmin": columns["qmin"],
            "ref": columns["ref"],
            "ksat": columns["ksat"],
            "ras": ras,
        },
        delt=float(timestep),
    )
    soilmois = moisture.soilmois

    scalar_names = (
        "cst", "snom", "edir1", "ec1", "eeta", "qfx", "hfx", "sublim",
        "fltot",
    )
    output = {
        name: np.zeros(horizontal_shape, dtype=np.float32)
        for name in scalar_names
    }
    output_flat = {name: value.reshape(ncolumn) for name, value in output.items()}
    tsnav_flat = snow_flat["tsnav"]
    soilt_flat = snow_flat["soilt"]
    soilmois_flat = soilmois.reshape(NUM_SOIL_LAYERS, ncolumn)
    told_flat = told.reshape(NUM_SOIL_LAYERS, ncolumn)
    smold_flat = smold.reshape(NUM_SOIL_LAYERS, ncolumn)
    keepfr_flat = keepfr.reshape(NUM_SOIL_LAYERS, ncolumn)
    soilice_flat = phase["soilice"].reshape(NUM_SOIL_LAYERS, ncolumn)
    for column in range(ncolumn):
        # :3671-3677 restore the snow-free average when the pack is gone.
        if snow_flat["snhei"][column] == zero:
            tsnav_flat[column] = np.float32(soilt_flat[column] - freeze)
        output_flat["snom"][column] = np.float32(
            column_flat["snom"][column]
            + np.float32(
                np.float32(snow_flat["smelt"][column] * timestep)
                * np.float32(1.0e3)
            )
        )
        # :3688-3696 latch keepfr where rain fell on frozen soil.
        for level in range(NUM_SOIL_LAYERS):
            if soilice_flat[level, column] > zero:
                if (
                    tso_flat[level, column] > told_flat[level, column]
                    and soilmois_flat[level, column] > smold_flat[level, column]
                ):
                    keepfr_flat[level, column] = one
                else:
                    keepfr_flat[level, column] = zero

        # :3699-3701 t3/upflux/xinet are assigned and never read.
        sensible = np.float32(
            np.float32(column_flat["tkms"][column] * cp_air)
            * column_flat["rho"][column]
        )
        sensible = np.float32(
            sensible
            * np.float32(column_flat["tabs"][column] - soilt_flat[column])
        )
        hft = np.float32(-sensible)
        exner = np.float32(
            np.power(
                np.float64(
                    np.float32(reference / column_flat["patm"][column])
                ),
                np.float64(rovcp),
            )
        )
        output_flat["hfx"][column] = np.float32(-np.float32(sensible * exner))
        q1 = np.float32(
            -np.float32(
                np.float32(column_flat["qkms"][column] * ras_flat[column])
                * np.float32(column_flat["qvatm"][column] - qsg_flat[column])
            )
        )
        umveg = np.float32(one - column_flat["vegfrac"][column])
        if q1 < zero:
            dew_flat[column] = np.float32(
                column_flat["qkms"][column]
                * np.float32(column_flat["qvatm"][column] - qsg_flat[column])
            )
            eeta = np.float32(
                -np.float32(column_flat["rho"][column] * dew_flat[column])
            )
            output_flat["cst"][column] = np.float32(
                column_flat["cst"][column]
                + np.float32(
                    np.float32(
                        np.float32(timestep * dew_flat[column])
                        * ras_flat[column]
                    )
                    * column_flat["vegfrac"][column]
                )
            )
            output_flat["qfx"][column] = np.float32(xlvm * eeta)
            eeta = np.float32(
                -np.float32(column_flat["rho"][column] * dew_flat[column])
            )
            ett_flat[column] = zero
        else:
            edir1 = np.float32(np.float32(q1 * umveg) * beta_flat[column])
            ec1 = np.float32(
                np.float32(q1 * wet_flat[column])
                * column_flat["vegfrac"][column]
            )
            output_flat["edir1"][column] = edir1
            output_flat["ec1"][column] = ec1
            output_flat["cst"][column] = np.float32(
                max(
                    zero,
                    np.float32(
                        column_flat["cst"][column] - np.float32(ec1 * timestep)
                    ),
                )
            )
            total = np.float32(np.float32(edir1 + ec1) + ett_flat[column])
            eeta = np.float32(total * np.float32(1.0e3))
            output_flat["qfx"][column] = np.float32(xlvm * eeta)
            eeta = np.float32(total * np.float32(1.0e3))
        output_flat["eeta"][column] = eeta
        output_flat["sublim"][column] = np.float32(
            output_flat["edir1"][column] * np.float32(1.0e3)
        )
        balance = np.float32(column_flat["rnet"][column] - hft)
        balance = np.float32(balance - np.float32(xlvm * eeta))
        balance = np.float32(balance - snow_flat["snflx"][column])
        balance = np.float32(balance - snow_flat["snoh"][column])
        balance = np.float32(balance - snow_flat["x"][column])
        output_flat["fltot"][column] = balance

    result = RucSnowSoilStep(
        soilmois=soilmois,
        tso=tso,
        smfrkeep=smfrkeep,
        keepfr=keepfr,
        soilice=phase["soilice"],
        soiliqw=moisture.soiliqw,
        cst=output["cst"],
        dew=dew,
        soilt=snow["soilt"],
        soilt1=snow["soilt1"],
        tsnav=snow["tsnav"],
        qvg=snow["qvg"],
        qsg=snow["qsg"],
        qcg=snow["qcg"],
        snwe=snwe,
        snhei=snow["snhei"],
        rhosn=snow["rhosn"],
        ilnb=layers_flat.reshape(horizontal_shape),
        snweprint=snow["snweprint"],
        snheiprint=snow["snheiprint"],
        rsm=snow["rsm"],
        smelt=snow["smelt"],
        snoh=snow["snoh"],
        snflx=snow["snflx"],
        snom=output["snom"],
        edir1=output["edir1"],
        ec1=output["ec1"],
        ett1=ett1,
        eeta=output["eeta"],
        qfx=output["qfx"],
        hfx=output["hfx"],
        # :3768 snowsoil reports the snow-layer flux, not soiltemp's s.
        s=snow["snflx"],
        sublim=output["sublim"],
        prcpl=np.array(columns["prcpms"], dtype=np.float32, copy=True),
        fltot=output["fltot"],
        runoff1=moisture.runoff,
        runoff2=moisture.runoff2,
        mavail=moisture.mavail,
        infiltrp=moisture.infiltrp,
    )
    for name in RucSnowSoilStep.__dataclass_fields__:
        array = getattr(result, name)
        if array.dtype != np.int32 and not np.all(np.isfinite(array)):
            raise ValueError(f"RUC snowsoil produced non-finite {name}")
    return result


__all__ += [
    "RucSnowSoilStep",
    "RUC_SNOW_SOIL_COLUMN_INPUTS",
    "RUC_SNOW_SOIL_PROFILE_INPUTS",
    "ruc_snow_soil_step",
]


# ---------------------------------------------------------------------------
# WRF v4.6.1 sfctmp dispatch + mosaic recombination
# (phys/module_sf_ruclsm.F:1767-2196)
# ---------------------------------------------------------------------------

RUC_SFCTMP_PROFILE_INPUTS = (
    "soilm1d",
    "ts1d",
    "smfrkeep",
    "keepfr",
    "soilice",
    "soiliqw",
)
#: Every ``sfctmp`` argument the routine reads before something overwrites it.
#: The preparation prologue's own inputs come first, in the order
#: ``RUC_SNOW_PREP_COLUMN_INPUTS`` lists them, then the arguments only the
#: dispatch reads.  Arguments WRF passes that no path can observe are listed
#: in ``ruc_surface_temperature_step``'s docstring with the line that kills
#: them.
RUC_SFCTMP_COLUMN_INPUTS = RUC_SNOW_PREP_COLUMN_INPUTS + (
    "meltfactor",
    "patm",
    "qvatm",
    "qcatm",
    "rho",
    "glw",
    "qkms",
    "tkms",
    "pc",
    "mavail",
    "qwrtz",
    "rhocs",
    "dqm",
    "qmin",
    "ref",
    "wilt",
    "psis",
    "bclh",
    "ksat",
    "cn",
    "soilt1",
    "qvg",
    "qsg",
    "qcg",
    "snoh",
    "snflx",
    "snom",
    "acsnow",
    "s",
    "sublim",
    "evapl",
)
RUC_SFCTMP_PROFILE_OUTPUTS = RUC_SFCTMP_PROFILE_INPUTS
RUC_SFCTMP_COLUMN_OUTPUTS = (
    "snwe",
    "snhei",
    "snowfrac",
    "rhosn",
    "rhonewsn",
    "rhosnfall",
    "snowrat",
    "grauprat",
    "icerat",
    "curat",
    "alb_snow",
    "emiss",
    "mavail",
    "alb",
    "cst",
    "znt",
    "soilt",
    "soilt1",
    "tsnav",
    "dew",
    "qvg",
    "qsg",
    "qcg",
    "smelt",
    "snoh",
    "snflx",
    "snom",
    "snowfallac",
    "acsnow",
    "edir1",
    "ec1",
    "ett1",
    "eeta",
    "qfx",
    "hfx",
    "s",
    "sublim",
    "evapl",
    "prcpl",
    "fltot",
    "runoff1",
    "runoff2",
    "infiltr",
    "smf",
    "rsm",
    "snweprint",
    "snheiprint",
)


@dataclass(frozen=True)
class RucSurfaceTemperatureStep:
    """Every ``intent(inout)``/``intent(out)`` argument WRF ``sfctmp`` returns.

    The snow-preparation prologue (``:1400-1766``) and the dispatch
    (``:1767-2196``) together, so this is the complete routine.  Fields whose
    value no ``sfctmp`` path assigns -- ``snowrat``, ``grauprat``, ``icerat``,
    ``curat``, ``alb_snow`` and ``acsnow`` -- are carried so a caller sees the
    full returned state rather than having to know which arguments survive.
    """

    soilm1d: np.ndarray
    ts1d: np.ndarray
    smfrkeep: np.ndarray
    keepfr: np.ndarray
    soilice: np.ndarray
    soiliqw: np.ndarray
    iland: np.ndarray
    snwe: np.ndarray
    snhei: np.ndarray
    snowfrac: np.ndarray
    rhosn: np.ndarray
    rhonewsn: np.ndarray
    rhosnfall: np.ndarray
    snowrat: np.ndarray
    grauprat: np.ndarray
    icerat: np.ndarray
    curat: np.ndarray
    alb_snow: np.ndarray
    emiss: np.ndarray
    mavail: np.ndarray
    alb: np.ndarray
    cst: np.ndarray
    znt: np.ndarray
    soilt: np.ndarray
    soilt1: np.ndarray
    tsnav: np.ndarray
    dew: np.ndarray
    qvg: np.ndarray
    qsg: np.ndarray
    qcg: np.ndarray
    smelt: np.ndarray
    snoh: np.ndarray
    snflx: np.ndarray
    snom: np.ndarray
    snowfallac: np.ndarray
    acsnow: np.ndarray
    edir1: np.ndarray
    ec1: np.ndarray
    ett1: np.ndarray
    eeta: np.ndarray
    qfx: np.ndarray
    hfx: np.ndarray
    s: np.ndarray
    sublim: np.ndarray
    evapl: np.ndarray
    prcpl: np.ndarray
    fltot: np.ndarray
    runoff1: np.ndarray
    runoff2: np.ndarray
    infiltr: np.ndarray
    smf: np.ndarray
    rsm: np.ndarray
    snweprint: np.ndarray
    snheiprint: np.ndarray
    ilnb: np.ndarray


def _ruc_tanh_array(values, *, arrays=None) -> np.ndarray:
    """``_f32_tanh`` mapped elementwise.

    ``_f32_tanh`` spells out fdlibm's reduction with Python control flow, so
    it is scalar.  The dispatch needs it over a column field.

    This is the ONE call in the ``sfctmp`` dispatch that is not array
    arithmetic, and on a snow-covered grid it runs on every column: the
    snow-cover rebuild at ``:2087``/``:2098`` is inside the melt-out block, so
    it is reached once per column per call and its cost is a Python function
    call, not a float32 multiply.  An ``arrays`` namespace that carries a
    ``ruc_tanhf_glibc`` entry supplies the same reduction as a kernel; the
    two are the same source, transcribed once here and once in ``ruc.cu``,
    and ``tests/test_ruc_device_column.py`` pins them together.
    """

    device_tanh = (getattr(arrays, "ruc_tanhf_glibc", None)
                   if arrays is not None else None)
    if device_tanh is not None:
        return device_tanh(values)
    raw = np.asarray(values, dtype=np.float32)
    result = np.empty(raw.shape, dtype=np.float32)
    source = raw.reshape(-1)
    target = result.reshape(-1)
    for index in range(source.size):
        target[index] = _f32_tanh(source[index])
    return result


def _ruc_surface_net_radiation(emissivity, glw, soilt, gswnew, *, arrays=None):
    """``t3``/``upflux``/``xinet``/``rnet``, spelled in WRF's order.

    ``module_sf_ruclsm.F:1777-1780`` and its three copies at ``:1830-1833``,
    ``:1895-1898`` and ``:2125-2128``.  ``stbolt*soilt*soilt*soilt`` is
    left-associative, so the cube is never formed as ``soilt**3``.
    """

    np = arrays if arrays is not None else _NUMPY
    stbolt = np.float32(5.67051e-8)
    t3 = np.float32(
        np.float32(np.float32(stbolt * soilt) * soilt) * soilt
    )
    upflux = np.float32(t3 * soilt)
    xinet = np.float32(emissivity * np.float32(glw - upflux))
    return np.float32(gswnew + xinet), xinet


#: The four leaves ``sfctmp`` dispatches to.  Named so a backend can replace
#: them one at a time and so the set is a checkable list rather than four
#: call sites: ``:1770-1822`` mosaic land, ``:1823-1878`` mosaic sea ice,
#: ``:1905-1933`` the snow column on land, ``:1934-1976`` on sea ice, plus the
#: two snow-free arms at ``:2118-2195`` that reuse the same two routines.
RUC_SFCTMP_HOST_LEAVES: "Mapping[str, object]" = MappingProxyType({
    "soil": ruc_soil_step,
    "sea_ice": ruc_sea_ice_step,
    "snow_soil": ruc_snow_soil_step,
    "snow_sea_ice": ruc_snow_sea_ice_step,
})


#: The straight-line stage ``sfctmp`` runs before it dispatches to any leaf.
#:
#: It is deliberately NOT in :data:`RUC_SFCTMP_HOST_LEAVES`: that mapping is
#: the four branches ``:1767-2195`` chooses between, and a column takes
#: exactly one of them, whereas every column goes through the preparation
#: block.  It is injectable by the same mechanism and for the same reason --
#: it is a ``for column in range(ncolumn)`` loop, and once the leaf arithmetic
#: is on the card it is 60% of what a RUC land-surface call costs.
RUC_SFCTMP_HOST_STAGES: "Mapping[str, object]" = MappingProxyType({
    "snow_prep": ruc_snow_preparation,
})


def ruc_surface_temperature_step(
    values: Mapping[str, object],
    *,
    delt: float,
    conflx: float,
    ivgtyp,
    iland,
    nroot,
    ilnb: object = 1,
    isice: int = 15,
    c1sn: float = 0.026,
    c2sn: float = 21.0,
    myj: bool = False,
    cw: float = 4.183e6,
    xlv: float = 2.5e6,
    isncovr_opt: int = RUC_SNOW_COVER_OPTION,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    parameters: RucParameterBundle | None = None,
    leaves: "Mapping[str, object] | None" = None,
    stages: "Mapping[str, object] | None" = None,
    arrays=None,
) -> RucSurfaceTemperatureStep:
    """Transcribe WRF ``sfctmp`` end to end.

    ``phys/module_sf_ruclsm.F:1180-2198``.  The snow-preparation prologue is
    delegated to :func:`ruc_snow_preparation` (``:1400-1766``); everything
    below is the dispatch and the mosaic recombination (``:1767-2196``).

    The dispatch is four calls and two recombinations:

    * ``:1770-1822`` mosaic land -- ``soil`` runs on the snow-free fraction
      with ``iland=ivgtyp``, ``emiss_snowfree``, ``gswnew=gswin*(1-albedo of
      snow-free ground)`` and the canopy drip that the preparation block
      routed to liquid, writing a shadow copy of the whole column.
    * ``:1823-1878`` mosaic sea ice -- ``sice`` runs on the snow-free
      fraction with a literal emissivity of ``0.98``.
    * ``:1905-1933`` / ``:1934-1976`` the snow column itself, ``snowsoil`` on
      land and ``snowseaice`` on sea ice, over the whole cell when the mosaic
      is on (``snfr=1``) and over ``snowfrac`` when it is off.
    * ``:1979-2074`` recombination -- a ``snowfrac`` weighting of the shadow
      copy and the snow column, with a different (shorter) output list on sea
      ice.
    * ``:2077-2115`` melt-out reset, the snow-cover fraction rebuild, the
      urban cap and the snowfall accumulation.
    * ``:2118-2195`` the snow-free branch: ``soil`` on land, and on sea ice a
      rescale of the absorbed solar flux to the ice albedo followed by
      ``sice``.

    Arguments WRF passes to ``sfctmp`` that no path can observe are omitted
    from this signature, each for a reason in the source:

    ``xland``  never referenced in ``:1180-2198``.
    ``rhonewsn``  ``:1428`` assigns 100 before any read.
    ``smelt``  ``:1430``.  ``rsm``  ``:1432``.  ``infiltr``  ``:1434``.
    ``smf``  ``:1446``.  All four are zeroed by the preparation block.
    ``snweprint``/``snheiprint``  written by ``snowsoil``/``snowseaice`` on
    the snow path and zeroed at ``:2120-2121`` on the snow-free one.
    ``dew``, ``eeta``, ``qfx``, ``hfx``, ``prcpl``, ``fltot``, ``runoff1``,
    ``runoff2``  written by whichever of ``soil``/``sice``/``snowsoil``/
    ``snowseaice`` runs, on every one of the four paths.
    ``edir1``, ``ec1``, ``ett1``  the same, except on the two sea-ice paths
    where ``:1863-1865``/``:1961-1963``/``:2176-2178`` force them.
    ``isltyp``, ``ktau``, ``i``, ``j``, ``spp_lsm``, ``rstochcol``,
    ``fieldcol_sf``  carried through to leaves that gpuwm has already shown
    do not read them.

    ``s``, ``sublim`` and ``evapl`` are NOT in that list and are required
    inputs: ``snowseaice`` reads ``s``; ``sublim`` survives the snow-free
    branch untouched and is only scaled (``:2019``) or replaced (``:2058``)
    on the mosaic paths; ``evapl`` survives both snow paths untouched and is
    scaled at ``:2018``.

    ``ilnb`` is the one argument here that has no WRF counterpart, and it is
    not an invention: ``sfctmp`` declares ``ilnb`` as a plain local at
    ``:1385``, never initialises it, and passes it to ``snowsoil`` (``:1928``,
    ``intent(inout)`` at ``:3329``) and ``snowseaice`` (``:1956``,
    ``intent(inout)`` at ``:3896``).  Both of those assign it -- at ``:4038``/
    ``:4059`` for ``snowseaice`` and through ``snowtemp``'s ``:5124``/``:5148``
    for ``snowsoil`` -- ONLY when the pack clears the ``snth`` thin-snow
    threshold.  On the thin blended path nothing assigns it, and the
    ``if(ilnb.gt.1)`` at ``:4410``/``:5716`` then reads whatever gfortran left
    in that stack slot: the layer count from the PREVIOUS call to ``sfctmp``,
    i.e. the previous grid column.  It selects between the one-layer and
    two-layer ``tsnav`` forms, so it is observable.

    ``oracle/sfctmp.csv`` measures this directly.  Case 4
    (``shallow_snow_mosaic``, a 0.025 m pack) reproduces bitwise only with
    ``ilnb=2``, which is what case 3 (``deep_pack_densify``) left behind;
    with ``ilnb=1`` its ``tsnav`` is -9.5347 K against the fixture's
    -15.6717 K.  This is undefined behaviour in WRF, so gpuwm makes it an
    explicit argument rather than reproducing it by accident: a caller that
    wants WRF's answer threads the previous column's returned ``ilnb`` in,
    and a caller that wants a column-order-independent answer does not.
    Either way the dependence is visible.

    ``isncovr_opt`` is ``integer, parameter :: isncovr_opt=2`` at
    ``module_sf_ruclsm.F:78``.  Options 1 and 3 are transcribed at ``:2083``
    and ``:2098`` for completeness and are NOT oracle-verified.

    **``arrays``** is the array namespace the dispatch's masking, gathering
    and mosaic recombination run on.  ``None`` is numpy and is the reference
    path.  :data:`gpuwm.core.ruc_gpu.RUC_DEVICE_ARRAYS` is CuPy, and with it
    -- together with the device leaf and stage sets -- a column never returns
    to the host between the preparation block and the result: the inputs
    arrive as device arrays, every mask, gather, scatter and blend below is a
    kernel, and the returned dataclass holds device arrays.  Before that, the
    four leaves and the stage each crossed the host boundary once per field
    in and once per field out, which at 24,576 columns was more than three
    hundred separately synchronised copies for one call and 46 % of a warm
    call's wall clock.

    Nothing in the body knows which namespace it has.  There is one
    transcription of ``:1767-2196``, and the two paths are gated against each
    other at ``max_ulp 0`` over all 43 returned fields.
    """

    np = arrays if arrays is not None else _NUMPY
    if myj is not False:
        raise ValueError("RUC sfctmp lane supports myj=False only")
    timestep = np.float32(delt)
    constant_flux_depth = _ruc_constant_flux_depth(
        conflx, "sfctmp", arrays=arrays)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC sfctmp delt must be finite and positive")
    if isncovr_opt not in (1, 2, 3):
        raise ValueError("RUC isncovr_opt must be 1, 2, or 3")
    if type(isice) is not int:
        raise TypeError("RUC sfctmp isice must be an int")
    missing = [
        name
        for name in RUC_SFCTMP_PROFILE_INPUTS + RUC_SFCTMP_COLUMN_INPUTS
        if name not in values
    ]
    if missing:
        raise TypeError(f"missing RUC sfctmp inputs: {', '.join(missing)}")

    bundle = parameters if parameters is not None else load_ruc_parameters()
    vegetation = bundle.vegetation_for(mminlu)
    urban = int(vegetation.scalars["URBAN"])
    # ``leaves`` is how a device backend reaches this dispatch without
    # :mod:`gpuwm.core.ruc` importing cupy -- which it must not, because
    # ``tests/conftest.py`` auto-marks any module that does as ``gpu``, and
    # the whole RUC oracle suite runs on a machine with no card.  The four
    # entries are the only calls below that do arithmetic; everything else
    # here is masking and recombination.
    leaf = dict(RUC_SFCTMP_HOST_LEAVES)
    if leaves is not None:
        unknown = sorted(set(leaves) - set(leaf))
        if unknown:
            raise KeyError(f"unknown RUC sfctmp leaf override(s): {unknown}")
        leaf.update(leaves)
    stage = dict(RUC_SFCTMP_HOST_STAGES)
    if stages is not None:
        unknown = sorted(set(stages) - set(stage))
        if unknown:
            raise KeyError(f"unknown RUC sfctmp stage override(s): {unknown}")
        stage.update(stages)

    profiles: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in RUC_SFCTMP_PROFILE_INPUTS:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC sfctmp profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape}; expected shared profile shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    ncolumn = int(np.prod(horizontal_shape))
    columns = {
        name: _horizontal_float_field(
            values[name], horizontal_shape, name, arrays=arrays)
        for name in RUC_SFCTMP_COLUMN_INPUTS
    }
    vegetation_category = _horizontal_integer_field(
        ivgtyp, horizontal_shape, "ivgtyp", arrays=arrays
    ).reshape(ncolumn)
    roots = _root_count_field(
        nroot, horizontal_shape, arrays=arrays).reshape(ncolumn)

    prep = stage["snow_prep"](
        {
            "ts1d": profiles["ts1d"],
            **{name: columns[name] for name in RUC_SNOW_PREP_COLUMN_INPUTS},
        },
        delt=delt,
        ivgtyp=ivgtyp,
        iland=iland,
        isice=isice,
        c1sn=c1sn,
        c2sn=c2sn,
        isncovr_opt=isncovr_opt,
        mminlu=mminlu,
        bundle=bundle,
    )

    one = np.float32(1.0)
    zero = np.float32(0.0)

    # Working state.  Names track the Fortran; the preparation block's
    # results replace the incoming values wherever it wrote one.
    state: dict[str, np.ndarray] = {}
    for name in RUC_SFCTMP_PROFILE_INPUTS:
        state[name] = np.array(
            profiles[name].reshape(NUM_SOIL_LAYERS, ncolumn),
            dtype=np.float32,
            copy=True,
        )
    for name in RUC_SFCTMP_COLUMN_OUTPUTS:
        if hasattr(prep, name):
            state[name] = np.array(
                np.asarray(getattr(prep, name), dtype=np.float32).reshape(
                    ncolumn
                ),
                dtype=np.float32,
                copy=True,
            )
        elif name in columns:
            state[name] = np.array(
                columns[name].reshape(ncolumn), dtype=np.float32, copy=True
            )
        else:
            # Written on every path; the incoming value is unobservable.
            state[name] = np.zeros(ncolumn, dtype=np.float32)
    land_category = np.array(
        np.asarray(prep.iland, dtype=np.int32).reshape(ncolumn), copy=True
    )
    # Preparation-block locals the dispatch reads.
    local = {
        name: np.asarray(getattr(prep, name), dtype=np.float32).reshape(ncolumn)
        for name in (
            "snow_mosaic", "keep_snow_albedo", "newsn", "rainf", "drip",
            "dripsn", "dripliq", "infwater", "vegfrac", "gswin", "albice",
            "emissn", "emiss_snowfree", "snhei_crit",
        )
    }
    ice_profile = {
        name: np.asarray(getattr(prep, name), dtype=np.float32).reshape(
            NUM_SOIL_LAYERS, ncolumn
        )
        for name in ("capice", "thdifice")
    }
    flat = {name: columns[name].reshape(ncolumn) for name in columns}

    # WRF's uninitialised ``ilnb`` (:1385).  See the docstring: this is the
    # one place ``sfctmp`` reads a variable it never defines, and gfortran
    # hands it whatever the previous call to ``sfctmp`` left in the same
    # stack slot -- i.e. the PREVIOUS COLUMN's snow layer count.
    raw_layers = np.asarray(ilnb)
    if not np.issubdtype(raw_layers.dtype, np.integer):
        raise TypeError("RUC sfctmp ilnb must be an integer layer count")
    if raw_layers.shape != horizontal_shape:
        try:
            raw_layers = np.broadcast_to(raw_layers, horizontal_shape)
        except ValueError as exc:
            raise ValueError(
                f"ilnb shape {raw_layers.shape} is not broadcastable to "
                f"{horizontal_shape}"
            ) from exc
    snow_layers = np.array(
        raw_layers.astype(np.int32, copy=False).reshape(ncolumn), copy=True
    )

    # ``conflx`` carries the column axis now (see
    # :func:`_ruc_constant_flux_depth`), and every leaf below is called on a
    # MASKED subset of the columns, so the depth has to be masked with them.
    flux_depth = np.broadcast_to(constant_flux_depth, (ncolumn,))

    seaice_is_ice = flat["seaice"] >= np.float32(0.5)
    snow = state["snhei"] > zero
    mosaic = local["snow_mosaic"] == one

    # Shadow (snow-free fraction) copies -- WRF's ``*s`` locals.  Only the
    # columns the mosaic reaches are ever filled.
    shadow_profile = {
        name: np.zeros((NUM_SOIL_LAYERS, ncolumn), dtype=np.float32)
        for name in RUC_SFCTMP_PROFILE_INPUTS
    }
    shadow = {
        name: np.zeros(ncolumn, dtype=np.float32)
        for name in (
            "dew", "soilt", "qvg", "qsg", "qcg", "edir1", "ec1", "ett1",
            "eeta", "qfx", "hfx", "s", "evapl", "prcpl", "fltot", "runoff1",
            "runoff2", "mavail", "infiltr", "cst",
        )
    }

    # :1767-1822 -- mosaic land: ``soil`` on the snow-free fraction.
    mask = snow & mosaic & ~seaice_is_ice
    if np.any(mask):
        gswnew = np.float32(local["gswin"][mask] * np.float32(
            one - flat["alb_snow_free"][mask]
        ))
        rnet, _ = _ruc_surface_net_radiation(
            local["emiss_snowfree"][mask],
            flat["glw"][mask],
            state["soilt"][mask],
            gswnew,
            arrays=arrays,
        )
        free = leaf["soil"](
            {
                "soilmois": state["soilm1d"][:, mask],
                "tso": state["ts1d"][:, mask],
                "smfrkeep": state["smfrkeep"][:, mask],
                "keepfr": state["keepfr"][:, mask],
                "prcpms": flat["prcpms"][mask],
                "rainf": local["rainf"][mask],
                "patm": flat["patm"][mask],
                "qvatm": flat["qvatm"][mask],
                "qcatm": flat["qcatm"][mask],
                "glw": flat["glw"][mask],
                "gsw": gswnew,
                "gswin": local["gswin"][mask],
                "emiss": local["emiss_snowfree"][mask],
                "rnet": rnet,
                "qkms": flat["qkms"][mask],
                "tkms": flat["tkms"][mask],
                "pc": flat["pc"][mask],
                "cst": state["cst"][mask],
                "drip": local["dripliq"][mask],
                "infwater": local["infwater"][mask],
                "rho": flat["rho"][mask],
                "vegfrac": local["vegfrac"][mask],
                "lai": flat["lai"][mask],
                "qwrtz": flat["qwrtz"][mask],
                "rhocs": flat["rhocs"][mask],
                "dqm": flat["dqm"][mask],
                "qmin": flat["qmin"][mask],
                "ref": flat["ref"][mask],
                "wilt": flat["wilt"][mask],
                "psis": flat["psis"][mask],
                "bclh": flat["bclh"][mask],
                "ksat": flat["ksat"][mask],
                "sat": flat["sat"][mask],
                "cn": flat["cn"][mask],
                "tabs": flat["tabs"][mask],
                "soilt": state["soilt"][mask],
                "qvg": state["qvg"][mask],
                "qsg": state["qsg"][mask],
                "qcg": state["qcg"][mask],
                "mavail": flat["mavail"][mask],
            },
            # :1803 -- the snow-free fraction keeps the vegetation class.
            vegetation_category[mask],
            nroot=roots[mask],
            delt=delt,
            conflx=flux_depth[mask],
            myj=myj,
            mminlu=mminlu,
            parameters=bundle,
        )
        for name, source in (
            ("soilm1d", "soilmois"), ("ts1d", "tso"),
            ("smfrkeep", "smfrkeep"), ("keepfr", "keepfr"),
            ("soilice", "soilice"), ("soiliqw", "soiliqw"),
        ):
            shadow_profile[name][:, mask] = getattr(free, source)
        for name in shadow:
            shadow[name][mask] = getattr(
                free, "infiltrp" if name == "infiltr" else name
            )
        # :1822 -- ``soil``'s ``smf`` slot is the caller's own ``smf``.
        state["smf"][mask] = free.smf

    # :1823-1878 -- mosaic sea ice: ``sice`` on the snow-free fraction.
    mask = snow & mosaic & seaice_is_ice
    if np.any(mask):
        gswnew = np.float32(
            local["gswin"][mask] * np.float32(one - local["albice"][mask])
        )
        rnet, _ = _ruc_surface_net_radiation(
            local["emiss_snowfree"][mask],
            flat["glw"][mask],
            state["soilt"][mask],
            gswnew,
            arrays=arrays,
        )
        free_ice = leaf["sea_ice"](
            {
                "capice": ice_profile["capice"][:, mask],
                "thdifice": ice_profile["thdifice"][:, mask],
                "tso": state["ts1d"][:, mask],
                "prcpms": flat["prcpms"][mask],
                "rainf": local["rainf"][mask],
                "patm": flat["patm"][mask],
                "qvatm": flat["qvatm"][mask],
                # :1853 -- a literal 0.98, not ``emiss``.
                "emiss": np.full(
                    int(np.count_nonzero(mask)),
                    np.float32(0.98),
                    dtype=np.float32,
                ),
                "rnet": rnet,
                "qkms": flat["qkms"][mask],
                "tkms": flat["tkms"][mask],
                "rho": flat["rho"][mask],
                "tabs": flat["tabs"][mask],
                "soilt": state["soilt"][mask],
                "qvg": state["qvg"][mask],
                "qsg": state["qsg"][mask],
            },
            delt=delt,
            conflx=flux_depth[mask],
            myj=myj,
            cw=cw,
        )
        shadow_profile["ts1d"][:, mask] = free_ice.tso
        for name in (
            "dew", "soilt", "qvg", "qsg", "qcg", "eeta", "qfx", "hfx", "s",
            "evapl", "prcpl", "fltot",
        ):
            shadow[name][mask] = getattr(free_ice, name)
        # :1863-1877.  Every one of these is reassigned by :1961-1975 below
        # over exactly this mask, so the whole block is dead.  It is
        # transcribed anyway because "dead" is a claim about the source, not
        # a licence to skip it -- with one honest gap: :1863 multiplies the
        # ENTRY ``eeta``, which this port does not accept as an input
        # precisely because no path can observe it, so the factor here is
        # zero rather than WRF's entry value.  Both are overwritten before
        # ``sfctmp`` returns.
        state["edir1"][mask] = np.float32(
            state["eeta"][mask] * np.float32(1.0e-3)
        )
        state["ec1"][mask] = zero
        state["ett1"][mask] = zero
        state["runoff1"][mask] = flat["prcpms"][mask]
        state["runoff2"][mask] = zero
        state["mavail"][mask] = one
        state["infiltr"][mask] = zero
        state["cst"][mask] = zero
        for name, value in (
            ("soilm1d", one), ("soiliqw", zero), ("soilice", one),
            ("smfrkeep", one), ("keepfr", zero),
        ):
            state[name][:, mask] = value

    # :1892-1898 -- absorbed solar and net radiation for the snow albedo.
    snow_rnet = np.zeros(ncolumn, dtype=np.float32)
    mask = snow
    if np.any(mask):
        gswnew = np.float32(
            local["gswin"][mask] * np.float32(one - state["alb"][mask])
        )
        rnet, _ = _ruc_surface_net_radiation(
            state["emiss"][mask],
            flat["glw"][mask],
            state["soilt"][mask],
            gswnew,
            arrays=arrays,
        )
        snow_rnet[mask] = rnet

    # :1905-1933 -- the snow column on land.
    mask = snow & ~seaice_is_ice
    if np.any(mask):
        # :1907-1911
        snfr = np.where(mosaic[mask], one, state["snowfrac"][mask]).astype(
            np.float32
        )
        packed = leaf["snow_soil"](
            {
                "soilmois": state["soilm1d"][:, mask],
                "tso": state["ts1d"][:, mask],
                "smfrkeep": state["smfrkeep"][:, mask],
                "keepfr": state["keepfr"][:, mask],
                "meltfactor": flat["meltfactor"][mask],
                "rhonewsn": state["rhonewsn"][mask],
                "prcpms": flat["prcpms"][mask],
                "rainf": local["rainf"][mask],
                "newsnow": local["newsn"][mask],
                "snhei": state["snhei"][mask],
                "snwe": state["snwe"][mask],
                "snowfrac": snfr,
                "rhosn": state["rhosn"][mask],
                "patm": flat["patm"][mask],
                "qvatm": flat["qvatm"][mask],
                "qcatm": flat["qcatm"][mask],
                "emiss": state["emiss"][mask],
                "rnet": snow_rnet[mask],
                "qkms": flat["qkms"][mask],
                "tkms": flat["tkms"][mask],
                "pc": flat["pc"][mask],
                "cst": state["cst"][mask],
                "infwater": local["infwater"][mask],
                "rho": flat["rho"][mask],
                "vegfrac": local["vegfrac"][mask],
                "lai": flat["lai"][mask],
                "gswin": local["gswin"][mask],
                "qwrtz": flat["qwrtz"][mask],
                "rhocs": flat["rhocs"][mask],
                "dqm": flat["dqm"][mask],
                "qmin": flat["qmin"][mask],
                "ref": flat["ref"][mask],
                "wilt": flat["wilt"][mask],
                "psis": flat["psis"][mask],
                "bclh": flat["bclh"][mask],
                "ksat": flat["ksat"][mask],
                "sat": flat["sat"][mask],
                "cn": flat["cn"][mask],
                "tabs": flat["tabs"][mask],
                "soilt": state["soilt"][mask],
                "soilt1": state["soilt1"][mask],
                "qvg": state["qvg"][mask],
                "qsg": state["qsg"][mask],
                "qcg": state["qcg"][mask],
                "snom": state["snom"][mask],
            },
            land_category[mask],
            nroot=roots[mask],
            delt=delt,
            conflx=flux_depth[mask],
            ilnb=snow_layers[mask],
            myj=myj,
            cw=cw,
            mminlu=mminlu,
            parameters=bundle,
        )
        for name, source in (
            ("soilm1d", "soilmois"), ("ts1d", "tso"),
            ("smfrkeep", "smfrkeep"), ("keepfr", "keepfr"),
            ("soilice", "soilice"), ("soiliqw", "soiliqw"),
        ):
            state[name][:, mask] = getattr(packed, source)
        for name in (
            "cst", "dew", "soilt", "soilt1", "tsnav", "qvg", "qsg", "qcg",
            "snwe", "snhei", "rhosn", "snweprint", "snheiprint", "rsm",
            "smelt", "snoh", "snflx", "snom", "edir1", "ec1", "ett1", "eeta",
            "qfx", "hfx", "s", "sublim", "prcpl", "fltot", "runoff1",
            "runoff2", "mavail",
        ):
            state[name][mask] = getattr(packed, name)
        state["infiltr"][mask] = packed.infiltrp
        snow_layers[mask] = packed.ilnb

    # :1934-1976 -- the snow column on sea ice.
    mask = snow & seaice_is_ice
    if np.any(mask):
        snfr = np.where(mosaic[mask], one, state["snowfrac"][mask]).astype(
            np.float32
        )
        packed_ice = leaf["snow_sea_ice"](
            {
                "capice": ice_profile["capice"][:, mask],
                "thdifice": ice_profile["thdifice"][:, mask],
                "tso": state["ts1d"][:, mask],
                "meltfactor": flat["meltfactor"][mask],
                "rhonewsn": state["rhonewsn"][mask],
                "prcpms": flat["prcpms"][mask],
                "rainf": local["rainf"][mask],
                "newsnow": local["newsn"][mask],
                "snhei": state["snhei"][mask],
                "snwe": state["snwe"][mask],
                "snowfrac": snfr,
                "rhosn": state["rhosn"][mask],
                "patm": flat["patm"][mask],
                "qvatm": flat["qvatm"][mask],
                "emiss": state["emiss"][mask],
                "rnet": snow_rnet[mask],
                "qkms": flat["qkms"][mask],
                "tkms": flat["tkms"][mask],
                "rho": flat["rho"][mask],
                "alb": state["alb"][mask],
                "znt": state["znt"][mask],
                "tabs": flat["tabs"][mask],
                "soilt": state["soilt"][mask],
                "soilt1": state["soilt1"][mask],
                "tsnav": state["tsnav"][mask],
                "qvg": state["qvg"][mask],
                "qsg": state["qsg"][mask],
                "snom": state["snom"][mask],
                "s": state["s"][mask],
                "ilnb": snow_layers[mask],
            },
            delt=delt,
            conflx=flux_depth[mask],
            myj=myj,
            cw=cw,
            xlv=xlv,
        )
        state["ts1d"][:, mask] = packed_ice.tso
        for name in (
            "snweprint", "snheiprint", "rsm", "dew", "soilt", "soilt1",
            "tsnav", "qvg", "qsg", "qcg", "smelt", "snoh", "snflx", "snom",
            "eeta", "qfx", "hfx", "s", "sublim", "prcpl", "fltot", "snwe",
            "snhei", "rhosn", "emiss", "alb", "znt",
        ):
            state[name][mask] = getattr(packed_ice, name)
        snow_layers[mask] = packed_ice.ilnb
        # :1961-1975
        state["edir1"][mask] = np.float32(
            state["eeta"][mask] * np.float32(1.0e-3)
        )
        state["ec1"][mask] = zero
        state["ett1"][mask] = zero
        state["runoff1"][mask] = state["smelt"][mask]
        state["runoff2"][mask] = zero
        state["mavail"][mask] = one
        state["infiltr"][mask] = zero
        state["cst"][mask] = zero
        for name, value in (
            ("soilm1d", one), ("soiliqw", zero), ("soilice", one),
            ("smfrkeep", one), ("keepfr", zero),
        ):
            state[name][:, mask] = value

    # :1979-2039 -- mosaic recombination on land.
    mask = snow & mosaic & ~seaice_is_ice
    if np.any(mask):
        snowfrac = state["snowfrac"][mask]
        rest = np.float32(one - snowfrac)

        def blend(free_value, snow_value):
            return np.float32(
                np.float32(free_value * rest) + np.float32(snow_value * snowfrac)
            )

        for name in ("soilm1d", "smfrkeep", "soilice", "soiliqw", "ts1d"):
            state[name][:, mask] = blend(
                shadow_profile[name][:, mask], state[name][:, mask]
            )
        # :1997-2001 -- a selection, not a blend.
        state["keepfr"][:, mask] = np.where(
            snowfrac > np.float32(0.5),
            state["keepfr"][:, mask],
            shadow_profile["keepfr"][:, mask],
        ).astype(np.float32)
        for name in (
            "dew", "soilt", "qvg", "qsg", "qcg", "edir1", "ec1", "cst",
            "ett1", "eeta", "qfx", "hfx", "s", "prcpl", "fltot",
        ):
            state[name][mask] = blend(shadow[name][mask], state[name][mask])
        # :2018-2019 -- neither of these is a blend.
        state["evapl"][mask] = np.float32(shadow["evapl"][mask] * rest)
        state["sublim"][mask] = np.float32(state["sublim"][mask] * snowfrac)
        # :2023-2028
        state["alb"][mask] = np.maximum(
            np.float32(local["keep_snow_albedo"][mask] * state["alb"][mask]),
            np.minimum(
                np.float32(
                    flat["alb_snow_free"][mask]
                    + np.float32(
                        np.float32(
                            state["alb"][mask] - flat["alb_snow_free"][mask]
                        )
                        * snowfrac
                    )
                ),
                state["alb"][mask],
            ),
        )
        state["emiss"][mask] = np.maximum(
            np.float32(local["keep_snow_albedo"][mask] * local["emissn"][mask]),
            np.minimum(
                np.float32(
                    local["emiss_snowfree"][mask]
                    + np.float32(
                        np.float32(
                            local["emissn"][mask]
                            - local["emiss_snowfree"][mask]
                        )
                        * snowfrac
                    )
                ),
                local["emissn"][mask],
            ),
        )
        for name in ("runoff1", "runoff2"):
            state[name][mask] = blend(shadow[name][mask], state[name][mask])
        # :2032 -- ``1.*snowfrac`` is snowfrac.
        state["mavail"][mask] = np.float32(
            np.float32(shadow["mavail"][mask] * rest) + snowfrac
        )
        state["infiltr"][mask] = blend(
            shadow["infiltr"][mask], state["infiltr"][mask]
        )

    # :2040-2074 -- mosaic recombination on sea ice.  A shorter output list:
    # edir1, ec1, ett1, cst, mavail, infiltr and the soil arrays keep the
    # values :1961-1975 forced.
    mask = snow & mosaic & seaice_is_ice
    if np.any(mask):
        snowfrac = state["snowfrac"][mask]
        rest = np.float32(one - snowfrac)

        def blend(free_value, snow_value):
            return np.float32(
                np.float32(free_value * rest) + np.float32(snow_value * snowfrac)
            )

        state["ts1d"][:, mask] = blend(
            shadow_profile["ts1d"][:, mask], state["ts1d"][:, mask]
        )
        for name in (
            "dew", "soilt", "qvg", "qsg", "qcg", "eeta", "qfx", "hfx", "s",
        ):
            state[name][mask] = blend(shadow[name][mask], state[name][mask])
        # :2058 -- an assignment from the recombined eeta, not a blend.
        state["sublim"][mask] = state["eeta"][mask]
        for name in ("prcpl", "fltot"):
            state[name][mask] = blend(shadow[name][mask], state[name][mask])
        # :2062-2063 -- albice, but the slope is still alb-alb_snow_free.
        state["alb"][mask] = np.maximum(
            np.float32(local["keep_snow_albedo"][mask] * state["alb"][mask]),
            np.minimum(
                np.float32(
                    local["albice"][mask]
                    + np.float32(
                        np.float32(
                            state["alb"][mask] - flat["alb_snow_free"][mask]
                        )
                        * snowfrac
                    )
                ),
                state["alb"][mask],
            ),
        )
        state["emiss"][mask] = np.maximum(
            np.float32(local["keep_snow_albedo"][mask] * local["emissn"][mask]),
            np.minimum(
                np.float32(
                    local["emiss_snowfree"][mask]
                    + np.float32(
                        np.float32(
                            local["emissn"][mask]
                            - local["emiss_snowfree"][mask]
                        )
                        * snowfrac
                    )
                ),
                local["emissn"][mask],
            ),
        )
        for name in ("runoff1", "runoff2"):
            state[name][mask] = blend(shadow[name][mask], state[name][mask])

    # :2077-2115 -- melt-out reset, cover rebuild, urban cap, accumulation.
    mask = snow
    if np.any(mask):
        melted = mask & (state["snhei"] == zero)
        state["alb"][melted] = flat["alb_snow_free"][melted]
        land_category[melted] = vegetation_category[melted]

        remaining = mask & ~melted
        if np.any(remaining):
            snhei = state["snhei"][remaining]
            depth_fraction = np.minimum(
                one,
                np.float32(
                    snhei
                    / np.float32(
                        np.float32(2.0) * local["snhei_crit"][remaining]
                    )
                ),
            )
            if isncovr_opt == 1:
                state["snowfrac"][remaining] = depth_fraction
            elif isncovr_opt == 2:
                # (rhosn/rhonewsn)**1. is the quotient itself.
                cover = _ruc_tanh_array(np.float32(
                    snhei
                    / np.float32(
                        np.float32(
                            np.float32(2.5)
                            * np.minimum(
                                np.float32(0.2), state["znt"][remaining]
                            )
                        )
                        * np.float32(
                            state["rhosn"][remaining]
                            / state["rhonewsn"][remaining]
                        )
                    )
                ), arrays=arrays)
                state["snowfrac"][remaining] = np.float32(
                    np.float32(0.5) * np.float32(depth_fraction + cover)
                )
            else:
                factor = np.asarray(RUC_SNOW_COVER_FACTOR, dtype=np.float32)[
                    vegetation_category[remaining] - 1
                ]
                state["snowfrac"][remaining] = _ruc_tanh_array(np.float32(
                    snhei
                    / np.float32(
                        np.float32(np.float32(10.0) * factor)
                        * np.float32(
                            state["rhosn"][remaining]
                            / state["rhonewsn"][remaining]
                        )
                    )
                ), arrays=arrays)
        # :2111
        town = mask & (vegetation_category == urban)
        state["snowfrac"][town] = np.minimum(
            np.float32(0.75), state["snowfrac"][town]
        )
        # :2115
        state["snowfallac"][mask] = np.float32(
            state["snowfallac"][mask]
            + np.float32(local["newsn"][mask] * np.float32(1.0e3))
        )

    # :2118-2195 -- the snow-free branch.
    rnet_free = np.zeros(ncolumn, dtype=np.float32)
    xinet_free = np.zeros(ncolumn, dtype=np.float32)
    mask = ~snow
    if np.any(mask):
        state["snheiprint"][mask] = zero
        state["snweprint"][mask] = zero
        state["smelt"][mask] = zero
        # :2125-2128.  ``gswnew`` is still the :1460 value, the raw absorbed
        # solar flux, because the mosaic block is inside the snhei>0 branch.
        bare_rnet, bare_xinet = _ruc_surface_net_radiation(
            state["emiss"][mask],
            flat["glw"][mask],
            state["soilt"][mask],
            flat["gsw"][mask],
            arrays=arrays,
        )
        rnet_free[mask] = bare_rnet
        xinet_free[mask] = bare_xinet

    mask = ~snow & ~seaice_is_ice
    if np.any(mask):
        bare = leaf["soil"](
            {
                "soilmois": state["soilm1d"][:, mask],
                "tso": state["ts1d"][:, mask],
                "smfrkeep": state["smfrkeep"][:, mask],
                "keepfr": state["keepfr"][:, mask],
                "prcpms": flat["prcpms"][mask],
                "rainf": local["rainf"][mask],
                "patm": flat["patm"][mask],
                "qvatm": flat["qvatm"][mask],
                "qcatm": flat["qcatm"][mask],
                "glw": flat["glw"][mask],
                "gsw": flat["gsw"][mask],
                "gswin": local["gswin"][mask],
                "emiss": state["emiss"][mask],
                "rnet": rnet_free[mask],
                "qkms": flat["qkms"][mask],
                "tkms": flat["tkms"][mask],
                "pc": flat["pc"][mask],
                "cst": state["cst"][mask],
                "drip": local["drip"][mask],
                "infwater": local["infwater"][mask],
                "rho": flat["rho"][mask],
                "vegfrac": local["vegfrac"][mask],
                "lai": flat["lai"][mask],
                "qwrtz": flat["qwrtz"][mask],
                "rhocs": flat["rhocs"][mask],
                "dqm": flat["dqm"][mask],
                "qmin": flat["qmin"][mask],
                "ref": flat["ref"][mask],
                "wilt": flat["wilt"][mask],
                "psis": flat["psis"][mask],
                "bclh": flat["bclh"][mask],
                "ksat": flat["ksat"][mask],
                "sat": flat["sat"][mask],
                "cn": flat["cn"][mask],
                "tabs": flat["tabs"][mask],
                "soilt": state["soilt"][mask],
                "qvg": state["qvg"][mask],
                "qsg": state["qsg"][mask],
                "qcg": state["qcg"][mask],
                "mavail": flat["mavail"][mask],
            },
            land_category[mask],
            nroot=roots[mask],
            delt=delt,
            conflx=flux_depth[mask],
            myj=myj,
            mminlu=mminlu,
            parameters=bundle,
        )
        for name, source in (
            ("soilm1d", "soilmois"), ("ts1d", "tso"),
            ("smfrkeep", "smfrkeep"), ("keepfr", "keepfr"),
            ("soilice", "soilice"), ("soiliqw", "soiliqw"),
        ):
            state[name][:, mask] = getattr(bare, source)
        for name in (
            "cst", "dew", "soilt", "qvg", "qsg", "qcg", "edir1", "ec1",
            "ett1", "eeta", "qfx", "hfx", "s", "evapl", "prcpl", "fltot",
            "runoff1", "runoff2", "mavail", "smf",
        ):
            state[name][mask] = getattr(bare, name)
        state["infiltr"][mask] = bare.infiltrp

    mask = ~snow & seaice_is_ice
    if np.any(mask):
        # :2158-2160 -- rescale the absorbed solar flux to the ice albedo.
        gswnew = np.where(
            state["alb"][mask] != local["albice"][mask],
            np.float32(
                np.float32(
                    flat["gsw"][mask] / np.float32(one - state["alb"][mask])
                )
                * np.float32(one - local["albice"][mask])
            ),
            flat["gsw"][mask],
        ).astype(np.float32)
        state["alb"][mask] = local["albice"][mask]
        rnet = np.float32(gswnew + xinet_free[mask])
        bare_ice = leaf["sea_ice"](
            {
                "capice": ice_profile["capice"][:, mask],
                "thdifice": ice_profile["thdifice"][:, mask],
                "tso": state["ts1d"][:, mask],
                "prcpms": flat["prcpms"][mask],
                "rainf": local["rainf"][mask],
                "patm": flat["patm"][mask],
                "qvatm": flat["qvatm"][mask],
                "emiss": state["emiss"][mask],
                "rnet": rnet,
                "qkms": flat["qkms"][mask],
                "tkms": flat["tkms"][mask],
                "rho": flat["rho"][mask],
                "tabs": flat["tabs"][mask],
                "soilt": state["soilt"][mask],
                "qvg": state["qvg"][mask],
                "qsg": state["qsg"][mask],
            },
            delt=delt,
            conflx=flux_depth[mask],
            myj=myj,
            cw=cw,
        )
        state["ts1d"][:, mask] = bare_ice.tso
        for name in (
            "dew", "soilt", "qvg", "qsg", "qcg", "eeta", "qfx", "hfx", "s",
            "evapl", "prcpl", "fltot",
        ):
            state[name][mask] = getattr(bare_ice, name)
        # :2176-2190
        state["edir1"][mask] = np.float32(
            state["eeta"][mask] * np.float32(1.0e-3)
        )
        state["ec1"][mask] = zero
        state["ett1"][mask] = zero
        state["runoff1"][mask] = flat["prcpms"][mask]
        state["runoff2"][mask] = zero
        state["mavail"][mask] = one
        state["infiltr"][mask] = zero
        state["cst"][mask] = zero
        for name, value in (
            ("soilm1d", one), ("soiliqw", zero), ("soilice", one),
            ("smfrkeep", one), ("keepfr", zero),
        ):
            state[name][:, mask] = value

    result = RucSurfaceTemperatureStep(
        iland=land_category.reshape(horizontal_shape),
        ilnb=snow_layers.reshape(horizontal_shape),
        **{
            name: state[name].reshape(shape)
            for name in RUC_SFCTMP_PROFILE_OUTPUTS
        },
        **{
            name: state[name].reshape(horizontal_shape)
            for name in RUC_SFCTMP_COLUMN_OUTPUTS
        },
    )
    # One reduction per field, but ONE host read for all of them.  Read field
    # by field this loop is 53 separate stream synchronisations on the device
    # namespace, each costing what a synchronisation costs whether it moves
    # four bytes or a megabyte; stacked, the whole check is one.  The answer
    # is the same and so is the message.
    checked = [name for name in RucSurfaceTemperatureStep.__dataclass_fields__
               if getattr(result, name).dtype != np.int32]
    finite = np.stack([
        np.all(np.isfinite(getattr(result, name))) for name in checked])
    for name, ok in zip(checked, finite.tolist()):
        if not ok:
            raise ValueError(f"RUC sfctmp produced non-finite {name}")
    return result


__all__ += [
    "RucSurfaceTemperatureStep",
    "RUC_SFCTMP_COLUMN_INPUTS",
    "RUC_SFCTMP_COLUMN_OUTPUTS",
    "RUC_SFCTMP_PROFILE_INPUTS",
    "RUC_SFCTMP_PROFILE_OUTPUTS",
    "ruc_surface_temperature_step",
]


# ---------------------------------------------------------------------------
# WRF v4.6.1 ``phys/module_sf_ruclsm.F:84-1175`` subroutine ``LSMRUC``:
# the RUC land-surface driver.
# ---------------------------------------------------------------------------

#: Surface and lowest-model-level forcing ``LSMRUC`` reads.  ``z3d``, ``p8w``,
#: ``t3d``, ``qv3d``, ``qc3d`` and ``rho3d`` are three-dimensional in WRF, but
#: ``:602-610``, ``:836`` and ``:1063`` are the only reads and every one of
#: them is at level ``kms``, so they are taken here already reduced to that
#: level.
RUC_DRIVER_COLUMN_FORCING = (
    "z3d",
    "p8w",
    "t3d",
    "qv3d",
    "qc3d",
    "rho3d",
    "rainbl",
    "frzfrac",
    "glw",
    "gsw",
    "chs",
    "flqc",
    "flhc",
    "albbck",
    "xland",
    "xice",
    "tbot",
    "shdmin",
    "shdmax",
    "vegfra",
)

#: Every two-dimensional ``intent(inout)``/``intent(out)`` argument, in the
#: order WRF declares them.  ``vegfra`` is declared ``intent(inout)`` at
#: ``:236`` but ``SOILVEGIN`` takes it ``intent(in)`` (``:6560``) and no line
#: of ``LSMRUC`` assigns it, so it is carried as forcing above rather than as
#: state.
RUC_DRIVER_COLUMN_STATE = (
    "snow",
    "snowh",
    "snowc",
    "canwat",
    "snoalb",
    "alb",
    "emiss",
    "lai",
    "mavail",
    "sfcexc",
    "z0",
    "znt",
    "soilt",
    "hfx",
    "qfx",
    "lh",
    "sfcevp",
    "sfcrunoff",
    "udrunoff",
    "acrunoff",
    "grdflx",
    "acsnow",
    "snom",
    "qvg",
    "qcg",
    "dew",
    "qsfc",
    "qsg",
    "chklowq",
    "soilt1",
    "tsnav",
    "smavail",
    "smmax",
    "rhosnf",
    "precipfr",
    "snowfallac",
)

#: The soil-column state, in gpuwm's ``(soil_level, ...horizontal...)`` order.
RUC_DRIVER_PROFILE_STATE = (
    "soilmois",
    "sh2o",
    "tso",
    "smfr3d",
    "keepfr3dflag",
)

#: ``LSMRUC`` locals that ``SFCTMP`` reads before anything writes them.  WRF
#: zeroes all five in the ``ktau==1`` block (``:531-542``) and on every later
#: step reads whatever is left in that stack region -- they are automatic
#: arrays of the driver, not arguments, so nothing carries them forward by
#: design.  gpuwm defines them as zero on every call; see the
#: :func:`ruc_land_surface_step` docstring for the measurement that says the
#: choice is unobservable on the pinned fixture.
RUC_DRIVER_STALE_LOCALS = ("snoh", "snflx", "s", "sublim", "evapl")


def _ruc_concatenate_sfctmp(
    pieces: "list[RucSurfaceTemperatureStep]",
) -> "RucSurfaceTemperatureStep":
    """Join single-column ``sfctmp`` results back into one column field.

    Only the ``ilnb_chain=True`` path needs this.  That chain is a genuine
    sequential dependence -- column ``i``'s seed is column ``i-1``'s answer,
    because WRF never initialises ``ilnb`` -- so those columns are dispatched
    one at a time through the same call the batched path uses, and rejoined
    here rather than scattered field by field at the call site.
    """

    joined: dict[str, np.ndarray] = {}
    for name in RucSurfaceTemperatureStep.__dataclass_fields__:
        parts = [np.asarray(getattr(piece, name)) for piece in pieces]
        joined[name] = np.concatenate(parts, axis=parts[0].ndim - 1)
    return RucSurfaceTemperatureStep(**joined)


@dataclass(frozen=True)
class RucLandSurfaceStep:
    """Every argument WRF ``LSMRUC`` returns, plus the carried ``ilnb``.

    Field names and units are WRF's.  Note that ``qfx`` is ``SFCTMP``'s
    ``eeta`` -- the driver's actual argument list swaps the two at
    ``:971-972`` -- so ``qfx`` here is the surface moisture flux in
    kg m-2 s-1 and ``lh`` is the latent heat flux in W m-2, which is what the
    rest of WRF reads out of the RUC driver.
    """

    snow: np.ndarray
    snowh: np.ndarray
    snowc: np.ndarray
    canwat: np.ndarray
    snoalb: np.ndarray
    alb: np.ndarray
    emiss: np.ndarray
    lai: np.ndarray
    mavail: np.ndarray
    sfcexc: np.ndarray
    z0: np.ndarray
    znt: np.ndarray
    soilt: np.ndarray
    hfx: np.ndarray
    qfx: np.ndarray
    lh: np.ndarray
    sfcevp: np.ndarray
    sfcrunoff: np.ndarray
    udrunoff: np.ndarray
    acrunoff: np.ndarray
    grdflx: np.ndarray
    acsnow: np.ndarray
    snom: np.ndarray
    qvg: np.ndarray
    qcg: np.ndarray
    dew: np.ndarray
    qsfc: np.ndarray
    qsg: np.ndarray
    chklowq: np.ndarray
    soilt1: np.ndarray
    tsnav: np.ndarray
    smavail: np.ndarray
    smmax: np.ndarray
    rhosnf: np.ndarray
    precipfr: np.ndarray
    snowfallac: np.ndarray
    soilmois: np.ndarray
    sh2o: np.ndarray
    tso: np.ndarray
    smfr3d: np.ndarray
    keepfr3dflag: np.ndarray
    #: ``SFCTMP``'s uninitialised ``ilnb`` after the last column, so a caller
    #: that wants WRF's column-order-dependent answer can thread it into the
    #: next call.  Never required as an input.
    ilnb: np.ndarray
    #: Driver locals a caller wants as diagnostics.  WRF keeps them as
    #: automatic arrays and does not return them.
    infiltr: np.ndarray
    smelt: np.ndarray
    runoff1: np.ndarray
    runoff2: np.ndarray


def _ruc_driver_soil_geometry(
    levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``phys/module_sf_ruclsm.F:699-704``: ``zsmain`` and ``zshalf``.

    ``zsmain(1)`` is forced to 0 rather than copied from ``zs(1)``, and
    ``zshalf(1)`` is 0 rather than the midpoint the loop would give.
    """

    zsmain = np.empty(NUM_SOIL_LAYERS, dtype=np.float32)
    zshalf = np.empty(NUM_SOIL_LAYERS, dtype=np.float32)
    zsmain[0] = np.float32(0.0)
    zshalf[0] = np.float32(0.0)
    for level in range(1, NUM_SOIL_LAYERS):
        zsmain[level] = levels[level]
        zshalf[level] = np.float32(
            np.float32(0.5) * np.float32(zsmain[level - 1] + zsmain[level])
        )
    return zsmain, zshalf


def ruc_land_surface_step(
    values: Mapping[str, object],
    *,
    dt: float,
    ktau: int,
    zs,
    ivgtyp,
    isltyp,
    myj: bool = False,
    frpcpn: bool = True,
    rdlai2d: bool = False,
    mosaic_lu: int = 0,
    mosaic_soil: int = 0,
    iswater: int | None = None,
    isice: int | None = None,
    xice_threshold: float = 0.5,
    ilnb: int = 1,
    ilnb_chain: bool = True,
    c1sn: float = 0.026,
    c2sn: float = 21.0,
    isncovr_opt: int = RUC_SNOW_COVER_OPTION,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    parameters: RucParameterBundle | None = None,
    leaves: "Mapping[str, object] | None" = None,
    stages: "Mapping[str, object] | None" = None,
    arrays=None,
) -> RucLandSurfaceStep:
    """Transcribe WRF ``LSMRUC``, ``phys/module_sf_ruclsm.F:84-1175``.

    This is the RUC driver: the first-step initialisation block, the
    per-column unit conversions, the precipitation partition, the water /
    sea-ice / land dispatch, the call into ``SFCTMP``, and the diagnostics
    and accumulators on the way out.  Everything below ``SFCTMP`` is already
    ported -- :func:`ruc_surface_temperature_step` and the leaves under it --
    so the body here is orchestration, not new physics.

    **Configuration.**  ``EM_CORE==0``, which is how every RUC fixture in
    ``gpuwm/data/ruc/oracle`` is compiled.  Under ``EM_CORE==1`` WRF
    partitions precipitation from ``rainncv``/``snowncv``/``graupelncv``
    (``:618-652``) instead of from ``rainbl`` and ``frzfrac`` (``:654-660``),
    and adds SPP perturbation arrays and a lake bypass.  None of those three
    arms is in the pinned object, so none is transcribed here; ``frpcpn``
    selects between ``:654-660`` and ``:664-673`` only.

    ``mosaic_lu`` and ``mosaic_soil`` must be 0.  That is not an extra
    restriction: :func:`ruc_surface_parameters` is fail-closed on
    ``SOILVEGIN``'s mosaic arms, and ``LSMRUC``'s irrigation block
    (``:984-1009``) is gated on the same ``mosaic_lu == 1``, so the irrigation
    block is unreachable wherever ``SOILVEGIN`` is.  It is therefore not
    transcribed either.

    **Two WRF defects reproduced on purpose.**

    ``sfcevp`` is accumulated TWICE per column, at ``:1095`` and again at
    ``:1116``, with nothing in between changing ``qfx``.  The RUC driver
    therefore reports twice the accumulated surface moisture flux.  That is a
    duplicated statement, not undefined behaviour, so gpuwm reproduces it and
    ``oracle/lsmruc.csv`` pins it: ``sfcevp`` advances by ``2*qfx*dt``.

    ``rhosnf``, ``precipfr`` and ``snowfallac`` are declared ``intent(out)``
    at ``:344-347``, but ``rhosnf`` is read at ``:695`` into ``rhosnfall`` and
    ``snowfallac`` is read by ``SFCTMP`` at ``:1641`` to build
    ``snowfracnewsn`` before ``:2115`` accumulates it.  gfortran does not
    clear ``intent(out)`` reals, so in WRF both keep the caller's registry
    value and the accumulation works.  gpuwm takes both as live inputs, which
    is what the declaration should have said.

    **Dead computations omitted.**  ``:1109-1141`` builds ``wb``,
    ``waterbudget`` and ``acwaterbudget`` out of driver locals that no
    argument ever reaches (``ac`` at ``:1126`` even subtracts ``canwatold``
    in metres from ``canwat`` in millimetres), and ``:906-908`` is an empty
    ``if(ktau.gt.1)``.  Neither can change a returned value.

    **Undefined reads, and what gpuwm defines instead.**

    ``SFCTMP`` declares ``ilnb`` at ``:1385`` and never initialises it, so
    inside the column loop it carries the PREVIOUS COLUMN's snow-layer count
    (see :func:`ruc_surface_temperature_step`).  gpuwm threads that chain
    explicitly under ``ilnb_chain=True``: ``ilnb`` here seeds the first
    column, each column's returned value seeds the next, and the value after
    the last column comes back in the result.  The chain makes the answer
    depend on column order, which is WRF's behaviour and not something a
    data-parallel lane can reproduce.

    ``ilnb_chain=False`` is gpuwm's DEFINED behaviour, and it is what the
    forecast runtime uses (:mod:`gpuwm.core.ruc_runtime`): every column is
    seeded from ``ilnb`` and no column inherits another column's value, so
    the answer is order-independent.  ``ilnb=1`` is not an arbitrary pick.
    The unassigned read happens on exactly ``0 < snhei < snth``
    (``:4038``/``:4059`` and ``:5124``/``:5148`` assign it only inside
    ``if(snhei.ge.snth)``), and there ``deltsn = 5*snth > snhei``
    (``:3387-3388``/``:3945-3946``), so the two-layer form
    ``0.5/snhei*((soilt+soilt1)*deltsn + (soilt1+tso(1))*(snhei-deltsn))``
    weights a NEGATIVE thickness.  A pack thinner than the depth at which
    WRF models one layer cannot have two: ``ilnb=1`` is the defined answer,
    not merely a safe one.

    The default stays ``True`` because ``oracle/lsmruc.csv`` was generated by
    the unmodified module and therefore encodes the chain; a fixture test
    must be able to reproduce what WRF actually did.

    ``snoh``, ``snflx``, ``s``, ``sublim`` and ``evapl``
    (:data:`RUC_DRIVER_STALE_LOCALS`) are driver automatic arrays that
    ``SFCTMP`` reads.  WRF zeroes them in the ``ktau==1`` block and on every
    later step reads whatever is left in that stack region.  gpuwm defines
    them as zero on every call.  ``oracle/lsmruc_stackfill.csv`` measures the
    cost of that choice directly: it is the same driver with the callee stack
    filled with a nonzero pattern instead of zeroed, and over the fixture's
    432 rows x 122 numeric columns the two files differ only in ``tsnav``, on
    the two thin-snow columns where ``ilnb`` is read.  So on that fixture the
    five stale locals are unobservable and ``ilnb`` is not.

    **The column loop is a batching, not a loop over physics.**  ``LSMRUC``
    is written as ``DO j / DO i`` around a scalar ``SFCTMP`` call, and this
    port was transcribed the same way: a Python ``for i in range(ncolumn)``
    that called :func:`ruc_surface_temperature_step` once per column with
    one-element arrays.  That cost 1.7 ms per land column and did not fall
    with the column count, because ``sfctmp`` and every leaf under it are
    already written over a column axis -- the loop was paying ``sfctmp``'s
    fixed per-call cost 360,000 times for a nest instead of once.

    The prologue (``:612-935``), the water and sea-ice arms (``:824-895``)
    and the epilogue (``:1024-1116``) are therefore evaluated over the whole
    column axis here, and ``SFCTMP`` is called ONCE for all land columns.
    Every expression keeps WRF's float32 grouping; the only thing that
    changed is how many columns each evaluation covers.

    ``ilnb_chain=True`` is the exception and it cannot be batched, because
    the chain is a genuine sequential dependence: column ``i``'s seed is
    column ``i-1``'s answer.  Under that flag the SAME dispatch runs once per
    column, so there is one implementation of the physics and two batchings
    of it, not two implementations.  ``ilnb_chain=False`` -- gpuwm's defined
    behaviour, and what the forecast runtime uses -- has no such dependence
    and takes the single batched call.

    Soil arrays use gpuwm's ``(soil_level, ...horizontal...)`` order.  Inputs
    are copied to float32; this function never mutates caller arrays.

    **``arrays``** is the array namespace the whole driver runs on.  ``None``
    is numpy -- the reference path the oracle fixtures compare against WRF.
    :data:`gpuwm.core.ruc_gpu.RUC_DEVICE_ARRAYS`, with the resident leaf and
    stage sets, makes the WHOLE COLUMN device-resident: the caller hands in
    device arrays, the prologue's unit conversions, the water and sea-ice
    arms, the ``SFCTMP`` dispatch and the epilogue's accumulators are all
    kernels, and the returned dataclass holds device arrays.  Nothing crosses
    the host boundary except the handful of scalars that decide control flow.

    It is fail-closed on ``ilnb_chain=True``: that path is a genuine
    sequential dependence over columns -- WRF's uninitialised ``ilnb`` -- and
    dispatching one column at a time to the device would be slower than the
    host and is not what a forecast runs.  The forecast runtime uses
    ``ilnb_chain=False``, which is gpuwm's DEFINED behaviour.
    """

    np = arrays if arrays is not None else _NUMPY
    if arrays is not None and ilnb_chain:
        raise ValueError(
            "RUC driver ilnb_chain=True is a per-column sequential "
            "dependence on WRF's uninitialised ilnb and is refused on a "
            "non-host array namespace; the forecast runtime uses "
            "ilnb_chain=False, which is gpuwm's defined behaviour")
    if type(mosaic_lu) is not int or mosaic_lu != 0:
        raise ValueError("RUC driver currently requires mosaic_lu=0")
    if type(mosaic_soil) is not int or mosaic_soil != 0:
        raise ValueError("RUC driver currently requires mosaic_soil=0")
    if myj is not False:
        # Same gate as ``ruc_surface_temperature_step``: ``ruc_soil_step``
        # and ``ruc_snow_soil_step`` are fail-closed on ``myj=True``, so the
        # ``:681-682`` arm below is unreachable and unverified.
        raise ValueError("RUC driver lane supports myj=False only")
    if type(frpcpn) is not bool:
        raise TypeError("frpcpn must be bool")
    if type(rdlai2d) is not bool:
        raise TypeError("rdlai2d must be bool")
    if type(ktau) is not int or ktau < 1:
        raise ValueError("RUC driver ktau must be a positive int")
    if type(ilnb) is not int:
        raise TypeError("RUC driver ilnb seed must be an int")
    if type(ilnb_chain) is not bool:
        raise TypeError("RUC driver ilnb_chain must be bool")
    if isncovr_opt not in (1, 2, 3):
        raise ValueError("RUC isncovr_opt must be 1, 2, or 3")

    timestep = np.float32(dt)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC driver dt must be finite and positive")
    threshold = np.float32(xice_threshold)
    if not np.isfinite(threshold):
        raise ValueError("RUC driver xice_threshold must be finite")

    # ``zs`` stays on the HOST whatever the namespace is: it is a pinned
    # nine-element constant, ``_ruc_driver_soil_geometry`` walks it scalar by
    # scalar, and the depths it produces are broadcast against column fields
    # rather than indexed by them.
    levels = _NUMPY.asarray(zs, dtype=_NUMPY.float32)
    pinned_levels, _ = ruc_soil_geometry()
    if levels.shape != pinned_levels.shape or not _NUMPY.array_equal(
        levels, pinned_levels
    ):
        raise ValueError(
            "RUC driver zs must be the pinned nine-level RUC grid "
            f"{tuple(float(v) for v in pinned_levels)}"
        )

    bundle = parameters if parameters is not None else load_ruc_parameters()
    vegetation = bundle.vegetation_for(mminlu)
    ncategory = len(vegetation.rows)
    if iswater is None:
        water_category = 16 if vegetation.name == "USGS-RUC" else 17
    elif type(iswater) is int and 1 <= iswater <= ncategory:
        water_category = iswater
    else:
        raise ValueError(f"RUC iswater {iswater!r} is outside 1..{ncategory}")
    if isice is None:
        ice_category = 24 if vegetation.name == "USGS-RUC" else 15
    elif type(isice) is int and 1 <= isice <= ncategory:
        ice_category = isice
    else:
        raise ValueError(f"RUC isice {isice!r} is outside 1..{ncategory}")

    missing = [
        name
        for name in RUC_DRIVER_PROFILE_STATE
        + RUC_DRIVER_COLUMN_STATE
        + RUC_DRIVER_COLUMN_FORCING
        if name not in values
    ]
    if missing:
        raise TypeError(f"missing RUC driver inputs: {', '.join(missing)}")

    shape: tuple[int, ...] | None = None
    profiles: dict[str, np.ndarray] = {}
    for name in RUC_DRIVER_PROFILE_STATE:
        array = np.asarray(values[name], dtype=np.float32)
        if shape is None:
            if array.ndim < 2 or array.shape[0] != NUM_SOIL_LAYERS:
                raise ValueError(
                    "RUC driver profiles must have shape "
                    f"({NUM_SOIL_LAYERS}, ...horizontal...), got {array.shape}"
                )
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"{name} shape {array.shape}; expected {shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        profiles[name] = array
    assert shape is not None
    horizontal_shape = shape[1:]
    ncolumn = int(np.prod(horizontal_shape))

    columns: dict[str, np.ndarray] = {}
    for name in RUC_DRIVER_COLUMN_STATE + RUC_DRIVER_COLUMN_FORCING:
        columns[name] = np.array(
            _horizontal_float_field(
                values[name], horizontal_shape, name, arrays=arrays
            ).reshape(ncolumn),
            dtype=np.float32,
            copy=True,
        )
    vegetation_category = _horizontal_integer_field(
        ivgtyp, horizontal_shape, "ivgtyp", arrays=arrays
    ).reshape(ncolumn)
    soil_category = _horizontal_integer_field(
        isltyp, horizontal_shape, "isltyp", arrays=arrays
    ).reshape(ncolumn)
    if np.any((vegetation_category < 1) | (vegetation_category > ncategory)):
        raise ValueError(
            f"RUC ivgtyp is outside 1..{ncategory} for {vegetation.name}"
        )
    if np.any((soil_category < 1) | (soil_category > len(bundle.soil.rows))):
        raise ValueError(f"RUC isltyp is outside 1..{len(bundle.soil.rows)}")

    state = {
        name: np.array(
            profiles[name].reshape(NUM_SOIL_LAYERS, ncolumn),
            dtype=np.float32,
            copy=True,
        )
        for name in RUC_DRIVER_PROFILE_STATE
    }

    tbq = ruc_saturation_table()
    emissivity_table = np.asarray(
        [row.lemi for row in vegetation.rows], dtype=np.float32
    )
    # ``:779``.  CFACTR_DATA is a per-dataset VEGPARM scalar, not a literal.
    cn_exponent = np.float32(vegetation.scalars["CFACTR_DATA"])
    # ``:780``.  A driver literal, not a table entry.
    sat = np.float32(5.0e-4)
    cp_air = np.float32(1004.5)
    zero = np.float32(0.0)
    one = np.float32(1.0)
    freezing = np.float32(273.15)

    zsmain, zshalf = _ruc_driver_soil_geometry(levels)

    local = {
        name: np.zeros(ncolumn, dtype=np.float32)
        for name in RUC_DRIVER_STALE_LOCALS
    }
    infiltr = np.zeros(ncolumn, dtype=np.float32)
    smelt = np.zeros(ncolumn, dtype=np.float32)
    runoff1 = np.zeros(ncolumn, dtype=np.float32)
    runoff2 = np.zeros(ncolumn, dtype=np.float32)

    # ----------------------------------------------------------------------
    # ``:481-565`` first-step initialisation.
    # ----------------------------------------------------------------------
    if ktau == 1:
        state["keepfr3dflag"][:, :] = zero
        soilt = columns["soilt"]
        tso_top = state["tso"][0]
        repair = (columns["soilt1"] < np.float32(170.0)) | (
            columns["soilt1"] > np.float32(400.0))
        blended = (np.float32(0.5) * (soilt + tso_top)).astype(np.float32)
        columns["soilt1"] = np.where(
            repair,
            np.where(columns["snowc"] > zero, blended, tso_top),
            columns["soilt1"],
        ).astype(np.float32)
        columns["tsnav"] = (blended - freezing).astype(np.float32)
        patmb = (columns["p8w"] * np.float32(1.0e-2)).astype(np.float32)
        columns["qsg"] = (
            ruc_qsn(soilt, tbq, arrays=arrays).astype(np.float32) / patmb
        ).astype(np.float32)
        columns["qcg"] = np.where(
            (columns["qcg"] < zero) | (columns["qcg"] > np.float32(0.1)),
            columns["qc3d"], columns["qcg"]).astype(np.float32)
        columns["qvg"] = np.where(
            (columns["qvg"] <= zero) | (columns["qvg"] > np.float32(0.1)),
            (columns["qsg"] * columns["mavail"]).astype(np.float32),
            columns["qvg"]).astype(np.float32)
        columns["qsfc"] = (
            columns["qvg"] / (one + columns["qvg"]).astype(np.float32)
        ).astype(np.float32)
        for name in (
            "snom",
            "snowfallac",
            "precipfr",
            "dew",
            "sfcrunoff",
            "udrunoff",
            "acrunoff",
        ):
            columns[name][:] = zero
        columns["rhosnf"][:] = np.float32(-1.0e3)
        columns["chklowq"][:] = one

    # ----------------------------------------------------------------------
    # ``:577-1172``, over the whole column axis.  See the docstring: this is
    # WRF's ``DO i`` batched, not a different formulation of it.
    # ----------------------------------------------------------------------
    ilnb_seed = np.int32(ilnb)
    index = np.arange(ncolumn)

    tabs = columns["t3d"]
    qvatm = columns["qv3d"]
    qcatm = columns["qc3d"]
    patm = (columns["p8w"] * np.float32(1.0e-5)).astype(np.float32)
    conflx = (columns["z3d"] * np.float32(0.5)).astype(np.float32)
    rho = columns["rho3d"]

    # ``:612-673`` precipitation partition (EM_CORE==0 arms only).
    grauprat = np.zeros(ncolumn, dtype=np.float32)
    icerat = np.zeros(ncolumn, dtype=np.float32)
    curat = np.zeros(ncolumn, dtype=np.float32)
    rate = (
        (columns["rainbl"] / timestep).astype(np.float32) * np.float32(1.0e-3)
    ).astype(np.float32)
    if frpcpn:
        prcpms = (
            rate * (np.float32(1) - columns["frzfrac"]).astype(np.float32)
        ).astype(np.float32)
        newsnms = (rate * columns["frzfrac"]).astype(np.float32)
        # ``:657``.  Guarded rather than clipped: where ``newsnms`` is zero
        # WRF does not evaluate the quotient AT ALL, and ``prcpms`` may be
        # zero with it, so 0/0 is not a value being discarded -- it is a
        # division the source never performs.  The denominator is forced to
        # one there so no NaN is created; the selected arm is untouched.
        total = (newsnms + prcpms).astype(np.float32)
        falling = newsnms != zero
        snowrat = np.where(
            falling,
            np.minimum(
                one,
                (newsnms / np.where(falling, total, one).astype(np.float32)
                 ).astype(np.float32),
            ),
            zero,
        ).astype(np.float32)
    else:
        frozen = tabs <= freezing
        prcpms = np.where(frozen, zero, rate).astype(np.float32)
        newsnms = np.where(frozen, rate, zero).astype(np.float32)
        snowrat = np.where(frozen, one, zero).astype(np.float32)
    columns["precipfr"] = (
        (newsnms * timestep).astype(np.float32) * np.float32(1.0e3)
    ).astype(np.float32)

    # ``:680-688``.  ``qkms`` divides by the INCOMING ``mavail``; ``:1090``
    # only replaces it after ``SFCTMP`` has run.  The ``myj`` arm is
    # transcribed but NOT oracle-verified: the gate at the top of this
    # function rejects ``myj=True``, because ``ruc_soil_step`` and
    # ``ruc_snow_soil_step`` do.
    if myj:
        qkms = np.array(columns["chs"], dtype=np.float32, copy=True)
        tkms = np.array(columns["chs"], dtype=np.float32, copy=True)
    else:
        qkms = (
            (columns["flqc"] / rho).astype(np.float32) / columns["mavail"]
        ).astype(np.float32)
        tkms = (
            (columns["flhc"] / rho).astype(np.float32)
            / (cp_air * (
                one + (np.float32(0.84) * qvatm).astype(np.float32)
            ).astype(np.float32)).astype(np.float32)
        ).astype(np.float32)

    # ``:690-697``.
    snwe = (columns["snow"] * np.float32(1.0e-3)).astype(np.float32)
    snhei = np.array(columns["snowh"], dtype=np.float32, copy=True)
    canwatr = (columns["canwat"] * np.float32(1.0e-3)).astype(np.float32)
    snowfrac = np.array(columns["snowc"], dtype=np.float32, copy=True)
    rhosnfall = np.array(columns["rhosnf"], dtype=np.float32, copy=True)

    # ``:748-754``.
    packed_snow = (columns["snow"] > zero) & (columns["snowh"] > zero)
    rhosn = np.where(
        packed_snow,
        (columns["snow"]
         / np.where(packed_snow, columns["snowh"], one).astype(np.float32)
         ).astype(np.float32),
        np.float32(300.0),
    ).astype(np.float32)

    # ``:763-767`` SOILVEGIN, once for every column.
    surface = ruc_surface_parameters(
        soil_category, vegetation_category, columns["shdmin"],
        columns["shdmax"], columns["vegfra"], columns["znt"], columns["lai"],
        rdlai2d=rdlai2d, iswater=water_category, mminlu=mminlu,
        parameters=bundle, arrays=arrays,
    )
    iforest = np.asarray(surface.iforest, dtype=np.int32)
    emissl = np.array(surface.emiss, dtype=np.float32, copy=True)
    pc = np.asarray(surface.pc, dtype=np.float32)
    columns["znt"] = np.array(surface.znt, dtype=np.float32, copy=True)
    columns["lai"] = np.array(surface.lai, dtype=np.float32, copy=True)
    qwrtz = np.asarray(surface.qwrtz, dtype=np.float32)
    rhocs = np.asarray(surface.rhocs, dtype=np.float32)
    bclh = np.asarray(surface.bclh, dtype=np.float32)
    dqm = np.array(surface.dqm, dtype=np.float32, copy=True)
    ksat = np.asarray(surface.ksat, dtype=np.float32)
    psis = np.asarray(surface.psis, dtype=np.float32)
    qmin = np.array(surface.qmin, dtype=np.float32, copy=True)
    ref = np.array(surface.ref, dtype=np.float32, copy=True)
    wilt = np.array(surface.wilt, dtype=np.float32, copy=True)

    # ``:782-812`` root depth and the Egglston melt limiter.  ``nroot`` keeps
    # its ``:797`` default of 4 if no level clears the threshold.
    forest = iforest > 2
    meltfactor = np.where(
        forest, np.float32(2.0), np.float32(0.85)).astype(np.float32)
    nroot = np.empty(ncolumn, dtype=np.int32)
    for limit, arm in ((np.float32(0.4), forest), (np.float32(1.1), ~forest)):
        chosen = 4
        for level in range(1, NUM_SOIL_LAYERS):
            if zsmain[level] >= limit:
                chosen = level + 1
                break
        nroot[arm] = chosen

    # ``:824-847`` water.  ``q2sat`` at ``:840`` is a dead store and
    # ``chklowq`` is forced to 1, so no water column can reach the ``:1067``
    # humidity test.  These columns take WRF's ``CYCLE``: nothing below the
    # branch runs for them, which is why every later write is masked to
    # ``land``.
    water = columns["xland"] - np.float32(1.5) >= zero
    land = ~water
    patmb = (columns["p8w"] * np.float32(1.0e-2)).astype(np.float32)
    if np.any(water):
        columns["smavail"][water] = one
        columns["smmax"][water] = one
        columns["snow"][water] = zero
        columns["snowh"][water] = zero
        columns["snowc"][water] = zero
        water_qvg = (
            ruc_qsn(columns["soilt"][water], tbq,
                    arrays=arrays).astype(np.float32)
            / patmb[water]
        ).astype(np.float32)
        columns["qvg"][water] = water_qvg
        columns["qsfc"][water] = (
            water_qvg / (one + water_qvg).astype(np.float32)
        ).astype(np.float32)
        columns["chklowq"][water] = one
        state["soilmois"][:, water] = one
        state["sh2o"][:, water] = one
        state["tso"][:, water] = columns["soilt"][water]

    # ``:851-855``.
    seaice = np.where(
        columns["xice"] >= threshold, one, zero).astype(np.float32)
    iland = np.array(vegetation_category, dtype=np.int32, copy=True)
    ice = land & (seaice > np.float32(0.5))
    if np.any(ice):
        # ``:862-895``.
        iland[ice] = ice_category
        columns["znt"][ice] = np.float32(0.011)
        columns["snoalb"][ice] = np.float32(0.75)
        dqm[ice] = one
        ref[ice] = one
        qmin[ice] = zero
        wilt[ice] = zero
        emissl[ice] = np.float32(0.98)
        ice_qvg = (
            ruc_qsn(columns["soilt"][ice], tbq,
                    arrays=arrays).astype(np.float32)
            / patmb[ice]
        ).astype(np.float32)
        columns["qvg"][ice] = ice_qvg
        columns["qsg"][ice] = ice_qvg
        columns["qsfc"][ice] = (
            ice_qvg / (one + ice_qvg).astype(np.float32)
        ).astype(np.float32)
        state["soilmois"][:, ice] = one
        state["smfr3d"][:, ice] = one
        state["sh2o"][:, ice] = zero
        state["keepfr3dflag"][:, ice] = zero
        state["tso"][:, ice] = np.minimum(
            np.float32(271.4), state["tso"][:, ice])

    # ``:900-914``.  RUC carries soil moisture minus the residual.
    soilm1d = np.minimum(
        np.maximum(zero, (state["soilmois"] - qmin).astype(np.float32)), dqm,
    ).astype(np.float32)
    tso1d = np.array(state["tso"], dtype=np.float32, copy=True)
    soiliqw = np.minimum(
        np.maximum(zero, (state["sh2o"] - qmin).astype(np.float32)), soilm1d,
    ).astype(np.float32)
    soilice = (
        (soilm1d - soiliqw).astype(np.float32) / np.float32(0.9)
    ).astype(np.float32)
    smfrkeep = np.array(state["smfr3d"], dtype=np.float32, copy=True)
    keepfr = np.array(state["keepfr3dflag"], dtype=np.float32, copy=True)

    # ``:915``.  WRF reaches this line only after the water ``CYCLE``, and a
    # water category can carry ``ref == qmin``, so evaluating it everywhere
    # would divide 0 by 0 on exactly the columns whose answer is discarded.
    # Computed on the land columns alone, which is where the source computes
    # it; the rest keep the zero they are never read at.
    lmavail = np.zeros(ncolumn, dtype=np.float32)
    lmavail[land] = np.maximum(
        np.float32(0.00001),
        np.minimum(
            one,
            (soilm1d[0][land]
             / (ref[land] - qmin[land]).astype(np.float32)
             ).astype(np.float32),
        ),
    ).astype(np.float32)

    # ``:928-935``.  ``smtotold`` is formed and never read again: the
    # ``:1109-1141`` water budget it feeds is built from driver locals that
    # no argument reaches.  Evaluated anyway, because "dead" is a claim about
    # the source and not a licence to skip it.
    smtotold = np.zeros(ncolumn, dtype=np.float32)
    for level in range(NUM_SOIL_LAYERS - 1):
        smtotold = (smtotold + (
            (qmin + soilm1d[level]).astype(np.float32)
            * np.float32(zshalf[level + 1] - zshalf[level])
        ).astype(np.float32)).astype(np.float32)
    smtotold = (smtotold + (
        (qmin + soilm1d[NUM_SOIL_LAYERS - 1]).astype(np.float32)
        * np.float32(
            zsmain[NUM_SOIL_LAYERS - 1] - zshalf[NUM_SOIL_LAYERS - 1])
    ).astype(np.float32)).astype(np.float32)
    del smtotold

    # ----------------------------------------------------------------------
    # ``:951-984`` SFCTMP.
    # ----------------------------------------------------------------------
    run = index[land]
    sflx = np.zeros(ncolumn, dtype=np.float32)
    carried_ilnb = ilnb_seed

    def _sfctmp_values(take) -> dict[str, object]:
        values_out: dict[str, object] = {
            "soilm1d": soilm1d[:, take], "ts1d": tso1d[:, take],
            "smfrkeep": smfrkeep[:, take], "keepfr": keepfr[:, take],
            "soilice": soilice[:, take], "soiliqw": soiliqw[:, take],
            "seaice": seaice[take], "gsw": columns["gsw"][take],
            "tabs": tabs[take], "tsnav": columns["tsnav"][take],
            "prcpms": prcpms[take], "newsnms": newsnms[take],
            "vegfra": columns["vegfra"][take], "lai": columns["lai"][take],
            "sat": np.full(take.size, sat, dtype=np.float32),
            "soilt": columns["soilt"][take],
            "snowfallac": columns["snowfallac"][take],
            "alb_snow": columns["snoalb"][take],
            "alb_snow_free": columns["albbck"][take],
            "snowrat": snowrat[take], "grauprat": grauprat[take],
            "icerat": icerat[take], "curat": curat[take],
            "snwe": snwe[take], "snhei": snhei[take],
            "snowfrac": snowfrac[take], "rhosn": rhosn[take],
            "rhosnfall": rhosnfall[take], "cst": canwatr[take],
            "alb": columns["alb"][take], "emiss": emissl[take],
            "znt": columns["znt"][take], "meltfactor": meltfactor[take],
            "patm": patm[take], "qvatm": qvatm[take], "qcatm": qcatm[take],
            "rho": rho[take], "glw": columns["glw"][take],
            "qkms": qkms[take], "tkms": tkms[take], "pc": pc[take],
            "mavail": lmavail[take], "qwrtz": qwrtz[take],
            "rhocs": rhocs[take], "dqm": dqm[take], "qmin": qmin[take],
            "ref": ref[take], "wilt": wilt[take], "psis": psis[take],
            "bclh": bclh[take], "ksat": ksat[take],
            "cn": np.full(take.size, cn_exponent, dtype=np.float32),
            "soilt1": columns["soilt1"][take], "qvg": columns["qvg"][take],
            "qsg": columns["qsg"][take], "qcg": columns["qcg"][take],
            "snom": columns["snom"][take],
            "acsnow": columns["acsnow"][take],
        }
        for name in RUC_DRIVER_STALE_LOCALS:
            values_out[name] = local[name][take]
        return values_out

    def _dispatch(take, seeds):
        return ruc_surface_temperature_step(
            _sfctmp_values(take), delt=float(timestep), conflx=conflx[take],
            ivgtyp=vegetation_category[take], iland=iland[take],
            nroot=nroot[take], ilnb=seeds, isice=ice_category,
            c1sn=c1sn, c2sn=c2sn, myj=myj, isncovr_opt=isncovr_opt,
            mminlu=mminlu, parameters=bundle, leaves=leaves, stages=stages,
            arrays=arrays,
        )

    if run.size:
        if ilnb_chain:
            # WRF's uninitialised ``ilnb`` carries the PREVIOUS COLUMN's snow
            # layer count, which is a real sequential dependence and the one
            # thing here that cannot be batched.  The SAME dispatch runs, one
            # column at a time, so there is one implementation of the physics
            # and two batchings of it.
            pieces = []
            for position in run:
                piece = _dispatch(
                    index[position:position + 1],
                    np.asarray([carried_ilnb], dtype=np.int32))
                carried_ilnb = np.int32(piece.ilnb[0])
                pieces.append(piece)
            surface_step = _ruc_concatenate_sfctmp(pieces)
        else:
            # gpuwm's DEFINED behaviour: no column inherits another column's
            # uninitialised snow-layer count, so every column is seeded from
            # ``ilnb`` and the whole field is one call.  See the docstring.
            surface_step = _dispatch(
                run, np.full(run.size, ilnb_seed, dtype=np.int32))
            # ``int()`` first: on a device namespace ``ilnb[-1]`` is a 0-d
            # device array and ``np.int32`` of one is not defined, while the
            # value itself is a scalar the caller reads on the host anyway.
            carried_ilnb = np.int32(int(surface_step.ilnb[-1]))
        # ``:577``'s loop resets the seed at the TOP of every iteration when
        # the chain is off, and a water column then ``CYCLE``s before
        # assigning anything -- so a trailing water column leaves the
        # returned ``ilnb`` at the seed, not at the last land answer.
        if not ilnb_chain and bool(water[ncolumn - 1]):
            carried_ilnb = ilnb_seed

        soilm1d[:, run] = np.asarray(surface_step.soilm1d, dtype=np.float32)
        tso1d[:, run] = np.asarray(surface_step.ts1d, dtype=np.float32)
        smfrkeep[:, run] = np.asarray(surface_step.smfrkeep, dtype=np.float32)
        keepfr[:, run] = np.asarray(surface_step.keepfr, dtype=np.float32)
        soiliqw[:, run] = np.asarray(surface_step.soiliqw, dtype=np.float32)
        for name in ("soilt", "soilt1", "tsnav", "dew", "qvg", "qsg", "qcg"):
            columns[name][run] = np.asarray(
                getattr(surface_step, name), dtype=np.float32)
        columns["snom"][run] = np.asarray(surface_step.snom, dtype=np.float32)
        columns["snowfallac"][run] = np.asarray(
            surface_step.snowfallac, dtype=np.float32)
        columns["acsnow"][run] = np.asarray(
            surface_step.acsnow, dtype=np.float32)
        columns["alb"][run] = np.asarray(surface_step.alb, dtype=np.float32)
        columns["znt"][run] = np.asarray(surface_step.znt, dtype=np.float32)
        emissl[run] = np.asarray(surface_step.emiss, dtype=np.float32)
        snwe[run] = np.asarray(surface_step.snwe, dtype=np.float32)
        snhei[run] = np.asarray(surface_step.snhei, dtype=np.float32)
        snowfrac[run] = np.asarray(surface_step.snowfrac, dtype=np.float32)
        rhosnfall[run] = np.asarray(surface_step.rhosnfall, dtype=np.float32)
        canwatr[run] = np.asarray(surface_step.cst, dtype=np.float32)
        lmavail[run] = np.asarray(surface_step.mavail, dtype=np.float32)
        smelt[run] = np.asarray(surface_step.smelt, dtype=np.float32)
        runoff1[run] = np.asarray(surface_step.runoff1, dtype=np.float32)
        runoff2[run] = np.asarray(surface_step.runoff2, dtype=np.float32)
        infiltr[run] = np.asarray(surface_step.infiltr, dtype=np.float32)
        for name in RUC_DRIVER_STALE_LOCALS:
            local[name][run] = np.asarray(
                getattr(surface_step, name), dtype=np.float32)
        # ``:971-972``: the driver's ``qfx`` receives SFCTMP's ``eeta`` and
        # its ``lh`` receives SFCTMP's ``qfx``.
        columns["qfx"][run] = np.asarray(surface_step.eeta, dtype=np.float32)
        columns["lh"][run] = np.asarray(surface_step.qfx, dtype=np.float32)
        columns["hfx"][run] = np.asarray(surface_step.hfx, dtype=np.float32)
        sflx[run] = np.asarray(surface_step.s, dtype=np.float32)

    # ``:1024-1035`` soil moisture diagnostics.
    smavail = np.zeros(ncolumn, dtype=np.float32)
    smmax = np.zeros(ncolumn, dtype=np.float32)
    for level in range(NUM_SOIL_LAYERS - 1):
        thickness = np.float32(zshalf[level + 1] - zshalf[level])
        smavail = (smavail + (
            (qmin + soilm1d[level]).astype(np.float32) * thickness
        ).astype(np.float32)).astype(np.float32)
        smmax = (smmax + (
            (qmin + dqm).astype(np.float32) * thickness
        ).astype(np.float32)).astype(np.float32)
    bottom = np.float32(
        zsmain[NUM_SOIL_LAYERS - 1] - zshalf[NUM_SOIL_LAYERS - 1])
    smavail = (smavail + (
        (qmin + soilm1d[NUM_SOIL_LAYERS - 1]).astype(np.float32) * bottom
    ).astype(np.float32)).astype(np.float32)
    smmax = (smmax + (
        (qmin + dqm).astype(np.float32) * bottom
    ).astype(np.float32)).astype(np.float32)

    # ``:1038-1044``.
    surface_runoff = (
        (runoff1 * timestep).astype(np.float32) * np.float32(1000.0)
    ).astype(np.float32)
    under_runoff = (
        (runoff2 * timestep).astype(np.float32) * np.float32(1000.0)
    ).astype(np.float32)
    columns["sfcrunoff"][land] = (
        columns["sfcrunoff"] + surface_runoff).astype(np.float32)[land]
    columns["udrunoff"][land] = (
        columns["udrunoff"] + under_runoff).astype(np.float32)[land]
    columns["acrunoff"][land] = (
        columns["acrunoff"] + surface_runoff).astype(np.float32)[land]
    columns["smavail"][land] = (
        smavail * np.float32(1000.0)).astype(np.float32)[land]
    columns["smmax"][land] = (
        smmax * np.float32(1000.0)).astype(np.float32)[land]

    # ``:1046-1058``.
    moisture = (soilm1d + qmin).astype(np.float32)
    state["soilmois"][:, land] = moisture[:, land]
    state["sh2o"][:, land] = np.minimum(
        (soiliqw + qmin).astype(np.float32), moisture)[:, land]
    state["tso"][:, land] = tso1d[:, land]
    state["tso"][NUM_SOIL_LAYERS - 1, land] = columns["tbot"][land]
    state["smfr3d"][:, land] = smfrkeep[:, land]
    state["keepfr3dflag"][:, land] = keepfr[:, land]

    # ``:1060-1073``.
    columns["z0"][land] = columns["znt"][land]
    columns["sfcexc"][land] = tkms[land]
    q2sat = (ruc_qsn(tabs, tbq, arrays=arrays).astype(np.float32)
             / patmb).astype(np.float32)
    columns["qsfc"][land] = (
        columns["qvg"] / (one + columns["qvg"]).astype(np.float32)
    ).astype(np.float32)[land]
    humid = (
        qvatm >= (q2sat * np.float32(0.95)).astype(np.float32)
    ) & (qvatm < columns["qvg"])
    columns["chklowq"][land] = np.where(humid, zero, one).astype(
        np.float32)[land]

    # ``:1082-1090``.  ``snow`` is still the INCOMING value here: the
    # ``:1085`` rewrite from ``snwe`` comes after the test.
    emissl = np.where(
        columns["snow"] == zero,
        emissivity_table[vegetation_category - 1], emissl,
    ).astype(np.float32)
    columns["emiss"][land] = emissl[land]
    columns["snow"][land] = (snwe * np.float32(1000.0)).astype(np.float32)[land]
    columns["snowh"][land] = snhei[land]
    columns["canwat"][land] = (
        canwatr * np.float32(1000.0)).astype(np.float32)[land]
    # ``:1090``.  ``lmavail`` is ``intent(inout)`` into SFCTMP and SOILMOIST
    # rewrites it, so this is the POST-call value, not the ``:915`` seed.
    columns["mavail"][land] = lmavail[land]

    # ``:1095``.
    columns["sfcevp"][land] = (
        columns["sfcevp"] + (columns["qfx"] * timestep).astype(np.float32)
    ).astype(np.float32)[land]
    columns["grdflx"][land] = (np.float32(-1.0) * sflx).astype(
        np.float32)[land]

    # ``:1104-1110``.
    snowfrac = np.where(
        (snowfrac > zero) & (columns["xice"] >= threshold),
        (snowfrac * columns["xice"]).astype(np.float32), snowfrac,
    ).astype(np.float32)
    columns["snowc"][land] = snowfrac[land]
    columns["rhosnf"][land] = rhosnfall[land]

    # ``:1116``.  The second, duplicated accumulation; see the docstring.
    columns["sfcevp"][land] = (
        columns["sfcevp"] + (columns["qfx"] * timestep).astype(np.float32)
    ).astype(np.float32)[land]

    result = RucLandSurfaceStep(
        **{
            name: columns[name].reshape(horizontal_shape)
            for name in RUC_DRIVER_COLUMN_STATE
        },
        **{
            name: state[name].reshape(shape)
            for name in RUC_DRIVER_PROFILE_STATE
        },
        ilnb=np.asarray(carried_ilnb, dtype=np.int32),
        infiltr=infiltr.reshape(horizontal_shape),
        smelt=smelt.reshape(horizontal_shape),
        runoff1=runoff1.reshape(horizontal_shape),
        runoff2=runoff2.reshape(horizontal_shape),
    )
    # One reduction per field, one host read for all of them; see the same
    # shape at the end of :func:`ruc_surface_temperature_step`.
    checked = [name for name in RucLandSurfaceStep.__dataclass_fields__
               if np.asarray(getattr(result, name)).dtype != np.int32]
    finite = np.stack([
        np.all(np.isfinite(np.asarray(getattr(result, name))))
        for name in checked])
    for name, ok in zip(checked, finite.tolist()):
        if not ok:
            raise ValueError(f"RUC driver produced non-finite {name}")
    return result


__all__ += [
    "RUC_DRIVER_COLUMN_FORCING",
    "RUC_SFCTMP_HOST_LEAVES",
    "RUC_SFCTMP_HOST_STAGES",
    "RUC_DRIVER_COLUMN_STATE",
    "RUC_DRIVER_PROFILE_STATE",
    "RUC_DRIVER_STALE_LOCALS",
    "RucLandSurfaceStep",
    "ruc_land_surface_step",
]
