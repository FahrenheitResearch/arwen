"""CUDA implementation of the pinned WRF v4.6.1 RUC setup slices."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from functools import lru_cache

import cupy as cp
import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.ruc import (
    RUC_SOIL_PROPERTY_COLUMN_INPUTS,
    RUC_SOIL_PROPERTY_PROFILE_INPUTS,
    RUC_SOIL_MOISTURE_COLUMN_INPUTS,
    RUC_SOIL_MOISTURE_PROFILE_INPUTS,
    RUC_SOIL_TEMPERATURE_COLUMN_INPUTS,
    RUC_SOIL_TEMPERATURE_PROFILE_INPUTS,
    RUC_SOIL_STEP_COLUMN_INPUTS,
    RUC_SOIL_STEP_PROFILE_INPUTS,
    RUC_SEA_ICE_COLUMN_INPUTS,
    RUC_SEA_ICE_PROFILE_INPUTS,
    RucParameterBundle,
    RucSeaIceStep,
    RucSnowSeaIceStep,
    RucSnowSoilStep,
    RucSoilStep,
    load_ruc_parameters,
    ruc_soil_geometry,
    ruc_saturation_table,
)
from gpuwm.core.ruc_contract import NUM_SOIL_LAYERS
from gpuwm.core.state import DTYPE


@dataclass(frozen=True)
class RucSurfaceParametersCuda:
    """Device-resident dominant-category outputs from WRF ``soilvegin``."""

    iforest: cp.ndarray
    emiss: cp.ndarray
    pc: cp.ndarray
    znt: cp.ndarray
    lai: cp.ndarray
    qwrtz: cp.ndarray
    rhocs: cp.ndarray
    bclh: cp.ndarray
    dqm: cp.ndarray
    ksat: cp.ndarray
    psis: cp.ndarray
    qmin: cp.ndarray
    ref: cp.ndarray
    wilt: cp.ndarray


@dataclass(frozen=True)
class RucSoilPropertiesCuda:
    """Device-resident outputs from WRF ``soilprop``."""

    thdif: cp.ndarray
    diffu: cp.ndarray
    hydro: cp.ndarray
    cap: cp.ndarray


@dataclass(frozen=True)
class RucTranspirationCuda:
    """Device-resident root-zone weights from WRF ``transf``."""

    tranf: cp.ndarray
    transum: cp.ndarray


@dataclass(frozen=True)
class RucSoilMoistureCuda:
    """Device-resident state and fluxes from WRF ``soilmoist``."""

    soilmois: cp.ndarray
    soiliqw: cp.ndarray
    mavail: cp.ndarray
    runoff: cp.ndarray
    runoff2: cp.ndarray
    infiltrp: cp.ndarray
    infmax: cp.ndarray


@dataclass(frozen=True)
class RucSoilTemperatureCuda:
    """Device-resident heat state and diagnostics from WRF ``soiltemp``."""

    tso: cp.ndarray
    soilt: cp.ndarray
    qvg: cp.ndarray
    qsg: cp.ndarray
    qcg: cp.ndarray
    storage: cp.ndarray


@dataclass(frozen=True)
class RucSoilStepCuda:
    """Device-resident snow-free land state and fluxes from WRF ``soil``."""

    soilmois: cp.ndarray
    tso: cp.ndarray
    smfrkeep: cp.ndarray
    keepfr: cp.ndarray
    soilice: cp.ndarray
    soiliqw: cp.ndarray
    cst: cp.ndarray
    dew: cp.ndarray
    soilt: cp.ndarray
    qvg: cp.ndarray
    qsg: cp.ndarray
    qcg: cp.ndarray
    edir1: cp.ndarray
    ec1: cp.ndarray
    ett1: cp.ndarray
    eeta: cp.ndarray
    qfx: cp.ndarray
    hfx: cp.ndarray
    s: cp.ndarray
    evapl: cp.ndarray
    prcpl: cp.ndarray
    fltot: cp.ndarray
    runoff1: cp.ndarray
    runoff2: cp.ndarray
    mavail: cp.ndarray
    infiltrp: cp.ndarray
    smf: cp.ndarray


@dataclass(frozen=True)
class RucSeaIceStepCuda:
    """Device-resident snow-free sea-ice state and fluxes from WRF ``sice``.

    ``soilmois``/``soiliqw``/``soilice``/``smfrkeep``/``keepfr`` are not
    written by ``sice`` itself; ``sfctmp`` forces them to 1/0/1/1/0
    immediately after both call sites and this result carries that forcing.
    """

    tso: cp.ndarray
    soilmois: cp.ndarray
    soiliqw: cp.ndarray
    soilice: cp.ndarray
    smfrkeep: cp.ndarray
    keepfr: cp.ndarray
    dew: cp.ndarray
    soilt: cp.ndarray
    qvg: cp.ndarray
    qsg: cp.ndarray
    qcg: cp.ndarray
    eeta: cp.ndarray
    qfx: cp.ndarray
    hfx: cp.ndarray
    s: cp.ndarray
    evapl: cp.ndarray
    prcpl: cp.ndarray
    fltot: cp.ndarray


@dataclass(frozen=True)
class _RucDeviceTables:
    ifortbl: cp.ndarray
    z0tbl: cp.ndarray
    lemitbl: cp.ndarray
    pctbl: cp.ndarray
    laitbl: cp.ndarray
    rstbl: cp.ndarray
    rgltbl: cp.ndarray
    rsmax_data: float
    bb: cp.ndarray
    drysmc: cp.ndarray
    hc: cp.ndarray
    maxsmc: cp.ndarray
    refsmc: cp.ndarray
    satpsi: cp.ndarray
    satdk: cp.ndarray
    wltsmc: cp.ndarray
    qtz: cp.ndarray


def _upload_tables(
    bundle: RucParameterBundle,
    mminlu: str,
) -> tuple[_RucDeviceTables, int, int, int]:
    vegetation = bundle.vegetation_for(mminlu)
    rows = vegetation.rows
    soil = np.asarray([row.values for row in bundle.soil.rows], dtype=np.float32)
    tables = _RucDeviceTables(
        ifortbl=cp.asarray([row.ifor for row in rows], dtype=cp.int32),
        z0tbl=cp.asarray([row.z0 for row in rows], dtype=DTYPE),
        lemitbl=cp.asarray([row.lemi for row in rows], dtype=DTYPE),
        pctbl=cp.asarray([row.pc for row in rows], dtype=DTYPE),
        laitbl=cp.asarray([row.lai for row in rows], dtype=DTYPE),
        rstbl=cp.asarray([row.rs for row in rows], dtype=DTYPE),
        rgltbl=cp.asarray([row.rgl for row in rows], dtype=DTYPE),
        rsmax_data=float(vegetation.scalars["RSMAX_DATA"]),
        bb=cp.asarray(soil[:, 0], dtype=DTYPE),
        drysmc=cp.asarray(soil[:, 1], dtype=DTYPE),
        hc=cp.asarray(soil[:, 2], dtype=DTYPE),
        maxsmc=cp.asarray(soil[:, 3], dtype=DTYPE),
        refsmc=cp.asarray(soil[:, 4], dtype=DTYPE),
        satpsi=cp.asarray(soil[:, 5], dtype=DTYPE),
        satdk=cp.asarray(soil[:, 6], dtype=DTYPE),
        wltsmc=cp.asarray(soil[:, 8], dtype=DTYPE),
        qtz=cp.asarray(soil[:, 9], dtype=DTYPE),
    )
    default_water = 16 if vegetation.name == "USGS-RUC" else 17
    return tables, len(rows), len(bundle.soil.rows), default_water


@lru_cache(maxsize=None)
def _default_device_tables(
    device_id: int,
    mminlu: str,
) -> tuple[_RucDeviceTables, int, int, int]:
    with cp.cuda.Device(device_id):
        return _upload_tables(load_ruc_parameters(), mminlu)


@lru_cache(maxsize=None)
def _device_tbq(device_id: int) -> cp.ndarray:
    with cp.cuda.Device(device_id):
        return cp.asarray(ruc_saturation_table(), dtype=DTYPE)


def _integer_field(value, shape: tuple[int, ...], name: str) -> cp.ndarray:
    raw = cp.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{name} shape {raw.shape}; expected {shape}")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer WRF categories")
    return cp.ascontiguousarray(raw, dtype=cp.int32)


def _float_field(value, shape: tuple[int, ...], name: str) -> cp.ndarray:
    raw = cp.asarray(value, dtype=DTYPE)
    if raw.shape != shape:
        try:
            raw = cp.broadcast_to(raw, shape)
        except ValueError as exc:
            raise ValueError(
                f"{name} shape {raw.shape} is not broadcastable to {shape}"
            ) from exc
    if not bool(cp.all(cp.isfinite(raw))):
        raise ValueError(f"{name} must be finite")
    return cp.ascontiguousarray(raw)


def _float_profile(value, shape: tuple[int, ...], name: str) -> cp.ndarray:
    raw = cp.asarray(value, dtype=DTYPE)
    if raw.shape != shape:
        raise ValueError(f"{name} shape {raw.shape}; expected {shape}")
    if not bool(cp.all(cp.isfinite(raw))):
        raise ValueError(f"{name} must be finite")
    return cp.ascontiguousarray(raw)


def _root_count_field(value, shape: tuple[int, ...]) -> cp.ndarray:
    raw = cp.asarray(value)
    if raw.dtype.kind not in "iu":
        raise TypeError("nroot must contain integer root-zone level counts")
    if raw.shape != shape:
        try:
            raw = cp.broadcast_to(raw, shape)
        except ValueError as exc:
            raise ValueError(
                f"nroot shape {raw.shape} is not broadcastable to {shape}"
            ) from exc
    roots = cp.ascontiguousarray(raw, dtype=cp.int32)
    invalid = (roots < 1) | (roots >= NUM_SOIL_LAYERS)
    if bool(cp.any(invalid)):
        bad = int(roots[invalid][0])
        raise ValueError(f"RUC nroot {bad} is outside 1..8")
    return roots


def _soil_phase_partition_cuda(
    soilmois: cp.ndarray,
    tso: cp.ndarray,
    smfrkeep: cp.ndarray,
    keepfr: cp.ndarray,
    columns: dict[str, cp.ndarray],
    *,
    update_smfrkeep: bool,
) -> dict[str, cp.ndarray]:
    """Launch the freezing partition shared by the assembled RUC step."""

    shape = soilmois.shape
    horizontal_shape = shape[1:]
    outputs = {
        name: cp.empty(shape, dtype=DTYPE)
        for name in (
            "soiliqw", "soilice", "tav", "soilmoism", "soiliqwm",
            "soilicem", "lwsat", "fwsat",
        )
    }
    ncolumn = int(np.prod(horizontal_shape))
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_soil_phase_partition")
    kernel(
        (blocks,),
        (threads,),
        (
            soilmois, tso, smfrkeep, keepfr,
            *(columns[name] for name in ("dqm", "qmin", "psis", "bclh")),
            np.int32(update_smfrkeep),
            *(outputs[name] for name in (
                "soiliqw", "soilice", "tav", "soilmoism", "soiliqwm",
                "soilicem", "lwsat", "fwsat",
            )),
            np.int32(ncolumn),
        ),
    )
    return outputs


def _device_constant_flux_depth(conflx, ncolumn: int, label: str):
    """``conflx`` as a contiguous float32 device column field.

    The four kernels that read it -- ``ruc_soil_temperature_step``,
    ``ruc_sea_ice_step``, ``ruc_snow_sea_ice_step`` and
    ``ruc_snow_temperature_step`` -- take a pointer rather than a ``real``,
    because ``0.5*dz8w(i,1,j)`` is a per-column depth.  A scalar caller still
    means "the same depth in every column" and is broadcast here, so no call
    site has to know which it holds.
    """

    depth = cp.asarray(conflx, dtype=cp.float32)
    if depth.ndim > 1:
        raise ValueError(f"RUC CUDA {label} conflx must be scalar or 1-D")
    depth = cp.ascontiguousarray(
        cp.broadcast_to(cp.atleast_1d(depth), (ncolumn,)))
    if not bool(cp.all(cp.isfinite(depth))) or bool(
            cp.any(depth < cp.float32(0.0))):
        raise ValueError(
            f"RUC CUDA {label} conflx must be finite and nonnegative")
    return depth


def ruc_surface_parameters_cuda(
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
) -> RucSurfaceParametersCuda:
    """Evaluate WRF ``soilvegin`` directly on independent GPU columns."""

    if type(mosaic_lu) is not int or mosaic_lu != 0:
        raise ValueError("RUC CUDA surface setup currently requires mosaic_lu=0")
    if type(mosaic_soil) is not int or mosaic_soil != 0:
        raise ValueError("RUC CUDA surface setup currently requires mosaic_soil=0")
    if type(rdlai2d) is not bool:
        raise TypeError("rdlai2d must be bool")

    soil_raw = cp.asarray(isltyp)
    if soil_raw.ndim < 1:
        raise ValueError("RUC CUDA surface fields must have at least one dimension")
    shape = soil_raw.shape
    soil_type = _integer_field(soil_raw, shape, "isltyp")
    vegetation_type = _integer_field(ivgtyp, shape, "ivgtyp")
    inputs = tuple(
        _float_field(value, shape, name)
        for value, name in (
            (shdmin, "shdmin"),
            (shdmax, "shdmax"),
            (vegfrac, "vegfrac"),
            (znt, "znt"),
            (lai, "lai"),
        )
    )

    if parameters is None:
        device_id = int(cp.cuda.runtime.getDevice())
        tables, nvegetation, nsoil, default_water = _default_device_tables(
            device_id, mminlu
        )
    else:
        tables, nvegetation, nsoil, default_water = _upload_tables(
            parameters, mminlu
        )
    soil_min = int(cp.min(soil_type))
    soil_max = int(cp.max(soil_type))
    vegetation_min = int(cp.min(vegetation_type))
    vegetation_max = int(cp.max(vegetation_type))
    if soil_min < 1 or soil_max > nsoil:
        bad = soil_min if soil_min < 1 else soil_max
        raise ValueError(f"RUC isltyp {bad} is outside 1..{nsoil}")
    if vegetation_min < 1 or vegetation_max > nvegetation:
        bad = vegetation_min if vegetation_min < 1 else vegetation_max
        raise ValueError(
            f"RUC ivgtyp {bad} is outside 1..{nvegetation} for {mminlu}"
        )
    if iswater is None:
        water_category = default_water
    elif type(iswater) is int and 1 <= iswater <= nvegetation:
        water_category = iswater
    else:
        raise ValueError(f"RUC iswater {iswater!r} is outside 1..{nvegetation}")

    float_names = (
        "emiss", "pc", "znt", "lai", "qwrtz", "rhocs", "bclh",
        "dqm", "ksat", "psis", "qmin", "ref", "wilt",
    )
    float_outputs = {
        name: cp.empty(shape, dtype=DTYPE) for name in float_names
    }
    forest = cp.empty(shape, dtype=cp.int32)
    n = int(np.prod(shape))
    threads = 128
    blocks = (n + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_surface_parameters")
    kernel(
        (blocks,),
        (threads,),
        (
            soil_type,
            vegetation_type,
            *inputs,
            tables.ifortbl,
            tables.z0tbl,
            tables.lemitbl,
            tables.pctbl,
            tables.laitbl,
            tables.bb,
            tables.drysmc,
            tables.hc,
            tables.maxsmc,
            tables.refsmc,
            tables.satpsi,
            tables.satdk,
            tables.wltsmc,
            tables.qtz,
            forest,
            *(float_outputs[name] for name in float_names),
            np.int32(water_category),
            np.int32(rdlai2d),
            np.int32(n),
        ),
    )
    return RucSurfaceParametersCuda(
        iforest=forest,
        **float_outputs,
    )


def ruc_soil_properties_cuda(
    values: dict[str, object],
    *,
    riw: float = 0.9,
) -> RucSoilPropertiesCuda:
    """Evaluate deterministic WRF ``soilprop`` on nine-level GPU columns."""

    ice_water_ratio = np.float32(riw)
    if not np.isfinite(ice_water_ratio) or ice_water_ratio <= np.float32(0.0):
        raise ValueError("RUC CUDA soilprop riw must be finite and positive")
    required = RUC_SOIL_PROPERTY_PROFILE_INPUTS + RUC_SOIL_PROPERTY_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC soil-property inputs: {', '.join(missing)}")
    first = cp.asarray(values[RUC_SOIL_PROPERTY_PROFILE_INPUTS[0]])
    if first.ndim < 2 or first.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA soil-property profiles must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {first.shape}"
        )
    shape = first.shape
    profiles = {
        name: _float_profile(values[name], shape, name)
        for name in RUC_SOIL_PROPERTY_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SOIL_PROPERTY_COLUMN_INPUTS
    }
    if bool(cp.any(columns["bclh"] <= cp.float32(0.0))):
        raise ValueError("RUC bclh must be positive")
    if bool(cp.any(columns["psis"] >= cp.float32(0.0))):
        raise ValueError("RUC psis must be negative")
    if bool(cp.any(columns["ksat"] < cp.float32(0.0))):
        raise ValueError("RUC ksat must be nonnegative")

    outputs = {
        name: cp.empty(shape, dtype=DTYPE)
        for name in ("thdif", "diffu", "hydro", "cap")
    }
    ncolumn = int(np.prod(horizontal_shape))
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_soil_properties")
    kernel(
        (blocks,),
        (threads,),
        (
            *(profiles[name] for name in RUC_SOIL_PROPERTY_PROFILE_INPUTS),
            *(columns[name] for name in RUC_SOIL_PROPERTY_COLUMN_INPUTS),
            ice_water_ratio,
            *(outputs[name] for name in ("thdif", "diffu", "hydro", "cap")),
            np.int32(ncolumn),
        ),
    )
    return RucSoilPropertiesCuda(**outputs)


def ruc_transpiration_cuda(
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
) -> RucTranspirationCuda:
    """Evaluate WRF ``transf`` on per-column 1..8-level GPU root zones."""
    liquid = cp.asarray(soiliqw, dtype=DTYPE)
    if liquid.ndim < 2 or liquid.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA soiliqw must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {liquid.shape}"
        )
    if not bool(cp.all(cp.isfinite(liquid))):
        raise ValueError("soiliqw must be finite")
    liquid = cp.ascontiguousarray(liquid)
    horizontal_shape = liquid.shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    columns = {
        name: _float_field(value, horizontal_shape, name)
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
    land_type = _integer_field(iland, horizontal_shape, "iland")
    if parameters is None:
        device_id = int(cp.cuda.runtime.getDevice())
        tables, nvegetation, _, _ = _default_device_tables(device_id, mminlu)
    else:
        tables, nvegetation, _, _ = _upload_tables(parameters, mminlu)
    land_min = int(cp.min(land_type))
    land_max = int(cp.max(land_type))
    if land_min < 1 or land_max > nvegetation:
        bad = land_min if land_min < 1 else land_max
        raise ValueError(f"RUC iland {bad} is outside 1..{nvegetation} for {mminlu}")
    if bool(cp.any(columns["ref"] <= columns["wilt"])):
        raise ValueError("RUC ref must exceed wilt")

    zs, _ = ruc_soil_geometry()
    zshalf_host = np.zeros(NUM_SOIL_LAYERS, dtype=np.float32)
    for level in range(1, NUM_SOIL_LAYERS):
        zshalf_host[level] = np.float32(
            np.float32(zs[level - 1] + zs[level]) * np.float32(0.5)
        )
    zshalf = cp.asarray(zshalf_host)
    weights = cp.empty_like(liquid)
    totals = cp.empty(horizontal_shape, dtype=DTYPE)
    ncolumn = int(np.prod(horizontal_shape))
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_transpiration")
    kernel(
        (blocks,),
        (threads,),
        (
            liquid,
            *(columns[name] for name in (
                "tabs", "lai", "gswin", "dqm", "qmin", "ref", "wilt", "pc"
            )),
            land_type,
            tables.rstbl,
            tables.rgltbl,
            zshalf,
            roots,
            np.float32(tables.rsmax_data),
            weights,
            totals,
            np.int32(ncolumn),
        ),
    )
    return RucTranspirationCuda(tranf=weights, transum=totals)


def ruc_soil_moisture_step_cuda(
    values: dict[str, object],
    *,
    delt: float,
) -> RucSoilMoistureCuda:
    """Run the complete nine-level WRF ``soilmoist`` solve on GPU."""

    timestep = np.float32(delt)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC CUDA soilmoist delt must be finite and positive")
    required = RUC_SOIL_MOISTURE_PROFILE_INPUTS + RUC_SOIL_MOISTURE_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC soilmoist inputs: {', '.join(missing)}")
    first = cp.asarray(values[RUC_SOIL_MOISTURE_PROFILE_INPUTS[0]])
    if first.ndim < 2 or first.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA soilmoist profiles must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {first.shape}"
        )
    shape = first.shape
    profiles = {
        name: _float_profile(values[name], shape, name)
        for name in RUC_SOIL_MOISTURE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SOIL_MOISTURE_COLUMN_INPUTS
    }
    if bool(cp.any(columns["dqm"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA soilmoist dqm must be positive")
    if bool(cp.any(columns["ref"] <= columns["qmin"])):
        raise ValueError("RUC CUDA soilmoist ref must exceed qmin")
    if bool(cp.any(columns["ksat"] < cp.float32(0.0))):
        raise ValueError("RUC CUDA soilmoist ksat must be nonnegative")

    profile_outputs = {
        name: cp.empty(shape, dtype=DTYPE) for name in ("soilmois", "soiliqw")
    }
    horizontal_outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in ("mavail", "runoff", "runoff2", "infiltrp", "infmax")
    }
    ncolumn = int(np.prod(horizontal_shape))
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_soil_moisture_step")
    kernel(
        (blocks,),
        (threads,),
        (
            *(profiles[name] for name in RUC_SOIL_MOISTURE_PROFILE_INPUTS),
            *(columns[name] for name in RUC_SOIL_MOISTURE_COLUMN_INPUTS),
            timestep,
            profile_outputs["soilmois"],
            profile_outputs["soiliqw"],
            *(horizontal_outputs[name] for name in (
                "mavail", "runoff", "runoff2", "infiltrp", "infmax"
            )),
            np.int32(ncolumn),
        ),
    )
    return RucSoilMoistureCuda(**profile_outputs, **horizontal_outputs)


def ruc_soil_temperature_step_cuda(
    values: dict[str, object],
    *,
    delt: float,
    conflx: float = 0.5,
    nroot: object = 4,
    cvw: float = 4183.0,
) -> RucSoilTemperatureCuda:
    """Run WRF's snow-free nine-level ``soiltemp`` solve on GPU."""

    timestep = np.float32(delt)
    raw_flux_depth = conflx
    water_heat_capacity = np.float32(cvw)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC CUDA soiltemp delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC CUDA soiltemp cvw must be finite and positive")
    required = (
        RUC_SOIL_TEMPERATURE_PROFILE_INPUTS
        + RUC_SOIL_TEMPERATURE_COLUMN_INPUTS
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC soiltemp inputs: {', '.join(missing)}")
    first = cp.asarray(values[RUC_SOIL_TEMPERATURE_PROFILE_INPUTS[0]])
    if first.ndim < 2 or first.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA soiltemp profiles must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {first.shape}"
        )
    shape = first.shape
    profiles = {
        name: _float_profile(values[name], shape, name)
        for name in RUC_SOIL_TEMPERATURE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SOIL_TEMPERATURE_COLUMN_INPUTS
    }
    if bool(cp.any(profiles["thdif"][0] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA soiltemp top-level thdif must be positive")
    if bool(cp.any(profiles["cap"][0] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA soiltemp top-level cap must be positive")
    if bool(cp.any(columns["patm"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA soiltemp patm must be positive")
    if bool(cp.any(columns["rho"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA soiltemp rho must be positive")
    if bool(cp.any(
        (columns["mavail"] < cp.float32(0.0))
        | (columns["mavail"] > cp.float32(1.0))
    )):
        raise ValueError("RUC CUDA soiltemp mavail must be within 0..1")

    tso = cp.empty(shape, dtype=DTYPE)
    outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in ("soilt", "qvg", "qsg", "qcg", "storage")
    }
    device_id = int(cp.cuda.runtime.getDevice())
    tbq = _device_tbq(device_id)
    ncolumn = int(np.prod(horizontal_shape))
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "soiltemp")
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_soil_temperature_step")
    kernel(
        (blocks,),
        (threads,),
        (
            *(profiles[name] for name in (
                "thdif", "cap", "tso"
            )),
            *(columns[name] for name in (
                "prcpms", "rainf", "patm", "tabs", "qvatm", "emiss",
                "rnet", "qkms", "tkms", "rho", "vegfrac", "drycan",
                "wetcan", "transum", "mavail", "soilres", "soilt", "qvg",
            )),
            roots,
            tbq,
            timestep,
            constant_flux_depth,
            water_heat_capacity,
            tso,
            *(outputs[name] for name in (
                "soilt", "qvg", "qsg", "qcg", "storage"
            )),
            np.int32(ncolumn),
        ),
    )
    return RucSoilTemperatureCuda(tso=tso, **outputs)


def ruc_soil_step_cuda(
    values: dict[str, object],
    iland,
    *,
    nroot: object,
    delt: float,
    conflx: float,
    myj: bool = False,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
    parameters: RucParameterBundle | None = None,
) -> RucSoilStepCuda:
    """Run the complete snow-free WRF RUC land column on the GPU."""

    if myj is not False:
        raise ValueError("RUC CUDA first soil lane supports myj=False only")
    timestep = np.float32(delt)
    constant_flux_depth = conflx
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC CUDA soil delt must be finite and positive")
    required = RUC_SOIL_STEP_PROFILE_INPUTS + RUC_SOIL_STEP_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC CUDA soil inputs: {', '.join(missing)}")

    first = cp.asarray(values[RUC_SOIL_STEP_PROFILE_INPUTS[0]])
    if first.ndim < 2 or first.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA soil profiles must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {first.shape}"
        )
    shape = first.shape
    profiles = {
        name: _float_profile(values[name], shape, name)
        for name in RUC_SOIL_STEP_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    land_type = _integer_field(iland, horizontal_shape, "iland")
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SOIL_STEP_COLUMN_INPUTS
    }
    if bool(cp.any(columns["dqm"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA soil dqm must be positive")
    if bool(cp.any(columns["psis"] >= cp.float32(0.0))):
        raise ValueError("RUC CUDA soil psis must be negative")
    if bool(cp.any(columns["bclh"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA soil bclh must be positive")
    if bool(cp.any(columns["sat"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA soil canopy saturation must be positive")
    if bool(cp.any(columns["rho"] <= cp.float32(0.0))) or bool(
        cp.any(columns["patm"] <= cp.float32(0.0))
    ):
        raise ValueError("RUC CUDA soil rho and patm must be positive")
    if bool(cp.any(
        (columns["mavail"] < cp.float32(0.0))
        | (columns["mavail"] > cp.float32(1.0))
    )):
        raise ValueError("RUC CUDA soil mavail must be within 0..1")

    soilmois = profiles["soilmois"].copy()
    tso = profiles["tso"].copy()
    smfrkeep = profiles["smfrkeep"].copy()
    keepfr = profiles["keepfr"].copy()
    told = tso.copy()
    smold = soilmois.copy()
    phase = _soil_phase_partition_cuda(
        soilmois, tso, smfrkeep, keepfr, columns, update_smfrkeep=True
    )
    source_riw = np.float32(np.float32(900.0) * np.float32(1.0e-3))
    properties = ruc_soil_properties_cuda(
        {
            **{name: phase[name] for name in (
                "fwsat", "lwsat", "tav", "soilmoism", "soiliqwm",
                "soilicem",
            )},
            "keepfr": keepfr,
            "soilmois": soilmois,
            "soiliqw": phase["soiliqw"],
            "soilice": phase["soilice"],
            **{name: columns[name] for name in (
                "qwrtz", "rhocs", "dqm", "qmin", "psis", "bclh", "ksat",
            )},
        },
        riw=float(source_riw),
    )

    ncolumn = int(np.prod(horizontal_shape))
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    canopy = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in ("dew", "wetcan", "drycan", "soilres")
    }
    canopy_kernel = get_kernel("ruc", "ruc_soil_canopy_setup")
    canopy_kernel(
        (blocks,),
        (threads,),
        (
            soilmois,
            *(columns[name] for name in (
                "qvatm", "qsg", "qvg", "qkms", "cst", "sat", "cn",
                "qmin", "ref",
            )),
            *(canopy[name] for name in ("dew", "wetcan", "drycan", "soilres")),
            np.int32(ncolumn),
        ),
    )
    transpiration = ruc_transpiration_cuda(
        phase["soiliqw"],
        columns["tabs"], columns["lai"], columns["gswin"],
        columns["dqm"], columns["qmin"], columns["ref"], columns["wilt"],
        columns["pc"], land_type, nroot=roots, mminlu=mminlu,
        parameters=parameters,
    )
    temperature = ruc_soil_temperature_step_cuda(
        {
            "thdif": properties.thdif,
            "cap": properties.cap,
            "tso": tso,
            **{name: columns[name] for name in (
                "prcpms", "rainf", "patm", "tabs", "qvatm", "qcatm",
                "emiss", "rnet", "qkms", "tkms", "pc", "rho", "vegfrac",
                "lai", "dqm", "qmin", "bclh",
            )},
            "drycan": canopy["drycan"],
            "wetcan": canopy["wetcan"],
            "transum": transpiration.transum,
            "dew": canopy["dew"],
            "mavail": columns["mavail"],
            "soilres": canopy["soilres"],
            "alfa": cp.ones(horizontal_shape, dtype=DTYPE),
            "soilt": columns["soilt"],
            "qvg": columns["qvg"],
            "qsg": columns["qsg"],
            "qcg": columns["qcg"],
        },
        delt=float(timestep),
        conflx=constant_flux_depth,
        nroot=roots,
        cvw=4.183e6,
    )
    tso = temperature.tso
    phase = _soil_phase_partition_cuda(
        soilmois, tso, smfrkeep, keepfr, columns, update_smfrkeep=False
    )

    prepared_profiles = {"transp": cp.empty(shape, dtype=DTYPE)}
    prepared = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in ("ett1", "dew", "prcp", "ras")
    }
    prepare_kernel = get_kernel("ruc", "ruc_soil_prepare_moisture")
    prepare_kernel(
        (blocks,),
        (threads,),
        (
            columns["qvatm"], temperature.qsg, columns["qkms"],
            columns["rho"], columns["vegfrac"], canopy["drycan"],
            transpiration.tranf, roots, columns["infwater"],
            prepared_profiles["transp"],
            *(prepared[name] for name in ("ett1", "dew", "prcp", "ras")),
            np.int32(ncolumn),
        ),
    )

    zeros = cp.zeros(horizontal_shape, dtype=DTYPE)
    moisture = ruc_soil_moisture_step_cuda(
        {
            "diffu": properties.diffu,
            "hydro": properties.hydro,
            "transp": prepared_profiles["transp"],
            "soilice": phase["soilice"],
            "soilmois": soilmois,
            "soiliqw": phase["soiliqw"],
            "qsg": temperature.qsg,
            "qvg": temperature.qvg,
            "qcg": temperature.qcg,
            "qcatm": columns["qcatm"],
            "qvatm": columns["qvatm"],
            "prcp": prepared["prcp"],
            "qkms": columns["qkms"],
            "drip": columns["drip"],
            "dew": prepared["dew"],
            "smelt": zeros,
            "vegfrac": columns["vegfrac"],
            "snowfrac": zeros,
            "soilres": canopy["soilres"],
            "dqm": columns["dqm"],
            "qmin": columns["qmin"],
            "ref": columns["ref"],
            "ksat": columns["ksat"],
            "ras": prepared["ras"],
        },
        delt=float(timestep),
    )
    soilmois = moisture.soilmois

    final_names = (
        "cst", "edir1", "ec1", "ett1", "eeta", "qfx", "hfx", "s",
        "evapl", "prcpl", "fltot", "smf",
    )
    final = {
        name: cp.empty(horizontal_shape, dtype=DTYPE) for name in final_names
    }
    finalize_kernel = get_kernel("ruc", "ruc_soil_finalize")
    finalize_kernel(
        (blocks,),
        (threads,),
        (
            phase["soilice"], tso, told, soilmois, smold, keepfr,
            properties.thdif, properties.cap,
            columns["cst"], prepared["dew"],
            temperature.soilt, temperature.qvg, temperature.qsg,
            temperature.qcg, prepared["ett1"], canopy["wetcan"],
            canopy["soilres"], prepared["ras"],
            *(columns[name] for name in (
                "tkms", "rho", "tabs", "patm", "qkms", "qvatm",
                "vegfrac", "rnet", "prcpms",
            )),
            temperature.storage, timestep,
            *(final[name] for name in final_names),
            np.int32(ncolumn),
        ),
    )

    result = RucSoilStepCuda(
        soilmois=soilmois,
        tso=tso,
        smfrkeep=smfrkeep,
        keepfr=keepfr,
        soilice=phase["soilice"],
        soiliqw=moisture.soiliqw,
        cst=final["cst"],
        dew=prepared["dew"],
        soilt=temperature.soilt,
        qvg=temperature.qvg,
        qsg=temperature.qsg,
        qcg=temperature.qcg,
        edir1=final["edir1"],
        ec1=final["ec1"],
        ett1=final["ett1"],
        eeta=final["eeta"],
        qfx=final["qfx"],
        hfx=final["hfx"],
        s=final["s"],
        evapl=final["evapl"],
        prcpl=final["prcpl"],
        fltot=final["fltot"],
        runoff1=moisture.runoff,
        runoff2=moisture.runoff2,
        mavail=moisture.mavail,
        infiltrp=moisture.infiltrp,
        smf=final["smf"],
    )
    for name in RucSoilStepCuda.__dataclass_fields__:
        if not bool(cp.all(cp.isfinite(getattr(result, name)))):
            raise ValueError(f"RUC CUDA soil produced non-finite {name}")
    return result

def _device_saturation_table(table: object | None) -> cp.ndarray:
    """Return the device ``tbq`` table, reusing the cached upload by default."""

    if table is None:
        return _device_tbq(int(cp.cuda.runtime.getDevice()))
    resolved = cp.ascontiguousarray(cp.asarray(table, dtype=DTYPE))
    if resolved.shape != (5001,):
        raise ValueError(
            f"RUC qsn table must have shape (5001,), got {resolved.shape}"
        )
    return resolved


def ruc_qsn_cuda(tn, table: object | None = None) -> cp.ndarray:
    """Evaluate WRF ``qsn`` on independent GPU points.

    ``qsn`` returns ``0.62198 * es(tn)`` in the table's pressure units; the
    callers divide by the surface pressure to obtain a mixing ratio.  The
    5001-entry table spans 173.15 K to 423.15 K in 0.05 K steps, and WRF
    clamps both ends onto the terminal nodes rather than extrapolating.
    """

    values = cp.asarray(tn, dtype=DTYPE)
    if not bool(cp.all(cp.isfinite(values))):
        raise ValueError("RUC qsn temperatures must be finite")
    saturation = _device_saturation_table(table)
    # ascontiguousarray promotes a scalar to shape (1,); the original shape
    # is restored on the way out so callers keep the layout they passed in.
    flat = cp.ascontiguousarray(values).reshape(-1)

    n = int(flat.size)
    result = cp.empty(n, dtype=DTYPE)
    threads = 128
    blocks = (n + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_qsn")
    kernel(
        (blocks,),
        (threads,),
        (flat, saturation, result, np.int32(n)),
    )
    return result.reshape(values.shape)


def ruc_sea_ice_step_cuda(
    values: dict[str, object],
    *,
    delt: float,
    conflx: float = 40.0,
    myj: bool = False,
    cw: float = 4.183e6,
) -> RucSeaIceStepCuda:
    """Run the complete snow-free WRF RUC sea-ice column on the GPU.

    Heat diffusion through the nine ice levels plus the surface energy
    balance closed by ``vilka``.  Every returned ice temperature is clipped
    at the 271.4 K sea-ice melting cap; melt itself is not modelled, the
    excess is absorbed by ``sice``'s local ``icemelt``.

    The arguments WRF passes but ``sice`` never reads -- ``qcatm``, ``gsw``,
    ``tice``, ``rhosice``, ``zshalf``, ``dtdzs2``, ``nroot``, ``xlv`` and
    ``glw`` -- are omitted, matching the CPU transcription.
    """

    timestep = np.float32(delt)
    raw_flux_depth = conflx
    water_heat_capacity = np.float32(cw)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC CUDA sice delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC CUDA sice cw must be finite and positive")
    if type(myj) is not bool:
        raise TypeError("RUC CUDA sice myj must be a bool")
    required = RUC_SEA_ICE_PROFILE_INPUTS + RUC_SEA_ICE_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC CUDA sice inputs: {', '.join(missing)}")

    first = cp.asarray(values[RUC_SEA_ICE_PROFILE_INPUTS[0]])
    if first.ndim < 2 or first.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA sice profiles must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {first.shape}"
        )
    shape = first.shape
    profiles = {
        name: _float_profile(values[name], shape, name)
        for name in RUC_SEA_ICE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SEA_ICE_COLUMN_INPUTS
    }
    if bool(cp.any(profiles["thdifice"][0] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA sice top-level thdifice must be positive")
    if bool(cp.any(profiles["capice"][0] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA sice top-level capice must be positive")
    if bool(cp.any(columns["patm"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA sice patm must be positive")
    if bool(cp.any(columns["rho"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA sice rho must be positive")

    scalar_names = (
        "dew", "soilt", "qvg", "qsg", "qcg", "eeta", "qfx", "hfx",
        "s", "evapl", "prcpl", "fltot",
    )
    tso = cp.empty(shape, dtype=DTYPE)
    outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in scalar_names
    }
    tbq = _device_tbq(int(cp.cuda.runtime.getDevice()))
    ncolumn = int(np.prod(horizontal_shape))
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "sice")
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_sea_ice_step")
    kernel(
        (blocks,),
        (threads,),
        (
            profiles["capice"],
            profiles["thdifice"],
            profiles["tso"],
            *(columns[name] for name in (
                "prcpms", "rainf", "patm", "qvatm", "emiss", "rnet",
                "qkms", "tkms", "rho", "tabs", "soilt", "qvg", "qsg",
            )),
            tbq,
            timestep,
            constant_flux_depth,
            water_heat_capacity,
            np.int32(myj),
            tso,
            *(outputs[name] for name in scalar_names),
            np.int32(ncolumn),
        ),
    )

    # module_sf_ruclsm.F:1871-1877 and :2184-2190.  sice never writes the
    # soil water arrays; sfctmp forces them after both call sites.
    forced = {
        "soilmois": np.float32(1.0),
        "soiliqw": np.float32(0.0),
        "soilice": np.float32(1.0),
        "smfrkeep": np.float32(1.0),
        "keepfr": np.float32(0.0),
    }
    result = RucSeaIceStepCuda(
        tso=tso,
        **{
            name: cp.full(shape, value, dtype=DTYPE)
            for name, value in forced.items()
        },
        **outputs,
    )
    for name in ("tso", *scalar_names):
        if not bool(cp.all(cp.isfinite(getattr(result, name)))):
            raise ValueError(f"RUC CUDA sice produced non-finite {name}")
    return result


__all__ = [
    "RucSeaIceStepCuda",
    "RucSoilPropertiesCuda",
    "RucSoilMoistureCuda",
    "RucSoilStepCuda",
    "RucSoilTemperatureCuda",
    "RucSurfaceParametersCuda",
    "RucTranspirationCuda",
    "ruc_qsn_cuda",
    "ruc_sea_ice_step_cuda",
    "ruc_soil_properties_cuda",
    "ruc_soil_moisture_step_cuda",
    "ruc_soil_step_cuda",
    "ruc_soil_temperature_step_cuda",
    "ruc_surface_parameters_cuda",
    "ruc_transpiration_cuda",
]


# --------------------------------------------------------------------------
# WRF v4.6.1 sfctmp snow preparation (phys/module_sf_ruclsm.F:1400-1766)
# --------------------------------------------------------------------------

# Appended import rather than an edit to the module's import block, so this
# lane's diff stays contiguous.
from gpuwm.core.ruc import (  # noqa: E402
    RUC_SNOW_COVER_OPTION,
    RUC_SNOW_PREP_COLUMN_INPUTS,
    RUC_SNOW_PREP_COLUMN_OUTPUTS,
    RUC_SNOW_PREP_PROFILE_OUTPUTS,
    RucSnowPreparation,
)


@dataclass(frozen=True)
class RucSnowPreparationCuda:
    """Device-resident state left by WRF ``sfctmp``'s snow-preparation block.

    Field for field the same contract as
    ``gpuwm.core.ruc.RucSnowPreparation``: every value written by
    ``phys/module_sf_ruclsm.F:1400-1766``, with ``iland`` the only integer.
    """

    tice: cp.ndarray
    rhosice: cp.ndarray
    capice: cp.ndarray
    thdifice: cp.ndarray
    snhei_crit: cp.ndarray
    snhei_crit_newsn: cp.ndarray
    zntsn: cp.ndarray
    snow_mosaic: cp.ndarray
    snfr: cp.ndarray
    newsn: cp.ndarray
    newsnowratio: cp.ndarray
    snowfracnewsn: cp.ndarray
    rhonewsn: cp.ndarray
    smelt: cp.ndarray
    rainf: cp.ndarray
    rsm: cp.ndarray
    dd1: cp.ndarray
    infiltr: cp.ndarray
    vegfrac: cp.ndarray
    drip: cp.ndarray
    dripsn: cp.ndarray
    dripliq: cp.ndarray
    smf: cp.ndarray
    interw: cp.ndarray
    intersn: cp.ndarray
    infwater: cp.ndarray
    intwratio: cp.ndarray
    gswnew: cp.ndarray
    gswin: cp.ndarray
    albice: cp.ndarray
    albsn: cp.ndarray
    emissn: cp.ndarray
    emiss_snowfree: cp.ndarray
    keep_snow_albedo: cp.ndarray
    snowfrac2: cp.ndarray
    snwe: cp.ndarray
    snhei: cp.ndarray
    snowfrac: cp.ndarray
    rhosn: cp.ndarray
    rhosnfall: cp.ndarray
    cst: cp.ndarray
    alb: cp.ndarray
    emiss: cp.ndarray
    znt: cp.ndarray
    iland: cp.ndarray


@lru_cache(maxsize=None)
def _snow_preparation_tables(
    device_id: int,
    mminlu: str,
) -> tuple[cp.ndarray, cp.ndarray, int, int]:
    """Upload the two VEGPARM columns the preparation block indexes.

    ``z0tbl`` reaches ``zntsn`` (``:1421``) and the roughness blend
    (``:1674-1678``); ``lemitbl`` reaches ``emiss_snowfree`` (``:1465``).  The
    ``URBAN`` category index gates the ``:1645`` snow-fraction clamp.
    """

    with cp.cuda.Device(device_id):
        vegetation = load_ruc_parameters().vegetation_for(mminlu)
        roughness = cp.asarray(
            [row.z0 for row in vegetation.rows], dtype=DTYPE
        )
        emissivity = cp.asarray(
            [row.lemi for row in vegetation.rows], dtype=DTYPE
        )
    return (
        roughness,
        emissivity,
        int(vegetation.scalars["URBAN"]),
        len(vegetation.rows),
    )


def ruc_snow_preparation_cuda(
    values: dict[str, object],
    *,
    delt: float,
    ivgtyp,
    iland,
    isice: int = 15,
    c1sn: float = 0.026,
    c2sn: float = 21.0,
    isncovr_opt: int = RUC_SNOW_COVER_OPTION,
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH",
) -> RucSnowPreparationCuda:
    """Run WRF ``sfctmp``'s snow-preparation block on the GPU.

    ``phys/module_sf_ruclsm.F:1400-1766``, one thread per column, stopping
    immediately before the ``:1767`` dispatch to ``soil``, ``snowsoil``,
    ``sice`` and ``snowseaice``.

    Every arithmetic boundary in the kernel is pinned with round-to-nearest
    intrinsics, and ``exp``/``tanh`` avoid the device library because the
    ``:1497`` compaction amplifies a 1 ULP transcendental difference into
    thousands of ULP of ``rhosn``.

    ``tanh`` is a genuine reproduction of glibc's ``tanhf`` -- the fdlibm
    ``s_tanhf.c`` reduction, spelled out.  ``exp`` is not: it is a float64
    ``exp`` rounded once, which is a third function rather than glibc's
    ``expf`` (glibc 2.39 is not correctly rounded).  It matches the host
    shim exactly, so CPU and GPU agree with each other; it does not match
    what gfortran linked.  See ``gpuwm/core/ruc.py``'s ``_f32_exp``.

    ``isncovr_opt`` is a compile-time parameter in WRF
    (``module_sf_ruclsm.F:78``), so only option 2 is oracle-verified; options
    1 and 3 are transcribed but unverified.
    """

    timestep = np.float32(delt)
    density_a = np.float32(c1sn)
    density_b = np.float32(c2sn)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError(
            "RUC CUDA snow preparation delt must be finite and positive"
        )
    if not np.isfinite(density_a) or not np.isfinite(density_b):
        raise ValueError("RUC CUDA snow preparation c1sn/c2sn must be finite")
    if isncovr_opt not in (1, 2, 3):
        raise ValueError("RUC isncovr_opt must be 1, 2, or 3")
    if type(isice) is not int:
        raise TypeError("RUC CUDA snow preparation isice must be an int")
    missing = [
        name
        for name in ("ts1d",) + RUC_SNOW_PREP_COLUMN_INPUTS
        if name not in values
    ]
    if missing:
        raise TypeError(
            f"missing RUC CUDA snow preparation inputs: {', '.join(missing)}"
        )

    profile = cp.asarray(values["ts1d"], dtype=DTYPE)
    if profile.ndim < 2 or profile.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA snow preparation ts1d must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {profile.shape}"
        )
    shape = profile.shape
    horizontal_shape = shape[1:]
    ts1d = _float_profile(values["ts1d"], shape, "ts1d")
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_PREP_COLUMN_INPUTS
    }
    vegetation_category = _integer_field(
        cp.asarray(ivgtyp), horizontal_shape, "ivgtyp"
    )
    land_category = _integer_field(cp.asarray(iland), horizontal_shape, "iland")

    device_id = int(cp.cuda.runtime.getDevice())
    roughness, emissivity, urban, ncategory = _snow_preparation_tables(
        device_id, mminlu
    )
    for name, category in (
        ("ivgtyp", vegetation_category), ("iland", land_category)
    ):
        if bool(cp.any((category < 1) | (category > ncategory))):
            raise ValueError(f"RUC {name} is outside 1..{ncategory}")
    if not 1 <= isice <= ncategory:
        raise ValueError(f"RUC isice is outside 1..{ncategory}")
    if bool(cp.any(columns["rhosn"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snow preparation rhosn must be positive")
    if bool(cp.any(columns["alb"] >= cp.float32(1.0))):
        raise ValueError("RUC CUDA snow preparation alb must be below 1")

    profiles = {
        name: cp.empty(shape, dtype=DTYPE)
        for name in RUC_SNOW_PREP_PROFILE_OUTPUTS
    }
    outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in RUC_SNOW_PREP_COLUMN_OUTPUTS
    }
    land_result = cp.empty(horizontal_shape, dtype=cp.int32)

    ncolumn = int(np.prod(horizontal_shape))
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_snow_preparation")
    kernel(
        (blocks,),
        (threads,),
        (
            ts1d,
            *(columns[name] for name in RUC_SNOW_PREP_COLUMN_INPUTS),
            vegetation_category,
            land_category,
            roughness,
            emissivity,
            timestep,
            density_a,
            density_b,
            np.int32(isice),
            np.int32(urban),
            np.int32(isncovr_opt),
            *(profiles[name] for name in RUC_SNOW_PREP_PROFILE_OUTPUTS),
            *(outputs[name] for name in RUC_SNOW_PREP_COLUMN_OUTPUTS),
            land_result,
            np.int32(ncolumn),
        ),
    )

    result = RucSnowPreparationCuda(**profiles, **outputs, iland=land_result)
    for name in RUC_SNOW_PREP_PROFILE_OUTPUTS + RUC_SNOW_PREP_COLUMN_OUTPUTS:
        if not bool(cp.all(cp.isfinite(getattr(result, name)))):
            raise ValueError(
                f"RUC CUDA snow preparation produced non-finite {name}"
            )
    return result


__all__ += [
    "RucSnowPreparationCuda",
    "ruc_snow_preparation_cuda",
]


# ---------------------------------------------------------------------------
# WRF v4.6.1 ``phys/module_sf_ruclsm.F:3789-4526`` subroutine ``snowseaice``.
# ---------------------------------------------------------------------------

from gpuwm.core.ruc import (  # noqa: E402
    RUC_SNOW_SEA_ICE_COLUMN_INPUTS,
    RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS,
    RUC_SNOW_SEA_ICE_INTEGER_INPUTS,
    RUC_SNOW_SEA_ICE_PROFILE_INPUTS,
)


@dataclass(frozen=True)
class RucSnowSeaIceStepCuda:
    """Device-resident snow-on-sea-ice state and fluxes from ``snowseaice``.

    Mirrors ``gpuwm.core.ruc.RucSnowSeaIceStep`` field for field, with every
    array left on the GPU.
    """

    tso: cp.ndarray
    ilnb: cp.ndarray
    snweprint: cp.ndarray
    snheiprint: cp.ndarray
    rsm: cp.ndarray
    dew: cp.ndarray
    soilt: cp.ndarray
    soilt1: cp.ndarray
    tsnav: cp.ndarray
    qvg: cp.ndarray
    qsg: cp.ndarray
    qcg: cp.ndarray
    smelt: cp.ndarray
    snoh: cp.ndarray
    snflx: cp.ndarray
    snom: cp.ndarray
    eeta: cp.ndarray
    qfx: cp.ndarray
    hfx: cp.ndarray
    s: cp.ndarray
    sublim: cp.ndarray
    prcpl: cp.ndarray
    fltot: cp.ndarray
    snwe: cp.ndarray
    snhei: cp.ndarray
    rhosn: cp.ndarray
    emiss: cp.ndarray
    alb: cp.ndarray
    znt: cp.ndarray


def ruc_snow_sea_ice_step_cuda(
    values: dict[str, object],
    *,
    delt: float,
    conflx: float = 40.0,
    myj: bool = False,
    cw: float = 4.183e6,
    xlv: float = 2.5e6,
) -> RucSnowSeaIceStepCuda:
    """Run the complete WRF RUC snow-on-sea-ice column on the GPU.

    One thread per column: one, two or blended snow layers over the nine ice
    levels, ``vilka`` closing the skin balance, the single non-iterated melt
    pass, and the 271.4 K ice cap.  The arguments WRF passes but
    ``snowseaice`` never reads -- ``snhei_crit``, ``qcatm``, ``gsw``,
    ``tice``, ``rhosice``, ``dtdzs2``, ``glw``, ``ktau``, ``i``, ``j``,
    ``iland``, ``isoil`` and the incoming ``qcg`` -- are omitted, matching
    the CPU transcription.
    """

    timestep = np.float32(delt)
    raw_flux_depth = conflx
    water_heat_capacity = np.float32(cw)
    vaporization = np.float32(xlv)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC CUDA snowseaice delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC CUDA snowseaice cw must be finite and positive")
    if not np.isfinite(vaporization) or vaporization <= 0.0:
        raise ValueError("RUC CUDA snowseaice xlv must be finite and positive")
    if type(myj) is not bool:
        raise TypeError("RUC CUDA snowseaice myj must be a bool")
    required = (
        RUC_SNOW_SEA_ICE_PROFILE_INPUTS
        + RUC_SNOW_SEA_ICE_COLUMN_INPUTS
        + RUC_SNOW_SEA_ICE_INTEGER_INPUTS
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(
            f"missing RUC CUDA snowseaice inputs: {', '.join(missing)}"
        )

    first = cp.asarray(values[RUC_SNOW_SEA_ICE_PROFILE_INPUTS[0]])
    if first.ndim < 2 or first.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA snowseaice profiles must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {first.shape}"
        )
    shape = first.shape
    profiles = {
        name: _float_profile(values[name], shape, name)
        for name in RUC_SNOW_SEA_ICE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_SEA_ICE_COLUMN_INPUTS
    }
    integers = {
        name: _integer_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_SEA_ICE_INTEGER_INPUTS
    }
    if bool(cp.any(profiles["thdifice"][0] <= cp.float32(0.0))):
        raise ValueError(
            "RUC CUDA snowseaice top-level thdifice must be positive"
        )
    if bool(cp.any(profiles["capice"][0] <= cp.float32(0.0))):
        raise ValueError(
            "RUC CUDA snowseaice top-level capice must be positive"
        )
    if bool(cp.any(columns["patm"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowseaice patm must be positive")
    if bool(cp.any(columns["rho"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowseaice rho must be positive")
    if bool(cp.any(columns["rhosn"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowseaice rhosn must be positive")
    if bool(cp.any(columns["snwe"] < cp.float32(0.0))):
        raise ValueError("RUC CUDA snowseaice snwe must be nonnegative")

    tso = cp.empty(shape, dtype=DTYPE)
    layer_count = cp.empty(horizontal_shape, dtype=cp.int32)
    outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS
    }
    tbq = _device_tbq(int(cp.cuda.runtime.getDevice()))
    ncolumn = int(np.prod(horizontal_shape))
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "snowseaice")
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_snow_sea_ice_step")
    kernel(
        (blocks,),
        (threads,),
        (
            profiles["capice"],
            profiles["thdifice"],
            profiles["tso"],
            *(columns[name] for name in RUC_SNOW_SEA_ICE_COLUMN_INPUTS),
            *(integers[name] for name in RUC_SNOW_SEA_ICE_INTEGER_INPUTS),
            tbq,
            timestep,
            constant_flux_depth,
            water_heat_capacity,
            vaporization,
            np.int32(myj),
            tso,
            layer_count,
            *(outputs[name] for name in RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS),
            np.int32(ncolumn),
        ),
    )

    result = RucSnowSeaIceStepCuda(tso=tso, ilnb=layer_count, **outputs)
    for name in ("tso", *RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS):
        if not bool(cp.all(cp.isfinite(getattr(result, name)))):
            raise ValueError(f"RUC CUDA snowseaice produced non-finite {name}")
    return result


__all__ += [
    "RucSnowSeaIceStepCuda",
    "ruc_snow_sea_ice_step_cuda",
]


# Appended after __all__ so the three concurrent RUC snow ports stay in
# separate contiguous blocks; RucSeaIceStepCuda sets the precedent that the
# snow-lane exports are reached by direct import.
from gpuwm.core.ruc import (  # noqa: E402
    RUC_SNOW_TEMPERATURE_COLUMN_INPUTS,
    RUC_SNOW_TEMPERATURE_PROFILE_INPUTS,
)


@dataclass(frozen=True)
class RucSnowTemperatureCuda:
    """Device-resident snow/soil heat state from WRF ``snowtemp``.

    ``storage`` is WRF's local ``x`` and ``ilnb`` is the snow layer count,
    which ``snowtemp`` declares ``intent(out)`` yet reads back at the final
    ``tsnav`` update (``phys/module_sf_ruclsm.F:5716``).
    """

    tso: cp.ndarray
    soilt: cp.ndarray
    soilt1: cp.ndarray
    tsnav: cp.ndarray
    qvg: cp.ndarray
    qsg: cp.ndarray
    qcg: cp.ndarray
    dew: cp.ndarray
    snwe: cp.ndarray
    snhei: cp.ndarray
    rhosn: cp.ndarray
    beta: cp.ndarray
    smelt: cp.ndarray
    snoh: cp.ndarray
    snflx: cp.ndarray
    s: cp.ndarray
    rsm: cp.ndarray
    snweprint: cp.ndarray
    snheiprint: cp.ndarray
    storage: cp.ndarray
    ilnb: cp.ndarray


_RUC_SNOW_TEMPERATURE_SCALAR_OUTPUTS = (
    "soilt", "soilt1", "tsnav", "qvg", "qsg", "qcg", "dew", "snwe",
    "snhei", "rhosn", "beta", "smelt", "snoh", "snflx", "s", "rsm",
    "snweprint", "snheiprint", "storage",
)


def ruc_snow_temperature_step_cuda(
    values: dict[str, object],
    *,
    delt: float,
    conflx: float = 40.0,
    nroot: object = 4,
    ilnb: object = 1,
    xlvm: float = 2.835e6,
    cvw: float = 4.183e6,
) -> RucSnowTemperatureCuda:
    """Run the complete WRF RUC ``snowtemp`` snow column on the GPU.

    ``phys/module_sf_ruclsm.F:4836-5728``.  One thread per column: the soil
    heat sweep, the one-layer, two-layer or blended snow coefficient row, the
    surface energy balance closed by ``vilka``, the melt iteration and the
    bottom-melt, density and flux epilogue.

    The arguments WRF passes but ``snowtemp`` never reads -- ``i``, ``j``,
    ``ktau``, ``iland``, ``isoil``, ``qcatm``, ``gsw``, ``pc``, ``dqm``,
    ``qmin``, ``psis``, ``bclh``, ``mavail``, ``rovcp``, ``g0_p``, ``glw``,
    ``cst`` and the incoming ``qsg``/``qcg``/``tsnav`` -- are omitted,
    matching the CPU transcription.
    """

    timestep = np.float32(delt)
    raw_flux_depth = conflx
    water_heat_capacity = np.float32(cvw)
    latent_heat = np.float32(xlvm)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC CUDA snowtemp delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC CUDA snowtemp cvw must be finite and positive")
    if not np.isfinite(latent_heat) or latent_heat <= 0.0:
        raise ValueError("RUC CUDA snowtemp xlvm must be finite and positive")
    required = (
        RUC_SNOW_TEMPERATURE_PROFILE_INPUTS
        + RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(
            f"missing RUC CUDA snowtemp inputs: {', '.join(missing)}"
        )

    first = cp.asarray(values[RUC_SNOW_TEMPERATURE_PROFILE_INPUTS[0]])
    if first.ndim < 2 or first.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA snowtemp profiles must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {first.shape}"
        )
    shape = first.shape
    profiles = {
        name: _float_profile(values[name], shape, name)
        for name in RUC_SNOW_TEMPERATURE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    }
    roots = _root_count_field(nroot, horizontal_shape)
    raw_layers = cp.asarray(ilnb)
    if raw_layers.dtype.kind not in "iu":
        raise TypeError("RUC CUDA snowtemp ilnb must contain integer counts")
    try:
        layers = cp.ascontiguousarray(
            cp.broadcast_to(raw_layers, horizontal_shape), dtype=cp.int32
        )
    except ValueError as exc:
        raise ValueError(
            f"ilnb shape {raw_layers.shape} is not broadcastable to "
            f"{horizontal_shape}"
        ) from exc
    if bool(cp.any(profiles["thdif"][0] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowtemp top-level thdif must be positive")
    if bool(cp.any(profiles["cap"][0] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowtemp top-level cap must be positive")
    for name in ("patm", "rho", "rhosn", "snhei", "snth", "deltsn"):
        if bool(cp.any(columns[name] <= cp.float32(0.0))):
            raise ValueError(f"RUC CUDA snowtemp {name} must be positive")

    tso = cp.empty(shape, dtype=DTYPE)
    outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in _RUC_SNOW_TEMPERATURE_SCALAR_OUTPUTS
    }
    layer_out = cp.empty(horizontal_shape, dtype=cp.int32)
    tbq = _device_tbq(int(cp.cuda.runtime.getDevice()))
    ncolumn = int(np.prod(horizontal_shape))
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "snowtemp")
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = get_kernel("ruc", "ruc_snow_temperature_step")
    kernel(
        (blocks,),
        (threads,),
        (
            profiles["cap"],
            profiles["thdif"],
            profiles["tranf"],
            profiles["tso"],
            *(columns[name] for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS),
            roots,
            layers,
            tbq,
            timestep,
            constant_flux_depth,
            latent_heat,
            water_heat_capacity,
            tso,
            *(
                outputs[name]
                for name in _RUC_SNOW_TEMPERATURE_SCALAR_OUTPUTS
            ),
            layer_out,
            np.int32(ncolumn),
        ),
    )

    result = RucSnowTemperatureCuda(tso=tso, ilnb=layer_out, **outputs)
    for name in ("tso", *_RUC_SNOW_TEMPERATURE_SCALAR_OUTPUTS):
        if not bool(cp.all(cp.isfinite(getattr(result, name)))):
            raise ValueError(f"RUC CUDA snowtemp produced non-finite {name}")
    return result


__all__ += [
    "RucSnowTemperatureCuda",
    "ruc_snow_temperature_step_cuda",
]


# ---------------------------------------------------------------------------
# WRF v4.6.1 phys/module_sf_ruclsm.F:3120-3786 subroutine snowsoil on the GPU.
# Appended as one contiguous block, exporting through __all__ +=, so the three
# parallel RUC snow ports stay mergeable.
# ---------------------------------------------------------------------------

from gpuwm.core.ruc import (  # noqa: E402
    RUC_SNOW_SOIL_COLUMN_INPUTS,
    RUC_SNOW_SOIL_PROFILE_INPUTS,
)


@dataclass(frozen=True)
class RucSnowSoilStepCuda:
    """Device-resident snow-covered land state and fluxes from ``snowsoil``.

    ``soilice``/``soiliqw`` carry the freezing-curve partition rebuilt after
    ``snowtemp`` but before ``soilmoist`` (``:3626-3648``); ``soilmoist``
    never writes ``soiliqw`` back, so they partition the moisture state as it
    stood on entry, not the ``soilmois`` returned beside them.
    """

    soilmois: cp.ndarray
    tso: cp.ndarray
    smfrkeep: cp.ndarray
    keepfr: cp.ndarray
    soilice: cp.ndarray
    soiliqw: cp.ndarray
    cst: cp.ndarray
    dew: cp.ndarray
    soilt: cp.ndarray
    soilt1: cp.ndarray
    tsnav: cp.ndarray
    qvg: cp.ndarray
    qsg: cp.ndarray
    qcg: cp.ndarray
    snwe: cp.ndarray
    snhei: cp.ndarray
    rhosn: cp.ndarray
    ilnb: cp.ndarray
    snweprint: cp.ndarray
    snheiprint: cp.ndarray
    rsm: cp.ndarray
    smelt: cp.ndarray
    snoh: cp.ndarray
    snflx: cp.ndarray
    snom: cp.ndarray
    edir1: cp.ndarray
    ec1: cp.ndarray
    ett1: cp.ndarray
    eeta: cp.ndarray
    qfx: cp.ndarray
    hfx: cp.ndarray
    s: cp.ndarray
    sublim: cp.ndarray
    prcpl: cp.ndarray
    fltot: cp.ndarray
    runoff1: cp.ndarray
    runoff2: cp.ndarray
    mavail: cp.ndarray
    infiltrp: cp.ndarray


def ruc_snow_soil_step_cuda(
    values: dict[str, object],
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
) -> RucSnowSoilStepCuda:
    """Run the complete snow-covered WRF RUC land column on the GPU."""

    if myj is not False:
        raise ValueError("RUC CUDA snow soil lane supports myj=False only")
    timestep = np.float32(delt)
    raw_flux_depth = conflx
    water_heat_capacity = np.float32(cw)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC CUDA snowsoil delt must be finite and positive")
    if not np.isfinite(water_heat_capacity) or water_heat_capacity <= 0.0:
        raise ValueError("RUC CUDA snowsoil cw must be finite and positive")
    required = RUC_SNOW_SOIL_PROFILE_INPUTS + RUC_SNOW_SOIL_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC CUDA snowsoil inputs: {', '.join(missing)}")

    first = cp.asarray(values[RUC_SNOW_SOIL_PROFILE_INPUTS[0]])
    if first.ndim < 2 or first.shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC CUDA snowsoil profiles must have shape "
            f"({NUM_SOIL_LAYERS}, ...horizontal...), got {first.shape}"
        )
    shape = first.shape
    profiles = {
        name: _float_profile(values[name], shape, name)
        for name in RUC_SNOW_SOIL_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape)
    land_type = _integer_field(iland, horizontal_shape, "iland")
    snow_layers = cp.asarray(ilnb)
    if snow_layers.dtype.kind not in "iu":
        raise TypeError("RUC CUDA snowsoil ilnb must be an integer layer count")
    if snow_layers.shape != horizontal_shape:
        try:
            snow_layers = cp.broadcast_to(snow_layers, horizontal_shape)
        except ValueError as exc:
            raise ValueError(
                f"ilnb shape {snow_layers.shape} is not broadcastable to "
                f"{horizontal_shape}"
            ) from exc
    snow_layers = cp.ascontiguousarray(snow_layers, dtype=cp.int32)
    columns = {
        name: _float_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_SOIL_COLUMN_INPUTS
    }
    if bool(cp.any(columns["dqm"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowsoil dqm must be positive")
    if bool(cp.any(columns["psis"] >= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowsoil psis must be negative")
    if bool(cp.any(columns["bclh"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowsoil bclh must be positive")
    if bool(cp.any(columns["sat"] <= cp.float32(0.0))):
        raise ValueError("RUC CUDA snowsoil canopy saturation must be positive")
    if bool(cp.any(columns["rho"] <= cp.float32(0.0))) or bool(
        cp.any(columns["patm"] <= cp.float32(0.0))
    ):
        raise ValueError("RUC CUDA snowsoil rho and patm must be positive")
    if bool(cp.any(columns["rhosn"] <= cp.float32(0.0))) or bool(
        cp.any(columns["rhonewsn"] <= cp.float32(0.0))
    ):
        raise ValueError("RUC CUDA snowsoil snow densities must be positive")
    if bool(cp.any(columns["snwe"] < cp.float32(0.0))) or bool(
        cp.any(columns["snhei"] < cp.float32(0.0))
    ):
        raise ValueError("RUC CUDA snowsoil snow depth must be nonnegative")
    if bool(cp.any(
        (columns["snowfrac"] < cp.float32(0.0))
        | (columns["snowfrac"] > cp.float32(1.0))
    )):
        raise ValueError("RUC CUDA snowsoil snowfrac must be within 0..1")

    xlv = np.float32(2.5e6)
    xlmelt = np.float32(3.35e5)
    # :3357 snowsoil closes its budget with the sublimation latent heat.
    xlvm = np.float32(xlv + xlmelt)
    source_riw = np.float32(np.float32(900.0) * np.float32(1.0e-3))

    soilmois = profiles["soilmois"].copy()
    tso = profiles["tso"].copy()
    smfrkeep = profiles["smfrkeep"].copy()
    keepfr = profiles["keepfr"].copy()
    told = tso.copy()
    smold = soilmois.copy()
    phase = _soil_phase_partition_cuda(
        soilmois, tso, smfrkeep, keepfr, columns, update_smfrkeep=True
    )
    properties = ruc_soil_properties_cuda(
        {
            **{name: phase[name] for name in (
                "fwsat", "lwsat", "tav", "soilmoism", "soiliqwm", "soilicem",
            )},
            "keepfr": keepfr,
            "soilmois": soilmois,
            "soiliqw": phase["soiliqw"],
            "soilice": phase["soilice"],
            **{name: columns[name] for name in (
                "qwrtz", "rhocs", "dqm", "qmin", "psis", "bclh", "ksat",
            )},
        },
        riw=float(source_riw),
    )

    ncolumn = int(np.prod(horizontal_shape))
    # snowsoil LAUNCHES ruc_snow_temperature_step itself rather than going
    # through ruc_snow_temperature_step_cuda, so it has to bind the device
    # conflx here too.  Missing this passed a host object straight into a
    # kernel parameter that is now a pointer.
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "snowsoil")
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    canopy_names = ("beta", "wetcan", "drycan", "snwe", "ras")
    canopy = {
        name: cp.empty(horizontal_shape, dtype=DTYPE) for name in canopy_names
    }
    get_kernel("ruc", "ruc_snow_soil_canopy_setup")(
        (blocks,),
        (threads,),
        (
            *(columns[name] for name in (
                "qvatm", "qsg", "qkms", "rho", "vegfrac", "snwe", "cst",
                "sat", "cn",
            )),
            timestep,
            *(canopy[name] for name in canopy_names),
            np.int32(ncolumn),
        ),
    )
    transpiration = ruc_transpiration_cuda(
        phase["soiliqw"],
        columns["tabs"], columns["lai"], columns["gswin"],
        columns["dqm"], columns["qmin"], columns["ref"], columns["wilt"],
        columns["pc"], land_type, nroot=roots, mminlu=mminlu,
        parameters=parameters,
    )

    # :3387-3398 deltsn and snth, in a kernel rather than in cupy elementwise
    # arithmetic so every operation boundary stays a pinned intrinsic.
    thresholds = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in ("deltsn", "snth")
    }
    get_kernel("ruc", "ruc_snow_layer_thresholds")(
        (blocks,),
        (threads,),
        (
            columns["rhosn"], columns["snhei"],
            thresholds["deltsn"], thresholds["snth"],
            np.int32(ncolumn),
        ),
    )

    # :3580 the one call to snowtemp.  The argument order is taken from the
    # same three sequences ruc_snow_temperature_step_cuda launches with, so
    # the two call sites cannot drift: a raw kernel checks nothing but dtype.
    snowtemp_inputs = {
        "cap": properties.cap,
        "thdif": properties.thdif,
        "tranf": transpiration.tranf,
        "tso": tso,
        "snwe": canopy["snwe"],
        "snwepr": columns["snwe"],
        "beta": canopy["beta"],
        "deltsn": thresholds["deltsn"],
        "snth": thresholds["snth"],
        "drycan": canopy["drycan"],
        "wetcan": canopy["wetcan"],
        "transum": transpiration.transum,
        # :3535 snowsoil passes a literal 0. for dew.
        "dew": cp.zeros(horizontal_shape, dtype=DTYPE),
        **{name: columns[name] for name in (
            "snhei", "newsnow", "snowfrac", "rhosn", "rhonewsn", "meltfactor",
            "prcpms", "rainf", "patm", "tabs", "qvatm", "emiss", "rnet",
            "qkms", "tkms", "rho", "vegfrac", "soilt", "soilt1", "qvg",
        )},
    }
    snow_names = _RUC_SNOW_TEMPERATURE_SCALAR_OUTPUTS
    snow = {
        name: cp.empty(horizontal_shape, dtype=DTYPE) for name in snow_names
    }
    updated_tso = cp.empty(shape, dtype=DTYPE)
    updated_layers = cp.empty(horizontal_shape, dtype=cp.int32)
    get_kernel("ruc", "ruc_snow_temperature_step")(
        (blocks,),
        (threads,),
        (
            *(
                snowtemp_inputs[name]
                for name in RUC_SNOW_TEMPERATURE_PROFILE_INPUTS
            ),
            *(
                snowtemp_inputs[name]
                for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
            ),
            roots, snow_layers, _device_tbq(int(cp.cuda.runtime.getDevice())),
            timestep, constant_flux_depth, xlvm, water_heat_capacity,
            updated_tso,
            *(snow[name] for name in _RUC_SNOW_TEMPERATURE_SCALAR_OUTPUTS),
            updated_layers,
            np.int32(ncolumn),
        ),
    )
    tso = updated_tso

    prepared_names = ("ett1", "dew", "prcp")
    prepared = {
        name: cp.empty(horizontal_shape, dtype=DTYPE) for name in prepared_names
    }
    transp = cp.empty(shape, dtype=DTYPE)
    get_kernel("ruc", "ruc_snow_soil_prepare_moisture")(
        (blocks,),
        (threads,),
        (
            columns["qvatm"], snow["qsg"], columns["qkms"],
            columns["vegfrac"], canopy["drycan"], canopy["ras"],
            transpiration.tranf, roots, columns["infwater"],
            transp,
            *(prepared[name] for name in prepared_names),
            np.int32(ncolumn),
        ),
    )

    phase = _soil_phase_partition_cuda(
        soilmois, tso, smfrkeep, keepfr, columns, update_smfrkeep=False
    )
    zeros = cp.zeros(horizontal_shape, dtype=DTYPE)
    moisture = ruc_soil_moisture_step_cuda(
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
            "prcp": prepared["prcp"],
            "qkms": columns["qkms"],
            "drip": zeros,
            "dew": zeros,
            "smelt": snow["smelt"],
            "vegfrac": columns["vegfrac"],
            "snowfrac": columns["snowfrac"],
            "soilres": cp.ones(horizontal_shape, dtype=DTYPE),
            "dqm": columns["dqm"],
            "qmin": columns["qmin"],
            "ref": columns["ref"],
            "ksat": columns["ksat"],
            "ras": canopy["ras"],
        },
        delt=float(timestep),
    )
    soilmois = moisture.soilmois

    final_names = (
        "tsnav", "snom", "cst", "dew", "ett1", "edir1", "ec1", "eeta",
        "qfx", "hfx", "sublim", "fltot",
    )
    final = {
        name: cp.empty(horizontal_shape, dtype=DTYPE) for name in final_names
    }
    get_kernel("ruc", "ruc_snow_soil_finalize")(
        (blocks,),
        (threads,),
        (
            phase["soilice"], tso, told, soilmois, smold, keepfr,
            snow["snhei"], snow["smelt"], columns["snom"], snow["snflx"],
            snow["snoh"], snow["storage"], snow["soilt"], snow["qsg"],
            snow["tsnav"], snow["beta"], canopy["wetcan"], prepared["ett1"],
            prepared["dew"], canopy["ras"], columns["cst"],
            *(columns[name] for name in (
                "tkms", "rho", "tabs", "patm", "qkms", "qvatm", "vegfrac",
                "rnet",
            )),
            timestep, xlvm,
            *(final[name] for name in final_names),
            np.int32(ncolumn),
        ),
    )

    result = RucSnowSoilStepCuda(
        soilmois=soilmois,
        tso=tso,
        smfrkeep=smfrkeep,
        keepfr=keepfr,
        soilice=phase["soilice"],
        soiliqw=moisture.soiliqw,
        cst=final["cst"],
        dew=final["dew"],
        soilt=snow["soilt"],
        soilt1=snow["soilt1"],
        tsnav=final["tsnav"],
        qvg=snow["qvg"],
        qsg=snow["qsg"],
        qcg=snow["qcg"],
        snwe=snow["snwe"],
        snhei=snow["snhei"],
        rhosn=snow["rhosn"],
        ilnb=updated_layers,
        snweprint=snow["snweprint"],
        snheiprint=snow["snheiprint"],
        rsm=snow["rsm"],
        smelt=snow["smelt"],
        snoh=snow["snoh"],
        snflx=snow["snflx"],
        snom=final["snom"],
        edir1=final["edir1"],
        ec1=final["ec1"],
        ett1=final["ett1"],
        eeta=final["eeta"],
        qfx=final["qfx"],
        hfx=final["hfx"],
        # :3768 snowsoil reports the snow-layer flux, not snowtemp's s
        # (:3782 is a format statement).  snow["s"] is therefore discarded.
        s=snow["snflx"],
        sublim=final["sublim"],
        prcpl=columns["prcpms"].copy(),
        fltot=final["fltot"],
        runoff1=moisture.runoff,
        runoff2=moisture.runoff2,
        mavail=moisture.mavail,
        infiltrp=moisture.infiltrp,
    )
    for name in RucSnowSoilStepCuda.__dataclass_fields__:
        array = getattr(result, name)
        if array.dtype == DTYPE and not bool(cp.all(cp.isfinite(array))):
            raise ValueError(f"RUC CUDA snowsoil produced non-finite {name}")
    return result


__all__ += [
    "RucSnowSoilStepCuda",
    "ruc_snow_soil_step_cuda",
]


# ---------------------------------------------------------------------------
# The device leaves, behind ``sfctmp``'s own argument lists.
# ---------------------------------------------------------------------------
#
# ``gpuwm.core.ruc.ruc_surface_temperature_step`` takes a ``leaves`` mapping
# so a backend can replace the four routines that do arithmetic without
# :mod:`gpuwm.core.ruc` importing cupy -- which it must not, because
# ``tests/conftest.py`` auto-marks any module that does as ``gpu`` and the
# whole RUC oracle suite runs on a machine with no card.
#
# Each wrapper below takes ``sfctmp``'s own numpy arguments, runs the CUDA
# leaf, and reconstructs the HOST dataclass.  That boundary is deliberate and
# it is where this conversion stops: the recombination arithmetic in
# ``sfctmp`` itself (``:1979-2115``) stays on the host, so every batch pays
# one upload and one download rather than one per column.  Making the whole
# column device-resident is the next conversion, not this one.


def _host_facing_leaf(device_call, host_result):
    """Wrap a ``*_cuda`` leaf so it takes and returns host arrays.

    The CUDA result dataclasses mirror their host counterparts field for
    field and in the same order -- asserted by
    ``tests/test_ruc_device_column.py`` rather than assumed -- so the
    reconstruction is a rename-free transfer.
    """

    fields = tuple(host_result.__dataclass_fields__)

    def call(*args, **keywords):
        result = device_call(*args, **keywords)
        return host_result(
            **{name: cp.asnumpy(getattr(result, name)) for name in fields})

    call.__name__ = f"{device_call.__name__}_host_facing"
    call.__qualname__ = call.__name__
    call.__doc__ = (
        f"``{device_call.__name__}`` with a host-array boundary; see "
        "``_host_facing_leaf``.")
    return call


#: The device counterpart of :data:`gpuwm.core.ruc.RUC_SFCTMP_HOST_LEAVES`.
#:
#: **All four are bitwise against the host**, warm and snow-covered, measured
#: on an RTX 5090 through the batched driver at 512, 4,096 and 24,576 columns
#: and confirmed in a second process.  Until 2026-07-26 the snow-covered case
#: was not: 10 ULP of ``infiltr``, 4 of ``acrunoff``/``sfcrunoff`` and 2 of
#: ``runoff1``, while every leaf was max_ulp 0 against its own WRF fixture --
#: the "a mirror is not an oracle" trap, and only driver-level composition
#: could surface it.  The cause was two ``**`` sites left on the CUDA device
#: libm while the host used the float64-rounded-once form; see
#: ``docs/wrf_ruc_runtime_admission.md``.
#:
#: :data:`RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE` is kept because a warm grid
#: never calls the other two, not because they are unverified.
RUC_SFCTMP_DEVICE_LEAVES: Mapping[str, object] = MappingProxyType({
    "soil": _host_facing_leaf(ruc_soil_step_cuda, RucSoilStep),
    "sea_ice": _host_facing_leaf(ruc_sea_ice_step_cuda, RucSeaIceStep),
    "snow_soil": _host_facing_leaf(ruc_snow_soil_step_cuda, RucSnowSoilStep),
    "snow_sea_ice": _host_facing_leaf(
        ruc_snow_sea_ice_step_cuda, RucSnowSeaIceStep),
})


#: The two leaves a grid with no snow can reach: ``soil`` (snow-free land)
#: and ``sea_ice`` (snow-free sea ice).  A snow-covered column still runs
#: ``snowsoil`` on the host under this set, which is correct but is why a
#: snow grid gets less of the speedup -- use
#: :data:`RUC_SFCTMP_DEVICE_LEAVES` there, which is now bitwise too.
RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE: Mapping[str, object] = MappingProxyType({
    name: RUC_SFCTMP_DEVICE_LEAVES[name] for name in ("soil", "sea_ice")
})


def _host_facing_snow_prep():
    """``ruc_snow_preparation_cuda`` behind the host stage's signature.

    Two things the leaf wrapper does not have to do.  The host stage takes a
    ``bundle``; the kernel indexes device tables built from
    :func:`gpuwm.core.ruc.load_ruc_parameters`, so a caller that supplied a
    DIFFERENT bundle would silently get the default tables.  That is refused
    rather than ignored -- fail-closed, checked against the three quantities
    the kernel actually reads (``z0tbl``, ``lemitbl`` and ``URBAN``).
    """

    fields = tuple(RucSnowPreparation.__dataclass_fields__)

    def call(values, *, delt, ivgtyp, iland, isice=15, c1sn=0.026,
             c2sn=21.0, isncovr_opt=RUC_SNOW_COVER_OPTION,
             mminlu="MODIFIED_IGBP_MODIS_NOAH", bundle=None):
        if bundle is not None:
            supplied = bundle.vegetation_for(mminlu)
            default = load_ruc_parameters().vegetation_for(mminlu)
            same = (
                [row.z0 for row in supplied.rows]
                == [row.z0 for row in default.rows]
                and [row.lemi for row in supplied.rows]
                == [row.lemi for row in default.rows]
                and supplied.scalars["URBAN"] == default.scalars["URBAN"]
            )
            if not same:
                raise ValueError(
                    "the RUC CUDA snow-preparation stage indexes device "
                    "tables built from the default parameter bundle; the "
                    "supplied bundle differs in z0tbl, lemitbl or URBAN"
                )
        result = ruc_snow_preparation_cuda(
            values, delt=delt, ivgtyp=ivgtyp, iland=iland, isice=isice,
            c1sn=c1sn, c2sn=c2sn, isncovr_opt=isncovr_opt, mminlu=mminlu)
        return RucSnowPreparation(
            **{name: cp.asnumpy(getattr(result, name)) for name in fields})

    call.__name__ = "ruc_snow_preparation_cuda_host_facing"
    call.__qualname__ = call.__name__
    call.__doc__ = (
        "``ruc_snow_preparation_cuda`` with a host-array boundary; see "
        "``_host_facing_snow_prep``.")
    return call


#: The device counterpart of :data:`gpuwm.core.ruc.RUC_SFCTMP_HOST_STAGES`.
#:
#: ``ruc_snow_preparation`` is the last per-column Python loop on the RUC
#: path, and once the four leaves are on the card it is the single largest
#: term left in a land-surface call.  The kernel behind this entry is
#: ``max_ulp 0`` against the unmodified WRF preparation block over all three
#: snow-cover options; see ``tests/test_ruc_gpu.py``.
RUC_SFCTMP_DEVICE_STAGES: Mapping[str, object] = MappingProxyType({
    "snow_prep": _host_facing_snow_prep(),
})


# ---------------------------------------------------------------------------
# The whole-column-device-resident path: the sfctmp DISPATCH on the card too.
# ---------------------------------------------------------------------------

def ruc_tanhf_glibc(values) -> cp.ndarray:
    """``gpuwm.core.ruc._f32_tanh`` over a column field, as a kernel.

    The host spelling is a Python ``for`` loop over fdlibm's reduction, and
    the snow-cover rebuild at ``module_sf_ruclsm.F:2087``/``:2098`` calls it
    once per snow-covered column inside ``sfctmp``'s DISPATCH -- not inside a
    leaf, so no leaf conversion reached it and every earlier decomposition
    charged it to "driver + recombination".  ``ruc.cu``'s
    ``ruc_tanhf_glibc_array`` is the same reduction, and
    ``tests/test_ruc_device_column.py`` pins the two together over the
    arguments ``:2087`` reaches.
    """

    source = cp.ascontiguousarray(cp.asarray(values, dtype=DTYPE))
    flat = source.reshape(-1)
    out = cp.empty(flat.shape, dtype=DTYPE)
    ncolumn = int(flat.size)
    if ncolumn:
        threads = 128
        blocks = (ncolumn + threads - 1) // threads
        get_kernel("ruc", "ruc_tanhf_glibc_array")(
            (blocks,), (threads,), (flat, out, np.int32(ncolumn)))
    return out.reshape(source.shape)


class _RucDeviceFloat32:
    """``np.float32``'s two jobs, split so CuPy can do both.

    ``gpuwm.core.ruc``'s two drivers spell ``np.float32`` for a cast AND for
    a dtype -- ``np.float32(a * b)`` to pin a float32 boundary, and
    ``dtype=np.float32`` to allocate.  ``np.float32(device_array)`` raises,
    because a CuPy array has no ``__float__``.  numpy accepts any object with
    a ``dtype`` attribute wherever a dtype is wanted, so this instance is a
    valid dtype AND a callable cast, and the drivers need no second spelling.

    The cast COPIES, exactly as ``np.float32(host_array)`` does; returning the
    input where the dtype already matches would alias a caller's array into a
    scattered assignment and is not the same function.
    """

    dtype = np.dtype(np.float32)

    def __call__(self, value):
        if isinstance(value, cp.ndarray):
            return value.astype(cp.float32)
        return np.float32(value)

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return "<RUC device float32 cast/dtype>"


def _dtype_normalising(function):
    """``function`` with any ``dtype=`` argument put through ``np.dtype``.

    numpy resolves an object with a ``dtype`` attribute wherever a dtype is
    wanted, which is what makes :class:`_RucDeviceFloat32` usable as both a
    cast and an allocation dtype.  CuPy forwards its ``dtype`` argument to
    several different constructors and this makes the resolution explicit at
    the boundary rather than depending on every one of them accepting the
    same duck type.  ``np.dtype(np.dtype('float32'))`` is the identity, so
    this is a no-op for every ordinary caller.
    """

    def call(*args, dtype=None, **keywords):
        if dtype is not None:
            dtype = np.dtype(dtype)
            return function(*args, dtype=dtype, **keywords)
        return function(*args, **keywords)

    call.__name__ = getattr(function, "__name__", "call")
    call.__qualname__ = call.__name__
    return call


#: CuPy behind the numpy surface :mod:`gpuwm.core.ruc`'s drivers use.
#:
#: Passed as ``arrays=`` to :func:`gpuwm.core.ruc.ruc_land_surface_step` it
#: rebinds ``np`` for the whole driver -- the prologue's unit conversions, the
#: water and sea-ice arms, the ``sfctmp`` dispatch's masking, gathers,
#: scatters and mosaic recombination, and the epilogue's accumulators -- so a
#: column never returns to the host.  The names here are exactly the ``np.*``
#: names those bodies reach and no others: a missing one is an
#: ``AttributeError`` at the call site rather than a silent host fallback,
#: which is why this is a namespace object and not a ``getattr`` shim onto
#: cupy.
#:
#: Two entries are deliberately NOT cupy's:
#:
#: ``float32``   see :class:`_RucDeviceFloat32`.
#: ``prod``      only ever applied to a shape tuple, which is host data.
#:
#: ``ruc_tanhf_glibc`` is not a numpy name at all.  It is how
#: ``gpuwm.core.ruc._ruc_tanh_array`` learns that this namespace can do
#: fdlibm's tanh reduction as a kernel instead of as a Python loop.
RUC_DEVICE_ARRAYS = SimpleNamespace(
    float32=_RucDeviceFloat32(),
    int32=np.int32,
    intp=np.intp,
    integer=np.integer,
    ndarray=cp.ndarray,
    issubdtype=np.issubdtype,
    prod=np.prod,
    abs=cp.abs,
    all=cp.all,
    any=cp.any,
    arange=_dtype_normalising(cp.arange),
    array=_dtype_normalising(cp.array),
    asarray=_dtype_normalising(cp.asarray),
    atleast_1d=cp.atleast_1d,
    broadcast_to=cp.broadcast_to,
    count_nonzero=cp.count_nonzero,
    empty=_dtype_normalising(cp.empty),
    full=_dtype_normalising(cp.full),
    isfinite=cp.isfinite,
    maximum=cp.maximum,
    minimum=cp.minimum,
    stack=cp.stack,
    where=cp.where,
    zeros=_dtype_normalising(cp.zeros),
    ruc_tanhf_glibc=ruc_tanhf_glibc,
)


#: The four ``sfctmp`` leaves with NO host boundary, for use with
#: :data:`RUC_DEVICE_ARRAYS`.
#:
#: These are the same kernels :data:`RUC_SFCTMP_DEVICE_LEAVES` launches.  The
#: difference is entirely the return: those wrap each result in
#: ``cp.asnumpy`` field by field, which is one separately synchronised copy
#: per field per call -- more than three hundred of them for one 24,576-column
#: land-surface call, and 46 % of that call's wall clock on a warm grid,
#: 67 % on a snow-covered one.  These return the device arrays the kernels
#: already wrote.
RUC_SFCTMP_DEVICE_LEAVES_RESIDENT: Mapping[str, object] = MappingProxyType({
    "soil": ruc_soil_step_cuda,
    "sea_ice": ruc_sea_ice_step_cuda,
    "snow_soil": ruc_snow_soil_step_cuda,
    "snow_sea_ice": ruc_snow_sea_ice_step_cuda,
})


def _resident_snow_prep():
    """``ruc_snow_preparation_cuda`` with the same fail-closed bundle check.

    The guard is :func:`_host_facing_snow_prep`'s, unchanged and for the same
    reason: the kernel indexes ``z0tbl``/``lemitbl``/``URBAN`` uploaded from
    the default bundle, so a caller who supplies a different one must be
    refused rather than silently handed the default tables.
    """

    def call(values, *, delt, ivgtyp, iland, isice=15, c1sn=0.026,
             c2sn=21.0, isncovr_opt=RUC_SNOW_COVER_OPTION,
             mminlu="MODIFIED_IGBP_MODIS_NOAH", bundle=None):
        if bundle is not None:
            supplied = bundle.vegetation_for(mminlu)
            default = load_ruc_parameters().vegetation_for(mminlu)
            same = (
                [row.z0 for row in supplied.rows]
                == [row.z0 for row in default.rows]
                and [row.lemi for row in supplied.rows]
                == [row.lemi for row in default.rows]
                and supplied.scalars["URBAN"] == default.scalars["URBAN"]
            )
            if not same:
                raise ValueError(
                    "the RUC CUDA snow-preparation stage indexes device "
                    "tables built from the default parameter bundle; the "
                    "supplied bundle differs in z0tbl, lemitbl or URBAN"
                )
        return ruc_snow_preparation_cuda(
            values, delt=delt, ivgtyp=ivgtyp, iland=iland, isice=isice,
            c1sn=c1sn, c2sn=c2sn, isncovr_opt=isncovr_opt, mminlu=mminlu)

    call.__name__ = "ruc_snow_preparation_cuda_resident"
    call.__qualname__ = call.__name__
    call.__doc__ = (
        "``ruc_snow_preparation_cuda`` returning device arrays; see "
        "``_resident_snow_prep``.")
    return call


#: The preparation stage with no host boundary; see
#: :data:`RUC_SFCTMP_DEVICE_LEAVES_RESIDENT`.
RUC_SFCTMP_DEVICE_STAGES_RESIDENT: Mapping[str, object] = MappingProxyType({
    "snow_prep": _resident_snow_prep(),
})


__all__ += [
    "RUC_DEVICE_ARRAYS",
    "RUC_SFCTMP_DEVICE_LEAVES",
    "RUC_SFCTMP_DEVICE_LEAVES_RESIDENT",
    "RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE",
    "RUC_SFCTMP_DEVICE_STAGES",
    "RUC_SFCTMP_DEVICE_STAGES_RESIDENT",
    "ruc_tanhf_glibc",
]
