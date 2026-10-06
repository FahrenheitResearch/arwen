"""CUDA implementation of the pinned WRF v4.6.1 RUC setup slices."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from functools import lru_cache
from gpuwm.core.device_cache import cuda_cache, cached_ready

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
    ruc_zshalf,
    _resolved_soil_levels,
)
from gpuwm.core.ruc_contract import NUM_SOIL_LAYERS
from gpuwm.core.ruc_validation import (VALIDATION_SCAN_GROUP,
                                       RucValidationBatch)
# The tier lives in its own CuPy-free module so the identity of the
# nine-level translation unit stays provable on a box with no card; see
# gpuwm/core/ruc_tier.py.  ``_ruc_kernel`` is the ONE place a RUC launcher
# chooses between the unspecialized loader and a specialized one.
from gpuwm.core.ruc_tier import (ruc_kernel as _ruc_kernel,
                                 ruc_kernel_source, ruc_module_defines)
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


@cuda_cache(maxsize=None, ready=True)
def _default_device_tables(
    device_id: int,
    mminlu: str,
) -> tuple[_RucDeviceTables, int, int, int]:
    with cp.cuda.Device(device_id):
        return _upload_tables(_default_parameter_bundle(), mminlu)


@lru_cache(maxsize=1)
def _default_parameter_bundle():
    """The canonical bundle used to build the cached default device tables."""
    return load_ruc_parameters()


_BUNDLE_DEVICE_TABLES = {}


def _bundle_device_tables(bundle, mminlu):
    """Cache uploads by device and the values the upload consumes.

    A value key also detects replacement rows in a caller's mapping.  Object
    identity alone would retain stale tables after such a replacement.
    """
    vegetation = bundle.vegetation_for(mminlu)
    key = (int(cp.cuda.Device().id), mminlu, vegetation.name,
           vegetation.rows, float(vegetation.scalars["RSMAX_DATA"]),
           bundle.soil.rows)
    return cached_ready(cp, _BUNDLE_DEVICE_TABLES, key,
                        lambda: _upload_tables(bundle, mminlu))


@cuda_cache(maxsize=None, ready=True)
def _device_soil_half_levels(device_id, nzs):
    with cp.cuda.Device(device_id):
        zs, _ = ruc_soil_geometry(nzs)
        return cp.asarray(ruc_zshalf(zs))


@cuda_cache(maxsize=None, ready=True)
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


@cuda_cache(maxsize=None)
def _validation_scan_kernel(count: int):
    """A read-only finiteness scan over ``count`` arrays, one flag word each.

    Pointers and lengths travel as launch arguments, so every batch of the
    same width shares one compiled kernel.  ``blockIdx.y`` selects the array
    and indexes the flag block, so a refusal still knows which field tripped
    -- a batched check that lost the field name would be a regression, not an
    optimisation.  The shape is ``mynn_validate_batch``'s
    (``gpuwm/core/mynn_pbl_gpu.py``), which is the same problem solved for
    MYNN in 2.7.4; explicit FTZ matches the ``cp.isfinite`` reduction this
    replaces, which flushes nothing but is never asked to.
    """
    arguments = ", ".join(f"const float* a{index}, unsigned long long n{index}"
                          for index in range(count))
    cases = "\n".join(
        f"case {index}: data=a{index}; size=n{index}; break;"
        for index in range(count))
    source = f"""
extern "C" __global__ void ruc_validate_finite({arguments}, int* flags) {{
    const float* data = nullptr;
    unsigned long long size = 0;
    switch (blockIdx.y) {{ {cases} }}
    unsigned int failed = 0;
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += (unsigned long long)gridDim.x * blockDim.x) {{
        float value = data[i];
        failed |= (isfinite(value) ? 0u : 1u);
    }}
    unsigned int any_failed = __ballot_sync(0xffffffffu, failed != 0);
    if ((threadIdx.x & 31) == 0 && any_failed)
        atomicOr(flags + blockIdx.y, 1);
}}
"""
    return cp.RawKernel(source, "ruc_validate_finite",
                        options=("-std=c++17", "--ftz=true"))


def _validation_scan_blocks(longest: int, count: int) -> int:
    """Blocks along x, enough to cover the card without oversubscribing it.

    The scan is memory-bound and grid-stride, so the useful width is the
    card's resident thread count divided across the arrays in the batch, and
    never more blocks than the longest array has 128-wide tiles.
    """
    tiles = max(1, (longest + 127) // 128)
    device = cp.cuda.Device(cp.cuda.runtime.getDevice())
    resident = (device.attributes["MultiProcessorCount"]
                * device.attributes["MaxThreadsPerMultiProcessor"])
    return max(1, min(tiles, max(1, resident // (128 * max(1, count)))))


def _ruc_validate_batch(arrays, flags) -> None:
    """Scan ``arrays`` for non-finite values, one flag word per array.

    This is what :class:`~gpuwm.core.ruc_validation.RucValidationBatch` finds
    on :data:`RUC_DEVICE_ARRAYS` and calls instead of one reduction per
    array; the batch reads ``flags`` once for the whole call.
    """
    group = tuple(arrays)
    if not group or len(group) > VALIDATION_SCAN_GROUP:
        raise ValueError(
            f"a RUC validation scan takes 1..{VALIDATION_SCAN_GROUP} arrays, "
            f"got {len(group)}; RucValidationBatch chunks a longer batch")
    launch = tuple(value for array in group
                   for value in (array, np.uint64(array.size)))
    blocks = _validation_scan_blocks(max(array.size for array in group),
                                     len(group))
    _validation_scan_kernel(len(group))(
        (blocks, len(group)), (128,), (*launch, flags))


#: The four names :class:`RucValidationBatch` reaches on a device batch.
#:
#: A leaf's admission tests run on cupy whatever namespace its CALLER uses,
#: because the leaf is the device implementation; this is that namespace,
#: kept to the names the batch needs so a fifth one is an AttributeError at
#: the batch rather than a silent host fallback.
_VALIDATION_ARRAYS = SimpleNamespace(
    zeros=cp.zeros, all=cp.all, any=cp.any, isfinite=cp.isfinite,
    ruc_validate_batch=_ruc_validate_batch,
)


def _validation_batch() -> RucValidationBatch:
    """A batch whose verdicts cost one host read for the whole call."""
    return RucValidationBatch(_VALIDATION_ARRAYS)


def _float_field(value, shape: tuple[int, ...], name: str, *,
                 batch: RucValidationBatch | None = None) -> cp.ndarray:
    raw = cp.asarray(value, dtype=DTYPE)
    if raw.shape != shape:
        try:
            raw = cp.broadcast_to(raw, shape)
        except ValueError as exc:
            if batch is not None:
                batch.flush()
            raise ValueError(
                f"{name} shape {raw.shape} is not broadcastable to {shape}"
            ) from exc
    field = cp.ascontiguousarray(raw)
    if batch is None:
        if not bool(cp.all(cp.isfinite(field))):
            raise ValueError(f"{name} must be finite")
        return field
    return batch.finite(field, name)


def _float_profile(value, shape: tuple[int, ...], name: str, *,
                   batch: RucValidationBatch | None = None) -> cp.ndarray:
    raw = cp.asarray(value, dtype=DTYPE)
    if raw.shape != shape:
        if batch is not None:
            batch.flush()
        raise ValueError(f"{name} shape {raw.shape}; expected {shape}")
    field = cp.ascontiguousarray(raw)
    if batch is None:
        if not bool(cp.all(cp.isfinite(field))):
            raise ValueError(f"{name} must be finite")
        return field
    return batch.finite(field, name)


def _root_count_field(value, shape: tuple[int, ...], *,
                      nzs: int = NUM_SOIL_LAYERS,
                      batch: RucValidationBatch | None = None) -> cp.ndarray:
    raw = cp.asarray(value)
    if raw.dtype.kind not in "iu":
        if batch is not None:
            batch.flush()
        raise TypeError("nroot must contain integer root-zone level counts")
    if raw.shape != shape:
        try:
            raw = cp.broadcast_to(raw, shape)
        except ValueError as exc:
            if batch is not None:
                batch.flush()
            raise ValueError(
                f"nroot shape {raw.shape} is not broadcastable to {shape}"
            ) from exc
    roots = cp.ascontiguousarray(raw, dtype=cp.int32)
    invalid = (roots < 1) | (roots >= nzs)
    if batch is not None:
        # The message still READS the offending count, and still only on the
        # failing path: the batch calls it after its own single host read.
        batch.refuse_if_any(
            invalid,
            lambda: f"RUC nroot {int(roots[invalid][0])} is outside 1..{nzs - 1}")
        return roots
    if bool(cp.any(invalid)):
        bad = int(roots[invalid][0])
        # The BOUND was un-pinned with the geometry; this MESSAGE was not,
        # and at six levels it said "RUC nroot 6 is outside 1..8" -- naming
        # the rejected value as inside the range that rejected it, and
        # disagreeing with the host lane's own wording for the identical
        # refusal (gpuwm/core/ruc.py:_root_count_field).  A refusal a user
        # cannot act on is worse than no refusal, so it reads from nzs.
        raise ValueError(f"RUC nroot {bad} is outside 1..{nzs - 1}")
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
    nzs = _resolved_soil_levels(soilmois, "RUC phase-partition profiles")
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
    kernel = _ruc_kernel("ruc_soil_phase_partition", nzs)
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


def _device_constant_flux_depth(conflx, ncolumn: int, label: str, *,
                                batch: RucValidationBatch | None = None):
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
    message = f"RUC CUDA {label} conflx must be finite and nonnegative"
    if batch is not None:
        batch.finite_message(depth, message)
        batch.refuse_if_any(depth < cp.float32(0.0), message)
    elif not bool(cp.all(cp.isfinite(depth))) or bool(
            cp.any(depth < cp.float32(0.0))):
        raise ValueError(message)
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
    landusef=None,
    soilctop=None,
    parameters: RucParameterBundle | None = None,
) -> RucSurfaceParametersCuda:
    """Evaluate WRF ``soilvegin`` directly on independent GPU columns."""

    from gpuwm.core.ruc_mosaic import mosaic_option, mosaic_fractions
    mosaic_option(mosaic_lu, "mosaic_lu")
    mosaic_option(mosaic_soil, "mosaic_soil")
    if type(rdlai2d) is not bool:
        raise TypeError("rdlai2d must be bool")

    soil_raw = cp.asarray(isltyp)
    if soil_raw.ndim < 1:
        raise ValueError("RUC CUDA surface fields must have at least one dimension")
    shape = soil_raw.shape
    soil_type = _integer_field(soil_raw, shape, "isltyp")
    vegetation_type = _integer_field(ivgtyp, shape, "ivgtyp")
    batch = _validation_batch()
    inputs = tuple(
        _float_field(value, shape, name, batch=batch)
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
        tables, nvegetation, nsoil, default_water = _bundle_device_tables(
            parameters, mminlu
        )
    # The bound is tested on the card and the offending CATEGORY is read
    # only when one is out of range, so the common path costs no read of its
    # own and the refusal still names the value it rejected.
    def _out_of_range(field, ceiling, name, suffix=""):
        def message():
            low = int(cp.min(field))
            bad = low if low < 1 else int(cp.max(field))
            return f"RUC {name} {bad} is outside 1..{ceiling}{suffix}"
        return message

    batch.refuse_if_any((soil_type < 1) | (soil_type > nsoil),
                        _out_of_range(soil_type, nsoil, "isltyp"))
    batch.refuse_if_any(
        (vegetation_type < 1) | (vegetation_type > nvegetation),
        _out_of_range(vegetation_type, nvegetation, "ivgtyp",
                      f" for {mminlu}"))
    batch.flush()
    if iswater is None:
        water_category = default_water
    elif type(iswater) is int and 1 <= iswater <= nvegetation:
        water_category = iswater
    else:
        raise ValueError(f"RUC iswater {iswater!r} is outside 1..{nvegetation}")

    land_fractions = (mosaic_fractions(landusef, shape, "landusef", nvegetation, arrays=cp)
                      if mosaic_lu else cp.empty((0,), dtype=cp.float32))
    soil_fractions = (mosaic_fractions(soilctop, shape, "soilctop", nsoil, arrays=cp)
                      if mosaic_soil else cp.empty((0,), dtype=cp.float32))

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
            land_fractions, soil_fractions,
            np.int32(land_fractions.shape[0]), np.int32(soil_fractions.shape[0]),
            np.int32(mosaic_lu), np.int32(mosaic_soil),
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
    spp_lsm: int = 0,
    rstochcol=None,
    fieldcol_sf=None,
    soilprop: str = "wrf_461",
) -> RucSoilPropertiesCuda:
    """Evaluate deterministic WRF ``soilprop`` on nine-level GPU columns.

    ``soilprop`` names the WRF lineage (``gpuwm.core.ruc_tier``
    ``RUC_SOILPROP_FORMS``); it selects the translation unit's define.
    """

    from gpuwm.core.ruc_spp import validate_spp_mode, hydraulic_spp_device
    enabled_spp = validate_spp_mode(spp_lsm)
    ice_water_ratio = np.float32(riw)
    if not np.isfinite(ice_water_ratio) or ice_water_ratio <= np.float32(0.0):
        raise ValueError("RUC CUDA soilprop riw must be finite and positive")
    required = RUC_SOIL_PROPERTY_PROFILE_INPUTS + RUC_SOIL_PROPERTY_COLUMN_INPUTS
    missing = [name for name in required if name not in values]
    if missing:
        raise TypeError(f"missing RUC soil-property inputs: {', '.join(missing)}")
    first = cp.asarray(values[RUC_SOIL_PROPERTY_PROFILE_INPUTS[0]])
    nzs = _resolved_soil_levels(first, "RUC CUDA soil-property profiles")
    shape = first.shape
    batch = _validation_batch()
    profiles = {
        name: _float_profile(values[name], shape, name, batch=batch)
        for name in RUC_SOIL_PROPERTY_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SOIL_PROPERTY_COLUMN_INPUTS
    }
    batch.refuse_if_any(columns["bclh"] <= cp.float32(0.0),
                        "RUC bclh must be positive")
    batch.refuse_if_any(columns["psis"] >= cp.float32(0.0),
                        "RUC psis must be negative")
    batch.refuse_if_any(columns["ksat"] < cp.float32(0.0),
                        "RUC ksat must be nonnegative")
    batch.flush()

    outputs = {
        name: cp.empty(shape, dtype=DTYPE)
        for name in ("thdif", "diffu", "hydro", "cap")
    }
    ncolumn = int(np.prod(horizontal_shape))
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = _ruc_kernel("ruc_soil_properties", nzs, soilprop)
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
    if enabled_spp:
        hydraulic_spp_device(outputs["hydro"], rstochcol, fieldcol_sf)
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
    """Evaluate WRF ``transf`` on per-column ``1..nzs-1``-level GPU root zones.

    The root-zone ceiling follows the resolved soil geometry -- 1..8 at nine
    levels, 1..5 at six -- rather than the nine-level literal this docstring
    carried while nine was the only geometry.
    """
    liquid = cp.asarray(soiliqw, dtype=DTYPE)
    nzs = _resolved_soil_levels(liquid, "RUC CUDA soiliqw")
    batch = _validation_batch()
    liquid = cp.ascontiguousarray(liquid)
    batch.finite_message(liquid, "soiliqw must be finite")
    horizontal_shape = liquid.shape[1:]
    roots = _root_count_field(nroot, horizontal_shape, nzs=nzs, batch=batch)
    columns = {
        name: _float_field(value, horizontal_shape, name, batch=batch)
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
        tables, nvegetation, _, _ = _bundle_device_tables(parameters, mminlu)
    def _bad_land():
        low = int(cp.min(land_type))
        bad = low if low < 1 else int(cp.max(land_type))
        return f"RUC iland {bad} is outside 1..{nvegetation} for {mminlu}"

    batch.refuse_if_any((land_type < 1) | (land_type > nvegetation), _bad_land)
    batch.refuse_if_any(columns["ref"] <= columns["wilt"],
                        "RUC ref must exceed wilt")
    batch.flush()

    zshalf = _device_soil_half_levels(int(cp.cuda.runtime.getDevice()), nzs)
    weights = cp.empty_like(liquid)
    totals = cp.empty(horizontal_shape, dtype=DTYPE)
    ncolumn = int(np.prod(horizontal_shape))
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = _ruc_kernel("ruc_transpiration", nzs)
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
    nzs = _resolved_soil_levels(first, "RUC CUDA soilmoist profiles")
    shape = first.shape
    batch = _validation_batch()
    profiles = {
        name: _float_profile(values[name], shape, name, batch=batch)
        for name in RUC_SOIL_MOISTURE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SOIL_MOISTURE_COLUMN_INPUTS
    }
    batch.refuse_if_any(columns["dqm"] <= cp.float32(0.0),
                        "RUC CUDA soilmoist dqm must be positive")
    batch.refuse_if_any(columns["ref"] <= columns["qmin"],
                        "RUC CUDA soilmoist ref must exceed qmin")
    batch.refuse_if_any(columns["ksat"] < cp.float32(0.0),
                        "RUC CUDA soilmoist ksat must be nonnegative")
    batch.flush()

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
    kernel = _ruc_kernel("ruc_soil_moisture_step", nzs)
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
    # WRF's `soil` binds this dummy to its own `cw`, the VOLUMETRIC heat
    # capacity of water: module_sf_ruclsm.F:731 `cw =4.183e6` (assigned
    # nowhere else in the file), :2435 `cvw=cw`, :2662 the `call soiltemp`
    # constants group, :4634 the dummy.  `rainf*cvw*prcpms` at :4743/:4752
    # needs J m-3 K-1 against `prcpms` in m s-1 to come out in W m-2, so the
    # mass-specific 4183.0 is a factor of 1000 short.
    cvw: float = 4.183e6,
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
    nzs = _resolved_soil_levels(first, "RUC CUDA soiltemp profiles")
    shape = first.shape
    batch = _validation_batch()
    profiles = {
        name: _float_profile(values[name], shape, name, batch=batch)
        for name in RUC_SOIL_TEMPERATURE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape, nzs=nzs, batch=batch)
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SOIL_TEMPERATURE_COLUMN_INPUTS
    }
    batch.refuse_if_any(
        profiles["thdif"][0] <= cp.float32(0.0),
        "RUC CUDA soiltemp top-level thdif must be positive")
    batch.refuse_if_any(
        profiles["cap"][0] <= cp.float32(0.0),
        "RUC CUDA soiltemp top-level cap must be positive")
    batch.refuse_if_any(columns["patm"] <= cp.float32(0.0),
                        "RUC CUDA soiltemp patm must be positive")
    batch.refuse_if_any(columns["rho"] <= cp.float32(0.0),
                        "RUC CUDA soiltemp rho must be positive")
    batch.refuse_if_any(
        (columns["mavail"] < cp.float32(0.0))
        | (columns["mavail"] > cp.float32(1.0)),
        "RUC CUDA soiltemp mavail must be within 0..1")

    tso = cp.empty(shape, dtype=DTYPE)
    outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in ("soilt", "qvg", "qsg", "qcg", "storage")
    }
    device_id = int(cp.cuda.runtime.getDevice())
    tbq = _device_tbq(device_id)
    ncolumn = int(np.prod(horizontal_shape))
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "soiltemp", batch=batch)
    batch.flush()
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = _ruc_kernel("ruc_soil_temperature_step", nzs)
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
    spp_lsm: int = 0,
    rstochcol=None,
    fieldcol_sf=None,
    soilprop: str = "wrf_461",
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
    nzs = _resolved_soil_levels(first, "RUC CUDA soil profiles")
    shape = first.shape
    batch = _validation_batch()
    profiles = {
        name: _float_profile(values[name], shape, name, batch=batch)
        for name in RUC_SOIL_STEP_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape, nzs=nzs, batch=batch)
    land_type = _integer_field(iland, horizontal_shape, "iland")
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SOIL_STEP_COLUMN_INPUTS
    }
    batch.refuse_if_any(columns["dqm"] <= cp.float32(0.0),
                        "RUC CUDA soil dqm must be positive")
    batch.refuse_if_any(columns["psis"] >= cp.float32(0.0),
                        "RUC CUDA soil psis must be negative")
    batch.refuse_if_any(columns["bclh"] <= cp.float32(0.0),
                        "RUC CUDA soil bclh must be positive")
    batch.refuse_if_any(columns["sat"] <= cp.float32(0.0),
                        "RUC CUDA soil canopy saturation must be positive")
    batch.refuse_if_any(
        (columns["rho"] <= cp.float32(0.0))
        | (columns["patm"] <= cp.float32(0.0)),
        "RUC CUDA soil rho and patm must be positive")
    batch.refuse_if_any(
        (columns["mavail"] < cp.float32(0.0))
        | (columns["mavail"] > cp.float32(1.0)),
        "RUC CUDA soil mavail must be within 0..1")
    batch.flush()

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
        spp_lsm=spp_lsm, rstochcol=rstochcol, fieldcol_sf=fieldcol_sf,
        soilprop=soilprop,
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
    prepare_kernel = _ruc_kernel("ruc_soil_prepare_moisture", nzs)
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
    finalize_kernel = _ruc_kernel("ruc_soil_finalize", nzs)
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
    outcome = _validation_batch()
    for name in RucSoilStepCuda.__dataclass_fields__:
        outcome.finite_message(
            getattr(result, name),
            f"RUC CUDA soil produced non-finite {name}")
    outcome.flush()
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
    batch = _validation_batch()
    batch.finite_message(cp.ascontiguousarray(values),
                         "RUC qsn temperatures must be finite")
    batch.flush()
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
    nzs = _resolved_soil_levels(first, "RUC CUDA sice profiles")
    shape = first.shape
    batch = _validation_batch()
    profiles = {
        name: _float_profile(values[name], shape, name, batch=batch)
        for name in RUC_SEA_ICE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SEA_ICE_COLUMN_INPUTS
    }
    batch.refuse_if_any(
        profiles["thdifice"][0] <= cp.float32(0.0),
        "RUC CUDA sice top-level thdifice must be positive")
    batch.refuse_if_any(
        profiles["capice"][0] <= cp.float32(0.0),
        "RUC CUDA sice top-level capice must be positive")
    batch.refuse_if_any(columns["patm"] <= cp.float32(0.0),
                        "RUC CUDA sice patm must be positive")
    batch.refuse_if_any(columns["rho"] <= cp.float32(0.0),
                        "RUC CUDA sice rho must be positive")

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
        raw_flux_depth, ncolumn, "sice", batch=batch)
    batch.flush()
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = _ruc_kernel("ruc_sea_ice_step", nzs)
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
    outcome = _validation_batch()
    for name in ("tso", *scalar_names):
        outcome.finite_message(
            getattr(result, name),
            f"RUC CUDA sice produced non-finite {name}")
    outcome.flush()
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


@cuda_cache(maxsize=None, ready=True)
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
        vegetation = _default_parameter_bundle().vegetation_for(mminlu)
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


_SNOW_PREPARATION_BUNDLE_TABLES = {}


def _snow_preparation_tables_for(parameters, mminlu):
    """Select the supplied bundle's snow-preparation columns.

    Bundles with default ``z0``/``lemi``/``URBAN`` values share the existing
    default upload. Other bundles are cached by their consumed values and
    device, with upload readiness preserved across streams.
    """
    device_id = int(cp.cuda.runtime.getDevice())
    if parameters is None:
        return _snow_preparation_tables(device_id, mminlu)
    supplied = parameters.vegetation_for(mminlu)
    default = _default_parameter_bundle().vegetation_for(mminlu)
    z0 = tuple(row.z0 for row in supplied.rows)
    lemi = tuple(row.lemi for row in supplied.rows)
    urban = int(supplied.scalars["URBAN"])
    if (z0 == tuple(row.z0 for row in default.rows)
            and lemi == tuple(row.lemi for row in default.rows)
            and urban == int(default.scalars["URBAN"])):
        return _snow_preparation_tables(device_id, mminlu)
    key = (device_id, mminlu, z0, lemi, urban)

    def upload():
        with cp.cuda.Device(device_id):
            return (cp.asarray(z0, dtype=DTYPE),
                    cp.asarray(lemi, dtype=DTYPE), urban, len(supplied.rows))

    return cached_ready(cp, _SNOW_PREPARATION_BUNDLE_TABLES, key, upload)


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
    snow: str = "wrf_461",
    parameters=None,
) -> RucSnowPreparationCuda:
    """Run WRF ``sfctmp``'s snow-preparation block on the GPU.

    ``snow`` names the lineage (``gpuwm.core.ruc_tier.RUC_SNOW_FORMS``), as
    for :func:`gpuwm.core.ruc.ruc_snow_preparation`.

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
    nzs = _resolved_soil_levels(profile, "RUC CUDA snow preparation ts1d")
    shape = profile.shape
    horizontal_shape = shape[1:]
    batch = _validation_batch()
    ts1d = _float_profile(values["ts1d"], shape, "ts1d", batch=batch)
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SNOW_PREP_COLUMN_INPUTS
    }
    vegetation_category = _integer_field(
        cp.asarray(ivgtyp), horizontal_shape, "ivgtyp"
    )
    land_category = _integer_field(cp.asarray(iland), horizontal_shape, "iland")

    roughness, emissivity, urban, ncategory = _snow_preparation_tables_for(
        parameters, mminlu
    )
    for name, category in (
        ("ivgtyp", vegetation_category), ("iland", land_category)
    ):
        batch.refuse_if_any((category < 1) | (category > ncategory),
                            f"RUC {name} is outside 1..{ncategory}")
    if not 1 <= isice <= ncategory:
        batch.flush()
        raise ValueError(f"RUC isice is outside 1..{ncategory}")
    batch.refuse_if_any(
        columns["rhosn"] <= cp.float32(0.0),
        "RUC CUDA snow preparation rhosn must be positive")
    batch.refuse_if_any(columns["alb"] >= cp.float32(1.0),
                        "RUC CUDA snow preparation alb must be below 1")
    batch.flush()

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
    kernel = _ruc_kernel("ruc_snow_preparation", nzs, snow=snow)
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
    outcome = _validation_batch()
    for name in RUC_SNOW_PREP_PROFILE_OUTPUTS + RUC_SNOW_PREP_COLUMN_OUTPUTS:
        outcome.finite_message(
            getattr(result, name),
            f"RUC CUDA snow preparation produced non-finite {name}")
    outcome.flush()
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
    nzs = _resolved_soil_levels(first, "RUC CUDA snowseaice profiles")
    shape = first.shape
    batch = _validation_batch()
    profiles = {
        name: _float_profile(values[name], shape, name, batch=batch)
        for name in RUC_SNOW_SEA_ICE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SNOW_SEA_ICE_COLUMN_INPUTS
    }
    integers = {
        name: _integer_field(values[name], horizontal_shape, name)
        for name in RUC_SNOW_SEA_ICE_INTEGER_INPUTS
    }
    batch.refuse_if_any(
        profiles["thdifice"][0] <= cp.float32(0.0),
        "RUC CUDA snowseaice top-level thdifice must be positive")
    batch.refuse_if_any(
        profiles["capice"][0] <= cp.float32(0.0),
        "RUC CUDA snowseaice top-level capice must be positive")
    batch.refuse_if_any(columns["patm"] <= cp.float32(0.0),
                        "RUC CUDA snowseaice patm must be positive")
    batch.refuse_if_any(columns["rho"] <= cp.float32(0.0),
                        "RUC CUDA snowseaice rho must be positive")
    batch.refuse_if_any(columns["rhosn"] <= cp.float32(0.0),
                        "RUC CUDA snowseaice rhosn must be positive")
    batch.refuse_if_any(columns["snwe"] < cp.float32(0.0),
                        "RUC CUDA snowseaice snwe must be nonnegative")

    tso = cp.empty(shape, dtype=DTYPE)
    layer_count = cp.empty(horizontal_shape, dtype=cp.int32)
    outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS
    }
    tbq = _device_tbq(int(cp.cuda.runtime.getDevice()))
    ncolumn = int(np.prod(horizontal_shape))
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "snowseaice", batch=batch)
    batch.flush()
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    kernel = _ruc_kernel("ruc_snow_sea_ice_step", nzs)
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
    outcome = _validation_batch()
    for name in ("tso", *RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS):
        outcome.finite_message(
            getattr(result, name),
            f"RUC CUDA snowseaice produced non-finite {name}")
    outcome.flush()
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
    snow: str = "wrf_461",
) -> RucSnowTemperatureCuda:
    """Run the complete WRF RUC ``snowtemp`` snow column on the GPU.

    ``snow`` names the lineage, as for
    :func:`gpuwm.core.ruc.ruc_snow_temperature_step`.

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
    nzs = _resolved_soil_levels(first, "RUC CUDA snowtemp profiles")
    shape = first.shape
    batch = _validation_batch()
    profiles = {
        name: _float_profile(values[name], shape, name, batch=batch)
        for name in RUC_SNOW_TEMPERATURE_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    }
    roots = _root_count_field(nroot, horizontal_shape, nzs=nzs, batch=batch)
    raw_layers = cp.asarray(ilnb)
    if raw_layers.dtype.kind not in "iu":
        batch.flush()
        raise TypeError("RUC CUDA snowtemp ilnb must contain integer counts")
    try:
        layers = cp.ascontiguousarray(
            cp.broadcast_to(raw_layers, horizontal_shape), dtype=cp.int32
        )
    except ValueError as exc:
        batch.flush()
        raise ValueError(
            f"ilnb shape {raw_layers.shape} is not broadcastable to "
            f"{horizontal_shape}"
        ) from exc
    batch.refuse_if_any(
        profiles["thdif"][0] <= cp.float32(0.0),
        "RUC CUDA snowtemp top-level thdif must be positive")
    batch.refuse_if_any(
        profiles["cap"][0] <= cp.float32(0.0),
        "RUC CUDA snowtemp top-level cap must be positive")
    for name in ("patm", "rho", "rhosn", "snhei", "snth", "deltsn"):
        batch.refuse_if_any(columns[name] <= cp.float32(0.0),
                            f"RUC CUDA snowtemp {name} must be positive")

    tso = cp.empty(shape, dtype=DTYPE)
    outputs = {
        name: cp.empty(horizontal_shape, dtype=DTYPE)
        for name in _RUC_SNOW_TEMPERATURE_SCALAR_OUTPUTS
    }
    layer_out = cp.empty(horizontal_shape, dtype=cp.int32)
    tbq = _device_tbq(int(cp.cuda.runtime.getDevice()))
    ncolumn = int(np.prod(horizontal_shape))
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "snowtemp", batch=batch)
    batch.flush()
    threads = 128
    blocks = (ncolumn + threads - 1) // threads
    # ONE OF TWO launch sites for this symbol; the other is in
    # ruc_snow_soil_step_cuda.  Both are tiered or neither may be: an
    # untiered load at nzs=6 returns the nine-level module, whose kernel
    # reads past every scratch array in the frame.
    kernel = _ruc_kernel("ruc_snow_temperature_step", nzs, snow=snow)
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
    outcome = _validation_batch()
    for name in ("tso", *_RUC_SNOW_TEMPERATURE_SCALAR_OUTPUTS):
        outcome.finite_message(
            getattr(result, name),
            f"RUC CUDA snowtemp produced non-finite {name}")
    outcome.flush()
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
    spp_lsm: int = 0,
    rstochcol=None,
    fieldcol_sf=None,
    soilprop: str = "wrf_461",
    snow: str = "wrf_461",
) -> RucSnowSoilStepCuda:
    """Run the complete snow-covered WRF RUC land column on the GPU."""

    # ``snow`` is the lineage name; the body reuses the word for its outputs.
    snow_form = snow
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
    nzs = _resolved_soil_levels(first, "RUC CUDA snowsoil profiles")
    shape = first.shape
    batch = _validation_batch()
    profiles = {
        name: _float_profile(values[name], shape, name, batch=batch)
        for name in RUC_SNOW_SOIL_PROFILE_INPUTS
    }
    horizontal_shape = shape[1:]
    roots = _root_count_field(nroot, horizontal_shape, nzs=nzs, batch=batch)
    land_type = _integer_field(iland, horizontal_shape, "iland")
    snow_layers = cp.asarray(ilnb)
    if snow_layers.dtype.kind not in "iu":
        batch.flush()
        raise TypeError("RUC CUDA snowsoil ilnb must be an integer layer count")
    if snow_layers.shape != horizontal_shape:
        try:
            snow_layers = cp.broadcast_to(snow_layers, horizontal_shape)
        except ValueError as exc:
            batch.flush()
            raise ValueError(
                f"ilnb shape {snow_layers.shape} is not broadcastable to "
                f"{horizontal_shape}"
            ) from exc
    snow_layers = cp.ascontiguousarray(snow_layers, dtype=cp.int32)
    columns = {
        name: _float_field(values[name], horizontal_shape, name, batch=batch)
        for name in RUC_SNOW_SOIL_COLUMN_INPUTS
    }
    batch.refuse_if_any(columns["dqm"] <= cp.float32(0.0),
                        "RUC CUDA snowsoil dqm must be positive")
    batch.refuse_if_any(columns["psis"] >= cp.float32(0.0),
                        "RUC CUDA snowsoil psis must be negative")
    batch.refuse_if_any(columns["bclh"] <= cp.float32(0.0),
                        "RUC CUDA snowsoil bclh must be positive")
    batch.refuse_if_any(
        columns["sat"] <= cp.float32(0.0),
        "RUC CUDA snowsoil canopy saturation must be positive")
    batch.refuse_if_any(
        (columns["rho"] <= cp.float32(0.0))
        | (columns["patm"] <= cp.float32(0.0)),
        "RUC CUDA snowsoil rho and patm must be positive")
    batch.refuse_if_any(
        (columns["rhosn"] <= cp.float32(0.0))
        | (columns["rhonewsn"] <= cp.float32(0.0)),
        "RUC CUDA snowsoil snow densities must be positive")
    batch.refuse_if_any(
        (columns["snwe"] < cp.float32(0.0))
        | (columns["snhei"] < cp.float32(0.0)),
        "RUC CUDA snowsoil snow depth must be nonnegative")
    batch.refuse_if_any(
        (columns["snowfrac"] < cp.float32(0.0))
        | (columns["snowfrac"] > cp.float32(1.0)),
        "RUC CUDA snowsoil snowfrac must be within 0..1")

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
        spp_lsm=spp_lsm, rstochcol=rstochcol, fieldcol_sf=fieldcol_sf,
        soilprop=soilprop,
    )

    ncolumn = int(np.prod(horizontal_shape))
    # snowsoil LAUNCHES ruc_snow_temperature_step itself rather than going
    # through ruc_snow_temperature_step_cuda, so it has to bind the device
    # conflx here too.  Missing this passed a host object straight into a
    # kernel parameter that is now a pointer.
    constant_flux_depth = _device_constant_flux_depth(
        raw_flux_depth, ncolumn, "snowsoil", batch=batch)
    batch.flush()
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
    # ONE OF TWO launch sites for this symbol; the other is in
    # ruc_snow_temperature_step_cuda.  See the note there.
    _ruc_kernel("ruc_snow_temperature_step", nzs, snow=snow_form)(
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
    _ruc_kernel("ruc_snow_soil_prepare_moisture", nzs)(
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
    _ruc_kernel("ruc_snow_soil_finalize", nzs)(
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
    outcome = _validation_batch()
    for name in RucSnowSoilStepCuda.__dataclass_fields__:
        array = getattr(result, name)
        if array.dtype == DTYPE:
            outcome.finite_message(
                array, f"RUC CUDA snowsoil produced non-finite {name}")
    outcome.flush()
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

    The supplied bundle's ``z0tbl``, ``lemitbl`` and ``URBAN`` reach the
    kernel. Default-valued columns share the cached default upload.
    """

    fields = tuple(RucSnowPreparation.__dataclass_fields__)

    def call(values, *, delt, ivgtyp, iland, isice=15, c1sn=0.026,
             c2sn=21.0, isncovr_opt=RUC_SNOW_COVER_OPTION,
             mminlu="MODIFIED_IGBP_MODIS_NOAH", bundle=None, snow="wrf_461"):
        result = ruc_snow_preparation_cuda(
            values, delt=delt, ivgtyp=ivgtyp, iland=iland, isice=isice,
            c1sn=c1sn, c2sn=c2sn, isncovr_opt=isncovr_opt, mminlu=mminlu,
            parameters=bundle, snow=snow)
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


_RUC_CONSTANT_CACHE = {}
#: One upload per key at a time.  Unlocked, the slabs of a split domain
#: (each on its own stream) each uploaded a missing constant and the last
#: replaced the others, so a slab holding a replaced copy read memory its
#: stream's pool had already handed back -- the defect measured on RRTMGP's
#: tables (rrtmgp._upload_once).  An evicted array may also still be read
#: on another stream, so eviction waits for the card first.
_RUC_CONSTANT_LOCK = threading.RLock()


def _constant_array(values, *, dtype):
    """Cache lookup uploads by exact contents, retaining at most 32 arrays."""
    dtype = np.dtype(dtype)
    if isinstance(values, cp.ndarray):
        return cp.asarray(values, dtype=dtype)
    host = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    key = (int(cp.cuda.runtime.getDevice()), host.dtype.str,
           host.shape, host.tobytes())
    stream = cp.cuda.get_current_stream()
    with _RUC_CONSTANT_LOCK:
        cached = _RUC_CONSTANT_CACHE.get(key)
        if cached is not None:
            if stream.ptr != cached[2]:
                stream.wait_event(cached[1])
            return cached[0]
        array = cp.asarray(host)
        ready = cp.cuda.Event(disable_timing=True)
        ready.record()
        if len(_RUC_CONSTANT_CACHE) >= 32:
            oldest = next(iter(_RUC_CONSTANT_CACHE))
            # The cache spans cards. Its oldest array can still be read on
            # another card, whose streams the current card cannot wait for.
            with cp.cuda.Device(oldest[0]):
                cp.cuda.Device().synchronize()
            _RUC_CONSTANT_CACHE.pop(oldest)
        _RUC_CONSTANT_CACHE[key] = (array, ready, stream.ptr)
        return array


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
    sum=cp.sum,
    arange=_dtype_normalising(cp.arange),
    array=_dtype_normalising(cp.array),
    asarray=_dtype_normalising(cp.asarray),
    ascontiguousarray=_dtype_normalising(cp.ascontiguousarray),
    shares_memory=cp.shares_memory,
    atleast_1d=cp.atleast_1d,
    broadcast_to=cp.broadcast_to,
    count_nonzero=cp.count_nonzero,
    empty=_dtype_normalising(cp.empty),
    full=_dtype_normalising(cp.full),
    isfinite=cp.isfinite,
    maximum=cp.maximum,
    minimum=cp.minimum,
    nonzero=cp.nonzero,
    stack=cp.stack,
    where=cp.where,
    zeros=_dtype_normalising(cp.zeros),
    ruc_tanhf_glibc=ruc_tanhf_glibc,
    ruc_validate_batch=_ruc_validate_batch,
    ruc_constant_array=_constant_array,
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
    """``ruc_snow_preparation_cuda`` returning device arrays.

    The supplied bundle's ``z0tbl``/``lemitbl``/``URBAN`` reach the kernel.
    """

    def call(values, *, delt, ivgtyp, iland, isice=15, c1sn=0.026,
             c2sn=21.0, isncovr_opt=RUC_SNOW_COVER_OPTION,
             mminlu="MODIFIED_IGBP_MODIS_NOAH", bundle=None, snow="wrf_461"):
        return ruc_snow_preparation_cuda(
            values, delt=delt, ivgtyp=ivgtyp, iland=iland, isice=isice,
            c1sn=c1sn, c2sn=c2sn, isncovr_opt=isncovr_opt, mminlu=mminlu,
            parameters=bundle, snow=snow)

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


# ---------------------------------------------------------------------------
# The full-width sfctmp contract the fused RUC path is built against.
# ---------------------------------------------------------------------------

def ruc_sfctmp_full_width_reference(
    values: Mapping[str, object],
    *,
    run,
    delt: float,
    conflx,
    ivgtyp,
    iland,
    nroot,
    ilnb,
    isice: int,
    c1sn: float,
    c2sn: float,
    isncovr_opt: int,
    mminlu: str,
    parameters: RucParameterBundle | None,
    soilprop: str = "wrf_461",
    snow: str = "wrf_461",
) -> dict[str, cp.ndarray]:
    """``sfctmp`` on the columns ``run`` selects, returned at FULL width.

    ``values`` holds full-width device arrays under the names
    :func:`gpuwm.core.ruc.ruc_land_surface_step` hands its ``sfctmp``
    dispatch: ``(nzs, n)`` soil profiles and ``(n,)`` columns.  ``conflx``,
    ``ivgtyp``, ``iland``, ``nroot`` and ``ilnb`` are ``(n,)`` device arrays
    and ``run`` is an ``(n,)`` boolean device mask.

    This is the reference: it gathers the ``run`` columns, calls the resident
    device ``sfctmp`` exactly as the driver does, and scatters every returned
    field into a zero-filled full-width array.  The column is column-local,
    so a gather changes no bit; a fused implementation of the same contract
    is graded against this one on the ``run`` columns only.
    """

    from gpuwm.core.ruc import (RucSurfaceTemperatureStep,
                                ruc_surface_temperature_step)

    index = cp.nonzero(run)[0]
    ncolumn = int(run.shape[0])
    take = {
        name: (array[:, index] if array.ndim == 2 else array[index])
        for name, array in values.items()
    }
    step = ruc_surface_temperature_step(
        take, delt=float(delt), conflx=conflx[index],
        ivgtyp=ivgtyp[index], iland=iland[index], nroot=nroot[index],
        ilnb=ilnb[index], isice=int(isice), c1sn=c1sn, c2sn=c2sn,
        myj=False, isncovr_opt=int(isncovr_opt), mminlu=mminlu,
        parameters=parameters, leaves=RUC_SFCTMP_DEVICE_LEAVES_RESIDENT,
        stages=RUC_SFCTMP_DEVICE_STAGES_RESIDENT, arrays=RUC_DEVICE_ARRAYS,
        soilprop=soilprop, snow=snow)
    out: dict[str, cp.ndarray] = {}
    for name in RucSurfaceTemperatureStep.__dataclass_fields__:
        part = cp.asarray(getattr(step, name))
        full = cp.zeros(part.shape[:-1] + (ncolumn,), dtype=part.dtype)
        full[..., index] = part
        out[name] = full
    return out


__all__ += ["ruc_sfctmp_full_width_reference"]

# The fused sfctmp layout is generated with its kernel source by
# tools/ruc_fused/gen_sfctmp.py.
from gpuwm.core.ruc_sfctmp_layout import (  # noqa: E402
    _SFCTMP_ARRAYS, _SFCTMP_CHECKS, _SFCTMP_CPU_CHECKS, _SFCTMP_OUTPUTS,
    _SFCTMP_SLOTS)

_SFCTMP_FLAG_WORDS = (len(_SFCTMP_CHECKS) + 63) // 64
RUC_SFCTMP_FLAGS_SIZE = _SFCTMP_FLAG_WORDS + len(_SFCTMP_CHECKS)
_SFCTMP_SCRATCH = {}
_SFCTMP_FLAG_CONTEXT = {}
_SFCTMP_CONTEXT_STREAMS = {}
_SFCTMP_TABLE_CACHE = {}
_SFCTMP_UPLOADS = {}


def release_ruc_stream_scratch(*, device_id, stream):
    """Retire only a completed stream's fused RUC mutable scratch.

    Independent ensemble members use separate streams. Their fused scratch
    and pending pinned uploads cannot remain resident after their models
    finish, or successive concurrent waves accumulate their full RUC working
    sets. Shared immutable device tables retain their existing cache owners.
    The caller must own the supplied stream and have selected its device.
    """
    device_id = int(device_id)
    if int(cp.cuda.runtime.getDevice()) != device_id:
        raise ValueError("RUC scratch release requires the owning CUDA device")
    stream.synchronize()
    owner = (device_id, int(stream.ptr))
    retired = {}
    for label, cache in (("scratch", _SFCTMP_SCRATCH),
                         ("tables", _SFCTMP_TABLE_CACHE),
                         ("uploads", _SFCTMP_UPLOADS)):
        # Other member streams can add keys while this one retires. Snapshot
        # the keys first, then remove only this inactive stream's entries.
        keys = [key for key in tuple(cache) if key[:2] == owner]
        for key in keys:
            del cache[key]
        retired[label] = len(keys)
    keys = [key for key, stream_id in tuple(_SFCTMP_CONTEXT_STREAMS.items())
            if key[0] == device_id and stream_id == owner[1]]
    for key in keys:
        _SFCTMP_FLAG_CONTEXT.pop(key, None)
        del _SFCTMP_CONTEXT_STREAMS[key]
    retired["flag_contexts"] = len(keys)
    return retired


def _sfctmp_tables(parameters, mminlu, nzs):
    """Device tables for the fused sfctmp, cached by the values they hold.

    The key carries the rows the uploads read, so a caller's bundle with a
    replaced row gets its own tables; object identity alone could hand a
    new bundle at a recycled address the old one's tables.
    """
    device = int(cp.cuda.runtime.getDevice())
    default = _default_parameter_bundle()
    bundle = default if parameters is None else parameters
    supplied = bundle.vegetation_for(mminlu)
    key = (device, cp.cuda.get_current_stream().ptr, mminlu, nzs,
           parameters is None, supplied.name,
           supplied.rows, float(supplied.scalars['RSMAX_DATA']),
           int(supplied.scalars['URBAN']), bundle.soil.rows)
    cached = _SFCTMP_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    table, ncategory, _, _ = (_default_device_tables(device, mminlu)
                             if parameters is None
                             else _bundle_device_tables(bundle, mminlu))
    bound = dict(rstbl=table.rstbl, rgltbl=table.rgltbl,
                 z0tbl=table.z0tbl, lemitbl=table.lemitbl,
                 tbq=_device_tbq(device),
                 zshalf=_device_soil_half_levels(device, nzs))
    cached = (bundle, bound, ncategory, int(supplied.scalars['URBAN']),
              np.float32(table.rsmax_data))
    _SFCTMP_TABLE_CACHE[key] = cached
    return cached


def _sfctmp_scratch(n, nzs):
    stream = cp.cuda.get_current_stream()
    key = (int(cp.cuda.runtime.getDevice()), stream.ptr, n, nzs)
    result = _SFCTMP_SCRATCH.get(key)
    if result is None:
        offsets = []
        units = 0
        for size in _SFCTMP_SLOTS:
            offsets.append(units * n)
            units += nzs if size == 9 else 1
        scratch = cp.empty(units * n, dtype=cp.float32)
        pointers = cp.empty(len(_SFCTMP_ARRAYS), dtype=cp.uint64)
        host_memory = cp.cuda.alloc_pinned_memory(pointers.nbytes)
        host_pointers = np.frombuffer(host_memory, dtype=np.uint64, count=len(_SFCTMP_ARRAYS))
        reset_memory = cp.cuda.alloc_pinned_memory(RUC_SFCTMP_FLAGS_SIZE * 8)
        reset = np.frombuffer(reset_memory, dtype=np.uint64, count=RUC_SFCTMP_FLAGS_SIZE)
        reset[:_SFCTMP_FLAG_WORDS] = 0
        reset[_SFCTMP_FLAG_WORDS:] = np.iinfo(np.uint64).max
        private_flags = cp.empty(RUC_SFCTMP_FLAGS_SIZE, dtype=cp.uint64)
        alive = cp.empty(n, dtype=cp.bool_)
        result = (scratch, offsets, pointers, host_memory, host_pointers,
                  reset_memory, reset, private_flags, alive)
        _SFCTMP_SCRATCH[key] = result
    return result


def ruc_sfctmp_raise_from_flags(flags):
    """Read one flag buffer and raise the first retained admission failure."""
    _sfctmp_raise_from_words(
        cp.asnumpy(flags), (int(cp.cuda.runtime.getDevice()), flags.data.ptr))


def _sfctmp_raise_from_words(words, context_key):
    """Raise the first failure recorded in host copies of the flag words.

    ``words`` is the whole ``RUC_SFCTMP_FLAGS_SIZE`` buffer read to the host;
    ``context_key`` is ``(device, device pointer)`` of the buffer the fused
    call wrote, which names the geometry and category count it ran with.
    """
    context = _SFCTMP_FLAG_CONTEXT.get(context_key,
                                      (9, 30, 'MODIFIED_IGBP_MODIS_NOAH', {}))
    nzs, ncategory, mminlu, overrides = context
    for word_index in range(_SFCTMP_FLAG_WORDS):
        word = int(words[word_index])
        if not word:
            continue
        bit = (word & -word).bit_length() - 1
        index = word_index * 64 + bit
        if index in overrides:
            exception, message = overrides[index]
            raise exception(message)
        message = _SFCTMP_CHECKS[index]
        if isinstance(message, dict):
            value = int(words[_SFCTMP_FLAG_WORDS + index])
            if message['kind'] == 'root':
                count = int(np.asarray(value & 0xffffffff, dtype=np.uint32).view(np.int32))
                message = f'RUC nroot {count} is outside 1..{nzs - 1}'
            else:
                encoded = value & 0xffffffff
                if value >> 32:
                    encoded = 0xffffffff - encoded
                category = encoded - 2147483648
                message = f'RUC iland {category} is outside 1..{ncategory} for {mminlu}'
        else:
            message = message.replace('1..30', f'1..{ncategory}')
        raise ValueError(message)


def ruc_sfctmp_full_width_fused(
    values: Mapping[str, object], *, run, delt: float, conflx, ivgtyp,
    iland, nroot, ilnb, isice: int, c1sn: float, c2sn: float,
    isncovr_opt: int, mminlu: str, parameters: RucParameterBundle | None,
    flags=None, soilprop: str = "wrf_461", snow: str = "wrf_461",
) -> dict[str, cp.ndarray]:
    """Execute full-width sfctmp in three stages with deferred device flags.

    A supplied flags buffer is uint64 with RUC_SFCTMP_FLAGS_SIZE entries.
    The function resets it asynchronously. Call ruc_sfctmp_raise_from_flags
    before publishing outputs. Every returned field owns its allocation.
    """
    from gpuwm.core.ruc import RUC_SFCTMP_PROFILE_INPUTS, RUC_SFCTMP_COLUMN_INPUTS
    from gpuwm.core.ruc_tier import ruc_fused_kernel

    errors = {}
    def refuse(label, error):
        errors[_SFCTMP_CPU_CHECKS[label]] = error

    timestep = np.float32(delt)
    if not np.isfinite(timestep) or timestep <= np.float32(0):
        refuse('delt', ValueError('RUC sfctmp delt must be finite and positive'))
    if isncovr_opt not in (1, 2, 3):
        refuse('isncovr_opt', ValueError('RUC isncovr_opt must be 1, 2, or 3'))
    elif isncovr_opt != 2:
        raise ValueError('RUC fused sfctmp supports isncovr_opt=2 only')
    missing = [name for name in RUC_SFCTMP_PROFILE_INPUTS + RUC_SFCTMP_COLUMN_INPUTS
               if name not in values]
    if missing:
        refuse('missing', TypeError(f"missing RUC sfctmp inputs: {', '.join(missing)}"))
    n = int(run.shape[0])
    run = cp.ascontiguousarray(run, dtype=cp.bool_)
    first_name = RUC_SFCTMP_PROFILE_INPUTS[0]
    first = cp.asarray(values[first_name]) if first_name in values else cp.empty((9, n), dtype=cp.float32)
    try:
        nzs = _resolved_soil_levels(first, 'RUC sfctmp profiles')
    except (ValueError, TypeError) as error:
        refuse('profile:' + first_name, error)
        nzs = 9
    profile_shape = (nzs, n)
    inputs = {}
    for name in RUC_SFCTMP_PROFILE_INPUTS:
        array = cp.asarray(values[name], dtype=cp.float32) if name in values else cp.empty(profile_shape, dtype=cp.float32)
        if array.shape != profile_shape:
            refuse('profile:' + name, ValueError(f'{name} shape {array.shape}; expected shared profile shape {profile_shape}'))
            array = cp.empty(profile_shape, dtype=cp.float32)
        inputs['value:' + name] = cp.ascontiguousarray(array)
    for name in RUC_SFCTMP_COLUMN_INPUTS:
        if name not in values:
            inputs['value:' + name] = cp.empty(n, dtype=cp.float32)
            continue
        raw = cp.asarray(values[name], dtype=cp.float32)
        try:
            inputs['value:' + name] = cp.ascontiguousarray(cp.broadcast_to(raw, (n,)))
        except ValueError:
            refuse('column:' + name, ValueError(f'{name} shape {raw.shape} is not broadcastable to {(n,)}'))
            inputs['value:' + name] = cp.empty(n, dtype=cp.float32)
    for name, raw in (('ivgtyp', ivgtyp), ('iland', iland), ('nroot', nroot), ('ilnb', ilnb)):
        raw = cp.asarray(raw)
        if raw.dtype.kind not in 'iu':
            if name == 'nroot':
                error = TypeError('nroot must contain integer root-zone level counts')
            elif name == 'ilnb':
                error = TypeError('RUC sfctmp ilnb must be an integer layer count')
            else:
                error = TypeError(f'{name} must contain integer WRF categories')
            refuse(name if name != 'iland' else 'prep_iland', error)
            raw = cp.empty(n, dtype=cp.int32)
        inputs[name] = cp.ascontiguousarray(cp.broadcast_to(raw, (n,)), dtype=cp.int32)
    depth = cp.asarray(conflx, dtype=cp.float32)
    if depth.ndim > 1:
        refuse('conflx', ValueError('RUC sfctmp conflx must be scalar or 1-D'))
        depth = cp.empty(n, dtype=cp.float32)
    inputs['conflx'] = cp.ascontiguousarray(cp.broadcast_to(depth, (n,)))
    try:
        bundle, tables, ncategory, urban, rsmax = _sfctmp_tables(parameters, mminlu, nzs)
    except ValueError as error:
        refuse('prep_bundle', error)
        bundle, tables, ncategory, urban, rsmax = _sfctmp_tables(None, mminlu, nzs)
    except KeyError as error:
        refuse('bundle', error)
        bundle, tables, ncategory, urban, rsmax = _sfctmp_tables(None, 'MODIFIED_IGBP_MODIS_NOAH', nzs)
    inputs.update(tables)
    ice = int(isice)
    if not 1 <= ice <= ncategory:
        index = _SFCTMP_CHECKS.index('RUC isice is outside 1..30')
        errors[index] = ValueError(f'RUC isice is outside 1..{ncategory}')
        ice = 1
    if not np.isfinite(np.float32(c1sn)) or not np.isfinite(np.float32(c2sn)):
        refuse('snow_density_scalars', ValueError('RUC CUDA snow preparation c1sn/c2sn must be finite'))
    scratch, offsets, pointers, host_memory, host_pointers, reset_memory, reset, private_flags, alive = _sfctmp_scratch(n, nzs)
    if flags is None:
        active_flags = private_flags
    else:
        if flags.dtype != cp.uint64 or flags.shape != (RUC_SFCTMP_FLAGS_SIZE,):
            raise ValueError(f'RUC sfctmp flags must be uint64 shape ({RUC_SFCTMP_FLAGS_SIZE},)')
        active_flags = flags
    out = {name: cp.empty(profile_shape if _SFCTMP_ARRAYS[index][2] else (n,),
                          dtype=_SFCTMP_ARRAYS[index][1])
           for name, index in _SFCTMP_OUTPUTS.items()}
    output_pointers = {index: out[name].data.ptr for name, index in _SFCTMP_OUTPUTS.items()}
    for index, (binding, dt, profile, location) in enumerate(_SFCTMP_ARRAYS):
        host_pointers[index] = (inputs[binding].data.ptr if binding else
                                output_pointers[index] if index in output_pointers else
                                scratch.data.ptr + offsets[location] * 4)
    stream = cp.cuda.get_current_stream()
    upload_key = (int(cp.cuda.runtime.getDevice()), stream.ptr)
    pending = _SFCTMP_UPLOADS.setdefault(upload_key, [])
    pending[:] = [entry for entry in pending if not cp.cuda.runtime.eventQuery(entry[0].ptr)]
    # Keep the pinned source alive and immutable until its transfer ends.
    upload_memory = cp.cuda.alloc_pinned_memory(pointers.nbytes)
    upload = np.frombuffer(upload_memory, dtype=np.uint64, count=len(_SFCTMP_ARRAYS))
    upload[:] = host_pointers
    pointers.set(upload, stream=stream)
    uploaded = cp.cuda.Event(disable_timing=True)
    uploaded.record(stream)
    pending.append((uploaded, upload_memory, upload))
    if errors:
        error_memory = cp.cuda.alloc_pinned_memory(reset.nbytes)
        error_reset = np.frombuffer(error_memory, dtype=np.uint64, count=RUC_SFCTMP_FLAGS_SIZE)
        error_reset[:] = reset
        for index in errors:
            error_reset[index // 64] |= np.uint64(1) << np.uint64(index % 64)
        active_flags.set(error_reset, stream=stream)
        reset_done = cp.cuda.Event(disable_timing=True)
        reset_done.record(stream)
        pending.append((reset_done, error_memory, error_reset))
    else:
        active_flags.set(reset, stream=stream)
    context_key = (int(cp.cuda.runtime.getDevice()), active_flags.data.ptr)
    _SFCTMP_FLAG_CONTEXT[context_key] = (
        nzs, ncategory, mminlu, {index: (type(error), str(error)) for index, error in errors.items()})
    _SFCTMP_CONTEXT_STREAMS[context_key] = int(stream.ptr)
    if n:
        scalars = (timestep, np.float32(c1sn), np.float32(c2sn),
                   rsmax, np.int32(ice), np.int32(urban), np.int32(ncategory),
                   np.int32(min(errors, default=len(_SFCTMP_CHECKS))), np.int32(n))
        for stage in range(3):
            ruc_fused_kernel(f'ruc_sfctmp_stage{stage}', nzs, soilprop, snow)(
                ((n + 127) // 128,), (128,), (pointers, run, active_flags, alive, *scalars))
    if flags is None:
        ruc_sfctmp_raise_from_flags(active_flags)
    return out


__all__ += ['ruc_sfctmp_full_width_fused', 'ruc_sfctmp_raise_from_flags',
            'RUC_SFCTMP_FLAGS_SIZE', 'release_ruc_stream_scratch']
