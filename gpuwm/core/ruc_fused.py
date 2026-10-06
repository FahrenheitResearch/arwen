"""Full-width CUDA orchestration for the RUC runtime seam.

One RUC land-surface call is six kernels: the driver prologue (WRF's
surface-driver seam and LSMRUC's prologue), the three fused ``sfctmp``
stages (:func:`gpuwm.core.ruc_gpu.ruc_sfctmp_full_width_fused`), the driver
epilogue and a commit that writes the fields only when no check failed.
Every check records a flag bit on the card; one read of the flag words at
the end raises the message of the first check the array orchestration
(``gpuwm.core.ruc_runtime._ruc_lsm_step_reference``) would have failed.
The column arithmetic and every output word are that orchestration's.
"""

from functools import lru_cache

import cupy as cp
import numpy as np

from gpuwm.core import ruc, ruc_gpu
from gpuwm.core.ruc_tier import ruc_fused_kernel


# Bit numbers of the driver's uint32 flag words, ordered by the reference's
# checks.  Driver admissions use 0..79, prologue qsn checks use 80..85,
# epilogue qsn checks use 1024..1025, and driver outputs use 1056..1151.
# Words 36..39 carry the land, water, lake and sea-ice census.  sfctmp's own
# checks sit between the prologue's and the epilogue's in the reference and
# live in the fused sfctmp's uint64 words, which share one device slab with
# these so the call ends in ONE flag read.
OUTPUT_FLAG_FIRST = 1056
FLAG_WORDS = 36
_DRIVER_FLAG_SLOTS = 20  # the 40 uint32 driver words as uint64 slab slots


def _runtime_sfctmp(values, *, run, delt, conflx, ivgtyp, iland, nroot, ilnb,
                    isice, c1sn, c2sn, isncovr_opt, mminlu, parameters, flags,
                    soilprop="wrf_45", snow="wrf_461"):
    """The fused sfctmp, behind the runtime's leaf-set wiring seam.

    Tests replace the runtime leaf sets to prove that a host leaf cannot be
    reached silently; a replaced set takes the array orchestration with
    those leaves, so the tripwire still fires.  The ordinary call is the
    fused kernels, recording its checks in ``flags`` without a host read.
    """
    from gpuwm.core.ruc_runtime import ruc_device_sfctmp_sets
    leaves, stages, arrays = ruc_device_sfctmp_sets()
    kwargs = dict(delt=delt, conflx=conflx, ivgtyp=ivgtyp, iland=iland,
                  nroot=nroot, ilnb=ilnb, isice=isice, c1sn=c1sn, c2sn=c2sn,
                  isncovr_opt=isncovr_opt, mminlu=mminlu, parameters=parameters)
    if (leaves is ruc_gpu.RUC_SFCTMP_DEVICE_LEAVES_RESIDENT
            and stages is ruc_gpu.RUC_SFCTMP_DEVICE_STAGES_RESIDENT
            and arrays is ruc_gpu.RUC_DEVICE_ARRAYS):
        return ruc_gpu.ruc_sfctmp_full_width_fused(values, run=run, flags=flags,
                                                   soilprop=soilprop, snow=snow,
                                                   **kwargs)
    flags.fill(0)
    index = cp.nonzero(run)[0]
    take = {name: array[..., index] for name, array in values.items()}
    for name in ("conflx", "ivgtyp", "iland", "nroot", "ilnb"):
        kwargs[name] = kwargs[name][index]
    result = ruc.ruc_surface_temperature_step(take, myj=False, leaves=leaves,
                                             stages=stages, arrays=arrays,
                                             soilprop=soilprop, snow=snow,
                                             **kwargs)
    output = {}
    for name in ruc.RucSurfaceTemperatureStep.__dataclass_fields__:
        part = cp.asarray(getattr(result, name))
        full = cp.zeros(part.shape[:-1] + (run.size,), dtype=part.dtype)
        full[..., index] = part
        output[name] = full
    return output


_RUC_SFCTMP_FULL_WIDTH = _runtime_sfctmp

_COLUMNS = (ruc.RUC_DRIVER_COLUMN_STATE + ruc.RUC_DRIVER_COLUMN_FORCING
            + ruc.RUC_DRIVER_ARW_FORCING)
_EXTRAS = ("psfc", "chs2", "cqs2", "cpm", "qgh", "tsk_save", "tsk_sea",
           "flhc_sea", "flqc_sea", "cpm_sea", "cqs2_sea", "chs2_sea",
           "chs_sea", "qsfc_sea", "qgh_sea", "hfx_sea", "qfx_sea", "lh_sea")
_LOCALS = ("patm", "conflx", "prcpms", "newsnms", "snowrat", "grauprat",
           "icerat", "curat", "qkms", "tkms", "snwe", "snhei", "canwatr",
           "snowfrac", "rhosnfall", "rhosn", "emissl", "pc", "qwrtz",
           "rhocs", "bclh", "dqm", "ksat", "psis", "qmin", "ref", "wilt",
           "meltfactor", "lmavail", "sat", "cn", "snoh", "snflx", "s",
           "sublim", "evapl", "infiltr", "smelt", "runoff1", "runoff2",
           "t2", "th2", "q2", "scale", "inverse")
_PROFILES = ruc.RUC_DRIVER_PROFILE_STATE
_WORK_PROFILES = ("soilm1d", "tso1d", "smfrkeep", "keepfr", "soiliqw", "soilice")
_SCRATCH_NAMES = _COLUMNS + _EXTRAS + _LOCALS + _PROFILES + _WORK_PROFILES
_INPUT_NAMES = _COLUMNS + _EXTRAS + _PROFILES
_SF_OUTPUTS = tuple(ruc.RucSurfaceTemperatureStep.__dataclass_fields__)
_OUTPUT_NAMES = tuple(name for name in ruc.RucLandSurfaceStep.__dataclass_fields__
                      if name != "ilnb")

_SF_VALUES = {
    "soilm1d": "soilm1d", "ts1d": "tso1d", "smfrkeep": "smfrkeep",
    "keepfr": "keepfr", "soilice": "soilice", "soiliqw": "soiliqw",
    "seaice": "seaice", "gsw": "gsw", "tabs": "t3d", "tsnav": "tsnav",
    "prcpms": "prcpms", "newsnms": "newsnms", "vegfra": "vegfra",
    "lai": "lai", "sat": "sat", "soilt": "soilt", "snowfallac": "snowfallac",
    "alb_snow": "snoalb", "alb_snow_free": "albbck", "snowrat": "snowrat",
    "grauprat": "grauprat", "icerat": "icerat", "curat": "curat", "snwe": "snwe",
    "snhei": "snhei", "snowfrac": "snowfrac", "rhosn": "rhosn",
    "rhosnfall": "rhosnfall", "cst": "canwatr", "alb": "alb", "emiss": "emissl",
    "znt": "znt", "meltfactor": "meltfactor", "patm": "patm", "qvatm": "qv3d",
    "qcatm": "qc3d", "rho": "rho3d", "glw": "glw", "qkms": "qkms",
    "tkms": "tkms", "pc": "pc", "mavail": "lmavail", "qwrtz": "qwrtz",
    "rhocs": "rhocs", "dqm": "dqm", "qmin": "qmin", "ref": "ref",
    "wilt": "wilt", "psis": "psis", "bclh": "bclh", "ksat": "ksat",
    "cn": "cn", "soilt1": "soilt1", "qvg": "qvg", "qsg": "qsg", "qcg": "qcg",
    "snom": "snom", "acsnow": "acsnow", **{n: n for n in ruc.RUC_DRIVER_STALE_LOCALS},
}
_SCRATCH_NAMES += ("seaice",)


def _pinned(shape, dtype):
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    owner = cp.cuda.alloc_pinned_memory(size)
    return np.ndarray(shape, dtype=dtype, buffer=owner)


class _Workspace:
    def __init__(self, shape, nzs):
        self.shape, self.nzs = shape, nzs
        self.n = int(np.prod(shape))
        profiles = set(_PROFILES + _WORK_PROFILES)
        rows = sum(nzs if name in profiles else 1 for name in _SCRATCH_NAMES)
        self.storage = cp.empty((rows, self.n), dtype=cp.float32)
        self.arrays = {}
        row = 0
        for name in _SCRATCH_NAMES:
            count = nzs if name in profiles else 1
            self.arrays[name] = (self.storage[row:row + count] if name in profiles
                                 else self.storage[row])
            row += count
        self.integer = cp.empty((5, self.n), dtype=cp.int32)
        self.run = cp.empty(self.n, dtype=cp.bool_)
        # One slab, one read: the driver's 40 uint32 words, then the fused
        # sfctmp's uint64 buffer.
        self.flag_slab = cp.empty(_DRIVER_FLAG_SLOTS + ruc_gpu.RUC_SFCTMP_FLAGS_SIZE,
                                  dtype=cp.uint64)
        self.flags = self.flag_slab[:_DRIVER_FLAG_SLOTS].view(cp.uint32)
        self.sflags = self.flag_slab[_DRIVER_FLAG_SLOTS:]
        self.sptr = cp.asarray(np.asarray([self.storage.data.ptr], dtype=np.uint64))
        self.iptr = cp.empty(len(_INPUT_NAMES) + 2, dtype=cp.uint64)
        self.optr = cp.empty(len(_SF_OUTPUTS), dtype=cp.uint64)
        self.cptr = cp.empty(len(_INPUT_NAMES), dtype=cp.uint64)
        self.ihost = _pinned(self.iptr.shape, np.uint64)
        self.ohost = _pinned(self.optr.shape, np.uint64)
        self.chost = _pinned(self.cptr.shape, np.uint64)
        self.psfc = _pinned(shape, np.float32)
        self.scale = _pinned(shape, np.float32)
        self.inverse = _pinned(shape, np.float32)
        self.side = cp.cuda.Stream(non_blocking=True)
        self.start = cp.cuda.Event()
        self.power_ready = cp.cuda.Event()


_WORKSPACES = {}


def release_ruc_driver_stream_scratch(*, device_id, stream):
    """Release the completed stream's driver banks and auxiliary copies."""
    device_id = int(device_id)
    if int(cp.cuda.runtime.getDevice()) != device_id:
        raise ValueError("RUC driver scratch release requires the owning CUDA device")
    stream.synchronize()
    keys = [key for key in tuple(_WORKSPACES)
            if key[0] == device_id and key[-1] == int(stream.ptr)]
    for key in keys:
        # A failed step can leave a pinned-memory copy on this side queue
        # before the main queue has recorded its ordinary waiting event.
        _WORKSPACES[key].side.synchronize()
        del _WORKSPACES[key]
    return {"driver_workspaces": len(keys)}


@lru_cache(maxsize=None)
def _host_soil_geometry(nzs):
    return ruc.ruc_soil_geometry(nzs)


def _refusal(slab, params, sflags_key):
    """Raise the first failure the reference would raise, from one read."""
    flags = slab[:_DRIVER_FLAG_SLOTS].view(np.uint32)
    swords = slab[_DRIVER_FLAG_SLOTS:]
    if not (np.any(flags[:FLAG_WORDS])
            or np.any(swords[:ruc_gpu._SFCTMP_FLAG_WORDS])):
        return
    admissions = _PROFILES + _COLUMNS
    messages = [f"{name} must be finite" for name in admissions]
    vegetation = params.bundle.vegetation_for(params.dataset_identifier)
    messages += [f"RUC ivgtyp is outside 1..{len(vegetation.rows)} for {vegetation.name}",
                 f"RUC isltyp is outside 1..{len(params.bundle.soil.rows)}"]
    for bit, message in enumerate(messages):
        if int(flags[bit // 32]) & (1 << (bit % 32)):
            raise ValueError(message)
    for bit in range(80, 86):
        if int(flags[bit // 32]) & (1 << (bit % 32)):
            raise ValueError("RUC qsn temperatures must be finite" if bit % 2 == 0
                             else "RUC qsn index expression overflows Fortran INT")
    # sfctmp's checks, in its own order, between the prologue's and the
    # epilogue's.
    ruc_gpu._sfctmp_raise_from_words(swords, sflags_key)
    for bit in (1024, 1025):
        if int(flags[bit // 32]) & (1 << (bit % 32)):
            raise ValueError("RUC qsn temperatures must be finite" if bit == 1024
                             else "RUC qsn index expression overflows Fortran INT")
    for index, name in enumerate(_OUTPUT_NAMES):
        bit = OUTPUT_FLAG_FIRST + index
        if int(flags[bit // 32]) & (1 << (bit % 32)):
            raise ValueError(f"RUC driver produced non-finite {name}")


def _observe_driver(values, fields, params, timestep, itimestep,
                    mosaic_lu, mosaic_soil, lakemodel, irrigation="wrf_461",
                    soilprop="wrf_45", qvg_cold_start="wrf", snow="wrf_461"):
    """Keep the tests' pre-seam result observer while still running fusion.

    The ordinary runtime callable is unchanged and returns immediately here.
    A replaced callable may collect the independent reference driver result;
    it cannot replace the fused computation or the conditional field commit.
    """
    from gpuwm.core import ruc_runtime as runtime
    callback = runtime.ruc_land_surface_step
    if callback is ruc.ruc_land_surface_step:
        return
    copied = dict(values)
    copied["albbck"], ice = runtime._ruc_seaice_albedo_override(
        copied["albbck"], copied["xice"], params.seaice_albedo_default, arrays=cp,
        xice_threshold=params.xice_threshold)
    copied["alb"] = runtime._ruc_fractional_deblend(
        copied["alb"], 0.08, copied["xice"], ice, arrays=cp)
    copied["emiss"] = runtime._ruc_fractional_deblend(
        copied["emiss"], 0.98, copied["xice"], ice, arrays=cp)
    copied["soilt"] = cp.where(ice, fields["tsk_save"], copied["soilt"]).astype(cp.float32)
    leaves, stages, arrays = runtime.ruc_device_sfctmp_sets()
    callback(copied, dt=float(timestep), ktau=int(itimestep), zs=params.zs,
             ivgtyp=fields["ivgtyp"], isltyp=fields["isltyp"],
             myj=False, em_core=1, lakemodel=lakemodel, frpcpn=True,
             rdlai2d=bool(params.rdlai2d),
             mosaic_lu=mosaic_lu, mosaic_soil=mosaic_soil,
             landusef=fields.get("landusef"), soilctop=fields.get("soilctop"),
             iswater=params.iswater, isice=params.isice,
             xice_threshold=float(params.xice_threshold), ilnb=1, ilnb_chain=False,
             c1sn=0.026, c2sn=21.0,
             isncovr_opt=ruc.RUC_SNOW_COVER_OPTION, mminlu=params.dataset_identifier,
             parameters=params.bundle, leaves=leaves, stages=stages, arrays=arrays,
             irrigation=irrigation, soilprop=soilprop,
             qvg_cold_start=qvg_cold_start, snow=snow)


def step(fields, atmosphere, *, params, precipitation, dt, itimestep,
         mosaic_lu=0, mosaic_soil=0, lakemodel=0, irrigation="wrf_461",
         soilprop="wrf_45", qvg_cold_start="wrf", diagnostic_2m="flux",
         snow="wrf_461"):
    from gpuwm.core.ruc_tier import (ruc_2m_diagnostic_form,
                                     ruc_qvg_cold_start_form, ruc_snow_form,
                                     ruc_soilprop_form)
    soilprop = ruc_soilprop_form(soilprop)
    snow = ruc_snow_form(snow)
    qvg_air = ruc_qvg_cold_start_form(qvg_cold_start)
    log_profile = ruc_2m_diagnostic_form(diagnostic_2m)
    from gpuwm.core.ruc_runtime import (RUC_PROFILE_BINDING, RUC_STATE_BINDING,
                                        sfcdiags_exner_powers)
    from gpuwm.core.ruc_mosaic import irrigation_form
    irrigation_selector = irrigation_form(irrigation)

    shape = tuple(fields["tsk"].shape)
    timestep = np.float32(dt)
    if not np.isfinite(timestep) or timestep <= np.float32(0.0):
        raise ValueError("RUC driver dt must be finite and positive")
    levels = np.asarray(params.zs, dtype=np.float32)
    if levels.ndim != 1 or levels.size not in (6, 9):
        raise ValueError("RUC driver zs must be one of WRF's tabulated RUC grids, "
                         f"with 6 or 9 levels; got shape {levels.shape}")
    nzs = int(levels.size)
    pinned, _ = _host_soil_geometry(nzs)
    if not np.array_equal(levels, pinned):
        raise ValueError(f"RUC driver zs must be the pinned {nzs}-level "
                         f"RUC grid {tuple(float(v) for v in pinned)}")
    stream = cp.cuda.get_current_stream()
    key = (int(cp.cuda.runtime.getDevice()), shape, nzs, stream.ptr)
    w = _WORKSPACES.get(key)
    if w is None:
        w = _WORKSPACES[key] = _Workspace(shape, nzs)
    w.start.record(stream)
    w.side.wait_event(w.start)
    with w.side:
        fields["psfc"].get(out=w.psfc, stream=w.side, blocking=False)

    values = {argument: fields[name] for name, argument in RUC_STATE_BINDING.items()}
    values.update({argument: fields[name] for name, argument in RUC_PROFILE_BINDING.items()})
    values.update({"z3d": atmosphere["dz"][0], "p8w": atmosphere["pressure"][0],
                   "t3d": atmosphere["temperature"][0], "qv3d": atmosphere["qv"][0],
                   "qc3d": atmosphere["qc"][0], "rho3d": atmosphere["rho"][0],
                   "frzfrac": fields["sr"], "tbot": fields["tmn"],
                   "rainncv": precipitation.rain_nonconvective,
                   "snowncv": precipitation.snow_nonconvective,
                   "graupelncv": precipitation.graupel_nonconvective})
    for name in _INPUT_NAMES:
        if name not in values:
            values[name] = fields[name]
    _observe_driver(values, fields, params, timestep, itimestep,
                    mosaic_lu, mosaic_soil, lakemodel, irrigation, soilprop,
                    qvg_cold_start, snow)
    keep = []
    for index, name in enumerate(_INPUT_NAMES):
        array = cp.ascontiguousarray(values[name], dtype=cp.float32)
        expected = (nzs,) + shape if name in _PROFILES else shape
        if array.shape != expected:
            raise ValueError(f"{name} shape {array.shape}; expected {expected}")
        keep.append(array)
        w.ihost[index] = array.data.ptr
    for offset, name in enumerate(("ivgtyp", "isltyp"), len(_INPUT_NAMES)):
        raw = cp.asarray(fields[name])
        if raw.shape != shape:
            raise ValueError(f"{name} shape {raw.shape}; expected {shape}")
        if raw.dtype.kind not in "iu":
            raise TypeError(f"{name} must contain integer WRF categories")
        array = cp.ascontiguousarray(raw, dtype=cp.int32)
        keep.append(array)
        w.ihost[offset] = array.data.ptr
    w.iptr.set(w.ihost, stream=stream)
    w.flags.data.memset_async(0, w.flags.nbytes, stream)
    sflags_key = (key[0], w.sflags.data.ptr)
    tables, nv, ns, _ = ruc_gpu._bundle_device_tables(params.bundle, params.dataset_identifier)
    geometry = ruc_gpu._device_soil_half_levels(key[0], nzs)
    vegetation = params.bundle.vegetation_for(params.dataset_identifier)
    iswater = params.iswater if params.iswater is not None else (16 if vegetation.name == "USGS-RUC" else 17)
    isice = params.isice if params.isice is not None else (24 if vegetation.name == "USGS-RUC" else 15)
    if params.iswater is not None and (type(params.iswater) is not int or not 1 <= params.iswater <= nv):
        raise ValueError(f"RUC iswater {params.iswater!r} is outside 1..{nv}")
    if params.isice is not None and (type(params.isice) is not int or not 1 <= params.isice <= nv):
        raise ValueError(f"RUC isice {params.isice!r} is outside 1..{nv}")
    from gpuwm.core.ruc_mosaic import mosaic_fractions
    # The wrf_45 floor reads the fractions whenever the run carries them
    # (WRF v4.5.2 :970 has no mosaic gate); wrf_461 only under mosaic_lu.
    carries_fractions = fields.get("landusef") is not None
    landusef = (mosaic_fractions(fields.get("landusef"), shape, "landusef", nv, arrays=cp, validate_values=False)
                if (mosaic_lu or (irrigation_selector == 0 and carries_fractions))
                else cp.empty((0,), dtype=cp.float32))
    soilctop = (mosaic_fractions(fields.get("soilctop"), shape, "soilctop", ns, arrays=cp, validate_values=False)
                if mosaic_soil else cp.empty((0,), dtype=cp.float32))
    crop, natural = (int(vegetation.scalars[name]) for name in ("CROP", "NATURAL"))
    if landusef.shape[0] and landusef.shape[0] < max(crop, natural):
        raise ValueError("RUC landusef omits the table's crop/natural categories; "
                         "LSMRUC irrigation cannot index the source fractions")
    mosaic_args = (landusef, soilctop, np.int32(landusef.shape[0]),
                   np.int32(soilctop.shape[0]), np.int32(mosaic_lu), np.int32(mosaic_soil))
    prologue = ruc_fused_kernel("ruc_driver_prologue", nzs, soilprop, snow)
    epilogue = ruc_fused_kernel("ruc_driver_epilogue", nzs, soilprop, snow)
    commit = ruc_fused_kernel("ruc_driver_commit", nzs, soilprop, snow)
    grid, block = ((w.n + 127) // 128,), (128,)
    table_args = tuple(getattr(tables, name) for name in (
        "ifortbl", "z0tbl", "lemitbl", "pctbl", "laitbl", "bb", "drysmc", "hc",
        "maxsmc", "refsmc", "satpsi", "satdk", "wltsmc", "qtz"))
    tbq = ruc_gpu._device_tbq(key[0])
    prologue(grid, block, (w.iptr, w.sptr, w.integer, w.run, w.flags, *table_args,
                           tbq, np.int32(w.n), np.int32(itimestep), timestep,
                           np.int32(iswater), np.int32(isice),
                           np.int32(nv), np.int32(ns),
                           np.float32(params.seaice_albedo_default),
                           np.float32(vegetation.scalars["CFACTR_DATA"]),
                           *mosaic_args, np.int32(lakemodel),
                           np.int32(qvg_air),
                           np.int32(1 if params.rdlai2d else 0),
                           np.float32(params.xice_threshold)))
    sfvalues = {name: w.arrays[source] for name, source in _SF_VALUES.items()}
    hook_kwargs = dict(run=w.run, delt=float(timestep), conflx=w.arrays["conflx"],
                       ivgtyp=w.integer[0], iland=w.integer[2], nroot=w.integer[3],
                       ilnb=w.integer[4], isice=isice, c1sn=0.026, c2sn=21.0,
                       isncovr_opt=ruc.RUC_SNOW_COVER_OPTION, mminlu=params.dataset_identifier,
                       parameters=params.bundle, flags=w.sflags)
    if soilprop != "wrf_45":
        # Passed only off its default, so a replaced sfctmp hook written
        # before the selector still runs the default lineage.
        hook_kwargs["soilprop"] = soilprop
    if snow != "wrf_461":
        hook_kwargs["snow"] = snow
    try:
        result = _RUC_SFCTMP_FULL_WIDTH(sfvalues, **hook_kwargs)
    except Exception:
        _refusal(w.flag_slab.get(), params, sflags_key)
        raise
    for index, name in enumerate(_SF_OUTPUTS):
        w.ohost[index] = result[name].data.ptr
    w.optr.set(w.ohost, stream=stream)
    # The two power expressions stay on the host with their float32
    # boundaries, and take glibc's powf there on every host, as the
    # reference orchestration does.
    w.side.synchronize()
    scale, inverse = sfcdiags_exner_powers(w.psfc)
    w.scale[...] = scale
    w.inverse[...] = inverse
    w.arrays["scale"].reshape(shape).set(w.scale, stream=w.side)
    w.arrays["inverse"].reshape(shape).set(w.inverse, stream=w.side)
    w.power_ready.record(w.side)
    stream.wait_event(w.power_ready)
    epilogue(grid, block, (w.sptr, w.optr, w.integer, w.run, w.flags, tbq,
                           tables.lemitbl, geometry, timestep, np.int32(w.n),
                           landusef, np.int32(landusef.shape[0]), np.int32(mosaic_lu),
                           np.int32(crop), np.int32(natural),
                           np.int32(irrigation_selector), np.int32(log_profile),
                           np.float32(params.xice_threshold)))
    targets = {argument: fields[name] for name, argument in RUC_STATE_BINDING.items()}
    targets.update({argument: fields[name] for name, argument in RUC_PROFILE_BINDING.items()})
    targets.update({name: fields[name] for name in _EXTRAS})
    targets.update({name: fields["ruc_" + name] for name in ("infiltr", "smelt", "runoff1", "runoff2")})
    targets.update({name: fields[name] for name in ("t2", "th2", "q2")})
    # Commit has a fixed target order shared with the generated CUDA source.
    targets.update({name: fields[name] for name in ("albbck", "chs", "flhc", "flqc")})
    target_names = ruc.RUC_DRIVER_COLUMN_STATE + ("albbck", "chs", "flhc", "flqc") + _EXTRAS + _PROFILES + (
        "infiltr", "smelt", "runoff1", "runoff2", "t2", "th2", "q2")
    if w.cptr.size != len(target_names):
        w.cptr = cp.empty(len(target_names), dtype=cp.uint64)
        w.chost = _pinned(w.cptr.shape, np.uint64)
    for index, name in enumerate(target_names):
        w.chost[index] = targets[name].data.ptr
    w.cptr.set(w.chost, stream=stream)
    commit(grid, block, (w.sptr, w.cptr, w.flags, w.sflags,
                         np.int32(ruc_gpu._SFCTMP_FLAG_WORDS), np.int32(w.n)))
    slab = w.flag_slab.get()
    _refusal(slab, params, sflags_key)
    census = slab[:_DRIVER_FLAG_SLOTS].view(np.uint32)[36:40]
    return dict(zip(("land", "water", "lake", "sea_ice"), map(int, census)))
