"""Member-indexed launches of the original full-width RUC driver.

The six science kernels use blockIdx.y for member ownership. Each member
keeps the original column index, soil stride, scratch, flags and conditional
commit. Only entry-point pointer bindings change; the RUC arithmetic comes
from the ordinary fused source and its compiler hooks.
"""
from __future__ import annotations

import hashlib
from math import prod
import re

ENTRIES = ("ruc_driver_prologue", "ruc_sfctmp_stage0", "ruc_sfctmp_stage1",
           "ruc_sfctmp_stage2", "ruc_driver_epilogue", "ruc_driver_commit")
CONTRACT = "gpuwm-packed-ruc-driver.v1"
_MODULES = {}
_EXNER_BIND = '''
extern "C" __global__ void packed_ruc_bind_exner(
    const unsigned* powers,const unsigned long long* storage,
    int scale_row,int inverse_row,int n) {
    const unsigned long long member = blockIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if(column >= n) return;
    unsigned* target = reinterpret_cast<unsigned*>(storage[member]);
    target[scale_row * n + column] = powers[(member * 2) * n + column];
    target[inverse_row * n + column] = powers[(member * 2 + 1) * n + column];
}
'''


def _layout():
    from gpuwm.core import ruc_memory as layout
    from gpuwm.core.ruc_sfctmp_layout import _SFCTMP_ARRAYS
    return layout, _SFCTMP_ARRAYS


def packed_ruc_source(nzs=6, soilprop="wrf_45", *, snow="wrf_461"):
    """The ordinary translation unit with member-private entry bindings."""
    from gpuwm.core.ruc_tier import ruc_fused_source
    layout, arrays = _layout()
    original = ruc_fused_source(nzs, soilprop=soilprop, snow=snow)
    total_flags = layout.DRIVER_FLAG_SLOTS + layout.SFCTMP_FLAGS_SIZE
    bindings = {
        "ruc_driver_prologue": (
            "const float* member_dt,const int* member_ktau,const int* member_qvg_air",
            f"ip += member * {len(layout.DRIVER_INPUT_NAMES) + 2}; sp += member;\n"
            f"integer += member * 5 * n; run += member * n; flags += member * {total_flags * 2};\n"
            "dt = member_dt[member]; ktau = member_ktau[member]; qvg_air = member_qvg_air[member];\n"
            "if(nlcat) landusef += member * nlcat * n;\n"
            "if(mosaic_soil) soilctop += member * nscat * n;\n"),
        "ruc_driver_epilogue": (
            "const float* member_dt,const int* member_irrigation,const int* member_log_profile",
            f"sp += member; op += member * {len(layout.SFCTMP_OUTPUT_NAMES)};\n"
            f"integer += member * 5 * n; run += member * n; flags += member * {total_flags * 2};\n"
            "dt = member_dt[member]; irrigation = member_irrigation[member]; log_profile = member_log_profile[member];\n"
            "if(nlcat) landusef += member * nlcat * n;\n"),
        "ruc_driver_commit": (
            "", f"sp += member; cp += member * {len(layout.DRIVER_TARGET_NAMES)};\n"
            f"flags += member * {total_flags * 2}; sflags += member * {total_flags};\n"),
    }
    for stage in range(3):
        bindings[f"ruc_sfctmp_stage{stage}"] = (
            "const float* member_dt",
            f"ptrs += member * {len(arrays)}; run += member * n; alive += member * n;\n"
            f"flags += member * {total_flags}; delt = member_dt[member];\n")
    source = original
    for name in ENTRIES:
        pattern = re.compile(r'extern "C" __global__ void ' + re.escape(name)
                             + r'\((?P<arguments>.*?)\)\s*\{', re.S)
        matches = list(pattern.finditer(source))
        if len(matches) != 1:
            raise ValueError(f"ordinary RUC entry {name} changed; member bindings need its current ABI")
        match = matches[0]
        extra, body = bindings[name]
        arguments = match.group("arguments") + ("," + extra if extra else "")
        replacement = (f'extern "C" __global__ void packed_{name}({arguments}) {{\n'
                       "const unsigned long long member = blockIdx.y;\n" + body)
        source = source[:match.start()] + replacement + source[match.end():]
    return source + _EXNER_BIND


def workspace_allocations(members, shape, nzs=6):
    """Price the actual member-private banks and shared metadata tables."""
    layout, arrays = _layout()
    members, n = int(members), prod(shape)
    if members < 1 or len(shape) != 2 or min(shape) < 1 or nzs not in (6, 9):
        raise ValueError("packed RUC needs positive member/grid extents and a supported soil layout")
    ordinary = layout.driver_workspace_allocations(n, nzs)
    result = {name: ((members,) + dims, dtype) for name, (dims, dtype) in ordinary.items()}
    result["cptr"] = ((members, len(layout.DRIVER_TARGET_NAMES)), "uint64")
    _, scratch = layout.sfctmp_scratch_layout(n, nzs)
    result.update(sf_scratch=((members, scratch), "float32"),
        sf_pointers=((members, len(arrays)), "uint64"), alive=((members, n), "bool"),
        member_dt=((members,), "float32"), member_ktau=((members,), "int32"),
        member_qvg_air=((members,), "int32"), member_irrigation=((members,), "int32"),
        member_log_profile=((members,), "int32"),
        exner_upload=((members, 2, n), "float32"))
    for name, (dims, dtype) in layout.sfctmp_output_allocations(n, nzs).items():
        result["sf_output:" + name] = ((members,) + dims, dtype)
    return result


def _scratch_views(slab, n, nzs):
    """Bind the generated typed pointers within float32-sized scratch slots."""
    from gpuwm.core.ruc_sfctmp_layout import _SFCTMP_OUTPUTS
    layout, arrays = _layout()
    offsets, _ = layout.sfctmp_scratch_layout(n, nzs)
    views = {}
    for index, (binding, dtype, profile, location) in enumerate(arrays):
        if binding or index in _SFCTMP_OUTPUTS.values():
            continue
        length = (nzs if profile else 1) * n
        # Bool intermediates address the first n bytes of a slot. Integer
        # intermediates reinterpret its words. Both retain its member pitch.
        view = slab[:, offsets[location]:offsets[location] + length].view(dtype)[:, :length]
        views[index] = view.reshape(slab.shape[0], nzs, n) if profile else view
    return views


def _module(nzs, soilprop, xp, *, snow="wrf_461"):
    source = packed_ruc_source(nzs, soilprop, snow=snow)
    digest = hashlib.sha256(source.encode()).hexdigest()
    key = (int(xp.cuda.runtime.getDevice()), digest)
    if key not in _MODULES:
        from gpuwm.certify.kernel_manifest import record_module
        from gpuwm.core.kernels import _compile_observed
        module = xp.RawModule(code=source, options=("-std=c++17",))
        identity = f"{CONTRACT}:soil={nzs}:source={digest}"
        _compile_observed(module, identity)
        record_module(identity, source=source, options=("-std=c++17",), module=module)
        _MODULES[key] = module
    return _MODULES[key], digest


class PackedRucDriver:
    """One stream owns a priced bank; a call advances all its member columns."""
    def __init__(self, members, shape, *, nzs=6, available_bytes, array_module=None):
        if array_module is None:
            import cupy as array_module
        import numpy as np
        from gpuwm.core import ruc_memory as layout
        self.xp, self.members, self.shape, self.nzs = array_module, int(members), tuple(shape), int(nzs)
        allocations = workspace_allocations(self.members, self.shape, self.nzs)
        required = layout.allocation_bytes(allocations)
        if required > int(available_bytes):
            raise MemoryError("packed RUC private workspaces exceed their admitted device reservation")
        self.required_bytes = required
        self.n = prod(self.shape)
        self.device = int(self.xp.cuda.runtime.getDevice())
        self.stream = self.xp.cuda.get_current_stream()
        self.arrays = {name: self.xp.empty(dims, dtype=dtype)
                       for name, (dims, dtype) in allocations.items()}
        self.host = {}
        for name in ("iptr", "optr", "cptr", "sptr", "sf_pointers"):
            array = self.arrays[name]
            owner = self.xp.cuda.alloc_pinned_memory(array.nbytes)
            self.host[name] = np.ndarray(array.shape, dtype=np.uint64, buffer=owner)
        self.driver, self.rows = {}, {}
        profiles = set(layout.DRIVER_PROFILES + layout.DRIVER_WORK_PROFILES)
        row = 0
        for name in layout.DRIVER_SCRATCH_NAMES:
            rows = self.nzs if name in profiles else 1
            self.rows[name] = row
            view = self.arrays["storage"][:, row:row + rows]
            self.driver[name] = view if name in profiles else view[:, 0]
            row += rows
        self.sf_outputs = {name: self.arrays["sf_output:" + name] for name in layout.SFCTMP_OUTPUT_NAMES}
        self.sf_scratch = _scratch_views(self.arrays["sf_scratch"], self.n, self.nzs)
        self.flag_slots = layout.DRIVER_FLAG_SLOTS + layout.SFCTMP_FLAGS_SIZE
        self.flag_slab = self.arrays["flag_slab"]
        self.sf_flags = self.flag_slab[:, layout.DRIVER_FLAG_SLOTS:]
        self.integer, self.run = self.arrays["integer"], self.arrays["run"]
        self.host_psfc = np.empty((self.members,) + self.shape, dtype=np.float32)
        self.host_exner = np.empty((self.members, 2, self.n), dtype=np.float32)
        self.reset = np.zeros(self.flag_slab.shape, dtype=np.uint64)
        self.reset[:, layout.DRIVER_FLAG_SLOTS + layout.SFCTMP_FLAG_WORDS:] = np.iinfo(np.uint64).max
        self.last_receipt = None

    def _owned(self):
        if (int(self.xp.cuda.runtime.getDevice()) != self.device
                or self.xp.cuda.get_current_stream().ptr != self.stream.ptr):
            raise ValueError("packed RUC bank must run on its owning device and stream")

    def _pointer(self, value, member, *, member_shape, dtype, label, shared=False):
        import numpy as np
        if not isinstance(value, self.xp.ndarray) or value.dtype != np.dtype(dtype):
            raise TypeError(f"packed RUC {label} needs device {dtype} words")
        if int(value.device.id) != self.device:
            raise ValueError(f"packed RUC {label} is on another device")
        if shared and value.shape == member_shape and value.flags.c_contiguous:
            return value.data.ptr
        if value.shape != (self.members,) + member_shape:
            raise ValueError(f"packed RUC {label} needs shape {(self.members,) + member_shape}")
        # Atmosphere bottom slices may stride between members. Every member's
        # own column plane remains contiguous and is bound without a copy.
        contiguous = tuple(value.strides[1:])
        expected, stride = [], np.dtype(dtype).itemsize
        for dimension in reversed(member_shape):
            expected.insert(0, stride)
            stride *= dimension
        if contiguous != tuple(expected) or value.strides[0] < stride:
            raise ValueError(f"packed RUC {label} requires nonoverlapping contiguous member slabs")
        return value.data.ptr + member * value.strides[0]

    def _bind(self, name, values):
        target = self.host[name]
        for index, value, shape, dtype, shared in values:
            for member in range(self.members):
                target[member, index] = self._pointer(value, member, member_shape=shape,
                    dtype=dtype, label=f"{name}[{index}]", shared=shared)
        self.arrays[name].set(target, stream=self.stream)

    def step(self, fields, atmosphere, *, params, precipitation, dt, itimestep,
             mosaic_lu=0, mosaic_soil=0, lakemodel=0, irrigation="wrf_461",
             soilprop="wrf_45", qvg_cold_start="wrf", diagnostic_2m="flux", snow="wrf_461"):
        """Launch prologue, three sfctmp stages, epilogue and commit once each."""
        self._owned()
        import numpy as np
        from gpuwm.core import ruc, ruc_fused as ordinary, ruc_gpu, ruc_memory as layout
        from gpuwm.core.ruc_runtime import RUC_STATE_BINDING, RUC_PROFILE_BINDING, sfcdiags_exner_powers
        from gpuwm.core.ruc_sfctmp_layout import _SFCTMP_ARRAYS, _SFCTMP_OUTPUTS
        from gpuwm.core.ruc_tier import (ruc_soilprop_form, ruc_snow_form,
            ruc_qvg_cold_start_form, ruc_2m_diagnostic_form)
        from gpuwm.core.ruc_mosaic import irrigation_form
        soilprop = ruc_soilprop_form(soilprop)
        snow = ruc_snow_form(snow)
        qvg_air = ruc_qvg_cold_start_form(qvg_cold_start)
        log_profile = ruc_2m_diagnostic_form(diagnostic_2m)
        irrigation_selector = irrigation_form(irrigation)
        levels = np.asarray(params.zs, dtype=np.float32)
        pinned, _ = ordinary._host_soil_geometry(self.nzs)
        if not np.array_equal(levels, pinned):
            raise ValueError("packed RUC parameter soil geometry differs from its priced bank")
        # Fractions require the ordinary mosaic shape and irrigation checks
        # before a member-indexed entry can bind their category planes.
        if mosaic_lu or mosaic_soil:
            raise ValueError("packed RUC mosaic fraction bindings are not implemented; use the ordinary member driver")
        steps = np.asarray(dt, dtype=np.float32)
        counts = np.asarray(itimestep, dtype=np.int32)
        if steps.shape == ():
            steps = np.full(self.members, steps, dtype=np.float32)
        if counts.shape == ():
            counts = np.full(self.members, counts, dtype=np.int32)
        if steps.shape != (self.members,) or counts.shape != (self.members,) or not np.isfinite(steps).all() or not (steps > 0).all():
            raise ValueError("packed RUC dt/itimestep must bind every member's finite positive ordinary clock")
        self.arrays["member_dt"].set(steps, stream=self.stream)
        self.arrays["member_ktau"].set(counts, stream=self.stream)
        for name, selector in (("member_qvg_air", qvg_air), ("member_irrigation", irrigation_selector),
                               ("member_log_profile", log_profile)):
            self.arrays[name].set(np.full(self.members, selector, dtype=np.int32), stream=self.stream)
        self.flag_slab.set(self.reset, stream=self.stream)
        values = {argument: fields[name] for name, argument in RUC_STATE_BINDING.items()}
        values.update({argument: fields[name] for name, argument in RUC_PROFILE_BINDING.items()})
        values.update(z3d=atmosphere["dz"][:, 0], p8w=atmosphere["pressure"][:, 0],
            t3d=atmosphere["temperature"][:, 0], qv3d=atmosphere["qv"][:, 0],
            qc3d=atmosphere["qc"][:, 0], rho3d=atmosphere["rho"][:, 0],
            frzfrac=fields["sr"], tbot=fields["tmn"],
            rainncv=precipitation.rain_nonconvective, snowncv=precipitation.snow_nonconvective,
            graupelncv=precipitation.graupel_nonconvective)
        for name in layout.DRIVER_INPUT_NAMES:
            if name not in values:
                values[name] = fields[name]
        bindings = [(index, values[name], ((self.nzs,) + self.shape if name in layout.DRIVER_PROFILES else self.shape), "float32", False)
                    for index, name in enumerate(layout.DRIVER_INPUT_NAMES)]
        bindings += [(len(layout.DRIVER_INPUT_NAMES) + index, fields[name], self.shape, "int32", True)
                     for index, name in enumerate(("ivgtyp", "isltyp"))]
        self._bind("iptr", bindings)
        self._bind("sptr", [(0, self.arrays["storage"], self.arrays["storage"].shape[1:], "float32", False)])
        tables, nv, ns, _ = ruc_gpu._bundle_device_tables(params.bundle, params.dataset_identifier)
        vegetation = params.bundle.vegetation_for(params.dataset_identifier)
        iswater = params.iswater if params.iswater is not None else (16 if vegetation.name == "USGS-RUC" else 17)
        isice = params.isice if params.isice is not None else (24 if vegetation.name == "USGS-RUC" else 15)
        for name, selected in (("iswater", params.iswater), ("isice", params.isice)):
            if selected is not None and (type(selected) is not int or not 1 <= selected <= nv):
                raise ValueError(f"RUC {name} {selected!r} is outside 1..{nv}")
        landusef, soilctop = fields.get("landusef"), fields.get("soilctop")
        carries_fractions = irrigation_selector == 0 and landusef is not None
        nlcat = landusef.shape[1] if carries_fractions else 0
        nscat = soilctop.shape[1] if mosaic_soil else 0
        if carries_fractions:
            if not 1 <= nlcat <= nv:
                raise ValueError("packed RUC landusef category count is outside its original parameter table")
            crop, natural = (int(vegetation.scalars[name]) for name in ("CROP", "NATURAL"))
            if nlcat < max(crop, natural):
                raise ValueError("RUC landusef omits the table's crop/natural categories; LSMRUC irrigation cannot index the source fractions")
            for member in range(self.members):
                self._pointer(landusef, member, member_shape=(nlcat,) + self.shape,
                              dtype="float32", label="landusef")
        else:
            landusef = self.xp.empty(0, dtype=np.float32)
        if not mosaic_soil:
            soilctop = self.xp.empty(0, dtype=np.float32)
        module, source_sha = _module(self.nzs, soilprop, self.xp, snow=snow)
        kernel = lambda name: module.get_function("packed_" + name)
        grid, block = ((self.n + 127) // 128, self.members), (128,)
        table_args = tuple(getattr(tables, name) for name in (
            "ifortbl", "z0tbl", "lemitbl", "pctbl", "laitbl", "bb", "drysmc", "hc",
            "maxsmc", "refsmc", "satpsi", "satdk", "wltsmc", "qtz"))
        tbq = ruc_gpu._device_tbq(self.device)
        kernel("ruc_driver_prologue")(grid, block, (self.arrays["iptr"], self.arrays["sptr"], self.integer,
            self.run, self.flag_slab, *table_args, tbq, np.int32(self.n), np.int32(0), np.float32(0),
            np.int32(iswater), np.int32(isice), np.int32(nv), np.int32(ns),
            np.float32(params.seaice_albedo_default), np.float32(vegetation.scalars["CFACTR_DATA"]),
            landusef, soilctop, np.int32(nlcat), np.int32(nscat), np.int32(mosaic_lu), np.int32(mosaic_soil),
            np.int32(lakemodel), np.int32(0), np.int32(bool(params.rdlai2d)), np.float32(params.xice_threshold),
            self.arrays["member_dt"], self.arrays["member_ktau"], self.arrays["member_qvg_air"]))
        sfvalues = {name: self.driver[source] for name, source in ordinary._SF_VALUES.items()}
        bundle, sf_tables, ncategory, urban, rsmax = ruc_gpu._sfctmp_tables(params.bundle, params.dataset_identifier, self.nzs)
        sf_inputs = {"value:" + name: value for name, value in sfvalues.items()}
        sf_inputs.update(ivgtyp=self.integer[:, 0], iland=self.integer[:, 2], nroot=self.integer[:, 3],
                         ilnb=self.integer[:, 4], conflx=self.driver["conflx"], **sf_tables)
        sf_bindings = []
        for index, (binding, dtype, profile, location) in enumerate(_SFCTMP_ARRAYS):
            if binding:
                value = sf_inputs[binding]
                shared = not binding.startswith("value:") and binding not in ("ivgtyp", "iland", "nroot", "ilnb", "conflx")
                member_shape = value.shape if shared else ((self.nzs, self.n) if profile else (self.n,))
            elif index in _SFCTMP_OUTPUTS.values():
                name = next(name for name, output_index in _SFCTMP_OUTPUTS.items() if output_index == index)
                value, shared = self.sf_outputs[name], False
                member_shape = (self.nzs, self.n) if profile else (self.n,)
            else:
                value, shared = self.sf_scratch[index], False
                member_shape = (self.nzs, self.n) if profile else (self.n,)
            sf_bindings.append((index, value, member_shape, dtype, shared))
        self._bind("sf_pointers", sf_bindings)
        scalars = (np.float32(0), np.float32(0.026), np.float32(21.0), rsmax, np.int32(isice),
                   np.int32(urban), np.int32(ncategory), np.int32(len(ruc_gpu._SFCTMP_CHECKS)), np.int32(self.n))
        for stage in range(3):
            kernel(f"ruc_sfctmp_stage{stage}")(grid, block, (self.arrays["sf_pointers"], self.run,
                self.sf_flags, self.arrays["alive"], *scalars, self.arrays["member_dt"]))
        self._bind("optr", [(index, self.sf_outputs[name], self.sf_outputs[name].shape[1:],
            str(self.sf_outputs[name].dtype), False) for index, name in enumerate(layout.SFCTMP_OUTPUT_NAMES)])
        fields["psfc"].get(out=self.host_psfc)
        scale, inverse = sfcdiags_exner_powers(self.host_psfc)
        self.host_exner[:, 0] = scale.reshape(self.members, self.n)
        self.host_exner[:, 1] = inverse.reshape(self.members, self.n)
        self.arrays["exner_upload"].set(self.host_exner, stream=self.stream)
        kernel("ruc_bind_exner")(grid, block, (self.arrays["exner_upload"], self.arrays["sptr"],
            np.int32(self.rows["scale"]), np.int32(self.rows["inverse"]), np.int32(self.n)))
        geometry = ruc_gpu._device_soil_half_levels(self.device, self.nzs)
        kernel("ruc_driver_epilogue")(grid, block, (self.arrays["sptr"], self.arrays["optr"], self.integer,
            self.run, self.flag_slab, tbq, tables.lemitbl, geometry, np.float32(0), np.int32(self.n),
            landusef, np.int32(nlcat), np.int32(mosaic_lu), np.int32(vegetation.scalars["CROP"]), np.int32(vegetation.scalars["NATURAL"]),
            np.int32(0), np.int32(0), np.float32(params.xice_threshold), self.arrays["member_dt"],
            self.arrays["member_irrigation"], self.arrays["member_log_profile"]))
        targets = {argument: fields[name] for name, argument in RUC_STATE_BINDING.items()}
        targets.update({argument: fields[name] for name, argument in RUC_PROFILE_BINDING.items()})
        targets.update({name: fields[name] for name in layout.DRIVER_EXTRAS})
        targets.update({name: fields["ruc_" + name] for name in ("infiltr", "smelt", "runoff1", "runoff2")})
        targets.update({name: fields[name] for name in ("t2", "th2", "q2", "albbck", "chs", "flhc", "flqc")})
        self._bind("cptr", [(index, targets[name], ((self.nzs,) + self.shape if name in layout.DRIVER_PROFILES else self.shape),
                           "float32", False) for index, name in enumerate(layout.DRIVER_TARGET_NAMES)])
        kernel("ruc_driver_commit")(grid, block, (self.arrays["sptr"], self.arrays["cptr"], self.flag_slab,
            self.sf_flags, np.int32(layout.SFCTMP_FLAG_WORDS), np.int32(self.n)))
        slab = self.flag_slab.get()
        census = []
        for member in range(self.members):
            context = (self.device, self.sf_flags.data.ptr + member * self.sf_flags.strides[0])
            ruc_gpu._SFCTMP_FLAG_CONTEXT[context] = (self.nzs, ncategory, params.dataset_identifier, {})
            try:
                ordinary._refusal(slab[member], params, context)
            except Exception as error:
                error.member_id = member
                raise
            finally:
                ruc_gpu._SFCTMP_FLAG_CONTEXT.pop(context, None)
            census.append(dict(zip(("land", "water", "lake", "sea_ice"), map(int, slab[member].view(np.uint32)[36:40]))))
        self.last_receipt = {"contract": CONTRACT, "members": self.members, "shape": self.shape,
            "soil_levels": self.nzs, "science_launches": 6, "source_sha256": source_sha,
            "exner_word_copy_launches": 1,
            "required_workspace_bytes": self.required_bytes, "member_census": census,
            "flags_and_commit": "independent per member", "arithmetic": "ordinary fused source unchanged"}
        self.last_receipt["selectors"] = {"ruc_irrigation": irrigation, "ruc_soilprop": soilprop,
            "ruc_qvg_cold_start": qvg_cold_start, "ruc_2m_diagnostic": diagnostic_2m, "ruc_snow": snow}
        self.last_receipt["landusef_member_private"] = carries_fractions
        return tuple(census)

    def close(self):
        self._owned()
        self.stream.synchronize()
        self.arrays.clear()
        self.driver.clear()
        self.sf_outputs.clear()
        self.sf_scratch.clear()
        self.host.clear()
        self.host_psfc = self.host_exner = self.reset = None
        self.flag_slab = self.sf_flags = self.integer = self.run = None


__all__ = ["CONTRACT", "ENTRIES", "PackedRucDriver", "packed_ruc_source", "workspace_allocations"]
