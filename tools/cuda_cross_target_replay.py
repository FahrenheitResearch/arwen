"""Diagnostic NVRTC-target replay of the randomized Smagorinsky kernels.

Each target compiles the production source with its ordinary options. A PTX
header copy is retargeted to one physical GPU and linked to a native cubin so
the target's generated arithmetic can execute there. This tests NVRTC target
choices. It does not prove the different physical targets' ptxas choices or
hardware execution. Paired physical-card forecasts provide that evidence.

The compile subcommand is CPU-only. GPU ownership belongs to the caller of
the replay subcommand. All outputs belong in caller-provided scratch.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import ctypes.util
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys

TARGETS = (100, 120)
DIAGNOSTIC_TARGETS = (80, 89, 90)
SCENARIOS = (
    (False, False, False, False, False),
    (True, False, False, False, False),
    (True, True, False, False, True),
    (True, True, True, False, False),
    (True, True, False, True, True),
    (False, True, True, True, False),
    (False, False, True, False, True),
    (False, True, False, True, True),
)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def sources(kernel_dir=None):
    from gpuwm.core.kernels import module_source, module_source_int_defines
    keywords = {} if kernel_dir is None else {"kernel_dir": Path(kernel_dir)}
    return {
        "production": module_source("smag2d", **keywords),
        "direct_w": module_source_int_defines(
            "smag2d", (("GPUWM_SMAG_DIRECT_W_REFERENCE", 1),), **keywords),
    }


def compiler_options(target):
    """Production RawModule options, including the CuPy NVRTC defaults."""
    from gpuwm.core.kernels import module_options

    return module_options("smag2d") + (
        "-ftz=true", f"-arch=compute_{target}",
        "--device-as-default-execution-space")


def validate_artifact_variants(variants, target, expected_names):
    """Reject stale options or a missing source variant before GPU loading."""
    if set(variants) != set(expected_names):
        raise AssertionError(f"compute_{target} replay source variants are incomplete")
    expected = compiler_options(target)
    for name, row in variants.items():
        if tuple(row.get("options", ())) != expected:
            raise AssertionError(f"compute_{target} {name} artifact options differ from production")


def find_nvjitlink():
    for key in ("CUDA_PATH", "CUDA_HOME"):
        raw = os.environ.get(key)
        if not raw:
            continue
        root = Path(raw)
        for folder in (root / "lib", root / "lib64", root / "bin"):
            for pattern in ("libnvJitLink.so*", "nvJitLink*.dll"):
                candidates = sorted(folder.glob(pattern))
                if candidates:
                    return str(candidates[-1])
    for root in (Path(sys.prefix), Path("/usr/local/cuda")):
        for pattern in ("lib/python*/site-packages/nvidia/*/lib/libnvJitLink.so*",
                        "lib64/libnvJitLink.so*", "Library/bin/nvJitLink*.dll"):
            candidates = sorted(root.glob(pattern))
            if candidates:
                return str(candidates[-1])
    found = ctypes.util.find_library("nvJitLink")
    if found:
        return found
    raise FileNotFoundError("nvJitLink is required for cross-target replay")


class NvJitLink:
    """Version-aware binding to the installed public nvJitLink API."""

    def __init__(self, library=None):
        self.library_path = library or find_nvjitlink()
        self.library = ct.CDLL(self.library_path)
        version = self.library.nvJitLinkVersion
        version.argtypes = [ct.POINTER(ct.c_uint), ct.POINTER(ct.c_uint)]
        version.restype = ct.c_int
        major, minor = ct.c_uint(), ct.c_uint()
        self.check("Version", version(ct.byref(major), ct.byref(minor)))
        self.version = [major.value, minor.value]
        self.symbols = {}
        signatures = {
            "Create": [ct.POINTER(ct.c_void_p), ct.c_uint32, ct.POINTER(ct.c_char_p)],
            "Destroy": [ct.POINTER(ct.c_void_p)],
            "AddData": [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_char_p],
            "Complete": [ct.c_void_p],
            "GetLinkedCubinSize": [ct.c_void_p, ct.POINTER(ct.c_size_t)],
            "GetLinkedCubin": [ct.c_void_p, ct.c_void_p],
            "GetErrorLogSize": [ct.c_void_p, ct.POINTER(ct.c_size_t)],
            "GetErrorLog": [ct.c_void_p, ct.c_void_p],
        }
        for name, arguments in signatures.items():
            choices = ["nvJitLink" + name]
            choices += [f"__nvJitLink{name}_{major.value}_{v}"
                        for v in range(minor.value, -1, -1)]
            for symbol in choices:
                function = getattr(self.library, symbol, None)
                if function is not None:
                    break
            else:
                raise AttributeError(f"nvJitLink API symbol {name} is absent")
            function.argtypes, function.restype = arguments, ct.c_int
            setattr(self, name, function)
            self.symbols[name] = symbol

    @staticmethod
    def check(operation, code):
        if code:
            raise RuntimeError(f"nvJitLink {operation} returned {int(code)}")

    def error_log(self, handle):
        size = ct.c_size_t()
        if self.GetErrorLogSize(handle, ct.byref(size)) or not size.value:
            return ""
        buffer = ct.create_string_buffer(size.value)
        if self.GetErrorLog(handle, buffer):
            return "error log unavailable"
        return buffer.value.decode("utf-8", errors="replace")

    def link(self, ptx, host_sm, name):
        handle = ct.c_void_p()
        options = (ct.c_char_p * 1)(f"-arch=sm_{host_sm}".encode())
        self.check("Create", self.Create(ct.byref(handle), 1, options))
        try:
            buffer = ct.create_string_buffer(ptx)
            self.check("AddData", self.AddData(handle, 2, buffer, len(ptx), name.encode()))
            try:
                self.check("Complete", self.Complete(handle))
            except RuntimeError as error:
                raise RuntimeError(f"{error}: {self.error_log(handle)}") from error
            size = ct.c_size_t()
            self.check("GetLinkedCubinSize", self.GetLinkedCubinSize(handle, ct.byref(size)))
            cubin = ct.create_string_buffer(size.value)
            self.check("GetLinkedCubin", self.GetLinkedCubin(handle, cubin))
            return bytes(cubin.raw)
        finally:
            self.check("Destroy", self.Destroy(ct.byref(handle)))


def retarget_ptx(ptx, host_sm):
    """Rewrite only the diagnostic PTX target declaration, never runtime code."""
    pattern = rb"(?m)^(\s*\.target\s+)sm_[0-9a-z]+"
    replacement = lambda match: match.group(1) + f"sm_{host_sm}".encode()
    replay, count = re.subn(pattern, replacement, ptx)
    if count != 1:
        raise ValueError(f"expected one PTX target declaration, found {count}")
    return replay


def compile_modules(outdir, host_sm, targets=TARGETS, *, kernel_dir=None,
                    base_kernel_dir=None):
    from gpuwm.nvrtc_cache_key import compile_program

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    linker = NvJitLink()
    source_set = sources(kernel_dir)
    receipt = {
        "schema": "gpuwm-cross-target-smag-replay-v1", "host_sm": int(host_sm),
        "runtime_ftz": True,
        "source_sha256": {name: digest(source.encode()) for name, source in source_set.items()},
        "nvjitlink_version": linker.version, "nvjitlink_symbols": linker.symbols,
        "method": "NVRTC compiler targets replayed through one native linker target",
        "limitations": "PTX target header is diagnostically rewritten; physical ptxas and hardware differences require paired-card proof",
        "targets": {},
    }
    for target in targets:
        target_rows = receipt["targets"][str(target)] = {}
        for variant, source in source_set.items():
            options = compiler_options(target)
            row = target_rows[variant] = {"options": options}
            try:
                ptx = compile_program(source, options)
                ptx = ptx.encode() if isinstance(ptx, str) else ptx
            except Exception as error:
                message = str(error)
                unsupported = ("gpu-architecture" in message and
                               ("invalid value" in message or "not supported" in message))
                row.update(status="unsupported" if unsupported else "compile_error", error=message)
                continue
            replay = retarget_ptx(ptx, host_sm)
            stem = f"compute_{target}-{variant}"
            (outdir / (stem + ".ptx")).write_bytes(ptx)
            (outdir / (stem + "-retargeted.ptx")).write_bytes(replay)
            row.update(ptx_sha256=digest(ptx), replay_ptx_sha256=digest(replay),
                       retargeted=ptx != replay, header_rewrites=[".target"],
                       original_ptx=stem + ".ptx", replay_ptx=stem + "-retargeted.ptx")
            try:
                cubin = linker.link(replay, host_sm, stem + ".ptx")
            except Exception as error:
                row.update(status="link_error", error=str(error))
                continue
            filename = stem + ".cubin"
            (outdir / filename).write_bytes(cubin)
            row.update(status="linked", cubin=filename, cubin_sha256=digest(cubin))
    if base_kernel_dir is not None:
        originals = sources(base_kernel_dir)
        control = receipt["blackwell_control"] = {
            "compute_target": 120, "host_sm": int(host_sm),
            "native_blackwell": int(host_sm) == 120,
            "source_sha256": {name: digest(source.encode()) for name, source in originals.items()},
            "variants": {},
        }
        for variant, source in originals.items():
            options = compiler_options(120)
            row = control["variants"][variant] = {"options": options}
            try:
                ptx = compile_program(source, options)
                ptx = ptx.encode() if isinstance(ptx, str) else ptx
                replay = retarget_ptx(ptx, host_sm)
                stem = f"original-compute_120-{variant}"
                (outdir / (stem + ".ptx")).write_bytes(ptx)
                (outdir / (stem + "-retargeted.ptx")).write_bytes(replay)
                cubin = linker.link(replay, host_sm, stem + ".ptx")
                filename = stem + ".cubin"
                (outdir / filename).write_bytes(cubin)
                row.update(status="linked", ptx_sha256=digest(ptx),
                           replay_ptx_sha256=digest(replay), cubin=filename,
                           cubin_sha256=digest(cubin))
            except Exception as error:
                row.update(status="control_error", error=str(error))
    (outdir / "compile-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def load_modules(outdir, *, kernel_dir=None, base_kernel_dir=None):
    import cupy as cp

    outdir = Path(outdir)
    receipt = json.loads((outdir / "compile-receipt.json").read_text())
    if receipt.get("runtime_ftz") is not True:
        raise AssertionError("cross-target artifacts omit the production RawModule flush mode")
    current = {name: digest(source.encode()) for name, source in sources(kernel_dir).items()}
    if receipt["source_sha256"] != current:
        raise AssertionError("cross-target replay artifacts were compiled from different source bytes")
    device = cp.cuda.Device()
    if int(device.compute_capability) != receipt["host_sm"]:
        raise AssertionError("cross-target cubins belong to a different physical GPU target")
    modules = {}
    for target, variants in receipt["targets"].items():
        validate_artifact_variants(variants, int(target), current)
        errors = [name for name, row in variants.items() if row["status"] not in ("linked", "unsupported")]
        if errors:
            raise AssertionError(f"target compute_{target} compilation/linking failed: {variants}")
        if any(row["status"] == "unsupported" for row in variants.values()):
            if int(target) in TARGETS:
                raise AssertionError(f"required target compute_{target} is unsupported by this toolkit")
            continue
        modules[int(target)] = {}
        for name, row in variants.items():
            artifact = outdir / row["cubin"]
            if digest(artifact.read_bytes()) != row["cubin_sha256"]:
                raise AssertionError("cross-target cubin checksum changed")
            modules[int(target)][name] = cp.RawModule(path=str(artifact))
    if not set(TARGETS) <= set(modules):
        raise AssertionError("required compute_100 and compute_120 replay artifacts are absent")
    control = receipt.get("blackwell_control")
    if control is not None:
        validate_artifact_variants(control["variants"], 120, current)
        if base_kernel_dir is not None:
            current_base = {name: digest(source.encode())
                            for name, source in sources(base_kernel_dir).items()}
            if current_base != control["source_sha256"]:
                raise AssertionError("Blackwell control source bytes changed")
        original_modules = modules[120]["blackwell_control"] = {}
        for name, row in control["variants"].items():
            if row["status"] != "linked" or "-ftz=true" not in row["options"]:
                raise AssertionError(f"original Blackwell control is invalid: {row}")
            artifact = outdir / row["cubin"]
            if digest(artifact.read_bytes()) != row["cubin_sha256"]:
                raise AssertionError("original Blackwell control cubin checksum changed")
            original_modules[name] = cp.RawModule(path=str(artifact))
    return modules, receipt


def fixture_calls(scenario):
    """Freeze one realistic randomized fixture before varying compiler target."""
    import cupy as cp
    import numpy as np
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import _save_time_t, _wrf_smag_grid_args
    from gpuwm.verify.npref import random_acoustic_state

    moist, terrain, boundary_x, boundary_y, time_t = scenario
    nz, ny, nx = 8, 9, 17
    state, cfg = random_acoustic_state(
        seed=734, nx=nx, ny=ny, nz=nz, stretch=1.4,
        hybrid_opt=2 if terrain else 0, hill_height=300.0 if terrain else 0.0,
        msf_amp=0.09, moist=moist)
    cfg = replace(cfg, km_opt=4, bl_pbl_physics=1, dx=900.0, dy=1100.0,
                  open_x=boundary_x, open_y=boundary_y)
    random = np.random.default_rng(908)

    def array(shape, low, high):
        return cp.asarray(random.uniform(low, high, shape), dtype=cp.float32)

    if moist:
        state.qv[...] = array(state.qv.shape, 0.001, 0.025)
        update_diagnostics(state, cfg.hypsometric_opt)
    _save_time_t(state)
    state.u[...] *= cp.float32(1.5)
    state.v[...] *= cp.float32(0.75)
    state.w[...] *= cp.float32(1.25)
    state.php[...] += cp.float32(0.03125)
    if moist:
        state.qv[...] *= cp.float32(0.8)
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx), np.int32(state.phb.ndim == 3),
            np.int32(boundary_x), np.int32(boundary_y)]
    mass = (nz, ny, nx)
    km, kh = array(mass, 0.0, 700.0), array(mass, 0.0, 2100.0)
    km[:, :, ::5], kh[:, :, ::5] = cp.float32(0), cp.float32(0)
    d11, d22, d12 = [array(mass, -0.02, 0.02) for _ in range(3)]
    fx, fy = array((nz, ny, nx + 1), -0.2, 0.2), array((nz, ny + 1, nx), -0.2, 0.2)
    scalar = state.thp0 if time_t else state.thp
    t_u, t_v, t_w = [array(shape, -0.2, 0.2) for shape in (state.u.shape, state.v.shape, state.w.shape)]
    t_s = array(mass, -0.2, 0.2)
    tke, bn2 = array(mass, 0.1, 4.0), array(mass, -1.0e-4, 1.0e-4)
    t_tke = array(mass, -0.2, 0.2)
    budget_terms = [cp.zeros(mass, dtype=cp.float32) for _ in range(4)]
    ustm, hfx = array((ny, nx), 0.1, 1.0), array((ny, nx), -50.0, 250.0)
    mut = state.mub2d + (state.mup0 if time_t else state.mup)
    qfx = array((ny, nx), -1.0e-5, 1.0e-5)
    t_q = array(mass, -0.2, 0.2)
    kmv, khv = array(mass, 0.0, 700.0), array(mass, 0.0, 2100.0)
    qc, qi = array(mass, 0.0, 2.0e-4), array(mass, 0.0, 2.0e-4)
    cached_what = array((nz + 1, ny, nx), -2.0, 2.0)
    cached_rdz = array((nz + 1, ny, nx), 0.001, 0.01)
    cached_zx = array((nz + 1, ny, nx + 1), -0.02, 0.02)
    cached_zy = array((nz + 1, ny + 1, nx), -0.02, 0.02)
    rdx, rdy = np.float32(1 / cfg.dx), np.float32(1 / cfg.dy)
    legacy_tail = [np.int32(nz), np.int32(ny), np.int32(nx)]
    grid = (1, ny, nz)
    face_grid = (1, ny + 1, nz)
    calls = [
        ("production", "smag2d_km", grid,
         [state.u, state.v, km, kh, rdx, rdy, np.float32(np.sqrt(cfg.dx * cfg.dy)),
          np.float32(cfg.c_s), np.float32(1 / 3)] + legacy_tail, (2, 3)),
        ("production", "smag_hd_s", grid,
         [scalar, kh, mut, state.c1h, state.c2h, rdx, rdy, t_s] + legacy_tail, (7,)),
        ("production", "smag_hd_u", grid,
         [state.u, km, mut, state.c1h, state.c2h, rdx, rdy, t_u]
         + legacy_tail + [np.int32(boundary_x)], (7,)),
        ("production", "smag_hd_v", (1, ny + 1, nz),
         [state.v, km, mut, state.c1h, state.c2h, rdx, rdy, t_v]
         + legacy_tail + [np.int32(boundary_y)], (7,)),
        ("production", "smag_hd_w", (1, ny, nz + 1),
         [state.w, km, mut, state.c1f, state.c2f, rdx, rdy, t_w] + legacy_tail, (7,)),
        ("production", "wrf_smag_deform", grid, common + [d11, d22, d12] + dims, (22, 23, 24)),
        ("production", "wrf_smag2d_km", grid,
         common + [np.float32(cfg.c_s), np.float32(1 / 3), d11, d22, d12, km, kh] + dims, (27, 28)),
        ("production", "wrf_smag_hd_u", (1, ny, nz), common + [km, d11, d12, t_u] + dims, (25,)),
        ("production", "wrf_smag_hd_v", (1, ny + 1, nz), common + [km, d22, d12, t_v] + dims, (25,)),
        ("production", "wrf_smag_flux_s", face_grid,
         common + [scalar, kh, state.thb, np.int32(1), np.int32(state.thb.ndim == 3), fx, fy] + dims, (27, 28)),
        ("production", "wrf_smag_hd_s", grid, common + [fx, fy, t_s] + dims, (24,)),
        ("production", "wrf_smag_w_stress", face_grid, common + [km, fx, fy] + dims, (23, 24)),
        ("production", "wrf_smag_hd_w_stress", (1, ny, nz + 1), common + [fx, fy, t_w] + dims, (24,)),
        ("direct_w", "wrf_smag_hd_w", (1, ny, nz + 1), common + [km, t_w] + dims, (23,)),
        ("production", "wrf_tke_rhs", grid,
         common + [scalar, state.thb, np.int32(state.thb.ndim == 3),
                   tke, bn2, d11, d22, d12, km, km, kh,
                   mut, state.c1h, state.c2h, ustm, hfx,
                   np.int32(1), np.int32(1), np.float32(0.15), np.float32(5.0),
                   np.float32(0.01), np.float32(0.02), np.int32(int(moist) + int(terrain)),
                   t_tke] + budget_terms + [np.int32(1)] + dims,
         (45, 46, 47, 48, 49)),
        ("production", "wrf_smag_km_bc", grid,
         [km, kh] + legacy_tail + [np.int32(boundary_x), np.int32(boundary_y)], (0, 1)),
        ("production", "wrf_smag_vd_u", grid, common + [km, t_u] + dims, (23,)),
        ("production", "wrf_smag_vd_v", (1, ny + 1, nz), common + [km, t_v] + dims, (23,)),
        ("production", "wrf_smag_vd_w", (1, ny, nz + 1), common + [km, t_w] + dims, (23,)),
        ("production", "wrf_smag_surface_u", (1, ny, 1),
         common + [ustm, np.int32(1), t_u] + dims, (24,)),
        ("production", "wrf_smag_surface_v", (1, ny + 1, 1),
         common + [ustm, np.int32(1), t_v] + dims, (24,)),
        ("production", "wrf_smag_surface_scalars", (1, ny, 1),
         common + [hfx, qfx, np.int32(1), np.int32(1), t_s, t_q] + dims, (26, 27)),
        ("production", "wrf_smag_w_primitives", (1, ny + 1, nz + 1),
         common + [cached_what, cached_rdz, cached_zx, cached_zy] + dims, (22, 23, 24, 25)),
        ("production", "wrf_smag_hd_w_cached", (1, ny, nz + 1),
         common + [km, cached_what, cached_rdz, cached_zx, cached_zy, t_w] + dims, (27,)),
        ("production", "wrf_calc_n2", grid,
         common + [scalar, state.thb, np.int32(state.thb.ndim == 3), state.p,
                   qc, qi, np.int32(moist), np.int32(moist), bn2] + dims, (30,)),
        ("production", "wrf_smag3d_km", grid,
         common + [np.float32(cfg.c_s), np.float32(1 / 3), np.float32(5.0),
                   np.float32(cfg.mix_upper_bound), np.int32(terrain),
                   d11, d22, d12, bn2, km, kh, kmv, khv] + dims, (31, 32, 33, 34)),
        ("production", "wrf_tke_km", grid,
         common + [scalar, state.thb, np.int32(state.thb.ndim == 3), state.p, tke, bn2,
                   np.float32(0.15), np.float32(1 / 3), np.float32(5.0),
                   np.float32(cfg.mix_upper_bound), np.int32(terrain), np.float32(0.0),
                   km, kh, kmv, khv] + dims, (34, 35, 36, 37)),
        ("production", "wrf_smag_vd_s", grid,
         common + [scalar, state.thb, np.int32(1), np.int32(state.thb.ndim == 3), khv, t_s]
         + dims, (27,)),
        ("production", "wrf_smag_surface_u_cd0", (1, ny, 1),
         common + [np.float32(0.01), t_u] + dims, (23,)),
        ("production", "wrf_smag_surface_v_cd0", (1, ny + 1, 1),
         common + [np.float32(0.01), t_v] + dims, (23,)),
        ("production", "wrf_smag_surface_heat_const", (1, ny, 1),
         common + [np.float32(0.02), hfx, np.int32(1), t_s] + dims, (23, 25)),
    ]
    # A new entry must not bypass compiler-target replay coverage.
    declared = set(re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(',
                              sources()["direct_w"]))
    exercised = {call[1] for call in calls}
    if declared != exercised:
        raise AssertionError(f"Smag replay coverage differs: missing={declared - exercised}, extra={exercised - declared}")
    return calls


def replay_scenario(modules, scenario):
    import cupy as cp
    import numpy as np

    rows = []
    def cloned(arguments):
        clones, replay = {}, []
        for arg in arguments:
            if isinstance(arg, cp.ndarray):
                if id(arg) not in clones:
                    clones[id(arg)] = arg.copy()
                replay.append(clones[id(arg)])
            else:
                replay.append(arg)
        return replay

    for variant, kernel, grid, arguments, outputs in fixture_calls(scenario):
        input_hash = hashlib.sha256()
        for index, arg in enumerate(arguments):
            if isinstance(arg, cp.ndarray):
                input_hash.update(str((index, arg.dtype.str, arg.shape)).encode())
                input_hash.update(cp.asnumpy(arg).tobytes())
            else:
                input_hash.update(str((index, type(arg).__name__, arg)).encode())
        words = {}
        for target in sorted(modules):
            replay = cloned(arguments)
            modules[target][variant].get_function(kernel)(grid, (128, 1, 1), tuple(replay))
            cp.cuda.Device().synchronize()
            words[target] = [cp.asnumpy(replay[index]).view(np.uint32).copy() for index in outputs]
            if any(not np.all(np.isfinite(value.view(np.float32))) for value in words[target]):
                raise AssertionError(f"nonfinite output in {kernel} compiled for compute_{target}")
        reference = words[120]
        original = modules[120].get("blackwell_control")
        if original is not None:
            replay = cloned(arguments)
            original[variant].get_function(kernel)(grid, (128, 1, 1), tuple(replay))
            cp.cuda.Device().synchronize()
            original_words = [cp.asnumpy(replay[index]).view(np.uint32).copy() for index in outputs]
            counts = [int(np.count_nonzero(actual != expected))
                      for actual, expected in zip(reference, original_words)]
            rows.append({"kernel": kernel, "variant": variant, "compute_target": 120,
                         "comparison": "original_blackwell", "input_sha256": input_hash.hexdigest(),
                         "output_arg_indices": outputs, "different_words": counts,
                         "output_sha256": [digest(x.tobytes()) for x in reference],
                         "original_output_sha256": [digest(x.tobytes()) for x in original_words]})
        for target in sorted(words):
            counts = [int(np.count_nonzero(actual != expected))
                      for actual, expected in zip(words[target], reference)]
            rows.append({"kernel": kernel, "variant": variant, "compute_target": target,
                         "comparison": "compiler_target",
                         "input_sha256": input_hash.hexdigest(), "output_arg_indices": outputs,
                         "different_words": counts,
                         "output_sha256": [digest(x.tobytes()) for x in words[target]]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("compile", "replay"))
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--host-sm", type=int)
    parser.add_argument("--kernel-dir", type=Path,
                        help="Compose diagnostic sources from an isolated kernel directory")
    parser.add_argument("--base-kernel-dir", type=Path,
                        help="Compile the immutable original Blackwell source as an answer control")
    parser.add_argument("--diagnostic-targets", action="store_true",
                        help="Also report older targets without requiring their words to match")
    options = parser.parse_args(argv)
    if options.action == "compile":
        if options.host_sm is None:
            parser.error("--host-sm is required for CPU-only compilation")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["GPUWM_NO_LOCAL_GPU"] = "1"
        targets = DIAGNOSTIC_TARGETS + TARGETS if options.diagnostic_targets else TARGETS
        receipt = compile_modules(options.outdir, options.host_sm, targets, kernel_dir=options.kernel_dir,
                                  base_kernel_dir=options.base_kernel_dir)
        print(json.dumps({target: {name: row["status"] for name, row in variants.items()}
                          for target, variants in receipt["targets"].items()}, indent=2))
        good = all(row["status"] in ("linked", "unsupported")
                   for variants in receipt["targets"].values() for row in variants.values())
        good = good and all(row["status"] == "linked"
                           for target in TARGETS for row in receipt["targets"].get(str(target), {}).values())
        good = good and all(row["status"] == "linked" for row in
                           receipt.get("blackwell_control", {}).get("variants", {}).values())
        return 0 if good else 2
    modules, receipt = load_modules(options.outdir, kernel_dir=options.kernel_dir,
                                    base_kernel_dir=options.base_kernel_dir)
    rows = []
    for scenario in SCENARIOS:
        rows.append({"scenario": scenario, "kernels": replay_scenario(modules, scenario)})
    report = {"schema": "gpuwm-smag-cross-target-results-v1", "compile": receipt,
              "scenarios": rows, "different_words": sum(sum(row["different_words"])
                for case in rows for row in case["kernels"]
                if row["comparison"] == "original_blackwell" or row["compute_target"] in TARGETS)}
    report["diagnostic_different_words"] = sum(sum(row["different_words"])
        for case in rows for row in case["kernels"]
        if row["comparison"] == "compiler_target" and row["compute_target"] not in TARGETS)
    report["blackwell_changed_words"] = sum(sum(row["different_words"])
        for case in rows for row in case["kernels"] if row["comparison"] == "original_blackwell")
    report["blackwell_control_present"] = "blackwell_control" in receipt
    (options.outdir / "replay-receipt.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"scenarios": len(rows), "targets": sorted(modules),
                      "different_words": report["different_words"],
                      "diagnostic_different_words": report["diagnostic_different_words"],
                      "blackwell_changed_words": report["blackwell_changed_words"],
                      "blackwell_control_present": report["blackwell_control_present"]}, indent=2))
    return int(report["different_words"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
