"""Measure native acoustic column launches against compiled WRF v4.7.1.

The launcher selector records the grid and arguments assembled by
``prepare_acoustic_substep_launch`` and executes its real vertical kernel.
This isolates a routine without a second copy of the CUDA argument list.
The reference calls WRF's own ``calc_coef_w``, ``advance_w`` and
``calc_p_rho`` through their complete Fortran argument lists.

WRF carries theta minus 300 K; the engine carries full theta.  The bridge
records that float32 conversion explicitly instead of hiding its roundoff
inside a tolerance.  Nothing in this module decides an acceptance bound.
"""

from __future__ import annotations

from unittest.mock import patch
from functools import lru_cache
from dataclasses import replace

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance


def vertical_cases():
    """Six shared states plus flat, dry, lid, damper and periodic branches."""
    from gpuwm.verify.smallstep_oracle import load_cases
    cases = list(load_cases())
    yield from cases
    name, raw, metadata = cases[0]
    flat = {key: value.copy() for key, value in raw.items()}
    for key in tuple(flat):
        if key.startswith("MAPFAC_"):
            flat[key].fill(np.float32(1.0))
    flat["HGT"].fill(np.float32(0.0))
    yield "flat_unity_map", flat, dict(metadata, stress="unity map factors and zero terrain height; terrain 3D base retained")
    yield "dry_loading", raw, dict(metadata, moist_cq=False, stress="dry momentum loading")
    yield "rigid_lid", raw, dict(metadata, top_lid=True, stress="rigid top boundary")
    yield "implicit_damper", raw, dict(metadata, damp_opt=3, stress="implicit vertical Rayleigh damping")
    periodic = {key: value.copy() for key, value in raw.items()}
    periodic["U"][..., -1] = periodic["U"][..., 0]
    periodic["V"][:, -1, :] = periodic["V"][:, 0, :]
    for key in tuple(periodic):
        if key.startswith("MAPFAC_U"):
            periodic[key][:, -1] = periodic[key][:, 0]
        elif key.startswith("MAPFAC_V"):
            periodic[key][-1, :] = periodic[key][0, :]
    yield "periodic_rows", periodic, dict(metadata, specified=False, periodic=True,
                                     stress="periodic halo and physical edge rows")


def _field(raw, name, default=None):
    if name in raw:
        return np.asarray(raw[name], dtype=np.float32)
    if name.lower() in raw:
        return np.asarray(raw[name.lower()], dtype=np.float32)
    if default is None:
        raise KeyError(name)
    return np.asarray(default, dtype=np.float32)


def _vcoord(raw, name, nz, default):
    out = _field(raw, name, default).reshape(-1)
    if out.size == nz:
        out = np.concatenate((out, out[-1:]))
    if out.size != nz + 1:
        raise ValueError(f"{name} has {out.size} levels, expected {nz + 1}")
    return out


def make_vertical_state(raw, metadata):
    """Build a CUDA state from a real WRF crop and its case perturbations."""
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState

    t = _field(raw, "T")
    nz, ny, nx = t.shape
    cfg = RunConfig(nx=nx, ny=ny, nz=nz,
                    dx=float(metadata.get("dx", metadata.get("DX", 3000.0))),
                    dy=float(metadata.get("dy", metadata.get("DY", 3000.0))),
                    ztop=float(metadata.get("ztop", 20000.0)), dt=1.0,
                    run_seconds=1.0, moist=False,
                    specified=bool(metadata.get("specified", False)),
                    spec_zone=1, relax_zone=4, spec_bdy_width=5,
                    top_lid=bool(metadata.get("top_lid", False)),
                    damp_opt=int(metadata.get("damp_opt", 0)),
                    dampcoef=float(metadata.get("dampcoef", 0.2)),
                    zdamp=float(metadata.get("zdamp", 5000.0)))
    s = DomainState(cfg)
    put = lambda name, value: setattr(s, name, cp.asarray(value, dtype=cp.float32))
    put("thp", t)
    put("thb", np.full_like(t, np.float32(300.0)))
    put("php", _field(raw, "PH"))
    put("phb", _field(raw, "PHB"))
    put("p", _field(raw, "P") + _field(raw, "PB"))
    put("pb", _field(raw, "PB"))
    put("alt", _field(raw, "ALT", _field(raw, "AL") + _field(raw, "ALB")))
    put("mup", _field(raw, "MU"))
    put("mub2d", _field(raw, "MUB"))
    put("u", _field(raw, "U"))
    put("v", _field(raw, "V"))
    put("w", _field(raw, "W"))
    put("ht", _field(raw, "HGT", np.zeros((ny, nx), dtype=np.float32)))
    for name in ("c1h", "c2h", "c1f", "c2f", "rdn", "rdnw", "dnw", "fnm", "fnp"):
        defaults = np.ones(nz + 1, dtype=np.float32)
        if name in ("c2h", "c2f"):
            defaults.fill(0.0)
        put(name, _vcoord(raw, name.upper(), nz, defaults))
    for name in ("cf1", "cf2", "cf3"):
        value = metadata.get(name, raw.get(name.upper(),
                          {"cf1": 1.75, "cf2": -1.0, "cf3": 0.25}[name]))
        setattr(s, name, np.float32(np.asarray(value).reshape(-1)[0]))
    for name, shape in (("msft", (ny, nx)), ("msfu", (ny, nx + 1)),
                        ("msfv", (ny + 1, nx))):
        wrf_name = {"msft": "MAPFAC_M", "msfu": "MAPFAC_U", "msfv": "MAPFAC_V"}[name]
        put(name, _field(raw, wrf_name, np.ones(shape, dtype=np.float32)))
    s.has_msf = bool(np.any(cp.asnumpy(s.msft) != np.float32(1.0)))

    # Deterministic acoustic increments from the real meteorological state.
    # No independent pseudo-random atmosphere is introduced.
    mass = _field(raw, "MUB") + _field(raw, "MU")
    c1h = cp.asnumpy(s.c1h[:nz])[:, None, None]
    c2h = cp.asnumpy(s.c2h[:nz])[:, None, None]
    c1f = cp.asnumpy(s.c1f)[:, None, None]
    c2f = cp.asnumpy(s.c2f)[:, None, None]
    mh = c1h * mass + c2h
    mf = c1f * mass + c2f
    increment = np.float32(metadata.get("acoustic_scale", 0.001))
    mu = increment * _field(raw, "MU")
    put("mu_pp", mu)
    s.scratch((ny, nx), "acoustic_mu_pp_old")[...] = cp.asarray(np.float32(0.75) * mu)
    theta = np.float32(300.0) + t
    th = c1h * mu * theta + mh * (increment * t)
    put("th_pp", th)
    s.scratch((nz, ny, nx), "acoustic_th_pp_old")[...] = cp.asarray(np.float32(0.75) * th)
    put("ph_pp", increment * _field(raw, "PH"))
    put("w_pp", mf * (increment * _field(raw, "W")) / cp.asnumpy(s.msft))
    # Pressure-coordinate vertical flux is Pa/s; its magnitude is a small
    # fraction of the real vertical wind, with zero physical endpoint flux.
    omega = increment * _field(raw, "W")
    omega[[0, -1]] = 0.0
    put("ww_pp", omega)
    put("rw_t", mf * (np.float32(0.00001) * _field(raw, "W")))
    put("rph_t", np.float32(9.81) * mf * (np.float32(0.00001) * _field(raw, "W")))
    # These coupled momentum increments are used only by the terrain bottom
    # condition.  Face mass follows the native periodic/clamped conventions.
    forced = cfg.specified
    for field, axis, out in (("U", 1, "u_pp"), ("V", 0, "v_pp")):
        before = np.concatenate((mass[:1] if axis == 0 else mass[:, :1], mass), axis=axis)
        after = np.concatenate((mass, mass[-1:] if axis == 0 else mass[:, -1:]), axis=axis)
        if not forced:
            if axis == 0:
                before[0] = mass[-1]
                after[-1] = mass[0]
            else:
                before[:, 0] = mass[:, -1]
                after[:, -1] = mass[:, 0]
        face = np.float32(0.5) * (before + after)
        mface = c1h * face + c2h
        msf = cp.asnumpy(s.msfu if out == "u_pp" else s.msfv)
        put(out, increment * mface * _field(raw, field) / msf)
    return s, cfg


@lru_cache(maxsize=12)
def _diagnostic_module(source, no_fma, preserve_subnormals):
    import cupy as cp
    from gpuwm import wrf_exact
    from gpuwm.core.kernels import diffusion_kernel, function_options, module_options
    options = function_options("acoustic", "advance_w_phi", module_options("acoustic"))
    if wrf_exact.ENABLED:
        options = wrf_exact.effective_options(options)
    if no_fma and "--fmad=false" not in options:
        options += ("--fmad=false",)
    if preserve_subnormals and "--ftz=false" not in options:
        options += ("--ftz=false",)
    if diffusion_kernel("acoustic", "advance_w_phi") or preserve_subnormals:
        from cupy.cuda import compiler, function
        binary, _ = compiler.compile_using_nvrtc(source, options=options)
        module = function.Module()
        module.load(binary)
        return module
    return cp.RawModule(code=source, options=options)


def selected_vertical_launch(state, cfg, dtau, coefficients, *, no_fma=False, workspace=None,
                             wrf_phi_order=False, preserve_subnormals=False, wrf_top_order=False,
                             wrf_update_order=False, correct_pi=False, wrf_eos_order=False,
                             replay_pi_defect=False):
    """Execute the real vertical launch selected from native driver assembly."""
    from gpuwm.core import acoustic

    captured = []
    original_get = acoustic.get_kernel
    original_w = acoustic._w_phi_kernel

    def recorder(module, name):
        actual = original_get(module, name)

        def save(grid, block, args):
            captured.append((name, actual, grid, block, args))
        return save

    def record_w(name, nz):
        # _w_phi_kernel itself resolves get_kernel through this module.
        # Restore that lookup while obtaining the native callable, so a
        # selected launch cannot accidentally retain another recorder.
        with patch.object(acoustic, "get_kernel", original_get):
            actual = original_w(name, nz)
        if (no_fma or workspace is not None or wrf_phi_order or preserve_subnormals
                or wrf_top_order or wrf_update_order or correct_pi or wrf_eos_order or replay_pi_defect):
            source = acoustic.wphi_kernel_source(nz)
            if replay_pi_defect:
                if source.count("1.5707963267948966f") == 2:
                    source = source.replace("1.5707963267948966f", "1.5707963f")
                elif source.count("1.5707963f") != 2:
                    raise ValueError("the two native damper pi literals changed")
            if correct_pi:
                if source.count("1.5707963f") == 2:
                    source = source.replace("1.5707963f", "1.5707963267948966f")
                elif source.count("1.5707963267948966f") != 2:
                    raise ValueError("the two native damper pi literals changed")
            if wrf_phi_order:
                from tools.smallstep_wrf471_oracle.vertical_phi_order import wrf_phi_order_source
                source = wrf_phi_order_source(source)
            if wrf_update_order:
                from tools.smallstep_wrf471_oracle.vertical_update_order import wrf_update_order_source
                source = wrf_update_order_source(source)
            if wrf_top_order:
                from tools.smallstep_wrf471_oracle.vertical_top_order import wrf_top_order_source
                source = wrf_top_order_source(source)
            if wrf_eos_order:
                from tools.smallstep_wrf471_oracle.vertical_eos_order import wrf_eos_order_source
                source = wrf_eos_order_source(source)
            if workspace is not None:
                from tools.smallstep_wrf471_oracle.vertical_workspace import workspace_source
                source = workspace_source(source)
            actual = _diagnostic_module(source, no_fma, preserve_subnormals).get_function(name)

        def save(grid, block, args):
            captured.append((name, actual, grid, block, args))
        return save

    with patch.object(acoustic, "get_kernel", recorder), \
            patch.object(acoustic, "_w_phi_kernel", record_w):
        launch = acoustic.prepare_acoustic_substep_launch(state, cfg, dtau, coefficients)
        launch(first=True)
    selected = [entry for entry in captured if entry[0] in
                ("advance_w_phi", "advance_w_phi_msf")]
    if len(selected) != 1:
        raise RuntimeError(f"native launcher selected {len(selected)} vertical kernels")
    name, kernel, grid, block, args = selected[0]
    if workspace is not None:
        position = next(i for i, value in enumerate(args) if value is state.al_pp) + 1
        args = args[:position] + tuple(workspace) + args[position:]
    kernel(grid, block, args)
    return name


def _host(state, name):
    import cupy as cp
    return cp.asnumpy(getattr(state, name))


def _measure(got, expected):
    got = np.ascontiguousarray(got, dtype=np.float32)
    expected = np.ascontiguousarray(expected, dtype=np.float32)
    if got.shape != expected.shape:
        raise ValueError(f"output shape {got.shape} != oracle {expected.shape}")
    finite = np.isfinite(got) & np.isfinite(expected)
    nonfinite_equal = np.array_equal(got[~finite].view(np.uint32),
                                     expected[~finite].view(np.uint32))
    ulp = fp32_ulp_distance(got[finite], expected[finite])
    words = got.view(np.uint32) != expected.view(np.uint32)
    return dict(max_ulp=int(ulp.max(initial=0)), differing_words=int(words.sum()),
                words=int(got.size), nonfinite_words=int((~finite).sum()),
                nonfinite_equal=bool(nonfinite_equal),
                max_abs=float(np.abs(got[finite].astype(np.float64)
                                     - expected[finite].astype(np.float64)).max(initial=0)))


def _vertical_coefficients(s, cfg, raw, metadata, *, no_fma=False, preserve_subnormals=False):
    """Prepare native coefficients and identically supplied moisture loading."""
    import cupy as cp
    from gpuwm.core import acoustic
    dtau = float(metadata.get("dtau", 0.5))
    moist_cq = bool(metadata.get("moist_cq", True))
    cqw_host = np.ones((cfg.nz + 1, cfg.ny, cfg.nx), dtype=np.float32)
    if moist_cq:
        qv = _field(raw, "QVAPOR", np.zeros_like(_field(raw, "T")))
        qsum = qv.copy()
        for name in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
            qsum += _field(raw, name, np.zeros_like(qsum))
        # Moisture loading at the w staggering is supplied identically to
        # both routines; calc_cq is compared by the moisture oracle lane.
        cqw_host[1:-1] = np.float32(1.0) / (np.float32(1.0)
                                                  + np.float32(0.5) * (qsum[:-1] + qsum[1:]))
        cqw_host[0] = np.float32(1.0) / (np.float32(1.0) + qsum[0])
        cqw_host[-1] = np.float32(1.0) / (np.float32(1.0) + qsum[-1])
    cq = (s.p, s.p, cp.asarray(cqw_host), moist_cq)
    if no_fma or preserve_subnormals:
        native_get = acoustic.get_kernel
        diagnostic_module = _diagnostic_module(acoustic.wphi_kernel_source(cfg.nz),
                                                no_fma, preserve_subnormals)
        def no_fma_get(module, name):
            return diagnostic_module.get_function(name) if module == "acoustic" else native_get(module, name)
        with patch.object(acoustic, "get_kernel", no_fma_get):
            coefficients = acoustic.prepare_acoustic_coefficients(s, cfg, dtau, cq=cq)
    else:
        coefficients = acoustic.prepare_acoustic_coefficients(s, cfg, dtau, cq=cq)
    return coefficients, cqw_host


def _pressure_history_words(p, pold, smdiv, *, no_fma=False, preserve_subnormals=False):
    """Expose the actual native pdmp helper, also consumed by advance_uv."""
    import cupy as cp
    from gpuwm.core.kernels import module_source
    source = module_source("acoustic") + """
extern "C" __global__ void oracle_pdmp_words(
    const real* p, const real* pold, real* out, real smdiv, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) out[index] = pdmp(p, pold, (size_t)index, smdiv);
}
"""
    kernel = _diagnostic_module(source, no_fma, preserve_subnormals).get_function("oracle_pdmp_words")
    p_device, old_device = cp.asarray(p), cp.asarray(pold)
    output = cp.empty_like(p_device)
    kernel(((p_device.size + 255) // 256,), (256,),
           (p_device, old_device, output, np.float32(smdiv), np.int32(p_device.size)))
    return cp.asnumpy(output)


def vertical_port_outputs(raw, metadata, *, reference_inputs=None):
    """Run the native path against packaged reference arrays, without Fortran."""
    import cupy as cp
    s, cfg = make_vertical_state(raw, metadata)
    dtau = float(metadata.get("dtau", 0.5))
    coefficients, _ = _vertical_coefficients(s, cfg, raw, metadata)
    selected_vertical_launch(s, cfg, dtau, coefficients)
    witness, witness_cfg = make_vertical_state(raw, metadata)
    old_th = cp.asnumpy(witness.scratch((cfg.nz, cfg.ny, cfg.nx), "acoustic_th_pp_old"))
    old_mu = cp.asnumpy(witness.scratch((cfg.ny, cfg.nx), "acoustic_mu_pp_old"))
    c1h = _host(witness, "c1h")[:cfg.nz, None, None]
    theta_offset = np.float32(metadata.get("theta_offset", 300.0))
    t2 = cp.asarray(old_th - (c1h * old_mu) * theta_offset)
    mu = cp.zeros((2, cfg.ny, cfg.nx), dtype=cp.float32)
    selected_vertical_launch(witness, witness_cfg, dtau, coefficients, workspace=(t2, mu))
    for name in ("w_pp", "ph_pp", "p_pp", "al_pp"):
        if _measure(_host(witness, name), _host(s, name))["differing_words"]:
            raise RuntimeError(f"workspace witness altered native {name}")
    diagnosis, diagnosis_cfg = make_vertical_state(raw, metadata)
    diagnosis_cfg = replace(diagnosis_cfg, top_lid=False, damp_opt=0)
    selected_vertical_launch(diagnosis, diagnosis_cfg, 0.0, coefficients)
    mask = np.ones((cfg.ny, cfg.nx), dtype=bool)
    if cfg.specified:
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False
    outputs = {name: cp.asnumpy(array) for name, array in
               zip(("c2a", "a", "alpha", "gamma"), coefficients[:4], strict=True)}
    outputs.update({key: _host(s, field) for key, field in (("w", "w_pp"), ("ph", "ph_pp"))})
    outputs.update({key: _host(diagnosis, field) for key, field in
                   (("p", "p_pp"), ("al", "al_pp"), ("ph_diag", "ph_pp"))})
    outputs.update(t2=cp.asnumpy(t2), muts=cp.asnumpy(mu)[0][mask],
                   muave=cp.asnumpy(mu)[1][mask])
    if reference_inputs is not None:
        outputs["history"] = _pressure_history_words(reference_inputs["history_p"],
                                                       reference_inputs["history_old"], cfg.smdiv)
    return outputs


def measure_vertical_case(raw, metadata, library_path, *, no_fma=False, output_arrays=None,
                          wrf_phi_order=False, preserve_subnormals=False, wrf_top_order=False,
                          wrf_update_order=False, correct_pi=False, wrf_eos_order=False,
                          replay_pi_defect=False):
    """Return every observable vertical output and its exact word distances."""
    import cupy as cp

    from gpuwm.verify.smallstep_oracle import WRFOracle

    s, cfg = make_vertical_state(raw, metadata)
    dtau = float(metadata.get("dtau", 0.5))
    oracle = WRFOracle(library_path, cfg.nx, cfg.ny, cfg.nz,
                       periodic=not cfg.specified)
    scalar = lambda x: np.float32(x)
    vectors = {name: oracle.array(_host(s, name)) for name in
               ("c1h", "c2h", "c1f", "c2f", "rdn", "rdnw", "dnw", "fnm", "fnp")}
    for name in ("c3h", "c4h", "c3f", "c4f"):
        vectors[name] = oracle.array(_vcoord(raw, name.upper(), cfg.nz,
                                            np.zeros(cfg.nz + 1, dtype=np.float32)))
    coefficients, cqw_host = _vertical_coefficients(s, cfg, raw, metadata, no_fma=no_fma,
                                                    preserve_subnormals=preserve_subnormals)
    c2a, a, alpha, gamma = coefficients[:4]
    shape_w = (cfg.nz + 1, cfg.ny, cfg.nx)
    zeros_w = np.zeros(shape_w, dtype=np.float32)
    coef_args = dict(a=oracle.array(zeros_w), alpha=oracle.array(zeros_w),
                     gamma=oracle.array(zeros_w),
                     mut=oracle.array(_host(s, "mub2d") + _host(s, "mup")),
                     cqw=oracle.array(cqw_host), c2a=oracle.array(cp.asnumpy(c2a)),
                     **{name: vectors[name] for name in
                        ("c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f", "rdn", "rdnw")},
                     dts=scalar(dtau), g=scalar(9.81), epssm=scalar(cfg.epssm),
                     top_lid=cfg.top_lid)
    oracle.call("calc_coef_w", **coef_args)
    results = {"calc_coef_w": {name: _measure(cp.asnumpy(device),
                 oracle.extract(coef_args[wrf], shape_w))
                for name, wrf, device in (("a", "a", a), ("alpha", "alpha", alpha),
                                          ("gamma", "gamma", gamma))}}

    before = {name: _host(s, name) for name in
              ("w_pp", "ph_pp", "th_pp", "mu_pp", "thp", "thb", "php", "phb", "alt",
               "mup", "mub2d", "u_pp", "v_pp", "rw_t", "rph_t", "ww_pp", "w", "ht", "msft")}
    old_mu = cp.asnumpy(s.scratch((cfg.ny, cfg.nx), "acoustic_mu_pp_old"))
    old_th = cp.asnumpy(s.scratch((cfg.nz, cfg.ny, cfg.nx), "acoustic_th_pp_old"))
    muave = scalar(0.5) * ((scalar(1) + scalar(cfg.epssm)) * before["mu_pp"]
                              + (scalar(1) - scalar(cfg.epssm)) * old_mu)
    c1h = _host(s, "c1h")[:cfg.nz, None, None]
    theta_offset = scalar(metadata.get("theta_offset", 300.0))
    t1 = before["thp"] if theta_offset == scalar(300) else \
        (before["thb"] + before["thp"]) - theta_offset
    t2 = before["th_pp"] - (c1h * before["mu_pp"]) * theta_offset
    t2old = old_th - (c1h * old_mu) * theta_offset
    mut = before["mub2d"] + before["mup"]
    wargs = dict(w=oracle.array(before["w_pp"]), ph=oracle.array(before["ph_pp"]),
                 t_2=oracle.array(t2), t_2ave=oracle.array(t2old),
                 t_1=oracle.array(t1),
                 rw_tend=oracle.array(before["rw_t"]), ww=oracle.array(before["ww_pp"]),
                 w_save=oracle.array(before["w"]),
                 u=oracle.array(before["u_pp"]), v=oracle.array(before["v_pp"]),
                 mu1=oracle.array(before["mup"]), mut=oracle.array(mut),
                 muave=oracle.array(muave), muts=oracle.array(mut + before["mu_pp"]),
                 ph_1=oracle.array(before["php"]), phb=oracle.array(before["phb"]),
                 ph_tend=oracle.array(before["rph_t"]), ht=oracle.array(before["ht"]),
                 c2a=oracle.array(cp.asnumpy(c2a)), cqw=oracle.array(cqw_host),
                 alt=oracle.array(before["alt"]), alb=oracle.array(before["alt"]),
                 a=oracle.array(cp.asnumpy(a)), alpha=oracle.array(cp.asnumpy(alpha)),
                 gamma=oracle.array(cp.asnumpy(gamma)), **vectors,
                 rdx=scalar(1.0 / cfg.dx), rdy=scalar(1.0 / cfg.dy),
                 dts=scalar(dtau), t0=theta_offset, epssm=scalar(cfg.epssm),
                 cf1=s.cf1, cf2=s.cf2, cf3=s.cf3,
                 msftx=oracle.array(before["msft"]), msfty=oracle.array(before["msft"]),
                 config_flags=dict(periodic_x=not cfg.specified, specified=cfg.specified,
                                   nested=False, phi_adv_z=1, damp_opt=cfg.damp_opt,
                                   dampcoef=cfg.dampcoef, zdamp=cfg.zdamp),
                 top_lid=cfg.top_lid)
    oracle.call("advance_w", **wargs)
    selected_vertical_launch(s, cfg, dtau, coefficients, no_fma=no_fma,
                              wrf_phi_order=wrf_phi_order, preserve_subnormals=preserve_subnormals,
                              wrf_top_order=wrf_top_order, wrf_update_order=wrf_update_order,
                              correct_pi=correct_pi, wrf_eos_order=wrf_eos_order,
                              replay_pi_defect=replay_pi_defect)
    results["advance_w"] = {name: _measure(_host(s, field),
                           oracle.extract(wargs[name], shape_w))
                            for name, field in (("w", "w_pp"), ("ph", "ph_pp"))}
    witness, witness_cfg = make_vertical_state(raw, metadata)
    # The unvisited frame retains the same original WRF workspace words.
    t2_workspace = cp.asarray(t2old, dtype=cp.float32)
    mu_workspace = cp.zeros((2, cfg.ny, cfg.nx), dtype=cp.float32)
    selected_vertical_launch(witness, witness_cfg, dtau, coefficients, no_fma=no_fma,
                              workspace=(t2_workspace, mu_workspace), wrf_phi_order=wrf_phi_order,
                              preserve_subnormals=preserve_subnormals, wrf_top_order=wrf_top_order,
                              wrf_update_order=wrf_update_order, correct_pi=correct_pi,
                              wrf_eos_order=wrf_eos_order, replay_pi_defect=replay_pi_defect)
    for field in ("w_pp", "ph_pp", "p_pp", "al_pp"):
        metric = _measure(_host(witness, field), _host(s, field))
        if metric["differing_words"]:
            raise RuntimeError(f"workspace instrumentation altered native {field}: {metric}")
    t2_expected = oracle.extract(wargs["t_2ave"], (cfg.nz, cfg.ny, cfg.nx))
    # WRF leaves physical boundary rows as the original coupled old theta;
    # they are outside the native column solve's workspace as well.
    if cfg.specified:
        workspace_mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
        workspace_mask[1:-1, 1:-1] = True
    else:
        workspace_mask = np.ones((cfg.ny, cfg.nx), dtype=bool)
    results["advance_w"]["t_2ave"] = _measure(cp.asnumpy(t2_workspace), t2_expected)
    results["advance_w"]["muts_workspace"] = _measure(cp.asnumpy(mu_workspace)[0][workspace_mask],
                                                     (mut + before["mu_pp"])[workspace_mask])
    results["advance_w"]["muave_workspace"] = _measure(cp.asnumpy(mu_workspace)[1][workspace_mask],
                                                      muave[workspace_mask])

    # A zero-time native vertical launch diagnoses the same initial fields
    # on every device.  This isolates the EOS from advance_w's separately
    # measured rounding.  Its open top keeps every initial phi word intact.
    diagnosis, diagnosis_cfg = make_vertical_state(raw, metadata)
    diagnosis_cfg = replace(diagnosis_cfg, top_lid=False, damp_opt=0)
    selected_vertical_launch(diagnosis, diagnosis_cfg, 0.0, coefficients,
                              no_fma=no_fma, preserve_subnormals=preserve_subnormals,
                              wrf_phi_order=wrf_phi_order, wrf_top_order=wrf_top_order,
                              wrf_update_order=wrf_update_order, correct_pi=correct_pi,
                              wrf_eos_order=wrf_eos_order, replay_pi_defect=replay_pi_defect)
    pshape = (cfg.nz, cfg.ny, cfg.nx)
    zero_h = np.zeros(pshape, dtype=np.float32)
    pargs = dict(al=oracle.array(zero_h), p=oracle.array(zero_h),
                 ph=oracle.array(before["ph_pp"]), pm1=oracle.array(zero_h),
                 alt=oracle.array(before["alt"]), t_2=oracle.array(t2),
                 t_1=oracle.array(t1),
                 c2a=oracle.array(cp.asnumpy(c2a)), mu=oracle.array(before["mu_pp"]),
                 mut=oracle.array(mut + before["mu_pp"]), **{name: vectors[name]
                    for name in ("c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f", "rdnw", "dnw")},
                 znu=oracle.array(_vcoord(raw, "ZNU", cfg.nz, np.zeros(cfg.nz + 1))),
                 t0=theta_offset, smdiv=scalar(0), non_hydrostatic=True, step=0)
    if cfg.specified:
        pargs.update(its=2, ite=cfg.nx - 1, jts=2, jte=cfg.ny - 1)
    oracle.call("calc_p_rho", **pargs)
    results["calc_p_rho"] = {name: _measure(_host(diagnosis, field),
                             oracle.extract(pargs[wrf], pshape))
        for name, field, wrf in (("p", "p_pp", "p"), ("al", "al_pp", "al"))}
    results["calc_p_rho"]["ph_unchanged"] = _measure(_host(diagnosis, "ph_pp"),
                                                   oracle.extract(pargs["ph"], shape_w))
    results["calc_p_rho"]["reference_pm1_copy"] = _measure(oracle.extract(pargs["p"], pshape),
                                                oracle.extract(pargs["pm1"], pshape))
    history_p = oracle.extract(pargs["p"], pshape)
    history_old = scalar(0.75) * history_p
    history_args = dict(pargs, p=oracle.array(zero_h), al=oracle.array(zero_h),
                         pm1=oracle.array(history_old), step=1, smdiv=scalar(cfg.smdiv))
    oracle.call("calc_p_rho", **history_args)
    history_native = _pressure_history_words(history_p, history_old, cfg.smdiv,
                                             no_fma=no_fma, preserve_subnormals=preserve_subnormals)
    results["calc_p_rho"]["history_weighted_p"] = _measure(history_native,
                                                       oracle.extract(history_args["p"], pshape))
    results["calc_p_rho"]["reference_pm1_history_copy"] = _measure(history_p,
                                                   oracle.extract(history_args["pm1"], pshape))
    if output_arrays is not None:
        for key, array in (("a", a), ("alpha", alpha), ("gamma", gamma)):
            output_arrays[key + "_native"] = cp.asnumpy(array)
            output_arrays[key + "_wrf"] = oracle.extract(coef_args[key], shape_w)
        output_arrays["c2a_native"] = cp.asnumpy(c2a)
        for key, field, args, wrf, shape in (
                ("w", "w_pp", wargs, "w", shape_w),
                ("ph", "ph_pp", wargs, "ph", shape_w),
                ("p", "p_pp", pargs, "p", pshape),
                ("al", "al_pp", pargs, "al", pshape)):
            output_arrays[key + "_native"] = _host(diagnosis if key in ("p", "al") else s, field)
            output_arrays[key + "_wrf"] = oracle.extract(args[wrf], shape)
        output_arrays["t2_native"] = cp.asnumpy(t2_workspace)
        output_arrays["t2_wrf"] = t2_expected
        output_arrays["muts_native"] = cp.asnumpy(mu_workspace)[0][workspace_mask]
        output_arrays["muts_wrf"] = (mut + before["mu_pp"])[workspace_mask]
        output_arrays["muave_native"] = cp.asnumpy(mu_workspace)[1][workspace_mask]
        output_arrays["muave_wrf"] = muave[workspace_mask]
        output_arrays["ph_diag_wrf"] = oracle.extract(pargs["ph"], shape_w)
        output_arrays["pm1_wrf"] = oracle.extract(pargs["pm1"], pshape)
        output_arrays["history_p_input"] = history_p
        output_arrays["history_old_input"] = history_old
        output_arrays["history_native"] = history_native
        output_arrays["history_wrf"] = oracle.extract(history_args["p"], pshape)
        output_arrays["pm1_history_wrf"] = oracle.extract(history_args["pm1"], pshape)
    return results
