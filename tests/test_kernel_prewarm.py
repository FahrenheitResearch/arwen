"""The dycore's uncached diffusion compile runs during the input read.

The defect (NV recipe dry run, D5): on a warm kernel cache the first step of
NCAR's CONUS 12 km restart leg still took 22 to 37 s; a sampler put 32 s of
a 33.5 s step 1 inside ``compile_diffusion_source`` (smag2d and diff_opt1
through CuPy's direct, uncached NVRTC route).  The WRF door's worker now
starts those same loader calls on a background thread before the read.

CPU only: the loader and CuPy's device are stood in for.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from gpuwm import kernel_prewarm


def _experiment(**run):
    defaults = {"diff_opt": 1, "km_opt": 4, "diff_6th_opt": 0}
    defaults.update(run)
    return SimpleNamespace(domains=(SimpleNamespace(run=SimpleNamespace(**defaults)),))


def test_ncar_conus_namelist_warms_smag2d_and_diff_opt1():
    # NCAR's v4.4 CONUS benchmark: diff_opt = 1, km_opt = 4, no sixth order.
    assert kernel_prewarm.modules_for(_experiment()) == ("smag2d", "diff_opt1")


def test_diff_opt2_and_sixth_order():
    assert kernel_prewarm.modules_for(_experiment(diff_opt=2, diff_6th_opt=2)) == (
        "smag2d", "diff6")


def test_no_domain_no_module():
    assert kernel_prewarm.modules_for(SimpleNamespace(domains=())) == ()
    assert kernel_prewarm.modules_for(object()) == ()


def test_env_turns_it_off(monkeypatch):
    monkeypatch.setenv(kernel_prewarm.ENV, "0")
    assert kernel_prewarm.start_for(_experiment()) is None


def _fake_cupy(monkeypatch):
    devices = []

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            devices.append(self.index)
            return self

        def __exit__(self, *exc):
            return False

    cupy = types.ModuleType("cupy")
    cupy.cuda = SimpleNamespace(Device=Device,
                                runtime=SimpleNamespace(getDeviceCount=lambda: 1))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    return devices


def test_prewarm_calls_the_runs_own_loader_on_device_0(monkeypatch):
    devices = _fake_cupy(monkeypatch)
    from gpuwm.core import kernels

    loaded = []
    monkeypatch.setattr(kernels, "_load_diffusion_module", lambda name: loaded.append(name))
    receipt = kernel_prewarm.Prewarm(("smag2d", "diff_opt1")).start().join(30)
    assert loaded == ["smag2d", "diff_opt1"]
    assert devices == [0]
    assert receipt["errors"] == {} and set(receipt["seconds"]) == {"smag2d", "diff_opt1"}


def test_prewarm_never_builds_a_physics_parameter_module(monkeypatch):
    _fake_cupy(monkeypatch)
    from gpuwm import physics_params
    from gpuwm.core import kernels

    loaded = []
    monkeypatch.setattr(kernels, "_load_diffusion_module", lambda name: loaded.append(name))
    affected = sorted(physics_params._affected_modules())
    assert affected, "the registry names modules with parameter literals"
    monkeypatch.setattr(kernels, "_DIFFUSION_MODULES",
                        frozenset(kernels._DIFFUSION_MODULES | {affected[0]}))
    receipt = kernel_prewarm.Prewarm((affected[0], "smag2d")).start().join(30)
    assert loaded == ["smag2d"]
    assert "physics-parameter" in receipt["errors"][affected[0]]


def test_prewarm_refuses_a_unit_off_the_diffusion_route(monkeypatch):
    _fake_cupy(monkeypatch)
    from gpuwm.core import kernels

    loaded = []
    monkeypatch.setattr(kernels, "_load_diffusion_module", lambda name: loaded.append(name))
    receipt = kernel_prewarm.Prewarm(("thompson", "smag2d")).start().join(30)
    assert loaded == []
    assert "not diffusion-route units" in receipt["errors"]["*"]


def test_every_warmed_unit_is_a_diffusion_route_unit():
    from gpuwm.core import kernels

    assert set(kernel_prewarm.DIFFUSION_ROUTE_MODULES) <= set(kernels._DIFFUSION_MODULES)


def test_a_failed_compile_is_recorded_not_raised(monkeypatch):
    _fake_cupy(monkeypatch)
    from gpuwm.core import kernels

    def boom(name):
        raise RuntimeError("nvrtc said no")

    monkeypatch.setattr(kernels, "_load_diffusion_module", boom)
    receipt = kernel_prewarm.Prewarm(("smag2d",)).start().join(30)
    assert receipt["errors"]["smag2d"] == "RuntimeError: nvrtc said no"


def test_more_than_one_visible_card_is_not_warmed(monkeypatch):
    _fake_cupy(monkeypatch)
    sys.modules["cupy"].cuda.runtime.getDeviceCount = lambda: 2
    monkeypatch.setattr("gpuwm.local_gpu.no_local_gpu", lambda: False)
    assert kernel_prewarm.start_for(_experiment()) is None


def test_the_worker_starts_it_before_the_read():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "gpuwm" / "wrfinput_forecast.py"
            ).read_text(encoding="utf-8")
    start = text.index("kernel_prewarm.start_for(getattr(run, 'experiment', None))")
    read = text.index("inputs = prepare_wrf_run(run, worker_output/'input'")
    assert start < read


def test_the_diffusion_compile_keeps_its_image_in_the_persistent_cache():
    """compile_diffusion_source compiles through gpuwm.nvrtc_ptx_cache.

    CuPy's direct NVRTC entry point has no disk cache, so before this every
    run recompiled smag2d on step 1 even on a warm kernel cache.  The alias
    keeps the call text (and with it the FTZ route inventory row) as it was.
    """
    import ast
    from pathlib import Path

    from gpuwm.core import kernels

    tree = ast.parse(Path(kernels.__file__).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "compile_diffusion_source")
    bound = {alias.asname or alias.name: f"{node.module}.{alias.name}"
             for node in ast.walk(function) if isinstance(node, ast.ImportFrom)
             for alias in node.names}
    assert bound["compiler"] == "gpuwm.nvrtc_ptx_cache"


def test_composed_inventory_uses_runtime_factory_not_nonexistent_file(monkeypatch):
    selected = tuple(kernel_prewarm._COMPOSED_LOADERS)
    monkeypatch.setattr(kernel_prewarm, 'static_modules', lambda experiment: selected)
    monkeypatch.setattr(kernel_prewarm, '_inline_jobs', lambda modules: [])
    jobs = kernel_prewarm.static_jobs(_experiment())
    assert {name for name, factory, args in jobs} == set(selected)
    assert all(factory is kernel_prewarm._warm_composed and args == (name,)
               for name, factory, args in jobs)
    # Execute every dispatcher against a substituted factory at the real
    # module/function address; SW must receive its actual table object.
    import importlib
    calls = []
    tables = object()
    for name, (module, function) in kernel_prewarm._COMPOSED_LOADERS.items():
        target = importlib.import_module('gpuwm.core.' + module)
        monkeypatch.setattr(target, function, lambda *args, n=name: calls.append((n,args)))
        if name == 'rrtmg_sw_legacy':
            monkeypatch.setattr(target, '_sw_tables', lambda: tables)
        kernel_prewarm._warm_composed(name)
    assert calls == [(n, (tables,) if n == 'rrtmg_sw_legacy' else ()) for n in selected]
