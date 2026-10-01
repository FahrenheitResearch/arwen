"""The port's package name is a row of the bridge's binding table.

The port renamed its package from ``mpas_port`` to ``hexcore`` in its
0.2.0 package rename.  The spine already priced a leg from whichever
layout the tree holds, while the bridge imported only ``mpas_port``, so
a hexcore tree was bound, priced, and then stopped the worker at its
first import with "No module named 'mpas_port'".  These tests hold the
seam to one table: the package is found from the tree, every port import
goes through it, and a tree holding neither package is refused by name
before any port tool loads.
"""

from __future__ import annotations

import ast
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from gpuwm.cycle import mpas_bridge
from mpas_cycle_bridge import portbind
from mpas_cycle_bridge.portbind import (PORT_PACKAGES, PortBindingError,
                                        bind_port, port_package)

#: What the stand-in port's ``cuda_backend.require_cuda`` raises, so a
#: worker that got past its port imports says so in its status file.
CUDA_REACHED = "stand-in port: require_cuda reached"

_PROOF = textwrap.dedent("""
    DT_SECONDS = 120.0

    def _construct_device_stack(**kwargs):
        raise AssertionError("no device stack in this test")

    def _run_steps(**kwargs):
        raise AssertionError("no steps in this test")

    def fingerprint_execution_boundary(*args, **kwargs):
        return {}

    def require_frozen_execution_sources():
        return {"sha256": "stand-in"}

    def require_fingerprint_identity(*args, **kwargs):
        return None
""")

_FORECAST = textwrap.dedent("""
    import importlib.util
    from pathlib import Path

    _spec = importlib.util.spec_from_file_location(
        "stand_in_v841_proof",
        Path(__file__).with_name("run_cuda_v841_full_physics_x4.py"))
    proof = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(proof)

    def verify_forecast_authorities(paths):
        return {"stand_in": True}

    def prepare_forecast_host(paths, authority, start_time_text=None):
        return {"stand_in": True}
""")

_MESH_BINDING = textwrap.dedent("""
    def bind_mesh(proof, mesh, *, grid, static, forecast):
        return None
""")

_STATE = textwrap.dedent("""
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class PrognosticState:
        rho: Any
        rho_theta: Any
        rho_u: Any
        rho_w: Any
        scalars: Any
        time_seconds: float
""")

_DRIVER = textwrap.dedent("""
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class DrySavedDiagnostics:
        theta_m: Any
        exner: Any
        density_perturbation: Any
        rho_theta_perturbation: Any
        pressure_perturbation: Any
        normal_velocity: Any
        vertical_velocity: Any
""")

_CUDA_BACKEND = textwrap.dedent(f"""
    class KernelCache:
        pass

    def require_cuda(**kwargs):
        raise RuntimeError({CUDA_REACHED!r})
""")


def _stand_in_port(root: Path, package: str | None,
                   *, tools_explode: bool = False) -> Path:
    """A port tree with the three tools and, optionally, one package."""
    (root / "tools").mkdir(parents=True)
    (root / "src").mkdir()
    bomb = 'raise AssertionError("a port tool loaded")\n'
    for relative, body in ((portbind.PORT_PROOF_RELPATH, _PROOF),
                           (portbind.PORT_FORECAST_RELPATH, _FORECAST),
                           (portbind.PORT_MESH_BINDING_RELPATH,
                            _MESH_BINDING)):
        (root / relative).write_text(bomb if tools_explode else body,
                                     encoding="utf-8")
    if package is not None:
        tree = root / "src" / package
        (tree / "cuda_backend").mkdir(parents=True)
        for name, body in (("__init__.py", ""), ("state.py", _STATE),
                           ("driver.py", _DRIVER),
                           ("cuda_dualrun.py",
                            "def fingerprint_atmosphere(atmosphere):\n"
                            "    return {}\n"),
                           ("cuda_backend/__init__.py", _CUDA_BACKEND)):
            (tree / name).write_text(body, encoding="utf-8")
    return root


@pytest.fixture
def isolated_imports(monkeypatch):
    """Keep the stand-in port's modules and path entries out of the run."""
    for name in list(sys.modules):
        if name.split(".")[0] in PORT_PACKAGES:
            monkeypatch.delitem(sys.modules, name)
    saved_path = list(sys.path)
    before = set(sys.modules)
    yield
    sys.path[:] = saved_path
    for name in set(sys.modules) - before:
        sys.modules.pop(name, None)


def test_the_package_rows_are_the_port_layouts_newest_first():
    assert PORT_PACKAGES == ("hexcore", "mpas_port")


def test_the_port_package_is_the_first_row_the_tree_holds(tmp_path):
    root = tmp_path / "port"
    (root / "src").mkdir(parents=True)
    assert port_package(root) is None

    (root / "src" / "mpas_port").mkdir()
    (root / "src" / "mpas_port" / "__init__.py").write_text("")
    assert port_package(root) == "mpas_port"

    # A directory left behind with no package in it (git keeps untracked
    # __pycache__ directories across a checkout) is not a layout.
    (root / "src" / "hexcore" / "__pycache__").mkdir(parents=True)
    assert port_package(root) == "mpas_port"

    (root / "src" / "hexcore" / "__init__.py").write_text("")
    assert port_package(root) == "hexcore"


def test_bind_port_refuses_a_tree_holding_neither_package(tmp_path):
    """Refused by name before a port tool loads: the tools here raise."""
    root = _stand_in_port(tmp_path / "port", None, tools_explode=True)
    with pytest.raises(PortBindingError) as excinfo:
        bind_port(root)
    observed = excinfo.value.observed
    assert observed["looked_for"] == ["src/hexcore", "src/mpas_port"]
    assert observed["port_root"] == str(root)
    assert "none of the port packages" in str(excinfo.value)


@pytest.mark.parametrize("package", PORT_PACKAGES)
def test_every_port_import_goes_through_the_bound_package(
        tmp_path, package, isolated_imports):
    root = _stand_in_port(tmp_path / "port", package)
    binding = bind_port(root)
    assert binding.package == package

    rng = np.random.default_rng(3)
    prognostic = {name: rng.random((2, 3), dtype=np.float32)
                  for name in portbind.PROGNOSTIC_FIELDS}
    prognostic["time_seconds"] = np.asarray(240.0)
    state = portbind.prognostic_state(binding, prognostic)
    assert type(state).__module__ == f"{package}.state"
    assert state.time_seconds == 240.0

    derived = {name: rng.random((2, 3), dtype=np.float32)
               for name in portbind.SAVED_DIAGNOSTIC_FIELDS}
    saved = portbind.saved_diagnostics(binding, derived)
    assert type(saved).__module__ == f"{package}.driver"
    assert sys.modules[f"{package}.state"].__file__.startswith(str(root))


def test_no_bridge_module_spells_a_port_package_in_an_import():
    """RED ON REVERT: the bridge imported ``mpas_port`` by name in four
    places, and a hexcore tree stopped at the first one."""
    package = Path(portbind.__file__).resolve().parent
    offences = []
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names]
                     if isinstance(node, ast.Import)
                     else [node.module or ""]
                     if isinstance(node, ast.ImportFrom) else [])
            offences += [f"{module.name}:{node.lineno} {name}"
                         for name in names
                         if name.split(".")[0] in PORT_PACKAGES]
    assert not offences


def test_the_spine_prices_a_leg_from_the_same_package_rows():
    row = mpas_bridge.PARENT_DEVICE_FOOTPRINT["mpas-cuda"]
    assert row["admission_module"] == tuple(
        f"src/{package}/device_admission.py" for package in PORT_PACKAGES)


def test_the_leg_runner_reports_a_tree_with_neither_package_unavailable(
        tmp_path, capsys):
    """Exit 2, as for any port the leg cannot reach, naming both paths."""
    from gpuwm.cycle.anchor import write_anchor
    from tools.cycle_mpas_leg import main

    rng = np.random.default_rng(11)
    prognostic = {name: rng.random((4, 9)) for name in
                  ("rho", "rho_theta", "rho_u", "rho_w")}
    prognostic["scalars"] = rng.random((6, 4, 9))
    prognostic["time_seconds"] = np.asarray(0.0)
    write_anchor(tmp_path, cycle_index=0, anchor_ticks=0,
                 valid_time="2026-09-28T00:00:00Z", parent_kind="replay",
                 prognostic=prognostic,
                 derived={"exner": np.ones((4, 9))}, mesh_id="stand-in")
    port = _stand_in_port(tmp_path / "port", None, tools_explode=True)
    config = tmp_path / "port-config.json"
    config.write_text("{}", encoding="utf-8")

    code = main(["--root", str(tmp_path), "--backend", "mpas-cuda",
                 "--port-root", str(port), "--port-config", str(config),
                 "--port-steps", "1", "--cycle-seconds", "120"])
    err = capsys.readouterr().err
    assert code == 2
    assert "none of the port packages" in err
    # Named through the looked-in list, whose repr doubles a Windows
    # separator, so the package names are what is compared.
    assert "hexcore" in err and "mpas_port" in err


@pytest.mark.parametrize("package", PORT_PACKAGES)
def test_the_worker_gets_past_its_port_imports_in_either_layout(
        tmp_path, package):
    """RED ON REVERT: on a hexcore tree the worker stopped at
    ``from mpas_port.cuda_backend import ...`` with "No module named
    'mpas_port'"; through the binding it reaches the card check."""
    root = _stand_in_port(tmp_path / "port", package)
    arwen = tmp_path / "arwen"
    arwen.mkdir()
    config = tmp_path / "port-config.json"
    config.write_text(json.dumps({
        "arwen_checkout": str(arwen), "mesh": "stand-in",
        "grid": str(tmp_path / "grid.nc"),
        "static": str(tmp_path / "static.nc"),
        "init": str(tmp_path / "init.nc"),
        "cache_root": str(tmp_path / "cache"),
        "start_time": "2026-09-28_00:00:00"}), encoding="utf-8")

    result = mpas_bridge.launch(phase="seed", port_root=root,
                                port_config=config, steps=1,
                                out=tmp_path / "out", timeout=120,
                                check=False)
    assert result["ok"] is False
    error = result["status"]["error"]
    assert error == f"RuntimeError: {CUDA_REACHED}", error
