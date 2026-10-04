"""Nonzero SPP against independent, unmodified WRF v4.6.1 modules.

The oracle builder supplies patterns at Fortran interfaces. It never edits
the reference physics. GF uses CUP_gf because stock GFDRV drops rand_clos.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from gpuwm.core.fp32_ulp import fp32_ulp_distance
import test_mynn_pbl as pbl
import test_mynn_surface_water as surface

DATA = Path(__file__).resolve().parents[1] / "gpuwm" / "data" / "spp"
NATIVE_ZIP_SHA256 = "9bc14a88c46b54547a6a19000830a3f4028fbce4b755f5bfb2648e853e4d0f4b"


@pytest.fixture(scope="session")
def native_directory(tmp_path_factory):
    directory = os.environ.get("GPUWM_SPP_ORACLE")
    if directory:
        return Path(directory)
    archive = DATA / "native-wrf461.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == NATIVE_ZIP_SHA256
    target = tmp_path_factory.mktemp("spp-native")
    with zipfile.ZipFile(archive) as source:
        assert all(Path(name).name == name for name in source.namelist())
        source.extractall(target)
    receipt = json.loads((target / "receipt.json").read_text())
    for path in target.glob("*.csv"):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["files"][path.name]["sha256"]
    return target


@pytest.fixture
def oracle(monkeypatch, native_directory):
    root = native_directory
    for attr, filename in [("TURBULENCE_ORACLE", "turbulence-spp.csv"),
                           ("CONDENSATION_ORACLE", "condensation-spp.csv"),
                           ("DMP_MF_ORACLE", "dmp_mf-spp.csv"),
                           ("DRIVER_ORACLE", "driver-spp.csv")]:
        monkeypatch.setattr(pbl, attr, root / filename)
    return root


@pytest.fixture
def backend():
    from gpuwm.core import mynn_pbl
    return "cpu", mynn_pbl, np.asarray, surface.mynn_surface_layer_default, np.asarray


def _call(backend, name, values, **kwargs):
    kind, module, host = backend[:3]
    result = getattr(module, name + ("_cuda" if kind == "gpu" else ""))(values, **kwargs)
    def get(key):
        return host(result[key] if isinstance(result, dict) else getattr(result, key))
    return get


def _measure(actual, expected):
    return {name: int(fp32_ulp_distance(actual[name], value).max(initial=0))
            for name, value in expected.items()}


def _gate(label, actual, expected):
    metrics = _measure(actual, expected)
    print("SPP_NATIVE " + json.dumps({"name": label, "ulp": {k:v for k,v in metrics.items() if v}, "fields": len(metrics)}, sort_keys=True))
    # The whole-column ports already carry named arithmetic residuals.
    # Store one measured bound per element: an exactly matching element has
    # no slack, and an inherited error in one cloud cannot hide a change in
    # another column. The new DMP and diffusivity operators are exact.
    with np.load(DATA / (label.split(".",1)[0] + "-ulp.npz"), allow_pickle=False) as budget:
        for name,value in expected.items():
            distance = fp32_ulp_distance(actual[name],value)
            bound = budget[label+"/"+name]
            assert distance.shape == bound.shape
            assert np.isfinite(actual[name]).all(), (label,name)
            assert np.all(distance <= bound), (label,name,metrics[name],int(bound.max(initial=0)))


def test_mynn_turbulence_spp_native(oracle, backend):
    _, fields = pbl._turbulence_oracle()
    inputs = {name: fields[name] for name in pbl.MYNN_TURBULENCE_INPUTS}
    for name in ("xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl", "psig_shcu"):
        inputs[name] = fields[name][:, 0]
    inputs["rstoch"] = fields["rstoch"]
    get = _call(backend, "mynn_turbulence_default", inputs, spp_pbl=1)
    actual = {name: get(name) for name in pbl.TURBULENCE_OUTPUTS}
    expected = {name: fields[name] for name in actual}
    _gate(backend[0] + ".turbulence", actual, expected)


def test_mynn_condensation_spp_native(oracle, backend):
    _, fields = pbl._condensation_oracle()
    get = _call(backend, "mynn_condensation_default", pbl._condensation_inputs(fields), spp_pbl=1)
    names = ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm")
    expected = {name: fields[name + ("_after" if name in ("vt", "vq", "sgm") else "")]
                for name in names}
    _gate(backend[0] + ".condensation", {name: get(name) for name in names}, expected)


def test_mynn_mass_flux_spp_native(oracle, backend):
    rows = pbl._dmp_mf_oracle()
    actual, expected = {}, {}
    for case in pbl.DMP_MF_CASES:
        selected = pbl._dmp_mf_case(rows, case)
        get = _call(backend, "mynn_dmp_mf", pbl._dmp_mf_inputs(selected), spp_pbl=1)
        for name, value in pbl._dmp_mf_expected(selected).items():
            got = get(name)
            key = case + "/" + name
            actual[key] = got[0] if got.ndim == 2 else got
            expected[key] = value
    _gate(backend[0] + ".mass_flux", actual, expected)


@pytest.mark.parametrize("step", [1, 2])
def test_mynn_driver_spp_native(oracle, backend, step):
    blocks, values, initflag, delt = pbl._driver_step(step)
    values["rstoch"] = np.broadcast_to(
        np.float32(.1) * (np.arange(5, dtype=np.float32)[:, None] - np.float32(2)),
        (5, pbl.DRIVER_NZ)).copy()
    get = _call(backend, "mynn_bl_driver", values, initflag=initflag, delt=delt,
                flag_qs=True, spp_pbl=1)
    expected = {name: np.asarray([[np.float32(row[pbl.DRIVER_OUTPUT_CSV.get(name, name)])
                                  for row in block] for block in blocks])
                for name in pbl.DRIVER_PROFILE_OUTPUTS}
    expected.update({name: np.asarray([np.float32(block[0][name]) for block in blocks])
                     for name in ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume")})
    for name in ("kpbl", "ktop_plume"):
        np.testing.assert_array_equal(get(name).reshape(-1), [int(block[0][name]) for block in blocks])
    actual = {name: get(name).reshape(value.shape) for name, value in expected.items()}
    _gate(f"{backend[0]}.driver.step{step}", actual, expected)


def test_mynn_surface_spp_native(oracle, backend):
    with (oracle / "surface-spp.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    actual, expected = {}, {}
    for flux in range(4):
        for step in (1, 2):
            selected = [r for r in rows if int(r["isftcflx"]) == flux and int(r["itimestep"]) == step]
            def field(name):
                return np.asarray([[np.float32(row[name]) for row in selected]])
            values = {name: field(surface.INPUT_ALIASES.get(name, name)) for name in surface.INPUT_NAMES}
            kwargs = dict(dx=float(selected[0]["dx"]), itimestep=step, isfflx=1, isftcflx=flux,
                          mol=field("mol_input"), ustm=field("ustm_input"), spp_pbl=1,
                          pattern_spp_pbl=field("rstoch"))
            kwargs["pattern_spp_pbl"] = backend[4](kwargs["pattern_spp_pbl"])
            if backend[0] == "cpu":
                values = {key: value.ravel() for key,value in values.items()}
                for key in ("mol", "ustm", "pattern_spp_pbl"):
                    kwargs[key] = kwargs[key].ravel()
            result = backend[3](values, **kwargs)
            for name in surface.OUTPUT_NAMES:
                key = f"{flux}.{step}.{name}"
                actual[key] = backend[2](result[name] if isinstance(result, dict) else getattr(result, name)).reshape(1,-1)
                expected[key] = field(name)
    _gate(backend[0] + ".surface", actual, expected)


def test_spp_specializations_leave_default_compiler_sources_frozen():
    from gpuwm.core.kernels import module_source
    from gpuwm.core.spp_kernel_sources import specialized_source
    pins = {"gf": "53785cdbe6d07c950bb4d8017ecf84a83d0f3bc84b81875e96bcda14ee8d9191",
            "mynn_pbl": "8771aadac8ba94ec75772d616db2acb0fd8fd768fba4b032595c3d07185fa6b2",
            "mynn_surface": "698e7f36bac844c7727b2fee5f15b5673276c87bfb4b36edbf367b0369aa87a4"}
    for name, pin in pins.items():
        baseline = module_source(name)
        assert hashlib.sha256(baseline.encode()).hexdigest() == pin
        assert specialized_source(name) != baseline
        assert module_source(name) == baseline


def test_enabled_spp_compiler_inputs_are_in_the_literal_division_census():
    from tools.literal_division_census import production_units
    from gpuwm.core.spp_kernel_sources import specialized_source
    units = {unit.key: unit for unit in production_units(DATA.parents[2])}
    for name,capacity in (("gf",40),("gf",60),("mynn_pbl",40),("mynn_surface",40)):
        assert units[f"spp:{name}[capacity={capacity}]"].source == specialized_source(name,capacity=capacity)
    assert "kernels:ruc_spp" in units


def test_native_spp_controls_exercise_the_parameter_sites(oracle):
    def rows(name):
        with (oracle / name).open(newline="") as stream:
            return list(csv.DictReader(stream))
    def values(data, field):
        return np.asarray([row[field] for row in data], dtype=np.float32)
    for stem, field in (("turbulence", "dfm"), ("condensation", "cldfra"),
                        ("dmp_mf", "edmf_ent"), ("surface", "hfx")):
        on, off = rows(stem + "-spp.csv"), rows(stem + "-off.csv")
        assert np.any(values(on,field) != values(off,field)), (stem,field)
        if stem == "turbulence":
            np.testing.assert_array_equal(values(on,"dfq"), values(off,"dfq"))
        if stem == "surface":
            first = [k for k,row in enumerate(on) if int(row["itimestep"]) == 1]
            np.testing.assert_array_equal(values(on,"znt")[first], values(off,"znt")[first])


def test_disabled_spp_config_preserves_old_prepared_cache_identity():
    from gpuwm.ingest.prepared_cache import compare_prepared_domain_config
    fields = {"spp_conv":0,"spp_pbl":0,"spp_lsm":0}
    defaults = {"run."+key:value for key,value in fields.items()}
    cached = {"run":{"dx":3000.0}}
    live = {"run":{"dx":3000.0,**fields}}
    tolerated,differing = compare_prepared_domain_config(cached,live,not_in_use=defaults)
    assert set(tolerated) == set(defaults)
    assert differing == []
    for key in fields:
        live["run"][key] = 1
        assert compare_prepared_domain_config(cached,live,not_in_use=defaults)[1] == ["run."+key]
        live["run"][key] = 0
