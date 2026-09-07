"""Fresh GFS scenarios preserve preparation authority and defer state changes.

These are CPU orchestration gates. The real wizard, experiment loader,
input-manifest verifier, preparation control flow and prepared identity
builder run; decoding and array numerics use small immutable fixtures.
The numerical application seam has separate test_initial_perturbation tests.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import gfs_direct, stage_cli, wrf_direct
from gpuwm.experiment import load_experiment
from gpuwm.ingest.prepared_cache import prepared_cache_identity


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config(tmp_path, *, domains=2):
    from gpuwm.cli import main

    config = tmp_path / "experiment.toml"
    args = ["domain", "--point=35.3,-97.5", "--root-dx", "12",
            "--vram-gib", "8", "--source", "gfs", "--hours", "3",
            "--cycle", "2026-09-05T00", "--out", str(config)]
    if domains > 1:
        args += ["--chain", ",".join(["3"] * (domains - 1))]
    assert main(args) == 0
    assert len(load_experiment(config).domains) == domains
    return config


def _with_bubble(config):
    result = config.with_name("scenario.toml")
    result.write_bytes(config.read_bytes() + b"""
# Hypothetical initial-state scenario; preserve the source experiment.
[[perturbation.bubbles]]
center_lat = 35.3
center_lon = -97.5
center_height_m = 1500.0
radius_km = 10.0
depth_m = 1500.0
amplitude_k = 2.5
rh_preserve = true
""")
    return result


def _inputs(tmp_path, config, name):
    root = tmp_path / name
    root.mkdir()
    roles = {"experiment_config": config,
             "wps_namelist": tmp_path / "experiment.namelist.wps"}
    for role in ("bridge", "grib-f000", "grib-f003"):
        roles[role] = root / role
        roles[role].write_text(role, encoding="utf-8")
    roles["bridge"].chmod(0o700)
    roles["series"] = root / "series.tsv"
    roles["series"].write_text("0\tgrib-f000\t81\n3\tgrib-f003\t96\n")
    manifest = root / "input-manifest.json"
    manifest.write_text(json.dumps({
        "schema": gfs_direct.INPUT_MANIFEST_SCHEMA,
        "source": {"model": "GFS", "product": "pgrb2.0p25",
                   "cycle": "2026-09-05T00:00:00Z"},
        "files": {role: {"name": path.name, "sha256": _sha(path)}
                  for role, path in roles.items()},
    }))
    return dict(series=roles["series"], cycle="2026-09-05_00:00:00",
                bridge=roles["bridge"], wps_namelist=roles["wps_namelist"],
                experiment_config=config, input_manifest=manifest,
                input_manifest_sha256=_sha(manifest), output_root=root / "prepared",
                static_input=None, static_receipt=None, geog_root=tmp_path / "geog",
                preprocess_backend="cpu")


def _cpu_preparation(monkeypatch, exp):
    """Substitute only expensive source/array work, retaining GFS orchestration."""
    captures = {"root_initializations": [], "hierarchies": []}
    monkeypatch.setattr(gfs_direct, "_implementation_sha256", lambda: "a" * 64)
    monkeypatch.setattr(gfs_direct, "_git_source_identity", lambda: {"available": False})
    monkeypatch.setattr(gfs_direct, "resolve_preprocess_backend",
                        lambda *_a, **_k: SimpleNamespace(receipt=lambda: {"backend": "cpu"}))
    monkeypatch.setattr(gfs_direct, "release_backend_memory", lambda *_a: None)
    statics = {name: np.ones((3, 3)) for name in (
        "LANDMASK", "LU_INDEX", "HGT_M", "SCT_DOM", "TMN", "MAPFAC_M",
        "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA")}
    monkeypatch.setattr(gfs_direct, "_static_from_geog",
                        lambda *_a: (statics, {}, None))
    grid = SimpleNamespace(**{name: lambda: (np.zeros((3, 3)), np.zeros((3, 3)))
                            for name in ("latlon_mass", "latlon_u", "latlon_v")})
    monkeypatch.setattr(gfs_direct, "validate_native_lambert_contracts",
                        lambda actual, *_a, **_k: tuple(grid for _ in actual.domains))
    monkeypatch.setattr(gfs_direct, "_validate_grid_and_vertical_contract",
                        lambda *_a, **_k: grid)

    def bridge(command, **_kwargs):
        assert command[0].endswith("bridge")
        decoded = Path(command[3])
        decoded.mkdir()
        for name in ("gate.tsv", "inventory.tsv", "decoded-sha256.tsv"):
            (decoded / name).write_text("CPU fixture\n")
        return SimpleNamespace(returncode=0, stdout="CPU fixture", stderr="")

    monkeypatch.setattr(gfs_direct.subprocess, "run", bridge)
    snapshots = tuple(SimpleNamespace(valid_time=exp.start_time + timedelta(hours=hour),
                                      fields={"SKINTEMP": np.ones((3, 3))})
                      for hour in (0, 3))
    monkeypatch.setattr(gfs_direct, "_load_bridge_snapshots", lambda *_a, **_k: snapshots)
    monkeypatch.setattr(gfs_direct, "orient_global_source_longitudes", lambda source, *_a: source)
    monkeypatch.setattr(gfs_direct, "_source_coverage_receipt", lambda *_a: {})
    monkeypatch.setattr(gfs_direct, "interpolate_era5_to_lambert", lambda source, *_a, **_k: source)
    monkeypatch.setattr(gfs_direct, "interpolate_lake_skin_temperature",
                        lambda *_a: np.zeros((3, 3)))
    monkeypatch.setattr(gfs_direct, "soil_source_orography", lambda *_a: None)
    monkeypatch.setattr(gfs_direct, "soil_mesh_plan_from_case", lambda *_a: None)
    monkeypatch.setattr(gfs_direct, "preprocess_land_surface_soil",
                        lambda *_a, **_k: SimpleNamespace(soil_texture_downscale={}))

    def initialize(met, *_args, **kwargs):
        # Neither the initial state nor any boundary time may get an applier.
        assert kwargs.get("initial_perturbation") is None
        payload = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        payload.flags.writeable = False
        state = SimpleNamespace(thp=payload, set_map_coriolis=lambda *_a, **_k: None)
        captures["root_initializations"].append((met.valid_time, payload.copy()))
        return SimpleNamespace(state=state, initial_perturbation={})

    monkeypatch.setattr(gfs_direct, "initialize_real", initialize)

    class Boundaries:
        def __init__(self, **_kwargs):
            self.frames = {}

        def add_state(self, state, *, index):
            self.frames[index] = state.thp.copy()

        def build(self, times):
            return self.frames

    monkeypatch.setattr(gfs_direct, "StateBoundaryFrames", Boundaries)
    monkeypatch.setattr(gfs_direct, "attach_lateral_boundaries",
                        lambda state, frames: setattr(state, "lateral_boundaries", frames))
    monkeypatch.setattr(gfs_direct, "_write_static_cache",
                        lambda path, values: path.write_bytes(b"CPU static fixture"))
    monkeypatch.setattr(gfs_direct, "_write_geometry_receipt",
                        lambda path, *_a: path.write_text("{}"))

    def hierarchy(**kwargs):
        # Exercise the real cache identity builder with the exact experiment
        # and digest that preparation forwards to the hierarchy writer.
        actual = kwargs["exp"]
        assert actual.perturbation == exp.perturbation
        identity = prepared_cache_identity(
            bridge_manifest_sha256=kwargs["bridge_manifest_sha256"],
            source_manifest_sha256=kwargs["source_manifest_sha256"],
            static_cache_sha256="b" * 64, namelist_sha256=kwargs["namelist_sha256"],
            domain_config=actual.root, forcing_hours=kwargs["forcing_hours"],
            source_identity=kwargs["source_identity"])
        captures["hierarchies"].append({
            "experiment": asdict(actual), "identity": identity,
            "initial": kwargs["root_initial_result"].state.thp.copy(),
            "boundaries": kwargs["root_boundaries"],
            "stock_wrf_export": kwargs["stock_wrf_export"],
        })
        return SimpleNamespace(
            static_catalog_receipt={}, source_coverage_receipt={}, topology_receipt={},
            statics_corridor_receipt=None,
            hierarchy=SimpleNamespace(artifacts=SimpleNamespace(receipt={"fixture": True}),
                                      wrf_manifest={"status": "NOT_REQUESTED"}, timings_seconds={}))

    monkeypatch.setattr(gfs_direct, "initialize_and_export_regular_source_hierarchy", hierarchy)
    return captures


@pytest.mark.parametrize("domains", [2, 3])
def test_fresh_tree_defers_bubbles_preserves_arrays_and_binds_exact_config(
        tmp_path, monkeypatch, capsys, domains):
    config = _config(tmp_path, domains=domains)
    scenario = _with_bubble(config)
    original_bytes, scenario_bytes = config.read_bytes(), scenario.read_bytes()
    observed = []
    for path, name in ((config, "baseline"), (scenario, "changed-initial-state")):
        exp = load_experiment(path)
        before = asdict(exp)
        with monkeypatch.context() as patch:
            captures = _cpu_preparation(patch, exp)
            arguments = _inputs(tmp_path, path, name)
            proof = gfs_direct.prepare_gfs_wrf(**arguments)
        assert asdict(exp) == before
        (prepared,) = captures["hierarchies"]
        assert prepared["experiment"] == before
        assert prepared["identity"]["namelist_sha256"] == _sha(path)
        assert len(captures["root_initializations"]) == 2
        assert prepared["stock_wrf_export"] == "optional"
        assert proof["schema"] == gfs_direct.HIERARCHY_PROOF_SCHEMA
        assert proof["domain_count"] == domains
        assert json.loads((arguments["output_root"] / "proof.json").read_text()) == proof
        observed.append((proof, prepared))

        # This is the exact bundle/command seam used by prepared:go, not a
        # runner chosen from the caller's requested number of domains.
        from gpuwm.runplan import prepared_chain_for_source
        assert prepared_chain_for_source("gfs") == "prepared:go"
        bundle = stage_cli.resolve_bundle(arguments["output_root"])
        assert bundle["layout"] == "tree"
        command = stage_cli.sim_command(bundle, experiment_config=path, wps_namelist=None,
                                        outdir=tmp_path / "forecast-not-started")
        assert command[2] == "gpuwm.prepared_domain_tree_forecast"
        assert command[command.index("--experiment-config") + 1] == str(path)
        assert command[command.index("--experiment-config-sha256") + 1] == _sha(path)

    (baseline, baseline_state), (changed, scenario_state) = observed
    assert "initial_perturbation" not in baseline
    assert "initial_perturbation" not in baseline_state["identity"]["source_identity"]
    deferred = changed["initial_perturbation"]
    assert deferred["status"] == "DEFERRED_TO_FORECAST_INITIALIZATION"
    assert deferred["config"] == load_experiment(scenario).perturbation.receipt()
    assert deferred["applied_on_restart"] is False
    assert deferred["applied_to_delayed_domains"] is False
    assert scenario_state["identity"]["source_identity"]["initial_perturbation"] == deferred
    np.testing.assert_array_equal(baseline_state["initial"], scenario_state["initial"])
    for index in baseline_state["boundaries"]:
        np.testing.assert_array_equal(baseline_state["boundaries"][index],
                                      scenario_state["boundaries"][index])
    assert config.read_bytes() == original_bytes
    assert scenario.read_bytes() == scenario_bytes
    assert "deferred to prepared-tree forecast initialization" in capsys.readouterr().err
    assert not (tmp_path / "forecast-not-started").exists()


@pytest.mark.parametrize("domains,spawn", [(1, False), (2, True)])
def test_unsupported_single_or_spawn_scenario_refuses_before_static_or_decode(
        tmp_path, monkeypatch, domains, spawn):
    config = _config(tmp_path, domains=domains)
    scenario = _with_bubble(config)
    if spawn:
        with scenario.open("a", encoding="utf-8") as target:
            target.write('\n[domain.spawn]\ntrigger = "time"\nat_s = 600.0\n')
    exp = load_experiment(scenario)
    _cpu_preparation(monkeypatch, exp)
    monkeypatch.setattr(gfs_direct, "_static_from_geog",
                        lambda *_a: pytest.fail("unsupported scenario reached static preparation"))
    arguments = _inputs(tmp_path, scenario, "refused")
    with pytest.raises(ValueError, match="spawn-triggered" if spawn else "does not apply.*perturbation"):
        gfs_direct.prepare_gfs_wrf(**arguments)
    assert not arguments["output_root"].exists()


def test_absent_perturbation_policy_is_noop_for_single_and_tree(tmp_path):
    for domains in (1, 2):
        root = tmp_path / str(domains)
        root.mkdir()
        exp = load_experiment(_config(root, domains=domains))
        before = asdict(exp)
        assert gfs_direct._prepared_initial_perturbation(exp) is None
        assert asdict(exp) == before


def _admission_plan(tmp_path, config, *, route="prepared"):
    from gpuwm.runplan import PLAN_SCHEMA, load_plan

    path = tmp_path / "admission-plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "scenario-admission",
        "route": route, "config": {"path": str(config)},
        "output_root": str(tmp_path / "no-forecast-output"),
    }), encoding="utf-8")
    return path, load_plan(path)


@pytest.mark.parametrize("door", ["check", "go", "run-plan", "stage-plan"])
def test_public_single_prepared_scenario_refuses_before_resources_or_outputs(
        tmp_path, monkeypatch, capsys, door):
    from gpuwm.cli import main
    from gpuwm import data_assets, go_cli

    config = _with_bubble(_config(tmp_path, domains=1))
    plan_path, _ = _admission_plan(tmp_path, config)
    before = {p.relative_to(tmp_path): p.read_bytes()
              for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(data_assets, "companion_root",
                        lambda: pytest.fail("unsupported scenario reached resource checks"))
    if door == "stage-plan":
        with pytest.raises(go_cli.GoRefusal, match="single-domain prepared.*does not apply"):
            go_cli.plan_from_config(config, outdir=tmp_path / "no-forecast-output")
    else:
        argv = {
            "check": ["check", str(config), "--vram-gib", "32", "--budget-gib", "24"],
            "go": ["go", str(config), "--dry-run", "--outdir", str(tmp_path / "no-forecast-output")],
            "run-plan": ["run-plan", str(plan_path), "--resolve"],
        }[door]
        capsys.readouterr()
        assert main(argv) == 2
        captured = capsys.readouterr()
        message = captured.out + captured.err
        assert "single-domain prepared" in message and "does not apply [perturbation]" in message
        assert "Traceback" not in message
    assert not (tmp_path / "no-forecast-output").exists()
    assert before == {p.relative_to(tmp_path): p.read_bytes()
                      for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("domains,bubble", [(1, False), (2, False), (2, True)])
def test_public_prepared_scenario_admission_preserves_supported_controls(tmp_path, domains, bubble):
    from gpuwm.core.preflight import _load_experiment_any
    from gpuwm.go_cli import plan_from_config
    from gpuwm.runplan import resolve_plan

    config = _config(tmp_path, domains=domains)
    if bubble:
        config = _with_bubble(config)
    original = config.read_bytes()
    exp = _load_experiment_any(config)
    assert len(exp.domains) == domains
    assert (exp.perturbation is not None) == bubble
    assert plan_from_config(config, outdir=tmp_path / "no-forecast-output")
    _, plan = _admission_plan(tmp_path, config)
    _, resolved, _ = resolve_plan(plan, require_inputs=False)
    assert len(resolved.domains) == domains
    assert (resolved.perturbation is not None) == bubble
    assert config.read_bytes() == original
    assert not (tmp_path / "no-forecast-output").exists()


def test_public_config_driven_single_scenario_keeps_supported_runtime_route(tmp_path):
    from gpuwm.core.preflight import _load_experiment_any
    from gpuwm.runplan import resolve_plan
    from test_case_data import make_case_toml

    config = _with_bubble(make_case_toml(tmp_path))
    original = config.read_bytes()
    exp = _load_experiment_any(config)
    assert len(exp.domains) == 1 and exp.perturbation is not None
    _, plan = _admission_plan(tmp_path, config, route="experiment")
    _, resolved, data = resolve_plan(plan, require_inputs=False)
    assert len(resolved.domains) == 1 and resolved.perturbation is not None
    assert data is not None
    assert config.read_bytes() == original
    assert not (tmp_path / "no-forecast-output").exists()


@pytest.mark.parametrize("mode", ["optional", "required"])
def test_stock_wrf_export_cannot_drop_deferred_bubbles(tmp_path, monkeypatch, mode):
    from test_native_hierarchy import _hierarchy_with, _run

    config = _with_bubble(_config(tmp_path))
    exp = load_experiment(config)
    output = tmp_path / "direct-wrf"
    with pytest.raises(wrf_direct.StockWrfExportUnsupported, match="deferred.*not stock-WRF"):
        wrf_direct.export_prepared_wrf_hierarchy(exp, (), output)
    assert not output.exists()

    # Retain the actual native hierarchy's optional-export error handling,
    # using its established CPU composition fixture for array preparation.
    inputs = _hierarchy_with(monkeypatch, tmp_path,
                             exporter=wrf_direct.export_prepared_wrf_hierarchy)
    inputs[0].perturbation = exp.perturbation
    if mode == "required":
        with pytest.raises(wrf_direct.StockWrfExportUnsupported, match="perturbation"):
            _run(inputs, tmp_path, stock_wrf_export=mode)
    else:
        result = _run(inputs, tmp_path, stock_wrf_export=mode)
        assert result.artifacts is inputs[4]
        assert result.wrf_manifest["status"] == "REFUSED"
        assert "perturbation" in result.wrf_manifest["unsupported"]
        assert "files" not in result.wrf_manifest
    assert not (tmp_path / "wrf").exists()
