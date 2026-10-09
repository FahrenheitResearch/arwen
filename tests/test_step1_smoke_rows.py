"""The step-1 smoke matrix enumerates every shipped config and profile (CPU).

``tools/battery/step1_smoke.py`` steps each row on a card; this file holds
its row list on CPU, so the GPU leg cannot quietly shrink.  THE BREAKAGE:
2.8.8's candidate died at model step 1 on every real forecast and no row of
any test stepped a shipped configuration on a card.  A matrix that silently
dropped a config (a loader refusal swallowed, a profile renamed) would
recreate that gap one config at a time, so here every shipped TOML must
become a steppable row or a SKIP that names why it is not a model
configuration, and every shipped physics profile must be a row.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class _TwinConfig:
    """A picklable stand-in RunConfig: two rows carrying it share a signature."""
    mp_physics: int = 8
    nz: int = 40


@pytest.fixture(scope="module")
def smoke():
    from tools.battery import step1_smoke
    return step1_smoke


@pytest.fixture(scope="module")
def rows(smoke):
    return smoke.enumerate_rows(dedupe=False)


def test_every_shipped_config_is_a_row(smoke, rows):
    covered = {name.split("#", 1)[0].removeprefix("config:")
               for row in rows if row.kind == "config"
               for name in (row.row_id, *row.covers)}
    shipped = {path.relative_to(ROOT).as_posix()
               for path in smoke.shipped_configs()}
    assert shipped, "no configs/ TOMLs found"
    assert shipped <= covered, sorted(shipped - covered)
    assert any(name.startswith("configs/recipes/") for name in covered)


def test_every_shipped_config_loads_and_shrinks(rows):
    """A shipped config the loader refuses is a defect, not a smoke row."""
    broken = [f"{row.row_id} [{row.phase}] {row.error}" for row in rows
              if row.cfg is None and row.status == "FAIL"]
    assert not broken, "\n".join(broken)


def test_skips_are_only_files_with_no_model_table(smoke, rows):
    import tomllib
    for row in rows:
        if row.status != "SKIP":
            continue
        path = ROOT / row.row_id.removeprefix("config:")
        tables = set(tomllib.loads(path.read_text(encoding="utf-8")))
        assert not tables & smoke.MODEL_TABLES, (row.row_id, sorted(tables))


def test_every_shipped_physics_profile_is_a_row(rows):
    from gpuwm.physics_compat import SINGLE_DOMAIN_PHYSICS_PROFILES
    profiles = {row.row_id.removeprefix("profile:") for row in rows
                if row.kind == "profile" and row.cfg is not None}
    assert set(SINGLE_DOMAIN_PHYSICS_PROFILES) <= profiles


def test_a_row_keeps_its_physics_and_drops_only_geometry(smoke, rows):
    from gpuwm.experiment import load_experiment
    path = ROOT / "configs" / "gfs_wrf_hierarchy_proof.toml"
    run = load_experiment(path).domains[0].run
    row = next(row for row in rows
               if row.row_id == "config:configs/gfs_wrf_hierarchy_proof.toml#d01")
    assert (row.cfg.nx, row.cfg.ny) == (32, 32)
    assert row.cfg.open_x and row.cfg.open_y
    assert not (row.cfg.specified or row.cfg.nested)
    for name in smoke.SUMMARY_FIELDS:
        assert getattr(row.cfg, name) == getattr(run, name), name


def test_identical_rows_run_once_and_say_what_they_cover(smoke):
    merged = smoke.enumerate_rows()
    signatures = [smoke._signature(row.cfg) for row in merged
                  if row.cfg is not None]
    assert len(signatures) == len(set(signatures))
    assert sum(len(row.covers) for row in merged) >= len(merged)


def test_the_list_door_needs_no_card(smoke, capsys):
    assert smoke.main(["--list", "--only", "profile:"]) == 0
    assert "profile:" in capsys.readouterr().out


def test_ruc_mosaic_rows_declare_one_hot_fractions(smoke, rows):
    """A RUC mosaic row carries the static fractions its builder lacks.

    Without them ``initialize_physics`` refuses the row at build ("RUC
    mosaic needs LANDUSEF category fractions"), a refusal about the
    harness, not the model, which would keep the matrix red for ever.
    """
    import numpy as np
    mosaic = [row for row in rows if row.cfg is not None
              and int(row.cfg.sf_surface_physics) == 3
              and int(row.cfg.mosaic_lu)]
    assert mosaic, "no shipped config runs the RUC land-use mosaic"
    declared = smoke._declared_mosaic_fractions(mosaic[0].cfg, 4, 5)
    landusef = declared["landusef"]
    assert landusef.shape[1:] == (4, 5)
    np.testing.assert_array_equal(landusef.sum(axis=0), 1.0)
    assert smoke._declared_mosaic_fractions(
        next(row.cfg for row in rows if row.cfg is not None
             and int(row.cfg.sf_surface_physics) != 3), 4, 5) == {}


def test_terrain_drag_rows_declare_flat_ground_statistics(smoke, rows):
    """A ``gwd_opt``/``topo_wind`` row carries the zero statistics of flat
    ground; without them ``initialize_physics`` refuses the row at build
    ("this initialization route hands the physics no static field set")."""
    from gpuwm.core.terrain_drag import required_static_fields
    drag = [row for row in rows if row.cfg is not None
            and (int(row.cfg.gwd_opt) or int(row.cfg.topo_wind))]
    assert drag, "no shipped config runs the terrain drag"
    cfg = drag[0].cfg
    static = smoke._declared_terrain_drag_static(cfg, 4, 5)["terrain_drag_static"]
    need = {"HGT_M", *required_static_fields(int(cfg.topo_wind), int(cfg.gwd_opt))}
    assert set(static) == need
    assert all(value.shape == (4, 5) and not value.any()
               for value in static.values())


def test_a_crashed_twin_row_never_takes_its_twins_verdict(smoke, monkeypatch):
    """Under ``--no-dedupe`` identical configs run as separate rows.  When
    their scratch files were keyed by config signature, a row whose worker
    crashed before writing a result read its twin's result file and was
    reported with the twin's PASS.  Each launch now owns its own files."""
    import argparse
    import json
    import subprocess

    calls = []

    def fake_run(command, **kwargs):
        result_file = command[command.index("--result-file") + 1]
        calls.append(result_file)
        if len(calls) == 1:
            Path(result_file).write_text(json.dumps(
                {"status": "PASS", "phase": None, "error": None,
                 "detail": None, "seconds": 0.1}), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command, 1, "", "Traceback\nRuntimeError: worker crashed")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    cfg = _TwinConfig()
    twins = [smoke.Row("config:a.toml#d01", "config", ["a"], cfg=cfg),
             smoke.Row("config:b.toml#d01", "config", ["b"], cfg=cfg)]
    args = argparse.Namespace(cards=1, per_card=1, steps=3, seed=0,
                              timeout=60)
    records = smoke.run_matrix(twins, args)
    assert [r["status"] for r in records] == ["PASS", "FAIL"]
    assert len(set(calls)) == 2
    assert records[1]["phase"] == "process"
    assert "worker crashed" in records[1]["error"]


def test_a_launch_slot_is_the_granted_device_not_a_bare_index(smoke, monkeypatch):
    """Slot i maps to the i-th device the card queue granted (often a UUID)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa,GPU-bbb,GPU-ccc")
    assert [smoke._card_device(i) for i in range(3)] == ["GPU-aaa", "GPU-bbb", "GPU-ccc"]
    with pytest.raises(SystemExit):
        smoke._card_device(3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    assert [smoke._card_device(i) for i in range(2)] == ["4", "6"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert smoke._card_device(2) == "2"
