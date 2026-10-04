"""The real doors, end to end: tiny two-member recipe ensembles on a card.

These run the commands a user types.  ``gpuwm domain`` writes a small
one-domain HRRR config.  ``gpuwm ensemble CONFIG --recipe time-lagged
--members 2`` fetches two hourly cycles, and ``gpuwm ensemble CONFIG
--trajectories FILE`` fetches one HRRR and one RAP cycle; each member is
prepared by its source's own chain and both run through the ensemble
session.  Nothing is mocked, so the tests need a CUDA card, the provider
network (about 4.5 GB of source files, cached under
``GPUWM_RECIPE_DOOR_OUTDIR`` when that is set) and the WPS geography tree.
They are gated on ``GPUWM_NETWORK_TESTS=1``.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
#: A fixed archived cycle: the providers keep these cycles, so the tests ask
#: for the same bytes on every run.
CYCLE = os.environ.get("GPUWM_RECIPE_DOOR_CYCLE", "2026-10-02T18")

pytestmark = [pytest.mark.gpu, pytest.mark.network, pytest.mark.slow, requires_gpu,
              pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                                 reason="live source fetch; set GPUWM_NETWORK_TESTS=1")]


def _gpuwm(*argv, timeout):
    environment = dict(os.environ, PYTHONIOENCODING="utf-8",
                       PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", "gpuwm", *argv], cwd=REPO, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def _tiny_config(tmp_path):
    from gpuwm.geog_assets import default_geog_root
    if not Path(default_geog_root()).is_dir():
        pytest.skip(f"no WPS geography tree at {default_geog_root()}")
    config = tmp_path / "recipedoor.toml"
    made = _gpuwm("domain", "--point", "35.3,-97.5", "--point-extent-km", "150", "--root-dx", "3",
                  "--card", "32gb", "--hours", "1", "--source", "hrrr", "--cycle", CYCLE,
                  "--name", "recipedoor", "--out", str(config), timeout=600)
    assert made.returncode == 0, made.stdout + made.stderr
    return config, Path(os.environ.get("GPUWM_RECIPE_DOOR_OUTDIR") or tmp_path / "out")


def _run_door(config, case_root, *flags):
    planned = _gpuwm("ensemble", str(config), *flags, "--dry-run", timeout=600)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert "member 0: " in planned.stdout and "member 1: " in planned.stdout
    before = set(case_root.glob("run-*")) if case_root.is_dir() else set()
    ran = _gpuwm("ensemble", str(config), *flags, "--outdir", str(case_root), timeout=3000)
    assert ran.returncode == 0, (ran.stdout + ran.stderr)[-6000:]
    (run,) = set(case_root.glob("run-*")) - before
    return run


def _check_run(run, kind):
    import netCDF4
    import numpy as np

    receipt = json.loads((run / "ensemble-recipe.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete" and receipt["recipe"]["kind"] == kind
    members = receipt["members"]
    assert [member["member_id"] for member in members] == [0, 1]
    assert len({member["trajectory_sha256"] for member in members}) == 2
    assert len({member["prepared_root"] for member in members}) == 2
    for member in members:
        assert Path(member["prepared_root"]).is_dir()

    manifest = json.loads((run / "run" / "ensemble-run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["request"]["recipe"] == kind and manifest["request"]["members"] == 2
    assert sorted(manifest["members_completed"]) == [0, 1]

    products = sorted((run / "run").glob("d01/ensemble/products/ensemble_*.nc"))
    assert len(products) >= 2
    spreads = {}
    for path in products:
        with netCDF4.Dataset(path) as dataset:
            assert int(dataset.ensemble_members) == 2
            field = np.asarray(dataset.variables["temperature2_spread"][:], dtype=np.float64)
            assert np.isfinite(field).all()
            spreads[path.name] = float(field.max())
    # Real trajectories differ from the first frame on: an ensemble of two
    # copies of one trajectory would have zero spread everywhere.
    assert all(value > 0.0 for value in spreads.values()), spreads
    assert any((run / "run" / "maps" / "d01").glob("ens_spread_temperature2/*/*.png"))
    return members


def test_two_member_time_lagged_ensemble_runs_from_the_front_door(tmp_path):
    config, case_root = _tiny_config(tmp_path)
    run = _run_door(config, case_root, "--recipe", "time-lagged", "--members", "2")
    members = _check_run(run, "time-lagged")
    assert {member["source"] for member in members} == {"hrrr"}
    # Two different cycles reach the same start: the second starts one lead later.
    assert [member["start_lead_hours"] for member in members] == [0, 1]
    assert len({member["cycle"] for member in members}) == 2


def test_two_member_multi_model_ensemble_runs_from_the_front_door(tmp_path):
    config, case_root = _tiny_config(tmp_path)
    listed = tmp_path / "members.json"
    listed.write_text(json.dumps([{"source": "hrrr", "cycle": CYCLE}, {"source": "rap", "cycle": CYCLE}]))
    run = _run_door(config, case_root, "--trajectories", str(listed))
    members = _check_run(run, "multi-model")
    assert [member["source"] for member in members] == ["hrrr", "rap"]
    assert [member["start_lead_hours"] for member in members] == [0, 0]
