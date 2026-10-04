"""The plain ``--members N`` door, end to end on a card: real members or a refusal.

These run the commands a user types.  ``gpuwm domain`` writes a small
one-domain config; ``gpuwm ensemble CONFIG --members 2`` names no recipe.

* On a source whose adapter row declares a runnable operational ensemble,
  the door plans two of its members, fetches and prepares each through
  that source's own chain, and runs them: exit 0, and the members differ
  from the first frame on.
* On a source that declares none, the door exits 2 with the breakage and
  the remedy before anything is downloaded.

The defect this holds closed: both commands used to fetch ONE trajectory,
run it N times, and publish zero spread and 0 or 1 probabilities.

Nothing is mocked, so the tests need a CUDA card, the provider network
(about 0.5 GB of source files, cached under ``GPUWM_RECIPE_DOOR_OUTDIR``
when that is set) and the WPS geography tree.  They are gated on
``GPUWM_NETWORK_TESTS=1``.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
#: A fixed archived cycle, the one the recipe door's own tests ask for.
CYCLE = os.environ.get("GPUWM_RECIPE_DOOR_CYCLE", "2026-10-02T18")

pytestmark = [pytest.mark.gpu, pytest.mark.network, pytest.mark.slow, requires_gpu,
              pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                                 reason="live source fetch; set GPUWM_NETWORK_TESTS=1")]


def _gpuwm(*argv, timeout):
    environment = dict(os.environ, PYTHONIOENCODING="utf-8",
                       PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", "gpuwm", *argv], cwd=REPO, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout)


HOURS = 3


def _sources():
    """One source whose operational ensemble plans this window and one with none.

    Read from the adapter table and the planner, so the tests follow the
    table: no source is named here.
    """
    from datetime import datetime, timedelta, timezone
    from gpuwm.ensemble.recipes import build_recipe
    from gpuwm.source_adapters import get_source_adapter, wizard_planable_source_ids
    start = datetime.fromisoformat(CYCLE + ":00:00").replace(tzinfo=timezone.utc)
    declared = undeclared = None
    for source in wizard_planable_source_ids():
        adapter = get_source_adapter(source)
        if adapter.member_set:
            continue        # itself an ensemble: its config names one member
        if not adapter.ensemble_source:
            undeclared = undeclared or source
            continue
        if declared is None:
            try:
                build_recipe(source=source, cycle=start, start=start,
                             end=start + timedelta(hours=HOURS), count=2, base_seed=0)
            except ValueError:
                continue
            declared = (source, adapter.ensemble_source)
    return declared, undeclared


def _tiny_config(tmp_path, source, hours):
    from gpuwm.geog_assets import default_geog_root
    if not Path(default_geog_root()).is_dir():
        pytest.skip(f"no WPS geography tree at {default_geog_root()}")
    config = tmp_path / "membersdoor.toml"
    made = _gpuwm("domain", "--point", "35.3,-97.5", "--point-extent-km", "150", "--root-dx", "3",
                  "--card", "32gb", "--hours", str(hours), "--source", source, "--cycle", CYCLE,
                  "--name", "membersdoor", "--out", str(config), timeout=600)
    assert made.returncode == 0, made.stdout + made.stderr
    return config, Path(os.environ.get("GPUWM_RECIPE_DOOR_OUTDIR") or tmp_path / "out")


def test_plain_member_count_runs_the_sources_operational_ensemble(tmp_path):
    import netCDF4
    import numpy as np

    declared, _ = _sources()
    if declared is None:
        pytest.skip("no wizard source declares a runnable operational ensemble")
    source, ensemble = declared
    # Three hours: operational ensembles post on a coarser lead ladder than
    # their deterministic model, and the window must end on one of its leads.
    config, case_root = _tiny_config(tmp_path, source, hours=HOURS)
    planned = _gpuwm("ensemble", str(config), "--members", "2", "--dry-run", timeout=600)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert "input-ensemble recipe, 2 members" in planned.stdout
    before = set(case_root.glob("run-*")) if case_root.is_dir() else set()
    ran = _gpuwm("ensemble", str(config), "--members", "2", "--outdir", str(case_root), timeout=3000)
    assert ran.returncode == 0, (ran.stdout + ran.stderr)[-6000:]
    (run,) = set(case_root.glob("run-*")) - before

    receipt = json.loads((run / "ensemble-recipe.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete" and receipt["recipe"]["kind"] == "input-ensemble"
    members = receipt["members"]
    assert [member["member_id"] for member in members] == [0, 1]
    assert {member["source"] for member in members} == {ensemble}
    assert len({member["source_member"] for member in members}) == 2
    assert len({member["trajectory_sha256"] for member in members}) == 2
    assert len({member["prepared_root"] for member in members}) == 2

    manifest = json.loads((run / "run" / "ensemble-run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["request"]["members"] == 2
    # A plain member count: the request names no recipe and declares no copies.
    assert "recipe" not in manifest["request"] and "identical_members" not in manifest
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
    # Two members of a real ensemble differ from the first frame on.  Two
    # copies of one trajectory, which is what this door used to run, have
    # zero spread everywhere.
    assert all(value > 0.0 for value in spreads.values()), spreads


def test_plain_member_count_answers_the_go_and_run_flags_for_its_members(tmp_path):
    """The flags of the one-trajectory doors, on the install a user has.

    Each of these was read for the config's one trajectory, or by nothing:
    ``gpuwm run --gpu-uuid`` pinned no card (its members ran on every
    visible one), ``--readiness`` answered for the config's own source,
    and ``--cycle`` was refused as a flag the member plan would drop.
    """
    from datetime import datetime, timedelta

    declared, _ = _sources()
    if declared is None:
        pytest.skip("no wizard source declares a runnable operational ensemble")
    source, ensemble = declared
    config, _case_root = _tiny_config(tmp_path, source, hours=HOURS)

    pinned = tmp_path / "pinned"
    ran = _gpuwm("run", str(config), "--members", "2", "--outdir", str(pinned),
                 "--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000000", timeout=600)
    assert ran.returncode == 2, (ran.stdout + ran.stderr)[-4000:]
    assert "does not use --gpu-uuid" in ran.stderr
    assert "they would land on cards the pin excludes" in ran.stderr
    assert "ensemble: member " not in ran.stdout and not pinned.exists()

    asked = _gpuwm("go", str(config), "--members", "2", "--readiness", "--no-probe",
                   "--outdir", str(tmp_path / "asked"), timeout=600)
    assert asked.returncode == 0, (asked.stdout + asked.stderr)[-4000:]
    document = json.loads(asked.stdout)
    assert document["source"] == ensemble
    assert document["recipe"]["kind"] == "input-ensemble"
    assert [window["source"] for window in document["recipe"]["member_windows"]] == [ensemble] * 2
    assert not (tmp_path / "asked").exists()

    earlier = datetime.fromisoformat(CYCLE + ":00:00") - timedelta(hours=24)
    moved = _gpuwm("ensemble", str(config), "--members", "2", "--dry-run",
                   "--cycle", earlier.strftime("%Y-%m-%dT%H"),
                   "--outdir", str(tmp_path / "moved"), timeout=600)
    assert moved.returncode == 0, (moved.stdout + moved.stderr)[-4000:]
    plan = [line for line in moved.stdout.splitlines() if line.startswith("ensemble: member ")]
    assert len(plan) == 2 and len(set(plan)) == 2, moved.stdout
    assert all(f"{ensemble} {earlier:%Y-%m-%dT%H}Z member " in line for line in plan), moved.stdout
    assert "dry run: nothing was fetched, prepared or run" in moved.stdout


def test_plain_member_count_with_no_ensemble_to_draw_from_exits_2_before_any_download(tmp_path):
    _, source = _sources()
    if source is None:
        pytest.skip("every wizard source declares an operational ensemble")
    config, _ = _tiny_config(tmp_path, source, hours=1)
    case_root = tmp_path / "refused"
    for command in ("ensemble", "go"):
        ran = _gpuwm(command, str(config), "--members", "2", "--outdir", str(case_root), timeout=600)
        assert ran.returncode == 2, (ran.stdout + ran.stderr)[-4000:]
        assert ("2 members from one input are 2 copies of one forecast: spread is zero and "
                "probabilities are 0 or 1.") in ran.stderr
        assert f"{source} declares none" in ran.stderr
        assert "--recipe time-lagged" in ran.stderr and "--trajectories FILE" in ran.stderr
        # Nothing was fetched, prepared or claimed.
        assert not case_root.exists()
        assert "fetch " not in ran.stdout and "preparing member" not in ran.stdout
