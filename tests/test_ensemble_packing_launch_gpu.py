"""Three real source trajectories, one model hour, through the public go door.

Run once inside each one-card and two-card queue grant. No launch or
preparation stage is mocked. The receipt records the visible grant and
the physical budgets that the production session actually sampled.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, pytest.mark.network, pytest.mark.slow, requires_gpu,
              pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                                 reason="set GPUWM_NETWORK_TESTS=1 inside the card queue")]


def _command(*args):
    completed = subprocess.run([sys.executable, "-m", "gpuwm.cli", *map(str, args)],
                               capture_output=True, text=True, timeout=1500)
    return completed


@pytest.mark.parametrize("restart_interval", [3600, 0])
def test_three_trajectories_launch_on_the_granted_cards(tmp_path, restart_interval):
    import cupy as cp
    from gpuwm.companion_domains import candidate_wps_text
    from gpuwm.experiment import load_experiment
    from gpuwm.toml_document import emit_experiment_toml

    cards = int(os.environ["GPUWM_PACKING_TEST_CARDS"])
    assert cp.cuda.runtime.getDeviceCount() == cards
    root = Path(os.environ.get("GPUWM_PACKING_TEST_ROOT", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    config = root / f"packing-{cards}-{restart_interval}.toml"
    made = _command("domain", "--point", "35.3,-97.5", "--point-extent-km", "150",
                    "--root-dx", "12", "--nz", "24", "--card", "32gb", "--hours", "1",
                    "--physics-profile", "wsm6-ysu-mm5-noah-no-radiation-v1",
                    "--source", "hrrr-prs", "--cycle", "2026-10-07T18", "--name", "packing", "--out", config)
    (root / f"domain-{cards}-{restart_interval}.log").write_text(made.stdout + made.stderr)
    assert made.returncode == 0, made.stdout + made.stderr
    raw = tomllib.loads(config.read_text())
    raw["experiment"]["restart_interval_s"] = restart_interval
    raw["domain"][0].update(nx=40, ny=40, history_interval_s=3600)
    config.write_text(emit_experiment_toml(raw))
    exp = load_experiment(config)
    config.with_suffix(".namelist.wps").write_text(candidate_wps_text(raw, exp, exp, config))
    trajectories = root / "trajectories.json"
    trajectories.write_text(json.dumps([
        {"source": source, "cycle": "2026-10-07T18"} for source in ("hrrr-prs", "rap", "rrfs")]))
    out = root / f"case-{cards}-{restart_interval}"
    argv = ("go", config, "--trajectories", trajectories, "--outdir", out,
            "--geog-root", os.environ["GPUWM_PACKING_TEST_GEOG"], "--keep-member-files", "--products", "none")
    # The selected card is outside this grant. Admission prepares valid
    # source members, then the real production sampler refuses at forecast.
    raw["ensemble"] = {"members": 3, "member_device_ids": [cards]}
    config.write_text(emit_experiment_toml(raw))
    planned = _command(*argv, "--dry-run")
    (root / f"plan-{cards}-{restart_interval}.log").write_text(planned.stdout + planned.stderr)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    failed = _command(*argv)
    (root / f"failure-{cards}-{restart_interval}.log").write_text(failed.stdout + failed.stderr)
    assert failed.returncode != 0, failed.stdout + failed.stderr
    failed_run, = out.glob("run-*")
    prior = json.loads((failed_run / "ensemble-recipe.json").read_text())
    assert prior["failure"]["stage"] == "forecast" and prior["preparation_status"] == "ready"
    assert "non-visible physical card" in failed.stdout + failed.stderr
    raw.pop("ensemble")
    config.write_text(emit_experiment_toml(raw))
    ran = _command(*argv)
    log = root / f"go-{cards}-{restart_interval}.log"
    log.write_text(ran.stdout + ran.stderr)
    assert ran.returncode == 0, (ran.stdout + ran.stderr)[-10000:]
    run, = set(out.glob("run-*")) - {failed_run}
    recipe = json.loads((run / "ensemble-recipe.json").read_text())
    manifest = json.loads((run / "run" / "ensemble-run.json").read_text())
    assert recipe["status"] == "complete" and recipe["preparation_status"] == "ready"
    assert [row["preparation_reused_from_run"] for row in recipe["members"]] == [str(failed_run)] * 3
    assert manifest["status"] == "PASS" and sorted(manifest["members_completed"]) == [0, 1, 2]
    assert len(manifest["ordinary_memory_sampling"]) == cards
    assert len(manifest["packing"]["cards"]) == (1 if restart_interval else cards)
    from gpuwm.netcdf_bridge import open_dataset
    histories, fields = {}, {}
    for relative in manifest["member_history_files"]:
        path = run / "run" / relative
        assert path.is_file()
        histories[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with open_dataset(path) as dataset:
            dataset.set_auto_maskandscale(False)
            fields[str(path.relative_to(run))] = {
                name: {"dtype": str(variable.dtype), "shape": variable.shape,
                       "sha256": hashlib.sha256(variable[:].tobytes()).hexdigest()}
                for name, variable in dataset.variables.items() if variable.dtype.kind in "fiu"}
    assert histories, "the real member runners must produce history"
    receipt = {"cards_granted": cards, "restart_interval_s": restart_interval,
               "members": 3, "run_seconds": exp.run_seconds, "grid": [40, 40, 24],
               "log": str(log), "recipe": recipe, "manifest": manifest, "history_sha256": histories,
               "history_field_sha256": fields}
    (root / f"receipt-{cards}-{restart_interval}.json").write_text(json.dumps(receipt, indent=2))
