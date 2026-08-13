"""The wizard-emits/adapter-accepts round-trip gate (task #204).

``gpuwm domain --source X`` writes a config, and source X's OWN adapter
must accept that exact file at least through config validation.  Nothing
gated that until the 2.2.0 verification campaign ran the documented ERA5
chain end-to-end and found the door dead on arrival: the wizard emits
``[case_data]`` for ERA5, the ERA5 adapter loaded through
``gpuwm.experiment.load_experiment`` (which split off ``[fetch]`` but not
``[case_data]``), and the refusal claimed the config "does not have a
table 'case_data'" -- about a table that sat in the file, present and
valid.  Every emission here goes through the real wizard and every
acceptance through the real adapter seam, CPU-hermetic: no fetch, no
bridge, no card.

The negative controls at the bottom pin the refusal-message CLASS the
incident exposed: a table that is present but unconsumed must be
reported as a caller routing defect, never as absent, while a genuinely
unknown table keeps its "does not have a table" refusal.
"""
from __future__ import annotations

import datetime
import tomllib

import pytest

from gpuwm.cli import main as cli_main


def _run_wizard(tmp_path, source, cycle, *extra):
    out = tmp_path / "area.toml"
    rc = cli_main([
        "domain", "--point=39.7,-96.6", "--card", "16gb",
        "--ladder", "12-3", "--source", source, "--cycle", cycle,
        "--out", str(out), *extra])
    assert rc == 0, f"wizard refused its own documented {source} emission"
    return out


# ---------------------------------------------------------------------------
# ERA5: the one-file [case_data] config through the ERA5 adapter's door.
# ---------------------------------------------------------------------------

def test_era5_wizard_config_is_accepted_by_the_era5_adapter(
        tmp_path, capsys):
    out = _run_wizard(tmp_path, "era5", "1999-05-03T12")
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert "case_data" in raw  # the shape that was refused

    # The adapter's own config door (prepare_era5_wrf loads through it).
    from gpuwm.era5_direct import load_era5_adapter_config
    exp, declared = load_era5_adapter_config(out)
    assert declared is not None, (
        "the [case_data] table must be consumed, not dropped")
    assert declared.geog_root is not None
    assert declared.forcing_interval_s == 21600.0

    # And on through the geometry contract, against the wizard's own
    # companion namelist -- the same validation prepare_era5_wrf runs.
    from gpuwm.native_wrf_contract import validate_native_lambert_contracts
    grids = validate_native_lambert_contracts(
        exp, out.parent / "area.namelist.wps", source_name="ERA5")
    assert len(grids) == len(exp.domains) == 2


def test_era5_wizard_config_loads_through_load_experiment(tmp_path, capsys):
    """Every front door that wants only the experiment portion (fetch,
    go, stream, the profile identifier) loads through
    ``gpuwm.experiment.load_experiment``; the wizard's ERA5 emission must
    load there too, with [case_data] validated and detached."""
    out = _run_wizard(tmp_path, "era5", "1999-05-03T12")
    from gpuwm.experiment import load_experiment
    exp = load_experiment(out)
    assert len(exp.domains) == 2


def test_era5_adapter_door_accepts_the_bare_experiment_shape(
        tmp_path, capsys):
    """A config with no [case_data] (the pre-wizard adapter shape) keeps
    loading, with no declarations returned."""
    out = _run_wizard(tmp_path, "gfs", "2026-07-28T06")
    from gpuwm.era5_direct import load_era5_adapter_config
    exp, declared = load_era5_adapter_config(out)
    assert declared is None
    assert len(exp.domains) == 2


# ---------------------------------------------------------------------------
# GFS: the adapter loads the emission through load_experiment (its own
# seam, gpuwm/gfs_direct.py) and validates the geometry contract.
# ---------------------------------------------------------------------------

def test_gfs_wizard_config_is_accepted_by_the_gfs_adapter(tmp_path, capsys):
    out = _run_wizard(tmp_path, "gfs", "2026-07-28T06")
    from gpuwm.experiment import (
        load_experiment, refuse_unrouted_perturbation, refuse_unrouted_spawn)
    exp = load_experiment(out)
    refuse_unrouted_perturbation(exp, "GFS-direct prepared-cache")
    refuse_unrouted_spawn(exp, "GFS-direct prepared-cache")
    from gpuwm.native_wrf_contract import validate_native_lambert_contracts
    grids = validate_native_lambert_contracts(
        exp, out.parent / "area.namelist.wps", source_name="GFS")
    assert len(grids) == len(exp.domains) == 2


# ---------------------------------------------------------------------------
# HRRR: the route reads the wizard's companion files, not the TOML, so
# acceptance means every route input exists and passes its own reader.
# ---------------------------------------------------------------------------

def test_hrrr_wizard_outputs_are_accepted_by_the_hrrr_route(
        tmp_path, capsys):
    out = _run_wizard(tmp_path, "hrrr", "2026-07-28T05")
    from gpuwm.hrrr_route_inputs import route_input_paths
    paths = route_input_paths(out)
    for role, path in paths.items():
        assert path.is_file(), f"missing wizard-written route input {role}"
    from gpuwm.experiment import load_experiment
    exp = load_experiment(out)
    assert len(exp.domains) == 2
    # The launch-preflight coverage receipt over the wizard's own
    # target-domain document must be a PASS, not a REFUSED.
    from gpuwm.source_cli import _hrrr_domain_validation
    receipt = _hrrr_domain_validation(paths["target_domain"])
    assert receipt["status"] == "PASS", receipt["error"]
    assert receipt["window"] is not None


# ---------------------------------------------------------------------------
# The refusal-message class: present-but-unconsumed is never "absent".
# ---------------------------------------------------------------------------

def _minimal_experiment_raw():
    return {
        "experiment": {
            "name": "roundtrip-gate",
            "start_time": datetime.datetime(1999, 5, 3, 12),
            "run_seconds": 3600.0,
            "restart_interval_s": 0.0,
        },
        "domain": [{"grid_id": 1}],
    }


def test_a_present_companion_table_is_reported_as_unconsumed_not_absent():
    from gpuwm.experiment import build_experiment

    raw = _minimal_experiment_raw()
    raw["case_data"] = {"forcing": ["x.grib"]}
    with pytest.raises(ValueError) as refusal:
        build_experiment(raw, source="<test>")
    message = str(refusal.value)
    assert "does not have a table" not in message
    assert "case_data" in message
    assert "PRESENT" in message


def test_a_genuinely_unknown_table_keeps_the_absent_refusal():
    from gpuwm.experiment import build_experiment

    raw = _minimal_experiment_raw()
    raw["case_dtaa"] = {"forcing": ["x.grib"]}
    with pytest.raises(ValueError, match="does not have a table"):
        build_experiment(raw, source="<test>")


def test_load_experiment_validates_the_case_table_it_detaches(tmp_path):
    """Detached is not dropped: a [case_data] with an unknown key must
    still refuse through load_experiment, exactly as the case loader
    itself would refuse it."""
    out = _run_wizard(tmp_path, "era5", "1999-05-03T12")
    text = out.read_text(encoding="utf-8")
    text = text.replace("sfcp_to_sfcp", "sfcp_to_sfpc")
    broken = tmp_path / "broken.toml"
    broken.write_text(text, encoding="utf-8")
    from gpuwm.experiment import load_experiment
    with pytest.raises(ValueError, match="case_data"):
        load_experiment(broken)
