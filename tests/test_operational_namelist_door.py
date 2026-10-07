"""The unmodified operational 3 km namelist imports.

The file (NOAA-EMC/HRRR v4.1.21 parm/conus/hrrr_wrf.nl, pinned by hash)
carries groups for machinery WOOF does not run: WRF-Chem smoke, the
digital filter, MPI log routing, the fork's extra diagnostics and its
cycle-state switch.  Each is read whole: inert when off, a declared
divergence the report prints when on.  Nothing in it describes a forecast
WOOF would get wrong, so nothing in it refuses.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tomllib

import pytest

from gpuwm.namelist_import import NamelistRefusal, import_namelists

ROOT = Path(__file__).resolve().parent
OPERATIONAL = ROOT / "data" / "hrrr_wrf_v4_1_21.nl"
OPERATIONAL_SHA256 = (
    "50ac01dbeaca863dfc313eae7dd53865458b2bffdfcc1e402d350d860bef5694")
WPS_FIXTURE = ROOT / "fixtures" / "source_requests" / "hrrr_namelist.wps.c18"


def _wps(tmp_path: Path) -> Path:
    # The WPS fixture describes the same 1800 x 1060 3 km grid; only its
    # dates are aligned with the namelist's own start/end.
    text = WPS_FIXTURE.read_text()
    text = text.replace("2026-09-29_12:00:00", "2018-08-26_12:00:00")
    text = text.replace("2026-09-30_06:00:00", "2018-08-26_18:00:00")
    path = tmp_path / "namelist.wps"
    path.write_text(text)
    return path


def _input(tmp_path: Path, edit=None) -> Path:
    data = OPERATIONAL.read_bytes()
    if edit is None:
        return OPERATIONAL
    path = tmp_path / "hrrr_wrf.nl"
    path.write_text(edit(data.decode()))
    return path


def test_unmodified_operational_namelist_imports(tmp_path):
    assert hashlib.sha256(OPERATIONAL.read_bytes()).hexdigest() \
        == OPERATIONAL_SHA256
    text, report = import_namelists(_wps(tmp_path), OPERATIONAL)
    doc = tomllib.loads(text)
    root = doc["domain"][0]
    assert (root["nx"], root["ny"]) == (1799, 1059)
    # namelist.input writes 0 for the head grid's parent start, WPS 1;
    # the head grid has no parent, so WPS's value stands.
    assert (root["i_parent_start"], root["j_parent_start"]) == (1, 1)
    divergences = {s.key: s for s in report.substitutions if s.reason}
    for key in ("chem_opt", "gsd_diagnostics", "cycling", "hailcast_opt",
                "ra_sw_eclipse", "prec_acc_dt", "prec_acc_dt1",
                "interp_type", "lagrange_order", "rebalance",
                "seaice_threshold"):
        assert key in divergences, key
    # The smoke package asks for its direct radiative feedback; the
    # divergence says that feedback is absent, not that nothing changes.
    assert "simple_dir_fdb" in divergences["chem_opt"].reason
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    for section, key in (("dfi_control", "dfi_opt"),
                         ("dfi_control", "dfi_nfilter"),
                         ("domains", "time_step_dfi"),
                         ("dynamics", "km_opt_dfi"),
                         ("dynamics", "moist_adv_dfi_opt"),
                         ("logging", "stderr_logging"),
                         ("chem", "biomass_burn_opt"),
                         ("time_control", "history_interval2"),
                         ("time_control", "diag_int"),
                         ("physics", "co2tf"),
                         ("physics", "maxens"),
                         ("dynamics", "tke_mix6_off")):
        assert (section, key) in dropped, (section, key)
    assert "dfi_opt = 0" in dropped[("dfi_control", "dfi_opt")]
    fixed = {(f.section, f.key) for f in report.fixed}
    assert ("physics", "grav_settling") in fixed
    assert ("domains", "zap_close_levels") in fixed
    assert ("dynamics", "scalar_mix6_off") in fixed


def test_requested_digital_filter_is_a_declared_divergence(tmp_path):
    path = _input(tmp_path, lambda t: t.replace(
        "dfi_opt                             = 0,",
        "dfi_opt                             = 3,"))
    _, report = import_namelists(_wps(tmp_path), path)
    divergences = {s.key: s for s in report.substitutions if s.reason}
    assert "unfiltered" in divergences["dfi_opt"].gpuwm_name
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    assert "declared divergence" in dropped[("domains", "time_step_dfi")]


def test_chemistry_off_is_inert(tmp_path):
    path = _input(tmp_path, lambda t: t.replace(
        "chem_opt                            = 18,",
        "chem_opt                            = 0,"))
    _, report = import_namelists(_wps(tmp_path), path)
    assert "chem_opt" not in {s.key for s in report.substitutions}
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    assert "chem_opt = 0" in dropped[("chem", "biomass_burn_opt")]


def test_history_cadence_switch_inside_the_run_is_declared(tmp_path):
    path = _input(tmp_path, lambda t: t.replace(
        "history_interval_change             = 1080,",
        "history_interval_change             = 120,"))
    _, report = import_namelists(_wps(tmp_path), path)
    assert "history_interval2" in {s.key for s in report.substitutions}


def test_requested_fog_settling_still_refuses_under_mp28(tmp_path):
    # WRF overwrites grav_settling = 1 to 0 under mp_physics = 28; the
    # refusal names that overwrite.  0 is what WRF runs and imports.
    path = _input(tmp_path, lambda t: t.replace(
        "grav_settling                       = 0,",
        "grav_settling                       = 1,"))
    with pytest.raises(NamelistRefusal, match="grav_settling"):
        import_namelists(_wps(tmp_path), path)
