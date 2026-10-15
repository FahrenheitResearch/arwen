"""The operational 3 km namelist at the door: its smoke maps, or is refused.

The file (NOAA-EMC/HRRR v4.1.21 parm/conus/hrrr_wrf.nl, pinned by hash)
carries groups for machinery WOOF does not run as WRF does: the digital
filter, MPI log routing, the fork's extra diagnostics -- each read whole,
inert when off, a declared divergence the report prints when on -- and the
fork's HRRR-Smoke chemistry (chem_opt = 18).  Since 2026-10-12 (the owner's
ruling) chemistry is never imported to run with none: the smoke maps onto
WOOF's ``smoke`` set, and the four smoke settings WOOF does not carry
(smoke dry deposition, the fork's enhanced smoke mixing, its flaming
fraction, and the smoke's direct radiative feedback) are refused by name,
all four in one refusal, beside its HAILCAST request (refused by name: the
package does not run on this line).  Its cycle-state switch (cycling = .true.) is run,
so it reaches [shared] cycling.
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

from operational_namelist_edits import importable


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
    data = OPERATIONAL.read_bytes().decode()
    text = importable(data)
    if edit is not None:
        text = edit(text)
    path = tmp_path / "hrrr_wrf.nl"
    path.write_text(text)
    return path


def test_unmodified_operational_namelist_refuses_exactly_the_uncarried_smoke(
        tmp_path):
    assert hashlib.sha256(OPERATIONAL.read_bytes()).hexdigest() \
        == OPERATIONAL_SHA256
    with pytest.raises(NamelistRefusal) as caught:
        import_namelists(_wps(tmp_path), OPERATIONAL)
    message = str(caught.value)
    assert "refuses 5 setting(s)" in message
    for key in ("aer_drydep_opt", "enh_vermix", "flam_part",
                "simple_dir_fdb"):
        assert f"&chem {key} = " in message, key
    assert "&physics hailcast_opt = " in message
    assert "shortwave would not be reduced under smoke" in message


def test_operational_namelist_imports_with_its_smoke_mapped(tmp_path):
    text, report = import_namelists(_wps(tmp_path), _input(tmp_path))
    doc = tomllib.loads(text)
    root = doc["domain"][0]
    assert (root["nx"], root["ny"]) == (1799, 1059)
    # namelist.input writes 0 for the head grid's parent start, WPS 1;
    # the head grid has no parent, so WPS's value stands.
    assert (root["i_parent_start"], root["j_parent_start"]) == (1, 1)
    shared = doc["shared"]
    # The smoke runs: WOOF's smoke set, its fire emission and HRRR-Smoke's
    # own smoke as start and boundary state, with the namelist's washout
    # and hourly plume.
    assert shared["chem_sets"] == "smoke"
    assert shared["chem_sources"] == "rave-3km,hrrr-native-smoke"
    assert shared["wetscav_onoff"] == -1
    assert shared["plumerisefire_frq"] == 60
    divergences = {s.key: s for s in report.substitutions if s.reason}
    assert "chem_opt" not in divergences
    fixed = {(f.section, f.key): f for f in report.fixed}
    assert fixed[("chem", "chem_opt")].fixed_value == "smoke"
    # The namelist's cycling = .true. is run, not substituted: both MYNN
    # generations keep the input QKE, QC_BL and CLDFRA_BL on a cycled
    # start, so the import must not report a fresh start.
    assert shared["cycling"] is True
    assert "cycling" not in {s.key for s in report.substitutions}
    assert ("time_control", "cycling") not in {
        (d.section, d.key) for d in report.dropped}
    for key in ("gsd_diagnostics",
                "prec_acc_dt", "prec_acc_dt1",
                "interp_type", "lagrange_order", "rebalance",
                "seaice_threshold"):
        assert key in divergences, key
    # ra_sw_eclipse = 1 is exact here, not a divergence: WRF's eclipse table
    # has no eclipse on 2018-08-26, so WRF computes zero obscuration at
    # every call (tests/test_namelist_door_rule.py refuses a run that has
    # one).
    assert "ra_sw_eclipse" not in divergences
    assert "no eclipse" in fixed[("physics", "ra_sw_eclipse")].reason
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    for section, key in (("dfi_control", "dfi_opt"),
                         ("dfi_control", "dfi_nfilter"),
                         ("domains", "time_step_dfi"),
                         ("dynamics", "km_opt_dfi"),
                         ("dynamics", "moist_adv_dfi_opt"),
                         ("logging", "stderr_logging"),
                         ("chem", "phot_opt"),
                         ("chem", "simple_dir_fdb"),
                         ("time_control", "history_interval2"),
                         ("time_control", "diag_int"),
                         ("physics", "co2tf"),
                         ("dynamics", "tke_mix6_off")):
        assert (section, key) in dropped, (section, key)
    assert ("physics", "maxens") not in dropped
    assert "dfi_opt = 0" in dropped[("dfi_control", "dfi_opt")]
    assert ("physics", "grav_settling") in fixed
    assert ("domains", "zap_close_levels") in fixed
    assert ("dynamics", "scalar_mix6_off") in fixed


def test_requested_digital_filter_is_refused_by_name(tmp_path):
    # A declared divergence until 2026-10-12 ("forecast from the unfiltered
    # analysis"); the filter changes the state the forecast integrates from.
    path = _input(tmp_path, lambda t: t.replace(
        "dfi_opt                             = 0,",
        "dfi_opt                             = 3,"))
    with pytest.raises(NamelistRefusal) as caught:
        import_namelists(_wps(tmp_path), path)
    message = str(caught.value)
    assert "&dfi_control dfi_opt = " in message
    assert "Next: set dfi_opt = 0." in message


def test_chemistry_off_is_inert(tmp_path):
    path = _input(tmp_path, lambda t: t.replace(
        "chem_opt                            = 18,",
        "chem_opt                            = 0,"))
    text, report = import_namelists(_wps(tmp_path), path)
    assert "chem_opt" not in {s.key for s in report.substitutions}
    assert "chem_sets" not in tomllib.loads(text)["shared"]
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
