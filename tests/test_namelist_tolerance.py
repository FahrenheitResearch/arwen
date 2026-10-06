"""Synthetic WPS metadata, Registry omissions and WRF convention regressions."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import tomllib

import pytest

from gpuwm.namelist_contract import _base_pair
from gpuwm.namelist_import import (
    _NAMELIST_DEFAULTS, import_namelists, import_parsed_namelists,
)
from test_namelist_import import INPUT_TEXT, WPS_TEXT, _pair


def _import_parsed(wps, inp):
    return import_parsed_namelists(wps, inp, wps_path="namelist.wps",
                                   input_path="namelist.input")


def test_gui_and_utility_sections_and_omitted_mixing_import(tmp_path):
    auxiliary = """
&domain_wizard
 gui_version = 'synthetic',
/
&mod_levs
 press_pa = 95000, 85000,
/
&plotfmt
 ix = 12, jx = 15,
/
"""
    inp = INPUT_TEXT.replace(" mix_full_fields = .true., .true.,\n", "")
    text, report = import_namelists(*_pair(tmp_path, wps=WPS_TEXT + auxiliary, inp=inp))
    assert tomllib.loads(text)["shared"]["km_opt"] == 4
    auxiliary_keys = {(entry.section, entry.key): entry for entry in report.dropped}
    for pair in (("domain_wizard", "gui_version"), ("mod_levs", "press_pa"),
                 ("plotfmt", "ix"), ("plotfmt", "jx")):
        assert "do not read" in auxiliary_keys[pair].reason
    assert "mix_full_fields" in report.format()
    assert "Namelist defaults" in report.format()
    assert "established full-field mixing resolution" in report.format()


def test_genuinely_unknown_wps_section_is_named(tmp_path):
    with pytest.raises(ValueError, match="unknown section.*unrecognized_extension"):
        import_namelists(*_pair(tmp_path, wps=WPS_TEXT +
                               "\n&unrecognized_extension\n knob=1,\n/\n"))


def _registry_default_pair():
    wps, inp = deepcopy(_base_pair())
    for document in (wps, inp):
        for entries in document.values():
            for key, values in entries.items():
                if key != "eta_levels":
                    entries[key] = values[:1]
    wps["share"]["max_dom"] = inp["domains"]["max_dom"] = [1]
    wps["share"]["start_date"] = ["1993-03-13_00:00:00"]
    wps["share"]["end_date"] = ["1993-03-14_00:00:00"]
    wps["geogrid"].update(e_we=[32], e_sn=[32], i_parent_start=[1], j_parent_start=[1])
    inp["domains"]["eta_levels"] = [1 - index / 30 for index in range(31)]
    inp["time_control"].update(start_hour=[0], end_hour=[0])
    for key in ("run_days", "run_hours", "run_minutes", "run_seconds"):
        inp["time_control"].pop(key, None)
    for (role, section), defaults in _NAMELIST_DEFAULTS.items():
        assert role == "input"
        for key in defaults:
            inp[section].pop(key, None)
    wps["geogrid"].pop("parent_id")
    wps["geogrid"].pop("truelat2")
    return wps, inp


def test_all_seventeen_formerly_required_defaults_import_together():
    wps, inp = _registry_default_pair()
    originals = deepcopy((wps, inp))
    text, report = _import_parsed(wps, inp)
    tables = tomllib.loads(text)
    # The seventeen FILLED defaults, and the one that is recorded without
    # being filled: an omitted use_theta_m, whose default depends on the
    # WRF line the namelist was written for (the translation resolves it;
    # the row says which Registry answered and what chose the line).
    filled = [entry for entry in report.namelist_defaults
              if entry.key != "use_theta_m"]
    assert len(filled) == 17
    assert len(report.namelist_defaults) == 18
    assert tables["domain"][0]["nx"] == 31
    assert tables["shared"]["nz"] == 30
    assert tables["experiment"]["start_time"] == datetime(1993, 3, 13)
    assert (wps, inp) == originals


def test_default_dimensions_do_not_hide_geometry_mismatch():
    wps, inp = _base_pair()
    inp["domains"].pop("e_we")
    with pytest.raises(ValueError, match="nest layout mismatch for e_we"):
        _import_parsed(wps, inp)


def test_missing_required_inventory_names_every_key_and_concrete_reason():
    wps, inp = _base_pair()
    wps["geogrid"].pop("dx")
    inp["dynamics"].pop("km_opt")
    inp["physics"].pop("mp_physics")
    with pytest.raises(ValueError) as caught:
        _import_parsed(wps, inp)
    message = str(caught.value)
    for key in ("dx", "km_opt", "mp_physics"):
        assert key in message
    assert "grid spacing" in message
    assert "turbulence closure" in message
    assert "physics_suite" in message
    assert "add every listed key" in message


def test_output_controls_index_origins_and_self_parent_do_not_refuse(tmp_path):
    inp = INPUT_TEXT.replace("parent_id = 0, 1", "parent_id = 1, 1")
    inp = inp.replace("&time_control", """&time_control
 write_input = .true.,
 input_outname = 'aux_<domain>',
 inputout_begin_h = 0,
 inputout_end_h = 6,
 inputout_interval = 60,
""")
    inp = inp.replace("&domains", "&domains\n s_we=1, s_sn=1, s_vert=1,\n")
    wps = WPS_TEXT.replace("&share", "&share\n opt_output_from_geogrid_path='./geo',\n")
    text, report = import_namelists(*_pair(tmp_path, wps=wps, inp=inp))
    reference, _ = import_namelists(*_pair(tmp_path))
    assert text == reference
    dropped = {entry.key for entry in report.dropped}
    assert {"write_input", "input_outname", "inputout_begin_h", "inputout_end_h",
            "inputout_interval", "opt_output_from_geogrid_path", "parent_id (root)"} <= dropped


def test_wps_templates_and_inactive_dates_use_concrete_input_clock(tmp_path):
    templated = WPS_TEXT.replace(
        "start_date = '1999-05-03_12:00:00', '1999-05-03_12:00:00',",
        "start_date = '@year@-@month@-@day@_@hour@:00:00', "
        "'1999-05-03_12:00:00', 'inactive',")
    text, report = import_namelists(*_pair(tmp_path, wps=templated))
    reference, _ = import_namelists(*_pair(tmp_path))
    assert text == reference
    assert {"start_date (templates)", "start_date (inactive domains)"} <= {
        entry.key for entry in report.dropped}


def test_concrete_wps_date_mismatch_still_refuses(tmp_path):
    wps = WPS_TEXT.replace("1999-05-03_12:00:00", "1999-05-04_12:00:00")
    with pytest.raises(ValueError, match="per-domain start mismatch"):
        import_namelists(*_pair(tmp_path, wps=wps))


@pytest.mark.parametrize("key", ["s_we", "s_sn", "s_vert"])
def test_nonstandard_active_grid_origin_still_refuses(tmp_path, key):
    inp = INPUT_TEXT.replace("&domains", f"&domains\n {key}=2,\n")
    with pytest.raises(ValueError, match="shift its extent"):
        import_namelists(*_pair(tmp_path, inp=inp))


@pytest.mark.parametrize("count", [10**12, "invalid", True])
def test_invalid_domain_count_refuses_before_default_array_expansion(count):
    wps, inp = _registry_default_pair()
    inp["domains"]["max_dom"] = [count]
    with pytest.raises(ValueError, match="max_dom.*refusing to expand"):
        _import_parsed(wps, inp)


def test_positive_run_length_ignores_partial_end_columns(tmp_path):
    inp = INPUT_TEXT.replace(" end_year = 1999, 1999,\n", "")
    text, report = import_namelists(*_pair(tmp_path, inp=inp))
    reference, _ = import_namelists(*_pair(tmp_path))
    assert text == reference
    assert not any(entry.key.startswith("end_") for entry in report.namelist_defaults)
    assert any(entry.key == "end_month" and "not read" in entry.reason
               for entry in report.dropped)
