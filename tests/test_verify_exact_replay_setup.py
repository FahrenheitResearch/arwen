"""The CPU half of `gpuwm verify-exact`'s WOOF arm: how a recorded run is staged.

What breaks without these: a combo's WOOF-only settings (the RUC lineage
switches, the RRTMG variant, acknowledgements) silently not applied, so a
replay scores a configuration the combo list never asked for and every
mismatch is a settings error reported as an engine error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm.verify_exact.recording import ReplayRun
from gpuwm.verify_exact.replay import (WoofRefused, set_toml_key, stage_run_directory,
                                       time_step_seconds)

TOML = """[experiment]
name = "x"

[shared]
mp_physics = 6
ruc_soilprop = "wrf45"

[[domain]]
radt = 1.0
"""

NAMELIST = """&domains
 time_step = 7,
 time_step_fract_num = 1,
 time_step_fract_den = 2,
/
&physics
 ra_lw_physics                = 0,
 ra_sw_physics                = 90,
/
"""


def test_settings_replace_existing_keys_and_land_new_ones_in_shared():
    text = set_toml_key(TOML, "ruc_soilprop", "wrf461")
    text = set_toml_key(text, "bl_mynn_version", "wrf461")
    text = set_toml_key(text, "radt", 2.0)
    assert 'ruc_soilprop = "wrf461"' in text and "wrf45" not in text
    assert text.index('bl_mynn_version = "wrf461"') > text.index("[shared]")
    assert "radt = 2.0" in text
    assert set_toml_key(TOML, "use_x", True).count("use_x = true") == 1


def test_a_toml_without_shared_cannot_take_a_new_setting():
    with pytest.raises(ValueError, match=r"\[shared\]"):
        set_toml_key("[experiment]\n", "x", 1)


def _run(tmp_path, *, door="namelist import", settings=None, inputs=True):
    rec = tmp_path / "rec"
    (rec / "inputs").mkdir(parents=True, exist_ok=True)
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        (rec / "inputs" / name).write_bytes(b"bytes")
    (rec / "namelist.input").write_text(NAMELIST)
    return ReplayRun("c", "D001", {"id": "D001", "woof_door": door,
                                   "woof_settings": settings or {}},
                     rec / "inputs" if inputs else None, rec / "namelist.input")


def test_the_toml_door_imports_radiation_one_and_keeps_the_input_bytes(tmp_path):
    staged = stage_run_directory(_run(tmp_path, door="toml: no mapping"), tmp_path / "s")
    text = (staged / "namelist.input").read_text()
    assert "ra_lw_physics = 1," in text and "ra_sw_physics = 1," in text
    assert (staged / "wrfinput_d01").read_bytes() == b"bytes"


def test_acknowledged_longwave_off_imports_longwave_one(tmp_path):
    staged = stage_run_directory(
        _run(tmp_path, settings={"acknowledgements": ["lw-off"]}), tmp_path / "s")
    text = (staged / "namelist.input").read_text()
    assert "ra_lw_physics = 1," in text and "ra_sw_physics                = 90," in text


def test_a_run_without_inputs_is_refused_by_name(tmp_path):
    with pytest.raises(WoofRefused, match="real.exe"):
        stage_run_directory(_run(tmp_path, inputs=False), tmp_path / "s")


def test_fractional_time_steps(tmp_path):
    run = _run(tmp_path)
    assert time_step_seconds(Path(run.namelist)) == 7.5
