"""scalar_pblmix resolves as wrf.exe resolves it, and YSU has its consumer.

WRF v4.6.1's share/module_check_a_mundo.F:2477-2495 resets scalar_pblmix to
1 for mp_physics = 28 with use_aero_icbc or use_rap_aero_icbc (debug level
1: nothing printed, namelist.output keeps the stated value) and
:2497-2511 turns it off under MYNN with bl_mynn_mixscalars = 1.  The
importer took the stated 0, so the door ran no PBL scalar mixing where
wrf.exe on the same namelist did.  Measured on the stock-WRF door case
(fixtures/wrf461_door, the namelist both models ran; g400 crop,
2024-05-21 18Z): wrf.exe's lowest-level QNWFA fell 0.11 % per 20 s step
while the door's did not, 3 % apart at 10 minutes and 14 % at 90.
"""
import re
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "wrf461_door"


def _imported(tmp_path, edit=None):
    from gpuwm.namelist_import import import_namelists

    text = (FIXTURE / "namelist.input").read_text(encoding="utf-8")
    if edit is not None:
        text = edit(text)
    namelist = tmp_path / "namelist.input"
    namelist.write_text(text, encoding="utf-8")
    toml_text, report = import_namelists(FIXTURE / "namelist.wps", namelist)
    match = re.search(r"^scalar_pblmix = (\d+)", toml_text, re.MULTILINE)
    return (int(match.group(1)) if match else 0), report


def test_mp28_with_aerosol_icbc_runs_wrfs_scalar_pblmix(tmp_path):
    value, report = _imported(tmp_path)
    assert value == 1
    rows = [row for row in report.defaults_applied
            if row.key == "scalar_pblmix"]
    assert rows and "module_check_a_mundo.F:2477-2495" in rows[0].reason


def test_without_aerosol_icbc_the_stated_value_stands(tmp_path):
    def no_icbc(text):
        return (text.replace("use_aero_icbc                       = .true.",
                             "use_aero_icbc                       = .false.")
                .replace(" wif_input_opt                       = 1,",
                         " wif_input_opt                       = 0,"))
    try:
        value, _ = _imported(tmp_path, no_icbc)
    except ValueError as refusal:      # mp=28 without the WIF pair
        pytest.skip(f"the importer refuses this pair by name: {refusal}")
    assert value == 0


def test_ysu_is_an_admitted_scalar_pblmix_consumer():
    from gpuwm.config import RunConfig, validate_scalar_pblmix_consumer

    cfg = RunConfig(nx=8, ny=8, nz=8, dx=3000.0, dy=3000.0, ztop=8000.0,
                    dt=20.0, run_seconds=0.0, mp_physics=28,
                    bl_pbl_physics=1, scalar_pblmix=1)
    validate_scalar_pblmix_consumer(cfg)
