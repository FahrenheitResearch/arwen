"""The production-default mosaic UCM oracle families carry the shipped switches.

``ucm_table`` and ``ucm_lcz_table`` (run_mosaic_ucm.F90 families 5 and 6)
keep every urban switch exactly as WRF's urban_param_init reads it from
URBPARM.TBL / URBPARM_LCZ.TBL.  The GPU replay
(tests/test_noah_mosaic_ucm_wrf471_parity.py::
test_production_default_switches_every_output_word) is only a default-run
claim while these words stay the shipped tables' words; this CPU test pins it.
"""
from pathlib import Path

from gpuwm.core.urban_tables import load_urban_params
from gpuwm.core.urban_ucm import (
    UCM_GLOBAL_LAYOUT, UCM_SWITCHES, UCM_TABLE_COLUMNS,
    pack_params as pack_ucm, pack_params_from_rows)
from gpuwm.verify.noah_mosaic_oracle import load

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/noah_mosaic"


def test_table_families_carry_the_production_default_switches():
    for family, lcz in (("ucm_table", 0), ("ucm_lcz_table", 1)):
        steps = sorted(load(ROOT / family).values(), key=lambda f: int(f["itimestep"]))
        assert [int(f["itimestep"]) for f in steps] == [1, 2, 3, 4]
        first = steps[0]
        assert int(first["use_wudapt_lcz"]) == lcz
        table = {n: first["ucm_table_" + n] for n in (*UCM_TABLE_COLUMNS, "frc_urb")}
        switches = {n: first["ucm_" + n] for n in UCM_SWITCHES}
        switches.update({n: first["ucm_global_" + n] for n, _ in UCM_GLOBAL_LAYOUT})
        dumped = pack_params_from_rows(table, switches)
        shipped = pack_ucm(load_urban_params(1, lcz))
        assert dumped.switches.tobytes() == shipped.switches.tobytes(), family
        assert dumped.table.tobytes() == shipped.table.tobytes(), family
        assert dumped.globals_.tobytes() == shipped.globals_.tobytes(), family
        # The arms the forced-switch families never take.
        assert int(first["ucm_imp_scheme"]) == 1
        assert int(first["ucm_ch_scheme"]) == 2
        assert int(first["ucm_ahoption"]) == 0 and int(first["ucm_alhoption"]) == 0
