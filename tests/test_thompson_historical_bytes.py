"""Nine generic emitted-config byte controls from the original Thompson parent."""
from datetime import datetime
import tomllib

import pytest

from gpuwm.domain_wizard import render_config
from test_audit_parent_generic_bytes import assert_old_bytes, controls


PARENT = "7ab2e3dcf59acb668015e3172feefc0418ee91b3"
CASES = (
    "namelist/omitted_legacy_defaults", "namelist/explicit_vertical_order3",
    "namelist/ordinary_fixed60_triplet", "namelist/ruc_monthly_omitted",
    "namelist/ruc_monthly_false0", "recipe/gfs", "recipe/era5", "recipe/rap", "recipe/rrfs",
)


@pytest.mark.parametrize("key", CASES)
def test_thompson_generic_config_bytes_match_original_parent(key, tmp_path):
    matrix, pins = controls(PARENT)
    if key.startswith("namelist/"):
        from gpuwm.namelist_import import import_namelists
        documents = matrix["namelists"][key.split("/", 1)[1]]
        wps, inp = tmp_path / "namelist.wps", tmp_path / "namelist.input"
        wps.write_text(documents["wps"], encoding="utf-8")
        inp.write_text(documents["input"], encoding="utf-8")
        emitted, _ = import_namelists(wps, inp)
    else:
        recipe = matrix["recipes"]
        emitted = render_config(
            name=recipe["name"], start_time=datetime.fromisoformat(recipe["start_time"]),
            hours=recipe["hours"], projection=recipe["projection"],
            dims=[tuple(row) for row in recipe["dims"]], ratios=tuple(recipe["ratios"]),
            root_dx_m=recipe["root_dx_m"], fetch_hints={"source": key.split("/", 1)[1]},
            case_data=None, profile=recipe["profile"],
        )
    shared = tomllib.loads(emitted)["shared"]
    assert "thompson_version" not in shared
    assert "thompson_fork_snow_fall" not in shared
    assert_old_bytes(PARENT, pins, key, emitted)
