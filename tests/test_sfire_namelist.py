"""Fire registry columns reach the public experiment import door."""
from dataclasses import fields
from pathlib import Path
import re

import pytest

from gpuwm.experiment import load_experiment
from gpuwm.namelist_import import import_namelists
from gpuwm.sfire_config import FireRunFields
from test_namelist_import import INPUT_TEXT, _pair


def test_fire_registry_has_a_typed_config_field_for_every_input():
    path = Path(__file__).parents[1] / "tools/sfire_wrf471_oracle/reference/Registry/registry.fire"
    names = set(re.findall(r"(?m)^rconfig\s+\S+\s+(\S+)\s+namelist,fire\b", path.read_text()))
    assert names <= {field.name for field in fields(FireRunFields)}


def test_fire_activates_only_the_innermost_imported_domain(tmp_path):
    text = INPUT_TEXT.replace("&domains", "&domains\n sr_x=0,4, sr_y=0,3,") + """
&fire
 ifire=0,2, fire_fuel_read=0,0,
 fire_num_ignitions=0,1, fire_ignition_radius1=0,80,
 fire_ignition_start_x1=0,100, fire_ignition_start_y1=0,200,
 fire_ignition_ros1=0.01,0.05, fire_wind_height=1.,
/
"""
    translated, report = import_namelists(*_pair(tmp_path, inp=text))
    path = tmp_path / "fire.toml"
    path.write_text(translated)
    experiment = load_experiment(path)
    assert experiment.root.run.ifire == 0
    child = experiment.domain(2).run
    assert (child.ifire, child.sr_x, child.sr_y, child.fire_num_ignitions) == (2, 4, 3, 1)
    assert child.fire_ignition_start_x1 == 100 and child.fire_ignition_ros1 == 0.05
    # Omitted max_domains tails keep their Registry default, rather than
    # repeating the root's one-meter height.
    assert experiment.root.run.fire_wind_height == 1.0
    assert child.fire_wind_height == FireRunFields().fire_wind_height
    assert ("fire", "ifire") in {(row.section, row.key) for row in report.translated}


def test_fire_namelist_typo_is_not_silently_dropped(tmp_path):
    text = INPUT_TEXT + "\n&fire fire_wind_heigt=1. /\n"
    with pytest.raises(ValueError, match="fire_wind_heigt"):
        import_namelists(*_pair(tmp_path, inp=text))


@pytest.mark.parametrize("key", ["fire_fuel_left_irl", "fire_fuel_left_jrl"])
def test_native_fixed_subcell_geometry_is_checked_during_import(tmp_path, key):
    text = INPUT_TEXT.replace("&domains", "&domains\n sr_x=0,4, sr_y=0,3,") + f"""
&fire
 ifire=0,2, fire_fuel_read=0,0, {key}=2,4,
/
"""
    with pytest.raises(ValueError, match=key):
        import_namelists(*_pair(tmp_path, inp=text))
