"""Explicit Thompson choices survive a generated fork-content namelist."""
import tomllib

import pytest

from gpuwm.companion_domains import candidate_wps_text
from gpuwm.experiment import build_experiment
from gpuwm.hrrr_route_inputs import render_namelist_input
from gpuwm.namelist_import import import_namelists
from gpuwm.physics_source_defaults import read_physics_selector_comment
from test_namelist_import import INPUT_TEXT, _pair


CHOICES = (
    {"thompson_version": "wrf_461", "thompson_fork_snow_fall": "blend"},
    {"thompson_version": "wrf_39_noaa", "thompson_fork_snow_fall": "blend"},
    {"thompson_version": "wrf_39_noaa", "thompson_fork_snow_fall": "wrf_39_noaa"},
)


@pytest.mark.parametrize("choices", CHOICES)
@pytest.mark.parametrize("signature", ("albedo", "factor2"))
def test_actual_generated_fork_content_preserves_explicit_thompson_choices(tmp_path, choices, signature):
    text = INPUT_TEXT.replace("mp_physics = 55, 55", "mp_physics = 28, 28")
    if signature == "albedo":
        text = text.replace("&physics\n", "&physics\n alb_sol = 1,\n")
    else:
        text = text.replace("&dynamics\n", "&dynamics\n diff_6th_factor2 = 0.04, 0.04,\n")
    emitted, _ = import_namelists(*_pair(tmp_path, inp=text))
    raw = tomllib.loads(emitted)
    raw["shared"].update(choices)
    exp = build_experiment(raw, source="explicit Thompson generation")
    wps, inp = tmp_path / "emitted.wps", tmp_path / "emitted.input"
    wps.write_text(candidate_wps_text(raw, exp, exp, tmp_path / "emitted.toml"), encoding="utf-8")
    generated = render_namelist_input(exp)
    carried = read_physics_selector_comment(generated)
    assert {key: carried[key] for key in choices} == choices
    inp.write_text(generated, encoding="utf-8")
    restored_text, _ = import_namelists(wps, inp,
        rrtmg_variant=exp.root.run.ra_rrtmg_variant,
        rrtmg_compatibility=exp.root.run.wrf_rrtmg_compatibility)
    restored = build_experiment(tomllib.loads(restored_text), source="actual generated Thompson reimport")
    for domain in restored.domains:
        assert {key: getattr(domain.run, key) for key in choices} == choices
