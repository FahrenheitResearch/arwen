"""The packaged HRRR v4 recipe runs no ocean model at any forecast length.

Breakage: a 48 h run of hrrr_v4_gsd41 took the 3D PWP ocean by the >= 48 h
default (gpuwm/data/surface_ocean.json), which operational HRRR does not run,
and on the CONUS grid its WOA18 fallback refused the start ("WOA18 monthly MLD
has no valid value within one degree of forecast ocean", race-B 2026-10-21).
"""

from pathlib import Path

from gpuwm.experiment import load_experiment

RECIPE = Path(__file__).resolve().parents[1] / "gpuwm" / "data" / "recipes" / "hrrr_v4_gsd41.toml"
EDITS = (("run_seconds = 7200.0", "run_seconds = 172800.0"),
         ("\nhours = 3\n", "\nhours = 48\n"),
         # A 06Z start pairs with the 03Z RAP cycle, which reaches f051.
         ('cycle = "2026-10-02T21"', 'cycle = "2026-10-21T06"'),
         ("start_time = 2026-10-02T21:00:00", "start_time = 2026-10-21T06:00:00"))


def test_a_48_hour_hrrr_recipe_keeps_the_ocean_off(tmp_path):
    text = RECIPE.read_text(encoding="utf-8")
    for old, new in EDITS:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / "hrrr48.toml"
    path.write_text(text, encoding="utf-8")
    experiment = load_experiment(path)
    assert experiment.domains[0].run.sf_ocean_physics == 0
