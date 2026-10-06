"""Explicit disabled/default values preserve the historical emitted bytes."""
import tomllib

from gpuwm.experiment import build_experiment
from gpuwm.namelist_import import import_namelists
from test_namelist_import import _pair, INPUT_TEXT


def _public_ruc_input():
    return (INPUT_TEXT.replace("mp_physics = 55, 55", "mp_physics = 8, 8")
            .replace("sf_sfclay_physics = 91, 91", "sf_sfclay_physics = 5, 5")
            .replace("sf_surface_physics = 2, 2", "sf_surface_physics = 3, 3")
            .replace("bl_pbl_physics = 11, 11", "bl_pbl_physics = 5, 5")
            .replace(" bldt = 0, 0,", " num_soil_layers = 9,\n bldt = 0, 0,"))


def test_explicit_fractional_seaice_zero_matches_the_omitted_configuration(tmp_path):
    control = _public_ruc_input()
    omitted, _ = import_namelists(*_pair(tmp_path, inp=control))
    explicit, _ = import_namelists(*_pair(tmp_path, inp=control.replace(
        "&physics\n", "&physics\n fractional_seaice = 0,\n")))
    assert explicit == omitted
    assert "fractional_seaice" not in tomllib.loads(explicit)["shared"]
    assert build_experiment(tomllib.loads(explicit), source="default sea ice").root.run.fractional_seaice == 0


def test_explicit_fractional_seaice_one_still_selects_the_fractional_branch(tmp_path):
    text, _ = import_namelists(*_pair(tmp_path, inp=_public_ruc_input().replace(
        "&physics\n", "&physics\n fractional_seaice = 1,\n")))
    assert tomllib.loads(text)["shared"]["fractional_seaice"] == 1
    assert build_experiment(tomllib.loads(text), source="fractional sea ice").root.run.fractional_seaice == 1


def test_explicit_disabled_microphysics_zero_out_matches_omitted_bytes(tmp_path):
    omitted, _ = import_namelists(*_pair(tmp_path))
    text = INPUT_TEXT.replace("&physics\n", "&physics\n mp_zero_out = 0,\n"
                              " mp_zero_out_all = 0,\n mp_zero_out_thresh = 1e-8,\n")
    explicit, _ = import_namelists(*_pair(tmp_path, inp=text))
    assert explicit == omitted
