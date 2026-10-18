import importlib.util
from pathlib import Path
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / f"tools/wrf_exact_localize/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_heating_dump_keeps_the_installed_step_end_and_rejects_repeat(tmp_path):
    path = tmp_path / "dyn_em/solve_em.F"
    path.parent.mkdir()
    path.write_text("     LOCIT = 0\n     CALL locstage('z_end')\nsuffix\n")
    hook = load("hook_mp_heating")
    hook.hook(tmp_path)
    source = path.read_text()
    assert "grid%itimestep, 9, 0)" in source
    assert source.endswith("suffix\n")
    with pytest.raises(ValueError, match="fresh"):
        hook.hook(tmp_path)
    assert path.read_text() == source


def test_terrain_hook_validates_both_files_before_writing(tmp_path):
    surface = tmp_path / "phys/module_surface_driver.F"
    part2 = tmp_path / "dyn_em/module_first_rk_step_part2.F"
    surface.parent.mkdir()
    part2.parent.mkdir()
    original = "      CALL locstage('p20_noah_in')\n"
    surface.write_text(original)
    part2.write_text("no installed coupling hook\n")
    with pytest.raises(ValueError, match="fresh p40"):
        load("hook_coupled_terrain").hook(tmp_path)
    assert surface.read_text() == original
