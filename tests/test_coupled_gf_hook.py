from pathlib import Path
import importlib.util

import pytest


spec = importlib.util.spec_from_file_location(
    "hook_gf_inputs", Path(__file__).parents[1] / "tools/wrf_exact_localize/hook_gf_inputs.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def test_missing_anchor_preserves_source(tmp_path):
    path = tmp_path / "phys/module_cumulus_driver.F"
    path.parent.mkdir()
    path.write_text("CALL GFDRV()\n")
    with pytest.raises(ValueError, match="one stock"):
        hook.hook(tmp_path)
    assert path.read_text() == "CALL GFDRV()\n"


def test_hook_keeps_call_and_rejects_repeat(tmp_path):
    path = tmp_path / "phys/module_cumulus_driver.F"
    path.parent.mkdir()
    original = "prefix\n         CALL GFDRV(spp_conv,pattern_spp_conv,field_conv,       &\nsuffix\n"
    path.write_text(original)
    hook.hook(tmp_path)
    modified = path.read_text()
    assert modified.count("CALL locdumpk(") == 24
    assert modified.index("CALL locstage('cu_gf_input')") < modified.index("CALL GFDRV(")
    assert modified.endswith(original.split("prefix\n", 1)[1])
    with pytest.raises(ValueError, match="already installed"):
        hook.hook(tmp_path)
    assert path.read_text() == modified
