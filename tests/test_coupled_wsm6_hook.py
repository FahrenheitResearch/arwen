import importlib.util
from pathlib import Path
import pytest


spec = importlib.util.spec_from_file_location(
    "hook_wsm6_inputs", Path(__file__).parents[1] / "tools/wrf_exact_localize/hook_wsm6_inputs.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def test_hook_brackets_the_complete_continued_call(tmp_path):
    path = tmp_path / "phys/module_microphysics_driver.F"
    path.parent.mkdir()
    call = ("             CALL wsm6( &\n                  TH=th &\n"
            "#if ( WRF_CHEM == 1 )\n ,WETSCAV_ON=.true. &\n#endif\n"
            "                   )\n")
    path.write_text("prefix\n" + call + "suffix\n")
    hook.hook(tmp_path)
    source = path.read_text()
    assert source.count("CALL locdumpk(") == 22
    assert source.index("mp_wsm6_in") < source.index(call) < source.index("mp_wsm6_out")
    with pytest.raises(ValueError, match="fresh"):
        hook.hook(tmp_path)
    assert path.read_text() == source
