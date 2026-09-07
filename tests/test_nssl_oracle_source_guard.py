"""A wrong WRF source must stop before any oracle build is created."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_nssl_oracle_refuses_changed_source_before_creating_build(tmp_path):
    if sys.platform != "linux" or not shutil.which("bash"):
        pytest.skip("the standalone WRF oracle builder uses Linux shell tools")
    source = tmp_path / "wrf" / "phys" / "module_mp_nssl_2mom.F"
    source.parent.mkdir(parents=True)
    source.write_text("module edited_nssl\nend module\n", encoding="ascii")
    build = tmp_path / "new-build"
    script = Path(__file__).resolve().parents[1] / "tools/nssl2_wrf461_oracle/build.sh"
    result = subprocess.run(["bash", str(script), str(source.parents[1]), str(build)],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 2
    assert "unexpected module_mp_nssl_2mom.F SHA-256" in result.stderr
    assert not build.exists()
