"""Rebuild the native Lin fixture and require every numerical word to survive."""
import os
from pathlib import Path
import shutil
import shlex
import sys
import subprocess
import numpy as np
import pytest

@pytest.mark.parametrize("long_version", [False, True])
def test_native_lin_oracle_rebuild_completes_and_preserves_fixture(tmp_path, long_version):
    root = Path(__file__).resolve().parents[1]
    wrf = Path(os.environ.get("WRF_SOURCE_ROOT", "/work/wrf/src/WRF-4.6.1"))
    if not (wrf / "phys/module_mp_lin.F").is_file() or shutil.which("gfortran") is None:
        pytest.skip("requires the pinned WRF source and gfortran")
    env = os.environ.copy()
    if long_version:
        compiler = shutil.which("gfortran")
        wrapper_dir = tmp_path / "bin"
        wrapper_dir.mkdir()
        wrapper = wrapper_dir / "gfortran"
        writer = "import sys; sys.stdout.write('compiler note\\n' * 20000)"
        wrapper.write_text("#!/bin/bash\nif [[ $1 == --version ]]; then\n" +
                           shlex.quote(compiler) + " --version\n" +
                           shlex.quote(sys.executable) + " -c " + shlex.quote(writer) +
                           "\nelse\nexec " + shlex.quote(compiler) + " \"$@\"\nfi\n")
        wrapper.chmod(0o755)
        env["PATH"] = str(wrapper_dir) + os.pathsep + env["PATH"]
    fixture = tmp_path / "rebuilt.npz"
    result = subprocess.run(["bash", str(root / "tools/lin_wrf461_oracle/build.sh"),
                             str(wrf), str(tmp_path / "build"), str(fixture)],
                            capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(fixture) as got, np.load(root / "tests/data/oracles/lin/lin-wrf461-columns.npz") as want:
        assert set(got.files) == set(want.files)
        for name in want.files:
            if name != "provenance":
                assert got[name].dtype == want[name].dtype, name
                assert got[name].tobytes() == want[name].tobytes(), name
