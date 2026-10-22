"""Preparation-only staging must carry vertical metadata without CUDA execution."""
import subprocess,sys,hashlib
from pathlib import Path

def test_preparation_stage_carries_family_bounds_without_the_executor(tmp_path,monkeypatch):
    from tools import build_rw_wps_release as release
    monkeypatch.setattr(release,'_require_tracked',lambda path:None)
    stage=tmp_path/'standalone'
    report=release._stage_rw_wps_python_project(stage)
    leaf='gpuwm/core/wsm_family_constants.py'
    table='gpuwm/data/wsm_family/schemes.json'
    assert leaf in report['files'] and table in report['files']
    assert not (stage/'gpuwm/core/wsm_family.py').exists()
    assert not (stage/'gpuwm/smoke_execution.py').exists()
    assert not (stage/'gpuwm/companion_setups.py').exists()
    assert (stage/'gpuwm/smoke_receipts.py').is_file()
    root=Path(__file__).resolve().parents[1]
    assert hashlib.sha256((stage/table).read_bytes()).digest()==hashlib.sha256((root/table).read_bytes()).digest()
    code='import sys; sys.path.insert(0,sys.argv[1]); from gpuwm.physics_vertical_contract import WSM7_VERTICAL_LEVEL_BOUNDS; assert WSM7_VERTICAL_LEVEL_BOUNDS==(2,80); assert "gpuwm.core.wsm_family" not in sys.modules; assert "cupy" not in sys.modules'
    r=subprocess.run([sys.executable,'-I','-c',code,str(stage)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
