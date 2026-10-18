"""Captured radar inputs keep their full volume and rounded temperature."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/"tools/coupled_reflectivity_probe.py"
spec=importlib.util.spec_from_file_location("radar_probe",path)
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_radar_capture_has_complete_volume_and_exact_temperature(tmp_path):
    shape=(3,4,5)
    a=np.arange(60,dtype=np.float32).reshape(shape)
    theta=np.full(shape,np.float32(301.03))
    pii=np.full(shape,np.float32(.8173))
    np.savez(tmp_path/"0001_s001_rk3_it04_mp_wsm6_out.npz",
             x_qv=a,x_qr=a+1,x_qs=a+2,x_qg=a+3,
             x_theta=theta,x_pii=pii,x_pressure=a+10000)
    got=probe.capture(tmp_path,"woof",shape)
    for v,w in zip(got,(a,a+1,a+2,a+3,theta*pii,a+10000)):
        assert np.array_equal(v.view(np.uint32),w.view(np.uint32))
    with pytest.raises(ValueError,match="complete float32 domain"):
        probe.capture(tmp_path,"woof",(3,4,4))
