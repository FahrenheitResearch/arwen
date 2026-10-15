"""Allocation and history metadata for the real diagnostic carriers."""
from dataclasses import replace
import pytest
from gpuwm.config import RunConfig
from gpuwm.core.physics_inventory import cumulus_state_layout, CUMULUS_HISTORY_LAYOUT
from gpuwm.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS, HISTORY_FIELDS_BY_NETCDF_NAME


def test_diagnostics_are_scoped_and_have_real_output_schema():
    from gpuwm.core.g3 import run_g3_diag
    assert callable(run_g3_diag)
    cfg=RunConfig(nx=16,ny=16,nz=30,dx=12000.,dy=12000.,ztop=16000.,dt=60.,run_seconds=0.)
    assert not cumulus_state_layout(cfg)
    assert 'gd_cldfr' not in cumulus_state_layout(replace(cfg,cu_physics=5))
    layouts=cumulus_state_layout(replace(cfg,cu_physics=5,cu_diag=1,ensdim=272))
    assert layouts['g3_xf_ens']==((272,16,16),'f4')
    assert layouts['gd_cloud_a']==((30,16,16),'f4')
    assert layouts['g3_diag_clock']==((4,),'i4')
    for name in CUMULUS_HISTORY_LAYOUT:
        row=SCHEME_OUTPUT_FIELDS[name]
        assert HISTORY_FIELDS_BY_NETCDF_NAME[row.netcdf_name]==row
        assert row.wrf_history is False


def test_real_forecast_disk_review_prices_the_diagnostic_inventory():
    from gpuwm.io.history_layout import produced_history_shapes, history_frame_bytes
    cfg=RunConfig(nx=40,ny=40,nz=50,dx=3000.,dy=3000.,ztop=20000.,dt=15.,run_seconds=300.,moist=True,mp_physics=1,cu_physics=5,cudt_minutes=0.,cu_diag=1)
    shapes=produced_history_shapes(cfg)
    for name in ('GD_CLOUD','GD_CLOUD2','GD_CLOUD_A','GD_CLOUD2_A','QC_CU','QI_CU','GD_CLDFR'):
        assert shapes[name]==(50,40,40)
    assert shapes['RAINCV_A']==(40,40) and shapes['RAINCV_B']==(40,40)
    assert history_frame_bytes(cfg,None)>0


def test_native_integer_carriers_and_workspace_have_manifest_policies():
    import numpy as np
    from gpuwm.core.g3 import Grell3D
    from gpuwm.core.health import gpu_integer_policy
    from gpuwm.io.restart import _callable_state_check, CUMULUS_CALLABLE_ARRAYS, CUMULUS_REBUILT_ARRAYS, CUMULUS_CALLABLE_CONTAINERS
    for name in ('surface.g3_isfc','surface.g3_diag_clock'):
        assert gpu_integer_policy(name,np.dtype('i4')) is not None
    adapter=Grell3D()
    adapter._ws=np.zeros(1024,dtype='f4')
    _callable_state_check(adapter,CUMULUS_CALLABLE_ARRAYS | CUMULUS_REBUILT_ARRAYS,CUMULUS_CALLABLE_CONTAINERS,'cumulus')
    assert 'diagnostics' not in vars(adapter)


@pytest.mark.parametrize("boundary,expected", [
    ({"open_x":False,"open_y":True},(True,False)),
    ({"specified":True},(False,False)),
    ({"nested":True},(False,False)),
])
def test_driver_preserves_the_domain_boundary_flags(monkeypatch,boundary,expected):
    import sys
    import numpy as np
    from types import SimpleNamespace
    import gpuwm.core.g3 as g3
    class StopAtColumn(Exception): pass
    def capture(**kw):
        assert (kw['periodic_x'],kw['periodic_y']) == expected
        raise StopAtColumn
    monkeypatch.setitem(sys.modules,'cupy',np)
    monkeypatch.setattr(g3,'run_g3',capture)
    cfg=RunConfig(nx=16,ny=16,nz=30,dx=12000.,dy=12000.,ztop=16000.,dt=60.,run_seconds=0.,cu_physics=5,cudt_minutes=0.,**boundary)
    vol=np.ones((30,16,16),dtype='f4')
    state=SimpleNamespace(p=vol,w=np.ones((31,16,16),dtype='f4'),ht=np.zeros((16,16),dtype='f4'))
    atmosphere={key:vol for key in ('exner','u','v','temperature','qv','pressure','rho')}
    atmosphere['p_interface']=np.ones((31,16,16),dtype='f4')
    fields={'xland':np.ones((16,16),dtype='f4')}
    import pytest
    with pytest.raises(StopAtColumn):
        g3.Grell3D()(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)


@pytest.mark.parametrize("option", (0, 11))
def test_grell3d_publishes_wrfs_ktop_deep_for_pr92_cloud_top(monkeypatch, option):
    """lightning_option = 11 reads the cumulus scheme's deep-cloud top.

    G3DRV writes ktop_deep (module_cu_g3.F, graded word for word by
    tests/test_g3_cu5_parity.py); the driver must hand that plane to the
    ltng_ktop_deep slot lightning reads, or PR92z refuses for want of a cloud
    top. Off, the slot is never created.
    """
    import sys
    import numpy as np
    from types import SimpleNamespace
    import gpuwm.core.g3 as g3

    class Published(Exception):
        pass

    top = (np.arange(16 * 16, dtype=np.int32).reshape(16, 16) % 41)

    def column(**kw):
        isfc = np.zeros((len(g3.ISFC_OUT), 16, 16), dtype=np.int32)
        isfc[g3.ISFC_OUT.index("ktop_deep")] = top
        return SimpleNamespace(ws=kw["ws"], isfc=isfc, lev=None, sfc=None)

    class Fields(dict):
        # The first read after publication copies the column outputs into
        # the carriers; stop there, the publication is already done.
        def __getitem__(self, key):
            if key == "g3_lev":
                raise Published
            return super().__getitem__(key)

    slots = {}

    def scratch(shape, name, dtype=np.float32):
        slots[name] = np.full(shape, -7, dtype=dtype)
        return slots[name]

    monkeypatch.setitem(sys.modules, "cupy", np)
    monkeypatch.setattr(g3, "run_g3", column)
    cfg = RunConfig(nx=16, ny=16, nz=30, dx=12000., dy=12000., ztop=16000.,
                    dt=60., run_seconds=0., cu_physics=5, cudt_minutes=0.,
                    moist=True, mp_physics=8, lightning_option=option)
    vol = np.ones((30, 16, 16), dtype="f4")
    state = SimpleNamespace(p=vol, w=np.ones((31, 16, 16), dtype="f4"),
                            ht=np.zeros((16, 16), dtype="f4"), scratch=scratch)
    atmosphere = {key: vol for key in ("exner", "u", "v", "temperature", "qv", "pressure", "rho")}
    atmosphere["p_interface"] = np.ones((31, 16, 16), dtype="f4")
    with pytest.raises(Published):
        g3.Grell3D()(atmosphere=atmosphere, fields=Fields(xland=np.ones((16, 16), dtype="f4")),
                     state=state, cfg=cfg)
    if option == 11:
        assert slots["ltng_ktop_deep"].dtype == np.int32
        np.testing.assert_array_equal(slots["ltng_ktop_deep"], top)
    else:
        assert "ltng_ktop_deep" not in slots
