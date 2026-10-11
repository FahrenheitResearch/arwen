"""Physical cell-depth fallout and unbounded column workspace regressions."""
import numpy as np
import pytest
from tools.fast_sbm_host import host as H,replay as RP
from tools.fast_sbm_wrf480_oracle import fixtures as fx

@pytest.mark.parametrize("family",("kc1","kc2","kc2g"))
def test_fallout_conserves_stretched_cell_mass(family):
    case=RP.Case(family,"o64")
    step,pre,_=list(fx.iter_replay(f"{family}_replay_o64"))[-1]
    flags=H.flags_corrected(hail=0 if family=="kc2g" else 1)
    host=RP.host_for("o64",flags)
    state=case.state(pre.fields); ledger=np.zeros(10)
    status,_=host.call(step,12.,1000.,1000.,case.tile.dims(),state,ledger=ledger)
    assert status==0
    residual=ledger[3]+ledger[4]-ledger[2]-ledger[1]
    assert abs(residual)<2e-10,residual
    flags["sedimentation_cell_depth"]=0
    host=RP.host_for("o64",flags); state=case.state(pre.fields); ledger=np.zeros(10)
    status,_=host.call(step,12.,1000.,1000.,case.tile.dims(),state,ledger=ledger)
    assert status==0
    assert abs(ledger[3]+ledger[4]-ledger[2]-ledger[1])>1e-8

@pytest.mark.parametrize("nz",(2,200,201,257))
def test_column_workspace_has_no_200_level_stack_limit(nz):
    tile=H.Tile(1,2,1,2,1,nz+1,0,2,0,2,1,nz+1,1,1,1,1,1,nz)
    state=tile.blank(np.float64); state["dz8w"].fill(100.)
    host=RP.host_for("o64",H.flags_corrected())
    status,_=host.call(1,12.,1000.,1000.,tile.dims(),state)
    assert status==0
    assert all(np.all(np.isfinite(a)) for a in state.values())
    assert np.all(state["qv"]==0.)


@pytest.mark.parametrize("family",("kc1","kc2","kc2g"))
def test_three_hour_physical_budget_closes(family):
    from tools.fast_sbm_host.column_budget import run
    result=run(family,3.)
    assert result["steps"]==900
    assert result["all_fields_finite"]
    assert abs(result["relative_residual"])<1e-9,result
    flags=H.flags_corrected(hail=0 if family=="kc2g" else 1)
    flags["collision_mass_conserved"]=0
    witness=run(family,3.,flags=flags)
    assert abs(witness["relative_residual"])>1e-9,witness
