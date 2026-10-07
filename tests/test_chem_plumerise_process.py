"""Process declarations and table-driven frame bindings, with no device use."""
from types import SimpleNamespace
import numpy as np
import pytest
from gpuwm.core import chem_plumerise as plume
from gpuwm.core.chem_plumerise_cache import PlumeParameters


def context():
    row=SimpleNamespace(emissions=({'source':'reference-source','vertical':'plumerise'},))
    source=SimpleNamespace(variables={'power':{'process':plume.KEY,'input':'frp_inst'}})
    table=SimpleNamespace(rows_for=lambda key:(row,),sources={'reference-source':source})
    return SimpleNamespace(table=table,cfg=SimpleNamespace(),diag={},frames=None,clock=SimpleNamespace(curr_secs=36.))


def test_contract():
    assert plume.KEY=='plumerise.freitas'; assert plume.LEDGER=='emitted'
    assert {'rho','qv','z_at_w'}<=set(plume.REQUIRES)
    assert all(a.restart=='serialize' for a in plume.ALLOCATES)
    output=next(a for a in plume.ALLOCATES if a.output_name=='PLUME_TOP')
    assert output.shape=='2d' and output.units=='m'
    assert next(a for a in plume.ALLOCATES if a.name=='plume_k_min').dtype=='int32'
    assert len(plume.fire_group_properties())==4


def test_frame_roles_come_from_rows():
    ctx=context(); calls=[]; data=np.ones((2,3),dtype=np.float32)
    def at(source,field,time): calls.append((source,field,time)); return data
    ctx.frames=SimpleNamespace(at=at)
    assert plume.frame_inputs(ctx,'valid-hour')=={'frp_inst':data}
    assert calls==[('reference-source','power','valid-hour')]


def test_duplicate_frame_refusal():
    ctx=context();ctx.frames=SimpleNamespace(at=lambda *args:np.zeros((1,1),dtype=np.float32))
    ctx.table.sources['reference-source'].variables['duplicate']={'process':plume.KEY,'input':'frp_inst'}
    with pytest.raises(ValueError,match='double fire heating'): plume.frame_inputs(ctx,0)


def test_clock_gate_preserves_fields_without_device_use():
    ctx=context(); sentinel=object(); ctx.diag['plume_top']=sentinel
    plume.step(ctx,36.,3)
    assert ctx.diag['plume_top'] is sentinel


def test_a_partial_fresh_strip_needs_plumes_before_the_regular_cadence():
    written = np.array([[5, 5, 0], [5, 5, 0]], np.int32)
    assert plume.rebuilt_mid_run(written, ktau=200)
    assert not plume.rebuilt_mid_run(written, ktau=2)
    assert not plume.rebuilt_mid_run(np.full((2, 3), 5, np.int32), ktau=200)


def test_frp_uses_wrapper_end_step_clock(monkeypatch):
    # ktau*dt = 3600 s makes the call due on GSL's end-of-step clock while
    # the start-of-step curr_secs (3564 s) would not; with no fire inputs
    # the due call is refused by name rather than skipped.
    from gpuwm.core import chem_fire
    monkeypatch.setattr(chem_fire, 'plume_inputs', lambda ctx, ktau: None)
    ctx=context();ctx.clock.curr_secs=3564.
    with pytest.raises(ValueError,match='fire_plume_inputs is not produced'):
        plume.step(ctx,36.,100)


def test_landuse_arm_is_refused_without_its_fire_inputs():
    ctx=context();ctx.cfg.plume_fire_properties='landuse';ctx.cfg.stepfirepl=1
    with pytest.raises(ValueError,match='mean_fct, firesize'):
        plume.step(ctx,36.,1)


def test_mechanism_refusal():
    ctx=context();ctx.cfg.scale_fire_emiss=True
    with pytest.raises(ValueError,match='MOZART mechanism'): plume.init(ctx)


def test_parameter_refusals():
    with pytest.raises(ValueError,match='divide by zero'): PlumeParameters(alpha=0.)
    with pytest.raises(ValueError,match='suppress real fires'): PlumeParameters(frp_min=float('nan'))
