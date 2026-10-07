"""AQ transport must consume the current IEVA split and reject unowned ledgers."""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.devices import DevicesRefused, validate_ranked_physics


def test_ranked_chemistry_refuses_before_any_device_allocation():
    with pytest.raises(DevicesRefused, match="shared chemistry ledger owner"):
        validate_ranked_physics(SimpleNamespace(grid_id=3, chem_sets="tracer_test"))
    validate_ranked_physics(SimpleNamespace(grid_id=3, chem_sets=""))


@pytest.mark.parametrize("advection,final", [(1, False), (1, True), (2, True)])
@pytest.mark.parametrize("vorder", [3, 5])
def test_chemistry_consumes_explicit_and_implicit_fluxes_before_update(
        monkeypatch, advection, final, vorder):
    from gpuwm.core import chem_advect_mono, chem_transport as transport

    shape = (4, 7, 8)
    surface = shape[1:]
    row = SimpleNamespace(state_attr="chem_test", time_attr="chem0_test")
    state = SimpleNamespace(
        p=np.zeros(shape, np.float32), mub2d=np.ones(surface, np.float32),
        mup0=np.zeros(surface, np.float32), mup=np.zeros(surface, np.float32),
        msft=np.ones(surface, np.float32), c1h=np.ones(shape[0], np.float32),
        c2h=np.zeros(shape[0], np.float32), has_msf=False,
        chem_test=np.ones(shape, np.float32), chem0_test=np.ones(shape, np.float32),
        chem=SimpleNamespace(transported=(row,)),
        scratch=lambda dims, slot: np.zeros(dims, np.float32))
    cfg = SimpleNamespace(specified=False, nested=False, open_x=False,
                          open_y=False, chem_adv_opt=advection, dx=1.0, dy=1.0,
                          v_sca_adv_order=vorder)
    explicit, implicit = object(), object()
    split = (explicit, implicit)
    events = []

    def flux(*args, **kwargs):
        assert args[4 if final else 3] is explicit
        if advection == 2:
            assert kwargs["rw_implicit"] is implicit
        assert kwargs["v_order" if advection == 2 else "vorder"] == vorder
        events.append("explicit")
        return None

    def solve(_state, _tend, _old, supplied, *_rest):
        assert supplied is split
        events.append("implicit")

    monkeypatch.setattr(transport, "launch_flux_div_scalar", flux)
    monkeypatch.setattr(transport, "launch_pd_fluxes", flux)
    monkeypatch.setattr(chem_advect_mono, "launch_mono_fluxes", flux)
    monkeypatch.setattr(transport, "launch_pd_renorm_apply", lambda *a, **k: None)
    monkeypatch.setattr(chem_advect_mono, "launch_mono_renorm_apply", lambda *a, **k: None)
    monkeypatch.setattr(transport, "_pd_fold_sources", lambda *a, **k: a[3])
    monkeypatch.setattr(transport, "_ieva_scalar", solve)
    monkeypatch.setattr(transport, "_update_scalar_in_place",
                        lambda *a, **k: events.append("update"))
    transport.advance_chem_stage(state, cfg, None, None, object(), 0.5,
                                 final=final, implicit=split)
    assert events == ["explicit", "implicit", "update"]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("variant", ["wrf_471", "wrf_legacy"])
@pytest.mark.parametrize("vorder", [3, 5])
def test_chem_ieva_matches_existing_qv_transport_on_card(variant, vorder):
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    from test_chem_transport import _bubble_case

    cfg, state = _bubble_case(chem_sets="tracer_test", zadvect_implicit=1,
                              zadvect_implicit_variant=variant,
                              v_sca_adv_order=vorder)
    state.chem_passive_1[...] = state.qv
    run_steps(state, cfg, 8)
    assert float(cp.abs(state.w).max()) > 0.5
    assert cp.asnumpy(state.chem_passive_1).tobytes() == cp.asnumpy(state.qv).tobytes()
