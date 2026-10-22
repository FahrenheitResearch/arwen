"""CUDA interval integration: published coefficients, gates, missing history."""
import numpy as np
import pytest
from conftest import requires_gpu
pytestmark = [pytest.mark.gpu, requires_gpu]

def test_fram_accumulates_with_surface_and_type_gates_without_state_mutation():
    import cupy as cp
    from gpuwm.ensemble.ice_accretion import SavedFrameIce
    shape = (240, 300)
    def full(value):
        return cp.full(shape, value, cp.float32)
    # Saturated -2 C: psychrometric Tw=-2 C. Wind exactly 10 kt.
    t = 271.15
    es = 6.112 * np.exp(17.67 * -2 / (-2 + 243.5))
    q = .622 * es / (1000 - es)
    fields = dict(T2=full(t), Q2=full(q), PSFC=full(100000),
                  U10=full(10*1852/3600), V10=full(0),
                  QRAIN=full(.001), QSNOW=full(0))
    fields["QSNOW"][:, 0] = .02 # snow, never freezing rain
    fields["T2"][:, 1] = 274.15 # above freezing
    fields["Q2"][:, 2] = cp.nan # missing, must not be zero hazard
    original = {k: v.get().tobytes() for k,v in fields.items()}
    owner = SavedFrameIce(cp)
    key = (1,0,0,"grid")
    owner.submit(key, 0, fields, full(0))
    a = owner.submit(key, 3600000000, fields, full(2.54))
    # FRAM 0.1 in/h, Tw=-2 C, 10 kt, low-wind cold branch.
    expected_ilr = .79*(.1395*.1**(-.541)) + .20*(-.0071*(-2)**3-.1039*(-2)**2-.3904*(-2)+.5545) + .01*(.0014*100+.0027*10+.7574)
    np.testing.assert_allclose(a["ice_accretion"].get()[:,3:], 2.54*expected_ilr*.394, rtol=2e-5)
    assert cp.all(a["ice_accretion"][:,:2] == 0)
    assert cp.all(cp.isnan(a["ice_accretion"][:,2]))
    assert cp.all(a["freezing_rain_total"][:,3:] == cp.float32(2.54))
    b = owner.submit(key, 7200000000, fields, full(5.08))
    np.testing.assert_allclose(b["ice_accretion"].get()[:,3:], 2*2.54*expected_ilr*.394, rtol=2e-5)
    assert {k:v.get().tobytes() for k,v in fields.items()} == original
    with pytest.raises(ValueError, match="clock must advance"):
        owner.submit(key, 7200000000, fields, full(5.08))
    # Missing type sample invalidates subsequent accumulation.
    owner.submit(key, 10800000000, {k:v for k,v in fields.items() if k!="QSNOW"}, full(6))
    c = owner.submit(key, 14400000000, fields, full(7))
    assert cp.all(cp.isnan(c["ice_accretion"]))
    # No observed t=0 means no fabricated zero accumulation.
    late = SavedFrameIce(cp).submit(key, 3600000000, fields, full(2.54))
    assert cp.all(cp.isnan(late["ice_accretion"]))


def test_native_ice_and_cold_probabilities_use_strict_float32_boundaries():
    import cupy as cp
    from gpuwm.ensemble.batch_products import default_product_requests, prepare_product_frame
    fields = {}
    for name, threshold in (("ice_accretion", 2.54), ("temperature2_below_freezing", 273.15)):
        edge = np.float32(threshold)
        values = np.array([np.nextafter(edge, np.float32(-np.inf)), edge,
                           np.nextafter(edge, np.float32(np.inf))], np.float32)
        fields[name] = cp.broadcast_to(cp.asarray(values)[:, None, None], (3,240,300)).copy()
    requests, _ = default_product_requests(fields, postage_stamp=False)
    frame = prepare_product_frame(fields, requests, available_bytes=int(cp.cuda.runtime.memGetInfo()[0]), array_module=cp)
    output = frame()
    # Exactly one member exceeds .10 in; exactly one is below 0 C.
    for name in fields:
        np.testing.assert_array_equal(output[name+":probability"][0].get(), np.full((240,300), np.float32(1/3)))
    assert cp.all(output["ice_accretion:probability"][1:] == 0)
