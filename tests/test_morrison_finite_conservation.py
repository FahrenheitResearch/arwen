"""Small device contracts for finite transfers and final vapor ownership."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core import constants as c


MASS = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBER = ("nc", "nr", "ni", "ns", "ng")
SURFACE = ("rainnc", "rainncv", "snownc", "snowncv",
           "graupelnc", "graupelncv", "sr")
EFFECTIVE = ("effc", "effr", "effi", "effs")


def _column(temp=156.0, rain=1.2e-13):
    shape = (2, 1, 1)
    host = {name: np.zeros(shape, np.float32)
            for name in MASS + NUMBER}
    host["pressure"] = np.array([300.0, 250.0], np.float32).reshape(shape)
    host["pii"] = ((host["pressure"] / np.float32(c.P0))
                   ** np.float32(c.RCP))
    host["theta"] = np.float32(temp) / host["pii"]
    host["rho"] = host["pressure"] / (np.float32(c.RD)
                                         * host["theta"] * host["pii"])
    host["dz"] = np.full(shape, 3000.0, np.float32)
    host["qv"][:, 0, 0] = (4.4e-8, 5.28e-8)
    host["qr"].fill(rain)
    host["nr"].fill(0.00475)
    return host


def _launch(host, dt, mode, *, adapter=False, monkeypatch=None):
    """Run the real launcher, optionally through the actual state adapter."""
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core import morrison
    from gpuwm.core.state import DomainState

    shape = host["theta"].shape
    stages = []
    if adapter:
        cfg = RunConfig(nx=shape[2], ny=shape[1], nz=shape[0],
                        dx=1000.0, dy=1000.0, ztop=6000.0, dt=dt,
                        run_seconds=0.0,
                        moist=True, mp_physics=10, morr_rimed_ice=mode)
        state = DomainState(cfg)
        state.p[...] = cp.asarray(host["pressure"])
        state.thb[...] = cp.asarray(host["theta"][:, 0, 0])
        z = np.concatenate(([0.0], np.cumsum(host["dz"][:, 0, 0])))
        state.phb[...] = cp.asarray(z * c.G, dtype=cp.float32)
        for name in MASS + NUMBER:
            getattr(state, name)[...] = cp.asarray(host[name])
        morrison.apply(state, cfg, dt)
        fields = {name: getattr(state, name)
                  for name in MASS + NUMBER + EFFECTIVE}
        fields.update({name: state.scratch(shape[1:], "mp_" + name)
                       for name in SURFACE})
        fields["theta"] = state.thb[:, None, None] + state.thp
        fields["h_diabatic"] = state.h_diabatic
        density = state.scratch(shape, "morr_rho")
    else:
        fields = {name: cp.asarray(value) for name, value in host.items()}
        fields.update({name: cp.zeros(shape[1:], cp.float32)
                       for name in SURFACE})
        fields.update({name: cp.zeros(shape, cp.float32)
                       for name in EFFECTIVE})
        density = cp.zeros(shape, cp.float32)
        if monkeypatch is not None:
            original = morrison.get_kernel

            def observed(module, name):
                actual = original(module, name)

                def call(*args, **kwargs):
                    actual(*args, **kwargs)
                    stages.append((name, {key: cp.asnumpy(value)
                                          for key, value in fields.items()}))

                return call

            monkeypatch.setattr(morrison, "get_kernel", observed)
        morrison.launch_morrison(**fields, dt=dt, morr_rimed_ice=mode,
                                _rhoa_scratch=density)
    got = {name: cp.asnumpy(value) for name, value in fields.items()}
    return got, cp.asnumpy(density).astype(np.float64), stages


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", (0, 1))
@pytest.mark.parametrize("rain", (1.2e-13, 5.0e-4))
@pytest.mark.parametrize("adapter", (False, True))
def test_cold_rain_keeps_finite_state_and_condensed_water(mode, rain, adapter):
    host = _column(rain=rain)
    got, density, _ = _launch(host, 50.0, mode, adapter=adapter)
    for name, values in got.items():
        assert np.isfinite(values).all(), name
    for name in MASS + NUMBER + SURFACE:
        assert (got[name] >= 0.0).all(), name
    np.testing.assert_array_equal(got["qv"], host["qv"])
    if not adapter:
        np.testing.assert_array_equal(got["rho"], host["rho"])
    # Vapor is much larger than the thin-rain control and must not hide its
    # loss. Use entry dry density for both inventories; rainncv is total
    # surface water, including frozen precipitation.
    weight = density * host["dz"].astype(np.float64)
    initial = float(np.sum(host["qr"].astype(np.float64) * weight))
    condensed = sum(got[name].astype(np.float64) for name in MASS[1:])
    final = float(np.sum(condensed * weight) + np.sum(got["rainncv"]))
    # These two levels need at most eight substeps at the scheme's speed
    # caps. Allow eight rounded operations per level/substep and 32 for
    # transfer, density conversion and surface summation.
    unit_roundoff = np.finfo(np.float32).eps / 2.0
    nops = 8 * 2 * 8 + 32
    gamma = nops * unit_roundoff / (1.0 - nops * unit_roundoff)
    assert abs(final - initial) <= gamma * initial


@pytest.mark.gpu
@requires_gpu
def test_vapor_only_cold_control_has_no_spurious_water_or_heating():
    host = _column(rain=0.0)
    got, _, _ = _launch(host, 50.0, 1, adapter=True)
    for name in MASS:
        np.testing.assert_array_equal(got[name], host[name])
    np.testing.assert_array_equal(got["h_diabatic"], 0.0)
    np.testing.assert_array_equal(got["rainncv"], 0.0)


@pytest.mark.gpu
@requires_gpu
def test_exceptional_rain_shares_its_donor_with_every_competing_sink():
    import cupy as cp

    from gpuwm.core.kernels import module_source

    source = module_source("morrison") + r'''
extern "C" __global__ void inspect_rain_budget(real* out) {
    MorrRates r = {};
    r.pre = -1.e9f; r.pracs = 2.e9f; r.qmultr = 3.e9f;
    r.qmultrg = 4.e9f; r.piacr = 5.e9f; r.piacrs = 6.e9f;
    r.pgracs = 7.e9f; r.pracg = 8.e9f;
    r.prc = .001f; r.pra = .002f; r.nnuccr = .00475f / 2.f;
    real lambda = 49863.8515625f;
    real bigg = expf(.66f * (273.15f - 156.f)) - 1.f;
    r.mnuccr = 20.f * MPI * MPI * MRHOW * 100.f * .00475f * bigg
               / powf(lambda, 6.f);
    out[0] = r.mnuccr;
    morr_limit_cold_rain(r, .002f, .00475f, lambda, 156.f, 2.f);
    out[1] = -r.pre; out[2] = r.pracs; out[3] = r.qmultr;
    out[4] = r.qmultrg; out[5] = r.piacr; out[6] = r.piacrs;
    out[7] = r.pgracs; out[8] = r.pracg; out[9] = r.mnuccr;
    out[10] = r.nnuccr;
    out[11] = 20.f * MPI * MPI * MRHOW * 100.f;
    out[12] = .66f * (273.15f - 156.f);
}
'''
    module = cp.RawModule(code=source, options=("-std=c++17",))
    out = cp.zeros(13, cp.float32)
    module.get_function("inspect_rain_budget")((1,), (1,), (out,))
    got = cp.asnumpy(out).astype(np.float64)
    assert np.isinf(got[0])
    freezing = (got[11] * float(np.float32(.00475))
                * np.expm1(got[12]) / 49863.8515625 ** 6)
    sinks = np.r_[np.arange(1, 9, dtype=np.float64) * 1.e9, freezing]
    supply = (float(np.float32(.002)) / 2.0
              + float(np.float32(.001)) + float(np.float32(.002)))
    expected = sinks * (supply / sinks.sum())
    np.testing.assert_allclose(got[1:10], expected, rtol=4.e-7, atol=0.0)
    assert abs(got[1:10].sum() - supply) <= np.spacing(np.float32(supply))
    assert got[10] == np.float32(.00475) / np.float32(2.0)


def _settling_column():
    host = _column(temp=280.0, rain=0.0)
    host["pressure"][:, 0, 0] = (90000.0, 85000.0)
    host["pii"][:] = ((host["pressure"] / np.float32(c.P0))
                       ** np.float32(c.RCP))
    host["theta"][:] = np.float32(280.0) / host["pii"]
    host["rho"][:] = host["pressure"] / (np.float32(c.RD)
                                            * host["theta"] * host["pii"])
    host["qv"][:, 0, 0] = (1.0e-5, 0.007339452393352985)
    host["qr"][1, 0, 0] = 1.0e-7
    return host


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", (0, 1))
def test_settled_trace_rain_returns_to_vapor_and_closes_column(mode, monkeypatch):
    host = _settling_column()
    got, density, stages = _launch(host, 1.0, mode, monkeypatch=monkeypatch)
    assert [name for name, _ in stages] == [
        "morrison_process_levels", "morrison_sediment_64",
        "morrison_finalize_levels"]
    before = stages[-2][1]
    removed = sum(before[name].astype(np.float64) - got[name]
                  for name in MASS[1:])
    vapor_gain = got["qv"].astype(np.float64) - before["qv"]
    # Rain is absent at entry in the dry lower cell, arrives in the actual
    # sedimentation stage, then evaporates during finalization.
    assert before["qr"][0, 0, 0] > 0.0
    assert removed[0, 0, 0] > 1.e-10
    assert got["qr"][0, 0, 0] == 0.0
    assert vapor_gain[0, 0, 0] > 0.0
    np.testing.assert_allclose(vapor_gain, removed, rtol=0.0,
                               atol=float(np.spacing(before["qv"][0, 0, 0])))
    np.testing.assert_array_equal(got["qv"][1], host["qv"][1])
    weight = density * host["dz"].astype(np.float64)
    delta = sum(got[name].astype(np.float64) - host[name] for name in MASS)
    residual = float(np.sum(delta * weight) + np.sum(got["rainncv"]))
    # The saturated upper vapor is unchanged. Charge one stored-vapor ULP
    # in the lower cell and 64 rounded operations on transported condensate;
    # its much larger stationary vapor inventory cannot conceal the loss.
    floor = float(np.spacing(host["qv"][0, 0, 0]) * weight[0, 0, 0])
    condensate = float(np.sum(host["qr"].astype(np.float64) * weight))
    assert abs(residual) <= floor + 64 * np.finfo(np.float32).eps * condensate
    for name in MASS + NUMBER + EFFECTIVE + ("theta",):
        assert np.isfinite(got[name]).all(), name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("temperature", (240.0, 280.0))
def test_final_cleanup_stores_each_evaporated_or_sublimated_species(temperature):
    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    shape = (2, 1, 5)
    fields = {name: cp.zeros(shape, cp.float32) for name in MASS + NUMBER}
    fields["qv"].fill(1.e-6)
    amount = np.float32(5.e-9)
    for i, name in enumerate(MASS[1:]):
        fields[name][:, 0, i] = amount
    theta = cp.full(shape, temperature, cp.float32)
    pii = cp.ones(shape, cp.float32)
    rho = cp.ones(shape, cp.float32)
    pressure = cp.full(shape, 80000.0, cp.float32)
    mask = cp.zeros(shape, cp.float32)
    xlv = np.float32(3.1484e6) - np.float32(2370.0) * np.float32(temperature)
    cpm = np.float32(c.CP) * (np.float32(1.0) + np.float32(.887e-6))
    effc = cp.full(shape, xlv, cp.float32)
    effi = cp.full(shape, cpm, cp.float32)
    effs, effr = cp.zeros_like(theta), cp.zeros_like(theta)
    kernel = get_kernel("morrison", "morrison_finalize_levels")
    kernel((1,), (32,), (theta, *(fields[n] for n in MASS + NUMBER),
                         rho, pii, pressure, mask, effc, effi, effs, effr,
                         np.float32(900.0), np.int32(theta.size)))
    np.testing.assert_array_equal(cp.asnumpy(fields["qv"]),
                                   np.float32(1.e-6) + amount)
    for name in MASS[1:]:
        np.testing.assert_array_equal(cp.asnumpy(fields[name]), 0.0)
    latent = np.array([xlv, xlv, xlv + np.float32(335300.0),
                       xlv + np.float32(335300.0),
                       xlv + np.float32(335300.0)], np.float32)
    expected = np.float32(temperature) - amount * latent / cpm
    np.testing.assert_allclose(cp.asnumpy(theta)[:, 0, :],
                               np.broadcast_to(expected, (2, 5)), rtol=0.0,
                               atol=float(np.spacing(np.float32(temperature))))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("temperature", (240.0, 250.0))
def test_cloud_rates_retain_the_log_space_immersion_term(temperature):
    import cupy as cp

    from gpuwm.core.kernels import module_source

    # Observe rates inside the full production level function before donor
    # limiting. This only adds stores, and leaves every rate expression and
    # the ordinary preamble intact. The comparison uses independent double
    # arithmetic with the same CUDA gamma values, isolating log-space loss
    # from the separate WRF GAMMA implementation question.
    source = module_source("morrison")
    anchor = "    if (*qs >= 1.0e-8f) {\n        real cons15"
    assert source.count(anchor) == 1
    capture = r'''
    cloud_observation[0] = r.mnuccc; cloud_observation[1] = r.nnuccc;
    cloud_observation[2] = m.lc; cloud_observation[3] = *nc;
    cloud_observation[4] = mu; cloud_observation[5] = tgammaf(m.pg + 1.f);
    cloud_observation[6] = tgammaf(m.pg + 2.f);
    cloud_observation[7] = tgammaf(m.pg + 4.f);
    cloud_observation[8] = tgammaf(m.pg + 5.f);
    cloud_observation[9] = tgammaf(m.pg + 7.f);
'''
    source = ('extern "C" __device__ float cloud_observation[10];\n'
              + source.replace(anchor, capture + anchor))
    source += r'''
extern "C" __global__ void inspect_cloud_rates(real temp, real* out) {
    real qv=.001f, qc=1.2e-13f, qr=0.f, qi=0.f, qs=0.f, qg=0.f;
    real nc=0.f, nr=0.f, ni=0.f, ns=0.f, ng=0.f, stale=0.f, cloud_nc=0.f;
    real pressure=300.f, rhoa=pressure/(RD*temp);
    real xlv=3.1484e6f-2370.f*temp, xls=xlv+335300.f;
    morr_process_level(&qv,&qc,&qr,&qi,&qs,&qg,&nc,&nr,&ni,&ns,&ng,
                       &temp,pressure,rhoa,50.f,.001f,.001f,xlv,xls,CP,
                       false,&stale,&cloud_nc,114.5f,.5f,900.f);
    for (int i=0;i<10;++i) out[i]=cloud_observation[i];
}
'''
    module = cp.RawModule(code=source, options=("-std=c++17",))
    out = cp.zeros(10, cp.float32)
    module.get_function("inspect_cloud_rates")(
        (1,), (1,), (np.float32(temperature), out))
    mass, number, lam, nc, mu, g1, g2, g4, g5, g7 = cp.asnumpy(out).astype(float)
    pi = float(np.float32(np.pi))
    nuclei = np.exp(-2.8 + .262 * (273.15 - temperature)) * 1000.0
    slip = 7.37 * temperature / (2880.0 * 300.0) / 100.0
    dap = (4 * pi * 1.38e-23 / (6 * pi * 1.e-7)
           * temperature * (1 + slip / 1.e-7) / mu)
    distribution = nc / g1
    bigg = np.expm1(.66 * (273.15 - temperature))
    contact = pi ** 2 / 3 * 997 * dap * nuclei * distribution * g5 / lam ** 4
    immersion = pi ** 2 / 36 * 997 * 100 * distribution * g7 / lam ** 6 * bigg
    expected_number = min(2 * pi * dap * nuclei * distribution * g2 / lam
                          + pi / 6 * 100 * distribution * g4 / lam ** 3 * bigg,
                          nc / 50.0)
    assert immersion > 0.0
    # The argument sums of the three FP32 log operations have magnitude
    # below 120 here; 4e-5 covers their rounding and the exponential's error.
    assert abs(mass - (contact + immersion)) <= 4.e-5 * (contact + immersion)
    assert abs(number - expected_number) <= 4.e-5 * expected_number


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", (0, 1))
@pytest.mark.parametrize("temperature", (138.0, 156.0))
def test_cold_cloud_is_finite_through_adapter_and_heating(mode, temperature):
    host = _column(temp=temperature, rain=0.0)
    host["qc"].fill(1.2e-13)
    if temperature == 138.0:
        host["qv"][:, 0, 0] = (1.12e-11, 1.344e-11)
    got, density, _ = _launch(host, 50.0, mode, adapter=True)
    for name, values in got.items():
        assert np.isfinite(values).all(), name
    for name in MASS + NUMBER:
        assert (got[name] >= 0.0).all(), name
    weight = density * host["dz"].astype(np.float64)
    initial = float(np.sum(host["qc"].astype(np.float64) * weight))
    final = float(np.sum(sum(got[name].astype(np.float64)
                             for name in MASS[1:]) * weight)
                  + np.sum(got["rainncv"]))
    np.testing.assert_array_equal(got["qv"], host["qv"])
    assert abs(final - initial) <= 128 * np.finfo(np.float32).eps * initial


@pytest.mark.gpu
@requires_gpu
def test_cloud_joint_budget_keeps_wide_rate_until_after_donor_limiting():
    import cupy as cp

    from gpuwm.core.kernels import module_source

    source = module_source("morrison") + r'''
extern "C" __global__ void inspect_cloud_budget(real* out) {
    MorrRates r = {};
    r.prc=1.e30f; r.pra=2.e30f; r.psacws=3.e30f; r.psacwi=4.e30f;
    r.qmults=5.e30f; r.qmultg=6.e30f; r.psacwg=7.e30f; r.pgsacw=8.e30f;
    double freezing=1.e40;
    r.mnuccc=(real)freezing; r.nnuccc=17.f;
    morr_limit_cold_cloud(r, .01f, 2.f, freezing);
    out[0]=r.prc; out[1]=r.pra; out[2]=r.psacws; out[3]=r.psacwi;
    out[4]=r.qmults; out[5]=r.qmultg; out[6]=r.psacwg; out[7]=r.pgsacw;
    out[8]=r.mnuccc; out[9]=r.nnuccc;
}
'''
    module = cp.RawModule(code=source, options=("-std=c++17",))
    out = cp.zeros(10, cp.float32)
    module.get_function("inspect_cloud_budget")((1,), (1,), (out,))
    got = cp.asnumpy(out).astype(np.float64)
    sinks = np.r_[(np.arange(1, 9) * 1.e30).astype(np.float32).astype(np.float64),
                  1.e40]
    supply = float(np.float32(.01)) / 2.0
    expected = sinks * (supply / sinks.sum())
    np.testing.assert_allclose(got[:9], expected, rtol=4.e-7, atol=0.0)
    assert abs(got[:9].sum() - supply) <= np.spacing(np.float32(supply))
    assert got[9] == 17.0


@pytest.mark.gpu
@requires_gpu
def test_number_is_reconstructed_only_when_the_slope_is_clipped():
    import cupy as cp

    from gpuwm.core.kernels import module_source

    source = module_source("morrison") + r'''
extern "C" __global__ void inspect_number_bounds(real* out) {
    int species=threadIdx.x;
    if (species>=5) return;
    real six_c[5]={MPI*MRHOW,6.f*MCI,6.f*MCS,MPI*400.f,MPI*900.f};
    real lo[5]={1.f/2800.e-6f,1.f/350.e-6f,1.f/2000.e-6f,
                1.f/2000.e-6f,1.f/2000.e-6f};
    real hi[5]={1.f/20.e-6f,1.f/1.e-6f,1.f/10.e-6f,
                1.f/20.e-6f,1.f/20.e-6f};
    for (int sample=0;sample<5;++sample) {
        real n=sample==0?(species==0?100.f:1.e5f):
               sample==1?0.f:sample==2?1.e30f:100.f;
        real q=sample==3?0.f:sample==4?9.e-15f:.001f;
        real lambda;
        int idx=(species*5+sample)*8;
        out[idx]=n;
        out[idx+1]=cbrtf(six_c[species]*fmaxf(n,0.f)/fmaxf(q,MQSMALL));
        morr_bound_one(q,six_c[species],lo[species],hi[species],&n,&lambda);
        out[idx+2]=n; out[idx+3]=lambda; out[idx+4]=lo[species];
        out[idx+5]=hi[species]; out[idx+6]=six_c[species]; out[idx+7]=q;
    }
}
'''
    module = cp.RawModule(code=source, options=("-std=c++17",))
    out = cp.zeros((5, 5, 8), cp.float32)
    module.get_function("inspect_number_bounds")((1,), (32,), (out,))
    got = cp.asnumpy(out)
    for species in got:
        for original, raw, number, slope, lo, hi, six_c, mass in species:
            if mass < np.float32(1.e-14):
                assert number == 0.0 and slope == 0.0
            elif lo <= raw <= hi:
                assert number.view(np.uint32) == original.view(np.uint32)
                assert slope == raw
            else:
                target = float(lo if raw < lo else hi)
                assert slope == np.float32(target)
                expected = float(mass) * target ** 3 / float(six_c)
                assert abs(float(number) - expected) <= 4 * np.spacing(number)
