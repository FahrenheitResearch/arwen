"""Word identity against the original dycore ElementwiseKernel factories."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]
MEMBERS = (1, 4, 10, 20, 40)


def _same_words(cp, array, expected):
    assert cp.asnumpy(array).view(np.uint32).tobytes() == np.asarray(
        expected).view(np.uint32).tobytes()


def _omega_column_reference(ru, rv, dnw, c1h, rdx, rdy, j, i, msft=None,
                            prescribed_dmdt=None):
    """Explicit scalar WRF recurrence, with each operator rounded to FP32."""
    nz = ru.shape[0]
    divv = np.empty(nz, np.float32)
    dmdt = np.float32(0)
    for k in range(nz):
        du = np.float32(ru[k, j, i + 1] - ru[k, j, i])
        dv = np.float32(rv[k, j + 1, i] - rv[k, j, i])
        bracket = np.float32(np.float32(rdx * du) + np.float32(rdy * dv))
        weight = dnw[k] if msft is None else np.float32(msft[j, i] * dnw[k])
        divv[k] = np.float32(weight * bracket)
        dmdt = np.float32(dmdt + divv[k])
    if prescribed_dmdt is not None:
        dmdt = np.float32(prescribed_dmdt)
    ww = np.zeros(nz + 1, np.float32)
    current = np.float32(0)
    for k in range(1, nz):
        level_mass = np.float32(dnw[k - 1] * c1h[k - 1])
        current = np.float32(current - np.float32(level_mass * dmdt))
        current = np.float32(current - divv[k - 1])
        ww[k] = current
    return ww, divv, dmdt


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("has_msf", (False, True))
@pytest.mark.parametrize("core", ((1, 1, 3), (5, 3, 137), (7, 5, 11)))
def test_omega_words_equal_original_factory(members, has_msf, core):
    import cupy as cp
    from gpuwm.core.dycore import _omega_column_kernel
    from gpuwm.ensemble.batch_fluxes import prepare_omega_columns

    nz, ny, nx = core
    rng = np.random.default_rng(9143)
    ru_h = rng.uniform(-40000.0, 60000.0, (members, nz, ny, nx + 1)).astype(np.float32)
    rv_h = rng.uniform(-50000.0, 70000.0, (members, nz, ny + 1, nx)).astype(np.float32)
    ru_h += np.arange(members, dtype=np.float32)[:, None, None, None] * np.float32(71.0)
    rv_h += np.arange(members, dtype=np.float32)[:, None, None, None] * np.float32(53.0)
    dnw_h = np.linspace(-0.25, -0.03125, nz, dtype=np.float32)
    c1h_h = np.linspace(1.0, 0.125, nz, dtype=np.float32)
    msft_h = rng.uniform(0.75, 1.5, (ny, nx)).astype(np.float32)
    ru, rv = cp.asarray(ru_h), cp.asarray(rv_h)
    ref_ru, ref_rv = cp.asarray(ru_h), cp.asarray(rv_h)
    dnw, c1h, msft = cp.asarray(dnw_h), cp.asarray(c1h_h), cp.asarray(msft_h)
    ref_dnw, ref_c1h, ref_msft = cp.asarray(dnw_h), cp.asarray(c1h_h), cp.asarray(msft_h)
    ww = cp.full((members, nz + 1, ny, nx), 879.25, cp.float32)
    reference = cp.full_like(ww, -927.75)
    dx, dy = 950.0, 1350.0
    batch = prepare_omega_columns(ru, rv, ww, dnw, c1h,
                                  dx=dx, dy=dy, has_msf=has_msf, msft=msft)
    original = _omega_column_kernel(has_msf)
    rdx, rdy = np.float32(1.0) / np.float32(dx), np.float32(1.0) / np.float32(dy)
    for _ in range(2):
        batch()
        for member in range(members):
            args = [ref_ru[member].reshape(-1), ref_rv[member].reshape(-1), ref_dnw, ref_c1h]
            if has_msf:
                args.append(ref_msft.reshape(-1))
            original(*args, rdx, rdy, np.int32(nz), np.int32(ny), np.int32(nx),
                     reference[member].reshape(-1), size=ny * nx)
    cp.cuda.get_current_stream().synchronize()
    _same_words(cp, ww, cp.asnumpy(reference))
    actual = cp.asnumpy(ww)
    assert not actual[:, 0].view(np.uint32).any()
    assert not actual[:, -1].view(np.uint32).any()
    # Independent scalar recurrence also checks the prescribed level order.
    for member in range(members):
        expected, _, _ = _omega_column_reference(
            ru_h[member], rv_h[member], dnw_h, c1h_h, rdx, rdy, 0, 0,
            msft_h if has_msf else None)
        assert actual[member, :, 0, 0].view(np.uint32).tobytes() == expected.view(np.uint32).tobytes()
    for array, expected in ((ru, ru_h), (rv, rv_h), (dnw, dnw_h), (c1h, c1h_h), (msft, msft_h)):
        _same_words(cp, array, expected)


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("has_msf", (False, True))
@pytest.mark.parametrize("reciprocal", (False, True))
@pytest.mark.parametrize("stagger", ("x", "y"))
@pytest.mark.parametrize("core", ((5, 3, 137), (7, 5, 11)))
def test_momentum_words_equal_original_factory(members, has_msf, reciprocal, stagger, core):
    import cupy as cp
    from gpuwm.core.dycore import _couple_momentum_kernel
    from gpuwm.ensemble.batch_fluxes import prepare_couple_momentum

    nz, ny, nx = core
    ny += int(stagger == "y")
    nx += int(stagger == "x")
    rng = np.random.default_rng(4471)
    wind_h = rng.uniform(-15.0, 27.0, (members, nz, ny, nx)).astype(np.float32)
    mass_h = rng.uniform(65000.0, 120000.0, (members, ny, nx)).astype(np.float32)
    wind_h += np.arange(members, dtype=np.float32)[:, None, None, None] * np.float32(0.125)
    mass_h += np.arange(members, dtype=np.float32)[:, None, None] * np.float32(101.0)
    c1h_h = np.linspace(1.0, 0.125, nz, dtype=np.float32)
    c2h_h = np.linspace(1500.0, 38000.0, nz, dtype=np.float32)
    msf_h = rng.uniform(0.7, 1.6, (ny, nx)).astype(np.float32)
    wind, mass = cp.asarray(wind_h), cp.asarray(mass_h)
    ref_wind, ref_mass = cp.asarray(wind_h), cp.asarray(mass_h)
    c1h, c2h, msf = cp.asarray(c1h_h), cp.asarray(c2h_h), cp.asarray(msf_h)
    ref_c1h, ref_c2h, ref_msf = cp.asarray(c1h_h), cp.asarray(c2h_h), cp.asarray(msf_h)
    flux = cp.full_like(wind, 712.25)
    reference = cp.full_like(wind, -19.75)
    batch = prepare_couple_momentum(wind, mass, flux, c1h, c2h,
                                    has_msf=has_msf, msf=msf, reciprocal=reciprocal)
    original = _couple_momentum_kernel(has_msf, reciprocal)
    for _ in range(2):
        batch()
        for member in range(members):
            args = [ref_wind[member], ref_c1h, ref_c2h, ref_mass[member].reshape(-1)]
            if has_msf:
                args.append(ref_msf.reshape(-1))
            original(*args, np.int32(ny * nx), reference[member])
    cp.cuda.get_current_stream().synchronize()
    _same_words(cp, flux, cp.asnumpy(reference))
    for array, expected in ((wind, wind_h), (mass, mass_h), (c1h, c1h_h),
                            (c2h, c2h_h), (msf, msf_h)):
        _same_words(cp, array, expected)


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("has_msf", (False, True))
def test_omega_cancellation_fixture_exposes_sequential_level_fold(members, has_msf):
    import cupy as cp
    from gpuwm.core.dycore import _omega_column_kernel
    from gpuwm.ensemble.batch_fluxes import prepare_omega_columns

    nz, ny, nx = 5, 1, 2
    desired = np.array([1.0e20, 1.0, -1.0e20, 1.0, 1.0], np.float32)
    scale = np.exp2(np.arange(members, dtype=np.float32))
    ru_h = np.zeros((members, nz, ny, nx + 1), np.float32)
    ru_h[:, :, 0, 1] = np.float32(-5.0) * desired[None] * scale[:, None]
    rv_h = np.zeros((members, nz, ny + 1, nx), np.float32)
    dnw_h, c1h_h = np.full(nz, -0.2, np.float32), np.ones(nz, np.float32)
    msft_h = np.array([[1.0, 1.25]], np.float32)
    ru, rv, dnw, c1h, msft = (cp.asarray(value)
                              for value in (ru_h, rv_h, dnw_h, c1h_h, msft_h))
    ww = cp.empty((members, nz + 1, ny, nx), cp.float32)
    reference = cp.empty_like(ww)
    prepare_omega_columns(ru, rv, ww, dnw, c1h,
                          dx=1.0, dy=1.0, has_msf=has_msf, msft=msft)()
    original = _omega_column_kernel(has_msf)
    for member in range(members):
        args = [ru[member].reshape(-1), rv[member].reshape(-1), dnw, c1h]
        if has_msf:
            args.append(msft.reshape(-1))
        original(*args, np.float32(1.0), np.float32(1.0),
                 np.int32(nz), np.int32(ny), np.int32(nx),
                 reference[member].reshape(-1), size=ny * nx)
    cp.cuda.get_current_stream().synchronize()
    _same_words(cp, ww, cp.asnumpy(reference))
    actual = cp.asnumpy(ww)
    for member in range(members):
        expected, divv, sequential = _omega_column_reference(
            ru_h[member], rv_h[member], dnw_h, c1h_h,
            np.float32(1.0), np.float32(1.0), 0, 0, msft_h if has_msf else None)
        pair_a = np.float32(divv[0] + divv[1])
        pair_b = np.float32(divv[2] + divv[3])
        tree = np.float32(np.float32(pair_a + pair_b) + divv[4])
        tree_output, _, _ = _omega_column_reference(
            ru_h[member], rv_h[member], dnw_h, c1h_h,
            np.float32(1.0), np.float32(1.0), 0, 0,
            msft_h if has_msf else None, prescribed_dmdt=tree)
        assert sequential != tree
        assert expected[4].view(np.uint32) != tree_output[4].view(np.uint32)
        assert actual[member, :, 0, 0].view(np.uint32).tobytes() == expected.view(np.uint32).tobytes()


@pytest.mark.parametrize("members", MEMBERS)
@pytest.mark.parametrize("has_msf", (False, True))
def test_flux_components_each_use_one_launch_and_n1_original_argument_contract(
        monkeypatch, members, has_msf):
    import cupy as cp
    from gpuwm.ensemble import batch_fluxes as batch

    scalar_calls, raw_calls = [], []
    original_factory, raw_factory = batch._original, batch._compiled

    class ObservedElementwise:
        def __init__(self, kind, kernel):
            self.kind, self.kernel = kind, kernel

        def __getattr__(self, name):
            return getattr(self.kernel, name)

        def __call__(self, *args, **kwargs):
            scalar_calls.append((self.kind, args, kwargs))
            return self.kernel(*args, **kwargs)

    def observed_original(kind, *options):
        return ObservedElementwise(kind, original_factory(kind, *options))

    def observed_raw(kind, *options):
        kernel = raw_factory(kind, *options)

        def launch(*args, **kwargs):
            raw_calls.append((kind, args, kwargs))
            return kernel(*args, **kwargs)

        return launch

    monkeypatch.setattr(batch, "_original", observed_original)
    monkeypatch.setattr(batch, "_compiled", observed_raw)
    nz, ny, nx = 5, 3, 137
    ru = cp.ones((members, nz, ny, nx + 1), cp.float32)
    rv = cp.ones((members, nz, ny + 1, nx), cp.float32)
    ww = cp.empty((members, nz + 1, ny, nx), cp.float32)
    dnw, c1h, c2h = cp.full(nz, -0.2, cp.float32), cp.ones(nz, cp.float32), cp.zeros(nz, cp.float32)
    msft = cp.ones((ny, nx), cp.float32)
    mass = cp.full((members, ny, nx + 1), 90000.0, cp.float32)
    msfu = cp.ones((ny, nx + 1), cp.float32)
    flux = cp.empty_like(ru)
    batch.prepare_omega_columns(ru, rv, ww, dnw, c1h,
                                dx=1000.0, dy=1250.0, has_msf=has_msf, msft=msft)()
    batch.prepare_couple_momentum(ru, mass, flux, c1h, c2h,
                                  has_msf=has_msf, msf=msfu, reciprocal=True)()
    cp.cuda.get_current_stream().synchronize()
    if members == 1:
        assert len(scalar_calls) == 2 and not raw_calls
        assert scalar_calls[0][0] == "omega" and scalar_calls[0][2] == {"size": ny * nx}
        assert scalar_calls[0][1][0].shape == (nz * ny * (nx + 1),)
        assert scalar_calls[0][1][-1].shape == ((nz + 1) * ny * nx,)
        assert scalar_calls[1][0] == "momentum" and scalar_calls[1][2] == {}
        assert scalar_calls[1][1][0].shape == (nz, ny, nx + 1)
        assert scalar_calls[1][1][-2] == ny * (nx + 1)
        assert scalar_calls[1][1][-1].shape == (nz, ny, nx + 1)
    else:
        assert not scalar_calls and len(raw_calls) == 2
        assert raw_calls[0][0] == "omega" and raw_calls[1][0] == "momentum"
        omega_args, momentum_args = raw_calls[0][1][2], raw_calls[1][1][2]
        assert omega_args[0] is ru and omega_args[1] is rv and omega_args[-3] is ww
        assert momentum_args[0] is ru and momentum_args[-3] is flux
        assert all(value.ndim == 4 for value in (omega_args[0], omega_args[1], omega_args[-3], momentum_args[0], momentum_args[-3]))
        _, omega_spec, _ = batch._raw_source("omega", has_msf)
        _, momentum_spec, _ = batch._raw_source("momentum", has_msf, True)
        omega_strides = {p.name: int(value) for p, value in zip(
            omega_spec.pointers, np.frombuffer(omega_args[-1].tobytes(), np.uint64))}
        momentum_strides = {p.name: int(value) for p, value in zip(
            momentum_spec.pointers, np.frombuffer(momentum_args[-1].tobytes(), np.uint64))}
        assert omega_strides["ru"] == ru.strides[0]
        assert omega_strides["rv"] == rv.strides[0]
        assert omega_strides["ww"] == ww.strides[0]
        assert momentum_strides["wind_values"] == ru.strides[0]
        assert momentum_strides["flux_values"] == flux.strides[0]
        assert momentum_strides["muface"] == mass.strides[0]
        assert all(omega_strides[p.name] == 0 for p in omega_spec.pointers if p.role == "shared")
        assert all(momentum_strides[p.name] == 0 for p in momentum_spec.pointers if p.role == "shared")


def test_member_guards_and_binding_refusals():
    import cupy as cp
    from gpuwm.ensemble.batch_fluxes import prepare_couple_momentum, prepare_omega_columns

    members, nz, ny, nx = 4, 3, 5, 3
    sentinel = np.float32(-912.5)
    guarded_ww = cp.full((members + 2, nz + 1, ny, nx), sentinel, cp.float32)
    ww = guarded_ww[1:-1]
    ru = cp.ones((members, nz, ny, nx + 1), cp.float32)
    rv = cp.ones((members, nz, ny + 1, nx), cp.float32)
    dnw, c1h, c2h = cp.full(nz, -0.25, cp.float32), cp.ones(nz, cp.float32), cp.zeros(nz, cp.float32)
    prepare_omega_columns(ru, rv, ww, dnw, c1h, dx=1000.0, dy=1000.0)()
    guarded_flux = cp.full((members + 2, nz, ny, nx + 1), sentinel, cp.float32)
    flux = guarded_flux[1:-1]
    mass = cp.full((members, ny, nx + 1), 80000.0, cp.float32)
    prepare_couple_momentum(ru, mass, flux, c1h, c2h)()
    cp.cuda.get_current_stream().synchronize()
    for backing in (guarded_ww, guarded_flux):
        expected = np.full((1,) + backing.shape[1:], sentinel, np.float32)
        _same_words(cp, backing[:1], expected)
        _same_words(cp, backing[-1:], expected)
    with pytest.raises(ValueError, match="4-D"):
        prepare_omega_columns(ru[0], rv, ww, dnw, c1h, dx=1000.0, dy=1000.0)
    with pytest.raises(ValueError, match="rv must have member/C-grid shape"):
        prepare_omega_columns(ru, ru, ww, dnw, c1h, dx=1000.0, dy=1000.0)
    with pytest.raises(ValueError, match="overlaps ru"):
        prepare_omega_columns(ru, rv, ru.reshape(ww.shape), dnw, c1h, dx=1000.0, dy=1000.0)
    with pytest.raises(ValueError, match="overlaps wind"):
        prepare_couple_momentum(ru, mass, ru, c1h, c2h)
    with pytest.raises(ValueError, match="3-D"):
        prepare_couple_momentum(ru, mass[0], flux, c1h, c2h)
    with pytest.raises(TypeError, match="msf needs a CUDA array"):
        prepare_couple_momentum(ru, mass, flux, c1h, c2h, has_msf=True)
    with pytest.raises(ValueError, match="shared shape"):
        prepare_couple_momentum(ru, mass, flux, c1h[:2], c2h)


@pytest.mark.parametrize("kind,has_msf,reciprocal", [
    ("omega", False, False), ("omega", True, False),
    ("momentum", False, False), ("momentum", True, False),
    ("momentum", True, True),
])
def test_raw_component_source_contains_original_operation_verbatim(kind, has_msf, reciprocal):
    from gpuwm.ensemble.batch_fluxes import _OPTIONS, _original, _raw_source
    from gpuwm.ensemble.batch_kernel import generate_batch_source

    source, spec, _ = _raw_source(kind, has_msf, reciprocal)
    operation = _original(kind, has_msf, reciprocal).operation
    assert source.count(operation) == 1
    assert operation in generate_batch_source(source, spec, 40)
    assert spec.options == _OPTIONS == ("-std=c++17", "-fmad=false")
    assert generate_batch_source(source, spec, 1) == source
