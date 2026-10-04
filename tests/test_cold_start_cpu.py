"""Native closure keeps the pre-port host numbers, refusals and receipts."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

from gpuwm.core import portable_math as pm
from gpuwm.ingest import real


@pytest.fixture(autouse=True)
def native_closure():
    if not hasattr(pm._load(), "gpuwm_cold_start_numbers_f32"):
        pytest.skip("CPU bridge predates native cold-start numbers")


def make_state(scheme, shape=(3, 257, 263)):
    from gpuwm.boundary_fields import COLD_START_SEEDED_NUMBERS
    rng = np.random.default_rng(28284)
    state = SimpleNamespace()
    for mass, number in COLD_START_SEEDED_NUMBERS[scheme]:
        q = rng.uniform(1e-12, 0.003, shape).astype(np.float32)
        q.flat[::3] = 0
        q.flat[1::11] = np.float32(5e-13)
        q.flat[2::31] = np.float32(1e-35)
        n = rng.uniform(1, 1e6, shape).astype(np.float32)
        n.flat[::2] = 0
        setattr(state, mass, q)
        setattr(state, number, n)
    alt = rng.uniform(0.3, 3, shape)
    temp = rng.uniform(200, 315, shape).astype(np.float32)
    aerosol = rng.uniform(1e6, 1e8, shape).astype(np.float32)
    aerosol.flat[::2] = 0
    land = rng.integers(0, 2, shape[-2:])
    return state, alt, dict(aerosol_number=aerosol, landmask=land, temperature=temp)


@pytest.mark.parametrize("scheme", [8, 28])
@pytest.mark.parametrize("workers", [1, 8, 24])
@pytest.mark.parametrize("receipt", [False, True])
def test_native_numbers_and_receipt_match_old_host(scheme, workers, receipt):
    state, alt, kwargs = make_state(scheme)
    expected = deepcopy(state)
    cfg = SimpleNamespace(mp_physics=scheme)
    with np.errstate(all="ignore"), pm.worker_limit(workers):
        reference = real._thompson_cold_start_moment_closure_reference(
            expected, np, cfg, alt, **kwargs, receipt=receipt)
        actual = real._thompson_cold_start_moment_closure(
            state, np, cfg, alt, **kwargs, receipt=receipt)
    assert json.dumps(actual, sort_keys=True) == json.dumps(reference, sort_keys=True)
    for name, value in vars(expected).items():
        assert getattr(state, name).tobytes() == value.tobytes(), name


def test_native_temperature_is_the_same_portable_theta_conversion():
    state, alt, kwargs = make_state(8, (4, 43, 47))
    expected = deepcopy(state)
    rng = np.random.default_rng(391)
    theta = rng.uniform(250, 350, alt.shape)
    pressure = rng.uniform(8000, 101000, alt.shape)
    # Contiguity alone is not an aligned-buffer contract for typed FFI reads.
    def unaligned(value):
        result = np.ndarray(value.shape, dtype=value.dtype,
                            buffer=bytearray(value.nbytes + 1), offset=1)
        result[...] = value
        return result
    theta, pressure, alt = map(unaligned, (theta, pressure, alt))
    cfg = SimpleNamespace(mp_physics=8)
    def selected(indices):
        return real._temperature_from_potential_temperature(
            theta.ravel()[indices], pressure.ravel()[indices], column_workers=8).astype(np.float32)
    with pm.worker_limit(8):
        reference = real._thompson_cold_start_moment_closure_reference(
            expected, np, cfg, alt, temperature_at=selected)
        actual = real._thompson_cold_start_moment_closure(
            state, np, cfg, alt, temperature_fields=(theta, pressure))
    assert actual == reference
    assert state.nr.tobytes() == expected.nr.tobytes()
    assert state.ni.tobytes() == expected.ni.tobytes()


@pytest.mark.parametrize("case", ["zero_alt", "nan_alt", "missing_temperature", "missing_land",
                                   "nan_aerosol", "invalid_unseeded_number", "existing_numbers", "subnormal_seed",
                                   "large_finite_temperature", "infinite_alt_cloud"])
def test_native_refusals_and_untouched_numbers_match_reference(case):
    scheme = 28 if case in ("missing_land", "nan_aerosol", "infinite_alt_cloud") else 8
    state, alt, kwargs = make_state(scheme, (2, 7, 11))
    if case == "zero_alt":
        alt.flat[2:4] = (0, -0.)
    elif case == "nan_alt":
        alt.flat[2] = np.nan
    elif case == "missing_temperature":
        kwargs["temperature"] = None
    elif case == "missing_land":
        kwargs["landmask"] = None
        kwargs["aerosol_number"][:] = 0
    elif case == "nan_aerosol":
        kwargs["aerosol_number"][:] = np.nan
    elif case == "invalid_unseeded_number":
        state.qr.flat[0], state.nr.flat[0] = 0, -1
    elif case == "existing_numbers":
        state.nr[:] = 123
        state.ni[:] = 234
    elif case == "subnormal_seed":
        state.qr.flat[2] = np.nextafter(np.float32(0), np.float32(1))
        state.nr.flat[2] = 0
        alt.flat[2] = 3.0
    elif case == "large_finite_temperature":
        kwargs["temperature"].flat[4] = np.float32(1e20)
    elif case == "infinite_alt_cloud":
        state.qc.flat[4], state.nc.flat[4], alt.flat[4] = 1e-4, 0, np.inf
        kwargs["aerosol_number"].flat[4] = 1
    expected = deepcopy(state)
    cfg = SimpleNamespace(mp_physics=scheme)
    def run(function, target):
        with np.errstate(all="ignore"), pm.worker_limit(8):
            try:
                return function(target, np, cfg, alt, **kwargs)
            except (ValueError, IndexError) as error:
                return type(error).__name__, str(error)
    assert run(real._thompson_cold_start_moment_closure, state) == run(
        real._thompson_cold_start_moment_closure_reference, expected)
    for name, value in vars(expected).items():
        assert getattr(state, name).tobytes() == value.tobytes(), name


@pytest.mark.parametrize("alt", [0.0, np.nan])
def test_the_density_refusal_is_reached_without_a_state(alt):
    """THE BREAKAGE: the native closure read the state before anything else.

    The reference refuses an inverse density that is zero or not finite
    before it reads the state, and 2.8.4's host closure did too.  The
    native closure looked up the first mass field first, so the same call
    died with ``AttributeError: 'NoneType' object has no attribute 'qr'``
    (tests/test_thompson_cold_start_gpu.py holds host and device to one
    sentence with exactly this call).
    """
    cfg = SimpleNamespace(mp_physics=8)
    with pytest.raises(ValueError, match="inverse density is zero or not "
                                         "finite in 1 cell"):
        real._thompson_cold_start_moment_closure(
            None, np, cfg, np.array([alt], dtype=np.float32))


@pytest.mark.parametrize("closure", ["native", "reference"])
@pytest.mark.parametrize("case", ["nan_aerosol", "infinite_alt_cloud"])
def test_a_droplet_table_row_that_does_not_exist_is_refused_by_name(case, closure):
    """THE BREAKAGE: the cold start refused these states with NumPy's text.

    A NaN aerosol number, or a droplet number that is NaN at the entry
    block, has no gamma-table row.  The NumPy mirror failed there with
    ``IndexError: index 9223372036854775807 is out of bounds for axis 0
    with size 15`` (``-9223372036854775808`` and ``size 16`` at the entry
    block), and the native closure raised those two strings verbatim: a
    refusal that names neither the field, the cells nor what cannot be
    done.  Both closures now say which field, how many cells, and which
    table row could not be chosen, as the ValueError every other
    cold-start refusal is.
    """
    state, alt, kwargs = make_state(28, (2, 7, 11))
    if case == "nan_aerosol":
        kwargs["aerosol_number"][:] = np.nan
        cells = int(np.count_nonzero((state.qc > 0) & (state.nc <= 0)))
        assert cells > 1
        named = (f"{cells} cloudy cell\\(s\\) have a water-friendly aerosol "
                 "number per volume \\(QNWFA times density\\) that is not a "
                 "number.*make_DropletNumber.*no droplet number can be seeded")
    else:
        state.qc.flat[4], state.nc.flat[4], alt.flat[4] = 1e-4, 0, np.inf
        kwargs["aerosol_number"].flat[4] = 1
        named = ("1 cloudy cell\\(s\\) reach Thompson's entry block with a "
                 "droplet number per volume that is not a number.*"
                 "inverse density.*cannot be closed")
    before = deepcopy(state)
    cfg = SimpleNamespace(mp_physics=28)
    function = (real._thompson_cold_start_moment_closure if closure == "native"
                else real._thompson_cold_start_moment_closure_reference)
    with np.errstate(all="ignore"), pm.worker_limit(8):
        with pytest.raises(ValueError, match=named) as refused:
            function(state, np, cfg, alt, **kwargs)
    assert str(refused.value).startswith("mp_physics=28 cold start: ")
    assert "out of bounds" not in str(refused.value)
    # A refusal publishes nothing.
    for name, value in vars(before).items():
        assert getattr(state, name).tobytes() == value.tobytes(), name


@pytest.mark.parametrize("field,shape", [("landmask", (7, 1)), ("landmask", (11, 7)),
                                         ("aerosol_number", (154,))])
def test_unusual_broadcasts_keep_reference_contract(field, shape):
    state, alt, kwargs = make_state(28, (2, 7, 11))
    kwargs[field] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 2
    expected = deepcopy(state)
    cfg = SimpleNamespace(mp_physics=28)
    def run(function, target):
        with np.errstate(all="ignore"):
            try:
                result = function(target, np, cfg, alt, **kwargs)
                return json.dumps(result, sort_keys=True)
            except (ValueError, IndexError) as error:
                return type(error).__name__, str(error)
    assert run(real._thompson_cold_start_moment_closure, state) == run(
        real._thompson_cold_start_moment_closure_reference, expected)
    for name, value in vars(expected).items():
        assert getattr(state, name).tobytes() == value.tobytes(), name
