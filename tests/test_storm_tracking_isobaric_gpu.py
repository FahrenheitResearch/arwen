"""The tracker's device isobaric height equals the Rust crate word for word.

``gpuwm.core.storm_tracking`` reads a device state's isobaric surface with a
CUDA transcription of ``rw_isobaric::column_isobaric_height`` (explicit
round-to-nearest operations in the crate's order, ``plm_log`` for the
crate's ``libm`` logarithm).  This grades it on a card against the crate
itself, through :mod:`gpuwm.isobaric_bridge`, on the same float32 state
widened to float64: every plane must be the same 64-bit words, NaN where
the crate says NaN.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

cp = pytest.importorskip("cupy")

from gpuwm import isobaric_bridge  # noqa: E402
from gpuwm.core import storm_tracking as st  # noqa: E402


def _state(xp, *, nz=50, ny=60, nx=70, seed=11, profile_base=False):
    rng = np.random.default_rng(seed)
    znw = 1.0 - (np.arange(nz + 1) / nz) ** 1.25
    znu = 0.5 * (znw[:-1] + znw[1:])
    jj, ii = np.mgrid[0:ny, 0:nx]
    psfc = (101000.0 - 9000.0 * np.exp(-(((ii - 30) / 9.0) ** 2 + ((jj - 25) / 9.0) ** 2))
            - 18000.0 * (ii > 55) + rng.normal(0.0, 40.0, (ny, nx)))
    p_w = 2000.0 + znw[:, None, None] * (psfc[None] - 2000.0)
    p = (0.5 * (p_w[:-1] + p_w[1:])).astype(np.float32)
    temperature = 250.0 + rng.normal(0.0, 3.0, (nz + 1, ny, nx))
    phi = (287.0 * temperature * np.log(psfc[None] / p_w)).astype(np.float64)
    phb = (phi.mean(axis=(1, 2)) if profile_base else 0.6 * phi).astype(np.float32)
    php = (phi - (phb[:, None, None] if profile_base else phb)).astype(np.float32)
    from types import SimpleNamespace
    return SimpleNamespace(p=xp.asarray(p), php=xp.asarray(php), phb=xp.asarray(phb),
                           znu=znu, znw=znw)


@pytest.mark.parametrize("profile_base", [False, True], ids=["phb3d", "phb1d"])
def test_device_plane_is_the_crate_word_for_word(profile_base):
    if isobaric_bridge.unavailable_reason() is not None:
        pytest.skip(f"the rw-isobaric library is not built here "
                    f"({isobaric_bridge.unavailable_reason()})")
    device = _state(cp, profile_base=profile_base)
    host = _state(np, profile_base=profile_base)
    for level_hpa in (925.0, 850.0, 700.0, 500.0, 300.0, 250.0, 100.0):
        got = cp.asnumpy(st._device_isobaric_height(
            device, host.znu, host.znw, level_hpa * 100.0))
        crate = isobaric_bridge.isobaric_heights(
            host.php, host.p, (level_hpa * 100.0,), eta_interface=host.znw,
            eta_mass=host.znu, interface_plus=host.phb,
            per_metre=st.GRAVITY_M_S2)[0]
        assert np.array_equal(np.isnan(got), np.isnan(crate)), level_hpa
        finite = np.isfinite(crate)
        assert finite.any(), level_hpa
        assert np.array_equal(got[finite].view(np.uint64),
                              crate[finite].view(np.uint64)), level_hpa


def test_device_and_host_tracker_planes_agree():
    if isobaric_bridge.unavailable_reason() is not None:
        pytest.skip("the rw-isobaric library is not built here")
    device = _state(cp)
    host = _state(np)
    for level_hpa in (850.0, 500.0):
        a = st.level_height_m_from_state(device, level_hpa)
        b = st.level_height_m_from_state(host, level_hpa)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
