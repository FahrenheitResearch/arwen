"""The Rust isobaric-height reader through its Python seam.

``gpuwm.isobaric_bridge`` is how the Python consumers (the storm tracker on
a host state, the GNSS-RO operator, the verification diagnostics, the
flagship products) read a height between layer interfaces: the
``rw-isobaric`` crate every chart already uses, through its C ABI.  These
pin it on an isothermal eta column, where the height of any pressure is
known exactly and the layer-mean pairing it replaces is not exact.
"""

from pathlib import Path

import numpy as np
import pytest

from gpuwm import bridge_assets, bridges
from gpuwm import isobaric_bridge as ib

REPO_ROOT = Path(__file__).resolve().parents[1]

needs_library = pytest.mark.skipif(
    ib.unavailable_reason() is not None,
    reason=f"the rw-isobaric library is not built here "
           f"({ib.unavailable_reason()}); cd tools/rustwx && cargo build "
           "--release -p rw-isobaric --offline")


def _eta_columns(nz=40, psfc=(101000.0, 84000.0, 70500.0),
                 scale_height=(7200.0, 7600.0, 7000.0), p_top=2000.0):
    znw = 1.0 - (np.arange(nz + 1) / nz) ** 1.3
    psfc = np.asarray(psfc)[None, :]
    scale_height = np.asarray(scale_height)[None, :]
    p_w = p_top + znw[:, None] * (psfc - p_top)
    p_m = 0.5 * (p_w[:-1] + p_w[1:])
    z_w = scale_height * np.log(psfc / p_w)
    return znw, p_m, z_w, psfc[0], scale_height[0]


@needs_library
def test_heights_match_the_isothermal_truth_where_layer_means_read_high():
    znw, p_m, z_w, psfc, scale_height = _eta_columns()
    got = ib.isobaric_heights(z_w, p_m, (50000.0,), eta_interface=znw)[0]
    truth = scale_height * np.log(psfc / 50000.0)
    assert np.abs(got - truth).max() < 0.05
    # The pairing this replaces: layer means at the mass-level pressures.
    z_mean = 0.5 * (z_w[:-1] + z_w[1:])
    paired = np.array([np.interp(np.log(50000.0), np.log(p_m[::-1, c]),
                                 z_mean[::-1, c]) for c in range(3)])
    assert (paired - truth > 0.5).all()


@needs_library
def test_a_surface_under_the_ground_or_above_the_top_is_nan():
    znw, p_m, z_w, _, _ = _eta_columns()
    at_1000 = ib.isobaric_heights(z_w, p_m, (100000.0, 1000.0), eta_interface=znw)
    assert np.isfinite(at_1000[0, 0]) and np.isnan(at_1000[0, 1:]).all()
    assert np.isnan(at_1000[1]).all()


@needs_library
def test_split_geopotential_stencils_and_mass_heights_agree():
    znw, p_m, z_w, psfc, scale_height = _eta_columns()
    znu = 0.5 * (znw[:-1] + znw[1:])
    g = ib.STANDARD_GRAVITY
    one = ib.isobaric_heights(z_w, p_m, (70000.0, 30000.0), eta_interface=znw)
    split = ib.isobaric_heights(z_w * g * 0.25, p_m, (70000.0, 30000.0),
                                interface_plus=z_w * g * 0.75, per_metre=g,
                                eta_interface=znw, eta_mass=znu)
    assert np.allclose(one, split, rtol=0, atol=1e-6, equal_nan=True)
    rebuilt = ib.interfaces_from_layer_thickness(np.diff(znw))
    assert np.allclose(rebuilt, znw, rtol=0, atol=1e-12)
    z_m = ib.mass_level_heights(z_w, p_m, eta_interface=znw)
    assert np.abs(z_m - scale_height * np.log(psfc / p_m)).max() < 0.05
    log_p_w = ib.interface_log_pressure(p_m, eta_interface=znw)
    assert np.abs(log_p_w[0] - np.log(psfc)).max() < 1e-6


@needs_library
def test_bad_vertical_coordinates_are_refused_by_the_crate():
    znw, p_m, z_w, _, _ = _eta_columns(nz=4)
    with pytest.raises(ib.IsobaricBridgeError, match="eta interfaces"):
        ib.isobaric_heights(z_w, p_m, (50000.0,), eta_interface=znw[:-1])
    with pytest.raises(ib.IsobaricBridgeError, match="share one vertical coordinate"):
        ib.isobaric_heights(z_w, p_m, (50000.0,), eta_interface=znw,
                            eta_mass=np.full(4, 0.5))
    with pytest.raises(ib.IsobaricBridgeError, match="not a scale"):
        ib.isobaric_heights(z_w, p_m, (50000.0,), eta_interface=znw, per_metre=0.0)


def test_the_bundle_entry_and_handshake_match_the_module():
    """The bundle stages the file this seam opens, under its own variable,
    and the release probe asks it the question this module asks."""
    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == "rw_isobaric"]
    assert artifact.env_var == ib.ISOBARIC_BRIDGE_ENV
    assert artifact.kind == "library"
    assert artifact.crate == bridges.RUSTWX_CRATE_RELATIVE
    assert bridge_assets.library_abi_for("rw_isobaric") == (
        "gpuwm_isobaric_abi_version", ib.ISOBARIC_ABI)
    assert bridges.BRIDGE_ABI_MARKERS["rw_isobaric"] == ib.ABI_MARKER


def test_the_crate_injects_the_source_revision():
    """A bridge artifact the release cut cannot pin cannot ship."""
    crate = REPO_ROOT / "tools" / "rustwx" / "crates" / "rw-isobaric"
    assert "rustc-env=GPUWM_BRIDGE_SOURCE_REV" in (crate / "build.rs").read_text(
        encoding="utf-8")
    capi = (crate / "src" / "capi.rs").read_text(encoding="utf-8")
    assert "SOURCE_REV_STAMP" in capi
    assert 'env!("GPUWM_BRIDGE_SOURCE_REV")' in capi
    assert ib.ABI_MARKER.decode() in capi


def test_no_python_copy_of_the_arithmetic_remains():
    """The Python boundary: the height read is the crate's, and the module
    that once carried a Python copy of it is gone."""
    assert not (REPO_ROOT / "gpuwm" / "core" / "interface_height.py").exists()
