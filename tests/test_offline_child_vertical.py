"""The offline child may carry a deeper vertical ladder than its parent.

The shipped path -- a child at the parent's own level count -- must stay
BITWISE what it is today; that lock is the first test here and it is the one
that goes red if the remap leaks into the trajectory it does not belong on.
"""

import hashlib
from datetime import datetime

import netCDF4
import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.grid import compute_hybrid_coeffs
from gpuwm.offline_child import (
    OfflineChildContractError,
    OfflineChildPlacement,
    bind_parent_physics_from_wrf_namelist,
    build_offline_child_domain_state,
    interpolate_parent_boundary_snapshot,
    interpolate_parent_initial_state,
)
from gpuwm.vertical_remap import column_integral, dry_mass_edges

P_TOP = 10000.0
MUB = 80000.0
HGT = 350.0


def _ladder(nz, stretch=None):
    if stretch is None:
        return np.linspace(1.0, 0.0, nz + 1)
    zeta = np.linspace(0.0, 1.0, nz + 1)
    znw = np.tanh(stretch * (1.0 - zeta)) / np.tanh(stretch)
    znw[0] = 1.0
    znw[-1] = 0.0
    return znw


def _physics_binding(tmp_path, *, mp=8):
    path = tmp_path / f"namelist-mp{mp}.input"
    path.write_text(f"&physics\n mp_physics = {mp},\n/\n", encoding="utf-8")
    return bind_parent_physics_from_wrf_namelist(path)


def _variable(dataset, name, dims, value):
    dataset.createVariable(name, "f4", dims)[:] = np.asarray(
        value, dtype=np.float32)


def _deep_history(path, valid_time, *, nz, stretch=1.8, ny=12, nx=14,
                  hybrid_opt=2, etac=0.2, p_top=P_TOP):
    """A parent frame on an arbitrary ladder, hydrostatically consistent.

    Distinct from ``tests/test_offline_child.py::_history`` on purpose: that
    fixture's nz=2 arrays are pinned by fifty other tests.  This one builds
    PB/PHB through the same relations ``make_base_state`` uses, so the base
    state a remap reads back is the one the dycore would have built.
    """
    znw = _ladder(nz, stretch)
    hy = compute_hybrid_coeffs(znw, hybrid_opt, etac, float(c.P0), p_top)
    znu = 0.5 * (znw[:-1] + znw[1:])
    pb = hy["c3h"] * MUB + hy["c4h"] + p_top
    thb = 300.0 * (1.0 + 0.15 * (1.0 - znu))
    alb = c.RD * thb * (pb / c.P0) ** c.RCP / pb
    dnw = np.diff(znw)
    phb = np.empty(nz + 1)
    phb[0] = c.G * HGT
    for k in range(nz):
        phb[k + 1] = phb[k] - dnw[k] * (hy["c1h"][k] * MUB + hy["c2h"][k]) * alb[k]
    # A smooth profile plus a sharp cloud layer: a remap that smeared the
    # layer would still conserve, so the test measures the integral AND the
    # peak stays inside the column.
    qv = (8.0e-3 * np.exp(-3.0 * (1.0 - znu))
          + 4.0e-3 * np.exp(-((znu - 0.55) ** 2) / (2 * 0.02 ** 2)))

    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
                ("Time", 1), ("DateStrLen", 19), ("west_east", nx),
                ("south_north", ny), ("bottom_top", nz),
                ("west_east_stag", nx + 1), ("south_north_stag", ny + 1),
                ("bottom_top_stag", nz + 1)):
            dataset.createDimension(name, size)
        dataset.TITLE = "gpuwm offline-child deep-ladder fixture"
        dataset.DX = 1000.0
        dataset.DY = 1000.0
        dataset.MAP_PROJ = 1
        dataset.TRUELAT1 = 30.0
        dataset.TRUELAT2 = 60.0
        dataset.STAND_LON = -97.0
        dataset.CEN_LAT = 35.0
        dataset.CEN_LON = -97.0
        dataset.HYBRID_OPT = hybrid_opt
        dataset.ETAC = etac
        dataset.GPUWM_WRITE_COMPLETE = 1
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(
            valid_time.strftime("%Y-%m-%d_%H:%M:%S").encode(), dtype="S1")
        mass3 = ("Time", "bottom_top", "south_north", "west_east")
        u3 = ("Time", "bottom_top", "south_north", "west_east_stag")
        v3 = ("Time", "bottom_top", "south_north_stag", "west_east")
        w3 = ("Time", "bottom_top_stag", "south_north", "west_east")
        mass2 = ("Time", "south_north", "west_east")

        def col3(profile, dims=mass3, shape=None):
            shape = shape or (1, len(profile), ny, nx)
            return np.broadcast_to(
                np.asarray(profile, dtype=np.float32)[None, :, None, None],
                shape)

        for name in ("P", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
            _variable(dataset, name, mass3, np.zeros((1, nz, ny, nx)))
        _variable(dataset, "QVAPOR", mass3, col3(qv))
        _variable(dataset, "QNRAIN", mass3, np.zeros((1, nz, ny, nx)))
        _variable(dataset, "QNICE", mass3, np.zeros((1, nz, ny, nx)))
        _variable(dataset, "PB", mass3, col3(pb))
        _variable(dataset, "T", mass3, col3(thb - 300.0))
        _variable(dataset, "U", u3, col3(5.0 + 2.0 * znu,
                                         shape=(1, nz, ny, nx + 1)))
        _variable(dataset, "V", v3, col3(-2.0 + znu,
                                         shape=(1, nz, ny + 1, nx)))
        _variable(dataset, "W", w3, np.zeros((1, nz + 1, ny, nx)))
        _variable(dataset, "PH", w3, np.zeros((1, nz + 1, ny, nx)))
        _variable(dataset, "PHB", w3, col3(phb, shape=(1, nz + 1, ny, nx)))
        _variable(dataset, "MU", mass2, np.zeros((1, ny, nx)))
        _variable(dataset, "MUB", mass2, np.full((1, ny, nx), MUB))
        _variable(dataset, "MAPFAC_M", mass2, np.ones((1, ny, nx)))
        _variable(dataset, "MAPFAC_U",
                  ("Time", "south_north", "west_east_stag"),
                  np.ones((1, ny, nx + 1)))
        _variable(dataset, "MAPFAC_V",
                  ("Time", "south_north_stag", "west_east"),
                  np.ones((1, ny + 1, nx)))
        _variable(dataset, "HGT", mass2, np.full((1, ny, nx), HGT))
        _variable(dataset, "PSFC", mass2, np.full((1, ny, nx), MUB + p_top))
        _variable(dataset, "F", mass2, np.full((1, ny, nx), 8.0e-5))
        _variable(dataset, "E", mass2, np.full((1, ny, nx), 1.0e-4))
        _variable(dataset, "SINALPHA", mass2, np.zeros((1, ny, nx)))
        _variable(dataset, "COSALPHA", mass2, np.ones((1, ny, nx)))
        _variable(dataset, "XLAT", mass2, np.full((1, ny, nx), 35.0))
        _variable(dataset, "XLONG", mass2, np.full((1, ny, nx), -97.0))
        _variable(dataset, "P_TOP", ("Time",), [p_top])
        _variable(dataset, "ZNU", ("Time", "bottom_top"), [znu])
        _variable(dataset, "ZNW", ("Time", "bottom_top_stag"), [znw])
    return znw


def _placement():
    return OfflineChildPlacement(
        parent_nx=14, parent_ny=12, child_nx=6, child_ny=6,
        parent_grid_ratio=1, i_parent_start=4, j_parent_start=4)


def _cfg(nz, *, eta_levels=None, **kwargs):
    settings = dict(
        nx=6, ny=6, nz=nz, dx=1000.0, dy=1000.0, ztop=9000.0, dt=5.0,
        run_seconds=300.0, hybrid_opt=2, etac=0.2, moist=True, mp_physics=8,
        specified=True, nested=False, terrain_opt=1, map_proj=1,
        hypsometric_opt=1)
    settings.update(kwargs)
    if eta_levels is not None:
        settings["eta_levels"] = tuple(float(v) for v in eta_levels)
    return RunConfig(**settings)


# --------------------------------------------------------------------------
# The OFF trajectory: a child at the parent's own ladder must not move.
# --------------------------------------------------------------------------

def test_matched_ladder_offline_child_is_bitwise_unchanged(tmp_path):
    """A child that declares no ladder takes the shipped path untouched.

    Compared against arrays captured from the pre-change tree (``git show
    d9e48213d:gpuwm/offline_child.py``), not against the post-change file, so
    the baseline cannot drift with the change it is policing.
    """
    path = tmp_path / "parent.nc"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=8)
    initial = interpolate_parent_initial_state(
        path, _placement(), physics_binding=_physics_binding(tmp_path),
        backend="cpu")
    state = build_offline_child_domain_state(initial, _cfg(8),
                                             array_module=np)
    for name, (digest, shape, dtype) in _BASELINE_MATCHED_LADDER.items():
        actual = np.ascontiguousarray(np.asarray(getattr(state, name)))
        assert actual.shape == shape, name
        assert str(actual.dtype) == dtype, name
        assert hashlib.sha256(actual.tobytes()).hexdigest() == digest, (
            f"{name} moved off the shipped trajectory")


def test_a_one_ulp_child_interface_breaks_the_bitwise_lock(tmp_path):
    """NEGATIVE CONTROL: the lock above must be able to go red."""
    path = tmp_path / "parent.nc"
    znw = _deep_history(path, datetime(1974, 4, 3, 12), nz=8)
    nudged = znw.copy()
    nudged[4] = np.nextafter(nudged[4], 1.0)
    initial = interpolate_parent_initial_state(
        path, _placement(), physics_binding=_physics_binding(tmp_path),
        backend="cpu", child_eta_levels=nudged)
    state = build_offline_child_domain_state(
        initial, _cfg(8, eta_levels=nudged), array_module=np)
    actual = np.ascontiguousarray(np.asarray(state.thb))
    assert (hashlib.sha256(actual.tobytes()).hexdigest()
            != _BASELINE_MATCHED_LADDER["thb"][0])


# --------------------------------------------------------------------------
# The capability: a deeper child ladder.
# --------------------------------------------------------------------------

def test_offline_child_admits_a_deeper_child_ladder(tmp_path):
    """8 parent levels -> 16 child levels, water substance conserved."""
    path = tmp_path / "parent.nc"
    parent_znw = _deep_history(path, datetime(1974, 4, 3, 12), nz=8)
    child_znw = _ladder(16, 2.2)
    initial = interpolate_parent_initial_state(
        path, _placement(), physics_binding=_physics_binding(tmp_path),
        backend="cpu", child_eta_levels=child_znw)
    state = build_offline_child_domain_state(
        initial, _cfg(16, eta_levels=child_znw), array_module=np)

    assert state.qv.shape == (16, 6, 6)
    assert state.thb.shape == (16, 6, 6)
    assert state.php.shape == (17, 6, 6)

    mu = np.asarray(state.total_mu(), dtype=np.float64)
    parent_edges = dry_mass_edges(parent_znw, hybrid_opt=2, etac=0.2,
                                  p_top=P_TOP, mu=mu)
    child_edges = dry_mass_edges(child_znw, hybrid_opt=2, etac=0.2,
                                 p_top=P_TOP, mu=mu)
    znu = 0.5 * (parent_znw[:-1] + parent_znw[1:])
    parent_qv = np.broadcast_to(
        (8.0e-3 * np.exp(-3.0 * (1.0 - znu))
         + 4.0e-3 * np.exp(-((znu - 0.55) ** 2) / (2 * 0.02 ** 2))
         ).astype(np.float32).astype(np.float64)[:, None, None],
        (8, 6, 6))
    before = column_integral(parent_edges, parent_qv)
    after = column_integral(child_edges,
                            np.asarray(state.qv, dtype=np.float64))
    assert np.abs((after - before) / before).max() < 1e-6

    # The child's geopotential must still satisfy the dycore's own discrete
    # hydrostatic relation, and its model top must sit where the parent's did.
    assert np.isfinite(np.asarray(state.phb)).all()
    assert np.allclose(np.asarray(state.phb)[0], c.G * HGT, rtol=0, atol=1e-2)


def test_a_deeper_child_conserves_dry_mass_exactly(tmp_path):
    """The column's total dry mass is the same on both ladders."""
    path = tmp_path / "parent.nc"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=8)
    child_znw = _ladder(16, 2.2)
    initial = interpolate_parent_initial_state(
        path, _placement(), physics_binding=_physics_binding(tmp_path),
        backend="cpu", child_eta_levels=child_znw)
    state = build_offline_child_domain_state(
        initial, _cfg(16, eta_levels=child_znw), array_module=np)
    mu = np.asarray(state.total_mu(), dtype=np.float64)
    edges = dry_mass_edges(child_znw, hybrid_opt=2, etac=0.2, p_top=P_TOP,
                           mu=mu)
    assert np.abs((edges[:-1] - edges[1:]).sum(axis=0) - mu).max() < 1e-6


def test_a_deeper_child_lateral_boundary_lands_on_the_child_ladder(tmp_path):
    """Boundary strips must arrive at the CHILD's level count.

    Without this the child's own lateral tables would be sized for the
    parent's ladder and the first boundary blend would refuse on shape.
    """
    path = tmp_path / "parent.nc"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=8)
    child_znw = _ladder(16, 2.2)
    snapshot = interpolate_parent_boundary_snapshot(
        path, _placement(), source_mp_physics=8, child_eta_levels=child_znw)
    assert snapshot.fields["qv"].shape[0] == 16
    assert snapshot.fields["w"].shape[0] == 17


# --------------------------------------------------------------------------
# The three parameters that must stay shared.
# --------------------------------------------------------------------------

def test_a_child_ladder_that_is_not_a_ladder_is_refused(tmp_path):
    path = tmp_path / "parent.nc"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=8)
    with pytest.raises((OfflineChildContractError, ValueError)) as excinfo:
        interpolate_parent_initial_state(
            path, _placement(), physics_binding=_physics_binding(tmp_path),
            backend="cpu",
            child_eta_levels=np.array([1.0, 0.5, 0.6, 0.0]))
    assert "decreas" in str(excinfo.value).lower()


def test_the_child_ladder_count_must_match_the_child_config(tmp_path):
    """A ladder of a different length than cfg.nz is refused, not truncated."""
    path = tmp_path / "parent.nc"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=8)
    child_znw = _ladder(16, 2.2)
    initial = interpolate_parent_initial_state(
        path, _placement(), physics_binding=_physics_binding(tmp_path),
        backend="cpu", child_eta_levels=child_znw)
    with pytest.raises(OfflineChildContractError):
        build_offline_child_domain_state(
            initial, _cfg(12, eta_levels=_ladder(12, 2.2)), array_module=np)


# --------------------------------------------------------------------------
# The radiation ceiling, per domain.
# --------------------------------------------------------------------------

def test_offline_child_refuses_a_ladder_its_radiation_cannot_run(tmp_path):
    """nz=120 + RRTMGP + the parent's p_top=5000 Pa exceeds the 128-layer cap.

    Reachable only once a child may carry its own nz.  ``RunConfig`` has no
    p_top of its own, so ``validate_run_config`` runs the p_top=None branch
    of the vertical preflight and the cap-layer arithmetic is skipped; the
    run would then die at the first radiative call, after the fetch and the
    whole preparation had been paid for.
    """
    path = tmp_path / "parent.nc"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=8, p_top=5000.0)
    child_znw = _ladder(120, 2.5)
    initial = interpolate_parent_initial_state(
        path, _placement(), physics_binding=_physics_binding(tmp_path),
        backend="cpu", child_eta_levels=child_znw)
    with pytest.raises(OfflineChildContractError) as excinfo:
        build_offline_child_domain_state(
            initial,
            _cfg(120, eta_levels=child_znw, ra_lw_physics=4, ra_sw_physics=4),
            array_module=np)
    message = str(excinfo.value)
    assert "model plus cap layers <= 128" in message
    assert "120+13=133" in message


def test_a_ladder_inside_the_radiation_cap_is_admitted(tmp_path):
    """NEGATIVE CONTROL: 100+13=113 <= 128 must run.

    A gate that refuses every deep ladder is not a gate.
    """
    path = tmp_path / "parent.nc"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=8, p_top=5000.0)
    child_znw = _ladder(100, 2.5)
    initial = interpolate_parent_initial_state(
        path, _placement(), physics_binding=_physics_binding(tmp_path),
        backend="cpu", child_eta_levels=child_znw)
    state = build_offline_child_domain_state(
        initial,
        _cfg(100, eta_levels=child_znw, ra_lw_physics=4, ra_sw_physics=4),
        array_module=np)
    assert state.qv.shape[0] == 100


def _bare_fields(ny=4, nx=4, nz=8):
    return {
        "MUB": np.full((ny, nx), MUB),
        "MU": np.zeros((ny, nx)),
        "HGT": np.full((ny, nx), HGT),
        "PB": np.zeros((nz, ny, nx)),
        "PHB": (np.linspace(0.0, 1.0, nz + 1)[:, None, None]
                * np.ones((nz + 1, ny, nx)) * 1.0e5),
        "PH": np.zeros((nz + 1, ny, nx)),
        "T": np.zeros((nz, ny, nx)),
        "W": np.zeros((nz + 1, ny, nx)),
    }


def test_an_unweighted_parent_ladder_field_is_refused_not_passed_through():
    """A 3-D field with no remap weight must refuse, not ride through.

    Every field in ``_INITIAL_CORE_FIELDS`` has a weight today.  One added
    later without one would otherwise reach the child at the PARENT's level
    count inside a dict whose other members are on the child's, and the
    per-field shape checks downstream do not all catch that.
    """
    from gpuwm.offline_child import _remap_initial_state_to_child_ladder

    fields = _bare_fields()
    fields["SOMETHING_NEW"] = np.zeros((8, 4, 4))
    with pytest.raises(OfflineChildContractError) as excinfo:
        _remap_initial_state_to_child_ladder(
            fields, {}, parent_znw=_ladder(8, 1.8),
            child_znw=_ladder(16, 2.2), hybrid_opt=2, etac=0.2, p_top=P_TOP)
    assert "SOMETHING_NEW" in str(excinfo.value)


def test_a_two_dimensional_field_still_rides_through():
    """NEGATIVE CONTROL: the guard must not refuse ladder-free fields.

    ``QNWFA2D``/``QNIFA2D`` are per-domain surface constants that reach the
    child through ``initial.fields``; a guard that refused them would break
    every mp=28 offline child.
    """
    from gpuwm.offline_child import _remap_initial_state_to_child_ladder

    fields = _bare_fields()
    fields["QNWFA2D"] = np.full((4, 4), 4321.0)
    out, _, _ = _remap_initial_state_to_child_ladder(
        fields, {}, parent_znw=_ladder(8, 1.8), child_znw=_ladder(16, 2.2),
        hybrid_opt=2, etac=0.2, p_top=P_TOP)
    assert np.array_equal(out["QNWFA2D"], fields["QNWFA2D"])
    assert out["T"].shape[0] == 16

#: sha256 of every state array of a matched-ladder offline child, captured
#: from the PRE-CHANGE tree (``git show d9e48213d:gpuwm/offline_child.py``)
#: before this lane touched the file.  Compared against the pre-change tree
#: rather than against the post-change file so the baseline cannot drift with
#: the change it is policing.
_BASELINE_MATCHED_LADDER = {
    "u": ("00cb31169bb01a760088744c93df5c8b0b48ba192da76cc0d53e1df3fe9f55b1", (8, 6, 7), "float32"),
    "v": ("75774b951049092a71ed925bca277be0f97297242a6885547c9d65119965be06", (8, 7, 6), "float32"),
    "w": ("94637c6efefbdcc3d3bb74d61732b22250552654c8c11f0fa9c3b3ed11d38373", (9, 6, 6), "float32"),
    "php": ("94637c6efefbdcc3d3bb74d61732b22250552654c8c11f0fa9c3b3ed11d38373", (9, 6, 6), "float32"),
    "mup": ("81c611f35bff79491538b2f7cf201c7597a661a5c549633541c62bdc8af1613f", (6, 6), "float32"),
    "thp": ("f4372188ab3753a29474799ba8317003aac09a4917f7c2996902c6d9f7cd7314", (8, 6, 6), "float32"),
    "thb": ("cb2b02bb0fb20342c1a52e7971edaca093f2f741d657cd444c44efe0c848247b", (8, 6, 6), "float32"),
    "phb": ("81ffe56f006fe063b30b6c3517a1b3273cc784e114fb3feb2085703936761f12", (9, 6, 6), "float32"),
    "pb": ("75f3ee4f7b4fa38667b52f10ecda28acd0256d3dbe59c2af1b81be76e6ea92d0", (8, 6, 6), "float32"),
    "alb": ("68c5f510500074f2625b19b5bbe25ef547500e05bec78311e38a749c4ddbfff7", (8, 6, 6), "float32"),
    "qv": ("4bf13817651334ffdff9f0feea0355410e3517552aed88adbc42e5f04a4077aa", (8, 6, 6), "float32"),
    "qc": ("4cf9816ed1062189ff0c8d427fba5e912cc68fc9af76cf7f08fd255977de3b33", (8, 6, 6), "float32"),
    "qr": ("4cf9816ed1062189ff0c8d427fba5e912cc68fc9af76cf7f08fd255977de3b33", (8, 6, 6), "float32"),
    "qi": ("4cf9816ed1062189ff0c8d427fba5e912cc68fc9af76cf7f08fd255977de3b33", (8, 6, 6), "float32"),
    "qs": ("4cf9816ed1062189ff0c8d427fba5e912cc68fc9af76cf7f08fd255977de3b33", (8, 6, 6), "float32"),
    "qg": ("4cf9816ed1062189ff0c8d427fba5e912cc68fc9af76cf7f08fd255977de3b33", (8, 6, 6), "float32"),
    "alt": ("8093f42e0718c774c4a1248085c5e4d9b68b51af90a49ef79a1fce5b308df292", (8, 6, 6), "float32"),
    "p": ("efcc02d816407abdb2390c0c3f0a07b21b89ddffc7fe3fdc7080c5f9b289c519", (8, 6, 6), "float32"),
    "al": ("215e77fd69be2dc4d68c96ed50801a2469ed9d6e74573d1ff4ed9a3988f3e1b1", (8, 6, 6), "float32"),
}


# ---------------------------------------------------------------------------
# CROSS-LANE INTERACTION: a child on its OWN ladder, on the FP32 EOS spelling
# ---------------------------------------------------------------------------
#
# Two changes landed together on 2026-09-03.  Per-domain vertical ladders
# rebuild a child's base state on a grid its parent never had; the FP32 EOS
# spelling made the base state carry ``dphb_resid``, the float64 layer
# geopotential thickness minus the float32 subtraction the kernel performs
# on the stored ``phb``.  The residual is a function of the profile it was
# derived from, so a child that got its parent's residual against its own
# remapped ``phb`` would diagnose a pressure for a column it does not have.
#
# These pin that they compose: the remapped child base goes through
# ``DomainState.load_base`` -> ``set_base_geopotential``, so the residual is
# rebuilt from the CHILD's geopotential on the CHILD's ladder.


def _child_on_its_own_ladder(tmp_path, *, parent_nz=8, child_nz=16,
                             stretch=2.2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "parent.nc"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=parent_nz)
    child_znw = _ladder(child_nz, stretch)
    initial = interpolate_parent_initial_state(
        path, _placement(), physics_binding=_physics_binding(tmp_path),
        backend="cpu", child_eta_levels=child_znw)
    return build_offline_child_domain_state(
        initial, _cfg(child_nz, eta_levels=child_znw), array_module=np), \
        child_znw


def test_a_deeper_child_carries_its_own_geopotential_residual(tmp_path):
    """The residual describes the CHILD's phb, on the CHILD's ladder.

    THE DEFECT THIS PINS.  The remap builds the child's base geopotential
    in float64 on a ladder the parent never had, and then every field was
    rounded to float32 one line later, so ``set_base_geopotential``
    subtracted a float32 profile from itself and ``dphb_resid`` came out
    IDENTICALLY ZERO.  The FP32 EOS remedy was therefore off on precisely
    the deep-column route it exists for, and on by default everywhere
    else.
    """
    state, _ = _child_on_its_own_ladder(tmp_path)

    resid = np.asarray(state.dphb_resid, dtype=np.float64)
    stored = np.asarray(state.phb, dtype=np.float32)
    assert resid.shape == (16,) + stored.shape[1:], resid.shape
    assert np.isfinite(resid).all()

    # It IS the residual of the float64 geopotential the remap produced:
    # that thickness minus the float32 subtraction the kernel performs on
    # the stored profile.  _phb_host is the float64 snapshot the setter
    # keeps; state.phb is the float32 store the kernel reads.
    host = np.asarray(state._phb_host, dtype=np.float64)
    expected = np.asarray(
        np.diff(host, axis=0) - np.diff(stored, axis=0).astype(np.float64),
        dtype=np.float32)
    np.testing.assert_array_equal(resid.astype(np.float32), expected)

    # NEGATIVE CONTROL, and the defect stated as a number: the correction
    # is not zero, so the equality above is not passing on an empty array.
    assert np.abs(resid).max() > 0.0
    # It is worth what it should be worth: at most an ulp of the stored
    # geopotential, and of the order of one for the deepest layer.
    assert np.abs(resid).max() <= np.spacing(np.abs(stored).max())


def test_the_deeper_childs_residual_is_built_on_the_child_ladder(tmp_path):
    """A residual inherited from the parent would have the parent's shape."""
    deep, _ = _child_on_its_own_ladder(tmp_path / "deep", child_nz=16,
                                       stretch=2.2)
    shallow, _ = _child_on_its_own_ladder(tmp_path / "flat", child_nz=8,
                                          stretch=None)
    deep_resid = np.asarray(deep.dphb_resid, dtype=np.float64)
    shallow_resid = np.asarray(shallow.dphb_resid, dtype=np.float64)
    assert deep_resid.shape[0] == 16 and shallow_resid.shape[0] == 8

    # And the two describe genuinely different columns, so "it has the
    # right shape" is not the whole of the claim.
    deep_dz = np.diff(np.asarray(deep._phb_host, dtype=np.float64), axis=0)
    shallow_dz = np.diff(np.asarray(shallow._phb_host, dtype=np.float64),
                         axis=0)
    assert deep_dz.shape[0] == 2 * shallow_dz.shape[0]
    assert not np.allclose(deep_dz[::2], shallow_dz)


def test_a_deeper_childs_geopotential_thickness_is_recovered(tmp_path):
    """The two changes meet here: remapped base, residual-corrected EOS.

    The kernel adds ``dphb_resid`` to its own float32 difference of the
    stored profile.  That sum must come back to the float64 thickness the
    remap produced, which is what stops the diagnosed pressure degrading
    as 1/dz on a child that asked for more levels.
    """
    state, _ = _child_on_its_own_ladder(tmp_path)

    host = np.asarray(state._phb_host, dtype=np.float64)
    stored = np.asarray(state.phb, dtype=np.float32)
    resid = np.asarray(state.dphb_resid, dtype=np.float32)

    truth = np.diff(host, axis=0)
    recovered = (np.diff(stored, axis=0).astype(np.float64)
                 + resid.astype(np.float64))
    plain = np.diff(stored, axis=0).astype(np.float64)

    corrected_err = np.abs(recovered - truth).max() / np.abs(truth).min()
    plain_err = np.abs(plain - truth).max() / np.abs(truth).min()
    # The correction is exact to float32, and it is doing real work.
    assert corrected_err <= 1e-9
    assert plain_err > 100.0 * max(corrected_err, 1e-12)

    # The child's base state is self-consistent on its own ladder.
    pb = np.asarray(state.pb, dtype=np.float64)
    alb = np.asarray(state.alb, dtype=np.float64)
    assert np.isfinite(pb).all() and (pb > 0.0).all()
    assert np.isfinite(alb).all() and (alb > 0.0).all()
