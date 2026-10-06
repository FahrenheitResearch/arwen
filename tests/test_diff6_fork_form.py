"""The NOAA-EMC WRFV3.9 fork's sixth-order filter and mp_zero_out.

What the fork does (NOAA-EMC/HRRR v4.1.21, sorc/hrrr_wrfarw.fd/WRFV3.9):

- the moist and scalar arrays are filtered with their own factor
  ``diff_6th_factor2`` (solve_em.F:2418, :2974; Registry.EM_COMMON:2629,
  default 0.04), while u, v, w, theta and TKE keep ``diff_6th_factor``
  (rk_tendency; solve_em.F:2587 for tke_2);
- rk_scalar_tend rebuilds the full step ``dt_rk*(rk_order-rk_step+1)``
  (module_em.F:1216-1217) and hands it to the filter (:1409-1412), where
  WRF v4.6.1 hands ``dt_step = grid%dt/3.``;
- sixth_order_diffusion loops to the domain edge for every field
  (module_big_step_utilities_em.F:6596-6632) and reads the halo, which
  set_physical_bc3d fills with zero-gradient copies under specified and
  nested boundaries (share/module_bc.F);
- microphysics_zero_outb/_outa run on the moist, scalar, chem and tracer
  arrays with no mp_zero_out_all switch (solve_em.F:4078-4141), which is
  WRF v4.6.1's ``mp_zero_out_all = 1``.

The port selects these with ``diff_6th_form = "noaa_wrf39"`` and
``mp_zero_out``/``mp_zero_out_thresh``/``mp_zero_out_all``; an imported
namelist that names ``diff_6th_factor2`` (a key only the fork declares)
selects the fork form.  Every other configuration keeps WRF v4.6.1.
"""
from __future__ import annotations

import tomllib

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.config import RunConfig, validate_run_config


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=16, ny=14, nz=6, dx=3000., dy=3000., ztop=20000.,
                dt=20., run_seconds=60., diff_6th_opt=2)
    base.update(overrides)
    return RunConfig(**base)


# ------------------------------------------------------------ the config door

def test_the_default_is_the_wrf_461_filter_with_no_second_factor():
    cfg = _cfg()
    assert cfg.diff_6th_form == "wrf_461"
    assert cfg.diff_6th_factor2 is None
    assert (cfg.mp_zero_out, cfg.mp_zero_out_all) == (0, 0)
    assert cfg.mp_zero_out_thresh == 1.0e-8
    validate_run_config(cfg)


def test_a_second_factor_under_the_wrf_461_form_is_refused_by_name():
    with pytest.raises(ValueError, match="silently ignored"):
        validate_run_config(_cfg(diff_6th_factor2=0.04))


@pytest.mark.parametrize("bad", [
    dict(diff_6th_form="hrrr"), dict(mp_zero_out=3),
    dict(mp_zero_out_all=2),
    dict(diff_6th_form="noaa_wrf39", diff_6th_factor2=-0.1),
    dict(mp_zero_out_thresh=float("nan"))])
def test_unknown_values_are_refused(bad):
    with pytest.raises(ValueError):
        validate_run_config(_cfg(**bad))


def test_the_fork_form_validates_with_and_without_its_factor():
    validate_run_config(_cfg(diff_6th_form="noaa_wrf39"))
    validate_run_config(_cfg(diff_6th_form="noaa_wrf39",
                             diff_6th_factor2=0.04, mp_zero_out=2,
                             mp_zero_out_thresh=1.0e-12, mp_zero_out_all=1))


# ------------------------------------------------- per-row factor and step

DRY = ("smag_ru", "smag_rv", "smag_rw", "smag_rth")
SCALAR = ("smag_rqv", "smag_rqc", "smag_rqr", "smag_rqi", "smag_rqs",
          "smag_rqg", "smag_rni", "smag_rnr", "smag_rnc", "smag_rnwfa",
          "smag_rnifa")


def test_wrf_461_rows_take_one_factor_and_dt_over_three():
    from gpuwm.core.dycore import _diff6_dt, _diff6_factor
    cfg = _cfg(diff_6th_factor=0.12)
    for slot in DRY:
        assert (_diff6_factor(cfg, slot), _diff6_dt(cfg, slot)) == (0.12, 20.)
    for slot in SCALAR + ("smag_rtke",):
        assert (_diff6_factor(cfg, slot),
                _diff6_dt(cfg, slot)) == (0.12, 20. / 3.)


def test_fork_rows_take_factor2_on_the_full_step():
    from gpuwm.core.dycore import _diff6_dt, _diff6_factor
    cfg = _cfg(diff_6th_factor=0.12, diff_6th_form="noaa_wrf39")
    for slot in DRY:
        assert (_diff6_factor(cfg, slot), _diff6_dt(cfg, slot)) == (0.12, 20.)
    for slot in SCALAR:
        assert (_diff6_factor(cfg, slot), _diff6_dt(cfg, slot)) == (0.04, 20.)
    # TKE keeps diff_6th_factor in the fork (solve_em.F:2587) on the full
    # step rk_scalar_tend builds for every array.
    assert (_diff6_factor(cfg, "smag_rtke"),
            _diff6_dt(cfg, "smag_rtke")) == (0.12, 20.)
    cfg = _cfg(diff_6th_form="noaa_wrf39", diff_6th_factor2=0.02)
    assert _diff6_factor(cfg, "smag_rqv") == 0.02


def test_the_moisture_coefficient_is_nine_times_weaker_at_hrrr_settings():
    from gpuwm.core.dycore import _diff6_dt, _diff6_factor
    old = _cfg(diff_6th_factor=0.12)
    new = _cfg(diff_6th_factor=0.12, diff_6th_form="noaa_wrf39",
               diff_6th_factor2=0.04)
    ratio = ((_diff6_factor(old, "smag_rqv") / _diff6_dt(old, "smag_rqv"))
             / (_diff6_factor(new, "smag_rqv") / _diff6_dt(new, "smag_rqv")))
    assert ratio == pytest.approx(9.0, rel=1e-12)


def test_the_fork_step_is_the_float32_product_the_fork_forms():
    from gpuwm.core.dycore import _diff6_dt
    for dt in (20., 15., 18., 7.5, 13.):
        cfg = _cfg(dt=dt, diff_6th_form="noaa_wrf39")
        want = float(np.float32(np.float32(dt) / np.float32(3.))
                     * np.float32(3.))
        assert _diff6_dt(cfg, "smag_rqv") == want
    assert _diff6_dt(_cfg(dt=20., diff_6th_form="noaa_wrf39"),
                     "smag_rqv") == 20.0


def test_the_batched_ensemble_graph_declines_the_fork_form():
    from gpuwm.ensemble.batch_mixing import _require_wrf461_diff6
    from gpuwm.ensemble.batch_state import BatchStateUnsupported
    _require_wrf461_diff6(_cfg())
    _require_wrf461_diff6(_cfg(diff_6th_opt=0, diff_6th_form="noaa_wrf39"))
    with pytest.raises(BatchStateUnsupported, match="ordinary door"):
        _require_wrf461_diff6(_cfg(diff_6th_form="noaa_wrf39"))


# ------------------------------------------------------------ the importer

def _import(tmp_path, dynamics="", physics=""):
    from test_namelist_gaps import _with
    from test_namelist_import import _pair
    from gpuwm.namelist_import import import_namelists
    text, report = import_namelists(*_pair(
        tmp_path, inp=_with(dynamics=dynamics, physics=physics)))
    return tomllib.loads(text), report


def test_a_wrf_461_namelist_imports_with_none_of_the_fork_keys(tmp_path):
    doc, _ = _import(tmp_path)
    keys = set(doc["shared"]) | {k for d in doc["domain"] for k in d}
    assert not keys & {"diff_6th_form", "diff_6th_factor2", "mp_zero_out",
                       "mp_zero_out_thresh", "mp_zero_out_all"}


def test_a_fork_namelist_selects_the_fork_filter_per_domain(tmp_path):
    doc, _ = _import(tmp_path, dynamics=" diff_6th_factor2 = 0.04, 0.03,\n")
    assert doc["shared"]["diff_6th_form"] == "noaa_wrf39"
    assert [d["diff_6th_factor2"] for d in doc["domain"]] == [0.04, 0.03]


def test_an_unassigned_tail_keeps_the_fork_registry_default(tmp_path):
    doc, _ = _import(tmp_path, dynamics=" diff_6th_factor2 = 0.05,\n")
    assert [d["diff_6th_factor2"] for d in doc["domain"]] == [0.05, 0.04]


def test_hrrr_mp_zero_out_imports_on_every_array_for_the_fork(tmp_path):
    zero = " mp_zero_out = 2,\n mp_zero_out_thresh = 1.e-12,\n"
    doc, report = _import(tmp_path,
                          dynamics=" diff_6th_factor2 = 0.04, 0.04,\n",
                          physics=zero)
    shared = doc["shared"]
    assert (shared["mp_zero_out"], shared["mp_zero_out_all"]) == (2, 1)
    assert shared["mp_zero_out_thresh"] == pytest.approx(1e-12)
    assert any("mp_zero_out_all = 1" in n for n in report.notices)
    # The same keys in a v4.6.1 namelist keep v4.6.1's moist-only default.
    doc, _ = _import(tmp_path, physics=zero)
    assert doc["shared"]["mp_zero_out"] == 2
    assert "mp_zero_out_all" not in doc["shared"]


def test_the_imported_fork_experiment_loads(tmp_path):
    from gpuwm.namelist_import import import_namelists
    from test_namelist_gaps import _with
    from test_namelist_import import _pair
    from gpuwm.experiment import load_experiment
    text, _ = import_namelists(*_pair(tmp_path, inp=_with(
        dynamics=" diff_6th_factor2 = 0.04, 0.03,\n",
        physics=" mp_zero_out = 2,\n mp_zero_out_thresh = 1.e-12,\n")))
    path = tmp_path / "experiment.toml"
    path.write_text(text)
    exp = load_experiment(path)
    runs = [domain.run for domain in exp.domains]
    assert [r.diff_6th_form for r in runs] == ["noaa_wrf39"] * 2
    assert [r.diff_6th_factor2 for r in runs] == [0.04, 0.03]
    assert runs[0].mp_zero_out == 2 and runs[0].mp_zero_out_all == 1


# ------------------------------------------------- edge-to-edge filter (GPU)

def _clamp(n, i):
    return min(max(i, 0), n - 1)


def fork_sixth_order_diffusion(f, mut, c1, c2, factor, dt, name,
                               phb=None, msfu=None, thresh=0.05, dx=0.0):
    """Independent float64 loop transcription of the fork's routine on a
    specified domain, opt 2, x and y: every point to the edge, every read
    of a point outside storage replaced by the zero-gradient halo copy
    (the nearest stored datum), mass and base geopotential included.
    ``phb``/``msfu`` engage the slope taper (slopeopt 1)."""
    f = np.asarray(f, np.float64)
    mut = np.asarray(mut, np.float64)
    nlev, nys, nxs = f.shape
    ny, nx = mut.shape
    coef = factor * 0.015625 / (2.0 * dt)
    k0, k1 = (1, nlev - 2) if name == "w" else (0, nlev - 1)

    def F(k, j, i):
        return f[k, _clamp(nys, j), _clamp(nxs, i)]

    def M(k, j, i):
        return c1[k] * mut[_clamp(ny, j), _clamp(nx, i)] + c2[k]

    def P(k, j, i):
        return phb[k, _clamp(ny, j), _clamp(nx, i)]

    def MU(j, i):
        return msfu[_clamp(ny, j), _clamp(nx + 1, i)]

    out = np.zeros_like(f)
    for k in range(k0, k1 + 1):
        for j in range(nys):
            for i in range(nxs):
                tend = 0.0
                for axis in ("x", "y"):
                    def G(m, a=axis):
                        return F(k, j, i + m) if a == "x" else F(k, j + m, i)
                    p0 = 10 * (G(0) - G(-1)) - 5 * (G(1) - G(-2)) + (G(2) - G(-3))
                    p1 = 10 * (G(1) - G(0)) - 5 * (G(2) - G(-1)) + (G(3) - G(-2))
                    if p0 * (G(0) - G(-1)) <= 0.0:
                        p0 = 0.0
                    if p1 * (G(1) - G(0)) <= 0.0:
                        p1 = 0.0
                    s0 = s1 = 1.0
                    if phb is not None and axis == "x" and name == "":
                        thr = thresh * 9.81 * dx
                        s0 = max(1 - abs(P(k, j, i) - P(k, j, i - 1))
                                 * MU(j, i) / thr, 0.0)
                        s1 = max(1 - abs(P(k, j, i + 1) - P(k, j, i))
                                 * MU(j, i + 1) / thr, 0.0)
                    if name == "u" and axis == "x":
                        m0, m1 = M(k, j, i - 1), M(k, j, i)
                    elif name == "u":
                        m0 = 0.25 * (M(k, j - 1, i - 1) + M(k, j - 1, i)
                                     + M(k, j, i - 1) + M(k, j, i))
                        m1 = 0.25 * (M(k, j, i - 1) + M(k, j, i)
                                     + M(k, j + 1, i - 1) + M(k, j + 1, i))
                    elif axis == "x":
                        m0 = 0.5 * (M(k, j, i - 1) + M(k, j, i))
                        m1 = 0.5 * (M(k, j, i) + M(k, j, i + 1))
                    else:
                        m0 = 0.5 * (M(k, j - 1, i) + M(k, j, i))
                        m1 = 0.5 * (M(k, j, i) + M(k, j + 1, i))
                    tend += coef * (s1 * m1 * p1 - s0 * m0 * p0)
                out[k, j, i] = tend
    return out


def _edge_case(stagger, nz=4, ny=10, nx=11, seed=5):
    rng = np.random.default_rng(seed)
    nlev = nz + 1 if stagger == "z" else nz
    shape = {"": (nlev, ny, nx), "z": (nlev, ny, nx),
             "x": (nlev, ny, nx + 1)}[stagger]
    f = rng.standard_normal(shape).astype(np.float32)
    mut = (9.0e4 + 4.0e3 * rng.random((ny, nx))).astype(np.float32)
    c1 = np.linspace(1.0, 0.2, nlev).astype(np.float32)
    c2 = np.linspace(0.0, 8.0e3, nlev).astype(np.float32)
    return f, mut, c1, c2


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("stagger,name", [("", "m"), ("x", "u"), ("z", "w")])
def test_the_edge_form_matches_the_fork_loops_with_zero_gradient_halos(
        stagger, name):
    import cupy as cp
    from gpuwm.core.dycore import launch_diff6_to_edge
    f, mut, c1, c2 = _edge_case(stagger)
    tend = cp.zeros(f.shape, cp.float32)
    launch_diff6_to_edge(cp.asarray(f), tend, cp.asarray(mut),
                         cp.asarray(c1), cp.asarray(c2), 0.04, 20.0, 2,
                         stagger=stagger)
    got = tend.get().astype(np.float64)
    want = fork_sixth_order_diffusion(f, mut, c1, c2, 0.04, 20.0,
                                      {"m": "", "u": "u", "w": "w"}[name])
    scale = np.abs(want).max()
    assert scale > 0
    np.testing.assert_allclose(got, want, rtol=0, atol=2e-5 * scale)
    # The rows WRF v4.6.1 skips carry the fork's tendency here.
    assert np.abs(got[:, :, 1:3]).max() > 0
    assert np.abs(got[:, 1:3, :]).max() > 0


@requires_gpu
@pytest.mark.gpu
def test_the_edge_form_slope_taper_reads_halo_base_geopotential():
    import cupy as cp
    from gpuwm.core.dycore import launch_diff6_to_edge
    f, mut, c1, c2 = _edge_case("")
    nlev, ny, nx = f.shape
    rng = np.random.default_rng(9)
    phb = (rng.random((nlev + 1, ny, nx)) * 3000.0).astype(np.float32)
    msfu = (1.0 + 0.02 * rng.random((ny, nx + 1))).astype(np.float32)
    msfv = np.ones((ny + 1, nx), np.float32)
    tend = cp.zeros(f.shape, cp.float32)
    launch_diff6_to_edge(cp.asarray(f), tend, cp.asarray(mut),
                         cp.asarray(c1), cp.asarray(c2), 0.04, 20.0, 2,
                         stagger="", phb=cp.asarray(phb),
                         msfu=cp.asarray(msfu), msfv=cp.asarray(msfv),
                         slopeopt=1, thresh=0.05, dx=3000.0, dy=3000.0)
    # y taper reads msfv = 1 and the same phb; only x is tapered in the
    # oracle, so compare with the y slope taken out of the problem: a phb
    # that is constant in y.
    phb_y = np.repeat(phb[:, :1, :], ny, axis=1)
    tend_y = cp.zeros(f.shape, cp.float32)
    launch_diff6_to_edge(cp.asarray(f), tend_y, cp.asarray(mut),
                         cp.asarray(c1), cp.asarray(c2), 0.04, 20.0, 2,
                         stagger="", phb=cp.asarray(phb_y),
                         msfu=cp.asarray(msfu), msfv=cp.asarray(msfv),
                         slopeopt=1, thresh=0.05, dx=3000.0, dy=3000.0)
    want = fork_sixth_order_diffusion(f, mut, c1, c2, 0.04, 20.0, "",
                                      phb=phb_y, msfu=msfu, thresh=0.05,
                                      dx=3000.0)
    got = tend_y.get().astype(np.float64)
    np.testing.assert_allclose(got, want, rtol=0,
                               atol=2e-5 * np.abs(want).max())
    assert not np.array_equal(tend.get(), tend_y.get())


# ------------------------------------------- the edge form's declared workspace
#
# Breakage these gates prevent: until 2.8.6 launch_diff6_to_edge built its
# padded field, padded tendency and padded planes with cp.pad/cp.zeros_like on
# every call.  The fork form is the request default of every HRRR recipe
# source (request-defaults.v1.toml), so a HRRR-route run allocated about three
# padded 3-D fields plus five padded planes per filtered field that the run
# preflight never priced, growing with the nest.


def _edge_slot_values(nz, ny, nx):
    py, px = ny + 7, nx + 7
    field = max(nz * py * (px + 1), nz * (py + 1) * px, (nz + 1) * py * px)
    planes = 2 * py * px + py * (px + 1) + (py + 1) * px
    return field, planes, (nz + 1) * py * px


def test_the_edge_form_workspace_is_priced_where_the_edge_form_runs():
    from gpuwm.core import preflight as pf
    from gpuwm.core.diff6_edge_workspace import DIFF6_EDGE_SLOTS
    field, planes, phb = _edge_slot_values(6, 14, 16)
    for forced in (dict(specified=True), dict(nested=True)):
        reg = pf.scratch_slot_registry(_cfg(diff_6th_form="noaa_wrf39",
                                            **forced))
        assert reg["diff6_edge_field"] == (field,)
        assert reg["diff6_edge_tend"] == (field,)
        assert reg["diff6_edge_planes"] == (planes,)
        assert "diff6_edge_phb" not in reg     # no slope taper, no read
        reg = pf.scratch_slot_registry(_cfg(diff_6th_form="noaa_wrf39",
                                            diff_6th_slopeopt=1, **forced))
        assert reg["diff6_edge_phb"] == (phb,)
    # Nowhere else: the WRF v4.6.1 form, a periodic domain (the fork reads
    # the periodic copy launch_diff6 already reads) and a switched-off filter.
    for cfg in (_cfg(), _cfg(specified=True, diff_6th_slopeopt=1),
                _cfg(diff_6th_form="noaa_wrf39", diff_6th_slopeopt=1),
                _cfg(diff_6th_opt=0, diff_6th_form="noaa_wrf39",
                     specified=True)):
        assert not set(DIFF6_EDGE_SLOTS) & set(pf.scratch_slot_registry(cfg))
    # Step-local: every slot is admitted to the shared arena, and the
    # restart walk classifies it as rebuilt, never serialized.
    from gpuwm.io import restart
    for slot in DIFF6_EDGE_SLOTS:
        assert pf.scratch_slot_lifetime(slot).kind == "write_before_read"
        assert pf.scratch_slot_uses_arena(slot)
        assert restart.classify_scratch_slot(slot) == "rebuild"


def test_the_run_preflight_prices_the_edge_form_workspace():
    from gpuwm.core.preflight import estimate_domain
    from gpuwm.experiment import DomainConfig

    def price(**overrides):
        cfg = _cfg(nx=96, ny=80, nz=40, diff_6th_slopeopt=1, **overrides)
        return estimate_domain(DomainConfig(
            grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
            parent_grid_ratio=1, parent_time_step_ratio=1,
            history_interval_s=3600.0, run=cfg, time_step=1))

    fork = price(diff_6th_form="noaa_wrf39", specified=True)
    stock = price(specified=True)
    field, planes, phb = _edge_slot_values(40, 80, 96)
    want = 4 * (2 * field + planes + phb)
    items = {item.name: item for item in fork.items
             if "diff6_edge_" in item.name}
    assert sum(item.nbytes for item in items.values()) == want, items
    assert not [item for item in stock.items if "diff6_edge_" in item.name]


# --------------------------------------------------------- mp_zero_out (GPU)

def _zero_out_reference(arrays, first, mode, thresh, sz):
    out = {}
    for name, a in arrays.items():
        a = a.copy()
        ny, nx = a.shape[1:]
        for ring in (a[:, 0, :], a[:, ny - 1, :], a[:, :, 0], a[:, :, nx - 1]):
            np.maximum(ring, 0, out=ring)
        inner = a[:, sz:ny - sz, sz:nx - sz]
        if name == first:
            if mode == 2:
                np.maximum(inner, 0, out=inner)
        else:
            inner[inner < np.float32(thresh)] = 0
        out[name] = a
    return out


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("mode,every_array,specified",
                         [(2, 1, True), (1, 1, True), (2, 0, True),
                          (2, 1, False)])
def test_mp_zero_out_is_wrfs_routine_on_the_arrays_it_names(
        mode, every_array, specified):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core.microphysics import microphysics_zero_out
    rng = np.random.default_rng(3)
    shape = (3, 9, 10)
    moist = ("qv", "qc", "qr", "qi", "qs", "qg")
    scalar = ("nr", "ni", "nc", "nwfa", "nifa")
    host = {}
    for name in moist + scalar:
        a = rng.standard_normal(shape).astype(np.float32)
        a *= np.float32(10.0) ** rng.integers(-14, -2, shape)
        host[name] = a
    state = SimpleNamespace(**{n: cp.asarray(a) for n, a in host.items()})
    cfg = _cfg(mp_physics=28, mp_zero_out=mode, mp_zero_out_thresh=1e-12,
               mp_zero_out_all=every_array, specified=specified,
               spec_zone=1)
    microphysics_zero_out(state, cfg)
    sz = 1 if specified else 0
    want = _zero_out_reference({n: host[n] for n in moist}, "qv", mode,
                               1e-12, sz)
    if every_array:
        # WRF's scalar array under mp_physics 28 is qni, qnr, qnc, qnwfa,
        # qnifa: qni sits at the vapour index.
        want.update(_zero_out_reference({n: host[n] for n in scalar}, "ni",
                                        mode, 1e-12, sz))
    else:
        want.update({n: host[n] for n in scalar})
    for name in moist + scalar:
        np.testing.assert_array_equal(getattr(state, name).get(), want[name],
                                      err_msg=name)


# --------------------------------------- the edge form's workspace (GPU)

def _transient_edge_launch(cp, f, tend, mut, c1, c2, factor, dt, opt,
                           stagger="", phb=None, msfu=None, msfv=None,
                           msft=None, slopeopt=0, thresh=0.10, dx=0.0,
                           dy=0.0, bnd_x=False, bnd_y=False, work=None):
    """The launcher as 2.8.5 shipped it (75c95b9dd): every padded input a
    cp.pad transient, the padded tendency a cp.zeros_like.  ``work`` is
    accepted and ignored.  The reference the declared workspace must equal
    word for word.  The calling GPU test hands in its own ``cp`` so this
    helper imports no device module (conftest would otherwise mark the
    whole file gpu and skip its CPU gates)."""
    from gpuwm.core.dycore import launch_diff6
    from gpuwm.core.state import DTYPE

    def edge_pad(a, low=3):
        pad = [(0, 0)] * (a.ndim - 2) + [(low, low + 1), (low, low + 1)]
        return cp.pad(a, pad, mode="edge")

    del work
    nlev, nys, nxs = f.shape
    nx = nxs - 1 if stagger == "x" else nxs
    ny = nys - 1 if stagger == "y" else nys
    fp = edge_pad(f)
    tp = cp.zeros_like(fp)
    mutp = edge_pad(mut)
    phbp = edge_pad(phb) if phb is not None and phb.ndim == 3 else phb
    msfup = edge_pad(msfu if msfu is not None
                     else cp.ones((ny, nx + 1), dtype=DTYPE))
    msfvp = edge_pad(msfv if msfv is not None
                     else cp.ones((ny + 1, nx), dtype=DTYPE))
    msftp = edge_pad(msft if msft is not None
                     else cp.ones((ny, nx), dtype=DTYPE))
    launch_diff6(fp, tp, mutp, c1, c2, factor, dt, opt, stagger=stagger,
                 phb=phbp, msfu=msfup, msfv=msfvp, msft=msftp,
                 slopeopt=slopeopt, thresh=thresh, dx=dx, dy=dy,
                 bnd_x=False, bnd_y=False)
    tend += tp[:, 3:3 + nys, 3:3 + nxs]


def _fork_edge_states(cp, slopeopt):
    from gpuwm.core.moist import SPECIES
    from gpuwm.verify.npref import random_acoustic_state

    states = []
    for _ in range(2):
        state, cfg0 = random_acoustic_state(
            seed=719, nz=8, ny=10, nx=12, hybrid_opt=2, hill_height=400.0,
            msf_amp=0.02, moist=True, mp_physics=8)
        for name in ("u", "v", "w", "thp"):
            getattr(state, name + "0")[...] = getattr(state, name)
        for index, name in enumerate(SPECIES, 1):
            field = getattr(state, name)
            values = cp.arange(field.size, dtype=cp.float32).reshape(
                field.shape)
            field[...] = cp.float32(index) * cp.float32(1.0e-9) * values
            getattr(state, name + "0")[...] = field
        states.append(state)
    cfg = RunConfig(**{**cfg0.__dict__, "diff_6th_opt": 2,
                       "diff_6th_factor": 0.12,
                       "diff_6th_form": "noaa_wrf39",
                       "diff_6th_factor2": 0.04, "specified": True,
                       "diff_6th_slopeopt": slopeopt,
                       "diff_6th_thresh": 0.05})
    assert states[0].phb.ndim == 3
    return states, cfg


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("slopeopt", [0, 1])
@pytest.mark.parametrize("caller", ["prepare_fixed_tendencies",
                                    "apply_diff6"])
def test_the_edge_form_workspace_is_bitwise_the_transient_form(
        monkeypatch, caller, slopeopt):
    """Both forecast callers draw exactly the registry's diff6_edge_* slots,
    from a NaN-poisoned shared arena, and leave every word the 2.8.5
    cp.pad transients left."""
    import functools
    import types

    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core import preflight as pf
    from gpuwm.core.diff6_edge_workspace import diff6_edge_slot_shapes
    from gpuwm.core.state import build_shared_scratch_arena

    (reference, lane), cfg = _fork_edge_states(cp, slopeopt)
    assert dycore.diff6_to_edge(cfg)
    run = getattr(dycore, caller)

    monkeypatch.setattr(dycore, "launch_diff6_to_edge",
                        functools.partial(_transient_edge_launch, cp))
    run(reference, cfg)
    monkeypatch.undo()

    arena = build_shared_scratch_arena((types.SimpleNamespace(run=cfg),))
    lane._scratch_arena = arena
    arena.poison()
    requested = {}
    scratch = lane.scratch

    def recording(shape, slot, dtype=None):
        if slot.startswith("diff6_edge_"):
            requested[slot] = tuple(shape)
        return scratch(shape, slot, dtype)

    lane.scratch = recording
    run(lane, cfg)
    cp.cuda.get_current_stream().synchronize()

    registry = pf.scratch_slot_registry(cfg)
    assert requested == diff6_edge_slot_shapes(cfg)
    assert requested == {slot: registry[slot] for slot in requested}
    assert ("diff6_edge_phb" in requested) == (slopeopt == 1)
    if caller == "apply_diff6":
        names = ("u", "v", "w", "thp", *SPECIES_ROWS)
        pairs = [(getattr(lane, n), getattr(reference, n), n) for n in names]
    else:
        slots = ("smag_ru", "smag_rv", "smag_rw", "smag_rth",
                 *("smag_r" + n for n in SPECIES_ROWS))
        pairs = [(lane.existing_scratch(s), reference.existing_scratch(s), s)
                 for s in slots]
    for got, want, name in pairs:
        assert got is not None and want is not None, name
        assert bool(cp.isfinite(got).all()), name
        cp.testing.assert_array_equal(got, want, err_msg=name)
    moved = [name for got, _w, name in pairs if bool(cp.any(got != 0))]
    assert moved, "the filter wrote nothing; the comparison proves nothing"


SPECIES_ROWS = ("qv", "qc", "qr", "qi", "qs", "qg")


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("stagger", ["", "x", "y", "z"])
def test_the_bare_launch_is_bitwise_the_transient_form(stagger):
    """The verification-only fallback (no DomainState) and the declared
    workspace both reproduce the 2.8.5 transients, including the identity
    map factors written straight into the padded planes."""
    import cupy as cp
    from gpuwm.core.dycore import launch_diff6_to_edge

    rng = np.random.default_rng(23)
    nz, ny, nx = 5, 9, 11
    nlev = nz + 1 if stagger == "z" else nz
    shape = {"": (nlev, ny, nx), "z": (nlev, ny, nx),
             "x": (nlev, ny, nx + 1), "y": (nlev, ny + 1, nx)}[stagger]
    f = cp.asarray(rng.standard_normal(shape).astype(np.float32))
    mut = cp.asarray((9.0e4 + 4.0e3 * rng.random((ny, nx))).astype(np.float32))
    c1 = cp.asarray(np.linspace(1.0, 0.2, nlev).astype(np.float32))
    c2 = cp.asarray(np.linspace(0.0, 8.0e3, nlev).astype(np.float32))
    phb = cp.asarray((rng.random((nz + 1, ny, nx)) * 3000.0)
                     .astype(np.float32))
    msfu = cp.asarray((1 + 0.02 * rng.random((ny, nx + 1))).astype(np.float32))
    for kwargs in (dict(), dict(phb=phb, msfu=msfu, slopeopt=1, thresh=0.05,
                                dx=3000.0, dy=3000.0)):
        want = cp.zeros(shape, cp.float32)
        _transient_edge_launch(cp, f, want, mut, c1, c2, 0.12, 20.0, 2,
                               stagger=stagger, **kwargs)
        got = cp.zeros(shape, cp.float32)
        launch_diff6_to_edge(f, got, mut, c1, c2, 0.12, 20.0, 2,
                             stagger=stagger, **kwargs)
        cp.testing.assert_array_equal(got, want)
        assert bool(cp.any(got != 0))
